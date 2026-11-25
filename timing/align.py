from .common import ROOT, Manifest, read_json, digest, source_hash, atomic_json
from .corpus import han
from .measurement import ctc_align, energy_gap, agreement
import argparse
import math
import time
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

class Aligner:
    def __init__(self, model_name='align_zh_full'):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        path = ROOT/'models'/model_name
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).to('cuda').eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank = self.model.config.pad_token_id
        self.stride = math.prod(self.model.config.conv_stride)/16000
        receptive, step = 1, 1
        for k,s in zip(self.model.config.conv_kernel,self.model.config.conv_stride):
            receptive += (k-1)*step
            step *= s
        self.frame_width = receptive/16000

    def measure(self, audio, sr, setup, ending):
        units = han(setup+ending)
        boundary = len(han(setup))
        missing = sorted(set(units)-set(self.vocab))
        if missing:
            raise ValueError(f'Unsupported alignment characters: {missing}')
        divisor = math.gcd(sr,16000)
        x = resample_poly(audio,16000//divisor,sr//divisor).astype(np.float32)
        values = self.processor(x,sampling_rate=16000,return_tensors='pt')
        with torch.inference_mode():
            logits = self.model(**{k:v.to('cuda') for k,v in values.items()}).logits[0]
            lp = logits.log_softmax(-1).cpu().numpy()
        spans = ctc_align(lp,[self.vocab[c] for c in units],self.blank)
        # CTC label-frame support is a proxy, not a proven phonetic boundary.
        times = [{'unit': c, 'start': a*self.stride,
                  'end': (b-1)*self.stride+self.frame_width} for c,(a,b) in zip(units,spans)]
        left, right = times[boundary-1]['end'], times[boundary]['start']
        anchor = (left+right)/2
        greedy = self.processor.decode(lp.argmax(-1))
        return {'ctc_gap_ms': (right-left)*1000,
                'ctc_measurement_type': 'character_emission_support_gap_NOT_silence',
                'valid_as_primary_pause': False,
                'setup_end': left, 'ending_start': right,
                'energy': energy_gap(audio,sr,anchor), 'units': times,
                'greedy_transcript': greedy, 'frame_stride_ms': self.stride*1000,
                'frame_width_ms': self.frame_width*1000,
                'resample': {'input_sr': sr, 'output_sr': 16000, 'method': 'scipy.signal.resample_poly',
                             'float32_samples_sha256': __import__('hashlib').sha256(x.tobytes()).hexdigest()},
                'qc': 'overlapping_ctc_support' if right<left else 'automatic_unreviewed'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',default='kokoro-diagnostic-v1')
    p.add_argument('--analysis',default='alignment-v2')
    p.add_argument('--model',default='align_zh_full')
    args = p.parse_args()
    source = ROOT/'runs'/args.run/'manifest.json'
    inputs = read_json(source)
    config = {'source_manifest_sha256': digest(source), 'model_lock': read_json(ROOT/'runs/model-locks'/f'{args.model}.json'),
              'code_sha256': source_hash(), 'energy_frame_ms':10, 'energy_hop_ms':5,
              'energy_relative_threshold_db':-35, 'energy_nearest_tolerance_ms':50}
    out = ROOT/'runs'/args.run/args.analysis
    manifest = Manifest(out/'manifest.json',config)
    torch.set_num_threads(4)
    aligner = Aligner(args.model)
    for key,item in inputs['items'].items():
        if item['status'] != 'complete' or manifest.complete(key):
            continue
        path = ROOT/next(iter(item['artifacts']))
        if digest(path) != item['artifacts'][str(path.relative_to(ROOT)).replace('\\','/')]:
            raise ValueError('Input waveform checksum mismatch')
        manifest.record(key,status='running')
        try:
            audio,sr = sf.read(path,dtype='float32')
            start = time.perf_counter()
            result = aligner.measure(audio,sr,item['setup'],item['ending'])
            result.update(key=key, script_id=item['script_id'],voice=item['voice'],kind=item['kind'],
                          elapsed_seconds=time.perf_counter()-start)
            dest = out/f'{key}.json'
            atomic_json(dest,result)
            manifest.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
            print(f'{key}: CTC={result["ctc_gap_ms"]:.0f}ms RMS={result["energy"]["gap_ms"]}ms',flush=True)
        except Exception as exc:
            manifest.record(key,status='failed',error=f'{type(exc).__name__}: {exc}')
            print(f'{key}: FAILED {exc}',flush=True)
    rows = [read_json(ROOT/next(iter(i['artifacts']))) for i in manifest.data['items'].values() if i['status']=='complete']
    atomic_json(out/'agreement.json',agreement(rows))

if __name__ == '__main__':
    main()
