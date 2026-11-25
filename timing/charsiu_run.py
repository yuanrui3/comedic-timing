"""Pinned local Mandarin frame-classification phonetic alignment."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .charsiu_adapter import transcript_phones,occurrence_alignment,boundary_from_occurrences,textless_gap
from .acoustic import compare
import argparse,importlib.util,math,os,time
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='pause-fixtures-v1')
    p.add_argument('--out',default='charsiu-fixtures-v1')
    p.add_argument('--limit',type=int,default=10000)
    args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    from g2pM import G2pM
    from transformers import Wav2Vec2FeatureExtractor
    code=ROOT/'models/charsiu_code/src/models.py'
    spec=importlib.util.spec_from_file_location('charsiu_upstream_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    locks={name:read_json(ROOT/'runs/model-locks'/f'{name}.json') for name in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for name,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for file,sha in locks[name]['sha256'].items():
            if digest(ROOT/'models'/folder/file)!=sha:raise ValueError('Model/source checksum mismatch')
    torch.set_num_threads(4);torch.manual_seed(20261115)
    model,load_info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    if any(load_info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(load_info))
    model=model.to('cuda').eval()
    extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    vocab=read_json(ROOT/'models/charsiu_tokenizer/vocab.json');g2p=G2pM()
    src=ROOT/'runs'/args.source/'manifest.json';source=read_json(src)
    audit_path=ROOT/'runs'/('pause-validation-v2' if args.source=='pause-fixtures-v1' else 'pause-audit-v3')/'measurements.json'
    brackets={r['key']:r['bracket'] for r in read_json(audit_path)}
    out=ROOT/'runs'/args.out
    state=Manifest(out/'manifest.json',{'models':locks,'source_sha256':digest(src),'limit':args.limit,
        'code_sha256':{n:digest(ROOT/'timing'/n) for n in ('charsiu_run.py','charsiu_adapter.py')},
        'torch':torch.__version__,'silence_threshold_frames':4,'sensitivity_threshold_frames':2,
        'normalization':'upstream per-utterance zero mean/unit variance','frame_shift_ms':10,'g2pm_version':'0.1.2.5',
        'alignment':'upstream frame classifier and probability DTW; occurrence-ID mapping replaces greedy string mapping'})
    atomic_json(out/'load-info.json',load_info)
    for key,item in list(source['items'].items())[:args.limit]:
        if state.complete(key):continue
        path=ROOT/next(iter(item['artifacts']))
        if digest(path)!=next(iter(item['artifacts'].values())):raise ValueError('Input checksum mismatch')
        text=item.get('text',item['setup']+item['ending'])
        prepared=transcript_phones(g2p,text,item['setup'],vocab)
        x,sr=sf.read(path,dtype='float32');divisor=math.gcd(sr,16000)
        audio=resample_poly(x,16000//divisor,sr//divisor)
        features=extractor(audio,sampling_rate=16000,return_tensors='pt').input_values.to('cuda')
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.no_grad():cost=torch.softmax(model(features).logits,dim=-1)[0].cpu().numpy()
        outputs={}
        frame_arrays={'posterior':cost.astype('float32')}
        for minimum in (4,2):
            occurrence,silent,raw=occurrence_alignment(cost,prepared['phone_ids'],vocab['[SIL]'],minimum)
            outputs[str(minimum)]={'forced':boundary_from_occurrences(occurrence,prepared['split_phone_index']),
                                   'textless':textless_gap(silent,*brackets[key])}
            frame_arrays[f'occurrence_min{minimum}']=occurrence
        frame_arrays['raw_argmax']=raw
        result={'key':key,'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),'prepared':prepared,'methods':outputs,
                'elapsed_seconds':time.perf_counter()-start,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,
                'frame_count':len(cost),'native_audio_seconds':len(x)/sr,'status':'complete'}
        dest=out/'aligned'/f'{key}.json';atomic_json(dest,result)
        trace=ROOT/'data/audio'/args.out/f'{key}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,**frame_arrays)
        state.record(key,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,trace)})
        print(key,outputs['4']['forced'],flush=True)


if __name__=='__main__':main()
