"""Pinned Charsiu on the 60 complete-script Melo diagnostic recordings."""
from .common import ROOT,read_json,digest,atomic_json,Manifest
from .charsiu_adapter import occurrence_alignment,boundary_from_occurrences,textless_gap
from .corpus import han
import importlib.util,os,math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    torch.set_num_threads(4);torch.manual_seed(20261115)
    if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Need 4 GiB free GPU for Charsiu transfer')
    src=ROOT/'runs/melo-diagnostic-v1/manifest.json';items=read_json(src)['items'];assert len(items)==60
    assert all(r['status']=='complete' for r in items.values())
    ctc=ROOT/'runs/melo-diagnostic-v1/alignment-v2'
    assert len([p for p in ctc.glob('*.json') if p.stem.startswith('zh')])==60
    locks={n:read_json(ROOT/'runs/model-locks'/f'{n}.json') for n in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for n,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for file,sha in locks[n]['sha256'].items():assert digest(ROOT/'models'/folder/file)==sha
    out=ROOT/'runs/charsiu-melo-v1';state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'models':locks,
        'adapter_sha256':digest(ROOT/'timing/charsiu_adapter.py'),'source_sha256':digest(src),
        'ctc_manifest_sha256':digest(ctc/'manifest.json'),'minimum_frames':[4,2],'bracket':'same CTC last/first character support plus 40ms as earlier audit'})
    code=ROOT/'models/charsiu_code/src/models.py';spec=importlib.util.spec_from_file_location('charsiu_transfer_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=model.cuda().eval();extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    for key,item in items.items():
        if state.complete(key):continue
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path,dtype='float32');ctc_result=read_json(ctc/f'{key}.json');split=len(han(item['setup']))
        bracket=[max(0.,ctc_result['units'][split-1]['start']-.04),min(len(x)/sr,ctc_result['units'][split]['end']+.04)]
        assert bracket[0]<bracket[1]
        original_key=f'{item["script_id"]}_zf_001_{item["kind"]}'
        prepared=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{original_key}.json')['prepared']
        assert prepared['words']==han(item['text'])
        g=math.gcd(sr,16000);audio=resample_poly(x,16000//g,sr//g)
        values=extractor(audio,sampling_rate=16000,return_tensors='pt').input_values.cuda()
        with torch.inference_mode():cost=model(values).logits.softmax(-1)[0].cpu().numpy()
        methods={}
        for minimum in (4,2):
            occurrence,silent,raw=occurrence_alignment(cost,prepared['phone_ids'],sil_threshold=minimum)
            methods[str(minimum)]={'forced':boundary_from_occurrences(occurrence,prepared['split_phone_index']),
                                  'textless':textless_gap(silent,*bracket)}
        row={'key':key,'methods':methods,'bracket':bracket,'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),
             'prepared_source':original_key,'frame_count':len(cost)}
        dest=out/'aligned'/f'{key}.json';atomic_json(dest,row)
        trace=ROOT/'data/audio/charsiu-melo-v1'/f'{key}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,posterior=cost)
        state.record(key,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,trace)})
        print(key,methods['4']['forced']['gap_ms'],flush=True)

if __name__=='__main__':main()
