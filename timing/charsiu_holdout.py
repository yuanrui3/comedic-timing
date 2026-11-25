"""Run the frozen phonetic model on the expanded fixture set; resumable per clip."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .charsiu_adapter import occurrence_alignment,boundary_from_occurrences,textless_gap
import argparse,importlib.util,os,time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=288);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    source=ROOT/'runs/pause-holdout-v1/manifest.json';items=read_json(source)['items']
    brackets={r['key']:r['bracket'] for r in read_json(ROOT/'runs/pause-holdout-v1/brackets.json')}
    locks={n:read_json(ROOT/'runs/model-locks'/f'{n}.json') for n in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for n,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for f,sha in locks[n]['sha256'].items():assert digest(ROOT/'models'/folder/f)==sha
    out=ROOT/'runs'/f'charsiu-holdout-{args.device}-v1';state=Manifest(out/'manifest.json',{'models':locks,'source_sha256':digest(source),
        'code_sha256':digest(__file__),'adapter_sha256':digest(ROOT/'timing/charsiu_adapter.py'),'device':args.device,
        'expected_count':len(items),'limit_is_execution_only':True,'threads':4,'silence_min_frames':4})
    if all(state.complete(k) for k in items):print('Verified all',len(items),'completed outputs');return
    code=ROOT/'models/charsiu_code/src/models.py';spec=importlib.util.spec_from_file_location('charsiu_upstream_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);torch.set_num_threads(4);torch.manual_seed(20261115)
    model,info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=model.to(args.device).eval();extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    for key,item in list(items.items())[:args.limit]:
        if state.complete(key):continue
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        base=item['reference']['base_key'];prepared=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{base}.json')['prepared']
        x,sr=sf.read(path,dtype='float32');assert sr==24000;x=resample_poly(x,2,3)
        features=extractor(x,sampling_rate=16000,return_tensors='pt').input_values.to(args.device);start=time.perf_counter()
        with torch.no_grad():cost=torch.softmax(model(features).logits,dim=-1)[0].cpu().numpy()
        occ,sil,raw=occurrence_alignment(cost,prepared['phone_ids'])
        result={'key':key,'forced':boundary_from_occurrences(occ,prepared['split_phone_index']),
            'textless':textless_gap(sil,*brackets[key]),'elapsed_seconds':time.perf_counter()-start,'device':args.device}
        dest=out/'aligned'/f'{key}.json';atomic_json(dest,result)
        state.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
        print(key,result['elapsed_seconds'],result['forced']['gap_ms'],flush=True)

if __name__=='__main__':main()
