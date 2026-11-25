"""Intact-speech padding/gain equivariance, with condition-specific CMVN groups."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .charsiu_adapter import occurrence_alignment,boundary_from_occurrences,textless_gap
from .pause_audit import hmm_boundary
from .pause_evaluate import errors
import argparse,importlib.util,math,os,time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

VARIANTS={'baseline':(0,0,1.),'pad007':(7,200,1.),'pad100':(100,200,1.),'pad500':(500,200,1.),'gain010':(0,0,.1)}

def prepare():
    source=ROOT/'runs/kokoro-diagnostic-v1/manifest.json'
    state=Manifest(ROOT/'runs/intact-stability-v1/manifest.json',{'source_sha256':digest(source),'variants':VARIANTS,
        'code_sha256':digest(__file__),'purpose':'known translation and gain without changing speech samples',
        'CMVN_groups':'condition x original voice'})
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')};refs=[]
    for key,item in read_json(source)['items'].items():
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path,dtype='float32')
        for variant,(lead,tail,gain) in VARIANTS.items():
            name=key+'__'+variant;offset=round(lead/1000*sr)/sr
            refs.append({'key':name,'base_key':key,'variant':variant,'offset':offset,
                         'bracket':[v+offset for v in audit[key]['bracket']]})
            if state.complete(name):continue
            y=np.r_[np.zeros(round(offset*sr),dtype=x.dtype),x*gain,np.zeros(round(tail/1000*sr),dtype=x.dtype)]
            dest=ROOT/'data/audio/intact-stability-v1'/f'{name}.wav';dest.parent.mkdir(parents=True,exist_ok=True)
            sf.write(dest,y,sr,subtype='FLOAT')
            state.record(name,status='complete',script_id=item['script_id'],voice=item['voice']+'__'+variant,
                kind=item['kind'],text=item['text'],setup=item['setup'],ending=item['ending'],
                artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
    atomic_json(ROOT/'runs/intact-stability-v1/references.json',refs)
    print('Prepared',len(refs),'intact speech transformations')

def charsiu():
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    locks={name:read_json(ROOT/'runs/model-locks'/f'{name}.json') for name in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for name,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for file,sha in locks[name]['sha256'].items():assert digest(ROOT/'models'/folder/file)==sha
    code=ROOT/'models/charsiu_code/src/models.py';spec=importlib.util.spec_from_file_location('charsiu_upstream_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4);torch.manual_seed(20261115)
    model,info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=model.cuda().eval();extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    source=ROOT/'runs/intact-stability-v1/manifest.json';out=ROOT/'runs/charsiu-stability-v1'
    state=Manifest(out/'manifest.json',{'source_sha256':digest(source),'models':locks,'code_sha256':digest(__file__),
        'adapter_sha256':digest(ROOT/'timing/charsiu_adapter.py'),'timebase':'upstream 10ms; no fitted offset'})
    refs={r['key']:r for r in read_json(ROOT/'runs/intact-stability-v1/references.json')}
    for key,item in read_json(source)['items'].items():
        if state.complete(key):continue
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        prepared=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{refs[key]["base_key"]}.json')['prepared']
        x,sr=sf.read(path,dtype='float32');g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
        feat=extractor(x,sampling_rate=16000,return_tensors='pt').input_values.cuda();start=time.perf_counter()
        with torch.no_grad():cost=torch.softmax(model(feat).logits,dim=-1)[0].cpu().numpy()
        occ,silent,raw=occurrence_alignment(cost,prepared['phone_ids'])
        result={'key':key,'forced':boundary_from_occurrences(occ,prepared['split_phone_index']),
            'textless':textless_gap(silent,*refs[key]['bracket']),'elapsed_seconds':time.perf_counter()-start}
        dest=out/'aligned'/f'{key}.json';atomic_json(dest,result)
        state.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
        print(key,result['forced']['gap_ms'],flush=True)

def evaluate():
    refs=read_json(ROOT/'runs/intact-stability-v1/references.json');results={}
    for model,run in (('MFA2','mfa-stability-v1'),('MFA3','mfa3-stability-v1')):
        inp=read_json(ROOT/'runs'/run/'inputs.json')
        results[model]={p.stem:hmm_boundary(read_json(p),inp[p.stem]) for p in (ROOT/'runs'/run/'aligned').rglob('*.json')}
    results['Charsiu']={r['key']:read_json(ROOT/'runs/charsiu-stability-v1/aligned'/f'{r["key"]}.json')['forced'] for r in refs}
    rows=[]
    for model,values in results.items():
        for r in refs:
            if r['variant']=='baseline':continue
            b=values[r['base_key']+'__baseline'];v=values[r['key']]
            rows.append({**r,'model':model,'baseline':b,'modified':v,
                'start_error_ms':1000*(v['start']-r['offset']-b['start']),
                'end_error_ms':1000*(v['end']-r['offset']-b['end']),
                'gap_error_ms':v['gap_ms']-b['gap_ms']})
    summary={model:{variant:{endpoint:errors([{'error_ms':r[endpoint+'_error_ms']} for r in rows if r['model']==model and r['variant']==variant])
        for endpoint in ('start','end','gap')} for variant in VARIANTS if variant!='baseline'} for model in results}
    out=ROOT/'runs/intact-stability-analysis-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'source_sha256':digest(ROOT/'runs/intact-stability-v1/manifest.json'),
        'alignment_manifests_sha256':{r:digest(ROOT/'runs'/r/'manifest.json') for r in ('charsiu-stability-v1','mfa-stability-v1','mfa3-stability-v1')}})
    print(summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','charsiu','evaluate'));args=parser.parse_args()
    {'prepare':prepare,'charsiu':charsiu,'evaluate':evaluate}[args.action]()
