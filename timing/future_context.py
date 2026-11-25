"""Change future context while preserving all samples around the pause boundary."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .charsiu_adapter import textless_gap
from .acoustic import runs
from .pause_evaluate import errors
import argparse,importlib.util,os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SOURCE='future-context-audio-v2'

def prepare():
    original=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    categories={r['key']:r['category'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json')}
    out=ROOT/'runs'/SOURCE
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'source_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'plan_sha256':digest(ROOT/'docs/context-transfer-plan.md'),'guards_ms':[100,500],'edits':['reverse','mute'],
        'edits_after_resampling':True,'sample_rate':16000,'primary_export_modified':False})
    for key,item in original.items():
        x,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32');assert sr==24000
        x=resample_poly(x,2,3);bracket=audit[key]['bracket']
        variants=[('baseline',len(x),x)]
        for guard in (100,500):
            cut=round((bracket[1]+guard/1000)*16000)
            assert cut<len(x)-4800, (key,cut,len(x))
            for edit in ('reverse','mute'):
                y=x.copy();y[cut:]=x[cut:][::-1] if edit=='reverse' else 0
                assert np.array_equal(y[:cut],x[:cut])
                variants.append((f'{edit}_g{guard}',cut,y))
        for mode,cut,y in variants:
            name=key+'__'+mode
            if state.complete(name):continue
            path=ROOT/'data/audio'/SOURCE/f'{name}.wav';path.parent.mkdir(parents=True,exist_ok=True)
            sf.write(path,y,16000,subtype='FLOAT')
            state.record(name,status='complete',base_key=key,mode=mode,category=categories[key],bracket=bracket,cut_sample=cut,
                prefix_unchanged=True,source_audio=str(path.relative_to(ROOT)).replace('\\','/'),
                artifacts={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)})
    print('Prepared',len(state.data['items']),'baseline/context waveforms',flush=True)

def classify_gap(cost,bracket,minimum):
    silent=np.zeros(len(cost),dtype=bool)
    for a,b in runs(cost.argmax(1)==1):
        if b-a>=minimum:silent[a:b]=True
    return textless_gap(silent,*bracket)

def charsiu():
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    free,total=torch.cuda.mem_get_info()
    if free<4*1024**3:raise RuntimeError('Need at least 4 GiB free for bounded Charsiu context probe')
    torch.set_num_threads(4);torch.manual_seed(20261115)
    src=ROOT/'runs'/SOURCE/'manifest.json';source=read_json(src)['items'];out=ROOT/'runs/future-context-charsiu-v1'
    locks={n:read_json(ROOT/'runs/model-locks'/f'{n}.json') for n in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for n,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for file,sha in locks[n]['sha256'].items():assert digest(ROOT/'models'/folder/file)==sha
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'source_sha256':digest(src),'models':locks,
        'normalization':['ordinary','fixed_original'],'minimum_frames':[4,2],'endpoint':'textless; no transcript forced onto edited speech'})
    code=ROOT/'models/charsiu_code/src/models.py';spec=importlib.util.spec_from_file_location('charsiu_context_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=model.cuda().eval()
    extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    for key in sorted({r['base_key'] for r in source.values()}):
        base,sr=sf.read(ROOT/source[key+'__baseline']['source_audio'],dtype='float32');assert sr==16000
        base_input=extractor(base,sampling_rate=16000,return_tensors='np').input_values[0]
        mean=base.mean();std=np.sqrt(base.var()+1e-7)
        assert np.array_equal(base_input,(base-mean)/std)
        old=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{key}.npz')['posterior']
        for original_name,item in source.items():
            if item['base_key']!=key:continue
            x,sr=sf.read(ROOT/item['source_audio'],dtype='float32');cut=item['cut_sample'];assert np.array_equal(x[:cut],base[:cut])
            for normalization in (['ordinary'] if item['mode']=='baseline' else ['ordinary','fixed_original']):
                name=original_name+'__'+normalization
                if state.complete(name):continue
                values=extractor(x,sampling_rate=16000,return_tensors='np').input_values[0] if normalization=='ordinary' else (x-mean)/std
                prefix_diff=float(np.max(np.abs(values[:cut]-base_input[:cut])))
                if normalization=='fixed_original':assert prefix_diff==0
                with torch.inference_mode():cost=model(torch.from_numpy(values[None]).cuda()).logits.softmax(-1)[0].cpu().numpy()
                centers=(np.arange(len(cost))+.5)*.01;a,b=item['bracket'];mask=(centers>=a)&(centers<=b)
                row={'key':key,'mode':item['mode'],'category':item['category'],'normalization':normalization,
                    'cut_seconds':cut/16000,'protected_bracket':item['bracket'],'normalized_prefix_max_abs_change':prefix_diff,
                    'methods':{str(m):classify_gap(cost,item['bracket'],m) for m in (4,2)},
                    'bracket_silence_posterior_MAE':float(np.mean(np.abs(cost[mask,1]-old[mask,1]))),
                    'bracket_argmax_changed_fraction':float(np.mean(cost[mask].argmax(1)!=old[mask].argmax(1)))}
                if item['mode']=='baseline':
                    previous=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{key}.json')
                    for minimum in ('4','2'):
                        saved=previous['methods'][minimum]['textless'];fresh=row['methods'][minimum]
                        assert all(saved.get(k)==fresh.get(k) for k in ('start','end','gap_ms','status')), (key,saved,fresh)
                    row['saved_baseline_endpoints_identical']=True
                path=out/'cases'/f'{name}.json';atomic_json(path,row)
                trace=ROOT/'data/audio/future-context-charsiu-v1'/f'{name}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(trace,silence_probability=cost[:,1],argmax=cost.argmax(1))
                state.record(name,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (path,trace)})
                print(name,row['methods']['4'].get('end'),round(row['bracket_silence_posterior_MAE'],4),flush=True)

def silero():
    import onnxruntime as ort
    from .silero_probe import infer,silence_gap
    lock=read_json(ROOT/'runs/model-locks/silero_vad.json');root=ROOT/'models/silero_vad'
    for f,sha in lock['sha256'].items():assert digest(root/f)==sha
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/'silero_vad.onnx'),options,providers=['CPUExecutionProvider'])
    src=ROOT/'runs'/SOURCE/'manifest.json';source=read_json(src)['items'];out=ROOT/'runs/future-context-silero-v1'
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'source_sha256':digest(src),'model':lock,'thresholds':[.3,.5,.7]})
    for key in sorted({r['base_key'] for r in source.values()}):
        x,sr=sf.read(ROOT/source[key+'__baseline']['source_audio'],dtype='float32');baseline=infer(session,x)
        for name,item in source.items():
            if item['base_key']!=key or state.complete(name):continue
            x,sr=sf.read(ROOT/item['source_audio'],dtype='float32');p=infer(session,x)
            prefix_chunks=item['cut_sample']//512;difference=float(np.max(np.abs(p[:prefix_chunks]-baseline[:prefix_chunks])))
            assert difference==0,(name,difference)
            row={'key':key,'mode':item['mode'],'prefix_complete_chunks':prefix_chunks,'prefix_probability_max_abs_change':difference,
                 'methods':{str(t):silence_gap(p,*item['bracket'],t) for t in (.3,.5,.7)}}
            path=out/'cases'/f'{name}.json';atomic_json(path,row)
            state.record(name,status='complete',artifacts={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)})
        print('Silero exact-prefix control',key,flush=True)

def evaluate():
    ch=[read_json(p) for p in (ROOT/'runs/future-context-charsiu-v1/cases').glob('*.json')]
    assert len(ch)==360,len(ch)
    bases={r['key']:r for r in ch if r['mode']=='baseline'};rows=[]
    for r in ch:
        if r['mode']=='baseline':continue
        for minimum in ('4','2'):
            a=bases[r['key']]['methods'][minimum];b=r['methods'][minimum]
            rows.append({**{k:r[k] for k in ('key','mode','category','normalization','bracket_silence_posterior_MAE')},'minimum':minimum,
                'baseline_status':a['status'],'edited_status':b['status'],
                **{field+'_change_ms':1000*(b[field]-a[field]) if a.get(field) is not None and b.get(field) is not None else None for field in ('start','end')},
                'gap_change_ms':b['gap_ms']-a['gap_ms'] if b.get('gap_ms') is not None and a.get('gap_ms') is not None else None})
    summary={}
    for mode in sorted({r['mode'] for r in rows}):
        summary[mode]={}
        for norm in ('ordinary','fixed_original'):
            summary[mode][norm]={}
            for minimum in ('4','2'):
                subset=[r for r in rows if r['mode']==mode and r['normalization']==norm and r['minimum']==minimum]
                summary[mode][norm][minimum]={'n':len(subset),'status_changes':sum(r['baseline_status']!=r['edited_status'] for r in subset),
                    **{field:errors([{'error_ms':r[field+'_change_ms']} for r in subset]) for field in ('start','end','gap')},
                    'mean_bracket_silence_posterior_MAE':float(np.mean([r['bracket_silence_posterior_MAE'] for r in subset]))}
    si=[read_json(p) for p in (ROOT/'runs/future-context-silero-v1/cases').glob('*.json')];assert len(si)==200
    sb={r['key']:r for r in si if r['mode']=='baseline'}
    assert all(r['methods']==sb[r['key']]['methods'] for r in si)
    summary['silero']={'cases':len(si),'all_prefix_probabilities_identical':True,'all_boundary_results_identical':True}
    out=ROOT/'runs/future-context-analysis-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_manifests_sha256':{run:digest(ROOT/'runs'/run/'manifest.json') for run in (SOURCE,'future-context-charsiu-v1','future-context-silero-v1')}})
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','charsiu','silero','evaluate']);args=p.parse_args()
    globals()[args.action]()
