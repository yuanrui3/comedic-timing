"""Fixed four-phase chunk ensemble, tested without tuning on held-out scripts."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .silero_probe import infer,silence_gap
from .acoustic import runs,compare
from .pause_evaluate import errors
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def phase_gap(prob,left,right):
    candidates=[];censored=False
    for a,b in runs(np.asarray(prob)<.5):
        start,end=a*.008,b*.008
        if end<left or start>right:continue
        if a==0 or b==len(prob) or start<left or end>right:censored=True;continue
        if end-start<.02-1e-9:continue
        candidates.append({'start':start,'end':end,'gap_ms':8.*(b-a)})
    if len(candidates)==1:return {**candidates[0],'status':'ok','candidates':candidates}
    return {'gap_ms':None if candidates or censored else 0.,'candidates':candidates,
        'status':'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}

def main():
    import onnxruntime as ort
    lock=read_json(ROOT/'runs/model-locks/silero_vad.json');root=ROOT/'models/silero_vad'
    for f,sha in lock['sha256'].items():assert digest(root/f)==sha
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/'silero_vad.onnx'),opts,providers=['CPUExecutionProvider'])
    sources={'old':ROOT/'runs/pause-validation-v2/measurements.json','holdout':ROOT/'runs/pause-holdout-v1/brackets.json','natural':ROOT/'runs/pause-audit-v3/measurements.json'}
    out=ROOT/'runs/silero-phase-v2';state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),
        'adapter_sha256':digest(ROOT/'timing/silero_probe.py'),'model':lock,'sources':{m:digest(p) for m,p in sources.items()},
        'ensemble_phases_ms':[0,8,16,24],'ensemble_threshold':.5,'phase_mapping':'mean of four piecewise-constant 32ms cells on 8ms grid; no fitted offset'})
    rows=[]
    for dataset,path in sources.items():
        for r in read_json(path):
            key=r['key'];dest=out/'frames'/f'{key}.json'
            if state.complete(key):result=read_json(dest)
            else:
                x,sr=sf.read(ROOT/r['source_audio'],dtype='float32');g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
                centers=np.arange(math.ceil(len(x)/128))*.008+.004;series=[];phase_arrays=[]
                for pad in (0,128,256,384):
                    p=infer(session,np.r_[np.zeros(pad,dtype=x.dtype),x]);phase_arrays.append(p)
                    indices=np.floor((centers+pad/16000)/.032+1e-9).astype(int);indices=np.minimum(indices,len(p)-1)
                    series.append(p[indices])
                average=np.mean(series,axis=0)
                result={'key':key,'dataset':dataset,'source_audio':r['source_audio'],'bracket':r['bracket'],
                    'raw':{str(t):silence_gap(phase_arrays[0],*r['bracket'],t) for t in (.3,.5,.7)},
                    'phase':phase_gap(average,*r['bracket']),'ensemble_speech_probability':average.tolist()}
                atomic_json(dest,result);state.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
                print(dataset,key,result['phase']['gap_ms'],flush=True)
            rows.append(result)
    lookup={r['key']:r for r in rows};validation=[]
    for dataset,run in (('old','pause-fixtures-v1'),('holdout','pause-holdout-v1')):
        for ref in read_json(ROOT/'runs'/run/'references.json'):
            for method in ('raw0.3','raw0.5','raw0.7','phase0.5'):
                def value(key):return lookup[key]['phase']['gap_ms'] if method=='phase0.5' else lookup[key]['raw'][method[3:]]['gap_ms']
                v=value(ref['key']);z=value(ref['base_key']+'_'+ref['mode']+'_0000ms')
                recovered=v if ref['mode']=='absolute' else v-z if v is not None and z is not None else None
                validation.append({'dataset':dataset,'key':ref['key'],'mode':ref['mode'],'method':method,'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    summary={dataset:{method:{mode:errors([r for r in validation if r['dataset']==dataset and r['method']==method and r['mode']==mode]) for mode in ('absolute','incremental')}
        for method in ('raw0.3','raw0.5','raw0.7','phase0.5')} for dataset in ('old','holdout')}
    natural=read_json(sources['natural']);summary['natural_phase_hmm_difference']=compare([r['hmm']['gap_ms'] for r in natural],[lookup[r['key']]['phase']['gap_ms'] for r in natural])
    atomic_json(out/'validation.json',validation);atomic_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':main()
