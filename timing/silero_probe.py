"""Pinned streaming Silero raw probabilities, distinct from phonetic alignment."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .acoustic import runs,compare
from .pause_evaluate import errors
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def silence_gap(speech_probability,left,right,threshold=.5):
    candidates=[];censored=False
    for a,b in runs(np.asarray(speech_probability)<threshold):
        start,end=a*.032,b*.032
        if end<left or start>right:continue
        if a==0 or b==len(speech_probability) or start<left or end>right:censored=True;continue
        candidates.append({'start':start,'end':end,'gap_ms':(b-a)*32.})
    if len(candidates)==1:return {**candidates[0],'status':'ok','candidates':candidates}
    return {'gap_ms':None if candidates or censored else 0.,'candidates':candidates,
        'status':'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}

def infer(session,x):
    # Same state/context updates as pinned official OnnxWrapper, using NumPy.
    state=np.zeros((2,1,128),dtype='float32');context=np.zeros((1,64),dtype='float32');scores=[]
    for i in range(0,len(x),512):
        frame=np.asarray(x[i:i+512],dtype='float32')
        if len(frame)<512:frame=np.pad(frame,(0,512-len(frame)))
        inp=np.concatenate([context,frame[None]],axis=1)
        out,state=session.run(None,{'input':inp,'state':state,'sr':np.array(16000,dtype='int64')})
        scores.append(float(out.item()));context=inp[:,-64:]
    return np.asarray(scores)

def main():
    import onnxruntime as ort
    root=ROOT/'models/silero_vad';lock=read_json(ROOT/'runs/model-locks/silero_vad.json')
    for f,sha in lock['sha256'].items():assert digest(root/f)==sha
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/'silero_vad.onnx'),opts,providers=['CPUExecutionProvider'])
    sources={'natural':ROOT/'runs/pause-audit-v3/measurements.json','fixture':ROOT/'runs/pause-validation-v2/measurements.json'}
    out=ROOT/'runs/silero-probe-v1';state=Manifest(out/'manifest.json',{'model':lock,'code_sha256':digest(__file__),
        'sources':{m:digest(p) for m,p in sources.items()},'runtime':ort.__version__,'thresholds':[.3,.5,.7],
        'timebase':'native 512-sample/32ms chunks, no fitted offset; previous 64 samples and recurrent state',
        'postprocessing':'no silence merging, hangover, or speech padding; unique complete interval in original bracket'})
    rows=[]
    for mode,path in sources.items():
        for row in read_json(path):
            key=row['key'];dest=out/'frames'/f'{key}.json'
            if state.complete(key):result=read_json(dest)
            else:
                x,sr=sf.read(ROOT/row['source_audio'],dtype='float32');g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
                probability=infer(session,x)
                assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
                result={'key':key,'mode':mode,'speech_probability':probability.tolist(),'bracket':row['bracket'],
                    'audio_sha256':digest(ROOT/row['source_audio']),
                    'grid':{str(t):silence_gap(probability,*row['bracket'],t) for t in (.3,.5,.7)}}
                atomic_json(dest,result);state.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
            rows.append(result)
    lookup={r['key']:r for r in rows};validation=[]
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        for threshold in ('0.3','0.5','0.7'):
            v=lookup[ref['key']]['grid'][threshold]['gap_ms'];z=lookup[ref['base_key']+'_'+ref['mode']+'_0000ms']['grid'][threshold]['gap_ms']
            recovered=v if ref['mode']=='absolute' else v-z if v is not None and z is not None else None
            validation.append({'key':ref['key'],'mode':ref['mode'],'threshold':threshold,'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    originals=read_json(sources['natural'])
    summary={'fixtures':{t:{m:errors([r for r in validation if r['threshold']==t and r['mode']==m]) for m in ('absolute','incremental')} for t in ('0.3','0.5','0.7')},
        'natural':{t:compare([r['hmm']['gap_ms'] for r in originals],[lookup[r['key']]['grid'][t]['gap_ms'] for r in originals]) for t in ('0.3','0.5','0.7')}}
    atomic_json(out/'validation.json',validation);atomic_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':main()
