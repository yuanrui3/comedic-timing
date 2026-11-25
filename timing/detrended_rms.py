"""Frozen linear-detrending diagnostic with quiet-speech negative controls."""
from .common import ROOT,read_json,atomic_json,digest
from .acoustic import acoustic_gap,runs,compare
from .ac_rms import ac_gap,ac_track
from .measurement import rms_track
from .pause_evaluate import errors
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt

def linear_track(audio,sr,frame_ms=10,hop_ms=5):
    x=np.asarray(audio,dtype=float);n=round(sr*frame_ms/1000);hop=round(sr*hop_ms/1000)
    if x.ndim!=1 or len(x)<n or not np.isfinite(x).all():raise ValueError('Invalid waveform')
    starts=np.arange(0,len(x)-n+1,hop);i=np.arange(len(x));cs=[np.r_[0.,np.cumsum(z)] for z in (x,x*x,x*i)]
    sx,sxx,six=[s[starts+n]-s[starts] for s in cs];stx=six-(starts+(n-1)/2)*sx
    stt=n*(n*n-1)/12
    residual=(sxx-sx*sx/n-stx*stx/stt)/n
    return (starts+n/2)/sr,np.sqrt(np.maximum(0,residual))

def linear_gap(x,sr,left,right):
    if not 0<=left<right<=len(x)/sr:return {'gap_ms':None,'status':'invalid_search_bracket'}
    centers,energy=linear_track(x,sr);scale=float(np.quantile(energy,.95))
    if scale<1e-5:return {'gap_ms':None,'status':'silent_clip'}
    threshold=max(1e-6,scale*10**(-35/20));candidates=[];censored=False
    for a,b in runs(energy<threshold):
        start,end=float(centers[a]-.0025),float(centers[b-1]+.0025)
        if end<left or start>right:continue
        if a==0 or b==len(energy) or start<left or end>right:censored=True;continue
        if end-start<.02-1e-9:continue
        candidates.append({'start':start,'end':end,'gap_ms':1000*(end-start)})
    out={'candidates':candidates,'threshold_rms':threshold,'preprocessing':'per-10ms-frame least-squares linear detrend'}
    if len(candidates)==1:return {**out,**candidates[0],'status':'ok'}
    return {**out,'gap_ms':None if candidates or censored else 0.,
            'status':'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}

METHODS={'raw':(acoustic_gap,rms_track),'AC':(ac_gap,ac_track),'linear':(linear_gap,linear_track)}

def main():
    rows=[];coverage=[]
    for mode,run in (('natural','pause-audit-v3'),('fixture','pause-validation-v2')):
        for r in read_json(ROOT/'runs'/run/'measurements.json'):
            x,sr=sf.read(ROOT/r['source_audio'])
            row={'key':r['key'],'mode':mode,'hmm':r['hmm'],'methods':{m:fn(x,sr,*r['bracket']) for m,(fn,_) in METHODS.items()}};rows.append(row)
            if mode!='natural':continue
            path=next((ROOT/'runs/mfa-diagnostic-v1/aligned').rglob(r['key']+'.json'))
            phones=read_json(path)['tiers']['phones']['entries'];tracks={m:fn(x,sr)[:2] for m,(_,fn) in METHODS.items()}
            center=tracks['raw'][0];speech=np.zeros(len(center),dtype=bool)
            for a,b,label in phones:
                if label not in ('sil','sp','spn','<eps>',''):
                    speech|=(center>=a+.02)&(center<b-.02)
            low={m:e<max(1e-6,float(np.quantile(e,.95))*10**(-35/20)) for m,(_,e) in tracks.items()}
            coverage.append({'key':r['key'],'interior_model_speech_frames':int(speech.sum()),
                'quiet_inside_model_speech':{m:int((speech&v).sum()) for m,v in low.items()},
                'newly_quiet_vs_raw':{m:int((speech&v&~low['raw']).sum()) for m,v in low.items()},
                'note':'Model-labelled phone interiors, with 20ms margins; not labelled truth.'})
    lookup={r['key']:r for r in rows};validation=[]
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        for m in METHODS:
            value=lookup[ref['key']]['methods'][m]['gap_ms'];zero=lookup[ref['base_key']+'_'+ref['mode']+'_0000ms']['methods'][m]['gap_ms']
            recovered=value if ref['mode']=='absolute' else value-zero if value is not None and zero is not None else None
            validation.append({'key':ref['key'],'method':m,'mode':ref['mode'],'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    rng=np.random.default_rng(20261115);negative=[];stress=[]
    for sr in (16000,24000):
        for f0 in (40,60,80,120,220,440,'frication'):
            for relative in (.01,.02,.032,.05,.1,.2):
                for phase in (0.,.7,1.4):
                    t=np.arange(round(.4*sr))/sr;tone=.2*np.sin(2*np.pi*220*t+.5)
                    tb=np.arange(round(.12*sr))/sr
                    if f0=='frication':
                        bridge=sosfiltfilt(butter(3,[500,5000],btype='bandpass',fs=sr,output='sos'),rng.normal(size=len(tb)))
                        bridge=bridge/np.sqrt(np.mean(bridge**2))/np.sqrt(2)
                    else:bridge=np.sin(2*np.pi*f0*tb+phase)
                    x=np.r_[tone,.2*relative*bridge,tone]
                    for m,(fn,_) in METHODS.items():
                        r=fn(x,sr,.3,.62)
                        negative.append({'sr':sr,'bridge':f0,'relative_amplitude':relative,'phase':phase,'method':m,
                            'quiet_ms':r['gap_ms'],'status':r['status'],'interpretation':'No inserted silence; energy criterion can classify weak speech-like probes as quiet.'})
        for f0 in (80,120,220,440):
            for amp in (.002,.2):
                for bg in ('none','DC','10Hz','18Hz','linear_drift','white_minus60dB'):
                    for delay in (0,30,100,300,600,1000):
                        t=np.arange(round(.4*sr))/sr;tone=amp*np.sin(2*np.pi*f0*t+.5);x=np.r_[tone,np.zeros(round(delay/1000*sr)),tone]
                        if bg=='DC':x+=amp*.025
                        elif bg in ('10Hz','18Hz'):x+=amp*.025*np.sin(2*np.pi*int(bg[:-2])*np.arange(len(x))/sr+.7)
                        elif bg=='linear_drift':x+=np.linspace(-amp*.1,amp*.1,len(x))
                        elif bg=='white_minus60dB':x+=rng.normal(0,amp/np.sqrt(2)*.001,len(x))
                        r=linear_gap(x,sr,.2,len(x)/sr-.2)
                        stress.append({'sr':sr,'f0_hz':f0,'amplitude':amp,'background':bg,'delay_ms':delay,'error_ms':None if r['gap_ms'] is None else r['gap_ms']-delay,'status':r['status']})
    natural=[r for r in rows if r['mode']=='natural'];summary={
        'natural':{m:{'missing':[r['key'] for r in natural if r['methods'][m]['gap_ms'] is None],
            'hmm_minus_acoustic':compare([r['hmm']['gap_ms'] for r in natural],[r['methods'][m]['gap_ms'] for r in natural])} for m in METHODS},
        'fixtures':{m:{mode:errors([r for r in validation if r['method']==m and r['mode']==mode]) for mode in ('absolute','incremental')} for m in METHODS},
        'linear_synthetic_gaps':errors(stress),
        'negative_controls':{m:{'total':sum(r['method']==m for r in negative),
            'quiet_cases_all_levels':sum(r['method']==m and r['quiet_ms'] is not None and r['quiet_ms']>0 for r in negative),
            'quiet_cases_at_least_minus30dB':sum(r['method']==m and r['relative_amplitude']>=.032 and r['quiet_ms'] is not None and r['quiet_ms']>0 for r in negative),
            'ambiguous':sum(r['method']==m and r['quiet_ms'] is None for r in negative)} for m in METHODS},
        'natural_model_speech':{'interior_frames':sum(r['interior_model_speech_frames'] for r in coverage),
            'newly_quiet_vs_raw':{m:sum(r['newly_quiet_vs_raw'][m] for r in coverage) for m in METHODS}}}
    out=ROOT/'runs/detrended-rms-v1'
    for name,data in (('measurements',rows),('validation',validation),('coverage',coverage),('negative',negative),('stress',stress),('summary',summary)):atomic_json(out/f'{name}.json',data)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'ac_code_sha256':digest(ROOT/'timing/ac_rms.py'),'plan_sha256':digest(ROOT/'docs/second-aligner-plan.md')})
    print(summary)

if __name__=='__main__':main()
