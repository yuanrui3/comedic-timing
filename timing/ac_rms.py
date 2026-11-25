"""Fixed frame-mean removal: an acoustic quiet metric, not a phonetic oracle."""
from .common import ROOT,read_json,atomic_json,digest
from .acoustic import runs,acoustic_gap,compare
from .pause_evaluate import errors
import numpy as np
import soundfile as sf


def ac_track(audio,sr,frame_ms=10,hop_ms=5):
    x=np.asarray(audio,dtype=float)
    frame,hop=round(sr*frame_ms/1000),round(sr*hop_ms/1000)
    if x.ndim!=1 or len(x)<frame or not np.isfinite(x).all():raise ValueError('Invalid mono waveform')
    starts=np.arange(0,len(x)-frame+1,hop)
    sums=np.r_[0.,np.cumsum(x)];squares=np.r_[0.,np.cumsum(x*x)]
    mean=(sums[starts+frame]-sums[starts])/frame
    power=(squares[starts+frame]-squares[starts])/frame
    return (starts+frame/2)/sr,np.sqrt(np.maximum(0,power-mean*mean)),mean


def ac_gap(audio,sr,left,right,relative_db=-35):
    if not 0<=left<right<=len(audio)/sr:return {'gap_ms':None,'status':'invalid_search_bracket'}
    centers,energy,_=ac_track(audio,sr)
    scale=float(np.quantile(energy,.95))
    if scale<1e-5:return {'gap_ms':None,'status':'silent_clip'}
    threshold=max(1e-6,scale*10**(relative_db/20))
    candidates=[];censored=False
    for a,b in runs(energy<threshold):
        start,end=float(centers[a]-.0025),float(centers[b-1]+.0025)
        if end<left or start>right:continue
        if a==0 or b==len(energy) or start<left or end>right:censored=True;continue
        if end-start<.02-1e-9:continue
        candidates.append({'start':start,'end':end,'gap_ms':(end-start)*1000})
    info={'candidates':candidates,'threshold_rms':threshold,'preprocessing':'per-10ms-frame mean subtraction; no waveform filtering'}
    if len(candidates)==1:return {**info,**candidates[0],'status':'ok'}
    return {**info,'gap_ms':None if candidates or censored else 0.,
            'status':'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}


def main():
    rows=[]
    inputs={'natural':ROOT/'runs/pause-audit-v3/measurements.json','fixture':ROOT/'runs/pause-validation-v2/measurements.json'}
    for mode,path in inputs.items():
        for r in read_json(path):
            x,sr=sf.read(ROOT/r['source_audio'])
            rows.append({'key':r['key'],'mode':mode,'raw':r['rms_grid']['35'],'ac':ac_gap(x,sr,*r['bracket']),'hmm':r['hmm']})
    lookup={r['key']:r['ac']['gap_ms'] for r in rows}
    validation=[]
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        value=lookup[ref['key']];zero=lookup[ref['base_key']+'_'+ref['mode']+'_0000ms']
        recovered=value if ref['mode']=='absolute' else value-zero if value is not None and zero is not None else None
        validation.append({'key':ref['key'],'mode':ref['mode'],'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    rng=np.random.default_rng(20261115);stress=[]
    for sr in (16000,24000):
        for f0 in (80,120,220,440):
            for amp in (.002,.2):
                for background in ('none','DC','10Hz','18Hz','white_minus60dB'):
                    for delay in (0,30,100,300,600,1000):
                        t=np.arange(round(.4*sr))/sr;tone=amp*np.sin(2*np.pi*f0*t+.5)
                        x=np.r_[tone,np.zeros(round(delay/1000*sr)),tone]
                        if background=='DC':x+=amp*.025
                        elif background in ('10Hz','18Hz'):x+=amp*.025*np.sin(2*np.pi*int(background[:-2])*np.arange(len(x))/sr+.7)
                        elif background=='white_minus60dB':x+=rng.normal(0,amp/np.sqrt(2)*.001,len(x))
                        for method,fn in (('raw',acoustic_gap),('AC',ac_gap)):
                            r=fn(x,sr,.2,len(x)/sr-.2);v=r['gap_ms']
                            stress.append({'sr':sr,'f0_hz':f0,'amplitude':amp,'background':background,'delay_ms':delay,'method':method,
                                           'error_ms':None if v is None else v-delay,'status':r['status']})
    natural=[r for r in rows if r['mode']=='natural']
    summary={'natural_hmm_minus_ac':compare([r['hmm']['gap_ms'] for r in natural],[r['ac']['gap_ms'] for r in natural]),
        'natural_missing':[r['key'] for r in natural if r['ac']['gap_ms'] is None],
        'original_five':{r['key']:{'raw':r['raw']['status'],'ac':r['ac']} for r in natural if r['raw']['gap_ms'] is None},
        'fixtures':{m:errors([r for r in validation if r['mode']==m]) for m in ('absolute','incremental')},
        'stress':{method:{b:errors([r for r in stress if r['method']==method and r['background']==b])
                         for b in ('none','DC','10Hz','18Hz','white_minus60dB')} for method in ('raw','AC')}}
    out=ROOT/'runs/ac-rms-v1'
    for name,data in (('measurements',rows),('validation',validation),('stress',stress),('summary',summary)):atomic_json(out/f'{name}.json',data)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_sha256':{k:digest(v) for k,v in inputs.items()},
                                     'plan_sha256':digest(ROOT/'docs/second-aligner-plan.md')})
    print(summary)


if __name__=='__main__':main()
