"""Deterministic background/level stress checks for the waveform estimators."""
from .common import ROOT,atomic_json,digest
from .acoustic import acoustic_gap,spectral_gap
from .pause_evaluate import errors
import numpy as np

def main():
    rng=np.random.default_rng(20261114)
    rows=[]
    for sr in (16000,24000):
        for amplitude in (.2,.002):
            t=np.arange(round(.4*sr))/sr
            tone=amplitude*np.sin(2*np.pi*440*t+.5)
            for background in ('none','white_minus60db','dc_offset'):
                for delay in (0,30,100,300,600,1000):
                    x=np.r_[tone,np.zeros(round(sr*delay/1000)),tone]
                    if background=='white_minus60db':
                        x=x+rng.normal(0,amplitude/np.sqrt(2)*.001,len(x))
                    elif background=='dc_offset':
                        x=x+amplitude*.025
                    for name,fn in {'raw_unique':lambda:acoustic_gap(x,sr,.2,len(x)/sr-.2),
                                    'hysteresis':lambda:acoustic_gap(x,sr,.2,len(x)/sr-.2,hysteresis=True),
                                    'preemphasized':lambda:spectral_gap(x,sr,.2,len(x)/sr-.2)}.items():
                        result=fn()
                        value=result['gap_ms']
                        rows.append({'sr':sr,'amplitude':amplitude,'background':background,'injected_ms':delay,
                                     'method':name,'measured_ms':value,'status':result['status'],
                                     'error_ms':None if value is None else value-delay})
    summary={name:{b:errors([r for r in rows if r['method']==name and r['background']==b])
                   for b in ('none','white_minus60db','dc_offset')} for name in ('raw_unique','hysteresis','preemphasized')}
    atomic_json(ROOT/'runs/pause-noise-v1/results.json',{'seed':20261114,'code_sha256':digest(__file__),
             'scope':'Signal-absence fixtures with controlled backgrounds; not speech alignment validation',
             'rows':rows,'summary':summary})
    print(summary)

if __name__=='__main__':
    main()
