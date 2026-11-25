"""Controlled subsonic sensitivity, separate from natural-speech accuracy."""
from .common import ROOT, atomic_json, digest
from .acoustic import acoustic_gap
from .bandlimited import highpass
from .pause_evaluate import errors
import numpy as np


def main():
    rows = []
    for sr in (16000,24000):
        for frequency in (0,10,18):
            for amplitude in (.002,.2):
                for delay in (0,30,100,300,600,1000):
                    t = np.arange(round(.4*sr))/sr
                    tone = amplitude*np.sin(2*np.pi*440*t+.5)
                    x = np.r_[tone,np.zeros(round(sr*delay/1000)),tone]
                    if frequency:
                        x += amplitude*.025*np.sin(2*np.pi*frequency*np.arange(len(x))/sr+.7)
                    for name,y in (('raw',x),('hp20',highpass(x,sr))):
                        result = acoustic_gap(y,sr,.2,len(x)/sr-.2)
                        value = result['gap_ms']
                        rows.append({'sample_rate':sr,'frequency_hz':frequency,'amplitude':amplitude,
                                     'delay_ms':delay,'method':name,'status':result['status'],
                                     'error_ms':value-delay if value is not None else None})
    summary = {method:{str(f):errors([r for r in rows if r['method']==method and r['frequency_hz']==f])
                       for f in (0,10,18)} for method in ('raw','hp20')}
    atomic_json(ROOT/'runs/subsonic-stress-v1/results.json',{'code_sha256':digest(__file__),'rows':rows,'summary':summary})
    print(summary)


if __name__=='__main__':
    main()
