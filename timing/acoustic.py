"""Acoustic silence intervals; CTC is only a search bracket, never a duration oracle."""
import numpy as np
import math
from scipy.signal import resample_poly,lfilter
from .measurement import rms_track

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def close_interruptions(mask, max_frames):
    result=mask.copy()
    for a,b in runs(~mask):
        if a>0 and b<len(mask) and b-a<=max_frames:
            result[a:b]=True
    return result

def acoustic_gap(audio,sr,left,right,relative_db=-35,hysteresis=False):
    """Return unique internal quiet interval in a transcript-localized bracket.

    Grid comparator: one threshold. Hysteresis: -35 dB core and -25 dB
    outer boundaries, bridging <=20 ms interruptions. No HMM endpoints enter.
    Minimum detected interval 20 ms; sub-resolution dips are not pause evidence.
    """
    if not (0<=left<right<=len(audio)/sr):
        return {'gap_ms':None,'status':'invalid_search_bracket'}
    centers,rms=rms_track(audio,sr)
    scale=float(np.quantile(rms,.95))
    if scale<1e-5:
        return {'gap_ms':None,'status':'silent_clip'}
    low_threshold=max(1e-6,scale*10**(relative_db/20))
    core=rms<low_threshold
    outer_threshold=max(1e-6,scale*10**(-25/20)) if hysteresis else low_threshold
    mask=close_interruptions(rms<outer_threshold,4) if hysteresis else core
    candidates=[]
    censored=False
    for a,b in runs(mask):
        start,end=float(centers[a]-.0025),float(centers[b-1]+.0025)
        if end<left or start>right:
            continue
        if a==0 or b==len(mask) or start<left or end>right:
            censored=True
            continue
        duration=(end-start)*1000
        if duration<20-1e-6 or (hysteresis and np.count_nonzero(core[a:b])<2):
            continue
        candidates.append({'start':start,'end':end,'gap_ms':duration})
    info={'low_threshold_rms':low_threshold,'outer_threshold_rms':outer_threshold,
          'search_bracket':[left,right],'candidates':candidates,'hysteresis':hysteresis}
    if len(candidates)==1:
        return {**info,**candidates[0],'status':'ok'}
    if len(candidates)>1:
        return {**info,'gap_ms':None,'status':'ambiguous_multiple_intervals'}
    return {**info,'gap_ms':None if censored else 0.,
            'status':'censored_at_bracket' if censored else 'no_qualifying_interval'}

def compare(first,second):
    valid=[(a,b) for a,b in zip(first,second) if a is not None and b is not None]
    if not valid:
        return {'n':0,'missing':len(first)}
    a,b=np.array(valid).T
    d=a-b
    sd=float(d.std(ddof=1)) if len(d)>1 else None
    return {'n':len(d),'missing':len(first)-len(d),'bias_ms':float(d.mean()),
            'mae_ms':float(np.abs(d).mean()),'within_30ms':float(np.mean(np.abs(d)<=30+1e-6)),
            'limits_of_agreement_ms':[float(d.mean()-1.96*sd),float(d.mean()+1.96*sd)] if sd is not None else None,
            'max_abs_ms':float(np.abs(d).max())}

def spectral_gap(audio,sr,left,right,relative_db=-35):
    """Fixed 16 kHz / 0.97 pre-emphasis, then the unchanged RMS estimator."""
    divisor=math.gcd(sr,16000)
    x=resample_poly(np.asarray(audio,dtype=float),16000//divisor,sr//divisor)
    y=lfilter([1.,-.97],[1.],x)
    result=acoustic_gap(y,16000,left,min(right,len(y)/16000),relative_db=relative_db)
    return {**result,'preprocessing':{'sample_rate':16000,'preemphasis':.97,'relative_db':relative_db}}
