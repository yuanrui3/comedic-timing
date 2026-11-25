"""Qualification before censoring: explicit follow-up, never rewrites old exports."""
from .common import ROOT,read_json,atomic_json,digest
from .acoustic import runs
from .measurement import rms_track
from .ac_rms import ac_track
from .detrended_rms import linear_track
from .pause_evaluate import errors
import numpy as np
import soundfile as sf

def select(centers,energy,left,right,hop_seconds=.005,minimum_seconds=.02):
    scale=float(np.quantile(energy,.95))
    if scale<1e-5:return {'gap_ms':None,'status':'silent_clip'}
    threshold=max(1e-6,scale*10**(-35/20));candidates=[];censored=False
    for a,b in runs(energy<threshold):
        start,end=float(centers[a]-hop_seconds/2),float(centers[b-1]+hop_seconds/2)
        if end<left or start>right:continue
        # Entire run is available, including the part outside the search bracket.
        # If it is too short globally, it cannot hide a qualifying interval.
        if end-start<minimum_seconds-1e-9:continue
        if a==0 or b==len(energy) or start<left or end>right:censored=True;continue
        candidates.append({'start':start,'end':end,'gap_ms':1000*(end-start)})
    if len(candidates)==1:return {**candidates[0],'status':'ok','candidates':candidates}
    return {'gap_ms':None if candidates or censored else 0.,'candidates':candidates,
        'status':'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}

def main():
    rows=[]
    sources={'natural':'pause-audit-v3/measurements.json','old':'pause-validation-v2/measurements.json','holdout':'pause-holdout-v1/brackets.json'}
    old_natural={r['key']:r for r in read_json(ROOT/'runs/detrended-rms-v1/measurements.json')}
    old_holdout={r['key']:r for r in read_json(ROOT/'runs/holdout-interim-v1/measurements.json')}
    changes=[]
    for dataset,path in sources.items():
        for r in read_json(ROOT/'runs'/path):
            x,sr=sf.read(ROOT/r['source_audio'])
            for method,fn in (('raw',rms_track),('AC',ac_track),('linear',linear_track)):
                centers,energy=fn(x,sr)[:2];result=select(centers,energy,*r['bracket'])
                previous=old_holdout[r['key']]['measurements']['RMS_'+method] if dataset=='holdout' else old_natural[r['key']]['methods'][method]
                row={'key':r['key'],'dataset':dataset,'method':method,'previous':previous,'corrected_selector':result}
                rows.append(row)
                if result['gap_ms']!=previous['gap_ms']:changes.append(row)
    lookup={(r['key'],r['method']):r['corrected_selector']['gap_ms'] for r in rows};validation=[]
    for dataset,run in (('old','pause-fixtures-v1'),('holdout','pause-holdout-v1')):
        for ref in read_json(ROOT/'runs'/run/'references.json'):
            for method in ('raw','AC','linear'):
                v=lookup[ref['key'],method];z=lookup[ref['base_key']+'_'+ref['mode']+'_0000ms',method]
                recovered=v if ref['mode']=='absolute' else v-z if v is not None and z is not None else None
                validation.append({'dataset':dataset,'key':ref['key'],'mode':ref['mode'],'method':method,'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    summary={'changes':[{'key':r['key'],'dataset':r['dataset'],'method':r['method'],'before':r['previous']['gap_ms'],'after':r['corrected_selector']['gap_ms']} for r in changes],
        'natural_changed_count':sum(r['dataset']=='natural' for r in changes),
        'recovery':{dataset:{method:{mode:errors([r for r in validation if r['dataset']==dataset and r['method']==method and r['mode']==mode]) for mode in ('absolute','incremental')}
            for method in ('raw','AC','linear')} for dataset in ('old','holdout')},'primary_export_modified':False}
    out=ROOT/'runs/quiet-selector-v2'
    for name,data in (('cases',rows),('validation',validation),('summary',summary)):atomic_json(out/f'{name}.json',data)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'sources_sha256':{k:digest(ROOT/'runs'/p) for k,p in sources.items()},
        'change':'Reject globally <20ms runs before censoring against the search bracket; all thresholds unchanged'})
    print(summary)

if __name__=='__main__':main()
