from .common import ROOT,read_json,atomic_json,digest
from .pause_audit import audit
import argparse
import numpy as np

def errors(rows):
    valid=[r for r in rows if r['error_ms'] is not None]
    if not valid:
        return {'n':0,'expected':len(rows),'missing':len(rows),'pass_30ms':0}
    e=np.array([r['error_ms'] for r in valid])
    return {'n':len(valid),'expected':len(rows),'missing':len(rows)-len(valid),
            'bias_ms':float(e.mean()),'mae_ms':float(np.abs(e).mean()),
            'max_abs_ms':float(np.abs(e).max()),'pass_30ms':int((np.abs(e)<=30+1e-6).sum()),
            'all_pass':len(valid)==len(rows) and bool((np.abs(e)<=30+1e-6).all())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='pause-fixtures-v1')
    p.add_argument('--mfa',default='mfa-fixtures-v1')
    p.add_argument('--analysis',default='alignment-v3')
    p.add_argument('--out',default='pause-validation-v1')
    args=p.parse_args()
    measured,_=audit(args.source,args.mfa,args.analysis)
    references=read_json(ROOT/'runs'/args.source/'references.json')
    lookup={r['key']:r for r in measured}
    methods={'ctc_emission_gap':lambda r:r['legacy_ctc']['gap_ms'],
             'hmm_silence':lambda r:r['hmm']['gap_ms'],
             'legacy_rms':lambda r:r['legacy_rms']['gap_ms'],
             'rms_35_unique':lambda r:r['rms_grid']['35']['gap_ms'],
             'hysteresis':lambda r:r['hysteresis']['gap_ms'],
             'preemphasized_rms_35':lambda r:r['spectral_grid']['35']['gap_ms']}
    rows=[]
    for ref in references:
        row=lookup.get(ref['key'])
        zero_key=ref['base_key']+'_'+ref['mode']+'_0000ms'
        baseline=lookup.get(zero_key)
        for name,fn in methods.items():
            value=fn(row) if row else None
            base=fn(baseline) if baseline else None
            recovered=value-base if ref['mode']=='incremental' and value is not None and base is not None else value if ref['mode']=='absolute' else None
            rows.append({'key':ref['key'],'mode':ref['mode'],'method':name,'inserted_ms':ref['inserted_ms'],
                         'measured_gap_ms':value,'recovered_ms':recovered,
                         'error_ms':recovered-ref['inserted_ms'] if recovered is not None else None})
    summary={mode:{name:errors([r for r in rows if r['mode']==mode and r['method']==name]) for name in methods}
             for mode in ('absolute','incremental')}
    out=ROOT/'runs'/args.out
    atomic_json(out/'measurements.json',measured)
    atomic_json(out/'errors.json',rows)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'source':args.source,'mfa':args.mfa,'analysis':args.analysis,
             'code_sha256':{name:digest(ROOT/'timing'/name) for name in ('pause_evaluate.py','pause_audit.py','acoustic.py')},
             'references_sha256':digest(ROOT/'runs'/args.source/'references.json')})
    print(summary)

if __name__=='__main__':
    main()
