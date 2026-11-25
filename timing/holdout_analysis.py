"""Evaluate each fixed method on the same 288 new edited-speech cases."""
from .common import ROOT,read_json,atomic_json,digest
from .pause_audit import hmm_boundary
from .pause_evaluate import errors
from .acoustic import acoustic_gap
from .ac_rms import ac_gap
from .detrended_rms import linear_gap
import argparse
import soundfile as sf

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--without-charsiu',action='store_true');parser.add_argument('--out',default='holdout-analysis-v2');args=parser.parse_args()
    source=ROOT/'runs/pause-holdout-v1';refs=read_json(source/'references.json');brackets={r['key']:r for r in read_json(source/'brackets.json')}
    models={};inputs={}
    for model,run in (('MFA2','mfa-holdout-v1'),('MFA3','mfa3-holdout-v1')):
        prepared=read_json(ROOT/'runs'/run/'inputs.json')
        models[model]={p.stem:hmm_boundary(read_json(p),prepared[p.stem]) for p in (ROOT/'runs'/run/'aligned').rglob('*.json')}
        inputs[run]=digest(ROOT/'runs'/run/'manifest.json')
    if not args.without_charsiu:
        folder=ROOT/'runs/charsiu-holdout-cpu-v1';models['Charsiu']={p.stem:read_json(p)['forced'] for p in (folder/'aligned').glob('*.json')}
        if len(models['Charsiu'])!=len(refs):raise ValueError('Charsiu holdout incomplete; use --without-charsiu for a separate interim analysis')
        inputs['charsiu-holdout-cpu-v1']=digest(folder/'manifest.json')
    rows=[]
    for ref in refs:
        key=ref['key'];r=brackets[key];x,sr=sf.read(ROOT/r['source_audio'])
        measurements={name:data[key] for name,data in models.items()}
        for m,fn in (('RMS_raw',acoustic_gap),('RMS_AC',ac_gap),('RMS_linear',linear_gap)):measurements[m]=fn(x,sr,*r['bracket'])
        silero=read_json(ROOT/'runs/silero-phase-v2/frames'/f'{key}.json')
        for t in ('0.3','0.5','0.7'):measurements['Silero_'+t]=silero['raw'][t]
        measurements['Silero_phase_0.5']=silero['phase'];rows.append({'key':key,'measurements':measurements})
    lookup={r['key']:r['measurements'] for r in rows};validation=[]
    for ref in refs:
        for method,result in lookup[ref['key']].items():
            v=result['gap_ms'];z=lookup[ref['base_key']+'_'+ref['mode']+'_0000ms'][method]['gap_ms']
            recovered=v if ref['mode']=='absolute' else v-z if v is not None and z is not None else None
            validation.append({'key':ref['key'],'mode':ref['mode'],'method':method,'error_ms':None if recovered is None else recovered-ref['inserted_ms']})
    summary={method:{mode:errors([r for r in validation if r['method']==method and (mode=='all' or r['mode']==mode)]) for mode in ('absolute','incremental','all')}
        for method in rows[0]['measurements']}
    out=ROOT/'runs'/args.out;atomic_json(out/'measurements.json',rows);atomic_json(out/'validation.json',validation);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'source_sha256':digest(source/'manifest.json'),'alignment_manifests':inputs,
        'silero_manifest_sha256':digest(ROOT/'runs/silero-phase-v2/manifest.json'),'without_charsiu':args.without_charsiu})
    print(summary)

if __name__=='__main__':main()
