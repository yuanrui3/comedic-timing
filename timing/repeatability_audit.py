"""Fresh fixed-condition alignment batches; observed spread is not true error."""
from .common import ROOT,read_json,atomic_json,digest
from .pause_audit import hmm_boundary
from .pause_evaluate import errors
import argparse,subprocess,sys
import numpy as np

def run(family):
    for i in range(1,5):
        name=f'{family}-repeat-batch{i:02}-v1'
        command=[sys.executable,'-X','utf8','-m']
        if family=='mfa2':command+=['timing.mfa_variants','--baseline','mfa-diagnostic-v1','--run',name]
        else:command+=['timing.mfa3_run','--source','kokoro-diagnostic-v1','--run',name]
        result=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
        print(name,'complete',flush=True)

def evaluate():
    originals={'mfa2':'mfa-diagnostic-v1','mfa3':'mfa3-diagnostic-v3'};rows=[];inputs={}
    targets={r['key'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json') if r['category']=='RMS_endpoint_inside_first_HMM_phone'}
    for family,baseline in originals.items():
        names=[baseline]+[f'{family}-repeat-batch{i:02}-v1' for i in range(1,5)];all_runs=[]
        for name in names:
            folder=ROOT/'runs'/name;prepared=read_json(folder/'inputs.json');values={p.stem:hmm_boundary(read_json(p),prepared[p.stem]) for p in (folder/'aligned').rglob('*.json')}
            if len(values)!=40:raise ValueError('Incomplete repeat batch: '+name)
            all_runs.append(values);inputs[name]=digest(folder/'manifest.json')
        for key,base in all_runs[0].items():
            estimates=[r[key] for r in all_runs];row={'key':key,'model':family,'original_onset_target':key in targets,'estimates':estimates,
                'observed_range_ms':{side:1000*(max(r[side] for r in estimates)-min(r[side] for r in estimates)) for side in ('start','end')},
                'gap_range_ms':max(r['gap_ms'] for r in estimates)-min(r['gap_ms'] for r in estimates),
                'max_delta_original_ms':{side:max(1000*abs(r[side]-base[side]) for r in estimates) for side in ('start','end')},
                'max_gap_delta_original_ms':max(abs(r['gap_ms']-base['gap_ms']) for r in estimates)}
            rows.append(row)
    summary={family:{'clips':40,'estimates_per_clip':5,
        'end_range_max_ms':max(r['observed_range_ms']['end'] for r in rows if r['model']==family),
        'start_range_max_ms':max(r['observed_range_ms']['start'] for r in rows if r['model']==family),
        'gap_range_max_ms':max(r['gap_range_ms'] for r in rows if r['model']==family),
        'onsets_any_repeat_delta_over30ms':sum(r['model']==family and r['max_delta_original_ms']['end']>30+1e-6 for r in rows),
        'gaps_any_repeat_delta_over30ms':sum(r['model']==family and r['max_gap_delta_original_ms']>30+1e-6 for r in rows)} for family in originals}
    out=ROOT/'runs/repeatability-audit-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_manifests_sha256':inputs,
        'interpretation':'Fresh batch repeatability; no seed/dither/weight changes. Observed ranges are not natural-accuracy confidence intervals.'})
    print(summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('mfa2','mfa3','evaluate'));args=parser.parse_args()
    evaluate() if args.action=='evaluate' else run(args.action)
