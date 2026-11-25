"""Effect of measurement choices on paired contrasts, not a humor inference."""
from .common import ROOT,read_json,atomic_json,digest
import numpy as np

def main():
    cases=read_json(ROOT/'runs/phonetic-evidence-v1/cases.json')
    energy={r['key']:r for r in read_json(ROOT/'runs/detrended-rms-v1/measurements.json') if r['mode']=='natural'}
    items=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items'];pairs={}
    for c in cases:
        key=c['key'];item=items[key];g=item['script_id']+'_'+item['voice']
        values={m:c[m]['gap_ms'] for m in ('mfa2','mfa3','charsiu','raw')}
        values.update({m:energy[key]['methods'][m]['gap_ms'] for m in ('AC','linear')})
        pairs.setdefault(g,{})[item['kind']]={'key':key,'values':values,'original_flags':c['original_flags']}
    rows=[]
    for key,pair in pairs.items():
        j=pair['joke'];c=pair['control'];baseline=j['values']['mfa2']-c['values']['mfa2']
        for method in j['values']:
            a,b=j['values'][method],c['values'][method]
            change=None if a is None or b is None else (a-b)-baseline
            rows.append({'pair':key,'method':method,'change_in_paired_contrast_from_MFA2_ms':change,
                'original_flags':{'joke':j['original_flags'],'control':c['original_flags']},
                'interpretation':'difference between estimators, not true measurement error or an effect of humor'})
    summary={}
    for method in ('mfa3','charsiu','raw','AC','linear'):
        values=[r['change_in_paired_contrast_from_MFA2_ms'] for r in rows if r['method']==method and r['change_in_paired_contrast_from_MFA2_ms'] is not None]
        a=np.asarray(values);summary[method]={'expected_pairs':20,'n':len(a),'missing':20-len(a),
            'mean_signed_change_ms':float(a.mean()),'mean_absolute_change_ms':float(np.abs(a).mean()),
            'max_absolute_change_ms':float(np.abs(a).max()),'within30ms':int(np.sum(np.abs(a)<=30+1e-6))}
    out=ROOT/'runs/paired-measurement-sensitivity-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'source_sha256':digest(ROOT/'runs/phonetic-evidence-v1/cases.json'),
        'energy_sha256':digest(ROOT/'runs/detrended-rms-v1/measurements.json'),'all_original_flags_preserved':True,'hypothesis_test':False})
    print(summary)

if __name__=='__main__':main()
