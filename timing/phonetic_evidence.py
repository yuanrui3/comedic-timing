"""Model agreement, text-free evidence, and audit of truncated-phone fixtures."""
from .common import ROOT,read_json,atomic_json,digest
from .pause_audit import hmm_boundary
from .pause_evaluate import errors
import numpy as np


def stats(values):return errors([{'error_ms':float(v)} for v in values])

def main():
    cases=read_json(ROOT/'runs/flagged-followup-v1/cases.json')
    mfa3inputs=read_json(ROOT/'runs/mfa3-diagnostic-v3/inputs.json')
    mfa3={p.stem:hmm_boundary(read_json(p),mfa3inputs[p.stem]) for p in (ROOT/'runs/mfa3-diagnostic-v3/aligned').rglob('*.json')}
    vocab=read_json(ROOT/'models/charsiu_tokenizer/vocab.json');inverse={v:k for k,v in vocab.items()};rows=[]
    for case in cases:
        key=case['key'];c=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{key}.json')
        cost=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{key}.npz')['posterior'];centers=(np.arange(len(cost))+.5)*.01
        row={'key':key,'category':case['category'],'original_flags':case['original_flags'],'mfa2':case['hmm'],
             'mfa3':mfa3[key],'charsiu':c['methods']['4']['forced'],'charsiu_textless':c['methods']['4']['textless'],
             'charsiu_min2':c['methods']['2'],'raw':case['raw_rms'],'first_MFA2_phone':case['first_ending_phone']}
        row['delta_from_mfa2_ms']={model:{side:1000*(row[model][side]-row['mfa2'][side]) for side in ('start','end')} for model in ('mfa3','charsiu')}
        if case['raw_rms']['gap_ms'] is not None:
            a=case['hmm']['end'];b=case['raw_rms'].get('end',a)
            mask=(centers>=a)&(centers<b);p=cost[mask]
            if len(p):
                labels=p.argmax(1);values,counts=np.unique(labels,return_counts=True)
                split=c['prepared']['split_phone_index'];first=c['prepared']['phone_ids'][split]
                row['disputed_onset_frames']={'start':a,'end':b,'n':len(p),'nonsilence_argmax_fraction':float(np.mean(labels!=1)),
                    'mean_silence_posterior':float(p[:,1].mean()),'mean_expected_phone_posterior':float(p[:,first].mean()),
                    'expected_phone':inverse[first],'argmax_phones':{inverse[int(v)]:int(n) for v,n in zip(values,counts)},
                    'interpretation':'uncalibrated frame classifier outputs; no transcript used for argmax or silence posterior'}
        rows.append(row)
    target=[r for r in rows if r['category']=='RMS_endpoint_inside_first_HMM_phone']
    summary={'n_original_onset_targets':len(target),'comparison':{model:{side:stats([r['delta_from_mfa2_ms'][model][side] for r in target])
        for side in ('start','end')} for model in ('mfa3','charsiu')},
        'textless_equals_forced_all40':sum(r['charsiu_textless'].get('start')==r['charsiu']['start'] and r['charsiu_textless'].get('end')==r['charsiu']['end'] for r in rows),
        'textless_valid_onset_targets':sum(r['charsiu_textless']['status']=='ok' for r in target),
        'textless_onset_delta_mfa2':stats([1000*(r['charsiu_textless']['end']-r['mfa2']['end']) for r in target if r['charsiu_textless']['status']=='ok']),
        'target_onset_mean_nonsilence_argmax_fraction':float(np.mean([r['disputed_onset_frames']['nonsilence_argmax_fraction'] for r in target])),
        'target_onset_min_nonsilence_argmax_fraction':min(r['disputed_onset_frames']['nonsilence_argmax_fraction'] for r in target),
        'targets_majority_nonsilence':sum(r['disputed_onset_frames']['nonsilence_argmax_fraction']>.5 for r in target),
        'caveat':'Different architecture does not imply independent training labels; outputs are not human truth.'}
    fixture_rows=[]
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        if ref['mode']!='absolute':continue
        case=next(c for c in cases if c['key']==ref['base_key']);sr=ref['sample_rate'];left=ref['left_cut_sample']/sr;right=ref['right_cut_sample']/sr
        first=case['first_ending_phone'];last=case['last_setup_phone']
        cp=read_json(ROOT/'runs/charsiu-fixtures-v1/aligned'/f'{ref["key"]}.json')['methods']['4']['forced']
        fixture_rows.append({**ref,'first_original_MFA2_phone':first,'last_original_MFA2_phone':last,
            'cut_into_first_phone_ms':1000*max(0,min(right,first[1])-first[0]),
            'first_phone_duration_ms':1000*(first[1]-first[0]),
            'first_phone_entirely_removed_by_MFA2':right>=first[1],
            'cut_into_last_setup_phone_ms':1000*max(0,last[1]-max(left,last[0])),
            'charsiu':cp,'charsiu_start_error_ms':1000*(cp['start']-ref['reference_start']),
            'charsiu_end_error_ms':1000*(cp['end']-ref['reference_end']),
            'charsiu_gap_error_ms':cp['gap_ms']-ref['inserted_ms'],
            'interpretation':'Phone removal estimated by original model; exact cut and inserted zeros are known.'})
    summary['hard_splice_bases']={r['base_key']:{k:r[k] for k in ('cut_into_first_phone_ms','first_phone_duration_ms','first_phone_entirely_removed_by_MFA2','cut_into_last_setup_phone_ms')} for r in fixture_rows}
    out=ROOT/'runs/phonetic-evidence-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'hard-splices.json',fixture_rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (
        ROOT/'runs/flagged-followup-v1/cases.json',ROOT/'runs/charsiu-diagnostic-v1/manifest.json',ROOT/'runs/charsiu-fixtures-v1/manifest.json',ROOT/'runs/mfa3-diagnostic-v3/manifest.json')}})
    print(summary)


if __name__=='__main__':main()
