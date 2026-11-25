"""Export phonetic silence as the pause metric, retaining independent QC."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .pause_audit import audit
import argparse

def metric(row):
    hmm=row['hmm']
    acoustic=row['rms_grid']['35']
    flags=[]
    if hmm['gap_ms'] is None:
        flags.append('phonetic_alignment_failure')
    if acoustic['gap_ms'] is None:
        flags.append('ambiguous_waveform_interval')
    if hmm['gap_ms'] is not None and acoustic['gap_ms'] is not None:
        if abs(hmm['gap_ms']-acoustic['gap_ms'])>30+1e-6:
            flags.append('detector_difference_over_30ms')
    return {'key':row['key'],'script_id':row['script_id'],'voice':row['voice'],'kind':row['kind'],
            'pause_ms':hmm['gap_ms'],'definition':'duration of explicit silence phones between matched transcript words',
            'pause_estimator':'MFA Mandarin v2.0.0a, speaker independent',
            'interval':{k:hmm.get(k) for k in ('start','end')},
            'waveform_check':acoustic,'qc_flags':flags,
            'eligible_for_automatic_analysis':not flags,
            'human_review':'deferred','scientific_status':'exploratory_no_absolute_natural_speech_reference',
            'diagnostics':{'ctc_emission_gap_ms_not_silence':row['legacy_ctc']['gap_ms'],
                           'legacy_rms':row['legacy_rms'],'preemphasized_rms':row['spectral_grid']['35']}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='kokoro-diagnostic-v1')
    p.add_argument('--mfa',default='mfa-diagnostic-v1')
    p.add_argument('--out',default='corrected-pauses-v1')
    args=p.parse_args()
    out=ROOT/'runs'/args.out
    spec={'source':args.source,'mfa':args.mfa,'mfa_manifest_sha256':digest(ROOT/'runs'/args.mfa/'manifest.json'),
          'source_manifest_sha256':digest(ROOT/'runs'/args.source/'manifest.json'),
          'ctc_manifest_sha256':digest(ROOT/'runs'/args.source/'alignment-v2/manifest.json'),
          'code_sha256':{n:digest(ROOT/'timing'/n) for n in ('measure_pauses.py','pause_audit.py','acoustic.py')},
          'qc_rule':'Explicit HMM silence; unique -35 dB raw RMS; absolute difference <=30ms'}
    state=Manifest(out/'manifest.json',spec)
    if state.complete('metrics'):
        print('Resume: corrected pause metrics verified.')
        return
    rows,summary=audit(args.source,args.mfa)
    results=[metric(row) for row in rows]
    dest=out/'metrics.json'
    atomic_json(dest,results)
    summary['automatic_qc_eligible']=sum(r['eligible_for_automatic_analysis'] for r in results)
    summary['qc_flagged']=sum(bool(r['qc_flags']) for r in results)
    summary['gate_1']='not_accepted_residual_natural_speech_disagreement'
    atomic_json(out/'summary.json',summary)
    state.record('metrics',status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,out/'summary.json')})
    print(f'{len(results)} corrected metrics; {summary["automatic_qc_eligible"]} pass automatic detector QC; {summary["qc_flagged"]} flagged.')

if __name__=='__main__':
    main()
