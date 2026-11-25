"""Compare unlike legacy proxies separately from explicit phonetic silence."""
from .common import ROOT,read_json,atomic_json,digest
from .acoustic import acoustic_gap,spectral_gap,compare
import argparse
import numpy as np
import soundfile as sf

SILENCE={'','<eps>','sil','sp','spn','<unk>'}

def hmm_boundary(alignment,prepared):
    words=[r for r in alignment['tiers']['words']['entries'] if r[2] not in SILENCE]
    expected=prepared['tokens']
    if [r[2] for r in words]!=expected:
        return {'gap_ms':None,'status':'transcript_mismatch','observed':[r[2] for r in words]}
    split=prepared['setup_token_count']
    left,right=words[split-1][1],words[split][0]
    phones=[r for r in alignment['tiers']['phones']['entries'] if r[1]>left+1e-6 and r[0]<right-1e-6]
    if right<left:
        return {'gap_ms':None,'status':'overlapping_word_intervals'}
    nonsil=[r for r in phones if r[2].split('_')[0] not in {'','sil','sp'}]
    if nonsil:
        return {'gap_ms':None,'status':'non_silence_phone_in_gap','phones':phones}
    return {'start':left,'end':right,'gap_ms':(right-left)*1000,'status':'ok',
            'last_setup_word':words[split-1][2],'first_ending_word':words[split][2],'gap_phones':phones}

def audit(source_run,mfa_run,ctc_analysis='alignment-v2'):
    src=ROOT/'runs'/source_run
    mfa=ROOT/'runs'/mfa_run
    inputs=read_json(mfa/'inputs.json')
    source=read_json(src/'manifest.json')['items']
    rows=[]
    for path in sorted((mfa/'aligned').rglob('*.json')):
        key=path.stem
        item=source[key]
        old=read_json(src/ctc_analysis/f'{key}.json')
        prepared=inputs[key]
        hmm=hmm_boundary(read_json(path),prepared)
        x,sr=sf.read(ROOT/prepared['source_audio'],dtype='float32')
        boundary=prepared['setup_han_count']
        # Same broad CTC bracket for all waveform estimators. No HMM leakage.
        left=max(0.,old['units'][boundary-1]['start']-.04)
        right=min(len(x)/sr,old['units'][boundary]['end']+.04)
        grid={str(db):acoustic_gap(x,sr,left,right,relative_db=-db) for db in (25,30,35,40)}
        robust=acoustic_gap(x,sr,left,right,hysteresis=True)
        spectral={str(db):spectral_gap(x,sr,left,right,relative_db=-db) for db in (25,30,35,40)}
        rows.append({'key':key,'script_id':item['script_id'],'voice':item['voice'],'kind':item['kind'],
                     'legacy_ctc':{'start':old['setup_end'],'end':old['ending_start'],'gap_ms':old['ctc_gap_ms']},
                     'legacy_rms':old['energy'],'hmm':hmm,'rms_grid':grid,'hysteresis':robust,'spectral_grid':spectral,
                     'source_audio':prepared['source_audio'],'bracket':[left,right]})
    hmm=[r['hmm']['gap_ms'] for r in rows]
    legacy=[r['legacy_rms']['gap_ms'] for r in rows]
    robust=[r['hysteresis']['gap_ms'] for r in rows]
    summary={'n_expected':len(source),'n_rows':len(rows),
             'ctc_minus_legacy_rms':compare([r['legacy_ctc']['gap_ms'] for r in rows],legacy),
             'hmm_minus_legacy_rms':compare(hmm,legacy),
             'hmm_minus_hysteresis':compare(hmm,robust),
             'hmm_minus_spectral_grid':{str(db):compare(hmm,[r['spectral_grid'][str(db)]['gap_ms'] for r in rows]) for db in (25,30,35,40)},
             'hmm_minus_rms_grid':{str(db):compare(hmm,[r['rms_grid'][str(db)]['gap_ms'] for r in rows]) for db in (25,30,35,40)},
             'by_voice':{voice:{'hmm_minus_legacy_rms':compare([r['hmm']['gap_ms'] for r in rows if r['voice']==voice],
                                                              [r['legacy_rms']['gap_ms'] for r in rows if r['voice']==voice]),
                                'hmm_minus_hysteresis':compare([r['hmm']['gap_ms'] for r in rows if r['voice']==voice],
                                                              [r['hysteresis']['gap_ms'] for r in rows if r['voice']==voice])}
                         for voice in sorted({r['voice'] for r in rows})}}
    return rows,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='kokoro-diagnostic-v1')
    p.add_argument('--mfa',default='mfa-diagnostic-v1')
    p.add_argument('--out',default='pause-audit-v1')
    args=p.parse_args()
    rows,summary=audit(args.source,args.mfa)
    out=ROOT/'runs'/args.out
    atomic_json(out/'measurements.json',rows)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code':{p.name:digest(p) for p in (ROOT/'timing/pause_audit.py',ROOT/'timing/acoustic.py')},
                                     'source':args.source,'mfa':args.mfa,
                                     'plan_sha256':digest(ROOT/'docs/pause-resolution-plan.md')})
    print(summary)

if __name__=='__main__':
    main()
