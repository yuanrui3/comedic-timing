"""Transcript cross-check and exploratory gap-band diagnostics; no exclusions."""
from .common import ROOT,read_json,atomic_json,digest
from .corpus import han
from .transfer_analysis import edit_distance,describe
from .pause_audit import hmm_boundary
from pypinyin import lazy_pinyin,Style
from scipy.signal import butter,sosfiltfilt
import numpy as np
import soundfile as sf

def edits_at_positions(a,b):
    # Deterministic minimum-edit alignment; this assigns discrepancies, not time.
    d=np.zeros((len(a)+1,len(b)+1),dtype=int);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
    i,j=len(a),len(b);changes={};insertions=[]
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:changes[i-1]=b[j-1]
            i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:changes[i-1]=None;i-=1
        else:insertions.append({'after_target_index':i-1,'token':b[j-1]});j-=1
    assert len(changes)+len(insertions)==edit_distance(a,b)==d[-1,-1]
    return changes,insertions

def pinyin(text):return lazy_pinyin(''.join(han(text)),style=Style.TONE3,neutral_tone_with_five=True)

def main():
    rows=[];transfer={r['key']:r for r in read_json(ROOT/'runs/transfer-analysis-v1/cases.json')}
    flags={r['key']:r['category'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json')}
    original={r['key']:r['hmm'] for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    metadata={k:v for run in ('kokoro-diagnostic-v1','melo-diagnostic-v1') for k,v in read_json(ROOT/'runs'/run/'manifest.json')['items'].items()}
    for path in sorted((ROOT/'runs/whisper-transcript-qc-v1/transcripts').glob('*.json')):
        wh=read_json(path);key=wh['key'];family=wh['family'];run='melo-diagnostic-v1' if family=='Melo' else 'kokoro-diagnostic-v1'
        ctc=read_json(ROOT/'runs'/run/'alignment-v2'/f'{key}.json')['greedy_transcript']
        target=han(wh['target_text']);tp=pinyin(wh['target_text']);asr={};bypos={};split=len(han(metadata[key]['setup']))
        for name,text in (('CTC',ctc),('Whisper',wh['transcript'])):
            hp=pinyin(text);changes,insertions=edits_at_positions(tp,hp);bypos[name]=changes
            char_changes,char_insertions=edits_at_positions(target,han(text))
            asr[name]={'text':text,'char_edits':edit_distance(target,han(text)),'toned_pinyin_edits':edit_distance(tp,hp),
                'toneless_pinyin_edits':edit_distance([s.rstrip('12345') for s in tp],[s.rstrip('12345') for s in hp]),
                'pinyin_changes':changes,'pinyin_insertions':insertions,'exact_characters':target==han(text),
                'ending_character_edits':sum(i>=split for i in char_changes)+sum(v['after_target_index']>=split-1 for v in char_insertions),
                'near_boundary_character_edits':sum(split-2<=i<split+2 for i in char_changes)+sum(split-3<=v['after_target_index']<split+2 for v in char_insertions)}
        shared=[{'target_index':i,'target_character':target[i],'target_pinyin':tp[i],'recognized_pinyin':v}
                for i,v in bypos['CTC'].items() if i in bypos['Whisper'] and bypos['Whisper'][i]==v]
        for name in asr:asr[name]['character_rate']=asr[name]['char_edits']/len(target);asr[name]['pinyin_rate']=asr[name]['toned_pinyin_edits']/len(target)
        window=transfer[key]['methods']['MFA2'] if family=='Melo' else original[key]
        x,sr=sf.read(ROOT/wh['source_audio']);a=round((window['start']+.01)*sr);b=round((window['end']-.01)*sr)
        level=None
        if b>a:
            filtered=sosfiltfilt(butter(4,20,'highpass',fs=sr,output='sos'),x)
            rms=lambda y:float(np.sqrt(np.mean(y**2)))
            raw=rms(x[a:b]);hp=rms(filtered[a:b]);level={'raw_RMS':raw,'HP20_RMS':hp,'ratio':hp/raw,'samples':b-a}
        rows.append({'key':key,'family':family,'original_category':flags.get(key),'target':wh['target_text'],'characters':len(target),
            'ASR':asr,'shared_pinyin_substitutions_or_deletions':shared,'HP20_gap_level':level,
            'audio':wh['source_audio'],'Whisper_reached_token_limit':wh['reached_token_limit']})
    assert len(rows)==100 and not any(r['Whisper_reached_token_limit'] for r in rows)
    summary={}
    for family in ('Kokoro','Melo'):
        group=[r for r in rows if r['family']==family];den=sum(r['characters'] for r in group)
        summary[family]={'clips':len(group),'target_characters':den,'recognizers':{m:{
            'character_error_rate':sum(r['ASR'][m]['char_edits'] for r in group)/den,
            'toned_pinyin_error_rate':sum(r['ASR'][m]['toned_pinyin_edits'] for r in group)/den,
            'toneless_pinyin_error_rate':sum(r['ASR'][m]['toneless_pinyin_edits'] for r in group)/den,
            'clips_with_ending_character_edits':sum(r['ASR'][m]['ending_character_edits']>0 for r in group),
            'clips_with_near_boundary_character_edits':sum(r['ASR'][m]['near_boundary_character_edits']>0 for r in group),
            'exact_character_transcripts':sum(r['ASR'][m]['exact_characters'] for r in group)} for m in ('CTC','Whisper')},
            'clips_with_shared_pinyin_substitution_or_deletion':sum(bool(r['shared_pinyin_substitutions_or_deletions']) for r in group),
            'shared_target_positions':sum(len(r['shared_pinyin_substitutions_or_deletions']) for r in group),
            'HP20_RMS_ratio':describe([r['HP20_gap_level']['ratio'] for r in group if r['HP20_gap_level']])}
    artifact_rows=[r for r in rows if r['original_category']=='fragmented_RMS_subsonic_energy'];assert len(artifact_rows)==5
    summary['Kokoro_five_artifacts_HP20_RMS_ratio']=describe([r['HP20_gap_level']['ratio'] for r in artifact_rows if r['HP20_gap_level']])
    summary['caveats']=['ASR text mismatch is not independently verified synthesis error',
        'Toned pinyin reduces script/homophone differences but does not model acoustic tone sandhi or resolve polyphones',
        'Shared edit locations depend on a deterministic minimum-edit alignment and are QC signals, not annotations',
        'Toneless and ending/near-boundary subsets are post-hoc descriptions; near means last two setup plus first two ending characters, with adjacent insertions',
        'High-pass diagnostic is post hoc, on full waveforms, evaluated inside MFA2 windows trimmed by 10ms each side; not silence truth',
        'High-pass filtering is not promoted; original signals, timing estimates, and 34 flags remain unchanged']
    out=ROOT/'runs/context-transfer-qc-v2';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (
        ROOT/'runs/whisper-transcript-qc-v1/manifest.json',ROOT/'runs/transfer-analysis-v1/cases.json',
        ROOT/'runs/pause-audit-v3/measurements.json',ROOT/'runs/kokoro-diagnostic-v1/alignment-v2/manifest.json',ROOT/'runs/melo-diagnostic-v1/alignment-v2/manifest.json')}})
    print(summary)

if __name__=='__main__':main()
