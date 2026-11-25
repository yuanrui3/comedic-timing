"""Describe cross-family measurement behavior; never infer a comedic-timing effect."""
from .common import ROOT,read_json,atomic_json,digest
from .pause_audit import hmm_boundary
from .quiet_selector_v2 import select
from .measurement import rms_track
from .ac_rms import ac_track
from .detrended_rms import linear_track
from .acoustic import compare
from .kokoro_signals import lowpass
from .corpus import han
import numpy as np
import soundfile as sf

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ac in enumerate(a,1):
        current=[i]
        for j,bc in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ac!=bc)))
        previous=current
    return previous[-1]

def describe(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'median':float(np.median(values)),'min':float(min(values)),'max':float(max(values))} if values else {'n':0}

def main():
    source=read_json(ROOT/'runs/melo-diagnostic-v1/manifest.json')['items'];assert len(source)==60
    models={}
    for model,run in (('MFA2','mfa-melo-v1'),('MFA3','mfa3-melo-v1')):
        prepared=read_json(ROOT/'runs'/run/'inputs.json')
        models[model]={p.stem:hmm_boundary(read_json(p),prepared[p.stem]) for p in (ROOT/'runs'/run/'aligned').rglob('*.json')}
        assert len(models[model])==60
    rows=[]
    for key,item in source.items():
        c=read_json(ROOT/'runs/charsiu-melo-v1/aligned'/f'{key}.json');x,sr=sf.read(ROOT/c['source_audio'])
        methods={m:v[key] for m,v in models.items()};methods['Charsiu']=c['methods']['4']['forced']
        for name,fn in (('raw',rms_track),('AC',ac_track),('linear',linear_track)):
            centers,energy=fn(x,sr)[:2];methods[name]=select(centers,energy,*c['bracket'])
        asr=read_json(ROOT/'runs/melo-diagnostic-v1/alignment-v2'/f'{key}.json')['greedy_transcript']
        target=han(item['text']);observed=han(asr);distance=edit_distance(target,observed)
        from pypinyin import lazy_pinyin,Style
        pinyin_target=lazy_pinyin(''.join(target),style=Style.TONE3,neutral_tone_with_five=True)
        pinyin_observed=lazy_pinyin(''.join(observed),style=Style.TONE3,neutral_tone_with_five=True)
        hmm=methods['MFA2'];assert hmm['status']=='ok'
        a,b=round(hmm['start']*sr),round(hmm['end']*sr);lp=lowpass(x,sr)
        outside=np.ones(len(x),dtype=bool);outside[max(0,a-round(.1*sr)):min(len(x),b+round(.1*sr))]=False
        rms=lambda z:float(np.sqrt(np.mean(np.asarray(z,dtype=float)**2))) if len(z) else None
        slow=rms(lp[a:b]);level=rms(x[outside]);flags=[]
        if methods['raw']['gap_ms'] is None:flags.append('raw_interval_'+methods['raw']['status'])
        elif abs(methods['raw']['gap_ms']-hmm['gap_ms'])>30+1e-6:flags.append('MFA2_raw_gap_difference_over30ms')
        rows.append({'key':key,'script_id':item['script_id'],'kind':item['kind'],'take':item['take'],'seed':item['seed'],
            'source_audio':c['source_audio'],'bracket':c['bracket'],'methods':methods,'charsiu_textless':c['methods']['4']['textless'],
            'ASR':{'transcript':asr,'expected':item['text'],'char_edits':distance,'chars':len(target),'CER':distance/len(target),
                   'toned_pinyin_edits':edit_distance(pinyin_target,pinyin_observed),'note':'Automated model QC, not human proof of synthesis correctness.'},
            'slow_component':{'MFA2_window_sub20_RMS':slow,'outside_RMS':level,'relative_to_outside':slow/level if slow is not None else None},
            'transfer_QC_flags':flags,'original_Kokoro_flags_modified':False})
    comparisons={}
    for method in ('MFA3','Charsiu','raw','AC','linear'):
        comparisons[method]={}
        for field in ('gap_ms','start','end'):
            scale=1 if field=='gap_ms' else 1000
            comparisons[method][field]=compare([r['methods']['MFA2'].get(field)*scale if r['methods']['MFA2'].get(field) is not None else None for r in rows],
                                               [r['methods'][method].get(field)*scale if r['methods'][method].get(field) is not None else None for r in rows])
    take_ranges={}
    for method in ('MFA2','MFA3','Charsiu','raw','AC','linear'):
        values=[]
        for script in sorted({r['script_id'] for r in rows}):
            for kind in ('joke','control'):
                group=[r['methods'][method]['gap_ms'] for r in rows if r['script_id']==script and r['kind']==kind]
                if all(v is not None for v in group):values.append(max(group)-min(group))
        take_ranges[method]=describe(values)
    old_asr=[]
    for p in (ROOT/'runs/kokoro-diagnostic-v1/alignment-v2').glob('zh*.json'):
        r=read_json(p);expected=[u['unit'] for u in r['units']];observed=han(r['greedy_transcript'])
        old_asr.append({'key':p.stem,'edits':edit_distance(expected,observed),'chars':len(expected)})
    old_slow=read_json(ROOT/'runs/signal-origin-v1/cases.json')
    summary={'clips':60,'scripts':10,'stock_voices':1,'takes_per_text':3,'TTS_family':'MeloTTS','whole_script_calls_per_clip':1,
        'ASR_character_error_rate':sum(r['ASR']['char_edits'] for r in rows)/sum(r['ASR']['chars'] for r in rows),
        'ASR_toned_pinyin_error_rate':sum(r['ASR']['toned_pinyin_edits'] for r in rows)/sum(r['ASR']['chars'] for r in rows),
        'ASR_exact_transcripts':sum(r['ASR']['char_edits']==0 for r in rows),
        'Kokoro_same_ASR_character_error_rate':sum(r['edits'] for r in old_asr)/sum(r['chars'] for r in old_asr),
        'transfer_QC_flagged':sum(bool(r['transfer_QC_flags']) for r in rows),
        'raw_missing':sum(r['methods']['raw']['gap_ms'] is None for r in rows),
        'raw_multiple_candidates':sum(r['methods']['raw']['status']=='ambiguous_multiple_intervals' for r in rows),
        'MFA2_comparisons':comparisons,'three_take_gap_ranges_ms':take_ranges,
        'Melo_MFA2_gap_sub20_RMS':describe([r['slow_component']['MFA2_window_sub20_RMS'] for r in rows]),
        'Kokoro_MFA2_gap_sub20_RMS_all40':describe([r['native_sub20_rms'] for r in old_slow]),
        'caveats':['MFA-defined windows are not independently annotated silence','Different synthesis family, one stock voice; not independent subjects',
                   'ASR errors can reflect recognition errors and homophones','No original flag cleared; no main-study or humor inference']}
    out=ROOT/'runs/transfer-analysis-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'input_manifests_sha256':{run:digest(ROOT/'runs'/run/'manifest.json') for run in (
        'melo-diagnostic-v1','mfa-melo-v1','mfa3-melo-v1','charsiu-melo-v1')},'CTC_manifest_sha256':digest(ROOT/'runs/melo-diagnostic-v1/alignment-v2/manifest.json')})
    print({k:v for k,v in summary.items() if k not in ('MFA2_comparisons','caveats')})
    print(comparisons)

if __name__=='__main__':main()
