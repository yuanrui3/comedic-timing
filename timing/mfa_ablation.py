"""Run the same exact-zero challenge through both local HMM aligners."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .pause_audit import hmm_boundary
from .pause_evaluate import errors
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def prepare():
    src=ROOT/'runs/kokoro-diagnostic-v1/manifest.json';items=read_json(src)['items']
    edits=[read_json(p) for p in (ROOT/'runs/onset-ablation-v1/cases').glob('*.json')]
    out=ROOT/'runs/onset-ablation-audio-v1';state=Manifest(out/'manifest.json',{'source_sha256':digest(src),
        'charsiu_ablation_manifest_sha256':digest(ROOT/'runs/onset-ablation-v1/manifest.json'),'code_sha256':digest(__file__),
        'timebase':'16k float with exact digital zeros, applied after resampling','groups':'condition x voice'})
    refs=[]
    variants=[{'key':key,'extension_ms':None} for key in items]+edits
    for edit in variants:
        key=edit['key'];item=items[key];suffix='baseline' if edit['extension_ms'] is None else f'mute_plus{edit["extension_ms"]:02}ms'
        name=key+'__'+suffix;refs.append({'key':name,'base_key':key,'variant':suffix,**{k:edit[k] for k in ('mute_start','mute_end') if k in edit}})
        if state.complete(name):continue
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path,dtype='float32');assert sr==24000;x=resample_poly(x,2,3)
        if suffix!='baseline':x[round(edit['mute_start']*16000):round(edit['mute_end']*16000)]=0
        dest=ROOT/'data/audio/onset-ablation-audio-v1'/f'{name}.wav';dest.parent.mkdir(parents=True,exist_ok=True);sf.write(dest,x,16000,subtype='FLOAT')
        state.record(name,status='complete',script_id=item['script_id'],kind=item['kind'],voice=item['voice']+'__'+suffix,
            setup=item['setup'],ending=item['ending'],text=item['text'],artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
    atomic_json(out/'references.json',refs);print('Prepared',len(refs),'ablated/baseline cases')

def evaluate():
    refs=read_json(ROOT/'runs/onset-ablation-audio-v1/references.json');rows=[]
    for model,run in (('MFA2','mfa-ablation-v1'),('MFA3','mfa3-ablation-v1')):
        inp=read_json(ROOT/'runs'/run/'inputs.json')
        aligned={p.stem:hmm_boundary(read_json(p),inp[p.stem]) for p in (ROOT/'runs'/run/'aligned').rglob('*.json')}
        for r in refs:
            if r['variant']=='baseline':continue
            a=aligned[r['key']];baseline=aligned[r['base_key']+'__baseline']
            rows.append({**r,'model':model,'baseline':baseline,'modified':a,
                'onset_shift_ms':1000*(a['end']-baseline['end']),
                'onset_before_unmuted_audio_ms':1000*(r['mute_end']-a['end'])})
    summary={m:{variant:{'onset_vs_known_resume':errors([{'error_ms':-r['onset_before_unmuted_audio_ms']} for r in rows if r['model']==m and r['variant']==variant]),
        'early_onset_over30ms':sum(r['model']==m and r['variant']==variant and r['onset_before_unmuted_audio_ms']>30+1e-6 for r in rows)}
        for variant in ('mute_plus00ms','mute_plus40ms')} for m in ('MFA2','MFA3')}
    d0inputs=read_json(ROOT/'runs/mfa-stability-d0-v1/inputs.json')
    d0={p.stem:hmm_boundary(read_json(p),d0inputs[p.stem]) for p in (ROOT/'runs/mfa-stability-d0-v1/aligned').rglob('*.json')}
    gain_rows=[]
    for k,a in d0.items():
        if not k.endswith('__gain010'):continue
        b=d0[k.replace('__gain010','__baseline')]
        gain_rows.append({'key':k,**{side+'_error_ms':1000*(a[side]-b[side]) for side in ('start','end')},'gap_error_ms':a['gap_ms']-b['gap_ms']})
    summary['MFA2_zero_dither_gain010']={side:errors([{'error_ms':r[side+'_error_ms']} for r in gain_rows]) for side in ('start','end','gap')}
    out=ROOT/'runs/mfa-ablation-analysis-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'gain-dither.json',gain_rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'inputs_sha256':{r:digest(ROOT/'runs'/r/'manifest.json') for r in ('mfa-ablation-v1','mfa3-ablation-v1','mfa-stability-d0-v1')}})
    print(summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate'));args=parser.parse_args()
    {'prepare':prepare,'evaluate':evaluate}[args.action]()
