"""Check whether saved decoder-style effects are explained by overall gain."""
from .common import ROOT,read_json,atomic_json,digest
from .kokoro_signals import lowpass
import numpy as np
import soundfile as sf


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))


def main():
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    manifest=ROOT/'runs/decoder-localization-v1/manifest.json'
    rows=[];inputs={str(manifest.relative_to(ROOT)):digest(manifest)}
    for path in sorted((ROOT/'runs/decoder-localization-v1/cases').glob('*.json')):
        case=read_json(path);key=case['key'];mode=case['mode']
        original_path=ROOT/next(iter(source[key]['artifacts']))
        modified_path=ROOT/'data/audio/decoder-localization-v1'/f'{key}__{mode}.wav'
        original,sr=sf.read(original_path,dtype='float32');modified,msr=sf.read(modified_path,dtype='float32')
        assert sr==msr==24000 and len(original)==len(modified)
        h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr)
        outside=np.ones(len(original),dtype=bool);outside[max(0,a-round(.1*sr)):min(len(original),b+round(.1*sr))]=False
        outside_ratio=rms(modified[outside])/rms(original[outside])
        slow_ratio=rms(lowpass(modified)[a:b])/rms(lowpass(original)[a:b])
        assert abs(slow_ratio-case['sub20_RMS_ratio'])<1e-10
        rows.append({'key':key,'mode':mode,'target':case['target'],'outside_RMS_ratio':outside_ratio,
                     'raw_sub20_RMS_ratio':slow_ratio,'level_normalized_sub20_ratio':slow_ratio/outside_ratio,
                     'outside_gain_change_dB':20*np.log10(outside_ratio),
                     'normalized_sub20_change_dB':20*np.log10(slow_ratio/outside_ratio)})
        for p in (path,original_path,modified_path):inputs[str(p.relative_to(ROOT)).replace('\\','/')]=digest(p)
    summary={}
    for mode in sorted({r['mode'] for r in rows}):
        summary[mode]={}
        for target,label in ((True,'five_male_targets'),(False,'five_matched_female_reverse_swaps')):
            subset=[r for r in rows if r['mode']==mode and r['target']==target]
            summary[mode][label]={'n':len(subset),**{name+'_range':[min(r[name] for r in subset),max(r[name] for r in subset)]
                for name in ('outside_RMS_ratio','raw_sub20_RMS_ratio','level_normalized_sub20_ratio')}}
    out=ROOT/'runs/style-level-audit-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'plan_sha256':digest(ROOT/'docs/second-aligner-plan.md'),
                                     'inputs_sha256':inputs,'interpretation':'Fixed same-sample outside-gap level control, not perceptual normalization or a repair.'})
    print(summary)


if __name__=='__main__':main()
