"""Bounded stock-style exchanges at each vocoder residual block and stage."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .kokoro_signals import lowpass
from .style_level_audit import rms
import torch
import numpy as np
import soundfile as sf


def main():
    from kokoro import KModel
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise RuntimeError('Less than 2 GiB free GPU memory; defer stage audit')
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    five=[r['key'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy']
    targets=sorted(set(five+[k.replace('zm_010','zf_001') for k in five]))
    groups={**{f'noise_{i}':[f'noise_res.{i}'] for i in range(2)},
            **{f'residual_{i}':[f'resblocks.{i}'] for i in range(6)},
            'stage_0_residuals':[f'resblocks.{i}' for i in range(3)],
            'stage_1_residuals':[f'resblocks.{i}' for i in range(3,6)]}
    out=ROOT/'runs/vocoder-stage-audit-v1'
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),
        'source_manifest_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'model':read_json(ROOT/'runs/model-locks/kokoro_zh.json'),'groups':groups,'targets':targets,'seed':20261114,
        'plan_sha256':digest(ROOT/'docs/second-aligner-plan.md'),'purpose':'Diagnostic stock-style swaps; no original audio changes'})
    torch.set_num_threads(4);torch.manual_seed(20261114);root=ROOT/'models/kokoro_zh'
    model=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh',config=str(root/'config.json'),model=str(root/'kokoro-v1_1-zh.pth')).cuda().eval()
    packs={voice:torch.load(root/'voices'/f'{voice}.pt',weights_only=True).cuda() for voice in ('zf_001','zm_010')}
    modules=dict(model.decoder.generator.named_modules())
    assert model.decoder.generator.num_upsamples==2 and model.decoder.generator.num_kernels==3
    for key in targets:
        if all(state.complete(key+'__'+group) for group in groups):continue
        item=source[key];original,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32')
        native=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__baseline.npz')
        trace=read_json(ROOT/'runs/kokoro-signals-v1/cases'/f'{key}__baseline.json')
        own=packs[item['voice']][len(item['phonemes'])-1]
        other=packs['zf_001' if item['voice']=='zm_010' else 'zm_010'][len(item['phonemes'])-1][:,:128]
        def own_style_hook(module,inputs):return (inputs[0],own[:,:128])
        sham_handles=[modules[name].register_forward_pre_hook(own_style_hook) for name in [f'noise_res.{i}' for i in range(2)]+[f'resblocks.{i}' for i in range(6)]]
        try:
            torch.manual_seed(20261114);sham=model(item['phonemes'],own,1.,return_output=True)
        finally:
            for handle in sham_handles:handle.remove()
        assert np.array_equal(sham.audio.numpy(),original)
        h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr)
        original_low=lowpass(original);outside=np.ones(len(original),dtype=bool)
        outside[max(0,a-round(.1*sr)):min(len(original),b+round(.1*sr))]=False
        for group,names in groups.items():
            name=key+'__'+group
            if state.complete(name):continue
            captured={}
            def capture(module,inputs):
                captured['F0']=inputs[1].detach().cpu().numpy().squeeze();captured['N']=inputs[2].detach().cpu().numpy().squeeze()
            def swap(module,inputs):return (inputs[0],other)
            handles=[model.decoder.register_forward_pre_hook(capture)]
            handles.extend(modules[n].register_forward_pre_hook(swap) for n in names)
            try:
                torch.manual_seed(20261114);result=model(item['phonemes'],own,1.,return_output=True)
            finally:
                for handle in handles:handle.remove()
            assert result.pred_dur.tolist()==trace['duration_frames']
            assert np.array_equal(captured['F0'],native['F0_curve']) and np.array_equal(captured['N'],native['N_curve'])
            x=result.audio.numpy();assert len(x)==len(original) and np.isfinite(x).all()
            ratio=rms(lowpass(x)[a:b])/rms(original_low[a:b]);level=rms(x[outside])/rms(original[outside])
            row={'key':key,'group':group,'target':key in five,'sham_all_blocks_replay_identical':True,'duration_F0_N_preserved':True,
                 'raw_sub20_RMS_ratio':ratio,'outside_RMS_ratio':level,'level_normalized_sub20_ratio':ratio/level,
                 'outside_gap_waveform_SNR_dB':20*np.log10(rms(original[outside])/max(rms((x.astype(float)-original)[outside]),1e-20))}
            path=out/'cases'/f'{name}.json';atomic_json(path,row)
            wave=ROOT/'data/audio/vocoder-stage-audit-v1'/f'{name}.wav';wave.parent.mkdir(parents=True,exist_ok=True);sf.write(wave,x,sr,subtype='FLOAT')
            state.record(name,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (path,wave)})
            print(name,'normalized sub20 ratio',round(ratio/level,3),flush=True)
    rows=[read_json(p) for p in (out/'cases').glob('*.json')];assert len(rows)==100
    summary={}
    for group in groups:
        summary[group]={}
        for target,label in ((True,'male_targets'),(False,'female_reverse_swaps')):
            subset=[r for r in rows if r['group']==group and r['target']==target]
            summary[group][label]={'n':len(subset),**{field+'_range':[min(r[field] for r in subset),max(r[field] for r in subset)]
                for field in ('raw_sub20_RMS_ratio','outside_RMS_ratio','level_normalized_sub20_ratio','outside_gap_waveform_SNR_dB')}}
    atomic_json(out/'summary.json',summary);print('Completed 100 stage interventions',flush=True)


if __name__=='__main__':main()
