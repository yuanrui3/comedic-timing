"""Localize style sensitivity within decoder features versus vocoder AdaIN."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .kokoro_signals import lowpass
from .acoustic import acoustic_gap
import torch
import numpy as np
import soundfile as sf

def main():
    from kokoro import KModel
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise RuntimeError('Less than 2 GiB free GPU memory; defer this bounded probe')
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    five=[r['key'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy']
    targets=sorted(set(five+[k.replace('zm_010','zf_001') for k in five]));out=ROOT/'runs/decoder-localization-v1'
    modes=['generator_style_only','decoder_features_style_only','generator_style_gamma_only','generator_style_beta_only']
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'source_manifest_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'model':read_json(ROOT/'runs/model-locks/kokoro_zh.json'),'targets':targets,'modes':modes,
        'seed':20261114,'purpose':'decoder-localization counterfactuals only; never alter original study audio'})
    if all(state.complete(key+'__'+mode) for key in targets for mode in modes):print('Verified 40 completed decoder interventions');return
    torch.set_num_threads(4);torch.manual_seed(20261114);root=ROOT/'models/kokoro_zh'
    model=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh',config=str(root/'config.json'),model=str(root/'kokoro-v1_1-zh.pth')).cuda().eval()
    packs={voice:torch.load(root/'voices'/f'{voice}.pt',weights_only=True).cuda() for voice in ('zf_001','zm_010')}
    adains=[(name,module) for name,module in model.decoder.generator.named_modules() if type(module).__name__=='AdaIN1d']
    atomic_json(out/'intervention-sites.json',{'generator_AdaIN_modules':[name for name,_ in adains],'free_GPU_before_load_bytes':free})
    for key in targets:
        item=source[key];original,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32')
        native=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__baseline.npz')
        original_trace=read_json(ROOT/'runs/kokoro-signals-v1/cases'/f'{key}__baseline.json')
        own=packs[item['voice']][len(item['phonemes'])-1];other_voice='zf_001' if item['voice']=='zm_010' else 'zm_010'
        other=packs[other_voice][len(item['phonemes'])-1][:,:128]
        torch.manual_seed(20261114);baseline=model(item['phonemes'],own,1.,return_output=True)
        assert np.array_equal(baseline.audio.numpy(),original)
        h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr);original_low=lowpass(original)
        for mode in modes:
            name=key+'__'+mode
            if state.complete(name):continue
            captured={};handles=[]
            def decoder_pre(module,inputs):
                captured['F0']=inputs[1].detach().cpu().numpy().squeeze();captured['N']=inputs[2].detach().cpu().numpy().squeeze()
                if mode=='decoder_features_style_only':return (*inputs[:3],other)
            def generator_pre(module,inputs):
                if mode=='generator_style_only':return (inputs[0],other,inputs[2])
                if mode=='decoder_features_style_only':return (inputs[0],own[:,:128],inputs[2])
            def affine_post(module,inputs,output):
                alternate=torch.nn.functional.linear(other,module.weight,module.bias);half=output.shape[1]//2
                if mode=='generator_style_gamma_only':return torch.cat([alternate[:,:half],output[:,half:]],dim=1)
                return torch.cat([output[:,:half],alternate[:,half:]],dim=1)
            handles.append(model.decoder.register_forward_pre_hook(decoder_pre));handles.append(model.decoder.generator.register_forward_pre_hook(generator_pre))
            if mode in ('generator_style_gamma_only','generator_style_beta_only'):
                handles.extend(module.fc.register_forward_hook(affine_post) for _,module in adains)
            try:
                torch.manual_seed(20261114);torch.cuda.reset_peak_memory_stats()
                output=model(item['phonemes'],own,1.,return_output=True)
            finally:
                for handle in handles:handle.remove()
            assert output.pred_dur.tolist()==original_trace['duration_frames']
            assert np.array_equal(captured['F0'],native['F0_curve']) and np.array_equal(captured['N'],native['N_curve'])
            x=output.audio.numpy();lp=lowpass(x);outside=np.ones(len(x),dtype=bool);outside[max(0,a-2400):min(len(x),b+2400)]=False
            def rms(z):return float(np.sqrt(np.mean(np.asarray(z,dtype=float)**2)))
            delta=x.astype(float)-original
            result={'key':key,'mode':mode,'target':key in five,'native_baseline_replay_identical':True,'duration_F0_N_preserved':True,
                'sub20_RMS_ratio':rms(lp[a:b])/rms(original_low[a:b]),'sub20_correlation':float(np.corrcoef(lp[a:b],original_low[a:b])[0,1]),
                'outside_gap_speech_SNR_dB':20*np.log10(rms(original[outside])/max(rms(delta[outside]),1e-20)),
                'raw_RMS_interval':acoustic_gap(x,sr,*audit[key]['bracket']),'peak_GPU_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}
            dest=out/'cases'/f'{name}.json';atomic_json(dest,result)
            wave=ROOT/'data/audio/decoder-localization-v1'/f'{name}.wav';wave.parent.mkdir(parents=True,exist_ok=True);sf.write(wave,x,sr,subtype='FLOAT')
            state.record(name,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,wave)})
            print(name,result['sub20_RMS_ratio'],result['outside_gap_speech_SNR_dB'],flush=True)
    rows=[read_json(p) for p in (out/'cases').glob('*.json')]
    summary={mode:{'targets':sum(r['target'] and r['mode']==mode for r in rows),
        'sub20_ratio_range':[min(r['sub20_RMS_ratio'] for r in rows if r['target'] and r['mode']==mode),max(r['sub20_RMS_ratio'] for r in rows if r['target'] and r['mode']==mode)],
        'outside_gap_SNR_range_dB':[min(r['outside_gap_speech_SNR_dB'] for r in rows if r['target'] and r['mode']==mode),max(r['outside_gap_speech_SNR_dB'] for r in rows if r['target'] and r['mode']==mode)]} for mode in modes}
    atomic_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':main()
