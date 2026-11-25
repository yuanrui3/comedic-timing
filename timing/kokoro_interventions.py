"""Counterfactual source/style probes, never replacement study recordings."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .kokoro_signals import lowpass
from .acoustic import acoustic_gap
import numpy as np
import soundfile as sf
import torch


def main():
    from kokoro import KModel
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    five=[r['key'] for r in read_json(ROOT/'runs/flagged-followup-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy']
    targets=sorted(set(five+[k.replace('zm_010','zf_001') for k in five]))
    out=ROOT/'runs/kokoro-interventions-v1'
    state=Manifest(out/'manifest.json',{'source_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'code_sha256':digest(__file__),'seed':20261114,'targets':targets,
        'modes':['source_mean_removed','source_zeroed','decoder_other_stock_style'],
        'scope':'mechanistic counterfactuals only; durations and upstream F0/N fixed'})
    root=ROOT/'models/kokoro_zh';torch.set_num_threads(4);torch.manual_seed(20261114)
    model=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh',config=str(root/'config.json'),model=str(root/'kokoro-v1_1-zh.pth')).to('cuda').eval()
    packs={v:torch.load(root/'voices'/f'{v}.pt',weights_only=True).to('cuda') for v in ('zf_001','zm_010')}
    for key in targets:
        item=source[key];base,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32')
        saved=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__baseline.npz')
        base_trace=read_json(ROOT/'runs/kokoro-signals-v1/cases'/f'{key}__baseline.json')
        h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr)
        base_lp=lowpass(base)
        for mode in state.data['config']['modes']:
            run_key=key+'__'+mode
            if state.complete(run_key):continue
            captured={};handles=[]
            if mode.startswith('source_'):
                def change_source(module,inputs,outputs):
                    harmonic,noise,uv=outputs
                    if mode=='source_zeroed':harmonic=torch.zeros_like(harmonic)
                    else:harmonic=harmonic-harmonic.mean(dim=1,keepdim=True)
                    return harmonic,noise,uv
                handles.append(model.decoder.generator.m_source.register_forward_hook(change_source))
            def decoder_pre(module,inputs):
                captured['F0']=inputs[1].detach().cpu().numpy().squeeze()
                captured['N']=inputs[2].detach().cpu().numpy().squeeze()
                if mode=='decoder_other_stock_style':
                    other='zf_001' if item['voice']=='zm_010' else 'zm_010'
                    ref=packs[other][len(item['phonemes'])-1][:,:128]
                    return (*inputs[:3],ref)
            handles.append(model.decoder.register_forward_pre_hook(decoder_pre))
            try:
                torch.manual_seed(20261114)
                output=model(item['phonemes'],packs[item['voice']][len(item['phonemes'])-1],1.,return_output=True)
            finally:
                for handle in handles:handle.remove()
            x=output.audio.numpy()
            if output.pred_dur.tolist()!=base_trace['duration_frames']:raise ValueError('Changed token durations')
            if not np.array_equal(saved['F0_curve'],captured['F0']) or not np.array_equal(saved['N_curve'],captured['N']):
                raise ValueError('Upstream F0/noise conditioning changed')
            lp=lowpass(x);speech=np.ones(len(x),dtype=bool);speech[max(0,a-2400):min(len(x),b+2400)]=False
            delta=x.astype(float)-base
            orig_power=np.mean(base[speech].astype(float)**2);err_power=np.mean(delta[speech]**2)
            result={'key':key,'mode':mode,'original_target':key in five,'upstream_and_duration_identical':True,
                'original_lowpass20_rms':float(np.sqrt(np.mean(base_lp[a:b]**2))),
                'modified_lowpass20_rms':float(np.sqrt(np.mean(lp[a:b]**2))),
                'lowpass20_correlation':float(np.corrcoef(lp[a:b],base_lp[a:b])[0,1]),
                'speech_outside_gap_SNR_dB':float(10*np.log10(orig_power/max(err_power,1e-20))),
                'rms_interval':acoustic_gap(x,sr,*audit[key]['bracket']),
                'interpretation':'signal-source intervention, not an accuracy claim or replacement stimulus'}
            wave=ROOT/'data/audio/kokoro-interventions-v1'/f'{run_key}.wav';wave.parent.mkdir(parents=True,exist_ok=True)
            sf.write(wave,x,sr,subtype='FLOAT');dest=out/'cases'/f'{run_key}.json';atomic_json(dest,result)
            state.record(run_key,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (wave,dest)})
            print(key,mode,result['modified_lowpass20_rms']/result['original_lowpass20_rms'],flush=True)


if __name__=='__main__':main()
