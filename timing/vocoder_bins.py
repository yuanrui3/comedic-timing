"""Linear inverse-STFT decomposition of native local synthesis."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .kokoro_signals import lowpass
import numpy as np
import soundfile as sf
import torch

def main():
    from kokoro import KModel
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    five=[r['key'] for r in read_json(ROOT/'runs/flagged-followup-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy']
    targets=sorted(set(five+[k.replace('zm_010','zf_001') for k in five]));out=ROOT/'runs/vocoder-bins-v1'
    state=Manifest(out/'manifest.json',{'source_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'code_sha256':digest(__file__),'model_lock':read_json(ROOT/'runs/model-locks/kokoro_zh.json'),'targets':targets,
        'purpose':'diagnose cancellation; no corrected or replacement stimuli'})
    torch.set_num_threads(4);torch.manual_seed(20261114);root=ROOT/'models/kokoro_zh'
    model=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh',config=str(root/'config.json'),model=str(root/'kokoro-v1_1-zh.pth')).cuda().eval()
    generator=model.decoder.generator;captured={}
    def hook(module,inputs,output):captured['logits']=output.detach()
    handle=generator.conv_post.register_forward_hook(hook);packs={}
    try:
        for key in targets:
            if state.complete(key):continue
            item=source[key];path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
            original,sr=sf.read(path,dtype='float32')
            if item['voice'] not in packs:packs[item['voice']]=torch.load(root/'voices'/f'{item["voice"]}.pt',weights_only=True).cuda()
            torch.manual_seed(20261114);result=model(item['phonemes'],packs[item['voice']][len(item['phonemes'])-1],1.,return_output=True)
            assert np.array_equal(original,result.audio.numpy())
            logits=captured['logits'];bins=generator.post_n_fft//2+1;mag=torch.exp(logits[:,:bins]);phase=torch.sin(logits[:,bins:])
            individual=[]
            with torch.no_grad():
                for i in range(bins):
                    isolated=torch.zeros_like(mag);isolated[:,i]=mag[:,i]
                    individual.append(generator.stft.inverse(isolated,phase).squeeze().cpu().numpy())
            individual=np.asarray(individual);reconstructed=individual.astype(float).sum(0)
            error=float(np.max(np.abs(reconstructed-original)))
            if error>1e-5:raise ValueError('Bin sum does not reproduce native waveform')
            filtered=np.asarray([lowpass(x) for x in individual]);full=lowpass(original)
            h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr)
            def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
            components=[rms(x[a:b]) for x in filtered];total=rms(full[a:b]);joint01=rms(filtered[:2,a:b].sum(0))
            report={'key':key,'original_target':key in five,'native_replay_identical':True,'bin_sum_max_sample_error':error,
                'fft_samples':generator.post_n_fft,'bin_spacing_hz':sr/generator.post_n_fft,'hop_samples':generator.stft.hop_length,
                'sub20_rms':total,'bin_sub20_rms':components,'joint_bins_0_1_sub20_rms':joint01,
                'cancellation_ratio_sum_component_rms_over_total':sum(components)/total,
                'bin0_bin1_lowpass_correlation':float(np.corrcoef(filtered[0,a:b],filtered[1,a:b])[0,1]),
                'residual_without_bin0_sub20_rms':rms(full[a:b]-filtered[0,a:b]),
                'residual_without_bins01_sub20_rms':rms(full[a:b]-filtered[:2,a:b].sum(0)),
                'note':'Individual synthesis bins carry time-varying envelopes; they are not audio-band brick-wall filters.'}
            dest=out/'cases'/f'{key}.json';atomic_json(dest,report)
            trace=ROOT/'data/audio/vocoder-bins-v1'/f'{key}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace,bin_lowpass20=filtered.astype('float32'),native_lowpass20=full.astype('float32'))
            state.record(key,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,trace)})
            print(key,report['cancellation_ratio_sum_component_rms_over_total'],report['bin0_bin1_lowpass_correlation'],flush=True)
    finally:handle.remove()

if __name__=='__main__':main()
