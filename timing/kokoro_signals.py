"""Trace and intervene on local synthesis signals, preserving original stimuli."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .acoustic import acoustic_gap
from scipy.signal import butter,sosfiltfilt,resample_poly
import argparse
import numpy as np
import soundfile as sf
import torch


def lowpass(x,sr=24000):
    return sosfiltfilt(butter(4,20,'lowpass',fs=sr,output='sos'),np.asarray(x,dtype=float))


def region_stats(audio, f0, source, dc, left, right, sr=24000):
    a,b=round(left*sr),round(right*sr)
    y=np.asarray(audio[a:b],dtype=float)
    lp=lowpass(audio,sr)[a:b]
    src=lowpass(source,sr)[a:b]
    dc_lp=lowpass(dc,sr)[a:b]
    f0_region=f0[a:b]
    def rms(z):return float(np.sqrt(np.mean(np.asarray(z,dtype=float)**2)))
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
    power=np.abs(np.fft.rfft((y-y.mean())*np.hanning(len(y))))**2
    freq=np.fft.rfftfreq(len(y),1/sr)
    return {'raw_rms':rms(y),'lowpass20_rms':rms(lp),'lowpass20_to_raw_ratio':rms(lp)/max(rms(y),1e-12),
            'mean':float(y.mean()),'centered_hann_peak_hz':float(freq[power.argmax()]),
            'f0_median_hz':float(np.median(f0_region)),'f0_range_hz':[float(f0_region.min()),float(f0_region.max())],
            'fraction_f0_10_to_50':float(np.mean((f0_region>10)&(f0_region<50))),
            'source_lowpass20_rms':rms(src),'dc_bin_lowpass20_rms':rms(dc_lp),
            'rawLP_sourceLP_correlation':corr(lp,src),'rawLP_dcLP_correlation':corr(lp,dc_lp),
            'lowpass20_after_DC_removal_rms':rms(lp-dc_lp)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='kokoro-signals-v1')
    args=parser.parse_args()
    from kokoro import KModel
    root=ROOT/'models/kokoro_zh'
    original=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    targets={r['key'] for r in read_json(ROOT/'runs/flagged-followup-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy'}
    out=ROOT/'runs'/args.out
    spec={'model':read_json(ROOT/'runs/model-locks/kokoro_zh.json'),'source_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
          'code_sha256':digest(__file__),'kokoro_code_sha256':digest(ROOT/'.venv/Lib/site-packages/kokoro/istftnet.py'),
          'seed':20261114,'interventions':['baseline','source_voicing_threshold_50Hz','alternate_noise_seed_20261115'],
          'purpose':'causal signal diagnostics; never replace original stimuli'}
    state=Manifest(out/'manifest.json',spec)
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    model=KModel(repo_id=spec['model']['model']['repo'],config=str(root/'config.json'),model=str(root/'kokoro-v1_1-zh.pth')).to('cuda').eval()
    generator=model.decoder.generator
    captured={}
    def source_pre(module,inputs):captured['source_f0']=inputs[0].detach()
    def source_post(module,inputs,outputs):captured['source']=outputs[0].detach()
    def decoder_pre(module,inputs):
        captured['F0_curve']=inputs[1].detach();captured['N_curve']=inputs[2].detach()
    def conv_post(module,inputs,output):captured['spectral_logits']=output.detach()
    handles=[generator.m_source.register_forward_pre_hook(source_pre),generator.m_source.register_forward_hook(source_post),
             model.decoder.register_forward_pre_hook(decoder_pre),generator.conv_post.register_forward_hook(conv_post)]
    packs={}
    try:
        for key,item in original['items'].items():
            for mode in (spec['interventions'] if key in targets else ['baseline']):
                run_key=key+'__'+mode
                if state.complete(run_key):continue
                captured.clear()
                generator.m_source.l_sin_gen.voiced_threshold=50 if mode=='source_voicing_threshold_50Hz' else 10
                torch.manual_seed(20261115 if mode.startswith('alternate') else spec['seed'])
                if item['voice'] not in packs:packs[item['voice']]=torch.load(root/'voices'/f'{item["voice"]}.pt',weights_only=True).to('cuda')
                result=model(item['phonemes'],packs[item['voice']][len(item['phonemes'])-1],1.,return_output=True)
                audio=result.audio.numpy()
                path=ROOT/next(iter(item['artifacts']))
                if digest(path)!=next(iter(item['artifacts'].values())):raise ValueError('Original audio hash mismatch')
                base,sr=sf.read(path,dtype='float32')
                if mode=='baseline' and not np.array_equal(audio,base):raise ValueError('Baseline PCM mismatch')
                logits=captured['spectral_logits']
                bins=generator.post_n_fft//2+1
                spec_mag=torch.exp(logits[:,:bins,:]);phase=torch.sin(logits[:,bins:,:])
                with torch.no_grad():
                    replay=generator.stft.inverse(spec_mag,phase).squeeze().cpu().numpy()
                    dc_spec=torch.zeros_like(spec_mag);dc_spec[:,0,:]=spec_mag[:,0,:]
                    dc=generator.stft.inverse(dc_spec,phase).squeeze().cpu().numpy()
                if not np.array_equal(audio,replay):raise ValueError('Inverse-STFT replay mismatch')
                f0=captured['source_f0'].squeeze().cpu().numpy()
                source=captured['source'].squeeze().cpu().numpy()
                if len(f0)!=len(audio):raise ValueError('Source/output timebase mismatch')
                h=audit[key]['hmm']
                stats=region_stats(audio,f0,source,dc,h['start'],h['end'])
                result_json={'key':key,'mode':mode,'original_target':key in targets,'pcm_identical':bool(np.array_equal(audio,base)),
                    'duration_frames':result.pred_dur.tolist(),'stats_at_original_HMM_interval':stats,
                    'rms_interval':acoustic_gap(audio,sr,*audit[key]['bracket']),
                    'changed_audio_rms':float(np.sqrt(np.mean((audio.astype(float)-base)**2))),
                    'source_voicing_threshold_hz':generator.m_source.l_sin_gen.voiced_threshold,
                    'timebase':'native audio and source=24000 Hz; F0/noise curves=80 Hz'}
                signal_path=ROOT/'data/audio'/args.out/f'{run_key}.npz'
                signal_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(signal_path,F0_curve=captured['F0_curve'].squeeze().cpu().numpy(),
                    N_curve=captured['N_curve'].squeeze().cpu().numpy(),source_f0=f0,source=source,dc_bin_audio=dc,
                    output=audio,output_lowpass20=lowpass(audio).astype('float32'))
                json_path=out/'cases'/f'{run_key}.json';atomic_json(json_path,result_json)
                artifacts=[signal_path,json_path]
                if mode!='baseline':
                    wav=signal_path.with_suffix('.wav');sf.write(wav,audio,sr,subtype='FLOAT');artifacts.append(wav)
                state.record(run_key,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in artifacts})
                print(run_key,stats['f0_median_hz'],stats['fraction_f0_10_to_50'],result_json['rms_interval']['status'],flush=True)
    finally:
        for handle in handles:handle.remove()


if __name__=='__main__':main()
