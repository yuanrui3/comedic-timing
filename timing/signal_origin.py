"""Audit native waveform, sampling, window leakage, and generator interventions."""
from .common import ROOT,read_json,atomic_json,digest
from .kokoro_signals import lowpass
from .measurement import rms_track
from scipy.signal import resample_poly,welch
import numpy as np
import soundfile as sf


def main():
    cases=read_json(ROOT/'runs/flagged-followup-v1/cases.json');rows=[]
    for case in cases:
        key=case['key'];x,sr=sf.read(ROOT/case['source_audio'],dtype='float32')
        a,b=round(case['hmm']['start']*sr),round(case['hmm']['end']*sr)
        base=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__baseline.npz')
        assert np.array_equal(base['output'],x)
        x16=resample_poly(x,2,3)
        pcm=(np.clip(np.floor(x16.astype(float)*32768),-32768,32767)/32768).astype('float32')
        roundtrip=resample_poly(x16,3,2)[:len(x)]
        pcm_roundtrip=resample_poly(pcm,3,2)[:len(x)]
        native_lp=lowpass(x);resampled_lp=lowpass(roundtrip);pcm_lp=lowpass(pcm_roundtrip)
        z=x[a:b].astype(float)
        def rms(y):return float(np.sqrt(np.mean(np.asarray(y,dtype=float)**2)))
        centers,raw_rms=rms_track(x,sr)
        mean=np.convolve(x.astype(float),np.ones(round(.01*sr))/round(.01*sr),mode='valid')[::round(.005*sr)]
        region=(centers>=case['hmm']['start'])&(centers<=case['hmm']['end'])
        pad=round(4*sr)
        spectrum=np.abs(np.fft.rfft((z-z.mean())*np.hanning(len(z)),n=pad))**2
        freq=np.fft.rfftfreq(pad,1/sr);band=(freq>=2)&(freq<=40)
        phases=np.unwrap(np.angle(np.fft.rfft(z,n=pad)))
        r={'key':key,'original_target':case['category']=='fragmented_RMS_subsonic_energy',
           'native_PCM_replay_identical':True,'native_sub20_rms':rms(native_lp[a:b]),
           'resampling_sub20_error_rms':rms((native_lp-resampled_lp)[a:b]),
           'pcm16_sub20_error_rms':rms((native_lp-pcm_lp)[a:b]),
           'full_band_rms':rms(z),'10ms_means_rms':rms(mean[region]),
           'frequency_resolution_original_hz':sr/len(z),
           'zero_padded_low_band_peak_hz':float(freq[band][spectrum[band].argmax()]),
           'peak_interpretation':'short nonstationary interval; zero padding does not add frequency resolution'}
        if r['original_target']:
            alternate=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__alternate_noise_seed_20261115.npz')['output']
            other=lowpass(alternate)
            r['alternate_seed_lowpass_correlation']=float(np.corrcoef(native_lp[a:b],other[a:b])[0,1])
            r['alternate_seed_lowpass_change_rms']=rms((native_lp-other)[a:b])
            r['interventions']={}
            for mode in ('source_mean_removed','source_zeroed','decoder_other_stock_style'):
                v=read_json(ROOT/'runs/kokoro-interventions-v1/cases'/f'{key}__{mode}.json')
                r['interventions'][mode]={'sub20_rms_ratio':v['modified_lowpass20_rms']/v['original_lowpass20_rms'],
                                          'outside_gap_speech_SNR_dB':v['speech_outside_gap_SNR_dB'],
                                          'correlation':v['lowpass20_correlation']}
        rows.append(r)
    targets=[r for r in rows if r['original_target']]
    summary={'targets':len(targets),'all_native_replays_identical':all(r['native_PCM_replay_identical'] for r in rows),
        'max_resampling_relative_sub20_error':max(r['resampling_sub20_error_rms']/r['native_sub20_rms'] for r in targets),
        'max_PCM16_relative_sub20_error':max(r['pcm16_sub20_error_rms']/r['native_sub20_rms'] for r in targets),
        'alternate_seed_correlation_range':[min(r['alternate_seed_lowpass_correlation'] for r in targets),max(r['alternate_seed_lowpass_correlation'] for r in targets)],
        'source_zeroed_sub20_ratio_range':[min(r['interventions']['source_zeroed']['sub20_rms_ratio'] for r in targets),max(r['interventions']['source_zeroed']['sub20_rms_ratio'] for r in targets)],
        'decoder_style_swap_sub20_ratio_range':[min(r['interventions']['decoder_other_stock_style']['sub20_rms_ratio'] for r in targets),max(r['interventions']['decoder_other_stock_style']['sub20_rms_ratio'] for r in targets)],
        'conclusion':'Present in native deterministic synthesis; largely insensitive to excitation source, sensitive to decoder style; no claim of a single stationary 10-18 Hz oscillator'}
    out=ROOT/'runs/signal-origin-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'signals_manifest_sha256':digest(ROOT/'runs/kokoro-signals-v1/manifest.json'),
                                     'interventions_manifest_sha256':digest(ROOT/'runs/kokoro-interventions-v1/manifest.json')})
    print(summary)


if __name__=='__main__':main()
