"""Verify actual libsndfile PCM16 behavior and the historical alignment copies."""
from .common import ROOT,read_json,atomic_json,digest
from .kokoro_signals import lowpass
from scipy.signal import resample_poly
import soundfile as sf
import numpy as np
import io

def main():
    rows=[];source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    inputs=read_json(ROOT/'runs/mfa-diagnostic-v1/inputs.json');audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    five={r['key'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json') if r['category']=='fragmented_RMS_subsonic_energy'}
    for key,item in source.items():
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path,dtype='float32');x16=resample_poly(x,2,3)
        buffer=io.BytesIO();sf.write(buffer,x16,16000,format='WAV',subtype='PCM_16');buffer.seek(0);actual,_=sf.read(buffer,dtype='float32')
        approximation=(np.clip(np.floor(x16.astype(float)*32768),-32768,32767)/32768).astype('float32')
        path=ROOT/'.cache/mfa-corpora/mfa-diagnostic-v1'/item['voice']/f'{key}.wav'
        assert digest(path)==inputs[key]['input_wav_sha256'];cached,_=sf.read(path,dtype='float32')
        roundtrip=resample_poly(actual,3,2)[:len(x)];cached_roundtrip=resample_poly(cached,3,2)[:len(x)]
        h=audit[key]['hmm'];a,b=round(h['start']*sr),round(h['end']*sr);original_low=lowpass(x)
        def rms(z):return float(np.sqrt(np.mean(np.asarray(z,dtype=float)**2)))
        row={'key':key,'target':key in five,'floor_approximation_matches_actual':bool(np.array_equal(actual,approximation)),
            'actual_matches_original_MFA_copy':bool(np.array_equal(actual,cached)),
            'actual_vs_floor_max_LSB':float(np.max(np.abs(actual-approximation))*32768),
            'actual_vs_cached_max_LSB':float(np.max(np.abs(actual-cached))*32768),
            'actual_PCM16_relative_sub20_error':rms((original_low-lowpass(roundtrip))[a:b])/rms(original_low[a:b]),
            'historical_MFA_PCM16_relative_sub20_error':rms((original_low-lowpass(cached_roundtrip))[a:b])/rms(original_low[a:b])}
        rows.append(row)
    summary={'cases':len(rows),'floor_approximation_exact_matches':sum(r['floor_approximation_matches_actual'] for r in rows),
        'actual_matches_historical_MFA_count':sum(r['actual_matches_original_MFA_copy'] for r in rows),
        'max_actual_vs_floor_LSB':max(r['actual_vs_floor_max_LSB'] for r in rows),
        'target_max_actual_PCM16_relative_sub20_error':max(r['actual_PCM16_relative_sub20_error'] for r in rows if r['target']),
        'target_max_historical_MFA_PCM16_relative_sub20_error':max(r['historical_MFA_PCM16_relative_sub20_error'] for r in rows if r['target']),
        'interpretation':'Native artifact predates conversion. Actual encoding is distinct from an approximate floor quantizer.'}
    out=ROOT/'runs/pcm-audit-v1';atomic_json(out/'cases.json',rows);atomic_json(out/'summary.json',summary)
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'soundfile_version':sf.__version__,
        'libsndfile_version':sf.__libsndfile_version__,'source_manifest_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'historical_MFA_inputs_sha256':digest(ROOT/'runs/mfa-diagnostic-v1/inputs.json')})
    print(summary)

if __name__=='__main__':main()
