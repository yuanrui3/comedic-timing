from .common import ROOT, Manifest, read_json, atomic_json, digest, source_hash
from .measurement import energy_gap
import argparse
import numpy as np
import soundfile as sf

INJECTIONS = [0,30,100,300,600,1000]

def fixtures():
    rows=[]
    for amplitude in (.2,.002):
        for sr in (16000,24000):
            t=np.arange(round(sr*.4))/sr
            signal=amplitude*np.sin(2*np.pi*440*t+.5)
            for delay in INJECTIONS:
                x=np.r_[signal,np.zeros(round(sr*delay/1000)),signal]
                result=energy_gap(x,sr,.4+delay/2000)
                error=result['gap_ms']-delay
                rows.append({'sr':sr,'amplitude':amplitude,'injected_ms':delay,
                             'measured_ms':result['gap_ms'],'error_ms':error,'pass_30ms':abs(error)<=30})
    atomic_json(ROOT/'runs/validation/tone-fixtures.json',{'scope':'Arithmetic/RMS fixtures, not speech alignment validation',
                                                         'rows':rows,'all_pass':all(r['pass_30ms'] for r in rows)})
    print(f'Tone fixtures: {sum(r["pass_30ms"] for r in rows)}/{len(rows)} within 30ms')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='kokoro-diagnostic-v1')
    parser.add_argument('--fixtures-only',action='store_true')
    args=parser.parse_args()
    fixtures()
    if args.fixtures_only:
        return
    import torch
    from .align import Aligner
    torch.set_num_threads(4)
    source=ROOT/'runs'/args.run/'manifest.json'
    inputs=read_json(source)
    selection=[k for k in inputs['items'] if k.startswith('zh001_') and k.endswith('_joke')]
    config={'source_manifest_sha256':digest(source),'code_sha256':source_hash(),
            'model_lock':read_json(ROOT/'runs/model-locks/align_zh_full.json'),
            'selection':selection,'injected_ms':INJECTIONS,
            'criterion':'change in detected gap versus baseline equals inserted duration within 30ms',
            'limitation':'Insertion point automatically localized; tests incremental recovery, not absolute phonetic accuracy.'}
    out=ROOT/'runs'/args.run/'silence-validation'
    manifest=Manifest(out/'manifest.json',config)
    aligner=Aligner()
    results=[]
    for key in selection:
        item=inputs['items'][key]
        path=ROOT/next(iter(item['artifacts']))
        if digest(path)!=next(iter(item['artifacts'].values())):
            raise ValueError('Waveform checksum mismatch')
        x,sr=sf.read(path,dtype='float32')
        baseline=aligner.measure(x,sr,item['setup'],item['ending'])
        e=baseline['energy']
        # Do not cut a consonant if no low-energy internal interval was detected.
        if e['status']!='ok':
            manifest.record(key,status='failed',error='No internal low-energy insertion interval')
            continue
        cut=round((e['start']+e['end'])/2*sr)
        for delay in INJECTIONS:
            k=f'{key}_{delay}ms'
            dest=out/f'{k}.json'
            if manifest.complete(k):
                results.append(read_json(dest))
                continue
            changed=np.r_[x[:cut],np.zeros(round(sr*delay/1000),dtype=x.dtype),x[cut:]]
            measured=aligner.measure(changed,sr,item['setup'],item['ending'])
            actual=round(sr*delay/1000)/sr*1000
            ctc_error=measured['ctc_gap_ms']-baseline['ctc_gap_ms']-actual
            energy_error=None if measured['energy']['gap_ms'] is None else measured['energy']['gap_ms']-e['gap_ms']-actual
            result={'key':key,'injected_ms':actual,'cut_sample':cut,'ctc_increment_error_ms':ctc_error,
                    'rms_increment_error_ms':energy_error,'ctc_pass_30ms':abs(ctc_error)<=30,
                    'rms_pass_30ms':energy_error is not None and abs(energy_error)<=30,
                    'baseline_ctc_gap_ms':baseline['ctc_gap_ms'],'changed_ctc_gap_ms':measured['ctc_gap_ms'],
                    'baseline_energy':e,'changed_energy':measured['energy']}
            atomic_json(dest,result)
            manifest.record(k,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
            results.append(result)
            print(f'{k}: CTC error={ctc_error:.1f}ms RMS error={energy_error}',flush=True)
    atomic_json(out/'summary.json',{'criterion':config['criterion'],'limitation':config['limitation'],
                                  'n':len(results),'expected_n':len(selection)*len(INJECTIONS),'rows':results,
                                  'all_pass':bool(results) and len(results)==len(selection)*len(INJECTIONS) and all(r['ctc_pass_30ms'] and r['rms_pass_30ms'] for r in results)})

if __name__ == '__main__':
    main()
