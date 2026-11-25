"""Known edits to actual stock speech, kept separate from study audio."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .measurement import rms_track
import argparse
import numpy as np
import soundfile as sf

DELAYS=[0,30,100,300,600,1000]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='pause-fixtures-v1')
    args=p.parse_args()
    source=ROOT/'runs/kokoro-diagnostic-v1/manifest.json'
    items=read_json(source)['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v2/measurements.json')}
    config={'code_sha256':digest(__file__),'source_sha256':digest(source),
            'audit_sha256':digest(ROOT/'runs/pause-audit-v2/measurements.json'),
            'delays_ms':DELAYS,'incremental_scripts':['zh001','zh006','zh009'],
            'absolute_scripts':['zh003'],
            'absolute_note':'Speech hard-splice fixtures: remove natural gap plus weak edge speech. Endpoints known by construction; not intact natural phonetic ground truth.'}
    manifest=Manifest(ROOT/'runs'/args.run/'manifest.json',config)
    references=[]
    for key,item in items.items():
        mode='incremental' if item['script_id'] in config['incremental_scripts'] else 'absolute' if item['script_id'] in config['absolute_scripts'] else None
        if mode is None:
            continue
        a=audit[key]
        x,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32')
        if mode=='incremental':
            pause=a['spectral_grid']['35']
            if pause['status']!='ok':
                raise ValueError(f'No unique insertion interval: {key}')
            left=right=round((pause['start']+pause['end'])/2*sr)
        else:
            # Hard-cut in confidently non-quiet source speech on both sides.
            centers,rms=rms_track(x,sr)
            threshold=float(np.quantile(rms,.95))*.2
            center=(a['legacy_ctc']['start']+a['legacy_ctc']['end'])/2
            before=np.flatnonzero((centers>=a['bracket'][0])&(centers<center)&(rms>threshold))
            after=np.flatnonzero((centers>center)&(centers<=a['bracket'][1]+.2)&(rms>threshold))
            if not len(before) or not len(after):
                raise ValueError(f'Cannot construct hard-splice reference: {key}')
            left=round(centers[before[-1]]*sr)
            right=round(centers[after[0]]*sr)
        for delay in DELAYS:
            name=f'{key}_{mode}_{delay:04}ms'
            n=round(sr*delay/1000)
            dest=ROOT/'data/audio'/args.run/f'{name}.wav'
            reference={'key':name,'base_key':key,'mode':mode,'inserted_ms':n/sr*1000,
                       'left_cut_sample':left,'right_cut_sample':right,'sample_rate':sr,
                       'reference_start':left/sr,'reference_end':(left+n)/sr}
            references.append(reference)
            if manifest.complete(name):
                continue
            edited=np.r_[x[:left],np.zeros(n,dtype=x.dtype),x[right:]]
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.partial')
            sf.write(tmp,edited,sr,format='WAV',subtype='FLOAT')
            tmp.replace(dest)
            manifest.record(name,status='complete',script_id=item['script_id'],voice=item['voice'],kind=item['kind'],
                            text=item['text'],setup=item['setup'],ending=item['ending'],
                            purpose='validation_only',reference=reference,
                            artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
    atomic_json(ROOT/'runs'/args.run/'references.json',references)
    print(f'{len(references)} edited speech cases: {sum(r["mode"]=="absolute" for r in references)} absolute hard-splices, {sum(r["mode"]=="incremental" for r in references)} increments.')

if __name__=='__main__':
    main()
