"""Frozen expansion of recovery fixtures to six previously unused scripts."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .measurement import rms_track
import numpy as np
import soundfile as sf

def main():
    source=ROOT/'runs/kokoro-diagnostic-v1/manifest.json';items=read_json(source)['items']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    scripts=['zh002','zh004','zh005','zh007','zh008','zh010'];delays=[0,30,100,300,600,1000]
    out=ROOT/'runs/pause-holdout-v1';state=Manifest(out/'manifest.json',{'source_sha256':digest(source),'code_sha256':digest(__file__),
        'scripts':scripts,'delays_ms':delays,'cutting_rules':'same fixed signal rules as original fixtures; no new aligner used to choose cuts',
        'purpose':'held-out edited speech recovery; absolute splices remove weak phone portions'})
    refs=[];brackets=[]
    for key,item in items.items():
        if item['script_id'] not in scripts:continue
        path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path,dtype='float32');a=audit[key];centers,energy=rms_track(x,sr)
        for mode in ('incremental','absolute'):
            if mode=='incremental':
                p=a['spectral_grid']['35'];assert p['status']=='ok';left=right=round((p['start']+p['end'])/2*sr)
            else:
                threshold=float(np.quantile(energy,.95))*.2;center=(a['legacy_ctc']['start']+a['legacy_ctc']['end'])/2
                before=np.flatnonzero((centers>=a['bracket'][0])&(centers<center)&(energy>threshold))
                after=np.flatnonzero((centers>center)&(centers<=a['bracket'][1]+.2)&(energy>threshold))
                if not len(before) or not len(after):raise ValueError('No confident speech for fixed hard splice: '+key)
                left,right=round(centers[before[-1]]*sr),round(centers[after[0]]*sr)
            for delay in delays:
                n=round(sr*delay/1000);name=f'{key}_{mode}_{delay:04}ms'
                reference={'key':name,'base_key':key,'mode':mode,'inserted_ms':1000*n/sr,'left_cut_sample':left,'right_cut_sample':right,
                    'sample_rate':sr,'reference_start':left/sr,'reference_end':(left+n)/sr};refs.append(reference)
                def mapped(t):
                    sample=t*sr
                    if sample<=left:return t
                    if sample>=right:return (sample+n-(right-left))/sr
                    return (left+n*(sample-left)/(right-left))/sr
                bracket=[mapped(t) for t in a['bracket']]
                brackets.append({'key':name,'source_audio':f'data/audio/pause-holdout-v1/{name}.wav','bracket':bracket})
                if state.complete(name):continue
                y=np.r_[x[:left],np.zeros(n,dtype=x.dtype),x[right:]];dest=ROOT/brackets[-1]['source_audio'];dest.parent.mkdir(parents=True,exist_ok=True)
                sf.write(dest,y,sr,subtype='FLOAT')
                state.record(name,status='complete',script_id=item['script_id'],voice=item['voice'],kind=item['kind'],text=item['text'],
                    setup=item['setup'],ending=item['ending'],reference=reference,artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
    atomic_json(out/'references.json',refs);atomic_json(out/'brackets.json',brackets);print('Prepared',len(refs),'held-out fixtures')

if __name__=='__main__':main()
