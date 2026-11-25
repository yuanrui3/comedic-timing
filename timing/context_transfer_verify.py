"""Read-only checks of experiment evidence, with a new verification snapshot."""
from .common import ROOT,read_json,atomic_json,digest
from .exploration_verify import Links
import datetime,json,shutil
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def main():
    out=ROOT/'runs/context-transfer-report-v1';checked={};items=0;manifests=0;count=0
    def reject(v):raise ValueError(v)
    for p in (ROOT/'runs').rglob('*.json'):
        json.loads(p.read_text(encoding='utf-8-sig'),parse_constant=reject);count+=1
    for p in (ROOT/'runs').rglob('manifest.json'):
        s=read_json(p)
        if 'items' not in s:continue
        manifests+=1
        for item in s['items'].values():
            if item.get('status')!='complete':continue
            items+=1
            for rel,sha in item.get('artifacts',{}).items():
                if rel not in checked:checked[rel]=digest(ROOT/rel)
                assert checked[rel]==sha,rel
    original_sha='5991384df05895e17de474e88778cac303f000899466ab5e897a5baa5956aa34'
    assert digest(ROOT/'runs/corrected-pauses-v1/metrics.json')==original_sha
    metrics=read_json(ROOT/'runs/corrected-pauses-v1/metrics.json');assert len(metrics)==40
    assert sum(bool(v['qc_flags']) for v in metrics)==34
    originals=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    source=read_json(ROOT/'runs/future-context-audio-v2/manifest.json')['items'];prefixes=0
    for key,item in originals.items():
        x,sr=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32');assert sr==24000
        base=resample_poly(x,2,3)
        for mode in ('baseline','mute_g100','mute_g500','reverse_g100','reverse_g500'):
            row=source[key+'__'+mode];y,ysr=sf.read(ROOT/row['source_audio'],dtype='float32');assert ysr==16000
            cut=row['cut_sample'];expected=base.copy()
            if mode.startswith('mute'):expected[cut:]=0
            elif mode.startswith('reverse'):expected[cut:]=base[cut:][::-1]
            assert np.array_equal(y,expected);prefixes+=1
    new_assets={};model_files=0
    for name,folder in [('melo_zh','melo_zh'),('melo-code','melo_code'),('melo_bert','melo_bert'),('whisper_turbo','whisper_turbo')]:
        lock=read_json(ROOT/'runs/model-locks'/f'{name}.json');size=0
        for rel,sha in lock['sha256'].items():
            path=ROOT/'models'/folder/rel;assert digest(path)==sha;size+=path.stat().st_size;model_files+=1
        new_assets[name]=size
    prior=read_json(ROOT/'runs/exploration-environment-v1/downloads.json')['retained_evidence_total_bytes']
    assert prior+sum(new_assets.values())<15_000_000_000
    atomic_json(out/'downloads.json',{'prior_retained_bytes':prior,'new_assets_bytes':new_assets,
        'total_retained_bytes':prior+sum(new_assets.values()),'ceiling_bytes':15_000_000_000,
        'interpretation':'retained source/checkpoint/download evidence, not a network traffic meter; no package install in this continuation'})
    page=ROOT/'reports/context-transfer-clips.html';parser=Links();parser.feed(page.read_text(encoding='utf-8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):assert (page.parent/link.split('#')[0]).resolve().exists(),link
    assert len(parser.categories)==100
    hashes={}
    for relative in ('timing','tests'):
        for p in (ROOT/relative).glob('*.py'):
            dest=out/'source'/relative/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            hashes[str(dest.relative_to(ROOT)).replace('\\','/')]=digest(dest)
    for relative in ('docs/context-transfer-plan.md','docs/context-transfer-reproduce.md','reports/context-transfer-investigation.md'):
        p=ROOT/relative;dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        hashes[str(dest.relative_to(ROOT)).replace('\\','/')]=digest(dest)
    plan=read_json(ROOT/'runs/future-context-audio-v2/manifest.json')['config']['plan_sha256']
    assert digest(ROOT/'runs/plan-snapshots'/f'{plan}.md')==plan
    whplan=ROOT/'runs/whisper-transcript-qc-v1/declared-plan.md'
    assert digest(whplan)==read_json(ROOT/'runs/whisper-transcript-qc-v1/manifest.json')['config']['plan_sha256']
    qc1=read_json(ROOT/'runs/context-transfer-qc-v1/summary.json');qc2=read_json(ROOT/'runs/context-transfer-qc-v2/summary.json')
    for family in ('Kokoro','Melo'):
        for model in ('CTC','Whisper'):
            assert all(qc2[family]['recognizers'][model][k]==v for k,v in qc1[family]['recognizers'][model].items())
    # Verify derivation inputs from all current continuation analyses.
    provenance_inputs=0
    for run in ('future-context-analysis-v1','transfer-analysis-v1','context-transfer-qc-v2'):
        pr=read_json(ROOT/'runs'/run/'provenance.json')
        for field in ('input_manifests_sha256','input_sha256'):
            for rel,sha in pr.get(field,{}).items():
                path=ROOT/rel if field=='input_sha256' else ROOT/'runs'/rel/'manifest.json'
                assert digest(path)==sha;provenance_inputs+=1
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'JSON_files_parsed':count,
        'manifests_checked':manifests,'complete_items_checked':items,'unique_artifacts_hashed':len(checked),
        'future_context_waveforms_reconstructed':prefixes,'new_model_files_verified':model_files,
        'provenance_inputs_verified':provenance_inputs,'original_metrics_sha256':original_sha,'original_flags':34,
        'HTML_audio_cards':len(parser.categories),'HTML_local_links':len(parser.links),'source_snapshot_sha256':hashes}
    atomic_json(out/'verification.json',result)
    print({k:v for k,v in result.items() if k!='source_snapshot_sha256'});print('Retained download evidence GB:',(prior+sum(new_assets.values()))/1e9)

if __name__=='__main__':main()
