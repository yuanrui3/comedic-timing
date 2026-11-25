"""Verify immutable outputs, report links, fixture edits, and snapshot used code."""
from .common import ROOT,read_json,atomic_json,digest
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,shutil,datetime
import numpy as np
import soundfile as sf

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.categories=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for name in ('href','src'):
            if name in attrs:self.links.append(attrs[name])
        if tag=='article':self.categories.append(attrs.get('data-category'))

def save_plan_snapshots():
    plan=ROOT/'docs/second-aligner-plan.md';body=plan.read_text(encoding='utf-8-sig');lines=body.splitlines();wanted={}
    for run in ('ac-rms-v1','detrended-rms-v1','style-level-audit-v1'):
        wanted[run]=read_json(ROOT/'runs'/run/'provenance.json')['plan_sha256']
    wanted['vocoder-stage-audit-v1']=read_json(ROOT/'runs/vocoder-stage-audit-v1/manifest.json')['config']['plan_sha256']
    candidates={}
    for end in range(1,len(lines)+1):
        prefix='\n'.join(lines[:end]).rstrip('\n')
        for tail in ('','\n','\n\n'):
            for newline in ('\n','\r\n'):
                raw=(prefix+tail).replace('\n',newline).encode('utf-8')
                for bom in (b'',b'\xef\xbb\xbf'):
                    blob=bom+raw;sha=hashlib.sha256(blob).hexdigest()
                    if sha in wanted.values():candidates[sha]=blob
    saved={}
    for run,sha in wanted.items():
        if sha not in candidates:raise ValueError('Cannot recover historical declared-plan bytes: '+run)
        dest=ROOT/'runs/plan-snapshots'/f'{sha}.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(candidates[sha]);assert digest(dest)==sha
        saved[run]=str(dest.relative_to(ROOT)).replace('\\','/')
    return saved

def main():
    plan_snapshots=save_plan_snapshots();checked={};json_count=0
    def reject_constant(v):raise ValueError('Non-finite JSON value: '+v)
    for path in (ROOT/'runs').rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=reject_constant);json_count+=1
    def check_artifacts(mapping):
        for relative,expected in mapping.items():
            if relative not in checked:checked[relative]=digest(ROOT/relative)
            if checked[relative]!=expected:raise ValueError('Artifact checksum mismatch: '+relative)
    manifest_count=0;complete_items=0
    for path in (ROOT/'runs').rglob('manifest.json'):
        state=read_json(path)
        if 'items' not in state:continue
        manifest_count+=1
        for item in state['items'].values():
            if item.get('status')=='complete':check_artifacts(item.get('artifacts',{}));complete_items+=1
    input_mapping_count=0
    aliases={'natural':'runs/pause-audit-v3/measurements.json','fixture':'runs/pause-validation-v2/measurements.json',
             'old':'runs/pause-validation-v2/measurements.json','holdout':'runs/pause-holdout-v1/brackets.json'}
    for path in (ROOT/'runs').rglob('provenance.json'):
        result=read_json(path)
        for field in ('artifacts_sha256','artifact_sha256'):
            if field in result:check_artifacts(result[field])
        for field in ('input_sha256','input_hashes','inputs_sha256','sources_sha256','inputs','input_manifests_sha256','alignment_manifests','alignment_manifests_sha256'):
            if field not in result or not isinstance(result[field],dict):continue
            mapping={}
            for key,value in result[field].items():
                if (ROOT/key).is_file():relative=key
                elif key in aliases:relative=aliases[key]
                elif path.parent.name=='second-aligner-investigation-v1':relative=f'runs/{key}/summary.json'
                else:relative=f'runs/{key}/manifest.json'
                mapping[relative]=value
            check_artifacts(mapping);input_mapping_count+=1
    report=read_json(ROOT/'runs/second-aligner-investigation-v1/summary.json');assert not report['draft']
    original=read_json(ROOT/'runs/corrected-pauses-v1/metrics.json')
    assert len(original)==40 and sum(bool(r['qc_flags']) for r in original)==34
    assert sum(r['eligible_for_automatic_analysis'] for r in original)==6
    assert report['cleared_flags']==0
    old_holdout=read_json(ROOT/'runs/holdout-analysis-v1/summary.json');new_holdout=read_json(ROOT/'runs/holdout-analysis-v2/summary.json')
    assert {k:new_holdout[k] for k in old_holdout}==old_holdout
    # Verify known edits from their construction, not from any aligner's output.
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items'];waves={}
    for key,item in source.items():waves[key]=sf.read(ROOT/next(iter(item['artifacts'])),dtype='float32')
    fixture_count=0
    for run in ('pause-fixtures-v1','pause-holdout-v1'):
        for ref in read_json(ROOT/'runs'/run/'references.json'):
            x,sr=waves[ref['base_key']];y,ysr=sf.read(ROOT/'data/audio'/run/f'{ref["key"]}.wav',dtype='float32')
            n=round(ref['inserted_ms']/1000*sr);left=ref['left_cut_sample'];right=ref['right_cut_sample']
            assert sr==ysr and np.array_equal(y,np.r_[x[:left],np.zeros(n,dtype=x.dtype),x[right:]])
            fixture_count+=1
    # CPU/GPU parity is checked on exactly unchanged incremental-zero clips.
    cpu_gpu=[]
    for ref in read_json(ROOT/'runs/pause-holdout-v1/references.json'):
        if ref['mode']!='incremental' or ref['inserted_ms']!=0:continue
        cpu=read_json(ROOT/'runs/charsiu-holdout-cpu-v1/aligned'/f'{ref["key"]}.json')['forced']
        gpu=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{ref["base_key"]}.json')['methods']['4']['forced']
        cpu_gpu.append({'key':ref['base_key'],**{side+'_delta_ms':1000*(cpu[side]-gpu[side]) for side in ('start','end')},'gap_delta_ms':cpu['gap_ms']-gpu['gap_ms']})
    page=ROOT/'reports/second-aligner-clips.html';parser=Links();parser.feed(page.read_text(encoding='utf-8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):
            assert (page.parent/link.split('#')[0]).resolve().exists(),link
    assert {c:parser.categories.count(c) for c in set(parser.categories)}=={'onset':29,'artifact':5,'other':6}
    snapshot=ROOT/'runs/second-aligner-investigation-v1/source';snapshot.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'runs/second-aligner-investigation-v1/source-by-sha256';archive.mkdir(parents=True,exist_ok=True)
    for p in snapshot.rglob('*'):
        if p.is_file():
            version=archive/f'{digest(p)}_{p.name}'
            if not version.exists():shutil.copyfile(p,version)
    snapshot_hashes={}
    for folder in ('timing','tests'):
        for p in (ROOT/folder).glob('*.py'):
            dest=snapshot/folder/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            snapshot_hashes[str(dest.relative_to(ROOT)).replace('\\','/')]=digest(dest)
            version=archive/f'{digest(p)}_{p.name}'
            if not version.exists():shutil.copyfile(p,version)
    for relative in ('docs/second-aligner-plan.md','docs/second-aligner-reproduce.md','config/mfa3-alignment.yaml','requirements-exploration-lock.txt','requirements-mfa3-lock.txt','requirements-vad-lock.txt'):
        p=ROOT/relative;dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);snapshot_hashes[str(dest.relative_to(ROOT)).replace('\\','/')]=digest(dest)
        version=archive/f'{digest(p)}_{p.name}'
        if not version.exists():shutil.copyfile(p,version)
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'json_files_parsed':json_count,'manifests_checked':manifest_count,
        'complete_items_checked':complete_items,'unique_artifacts_hashed':len(checked),'provenance_input_mappings_checked':input_mapping_count,'fixture_constructions_verified':fixture_count,
        'original_metric_sha256':digest(ROOT/'runs/corrected-pauses-v1/metrics.json'),'original_flags':34,'original_eligible':6,
        'HTML_local_links':len(parser.links),'HTML_categories':{c:parser.categories.count(c) for c in set(parser.categories)},
        'Charsiu_CPU_GPU_baseline_parity':cpu_gpu,'plan_snapshots':plan_snapshots,'source_snapshot_sha256':snapshot_hashes}
    atomic_json(ROOT/'runs/second-aligner-investigation-v1/verification.json',result)
    print({k:v for k,v in result.items() if k not in ('source_snapshot_sha256','Charsiu_CPU_GPU_baseline_parity')})
    print('Charsiu CPU/GPU baseline max endpoint difference, ms:',max(abs(r[side+'_delta_ms']) for r in cpu_gpu for side in ('start','end')))

if __name__=='__main__':main()
