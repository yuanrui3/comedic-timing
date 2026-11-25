"""Snapshot local runtime versions, native artifacts, source and download evidence."""
from .common import ROOT,read_json,atomic_json,digest
from .mfa3_run import environment
from datetime import datetime
import subprocess,json

def main():
    code="import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':sorted({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}.items())}))"
    runtimes={}
    for name,relative in (('main','.venv/Scripts/python.exe'),('mfa3','.runtime/mfa3/python.exe'),('vad','.venv-vad/Scripts/python.exe')):
        result=subprocess.run([str(ROOT/relative),'-X','utf8','-c',code],env=environment() if name=='mfa3' else None,capture_output=True,text=True,check=True)
        runtimes[name]=json.loads(result.stdout)
    (ROOT/'requirements-mfa3-lock.txt').write_text(''.join(f'{name}=={version}\n' for name,version in runtimes['mfa3']['packages']),encoding='utf-8')
    native=[]
    for p in (ROOT/'.runtime/mfa3/conda-meta').glob('*.json'):
        r=read_json(p)
        if r['name'] in ('python','kaldi','kalpy','pynini','openfst'):
            native.append({k:r.get(k) for k in ('name','version','build','url','sha256','md5')})
    files=[ROOT/'.runtime/mfa3/Lib/site-packages/_kalpy.cp310-win_amd64.pyd',ROOT/'.runtime/mfa3/Lib/site-packages/_pywrapfst.cp310-win_amd64.pyd']
    files+=[ROOT/'.runtime/mfa3/Library/bin'/n for n in ('gmm-align-compiled.exe','compute-mfcc-feats.exe','compute-kaldi-pitch-feats.exe','kaldi-feat.dll','kaldi-gmm.dll')]
    atomic_json(ROOT/'runs/model-locks/mfa3-runtime.json',{'native_packages':native,'runtime':runtimes['mfa3'],
        'python_dependencies_sha256':digest(ROOT/'requirements-mfa3-lock.txt'),'binary_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files},
        'installation':'project-local hybrid native Conda + wheels; failed optional test-file extraction logged; no system changes'})
    start=datetime.fromisoformat('2026-11-15T22:32:45+00:00').timestamp()
    archives=[p for p in (ROOT/'.cache/mamba3/pkgs').rglob('*') if p.is_file() and (p.name.endswith('.conda') or p.name.endswith('.tar.bz2'))]
    models=[p for f in ('charsiu_zh','charsiu_code','charsiu_tokenizer','mfa3_zh','silero_vad') for p in (ROOT/'models'/f).rglob('*') if p.is_file()]
    cache=[p for p in (ROOT/'.cache/uv').rglob('*') if p.is_file() and p.stat().st_mtime>=start]
    budget={'ceiling_bytes':15_000_000_000,'model_and_source_files_bytes':sum(p.stat().st_size for p in models),
        'native_archives_bytes':sum(p.stat().st_size for p in archives),'recent_expanded_uv_cache_bytes':sum(p.stat().st_size for p in cache),
        'interpretation':'retained download evidence, not a network traffic meter; uv bytes are expanded and may include hardlinks; failed transaction archives retained',
        'archive_inventory':{str(p.relative_to(ROOT)).replace('\\','/'):p.stat().st_size for p in archives}}
    budget['retained_evidence_total_bytes']=sum(budget[k] for k in ('model_and_source_files_bytes','native_archives_bytes','recent_expanded_uv_cache_bytes'))
    out=ROOT/'runs/exploration-environment-v1';atomic_json(out/'runtime.json',runtimes);atomic_json(out/'downloads.json',budget)
    atomic_json(out/'source.json',{'timing_sha256':{p.name:digest(p) for p in (ROOT/'timing').glob('*.py')},
        'test_sha256':{p.name:digest(p) for p in (ROOT/'tests').glob('*.py')},
        'dependency_sha256':{p:digest(ROOT/p) for p in ('requirements-exploration-lock.txt','requirements-mfa3-lock.txt','requirements-vad-lock.txt')}})
    print({k:v for k,v in budget.items() if k!='archive_inventory'})

if __name__=='__main__':main()
