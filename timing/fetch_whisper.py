"""Download official pinned local ASR assets, without hosted inference."""
from .common import ROOT,read_json,atomic_json,digest
from .fetch_melo import read_url,fetch
import json

def main():
    repo='openai/whisper-large-v3-turbo';rev='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
    lockpath=ROOT/'runs/model-locks/whisper_turbo.json'
    if lockpath.exists():
        lock=read_json(lockpath)
        for name,sha in lock['sha256'].items():assert digest(ROOT/'models/whisper_turbo'/name)==sha
        print('Verified existing Whisper assets');return
    info=json.loads(read_url(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true'))
    assert info['sha']==rev and info['cardData']['license']=='mit'
    files=[r for r in info['siblings'] if r['rfilename']!='.gitattributes']
    total=sum(r['size'] for r in files);assert total<1_700_000_000,total
    hashes={};sizes={}
    for row in files:
        name=row['rfilename'];path=ROOT/'models/whisper_turbo'/name
        fetch(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',path)
        sha=digest(path);assert path.stat().st_size==row['size']
        if row.get('lfs'):assert sha==row['lfs']['sha256']
        hashes[name]=sha;sizes[name]=path.stat().st_size
        print(name,sizes[name],flush=True)
    atomic_json(lockpath,{'repository':repo,'revision':rev,'license':'MIT','sha256':hashes,'bytes':sizes,
        'purpose':'unprompted local transcript cross-check, not phonetic timing reference'})

if __name__=='__main__':main()
