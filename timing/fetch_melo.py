"""Pin and fetch official Melo code/metadata, with weights only on explicit action."""
from .common import ROOT,atomic_json,read_json,digest
import argparse,json,urllib.request

def read_url(url):
    request=urllib.request.Request(url,headers={'User-Agent':'local-research-provenance/1.0'})
    with urllib.request.urlopen(request,timeout=90) as response:return response.read()

def fetch(url,path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    request=urllib.request.Request(url,headers={'User-Agent':'local-research-provenance/1.0'})
    temp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(request,timeout=90) as response,temp.open('wb') as out:
        while block:=response.read(1024*1024):out.write(block)
    temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['metadata','weights']);args=p.parse_args()
    lockpath=ROOT/'runs/model-locks/melo_zh.json';codepath=ROOT/'runs/model-locks/melo-code.json'
    if args.action=='metadata':
        if lockpath.exists() and codepath.exists():print('Metadata already pinned');return
        hf=json.loads(read_url('https://huggingface.co/api/models/myshell-ai/MeloTTS-Chinese?blobs=true'))
        commit=json.loads(read_url('https://api.github.com/repos/myshell-ai/MeloTTS/commits/main'))['sha']
        tree=json.loads(read_url(f'https://api.github.com/repos/myshell-ai/MeloTTS/git/trees/{commit}?recursive=1'))['tree']
        files=[r['path'] for r in tree if r['type']=='blob' and (r['path'] in ('LICENSE','README.md','requirements.txt','setup.py') or r['path'].startswith('melo/') and not r['path'].endswith(('.png','.jpg','.mp3','.wav')))]
        hashes={}
        for relative in files:
            path=ROOT/'models/melo_code'/relative;fetch(f'https://raw.githubusercontent.com/myshell-ai/MeloTTS/{commit}/{relative}',path);hashes[relative]=digest(path)
        atomic_json(codepath,{'repository':'myshell-ai/MeloTTS','revision':commit,'license':'MIT','sha256':hashes})
        files_hf={r['rfilename']:r for r in hf['siblings']};hashes={}
        for name in ('README.md','config.json'):
            path=ROOT/'models/melo_zh'/name;fetch(f'https://huggingface.co/myshell-ai/MeloTTS-Chinese/resolve/{hf["sha"]}/{name}',path);hashes[name]=digest(path)
        atomic_json(lockpath,{'repository':'myshell-ai/MeloTTS-Chinese','revision':hf['sha'],'license':hf.get('cardData',{}).get('license'),
            'sha256':hashes,'checkpoint_lfs':files_hf['checkpoint.pth'].get('lfs'),'download_policy':'official files only, local inference; no hosted TTS'})
        print('Pinned',commit,hf['sha'],'files',len(files),'weight',files_hf['checkpoint.pth'].get('lfs'))
    else:
        lock=read_json(lockpath);path=ROOT/'models/melo_zh/checkpoint.pth'
        fetch(f'https://huggingface.co/{lock["repository"]}/resolve/{lock["revision"]}/checkpoint.pth',path)
        assert path.stat().st_size==lock['checkpoint_lfs']['size']
        assert digest(path)==lock['checkpoint_lfs']['sha256']
        lock['sha256']['checkpoint.pth']=digest(path);atomic_json(lockpath,lock);print('Verified checkpoint',path.stat().st_size)

if __name__=='__main__':main()
