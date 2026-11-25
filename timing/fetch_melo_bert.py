"""Fetch the pinned official multilingual BERT dependency used by Melo Chinese."""
from .common import ROOT,atomic_json,read_json,digest
from .fetch_melo import read_url,fetch
import json

def main():
    repo='google-bert/bert-base-multilingual-uncased';lockpath=ROOT/'runs/model-locks/melo_bert.json'
    if lockpath.exists():
        lock=read_json(lockpath)
        for f,sha in lock['sha256'].items():assert digest(ROOT/'models/melo_bert'/f)==sha
        print('Verified existing BERT files');return
    info=json.loads(read_url('https://huggingface.co/api/models/'+repo+'?blobs=true'))
    files={r['rfilename']:r for r in info['siblings']}
    names=['README.md','config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']
    total=sum(files[n].get('size',0) for n in names);assert total<1_000_000_000,total
    hashes={};sizes={}
    for name in names:
        path=ROOT/'models/melo_bert'/name;fetch(f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}',path)
        hashes[name]=digest(path);sizes[name]=path.stat().st_size
        if files[name].get('lfs'):assert hashes[name]==files[name]['lfs']['sha256']
    atomic_json(lockpath,{'repository':repo,'revision':info['sha'],'license':info.get('cardData',{}).get('license'),
        'sha256':hashes,'bytes':sizes,'purpose':'Melo Chinese native text conditioning'})
    print(info['sha'],sum(sizes.values()))

if __name__=='__main__':main()
