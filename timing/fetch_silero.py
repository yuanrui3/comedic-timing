"""Fetch a fixed public model/code revision; never call hosted inference."""
from .common import ROOT,atomic_json,digest
import urllib.request

def main():
    revision='5cd7945676eb32225748052e2e6a0580e4686a08'
    files={'silero_vad.onnx':'src/silero_vad/data/silero_vad.onnx','utils_vad.py':'src/silero_vad/utils_vad.py','LICENSE':'LICENSE'}
    root=ROOT/'models/silero_vad';root.mkdir(parents=True,exist_ok=True);hashes={};sizes={}
    for name,relative in files.items():
        path=root/name;url=f'https://raw.githubusercontent.com/snakers4/silero-vad/{revision}/{relative}'
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=90) as response,partial.open('wb') as out:
                while block:=response.read(1024*1024):out.write(block)
            partial.replace(path)
        hashes[name]=digest(path);sizes[name]=path.stat().st_size
    assert hashes['silero_vad.onnx']=='1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
    atomic_json(ROOT/'runs/model-locks/silero_vad.json',{'repository':'snakers4/silero-vad','tag':'v6.2.3','revision':revision,
        'license':'MIT','license_confirmation':'https://github.com/snakers4/silero-vad/discussions/817',
        'sha256':hashes,'bytes':sizes,'source_paths':files})
    (ROOT/'docs/licenses/silero-vad-LICENSE').write_bytes((root/'LICENSE').read_bytes())
    print(sizes)

if __name__=='__main__':main()
