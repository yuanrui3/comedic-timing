from .common import ROOT, read_json, atomic_json, digest
import argparse
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()
    configs = read_json(ROOT/'config/models.json')
    for name in args.models:
        spec = configs[name]
        target = ROOT/'models'/name
        hashes = {}
        for filename in spec['files']:
            print(f'{name}: {filename}', flush=True)
            p = hf_hub_download(spec['repo'], filename, revision=spec['revision'], local_dir=target)
            hashes[filename] = digest(p)
            atomic_json(target/'download-manifest.json', {'model': spec, 'sha256': hashes})
        atomic_json(ROOT/'runs/model-locks'/f'{name}.json', {'model': spec, 'sha256': hashes})

if __name__ == '__main__':
    main()
