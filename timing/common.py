from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
for name, relative in {
    'HF_HOME': '.cache/huggingface', 'TORCH_HOME': '.cache/torch',
    'MPLCONFIGDIR': '.cache/matplotlib', 'NUMBA_CACHE_DIR': '.cache/numba',
    'TEMP': '.cache/tmp', 'TMP': '.cache/tmp',
}.items():
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def source_hash():
    return hashlib.sha256(''.join(digest(p) for p in sorted((ROOT/'timing').glob('*.py'))).encode()).hexdigest()

class Manifest:
    def __init__(self, path, config):
        self.path = Path(path)
        # Compare the persisted JSON representation, so tuple-valued parameter
        # grids remain resumable after JSON has read them back as lists.
        config = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
        self.data = read_json(path) if self.path.exists() else {'config': config, 'items': {}}
        if self.data['config'] != config:
            raise ValueError('Run configuration changed: use a new run directory.')
        atomic_json(self.path, self.data)

    def complete(self, key):
        item = self.data['items'].get(key)
        return bool(item and item.get('status') == 'complete' and all(
            (ROOT/p).exists() and digest(ROOT/p) == sha for p, sha in item.get('artifacts', {}).items()))

    def record(self, key, **values):
        self.data['items'][key] = values
        atomic_json(self.path, self.data)
