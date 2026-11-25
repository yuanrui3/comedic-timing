"""Controlled inference sensitivity checks using the existing MFA weights."""
from .common import ROOT, read_json, atomic_json, digest, Manifest
from .mfa_run import environment
from .pause_audit import hmm_boundary
from .acoustic import compare
import argparse
import json
import shutil
import subprocess
import time
import zipfile


def model_without_dither():
    original = ROOT/'models/mfa_zh/mandarin_mfa.zip'
    target = ROOT/'models/mfa_zh/mandarin_mfa_dither0.zip'
    if not target.exists():
        with zipfile.ZipFile(original) as src, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                value = src.read(info.filename)
                if info.filename.endswith('meta.json'):
                    metadata = json.loads(value)
                    metadata['features']['dither'] = 0
                    value = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                dst.writestr(info, value)
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(target) as dst:
        assert src.namelist() == dst.namelist()
        for name in src.namelist():
            if name.endswith('meta.json'):
                expected = json.loads(src.read(name))
                expected['features']['dither'] = 0
                assert json.loads(dst.read(name)) == expected
            else:
                assert src.read(name) == dst.read(name), f'Learned model changed: {name}'
    atomic_json(ROOT/'runs/model-locks/mfa-dither0.json', {
        'parent': str(original.relative_to(ROOT)), 'parent_sha256': digest(original),
        'derivative': str(target.relative_to(ROOT)), 'sha256': digest(target),
        'change': 'features.dither = 0; all non-metadata archive entries bit-identical',
        'license': 'CC-BY-4.0; Michael McAuliffe and Morgan Sonderegger; local metadata modification'})
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='mfa-diagnostic-v1')
    p.add_argument('--run', required=True)
    p.add_argument('--dither-zero', action='store_true')
    p.add_argument('--per-clip', action='store_true')
    args = p.parse_args()
    base = ROOT/'runs'/args.baseline
    inputs = read_json(base/'inputs.json')
    out = ROOT/'runs'/args.run
    corpus = ROOT/'.cache/mfa-corpora'/args.run
    model = model_without_dither() if args.dither_zero else ROOT/'models/mfa_zh/mandarin_mfa.zip'
    spec = {'baseline': args.baseline, 'baseline_manifest_sha256': digest(base/'manifest.json'),
            'inputs_sha256': digest(base/'inputs.json'), 'model_sha256': digest(model),
            'dither_zero': args.dither_zero, 'normalization_groups': 'per_clip' if args.per_clip else 'per_voice',
            'code_sha256': {n: digest(ROOT/'timing'/n) for n in ('mfa_variants.py','mfa_run.py','pause_audit.py','acoustic.py')},
            'config_sha256': digest(ROOT/'config/mfa-alignment.yaml'),
            'runtime': read_json(ROOT/'runs/model-locks/mfa-runtime.json'),
            'adaptation': False, 'weight_training': False}
    state = Manifest(out/'manifest.json', spec)
    if state.complete('alignment'):
        print('Verified existing outputs; skip.')
        return
    for key, item in inputs.items():
        src = ROOT/'.cache/mfa-corpora'/args.baseline/item['voice']/f'{key}.wav'
        if digest(src) != item['input_wav_sha256']:
            raise ValueError('Prepared audio checksum mismatch')
        directory = corpus/(key if args.per_clip else item['voice'])
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, directory/f'{key}.wav')
        (directory/f'{key}.lab').write_text(' '.join(item['tokens']), encoding='utf-8')
    atomic_json(out/'inputs.json', inputs)
    shutil.copyfile(base/'lexicon.dict', out/'lexicon.dict')
    command = [str(ROOT/'.runtime/mfa/Scripts/mfa.exe'), 'align', str(corpus), str(out/'lexicon.dict'),
               str(model), str(out/'aligned'), '--config_path', str(ROOT/'config/mfa-alignment.yaml'),
               '--temporary_directory', str(ROOT/'.cache/mfa-work'/args.run), '--output_format','json',
               '--no_use_mp','--no_use_postgres','--no_textgrid_cleanup','--num_jobs','2']
    state.record('alignment', status='running', command=command)
    start = time.perf_counter()
    with (out/'process.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, env=environment(), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    files = list((out/'aligned').rglob('*.json'))
    success = result.returncode == 0 and len(files) == len(inputs)
    if not success:
        state.record('alignment', status='failed', returncode=result.returncode, count=len(files))
        print((out/'process.log').read_text(encoding='utf-8')[-5000:])
        raise SystemExit(1)
    old = {p.stem: hmm_boundary(read_json(p), inputs[p.stem]) for p in (base/'aligned').rglob('*.json')}
    rows = [{'key': path.stem, 'baseline': old[path.stem],
             'variant': hmm_boundary(read_json(path), inputs[path.stem])} for path in sorted(files)]
    summary = {'gap': compare([r['variant']['gap_ms'] for r in rows], [r['baseline']['gap_ms'] for r in rows])}
    for endpoint in ('start','end'):
        summary[endpoint] = compare([r['variant'].get(endpoint, 0)*1000 for r in rows],
                                    [r['baseline'].get(endpoint, 0)*1000 for r in rows])
    atomic_json(out/'comparison.json', rows)
    atomic_json(out/'summary.json', summary)
    artifacts = files+[out/'inputs.json', out/'lexicon.dict', out/'comparison.json', out/'summary.json', out/'process.log']
    state.record('alignment', status='complete', elapsed_seconds=time.perf_counter()-start,
                 artifacts={str(p.relative_to(ROOT)).replace('\\','/'): digest(p) for p in artifacts})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
