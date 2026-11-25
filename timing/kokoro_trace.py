"""Replay saved Kokoro inputs, retaining latent durations only after PCM identity."""
from .common import ROOT, read_json, atomic_json, digest, Manifest
import argparse
import os
import time
import numpy as np
import soundfile as sf
import torch


def punctuation_index(setup, text, phonemes):
    table = str.maketrans('，。！？；：', ',.!?;:')
    punctuation = set(',.!?;:')
    expected = [c for c in text.translate(table) if c in punctuation]
    observed = [(i, c) for i, c in enumerate(phonemes) if c in punctuation]
    if expected != [c for _, c in observed]:
        raise ValueError('Punctuation sequence does not map exactly to saved phonemes')
    prefix = [c for c in setup.translate(table) if c in punctuation]
    if not prefix or expected[:len(prefix)] != prefix or setup[-1].translate(table) not in punctuation:
        raise ValueError('Setup must end in an exactly matched punctuation mark')
    return observed[len(prefix)-1][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='kokoro-diagnostic-v1')
    parser.add_argument('--out', default='kokoro-trace-v1')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    from kokoro import KModel
    source_path = ROOT/'runs'/args.source/'manifest.json'
    source = read_json(source_path)
    model_dir = ROOT/'models/kokoro_zh'
    lock = read_json(ROOT/'runs/model-locks/kokoro_zh.json')
    for name, sha in lock['sha256'].items():
        if digest(model_dir/name) != sha:
            raise ValueError(f'Model hash mismatch: {name}')
    out = ROOT/'runs'/args.out
    spec = {'source_sha256': digest(source_path), 'model': lock,
            'code_sha256': digest(__file__), 'torch': torch.__version__,
            'local_model_code': digest(ROOT/'.venv/Lib/site-packages/kokoro/model.py'),
            'seed': source['config']['seed'], 'frame_ms': 25,
            'purpose': 'latent allocation, NOT acoustic ground truth'}
    state = Manifest(out/'manifest.json', spec)
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    model = KModel(repo_id=lock['model']['repo'], config=str(model_dir/'config.json'),
                   model=str(model_dir/'kokoro-v1_1-zh.pth')).to('cuda').eval()
    packs = {}
    for key, item in source['items'].items():
        if state.complete(key):
            continue
        wav = ROOT/next(iter(item['artifacts']))
        if digest(wav) != item['artifacts'][str(wav.relative_to(ROOT)).replace('\\', '/')]:
            raise ValueError('Original audio checksum mismatch')
        original, sr = sf.read(wav, dtype='float32')
        voice = item['voice']
        if voice not in packs:
            packs[voice] = torch.load(model_dir/'voices'/f'{voice}.pt', weights_only=True).to('cuda')
        ps = item['phonemes']
        if any(c not in model.vocab for c in ps):
            raise ValueError('Unknown saved phoneme')
        torch.manual_seed(spec['seed'])
        start = time.perf_counter()
        result = model(ps, packs[voice][len(ps)-1], 1., return_output=True)
        generated = result.audio.numpy()
        durations = result.pred_dur.numpy()
        identical = generated.shape == original.shape and np.array_equal(generated, original)
        if not identical:
            raise ValueError(f'{key}: replay is not PCM identical; cannot attach trace')
        if len(durations) != len(ps)+2 or len(original) != int(durations.sum())*600:
            raise ValueError('Unexpected duration-to-waveform mapping')
        edges = np.r_[0, np.cumsum(durations)]/40
        punct = punctuation_index(item['setup'], item['text'], ps)
        # Include following whitespace allocation, but never label it silence.
        stop = punct+1
        while stop < len(ps) and ps[stop].isspace():
            stop += 1
        trace = {'key': key, 'pcm_identical': identical, 'sample_rate': sr,
                 'source_audio_sha256': digest(wav), 'elapsed_seconds': time.perf_counter()-start,
                 'punctuation_index': punct, 'punctuation_only': {
                     'start': float(edges[punct+1]), 'end': float(edges[punct+2])},
                 'punctuation_and_space': {'start': float(edges[punct+1]), 'end': float(edges[stop+1])},
                 'tokens': [{'token': c, 'frames': int(d), 'start': float(a), 'end': float(b)}
                            for c, d, a, b in zip(['<bos>', *ps, '<eos>'], durations, edges[:-1], edges[1:])],
                 'interpretation': '25 ms latent allocation; decoder can spread acoustic content across boundaries'}
        path = out/'traces'/f'{key}.json'
        atomic_json(path, trace)
        state.record(key, status='complete', artifacts={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)})
        print(f'{key}: identical PCM; punctuation {trace["punctuation_and_space"]}', flush=True)


if __name__ == '__main__':
    main()
