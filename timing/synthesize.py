from .common import ROOT, read_json, digest, Manifest, source_hash
import argparse
import time
import numpy as np
import soundfile as sf
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='kokoro-diagnostic-v1')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    from kokoro import KModel, KPipeline
    corpus_path = ROOT/'data/scripts/open_zh_diagnostic.json'
    rows = read_json(corpus_path)[:args.limit]
    model_path = ROOT/'models/kokoro_zh'
    config = {'model_lock': read_json(ROOT/'runs/model-locks/kokoro_zh.json'),
              'corpus_sha256': digest(corpus_path), 'code_sha256': source_hash(),
              'torch': torch.__version__, 'voices': ['zf_001', 'zm_010'],
              'speed': 1.0, 'seed': 20261114, 'limit': args.limit,
              'delivery': 'plain', 'variant': 'original', 'takes': 1,
              'purpose': 'diagnostic_unfiltered_not_main', 'device': 'cuda'}
    manifest = Manifest(ROOT/'runs'/args.run/'manifest.json', config)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this run.')
    free, total = torch.cuda.mem_get_info()
    if free < 2*1024**3:
        raise RuntimeError('Less than 2 GiB free GPU memory.')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    model = KModel(repo_id=config['model_lock']['model']['repo'],
                   config=str(model_path/'config.json'), model=str(model_path/'kokoro-v1_1-zh.pth')).to('cuda').eval()
    pipeline = KPipeline(lang_code='z', repo_id=config['model_lock']['model']['repo'], model=model)
    for row in rows:
        for voice in config['voices']:
            for kind in ('joke', 'control'):
                key = f'{row["id"]}_{voice}_{kind}'
                if manifest.complete(key):
                    print(f'resume: {key}', flush=True)
                    continue
                text = row['setup'] + row[kind]
                manifest.record(key, status='running', text=text)
                try:
                    phonemes, _ = pipeline.g2p(text)
                    unknown = sorted(set(phonemes) - set(model.vocab))
                    if unknown or not phonemes or len(phonemes) > 510:
                        raise ValueError(f'Unsupported phonemes or length: {unknown}, {len(phonemes)}')
                    torch.manual_seed(config['seed'])
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    outputs = list(pipeline.generate_from_tokens(phonemes, voice=str(model_path/'voices'/f'{voice}.pt'), speed=1.0))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if len(outputs) != 1:
                        raise ValueError('Multi-chunk synthesis is excluded.')
                    wav = outputs[0].audio.detach().cpu().numpy()
                    if wav.ndim != 1 or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
                        raise ValueError('Invalid or silent waveform')
                    path = ROOT/'data/audio'/args.run/f'{key}.wav'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.partial')
                    sf.write(tmp, wav, 24000, format='WAV', subtype='FLOAT')
                    tmp.replace(path)
                    manifest.record(key, status='complete', script_id=row['id'], voice=voice, kind=kind,
                                    text=text, setup=row['setup'], ending=row[kind], phonemes=phonemes,
                                    seconds=len(wav)/24000, sample_rate=24000, elapsed_seconds=elapsed,
                                    peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
                                    artifacts={str(path.relative_to(ROOT)).replace('\\','/'): digest(path)})
                    print(f'{key}: {len(wav)/24000:.2f}s audio, {elapsed:.2f}s wall', flush=True)
                except Exception as exc:
                    manifest.record(key, status='failed', error=f'{type(exc).__name__}: {exc}', text=text)
                    raise

if __name__ == '__main__':
    main()
