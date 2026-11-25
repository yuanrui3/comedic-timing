"""Independent neural frame probabilities; deliberately no VAD hangover/padding."""
from .common import ROOT, read_json, atomic_json, digest, Manifest
from .acoustic import runs, compare
from .pause_evaluate import errors
import argparse
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def posterior_gap(probability, left, right, threshold=.5):
    candidates = []
    censored = False
    for a, b in runs(np.asarray(probability) >= threshold):
        # 25 ms window centers at 12.5 ms + frame * 10 ms; half-hop cells.
        start, end = a*.01+.0075, b*.01+.0075
        if end < left or start > right:
            continue
        if a == 0 or b == len(probability) or start < left or end > right:
            censored = True
            continue
        if b-a < 2:
            continue
        candidates.append({'start':start, 'end':end, 'gap_ms':(b-a)*10.})
    if len(candidates)==1:
        return {**candidates[0], 'status':'ok', 'candidates':candidates}
    return {'gap_ms': None if candidates or censored else 0., 'candidates':candidates,
            'status': 'ambiguous_multiple_intervals' if candidates else 'censored_at_bracket' if censored else 'no_qualifying_interval'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='fsmn-probe-v1')
    args = p.parse_args()
    import onnxruntime as ort
    import yaml
    from funasr_onnx.utils.frontend import WavFrontend
    folder = ROOT/'models/fsmn_vad'
    lock = read_json(ROOT/'runs/model-locks/fsmn_vad.json')
    for name, sha in lock['sha256'].items():
        if digest(folder/name) != sha:
            raise ValueError('Model checksum mismatch')
    cfg = yaml.safe_load((folder/'vad.yaml').read_text())
    frontend = WavFrontend(cmvn_file=str(folder/'vad.mvn'), **cfg['frontend_conf'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder/'model.onnx'), options, providers=['CPUExecutionProvider'])
    out = ROOT/'runs'/args.out
    sources = {'natural': ROOT/'runs/pause-audit-v3/measurements.json',
               'fixture': ROOT/'runs/pause-validation-v2/measurements.json'}
    spec = {'model':lock, 'code_sha256':digest(__file__), 'runtime_lock':digest(ROOT/'requirements-vad-lock.txt'),
            'frontend_code_sha256':digest(ROOT/'.venv-vad/Lib/site-packages/funasr_onnx/utils/frontend.py'),
            'sources':{k:digest(v) for k,v in sources.items()},
            'decision': 'class 0 >= 0.5; minimum 20ms; 0.3/0.7 sensitivity; no smoothing, padding, hangover',
            'timebase': '25ms window centers, 10ms hop', 'plan_sha256':digest(ROOT/'docs/flagged-followup-plan.md')}
    state = Manifest(out/'manifest.json', spec)
    rows = []
    for mode, source in sources.items():
        for row in read_json(source):
            key = row['key']
            dest = out/'frames'/f'{key}.json'
            if state.complete(key):
                result = read_json(dest)
            else:
                x, sr = sf.read(ROOT/row['source_audio'], dtype='float32')
                divisor = math.gcd(sr, 16000)
                x = resample_poly(x, 16000//divisor, sr//divisor)
                features, _ = frontend.fbank(x)
                features, _ = frontend.lfr_cmvn(features)
                inputs = {'speech':features[None].astype('float32')}
                for i in range(4):
                    inputs[f'in_cache{i}'] = np.zeros((1,128,19,1), dtype='float32')
                scores = session.run(None, inputs)[0][0]
                if not np.allclose(scores.sum(axis=1), 1, atol=1e-5) or not np.isfinite(scores).all():
                    raise ValueError('Invalid posterior output')
                silence = scores[:,0]
                result = {'key':key, 'mode':mode, 'audio_sha256':digest(ROOT/row['source_audio']),
                          'silence_probability':silence.tolist(), 'bracket':row['bracket'],
                          'grid':{str(t):posterior_gap(silence, *row['bracket'], t) for t in (.3,.5,.7)}}
                atomic_json(dest, result)
                state.record(key, status='complete', artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
            rows.append(result)
    natural = {r['key']:r for r in read_json(sources['natural'])}
    summary = {'natural':{}}
    for threshold in ('0.3','0.5','0.7'):
        selected = [r for r in rows if r['mode']=='natural']
        summary['natural'][threshold] = {
            'hmm_minus_fsmn': compare([natural[r['key']]['hmm']['gap_ms'] for r in selected], [r['grid'][threshold]['gap_ms'] for r in selected]),
            'rms_minus_fsmn': compare([natural[r['key']]['rms_grid']['35']['gap_ms'] for r in selected], [r['grid'][threshold]['gap_ms'] for r in selected])}
    lookup = {r['key']:r for r in rows if r['mode']=='fixture'}
    validation = []
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        for threshold in ('0.3','0.5','0.7'):
            value = lookup[ref['key']]['grid'][threshold]['gap_ms']
            baseline = lookup[ref['base_key']+'_'+ref['mode']+'_0000ms']['grid'][threshold]['gap_ms']
            recovered = value if ref['mode']=='absolute' else value-baseline if value is not None and baseline is not None else None
            validation.append({'key':ref['key'], 'mode':ref['mode'], 'threshold':threshold,
                               'error_ms':recovered-ref['inserted_ms'] if recovered is not None else None})
    summary['fixtures'] = {t:{mode:errors([r for r in validation if r['mode']==mode and r['threshold']==t])
                                  for mode in ('absolute','incremental')} for t in ('0.3','0.5','0.7')}
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'validation.json', validation)
    print(summary)


if __name__ == '__main__':
    main()
