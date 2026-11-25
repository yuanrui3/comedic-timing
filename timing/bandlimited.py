"""Fixed subsonic removal sensitivity analysis; no fitted thresholds."""
from .common import ROOT, read_json, atomic_json, digest
from .acoustic import acoustic_gap, compare
from .pause_evaluate import errors
from scipy.signal import butter, sosfiltfilt
import numpy as np
import soundfile as sf


def highpass(audio, sr):
    return sosfiltfilt(butter(2, 20, 'highpass', fs=sr, output='sos'), audio)


def main():
    rows = []
    sources = {'natural': ROOT/'runs/pause-audit-v3/measurements.json',
               'fixture': ROOT/'runs/pause-validation-v2/measurements.json'}
    for mode, source in sources.items():
        for row in read_json(source):
            x, sr = sf.read(ROOT/row['source_audio'])
            left, right = row['bracket']
            result = acoustic_gap(highpass(x, sr), sr, left, right)
            rows.append({'key': row['key'], 'mode': mode, 'result': result, 'hmm': row['hmm'],
                         'raw': row['rms_grid']['35']})
    natural = [r for r in rows if r['mode']=='natural']
    lookup = {r['key']: r['result']['gap_ms'] for r in rows if r['mode']=='fixture'}
    validation = []
    for ref in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        value = lookup[ref['key']]
        zero = lookup.get(ref['base_key']+'_'+ref['mode']+'_0000ms')
        recovered = value if ref['mode']=='absolute' else value-zero if value is not None and zero is not None else None
        validation.append({'key': ref['key'], 'mode': ref['mode'],
                           'error_ms': recovered-ref['inserted_ms'] if recovered is not None else None})
    summary = {'natural_hmm_minus_hp20': compare([r['hmm']['gap_ms'] for r in natural],
                                                [r['result']['gap_ms'] for r in natural]),
               'natural_ambiguities': [r['key'] for r in natural if r['result']['gap_ms'] is None],
               'fixtures': {mode: errors([r for r in validation if r['mode']==mode]) for mode in ('absolute','incremental')}}
    out = ROOT/'runs/bandlimited-v1'
    atomic_json(out/'measurements.json', rows)
    atomic_json(out/'validation.json', validation)
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'provenance.json', {'code': {n:digest(ROOT/'timing'/n) for n in ('bandlimited.py','acoustic.py')},
                                      'inputs': {k:digest(v) for k,v in sources.items()},
                                      'plan_sha256': digest(ROOT/'docs/flagged-followup-plan.md')})
    print(summary)


if __name__=='__main__':
    main()
