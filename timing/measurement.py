"""Explicit CTC state alignment and independent waveform-energy intervals."""
import numpy as np

def ctc_align(log_probs, tokens, blank):
    """Viterbi over blank-interleaved labels, including repeated-label constraints."""
    lp = np.asarray(log_probs)
    labels = np.full(2*len(tokens)+1, blank, dtype=int)
    labels[1::2] = tokens
    if not len(tokens) or len(lp) < len(tokens) + sum(a == b for a,b in zip(tokens,tokens[1:])):
        raise ValueError('Empty transcript or insufficient CTC frames')
    n = len(labels)
    prev = np.full(n, -np.inf)
    prev[:2] = lp[0, labels[:2]]
    back = np.zeros((len(lp), n), dtype=np.int8)
    skip = np.zeros(n, dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    for t in range(1, len(lp)):
        candidates = np.stack([prev, np.r_[-np.inf, prev[:-1]], np.r_[-np.inf, -np.inf, prev[:-2]]])
        candidates[2, ~skip] = -np.inf
        best = candidates.argmax(axis=0)
        back[t] = best
        prev = candidates[best, np.arange(n)] + lp[t, labels]
    state = n-1 if prev[-1] >= prev[-2] else n-2
    if not np.isfinite(prev[state]):
        raise ValueError('No finite CTC path')
    states = np.empty(len(lp), dtype=int)
    for t in range(len(lp)-1, -1, -1):
        states[t] = state
        if t:
            state -= int(back[t,state])
    spans = []
    for i, token in enumerate(tokens):
        frames = np.flatnonzero(states == 2*i+1)
        if not len(frames):
            raise ValueError('Skipped transcript unit')
        spans.append((int(frames[0]), int(frames[-1])+1))
    return spans

def rms_track(audio, sr, frame_ms=10, hop_ms=5):
    x = np.asarray(audio, dtype=np.float64)
    frame, hop = round(sr*frame_ms/1000), round(sr*hop_ms/1000)
    if x.ndim != 1 or len(x) < frame or not np.isfinite(x).all():
        raise ValueError('Invalid mono waveform')
    starts = np.arange(0, len(x)-frame+1, hop)
    cumulative = np.r_[0., np.cumsum(x*x)]
    rms = np.sqrt(np.maximum(0, (cumulative[starts+frame]-cumulative[starts])/frame))
    centers = (starts+frame/2)/sr
    return centers, rms

def energy_gap(audio, sr, anchor, radius=1.5):
    centers, rms = rms_track(audio, sr)
    scale = float(np.quantile(rms, .95))
    if scale < 1e-5:
        return {'gap_ms': None, 'status': 'silent_clip'}
    threshold = max(1e-6, scale * 10**(-35/20))
    low = rms < threshold
    edges = np.diff(np.r_[False, low, False].astype(int))
    intervals = []
    for a,b in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
        left, right = float(centers[a]-.0025), float(centers[b-1]+.0025)
        if a == 0 or b == len(low):
            continue  # leading/trailing padding is not an internal pause
        distance = max(left-anchor, anchor-right, 0)
        if distance <= .05 and left >= anchor-radius and right <= anchor+radius:
            intervals.append((distance, -(right-left), left, right))
    if not intervals:
        return {'gap_ms': 0., 'status': 'no_low_energy_interval_near_anchor', 'threshold_rms': threshold}
    _, _, left, right = min(intervals)
    return {'gap_ms': (right-left)*1000, 'start': left, 'end': right, 'threshold_rms': threshold, 'status': 'ok'}

def agreement(rows):
    d = np.array([r['ctc_gap_ms']-r['energy']['gap_ms'] for r in rows
                  if r.get('ctc_gap_ms') is not None and r['energy']['gap_ms'] is not None])
    if not len(d):
        return {'n': 0}
    sd = float(d.std(ddof=1)) if len(d)>1 else None
    bias = float(d.mean())
    return {'n': len(d), 'bias_ctc_minus_energy_ms': bias, 'mae_ms': float(np.abs(d).mean()),
            'fraction_within_30ms': float((np.abs(d)<=30).mean()),
            'loa_ms': [bias-1.96*sd, bias+1.96*sd] if sd is not None else None}

def paired_bootstrap(differences, seed=20261114, draws=10000):
    d = np.array(differences, dtype=float)
    if len(d)<2 or not np.isfinite(d).all():
        raise ValueError('At least two finite paired differences required')
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0,len(d),size=(draws,len(d)))].mean(axis=1)
    sd = float(d.std(ddof=1))
    return {'n_pairs': len(d), 'mean_ms': float(d.mean()),
            'ci95_ms': np.quantile(means,[.025,.975]).tolist(),
            'dz': float(d.mean()/sd) if sd else None, 'bootstrap_draws': draws, 'seed': seed}
