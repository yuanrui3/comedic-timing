"""Per-clip evidence, plots and a local review page; never clears QC by fiat."""
from .common import ROOT, read_json, atomic_json, digest
from .measurement import rms_track
from .acoustic import compare
import html
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows = read_json(ROOT/'runs/pause-audit-v3/measurements.json')
    prior = {r['key']:r for r in read_json(ROOT/'runs/corrected-pauses-v1/metrics.json')}
    variant_names = ('mfa-repeat-v1','mfa-dither0-v1','mfa-perclip-d0-v1','mfa-perclip-d0-repeat-v1')
    variants = {n:{r['key']:r['variant'] for r in read_json(ROOT/'runs'/n/'comparison.json')} for n in variant_names}
    figures = ROOT/'reports/figures/flagged-followup'
    figures.mkdir(parents=True, exist_ok=True)
    findings = []
    cards = []
    for row in rows:
        key = row['key']
        h, raw = row['hmm'], row['rms_grid']['35']
        x, sr = sf.read(ROOT/row['source_audio'])
        phone_path = next((ROOT/'runs/mfa-diagnostic-v1/aligned').rglob(key+'.json'))
        phones = read_json(phone_path)['tiers']['phones']['entries']
        first = next(p for p in phones if abs(p[0]-h['end'])<.001)
        last = next(p for p in phones if abs(p[1]-h['start'])<.001)
        flagged = bool(prior[key]['qc_flags'])
        case = {'key':key,'original_flags':prior[key]['qc_flags'],'flags_cleared':False,
                'hmm':h, 'raw_rms':raw, 'last_setup_phone':last, 'first_ending_phone':first,
                'variants':{n:v[key] for n,v in variants.items()}, 'source_audio':row['source_audio']}
        y = x[int(h['start']*sr):int(h['end']*sr)]
        spectrum = np.abs(np.fft.rfft(y*np.hanning(len(y))))**2
        frequency = np.fft.rfftfreq(len(y),1/sr)
        case['spectrum'] = {'window':'Hann, entire HMM interval; finite-window leakage applies',
                            'dominant_hz':float(frequency[spectrum.argmax()]),
                            'fraction_below_20hz':float(spectrum[frequency<20].sum()/spectrum.sum())}
        if raw['gap_ms'] is None:
            category = 'fragmented_RMS_subsonic_energy'
            explanation = ('Multiple quiet intervals; the HMM-region spectrum peaks at '
                           f'{case["spectrum"]["dominant_hz"]:.1f} Hz. Filtering does not establish an accurate pause.')
        elif flagged:
            category = 'RMS_endpoint_inside_first_HMM_phone'
            case['endpoint_delta_ms_RMS_minus_HMM'] = {k:(raw[k]-h[k])*1000 for k in ('start','end')}
            case['RMS_end_inside_first_phone'] = first[0] <= raw['end'] <= first[1]+.005
            explanation = (f'RMS resumes {case["endpoint_delta_ms_RMS_minus_HMM"]["end"]:.1f} ms after HMM, '
                           f'inside its initial /{first[2]}/ interval. Consistent with low-energy phone inclusion; '
                           'the HMM label is a model hypothesis, not independent truth.')
        else:
            category = 'prior_duration_agreement'
            explanation = 'Passed the original duration-only comparison; natural boundary accuracy remains unverified.'
            case['endpoint_delta_ms_RMS_minus_HMM'] = {k:(raw[k]-h[k])*1000 for k in ('start','end')}
        case['category'] = category
        case['explanation'] = explanation
        neural = read_json(ROOT/'runs/fsmn-probe-v1/frames'/f'{key}.json')
        case['fsmn'] = neural['grid']['0.5']
        trace = read_json(ROOT/'runs/kokoro-trace-v1/traces'/f'{key}.json')
        case['latent_punctuation'] = trace['punctuation_and_space']
        findings.append(case)

        left, right = max(0,row['bracket'][0]-.15), min(len(x)/sr,row['bracket'][1]+.15)
        fig, axes = plt.subplots(3,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1.2,1,1]},layout='constrained')
        time = np.arange(len(x))/sr
        mask = (time>=left)&(time<=right)
        axes[0].plot(time[mask][::3],x[mask][::3],color='#334155',lw=.6)
        axes[0].set_yscale('symlog',linthresh=1e-4)
        axes[0].set_ylabel('Amplitude\n(symmetric log)')
        axes[0].set_title(key+' — '+('original QC flag' if flagged else 'prior duration agreement'),loc='left')
        centers, rms = rms_track(x,sr)
        level = 20*np.log10(np.maximum(rms,1e-10)/np.quantile(rms,.95))
        axes[1].plot(centers,level,color='#2563eb',lw=1,label='Raw RMS')
        axes[1].axhline(-35,color='#7c3aed',ls='--',lw=1,label='−35 dB criterion')
        axes[1].set_ylim(-85,5)
        axes[1].set_ylabel('RMS dB\nrelative to p95')
        for candidate in raw['candidates']:
            axes[1].plot([candidate['start'],candidate['end']],[-74,-74],color='#7c3aed',lw=5)
        p = np.array(neural['silence_probability'])
        axes[2].plot(np.arange(len(p))*.01+.0125,p,color='#b91c1c',lw=1,label='FSMN silence probability')
        axes[2].axhline(.5,color='#b91c1c',ls=':',lw=1)
        axes[2].set_ylim(-.05,1.05)
        axes[2].set_ylabel('Neural silence\nprobability')
        axes[2].set_xlabel('Seconds in original audio')
        for ax in axes:
            ax.axvspan(h['start'],h['end'],color='#16a34a',alpha=.12,label='MFA silence')
            ax.axvline(h['end'],color='#16a34a',lw=1)
            ax.set_xlim(left,right)
            ax.grid(alpha=.15)
        axes[0].text(.99,.94,f'Adjacent MFA phones: /{last[2]}/ → /{first[2]}/',transform=axes[0].transAxes,
                     ha='right',va='top',fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        axes[1].legend(loc='upper left',fontsize=8,ncol=3)
        axes[2].legend(loc='upper left',fontsize=8)
        path = figures/f'{key}.png'
        fig.savefig(path,dpi=115)
        plt.close(fig)
        raw_text = 'ambiguous' if raw['gap_ms'] is None else f'{raw["gap_ms"]:.0f} ms'
        cards.append(f'''<article data-flagged="{str(flagged).lower()}" id="{key}">
<h2>{key}</h2><p>MFA {h['gap_ms']:.0f} ms · RMS {raw_text} · {html.escape(category)}</p>
<p>{html.escape(explanation)}</p><img loading="lazy" src="figures/flagged-followup/{key}.png" alt="Waveform, RMS and neural probability at the boundary">
<audio controls preload="none" src="../{row['source_audio']}"></audio></article>''')
    out = ROOT/'runs/flagged-followup-v1'
    atomic_json(out/'cases.json',findings)
    neural_summary = read_json(ROOT/'runs/fsmn-probe-v1/summary.json')
    hp = read_json(ROOT/'runs/bandlimited-v1/summary.json')
    old_validation = read_json(ROOT/'runs/pause-validation-v2/summary.json')
    variant_validation = {}
    for name in ('d0','perclip-d0','perclip-d1'):
        path = ROOT/'runs'/f'pause-validation-{name}-v1/summary.json'
        if path.exists():
            variant_validation[name] = read_json(path)
    summary = {'n_clips':40,'original_flagged':34,'cleared':0,'remaining_original_flags':34,
               'duration_disagreements':29,'waveform_ambiguities':5,
               'all_29_RMS_endpoints_within_first_HMM_phone':all(r.get('RMS_end_inside_first_phone',True) for r in findings),
               'neural_validation':neural_summary['fixtures']['0.5'], 'highpass_validation':hp['fixtures'],
               'mfa_variant_validation':{n:{mode:v[mode]['hmm_silence'] for mode in ('absolute','incremental')} for n,v in variant_validation.items()},
               'dither0_repeat':compare([r['variants']['mfa-perclip-d0-repeat-v1']['gap_ms'] for r in findings],
                                        [r['variants']['mfa-perclip-d0-v1']['gap_ms'] for r in findings]),
               'decision':'No estimator promoted; preserve original QC and Gate 1 hold'}
    atomic_json(out/'summary.json',summary)
    validation_table = []
    validation_table.append('| Original MFA configuration | 24/24 | 72/72 | Passed these edited fixtures only |')
    for name,v in summary['mfa_variant_validation'].items():
        validation_table.append(f'| MFA {name} | {v["absolute"]["pass_30ms"]}/24 | {v["incremental"]["pass_30ms"]}/72 | Sensitivity experiment |')
    validation_table.append('| FSMN raw posterior ≥0.5 | 8/24 | 23/72 | Reject as precise pause detector |')
    validation_table.append('| Fixed 20 Hz high-pass + RMS | 24/24 | 58/72 | Reject; ambiguous intervals count as failures |')
    clip_table = []
    for c in findings:
        if not c['original_flags']:
            continue
        raw = c['raw_rms']['gap_ms']
        raw_text = 'ambiguous' if raw is None else f'{raw:.0f}'
        evidence = f'peak {c["spectrum"]["dominant_hz"]:.1f} Hz' if raw is None else f'/{c["first_ending_phone"][2]}/; onset +{c["endpoint_delta_ms_RMS_minus_HMM"]["end"]:.1f} ms'
        clip_table.append(f'| [{c["key"]}](flagged-clips.html#{c["key"]}) | {c["hmm"]["gap_ms"]:.0f} | {raw_text} | {evidence} | Retained |')
    report = '''# Investigation of the 34 flagged clips

All 34 were investigated, along with the six original comparison passes.
**No original flag is defensibly cleared.** The disagreements are now localized,
and several tempting fixes have been rejected with controlled tests. Gate 1
remains unaccepted. No human annotations or billed API calls were used.

## What the clip-level evidence shows

- **29 duration disagreements:** in every case, the RMS detector's ending onset
  is over 30 ms later than MFA's, and falls inside the first phone MFA assigns to
  the ending (allowing 5 ms numerical/frame tolerance). In 26/29, the other endpoint
  already agrees within 30 ms. This concentrates the discrepancy at low-energy
  initial phones, rather than at the whole pause. It is consistent with an
  acoustic-threshold versus phonetic-boundary distinction. Because those phone
  labels come from MFA itself, this is a machine attribution, not proof that MFA
  is right.
- **Five fragmented waveform intervals:** all have dominant spectral peaks at
  10–18 Hz in the HMM interval. A Hann-window analysis puts 48–78% of its energy
  below 20 Hz. This is consistent with subsonic synthesis residue splitting the
  raw RMS mask. Spectral leakage and speech tails prevent treating this statistic
  as a definitive source attribution. Fixed high-pass filtering resolves three
  of the original five ambiguities but creates four others; it is not a fix.
- **Four of the six previous duration passes** have at least one endpoint
  disagreement over 30 ms. Similar length alone does not establish endpoint
  accuracy. The original eligibility rules and records remain unchanged.

## Local experiments and falsification

Kokoro replay produced bit-identical PCM for all 40 recordings. The recovered
punctuation-plus-space allocations span only 50–150 ms and do not represent the
full acoustic pause. No undocumented timestamp offset was fitted. Its durations
are retained as generator diagnostics, not promoted to ground truth.

Original MFA inference repeated within 0–30 ms in pause length (MAE 4.25 ms).
Removing feature dither and changing normalization groups did not remove natural
disagreement: waveform MAE remained about 50 ms. The per-clip, zero-dither run
repeated exactly for all 40, but failed edited-speech validation. Removing
dither changes the model's treatment of digital zeros; determinism is insufficient
evidence of accuracy. Learned weights were bit-identical in the metadata-only
derivative. The original feature configuration remains the reference.

The independent [FSMN VAD model](https://huggingface.co/funasr/fsmn-vad-onnx)
was pinned, downloaded, and run locally on all 40 clips and 96 fixtures. The
[official runtime](https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime)
provides 10 ms frame probabilities. We examined raw silence-class probabilities,
not its utterance padding/hangover. The fixed 0.5 decision underestimates absolute
known gaps by 137.9 ms on average, and passes only 31/96 cases. Sensitivity at 0.3
and 0.7 also fails. It cannot arbitrate a 30 ms disagreement. CPU ONNX inference
uses a separate `.venv-vad` to preserve the GPU environment.

Known-gap passes within 30 ms (missing results count as failures):

| Method | Absolute hard-splice | Incremental insertion | Decision |
|---|---:|---:|---|
'''+ '\n'.join(validation_table)+'''

The 20 Hz filter passed all 72 controlled tone cases, including 10/18 Hz
contamination, yet failed speech insertion coverage. This demonstrates why
clean synthetic signals alone cannot validate a natural-speech pause method.
The earlier 96/96 MFA result remains true for its saved configuration and batch;
it must not be generalized to all preprocessing/normalization choices.

## Every original flagged clip

Times are milliseconds. Positive onset offset means RMS resumes later than MFA.
The linked [audio/plot page](flagged-clips.html) contains all 40 clips with local
players, waveform, RMS, neural posterior, and modeled phone context. It requires
no server, external assets, annotation, or network requests.

| Clip | MFA pause | RMS pause | Localized evidence | QC |
|---|---:|---:|---|---|
'''+ '\n'.join(clip_table)+'''

## Decision and reproducibility

Keep phonetic silence and acoustic quiet duration as separately named quantities.
Do not force them to agree by shifting boundaries, choosing thresholds after seeing
the answers, dropping flagged rows, or treating another local model as an oracle.
No new automatic clearance rule or primary estimator has been introduced.
All 34 original flags remain, with specific machine-readable findings in
`runs/flagged-followup-v1/cases.json`.

Reproduction commands and model/environment locks are in README.md. Original
audio, measurements, and flags were preserved. Additional model assets, inference
outputs, source hashes, and this report's input hashes are recorded locally.
'''
    (ROOT/'reports/flagged-followup.md').write_text(report,encoding='utf-8')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flagged pause investigation</title><style>body{font:16px system-ui;background:#f1f5f9;color:#172033;margin:0 auto;padding:28px;max-width:1100px}header,article{background:white;padding:24px;border-radius:12px;margin:0 0 24px}h1{margin-top:0}h2{font-size:20px}img{width:100%;height:auto}audio{width:100%}p{line-height:1.5}label{display:block;margin-top:16px}article[hidden]{display:none}a{color:#1753a5}</style>
<header><h1>34 flags investigated · 0 cleared</h1><p>29 disagreements fall at the initial ending phone; 5 have fragmented RMS intervals with subsonic energy. These are machine findings, not human annotations or validated natural boundaries. Green shading is MFA silence. Purple bars are every qualifying RMS candidate. The neural detector failed known-gap validation.</p><p><a href="flagged-followup.md">Full findings</a> · <a href="../runs/flagged-followup-v1/cases.json">Machine-readable cases</a></p><label><input id="only" type="checkbox" checked> Show the original 34 flagged clips only</label></header>
'''+ '\n'.join(cards)+'''<script>const filter=document.getElementById('only');function apply(){document.querySelectorAll('article').forEach(a=>a.hidden=filter.checked&&a.dataset.flagged!=='true')}filter.addEventListener('change',apply);apply();</script></html>'''
    (ROOT/'reports/flagged-clips.html').write_text(page,encoding='utf-8')
    dependencies = [ROOT/'runs/pause-audit-v3/measurements.json',ROOT/'runs/corrected-pauses-v1/metrics.json',
                    ROOT/'runs/kokoro-trace-v1/manifest.json',ROOT/'runs/fsmn-probe-v1/manifest.json',
                    ROOT/'runs/fsmn-probe-v1/summary.json',ROOT/'runs/bandlimited-v1/summary.json',
                    ROOT/'runs/subsonic-stress-v1/results.json',ROOT/'docs/flagged-followup-plan.md']
    dependencies += [ROOT/'runs'/n/'manifest.json' for n in variant_names]
    dependencies += [ROOT/'runs'/f'pause-validation-{n}-v1/summary.json' for n in variant_validation]
    artifacts = list(figures.glob('*.png'))+[out/'cases.json',out/'summary.json',ROOT/'reports/flagged-followup.md',ROOT/'reports/flagged-clips.html']
    atomic_json(out/'provenance.json', {'code_sha256':digest(__file__),
        'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in dependencies},
        'artifact_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in artifacts}})
    print(summary)


if __name__=='__main__':
    main()
