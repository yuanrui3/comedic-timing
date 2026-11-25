"""Rebuild the resolution report/figures from saved, auditable measurements."""
from .common import ROOT,read_json,atomic_json,digest
from .measurement import rms_track
import numpy as np
import soundfile as sf
import html

def main():
    audit=read_json(ROOT/'runs/pause-audit-v3/summary.json')
    rows=read_json(ROOT/'runs/pause-audit-v3/measurements.json')
    metrics=read_json(ROOT/'runs/corrected-pauses-v1/metrics.json')
    validation=read_json(ROOT/'runs/pause-validation-v2/summary.json')
    errors=read_json(ROOT/'runs/pause-validation-v2/errors.json')
    noise=read_json(ROOT/'runs/pause-noise-v1/results.json')['summary']
    out=ROOT/'reports/figures/pause-resolution'
    out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    positions=np.arange(len(rows))
    before=np.array([r['legacy_ctc']['gap_ms']-r['legacy_rms']['gap_ms'] for r in rows])
    after=np.array([r['hmm']['gap_ms']-r['legacy_rms']['gap_ms'] for r in rows])
    axes[0].scatter(positions,before,color='#c75138',s=18,label='CTC emission gap minus legacy RMS')
    axes[0].scatter(positions,after,color='#2266a5',s=18,label='Phonetic silence minus legacy RMS')
    axes[0].axhspan(-30,30,color='#278763',alpha=.12,label='+/-30 ms band')
    axes[0].axhline(0,color='#777',linewidth=.7)
    axes[0].set(title='Same 40 unedited clips: residual differences remain',xlabel='Clip (20 per stock voice)',ylabel='Signed difference (ms)')
    axes[0].set_ylim(-165,510)
    axes[0].legend(fontsize=8,loc='upper right')
    for name,label,color in [('ctc_emission_gap','CTC emission gap','#c75138'),('hmm_silence','Phonetic silence','#2266a5'),('legacy_rms','RMS','#278763')]:
        selected=[r for r in errors if r['mode']=='absolute' and r['method']==name]
        durations=sorted({r['inserted_ms'] for r in selected})
        means=[np.mean([r['error_ms'] for r in selected if r['inserted_ms']==d]) for d in durations]
        axes[1].plot(durations,means,'o-',label=label,color=color)
    axes[1].axhspan(-30,30,color='#278763',alpha=.12)
    axes[1].axhline(0,color='#777',linewidth=.7)
    axes[1].set(title='Known hard-splice gaps: mean error over 4 clips',xlabel='Inserted silence (ms)',ylabel='Measured minus known gap (ms)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'before-after.png',dpi=160)
    plt.close(fig)
    chosen=[r for r in rows if r['key'] in ('zh001_zf_001_joke','zh009_zm_010_joke')]
    for r in chosen:
        x,sr=sf.read(ROOT/r['source_audio'])
        t=np.arange(len(x))/sr
        center=sum(r['bracket'])/2
        fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        axes[0].plot(t,x,linewidth=.5,color='#34495e')
        axes[0].set(title=r['key']+' — automatically estimated boundaries',ylabel='Amplitude')
        definitions=[('legacy_ctc','CTC emission gap (not silence)','#c75138'),
                     ('legacy_rms','Legacy low-energy interval','#a17c1a'),
                     ('hmm','Explicit phonetic silence','#2266a5')]
        for i,(key,label,color) in enumerate(definitions):
            d=r[key]
            axes[1].plot([d['start'],d['end']],[2-i,2-i],linewidth=9,color=color,solid_capstyle='butt')
            axes[1].text(d['end']+.025,2-i,f'{d["gap_ms"]:.0f} ms',va='center',fontsize=9)
            for ax in axes:
                ax.axvline(d['start'],color=color,linestyle=':',linewidth=.7,alpha=.65)
                ax.axvline(d['end'],color=color,linestyle=':',linewidth=.7,alpha=.65)
        axes[1].set_yticks([2,1,0],[d[1] for d in definitions],fontsize=9)
        axes[1].set_ylim(-.5,2.5)
        axes[1].set_xlabel('Seconds from start of native waveform')
        axes[1].set_xlim(center-.7,center+.7)
        fig.tight_layout()
        fig.savefig(out/f'{r["key"]}.png',dpi=160)
        plt.close(fig)
    old=audit['ctc_minus_legacy_rms']
    new=audit['hmm_minus_legacy_rms']
    eligible=sum(r['eligible_for_automatic_analysis'] for r in metrics)
    doc=[
        '# Pause detector investigation — 2026-11-14', '',
        '**The CTC measurement bug is fixed; full agreement on natural speech is not yet established.** The old code treated the interval between character-recognition spikes as silence. The corrected pause export measures explicit silence phones from a local phonetic aligner, checks the exact transcript boundary, and keeps CTC distances only as diagnostics. No constant 217 ms subtraction was applied.', '',
        f'On the same 40 clips, keeping the original RMS comparator fixed, mean signed disagreement changed from **+{old["bias_ms"]:.1f} ms to {new["bias_ms"]:.1f} ms**; MAE fell from **{old["mae_ms"]:.1f} to {new["mae_ms"]:.1f} ms**. Only {round(new["within_30ms"]*40)}/40 original-comparator differences are within 30 ms. The replacement therefore removes the large positive bias but does **not** meet a per-clip 30 ms agreement criterion.', '',
        f'The corrected export separately requires a unique waveform interval and agreement within 30 ms: **{eligible}/40 pass; {40-eligible}/40 are flagged**, including ambiguous waveform cases. All 40 estimates and reasons remain saved. Do not run a humor comparison on this small selected subset or silently discard the flagged clips. Gate 1 remains unaccepted for natural-speech measurement confidence; no pilot/main run was launched.', '',
        '![Before/after and absolute recovery](figures/pause-resolution/before-after.png)', '',
        '## What the stronger tests established', '',
        'The original insertion test measured only a change from baseline, so any fixed endpoint bias canceled. The new tests include 24 hard-spliced speech fixtures with exact gap lengths (four joke/control/voice combinations from one script), plus 72 incremental edits across three scripts, two voices and both endings. All use 0, 30, 100, 300, 600 and 1000 ms gaps. Both CTC and phonetic alignment were rerun from each edited waveform and its transcript; reference timestamps were not supplied to either aligner.', '',
        '| Estimator | Absolute cases within 30 ms | Worst absolute error | Increment cases within 30 ms | Worst increment error |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('ctc_emission_gap','Legacy CTC emission gap'),('hmm_silence','Phonetic silence (MFA)'),('legacy_rms','Legacy RMS selection'),('rms_35_unique','RMS with ambiguity flags'),('hysteresis','Amplitude hysteresis'),('preemphasized_rms_35','Pre-emphasized RMS')]:
        a,b=validation['absolute'][key],validation['incremental'][key]
        doc.append(f'| {label} | {a["pass_30ms"]}/{a["expected"]} | {a["max_abs_ms"]:.1f} ms | {b["pass_30ms"]}/{b["expected"]} | {b["max_abs_ms"]:.1f} ms |')
    doc += ['', 'The 6 missing incremental cases for ambiguity-aware RMS count as failures in the denominator. The original RMS selector sometimes latched onto a 5–10 ms dip and then jumped to a different interval after insertion. Its largest incremental error was 110 ms. The new selector reports multiple qualifying intervals as ambiguous rather than choosing whichever is closest to the midpoint. Truly adjacent spoken intervals remain zero; missing, noise-labelled, mismatched and overlapping alignments remain missing.', '',
            'Hard-splice fixtures remove the original pause and weak edge speech, so their silence boundaries are exact by construction. They verify acoustic measurement on edited speech, **not** absolute phonetic boundary accuracy in intact natural speech. The HMM passed 96/96 edited-speech cases; that does not make it ground truth for the original clips.', '',
            '## Independent checks and unsuccessful fixes', '',
            'The fixed threshold grid was 25/30/35/40 dB below the clip’s 95th-percentile frame RMS. Hysteresis was specified before its comparison and worsened natural-clip agreement (89.5 ms MAE). A subsequent fixed 16 kHz, 0.97 pre-emphasis check handled low-frequency residue and recovered all 72 incremental edits, but still disagreed with the HMM by 59.5 ms MAE on the original clips. Neither was promoted based on agreement tuning. The full grid and missing counts are retained in the audit JSON.', '',
            'Additional deterministic fixtures vary sample rate (16/24 kHz), signal amplitude (0.2/0.002), 60 dB-below-signal white noise and a DC offset. Pre-emphasized RMS passed all 72 cases within 30 ms. Raw RMS and hysteresis each passed 24/24 clean and 24/24 white-noise cases, but only 8/24 DC-offset cases. This is a waveform stress test, not an additional speech alignment result.', '',
            '![Female voice example](figures/pause-resolution/zh001_zf_001_joke.png)', '',
            '![Fragment-selection example](figures/pause-resolution/zh009_zm_010_joke.png)', '',
            '## Implementation and reproducibility', '',
            'MFA 2.2.17 / Kaldi 5.5.1068 runs locally in a separate Windows environment. It uses the [Mandarin MFA v2.0.0a acoustic model](https://mfa-models.readthedocs.io/en/latest/acoustic/Mandarin/Mandarin%20MFA%20acoustic%20model%20v2_0_0a.html) and the [Mandarin (China) v2.0.0 pronunciation dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/Mandarin/Mandarin%20(China)%20MFA%20dictionary%20v2_0_0.html), attributed to Michael McAuliffe and Morgan Sonderegger, CC BY 4.0. Exact release URLs, SHA256s, Python dependencies and key native executable hashes are saved. No speaker adaptation, model training, fine-tuning or PostgreSQL server was used. Feature normalization is part of fixed inference.', '',
            'The dictionary does not contain every individual character, so a deterministic minimum-token dictionary segmentation covers each setup and ending separately and preserves their shared boundary. All pronunciations are from the downloaded dictionary; no unknown-word substitutions were used. Pronunciation variants have equal weights and silence uses the model’s global prior, rather than word-specific silence probabilities. MFA receives a 16 kHz PCM16 resampled copy; original float waveforms remain unchanged.', '',
            'Runtime logs confirm 25 ms MFCC windows / 10 ms shift, model-default dither=1 and pre-emphasis=0.97. The YAML dither=0 request was ignored by this MFA version because it reads acoustic-model metadata; the report records the actual setting. Only first-pass decoding ran. “Compiling training graphs” in the program log is transcript-graph construction, not model training. The GPU CTC pass remains local and unchanged apart from an explicit “not silence” type label.', '',
            'The underlying CTC issue is also documented in [TorchAudio’s explanation of blank-token ambiguity](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html#inconsistent-treatment-of-blank-token). Peaky character emission supports do not specify complete acoustic durations. The independent HMM and waveform estimators differ in how they treat weak consonants, noise and transitions. Those residual differences are retained, not calibrated away using the comparison set.', '',
            'One command per result: `python -m timing.measure_pauses` exports the corrected metric; `python -m timing.pause_audit --out pause-audit-v3` rebuilds natural-clip comparisons; `python -m timing.pause_evaluate --out pause-validation-v2` recomputes edited-speech recovery; `python -m timing.pause_noise` recomputes controlled-background results; `python -m timing.pause_report` rebuilds this report and its figures. Use the project `.venv/Scripts/python.exe`; see [README](../README.md) for model execution commands.', '',
            'Artifacts: [corrected measurements](../runs/corrected-pauses-v1/metrics.json), [audit](../runs/pause-audit-v3/summary.json), [validation](../runs/pause-validation-v2/summary.json), [noise checks](../runs/pause-noise-v1/results.json), [pre-comparison decisions](../docs/pause-resolution-plan.md), and [local audio review page](pause-resolution-audio.html). No human annotation, billed API, new speech synthesis, system driver change or push was used in this continuation.', '']
    (ROOT/'reports/pause-resolution.md').write_text('\n'.join(doc),encoding='utf-8')
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Pause measurement investigation</title>',
          '<style>body{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172638}img{max-width:100%}audio{width:100%;max-width:500px}article{padding:24px 0;border-top:1px solid #ccd5dd}.note{padding:18px;background:#fff2dc}small{color:#536478}</style>',
          '<h1>Pause measurement investigation</h1><p class="note">The CTC emission-gap bug is fixed. Residual natural-speech disagreement remains; these are exploratory estimates, not validated humor findings. AI-generated stock voices only; human annotation deferred.</p>',
          '<img src="figures/pause-resolution/before-after.png" alt="Comparison of old and new measurements and recovery on known gaps">']
    for r in chosen:
        page += [f'<article><h2>{r["key"]}</h2><audio controls preload="none" src="../{r["source_audio"]}"></audio>',
                 f'<p>CTC emission gap: {r["legacy_ctc"]["gap_ms"]:.0f} ms (not silence). Phonetic silence: {r["hmm"]["gap_ms"]:.0f} ms. Legacy RMS: {r["legacy_rms"]["gap_ms"]:.0f} ms.</p>',
                 f'<img src="figures/pause-resolution/{r["key"]}.png" alt="Automatically estimated intervals for {r["key"]}"></article>']
    page.append('</html>')
    (ROOT/'reports/pause-resolution-audio.html').write_text('\n'.join(page),encoding='utf-8')
    atomic_json(ROOT/'runs/pause-report-v1.json',{'code_sha256':digest(__file__),'inputs':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (
        ROOT/'runs/pause-audit-v3/summary.json',ROOT/'runs/pause-validation-v2/summary.json',ROOT/'runs/pause-noise-v1/results.json',ROOT/'runs/corrected-pauses-v1/metrics.json')}})
    print('Wrote reports/pause-resolution.md, three figures, and local audio page.')

if __name__=='__main__':
    main()
