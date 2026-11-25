"""Generate the second-aligner investigation and auditable per-clip plots."""
from .common import ROOT,read_json,atomic_json,digest
from .ac_rms import ac_track
from .detrended_rms import linear_track
from .measurement import rms_track
from .pause_evaluate import errors
import argparse,html
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'MFA2':'#2563eb','MFA3':'#0d9488','Charsiu':'#9333ea','raw':'#d97706','AC':'#64748b','linear':'#dc2626','Silero':'#15803d'}

def charsiu_fixture_scores():
    lookup={p.stem:read_json(p)['methods']['4']['forced']['gap_ms'] for p in (ROOT/'runs/charsiu-fixtures-v1/aligned').glob('*.json')};rows=[]
    for r in read_json(ROOT/'runs/pause-fixtures-v1/references.json'):
        value=lookup[r['key']];zero=lookup[r['base_key']+'_'+r['mode']+'_0000ms']
        recovered=value if r['mode']=='absolute' else value-zero
        rows.append({'key':r['key'],'mode':r['mode'],'error_ms':recovered-r['inserted_ms']})
    return rows,{m:errors([r for r in rows if r['mode']==m]) for m in ('absolute','incremental')}

def clip_plot(case,energy,audit,dest):
    key=case['key'];x,sr=sf.read(ROOT/audit['source_audio']);left=max(0,audit['bracket'][0]-.08);right=min(len(x)/sr,audit['bracket'][1]+.08)
    fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True,gridspec_kw={'height_ratios':[1.1,1.2,1.1,1.]},layout='constrained')
    t=np.arange(len(x))/sr;mask=(t>=left)&(t<=right);axes[0].plot(t[mask],x[mask],color='#1e293b',lw=.6)
    axes[0].set_ylabel('Native amplitude');axes[0].set_title(key+' | '+('originally flagged' if case['original_flags'] else 'original duration pass'),loc='left',fontweight='bold')
    intervals=[('MFA2',case['mfa2']),('MFA3',case['mfa3']),('Charsiu',case['charsiu']),('raw',energy['methods']['raw']),('AC',energy['methods']['AC']),('linear',energy['methods']['linear'])]
    for i,(name,r) in enumerate(intervals):
        spans=r.get('candidates',[]) if name in ('raw','AC','linear') else [r]
        for span in spans:
            axes[1].broken_barh([(span['start'],span['end']-span['start'])],(i-.28,.56),facecolors=COLORS[name],alpha=.8)
        if r.get('gap_ms')==0:axes[1].plot(r.get('start',case['mfa2']['end']),i,marker='|',color=COLORS[name],ms=12)
        if r.get('gap_ms') is None:axes[1].text(right-.005,i,'ambiguous',ha='right',va='center',fontsize=8,color=COLORS[name])
    axes[1].set_yticks(range(len(intervals)),[n for n,_ in intervals]);axes[1].invert_yaxis();axes[1].set_ylabel('Estimated intervals')
    for name,fn in (('raw',rms_track),('AC',ac_track),('linear',linear_track)):
        centers,rms=fn(x,sr)[:2];scale=max(float(np.quantile(rms,.95)),1e-12);q=(centers>=left)&(centers<=right)
        axes[2].plot(centers[q],20*np.log10(np.maximum(rms[q],1e-10)/scale),label=name,color=COLORS[name],lw=1.4)
    axes[2].axhline(-35,color='#444',ls=':',lw=1);axes[2].set_ylim(-75,5);axes[2].set_ylabel('Frame energy (dB\nrelative to own p95)');axes[2].legend(ncol=3,fontsize=8)
    posterior=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{key}.npz')['posterior'];ct=(np.arange(len(posterior))+.5)*.01
    q=(ct>=left)&(ct<=right);axes[3].step(ct[q],posterior[q,1],where='mid',label='Charsiu SIL',color=COLORS['Charsiu'])
    silero=read_json(ROOT/'runs/silero-probe-v1/frames'/f'{key}.json');sp=np.asarray(silero['speech_probability']);st=(np.arange(len(sp))+.5)*.032
    q=(st>=left)&(st<=right);axes[3].step(st[q],1-sp[q],where='mid',label='Silero non-speech',color=COLORS['Silero'],alpha=.85)
    axes[3].axhline(.5,color='#444',ls=':',lw=1);axes[3].set_ylim(-.05,1.05);axes[3].set_ylabel('Model probability\n(uncalibrated)');axes[3].legend(ncol=2,fontsize=8)
    for ax in axes:
        ax.set_xlim(left,right);ax.grid(alpha=.18);ax.axvline(case['mfa2']['end'],color=COLORS['MFA2'],ls='--',lw=.8,alpha=.6)
    axes[-1].set_xlabel('Seconds in original recording; vertical line = original MFA ending onset')
    fig.savefig(dest,dpi=140);plt.close(fig)

def ablation_plot(dest):
    key='zh003_zf_001_joke';c=read_json(ROOT/'runs/onset-ablation-v1/cases'/f'{key}__mute_plus40ms.json')
    base,sr=sf.read(ROOT/'data/audio/kokoro-diagnostic-v1'/f'{key}.wav');t=np.arange(len(base))/sr
    left,right=c['mute_start']-.12,c['mute_end']+.13;mask=(t>=left)&(t<=right)
    orig=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{key}.npz')['posterior'][:,1]
    modified=np.load(ROOT/'data/audio/onset-ablation-v1'/f'{key}__mute_plus40ms.npz')['posterior'][:,1]
    fig,axes=plt.subplots(2,1,figsize=(10,5.7),sharex=True,layout='constrained')
    axes[0].plot(t[mask],base[mask],lw=.7,color='#334155',label='Original waveform')
    axes[0].plot([c['mute_start'],c['mute_end']],[0,0],lw=3,color='#dc2626',label='Replaced by exact zeros')
    axes[0].set_title('Context can anticipate speech inside exact digital silence',loc='left',fontweight='bold');axes[0].set_ylabel('Amplitude');axes[0].legend(fontsize=9)
    centers=(np.arange(len(orig))+.5)*.01;q=(centers>=left)&(centers<=right)
    axes[1].step(centers[q],orig[q],where='mid',label='Original SIL probability',color='#64748b')
    axes[1].step(centers[q],modified[q],where='mid',label='Muted SIL probability',color=COLORS['Charsiu'])
    axes[1].axvline(c['forced']['end'],color=COLORS['Charsiu'],ls='--',label='Muted predicted onset')
    for ax in axes:
        ax.axvspan(c['mute_start'],c['mute_end'],color='#fca5a5',alpha=.18);ax.axvline(c['mute_end'],color='#dc2626',ls=':',label='Waveform resumes' if ax is axes[1] else None);ax.grid(alpha=.15);ax.set_xlim(left,right)
    axes[1].set_ylim(-.05,1.05);axes[1].set_ylabel('Charsiu SIL probability');axes[1].set_xlabel('Seconds; upstream 10 ms cells, no fitted offset');axes[1].legend(fontsize=8,ncol=2)
    fig.savefig(dest,dpi=150);plt.close(fig)

def artifact_plot(cases,dest):
    fig,axes=plt.subplots(5,1,figsize=(11,10),layout='constrained')
    for ax,c in zip(axes,cases):
        key=c['key'];h=c['mfa2'];base=np.load(ROOT/'data/audio/kokoro-signals-v1'/f'{key}__baseline.npz');t=np.arange(len(base['output']))/24000
        q=(t>=h['start']-.03)&(t<=h['end']+.03)
        ax.plot(t[q],base['output'][q],lw=.65,color='#94a3b8',label='Native waveform')
        ax.plot(t[q],base['output_lowpass20'][q],lw=1.5,color='#d97706',label='Sub-20 Hz diagnostic')
        ax.axvspan(h['start'],h['end'],color='#2563eb',alpha=.07);ax.set_ylim(-.0035,.0035);ax.set_xlim(h['start']-.03,h['end']+.03)
        ax.set_title(key,loc='left',fontsize=10);ax.grid(alpha=.15);ax.set_ylabel('Amplitude')
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Seconds; shaded interval is original MFA silence, not ground truth')
    fig.suptitle('Five fragmented raw-RMS intervals: native baseline wander',fontweight='bold');fig.savefig(dest,dpi=140);plt.close(fig)

def stage_plot(dest):
    rows=[read_json(p) for p in (ROOT/'runs/vocoder-stage-audit-v1/cases').glob('*.json')]
    groups=['noise_0','noise_1',*[f'residual_{i}' for i in range(6)],'stage_0_residuals','stage_1_residuals']
    males=sorted({r['key'] for r in rows if r['target']});keys=males+[k.replace('zm_010','zf_001') for k in males]
    lookup={(r['key'],r['group']):r['level_normalized_sub20_ratio'] for r in rows}
    values=np.array([[lookup[key,group] for key in keys] for group in groups]);fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    bound=max(abs(np.log2(values)).max(),1);im=ax.imshow(np.log2(values),cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    for i in range(len(groups)):
        for j in range(len(keys)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(np.log2(values[i,j]))>bound*.6 else '#172033')
    ax.set_yticks(range(len(groups)),['Source block 0','Source block 1',*[f'Stage {i//3}: kernel {[3,7,11][i%3]}' for i in range(6)],'Stage 0: all three','Stage 1: all three'])
    ax.set_xticks(range(len(keys)),[k.replace('zh','').replace('_zm_010','').replace('_zf_001','') for k in keys],rotation=45,ha='right')
    ax.axvline(4.5,color='#172033',lw=2);ax.set_title('Vocoder block style swaps: slow component relative to surrounding level\nLeft: five male targets → female style | Right: matched female → male style',fontsize=12)
    ax.set_xlabel('Numbers are ratios: 1 = unchanged; each column keeps its own original timing and F0/noise')
    cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label('log2 ratio (color only)')
    fig.savefig(dest,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');parser.add_argument('--skip-plots',action='store_true');args=parser.parse_args()
    cases=read_json(ROOT/'runs/phonetic-evidence-v1/cases.json');audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    energy={r['key']:r for r in read_json(ROOT/'runs/detrended-rms-v1/measurements.json') if r['mode']=='natural'}
    summary=read_json(ROOT/'runs/phonetic-evidence-v1/summary.json');signal=read_json(ROOT/'runs/signal-origin-v1/summary.json');detrend=read_json(ROOT/'runs/detrended-rms-v1/summary.json')
    stability=read_json(ROOT/'runs/intact-stability-analysis-v1/summary.json');ablation=read_json(ROOT/'runs/onset-ablation-v1/summary.json');hmm_ablation=read_json(ROOT/'runs/mfa-ablation-analysis-v1/summary.json')
    paired=read_json(ROOT/'runs/paired-measurement-sensitivity-v1/summary.json')
    pcm=read_json(ROOT/'runs/pcm-audit-v1/summary.json');localization=read_json(ROOT/'runs/decoder-localization-v1/summary.json')
    repeats=read_json(ROOT/'runs/repeatability-audit-v1/summary.json');selector=read_json(ROOT/'runs/quiet-selector-v2/summary.json')
    level=read_json(ROOT/'runs/style-level-audit-v1/summary.json');stages=read_json(ROOT/'runs/vocoder-stage-audit-v1/summary.json')
    holdout_run='holdout-interim-v1' if args.draft else 'holdout-analysis-v2';holdout=read_json(ROOT/'runs'/holdout_run/'summary.json')
    char_rows,char_scores=charsiu_fixture_scores();out=ROOT/'runs/second-aligner-investigation-v1';atomic_json(out/'charsiu-fixture-validation.json',char_rows)
    targets=[r for r in cases if r['category']=='RMS_endpoint_inside_first_HMM_phone'];five=[r for r in cases if r['category']=='fragmented_RMS_subsonic_energy']
    figdir=ROOT/'reports/figures/second-aligner';figdir.mkdir(parents=True,exist_ok=True)
    if not args.skip_plots:
        for c in cases:clip_plot(c,energy[c['key']],audit[c['key']],figdir/f'{c["key"]}.png')
        ablation_plot(figdir/'onset-ablation.png');artifact_plot(five,figdir/'five-artifacts.png')
    if not args.skip_plots or not (figdir/'vocoder-stage-effects.png').exists():stage_plot(figdir/'vocoder-stage-effects.png')
    scores={r['key']:r for r in read_json(ROOT/'runs/signal-origin-v1/cases.json')};items=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    merged=[]
    for c in cases:
        intervals=[c[m] for m in ('mfa2','mfa3','charsiu')]
        merged.append({**c,'energy':energy[c['key']]['methods'],'source_audio':audit[c['key']]['source_audio'],
            'model_range_ms':{side:1000*(max(v[side] for v in intervals)-min(v[side] for v in intervals)) for side in ('start','end')},
            'range_interpretation':'uncalibrated model spread, not a confidence interval','original_flags_cleared':False,
            'investigation_status':'onset_mechanism_corroborated_accuracy_unproven' if c in targets else 'native_decoder_artifact_corroborated_no_validated_repair' if c in five else 'original_pass_retained_endpoint_QC_separate'})
    atomic_json(out/'cases.json',merged)
    mfa3validation=read_json(ROOT/'runs/pause-validation-mfa3-v1/summary.json')
    # Read current numerical results; no hand-entered table values.
    source_rows=[]
    for c in five:
        r=scores[c['key']];source_rows.append(f"| `{c['key']}` | {r['native_sub20_rms']:.6f} | {r['alternate_seed_lowpass_correlation']:.3f} | {100*r['interventions']['source_zeroed']['sub20_rms_ratio']:.1f}% | {100*r['interventions']['decoder_other_stock_style']['sub20_rms_ratio']:.1f}% |")
    holdout_rows=[]
    for method,v in holdout.items():
        a,b=v['absolute'],v['incremental'];holdout_rows.append(f"| {method} | {a['pass_30ms']}/{a['expected']} | {b['pass_30ms']}/{b['expected']} | {v['all']['missing']} | {v['all']['max_abs_ms']:.1f} |")
    comparison_rows=[]
    for model,label in (('charsiu','Charsiu frame classifier'),('mfa3','New MFA Mandarin v3')):
        s=summary['comparison'][model];comparison_rows.append(f"| {label} | {s['end']['pass_30ms']}/29 | {s['end']['mae_ms']:.1f} ms | {s['end']['max_abs_ms']:.1f} ms | {s['start']['pass_30ms']}/29 |")
    localization_rows=[]
    for method,label in (('generator_style_only','Vocoder style only'),('decoder_features_style_only','Pre-vocoder decoder style only'),('generator_style_gamma_only','Vocoder style scale only'),('generator_style_beta_only','Vocoder style shift only')):
        r=localization[method];lo,hi=r['sub20_ratio_range'];snrlo,snrhi=r['outside_gap_SNR_range_dB']
        localization_rows.append(f"| {label} | {100*lo:.1f}–{100*hi:.1f}% | {snrlo:.2f} to {snrhi:.2f} |")
    natural_rows=[]
    for c in merged:
        fmt=lambda r:'missing' if r['gap_ms'] is None else f"{r['gap_ms']:.0f}"
        natural_rows.append(f"| `{c['key']}` | {'yes' if c['original_flags'] else 'no'} | {fmt(c['mfa2'])} | {fmt(c['mfa3'])} | {fmt(c['charsiu'])} | {fmt(c['raw'])} | {c['model_range_ms']['end']:.0f} |")
    report=f'''# Second aligner and source investigation

{'**DRAFT: expanded Charsiu CPU recovery run is still in progress.**' if args.draft else '**Completed local exploration; original stimuli, primary estimates and all 34 flags are preserved.**'}

The new aligners corroborate an earlier phonetic onset in all 29 disputed clips.
That agreement is useful, but it does **not** establish 30 ms absolute accuracy:
all three phonetic models can anticipate speech inside deliberately muted audio.
The other five disagreements involve slow native Kokoro output that depends on
the decoder/style path. It is not introduced by resampling, and removing harmonic
excitation does not remove it. No candidate repair passed the broader validation
without limitations. Gate 1 remains unaccepted; no humor or main-study inference
was made. No human annotations, billed APIs, training or real-voice cloning.

[Per-clip plots and original audio](second-aligner-clips.html) include all 40 clips,
not only the 34 flagged ones. [Earlier investigation](flagged-followup.md) remains
available as the historical result. The original +217 ms CTC disagreement was a
different measurement-definition error; no constant offset is fitted here.

## What is corroborated about the 29 onsets

Both new methods agree with the original MFA ending onset within 30 ms in every
target. Setup endings are less consistent, so onset agreement must not be read as
agreement on the entire pause.

| New method versus original MFA | Onsets within 30 ms | Onset MAE | Largest onset difference | Setup ends within 30 ms |
|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

All three onset estimates span at most 30 ms in 27/29 cases. The two `zh009_zf_001`
endings span 40 ms across the three models, despite each new model agreeing with
the original individually. Model ranges are descriptive, not confidence intervals.

Charsiu's unconstrained frame argmax assigns a non-silence class to every analysed
frame between original MFA onset and later RMS onset, across all 29 clips. Its
textless detector gives the same onset corroboration in 28/29. The exception,
`zh006_zf_001_joke`, has a 30 ms gap that the upstream four-frame minimum suppresses;
the predeclared two-frame sensitivity recovers 30 ms. This is a known minimum-length
effect, not an ambiguous interval. The textless method uses a broad existing search
bracket to localize the relevant boundary, but no transcript to classify frames.

**A falsification check limits this evidence.** Zeroing the disputed region with a
20 ms left guard, and separately extending the mute 40 ms past RMS onset, makes
Charsiu predict onset over 30 ms before waveform resumption in **29/29 cases in
both conditions**. Its convolutional waveform windows can be wholly inside zeros
while the full-context network predicts a phone. With the longer mute, original
MFA does this in {hmm_ablation['MFA2']['mute_plus40ms']['early_onset_over30ms']}/29 cases and newer MFA in
{hmm_ablation['MFA3']['mute_plus40ms']['early_onset_over30ms']}/29. Muting removes speech while retaining the
transcript, so this is a context/phone-completion challenge, not proof that original
natural onsets are wrong. It does show why model agreement and posterior confidence
cannot serve as independent acoustic ground truth.

![An onset anticipated inside exact digital silence](figures/second-aligner/onset-ablation.png)

## Where the five low-frequency artifacts originate

All 40 native Kokoro waveforms were replayed bit-for-bit, and their inverse-STFT
reconstructions also matched. For the five targets, 24→16→24 kHz resampling changes
the sub-20-Hz diagnostic by at most {100*signal['max_resampling_relative_sub20_error']:.5f}% RMS relative
to its original level. Actual PCM16 encoding changes it by at most
{100*pcm['target_max_actual_PCM16_relative_sub20_error']:.2f}%. All 40 actual encodings match the historical MFA input copies exactly;
a simple floor-quantizer approximation differs by up to one least-significant bit.
The component exists before either conversion.

| Clip | Native sub-20-Hz RMS | Correlation after noise-seed change | Retained after excitation zeroing | Retained after decoder-style swap |
|---|---:|---:|---:|---:|
{chr(10).join(source_rows)}

Predicted F0 is mostly below the source's 10 Hz voicing gate in these intervals;
raising that gate to 50 Hz barely changes the artifact. Zeroing the entire harmonic
excitation branch retains {100*signal['source_zeroed_sub20_ratio_range'][0]:.1f}–{100*signal['source_zeroed_sub20_ratio_range'][1]:.1f}% of its RMS. Swapping only decoder
stock-voice style, while holding duration/F0/noise conditioning fixed, retains
{100*signal['decoder_style_swap_sub20_ratio_range'][0]:.1f}–{100*signal['decoder_style_swap_sub20_ratio_range'][1]:.1f}%. This supports a decoder/style-dependent residual; it does not identify
one faulty learned parameter or exclude every possible F0-conditioning effect.

A second intervention separates the decoder before the vocoder from style
conditioning inside the vocoder. Duration, F0 and noise conditioning stay fixed;
only existing stock-voice style components are exchanged. Native baselines replay
exactly. The following ranges cover the five artifact clips:

| Intervention | Retained sub-20-Hz RMS | Outside-gap waveform SNR versus original, dB |
|---|---:|---:|
{chr(10).join(localization_rows)}

The strongest reduction comes from vocoder style conditioning; changing only the
preceding decoder features does not reduce the slow component in these five cases.
Both scale and shift contribute. Applying the male vocoder style to the five
matched female clips increases this component by 2.1–2.9 times. These controlled
swaps localize a dependency, but speech changes substantially too: the SNR column
measures sample differences, not perceptual quality. They are not artifact repairs.

This effect is not explained by a simple overall gain reduction. With the same
outside-gap sample mask (100 ms margins), vocoder-style exchange raises surrounding
RMS by 12–16%. Dividing slow-component RMS changes by that level change leaves
{100*level['generator_style_only']['five_male_targets']['level_normalized_sub20_ratio_range'][0]:.1f}–{100*level['generator_style_only']['five_male_targets']['level_normalized_sub20_ratio_range'][1]:.1f}% of the original relative component.

A final 100-intervention audit exchanges style in each of eight residual blocks
and each of the two three-block waveform stages, for the five targets and five
matched reverse swaps. Own-style sham hooks reproduce all ten originals exactly.
The last stage's widest-kernel block (`resblocks.5`, kernel 11) alone leaves
{100*stages['residual_5']['male_targets']['level_normalized_sub20_ratio_range'][0]:.1f}–{100*stages['residual_5']['male_targets']['level_normalized_sub20_ratio_range'][1]:.1f}% of the relative slow component in every target. Reverse swaps increase it
{stages['residual_5']['female_reverse_swaps']['level_normalized_sub20_ratio_range'][0]:.1f}–{stages['residual_5']['female_reverse_swaps']['level_normalized_sub20_ratio_range'][1]:.1f} times. Surrounding waveforms still change greatly (target SNR 0.5–0.8 dB).
The figure reports all groups, including weak and mixed effects. Such partial
stock-style combinations are outside the native conditioning pattern and effects
are not additive; this identifies a sensitive pathway, not a uniquely faulty layer.

![All fixed vocoder stage style interventions](figures/second-aligner/vocoder-stage-effects.png)

The vocoder uses a 20-sample inverse FFT (1,200 Hz bin spacing). Summing individual
bin reconstructions reproduces native audio within floating-point tolerance. Its
slow component is almost entirely the residual of the lowest two synthesis-bin
contributions. Removing only the DC bin increases the artifact. These are
time-varying synthesis components, so deleting a bin is not a sub-20-Hz filter.
The style/source probes also change speech outside the gap and are not repairs.
The original 10–18 Hz peaks came from short windows; they do not establish a
stationary oscillator. Baseline wander is the more cautious description.

![Native baseline wander in the five clips](figures/second-aligner/five-artifacts.png)

## Why the proposed corrections were not promoted

Per-frame mean subtraction (AC RMS) makes three of the five old ambiguous intervals
unique, leaves the two `zh010_zm_010` cases ambiguous, and introduces ambiguity in
`zh005_zm_010_control`. Linear detrending makes all 40 natural intervals unique.
Both retain the original 10 ms frame, 5 ms hop, −35 dB relative threshold, 20 ms
minimum and missing/ambiguity rules; no threshold was fitted to MFA.

They both pass the original 96 edited-speech cases. AC RMS also passes 480 controlled
tone/background cases, and linear detrending passes 576 including linear drift.
But linear detrending labels 16 non-silent 40–60 Hz probes at ≥−30 dB relative
amplitude as quiet, whereas raw and AC RMS do not. Across 27,196 model-labelled
phone-interior frames (20 ms margins), AC adds 421 quiet frames and linear detrending
adds 632. Those phone labels are diagnostic, not truth. More coverage is therefore
not evidence of better phonetic measurement. Natural gap MAE versus MFA increases
to 63.2 ms for AC and 66.5 ms for linear detrending.

## Recovery beyond the original fixture set

The original 24 absolute fixtures cut approximately 50–70 ms from the first ending
phone and 45–65 ms from the last setup phone, as estimated by original MFA. Their
digital silence is exact, but the original phonetic transcript no longer fully
describes the waveform. Charsiu passes {char_scores['absolute']['pass_30ms']}/24 absolute cases and
{char_scores['incremental']['pass_30ms']}/72 increments. New MFA passes 22/24 and 72/72. Original MFA's historical
96/96 result remains true for that set and configuration.

A new held-out set uses six other scripts, both voices/endings, both edit types,
and the same six delays: 288 cases. The fixed cutting rules were retained. Each
method was run on the resulting waveform; timestamps were not merely shifted.
The broad search bracket is mapped through the known edit to provide localization.

| Method | Absolute within 30 ms /144 | Incremental within 30 ms /144 | Missing /288 | Largest observed error, ms |
|---|---:|---:|---:|---:|
{chr(10).join(holdout_rows)}

The streaming [Silero v6.2.3 model](https://github.com/snakers4/silero-vad/releases/tag/v6.2.3)
was added as a past-context contrast to Charsiu. Raw threshold 0.5 passes 93/96 old
cases, and the predeclared 0.7 sensitivity passes 96/96; neither generalizes
perfectly to the new set. Averaging four fixed 0/8/16/24 ms chunk phases at threshold
0.5 improves quantization but also fails some cases. It is not a selected replacement.
All missing values count as failures. A zero estimate for a 30 ms insertion can
pass the requested tolerance, so pass counts alone overstate short-gap detection.
Hard-splice results test response to edited audio; they do not supply natural labels.

A separate implementation audit found that the old quiet-interval selector could
mark a bracket-edge blip missing before checking whether it met the 20 ms minimum.
Version 2 qualifies the whole run first. This fixes {len(selector['changes'])} fixture/method entries
across two zero-insertion clips and changes **none of the 40 natural estimates**.
Held-out totals become 264/288 for raw RMS, 268/288 for AC RMS and 286/288 for linear
RMS; the table above retains the originally run versions. The two remaining
linear-RMS failures are genuinely censored by the search bracket. No original
flag is cleared, and the quiet-speech caveat remains.

## Stability, implementation and reproducibility

All three aligners keep both boundaries within 30 ms of their own baseline after
7/100/500 ms leading padding, for all 40 intact recordings. A 0.1 gain is less
stable for MFA: original MFA gap MAE is 27.0 ms with a 110 ms maximum, newer MFA
19.3 ms with a 50 ms maximum; Charsiu is unchanged. Repeating original MFA with
zero dither reduces gain sensitivity to 11.5 ms MAE (40 ms maximum), but the earlier
zero-dither absolute-recovery failures remain. Zero dither is not promoted.
The gain challenge includes MFA's PCM16 copy and feature dither, so this is
sensitivity of the complete measurement pipeline, not just the learned weights.

Four fresh batches plus the saved baseline give five estimates per clip for each
MFA version. Across all 40 clips, the largest observed onset range is
{repeats['mfa2']['end_range_max_ms']:.0f} ms for original MFA and {repeats['mfa3']['end_range_max_ms']:.0f} ms for newer MFA;
the largest gap range is 30 ms for both. These are observed repeat ranges, not
accuracy bounds or confidence intervals.

Estimator differences partly cancel in matched pairs, but not completely. Across
all 20 diagnostic script/voice pairs, switching from original MFA to newer MFA
changes the paired contrast by {paired['mfa3']['mean_absolute_change_ms']:.1f} ms on average in absolute value
(maximum {paired['mfa3']['max_absolute_change_ms']:.0f} ms); switching to Charsiu changes it by
{paired['charsiu']['mean_absolute_change_ms']:.1f} ms (maximum {paired['charsiu']['max_absolute_change_ms']:.0f} ms). Raw RMS lacks a value for
{paired['raw']['missing']}/20 pairs. These are differences between measurement methods with all flags
retained, not estimates of true error or a test of comedic timing.

Charsiu's 182 observed pinyin syllables match the pinned upstream preprocessing,
and all 40 DTW paths match upstream exactly. The adapter retains occurrence IDs to
avoid the upstream greedy repeated-phone word-mapping problem. Models were loaded
with no missing, unexpected or mismatched weights. No timestamp offset was fitted.
Charsiu's architecture differs from MFA, but automatically generated training
labels can share alignment assumptions; the Mandarin checkpoint's full training
provenance is not specified. See the [upstream code](https://github.com/lingjzhu/charsiu)
and [model card](https://huggingface.co/charsiu/zh_xlsr_fc_10ms).

MFA 3.0.7 and the official [Mandarin v3 acoustic model](https://mfa-models.readthedocs.io/en/latest/acoustic/Mandarin/Mandarin%20MFA%20acoustic%20model%20v3_0_0.html)
run in `.runtime/mfa3/`, separately from MFA 2. Two missing dictionary words received
automatic pronunciations from the matching official G2P model. `language: unknown`
disables re-tokenization of already dictionary-tokenized transcripts; the acoustic
model remains Mandarin. No adaptation, fine-tuning or database server was used.
Optional Conda test-file extraction hit Windows path limits; native components and
pinned wheels form the verified hybrid runtime. Failed attempts remain logged.

Licenses: MFA models/dictionaries are CC-BY-4.0 (McAuliffe and Sonderegger); Charsiu
code is MIT but checkpoint metadata does not state an explicit weight license, so
it stays a local exploratory dependency, not a redistributed asset. Silero's model
is MIT, [confirmed by its maintainer](https://github.com/snakers4/silero-vad/discussions/817).
The retained new model/native-download/cache inventory is approximately 2.95 GB,
below the approved 15 GB allowance; this is an artifact inventory, not a network
traffic meter. No system installation or driver change occurred.

Exact versions and hashes: `requirements-exploration-lock.txt`,
`requirements-mfa3-lock.txt`, `requirements-vad-lock.txt`, `runs/model-locks/`, and
`runs/exploration-environment-v1/`. All long inference runs checkpoint per clip or
aligner batch. A tuple/list manifest comparison bug was fixed without altering
stored scientific configurations. [Reproduction commands](../docs/second-aligner-reproduce.md)
cover each result; [the dated plan](../docs/second-aligner-plan.md) separates declared
tests from follow-ups motivated by their outcomes.

## Per-clip estimates (milliseconds)

These are exploratory estimates. Original eligibility is unchanged; the new
investigation status is separate from the original QC flag.

| Clip | Originally flagged | MFA2 | MFA3 | Charsiu | Raw RMS | Three-model onset span |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(natural_rows)}
'''
    report_path=ROOT/'reports/second-aligner-investigation.md';report_path.write_text(report,encoding='utf-8')
    cards=[]
    for c in merged:
        key=c['key'];item=items[key];flagged=bool(c['original_flags']);category='onset' if c in targets else 'artifact' if c in five else 'other'
        # Membership above uses original dictionaries; derive category explicitly.
        category='onset' if c['category']=='RMS_endpoint_inside_first_HMM_phone' else 'artifact' if c['category']=='fragmented_RMS_subsonic_energy' else 'other'
        cards.append(f'''<article class="clip" data-category="{category}" id="{key}"><h2>{key}</h2>
<p class="tag">{'Original QC flag preserved' if flagged else 'Original duration pass preserved'} · {html.escape(c['investigation_status'].replace('_',' '))}</p>
<p lang="zh">{html.escape(item['setup'])}<strong>{html.escape(item['ending'])}</strong></p>
<audio controls preload="none" src="../{c['source_audio']}"></audio>
<img loading="lazy" src="figures/second-aligner/{key}.png" alt="Waveform, three aligners, energy measures and neural probabilities for {key}">
<p class="small">Model estimates are not human labels. Original audio only. Energy methods estimate quiet intervals; they can include weak speech.</p></article>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Second aligner investigation</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#f8fafc;color:#172033;margin:0}main{max-width:1120px;margin:auto;padding:24px}h1{font-size:30px}h2{font-size:20px}.clip{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:20px;margin:24px 0}img{width:100%;height:auto}audio{width:100%;max-width:650px}.tag,.small{color:#475569;font-size:14px}nav{position:sticky;top:0;background:#e2e8f0;padding:12px;border-radius:6px;z-index:2}select{font:inherit;padding:6px}strong{background:#fef3c7}.notice{border-left:4px solid #9333ea;padding:12px;background:#f3e8ff}a{color:#1d4ed8}</style>
<main><h1>Second aligner and source investigation</h1><p class="notice">All 34 original flags remain. Agreement among models supports a phonetic explanation, but does not prove natural boundary accuracy. These are local diagnostic recordings, not a completed humor study.</p>
<p><a href="second-aligner-investigation.md">Full findings and limitations</a> · 40 original recordings · no external media or server</p>
<nav><label for="filter">Show </label><select id="filter"><option value="all">All 40 clips</option><option value="onset">29 onset disagreements</option><option value="artifact">5 low-frequency artifacts</option><option value="other">6 original duration passes</option></select><span id="count" style="margin-left:12px">40 clips</span></nav>'''+''.join(cards)+'''
</main><script>document.getElementById('filter').addEventListener('change',function(){let n=0;document.querySelectorAll('.clip').forEach(x=>{x.hidden=this.value!=='all'&&x.dataset.category!==this.value;if(!x.hidden)n++});document.getElementById('count').textContent=n+' clips'});</script></html>'''
    page_path=ROOT/'reports/second-aligner-clips.html';page_path.write_text(page,encoding='utf-8')
    atomic_json(out/'summary.json',{'draft':args.draft,'original_flags':34,'cleared_flags':0,'onset_targets':29,'artifact_targets':5,
        'charsiu_original_fixtures':char_scores,'holdout':holdout,'natural_comparison':summary['comparison']})
    artifacts=[report_path,page_path,*figdir.glob('*.png'),out/'cases.json',out/'summary.json',out/'charsiu-fixture-validation.json']
    atomic_json(out/'provenance.json',{'code_sha256':digest(__file__),'draft':args.draft,'input_sha256':{r:digest(ROOT/'runs'/r/'summary.json') for r in (
        'phonetic-evidence-v1','signal-origin-v1','detrended-rms-v1','intact-stability-analysis-v1','onset-ablation-v1','mfa-ablation-analysis-v1','paired-measurement-sensitivity-v1','pcm-audit-v1','decoder-localization-v1','style-level-audit-v1','vocoder-stage-audit-v1','repeatability-audit-v1','quiet-selector-v2',holdout_run)},
        'artifacts_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in artifacts}})
    print('Generated',len(cases),'clip plots/page and investigation report; draft=',args.draft)

if __name__=='__main__':main()
