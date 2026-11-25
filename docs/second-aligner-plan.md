# Second phonetic aligner and subsonic-source investigation

Owner-approved exploration: roughly 2026-11-15 22:32:45 to 2026-11-16 00:32:45 UTC.
Prioritize the 29 onset disagreements and five fragmented RMS intervals. Branch
into related experiments when useful. Additional download ceiling: 15 GB. Keep
all assets and environments in this project. No human annotation or billed APIs.
The owner has reviewed and included all four methods: Charsiu, newer MFA,
Kokoro signal tracing, and targeted signal correction.

## Frozen comparisons

- Preserve original audio, all 40 estimates, and the 34 original flags.
- Run the independent Mandarin Charsiu frame classifier and its phone-based
  forced alignment, first on known-gap fixtures, then all originals. Validate
  phone/transcript coverage; distinguish its silence labels from CTC blanks.
  Record source/model revisions, local compatibility changes, missing outputs,
  and both start/end differences, not just duration agreement.
- The 29 original onset cases are the target, not a selection of whichever clips
  happen to agree. Report Charsiu-versus-MFA and Charsiu-versus-RMS at the same
  boundary. Model agreement is corroboration, not natural-speech ground truth.
- If useful and feasible, install a compatible isolated MFA 3 runtime and the
  newer Mandarin acoustic model. This is an additional model version, not an
  independent architecture. Do not load the v3 archive into the v2 runtime.

## Low-frequency investigation

1. Check whether the low-frequency waveform is present in native float audio,
   resampling, PCM conversion, or finite-window spectral estimation. Compare
   full-window low-pass reconstruction, means, spectral leakage controls,
   dominant-period stability and all 40 clips, not only five selected cases.
2. Replay original Kokoro inputs and require bit-identical PCM. Capture predicted
   F0/noise and vocoder source/output signals with forward hooks. Inspect whether
   voiced-source behavior at very low F0 or inverse-STFT DC content tracks the
   artifacts. Save diagnostics without modifying original audio.
3. Controlled interventions may change one source at a time: noise seed, low-F0
   excitation gating, or vocoder DC bin. Preserve durations and all other model
   inputs. These are counterfactual generator experiments, not replacements for
   the recorded study stimuli or proof of phonetic truth.
4. Test any proposed waveform correction on known-gap fixtures and non-silent
   speech regions. Report false removals, boundary shifts, missing intervals and
   limitations. Do not choose parameters by maximizing agreement with MFA.

Keep failed experiments. Do not relax the 30 ms criterion to manufacture passes.
Conclude with per-clip evidence and a specific decision on which findings are
resolved, corroborated, or still uncertain.

## Extensions declared at 22:50 UTC, before their outcomes

- Test AC RMS: subtract each 10 ms frame's own mean before computing RMS, retaining
  the original 5 ms hop, -35 dB threshold, 20 ms minimum and ambiguity handling.
  This is a fixed offset-invariance correction, not a fitted phonetic boundary.
  Evaluate original 40, existing 96, and controlled drift/noise/low-pitch fixtures.
- Trace matched baseline and altered-noise-seed signals; compare native float,
  16 kHz resampling and PCM16 copies. Separate fixed subsonic oscillation from
  nonstationary baseline wander and finite-window spectral peaks.
- Additional generator interventions: remove only the mean of the harmonic
  source; remove that source branch entirely; replace only decoder stock-voice
  style while preserving male timing/F0/noise. Record speech changes outside the
  disputed region; these are causal probes, not candidate study stimuli.
- Audit the hard-splice validation itself: it intentionally removed weak speech
  while retaining the original transcript. Check how much of the first ending
  phone was removed and whether this contributes to apparent aligner error.
  Do not optimize a natural-speech offset using those fixture outcomes.

## Extensions declared at 23:02 UTC, before their outcomes

- Test all 40 intact clips under no edit, 7/100/500 ms leading zero padding
  (plus 200 ms trailing zeros), and a 0.1 gain. Compare shifted timestamps and
  silence durations with each model's own unedited baseline. Group MFA CMVN by
  condition and original stock voice, keeping baseline grouping intact. These
  equivariance tests establish stability, not absolute phonetic accuracy.
- Inspect Charsiu's textless silence and unconstrained phone posteriors at all
  29 disputed onsets, separately from transcript-constrained DTW output.
- Evaluate linear-detrended frame RMS as a separately named diagnostic. Remove
  the best straight line in each 10 ms frame, retaining existing threshold,
  hop, minimum and missing rules. Challenge mean-only and linear variants with
  quiet low-pitch tones and consonant-like noise, not only inserted silence.
  Do not select a winner using agreement with MFA.

## Follow-up to stability and correction outcomes

- The fixed linear detrend removed ambiguity but also marked non-silent low-pitch
  probes quiet; keep it diagnostic, with no replacement of primary measurements.
- The 0.1-gain challenge changed some MFA intervals. Repeat the same intact
  stability corpus using the already verified zero-dither MFA 2 derivative.
  This diagnoses scale/dither sensitivity; earlier zero-dither fixture failures
  remain part of the decision and are not superseded by this test.
- Decompose the five targets and matched female controls into native inverse-STFT
  bin contributions. Verify their sum reproduces the waveform; quantify slow
  cancellation between bins. A 20-point FFT cannot isolate a sub-20-Hz band by
  discarding a bin, so do not interpret this probe as a correction method.

- Challenge Charsiu's apparently strong onset evidence with waveform ablation:
  for the frozen 29 cases, digitally zero the disputed onset region, with a
  20 ms left guard, and separately extend the mute 40 ms past RMS's onset.
  Use the same transcript and full context. Inspect only classifier frames
  whose 25 ms convolutional receptive fields are wholly inside exact zeros.
  This tests acoustic sensitivity versus contextual anticipation; the edits
  do not establish the original phonetic onset or become study stimuli.

- Add one small streaming speech-activity probe, Silero VAD v6.2.3, to contrast
  Charsiu's full-utterance context with a recurrent past-context classifier.
  Pin official model/code and use CPU ONNX in the existing VAD environment.
  Freeze raw posterior threshold 0.5 with 0.3/0.7 sensitivity, 32 ms chunks,
  no default speech padding or 100 ms silence merging, and the existing broad
  search brackets. Require all 96 known-gap checks before any consideration as
  a replacement; do not infer phonetic accuracy from a VAD score alone.

## Held-out recovery check after the streaming VAD probe

Silero at the primary 0.5 threshold recovered 93/96 edited gaps; the three
failures were 34--36 ms. The already declared 0.7 sensitivity recovered 96/96.
Before judging either, freeze a new 288-case set from scripts 002/004/005/007/
008/010 (not used in the original 96): both voices/endings, absolute hard splices
and incremental insertions, and the same six delays. Reuse the original fixed
cutting rules. Map the broad pre-existing search bracket through the known edit;
this supplies localization, not an endpoint estimate.

Also test a fixed four-phase streaming ensemble: lead with 0/8/16/24 ms of zeros,
map each 32 ms probability cell back to original time, average the four cells
on an 8 ms grid, and use threshold 0.5. This addresses chunk quantization without
fitting an offset. Report raw 0.3/0.5/0.7 and ensemble 0.5 on all old and new
fixtures, including failures. These remain edited speech, not a natural-boundary
accuracy claim; 30 ms insertions can pass the tolerance even when estimated zero.

- Apply the held-out set to original/new MFA, Charsiu and both waveform corrections
  as well. Keep the same tolerance and count missing outputs as failures.
- Quantify how estimator changes propagate into matched-pair contrasts, using all
  20 diagnostic script/voice pairs with missing counts retained. Report differences
  between estimators' contrasts, not a comedy effect or a main-study hypothesis test.
  Keep original flags attached; do not select the six previously agreeing clips.

## Final robustness and selector audit

- Test each HMM pipeline in four fresh original-audio batches, keeping weights,
  dither and grouping fixed. Compare repeated estimates with the saved original;
  report observed range, not a calibrated confidence interval. This follows the
  newly observed gain sensitivity and earlier 30 ms repeat differences.
- A globally shorter-than-20-ms quiet run cannot conceal a qualifying interval,
  even if it intersects a search edge. Audit qualification-before-censoring as
  selector v2 without changing historical metrics. Apply it to all 40 natural,
  96 original and 288 new fixtures, and retain any remaining censored intervals.

## Decoder localization after the completed held-out run

The source-zero and whole-decoder-style probes left the exact internal origin
open. For the same five targets and their matched female controls, separate
generator-only style from pre-generator decoder-feature style. Also swap only
the generator AdaIN style scale (gamma) or shift (beta), using the existing other
stock voice. Hold duration, F0, noise conditioning and random seed fixed. Require
native baseline replay identity and record speech changes outside the gap. These
are causal interventions, never proposed replacements or voice-cloning stimuli.

Before interpreting the size of these style effects, compare each saved
intervention's slow-component RMS with its outside-gap waveform RMS. Keep the
original gap plus 100 ms margins excluded and use the identical sample mask for
baseline/intervention. Report both raw and level-normalized changes, including
reverse swaps on matched female clips. This checks a possible overall gain
confound without another inference run or a tuned correction.

Final bounded localization: exchange style inside each of the eight existing
vocoder residual blocks separately (two source/noise blocks and six waveform
feature blocks), plus the two groups of three waveform blocks at the two
upsampling stages. Use all five targets and all five matched reverse-swap controls,
fixed duration/F0/noise and seed, for 100 interventions. Report every group,
level-normalized slow-component ratios and outside-gap waveform differences.
These non-additive, off-distribution probes cannot establish a uniquely faulty
layer; do not select or publish a repaired waveform from them.
