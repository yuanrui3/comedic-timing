# Resolving the pause disagreement

Written before inspecting MFA outputs, 2026-11-14 (America/Indianapolis).

The old 217.375 ms mean difference compares two different quantities: the interval between CTC character-emission supports, and a strict low-energy interval. CTC blanks can cover ongoing speech. The [TorchAudio alignment tutorial](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html#inconsistent-treatment-of-blank-token) explicitly explains this ambiguity. Subtracting 217 ms would not repair it.

Use a local phonetic GMM-HMM aligner with explicit silence phones as an independent acoustic check. Pin MFA 2.2.17 and the Mandarin MFA v2.0.0a acoustic model plus Mandarin (China) v2.0.0 dictionary: this compatible native Windows combination uses a local SQLite workspace, without needing a persistent database service. Character-level dictionary entries will preserve the exact setup/ending boundary. No model training or fine-tuning. Preserve the original audio and all earlier results.

Before inspecting new alignment results, freeze these diagnostics:

- Use all 40 existing diagnostic clips, retaining failures. Character sequence must match completely, with no out-of-vocabulary substitutions.
- Report raw CTC, HMM word/phone boundaries, and original RMS separately. Never relabel CTC emission gaps as silence.
- For RMS, report a fixed sensitivity grid of 25, 30, 35, and 40 dB below the 95th percentile frame RMS. Keep -35 dB as the legacy comparator; do not choose a threshold to minimize disagreement with the HMM.
- Fix the interval-selection problem: search the bracket between adjacent character anchors, require a unique qualifying interval or flag ambiguity, and distinguish no detected interval from missing/censored localization. Do not select a 5 ms fragment just because it happens to touch the midpoint.
- Define a second, explicitly named acoustic estimator using hysteresis: a low-energy core at -35 dB, expanded to -25 dB boundaries with no more than 20 ms of brief high-energy interruptions. This is a waveform-derived estimator, **not** independent forced alignment. Its agreement with another RMS estimator is not validation.
- Test incremental silence insertions again at 0, 30, 100, 300, 600, and 1000 ms. Include both voices, jokes and controls, and more than the first script.
- Add synthetic fixtures with exact known silence, low amplitude, low-level noise and brief internal transients. These test acoustic recovery, not semantic annotation or absolute natural-speech phonetic ground truth.
- Report bias, MAE, within-30-ms rate, individual outliers and endpoint differences, not only correlation. No claim of resolving absolute accuracy follows from incremental recovery alone.

Only adopt a primary pause definition that actually describes a silent interval. If independent methods still disagree, retain an uncertainty/QC flag rather than hiding it by tuning or averaging. Human annotation and billed APIs remain deferred.

## Follow-up after the fixed first comparison

MFA reduced the legacy mean difference from +217.375 ms to -29.375 ms, with 47.875 ms MAE. The proposed amplitude hysteresis worsened agreement and will not be promoted. Quiet consonants and low-frequency residue remain plausible causes of RMS disagreement.

Before running the next comparison, add a fixed spectral preprocessing check: resample to 16 kHz and apply `y[n] = x[n] - 0.97*x[n-1]`, then use the same -35 dB RMS rule and unmodified search brackets. This standard [Kaldi pre-emphasis coefficient](https://kaldi-asr.org/doc/feature-window_8h_source.html) raises the relative contribution of weak high-frequency consonants and suppresses low-frequency/DC residue. Keep the full 25/30/35/40 dB sensitivity grid; do not choose the threshold by closeness to MFA. This is exploratory method development on the existing sample, not held-out confirmation. Test the frozen method on new acoustic fixtures and edited-speech recovery cases.

The installed MFA implementation takes MFCC settings from the acoustic-model metadata: logs show dither=1 even when the YAML requested 0. Retain and report the actual model default instead of claiming the override worked. Speaker adaptation is disabled separately; logs show only first-pass alignment and no fMLLR stage. MFA's phrase “compiling training graphs” names a fixed-transcript decoding step, not acoustic-model training.
