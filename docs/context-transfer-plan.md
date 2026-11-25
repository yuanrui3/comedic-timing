# Context sensitivity and transfer exploration

Started 2026-11-15, 19:15 America/Indianapolis (2026-11-16 00:15 UTC).
Owner requested autonomous exploration until a critical crossroad. Existing
constraints persist: local inference, no billed calls, no human annotation,
project-local assets, at most 15 GB of additional downloads across this exploration
sequence (about 2.95 GB already retained). Preserve every original estimate/flag.

## First: future context with the boundary waveform unchanged

Use all 40 recordings, retaining the original 29/5/6 categories. Edit only after
the existing search bracket's right edge plus either 100 or 500 ms. Reverse the
remaining suffix, or replace it by zeros, without changing length. Save edits
after 16 kHz resampling and verify prefix sample identity. This protects the whole
measurement bracket and an additional guard, not only a single estimated onset.

Run the pinned Charsiu frame classifier with both ordinary per-utterance input
normalization and original-normalization held fixed. The primary endpoint is its
textless silence interval (upstream minimum four frames, predeclared two-frame
sensitivity). Do not force the unchanged transcript onto reversed/muted speech.
Report start/end/gap changes, missing status, and posterior changes in the intact
bracket, retaining failures. The existing 30 ms tolerance is a stability criterion,
not a natural-accuracy label. Verify unmodified baseline endpoints against saved
outputs. Changes to global normalization must be distinguished from later model
context effects; even frozen input normalization does not freeze all internal
feature statistics.

Run streaming Silero on the same edits as a causal control. At the unchanged
prefix, probabilities should remain identical through complete 32 ms chunks
before the edit; use raw 0.5 with the existing 0.3/0.7 sensitivities. Do not fit a
timestamp offset or select thresholds using the results.

## Then: investigate transfer where feasible

Survey a distinct openly licensed Mandarin TTS family that can produce a whole
script in one inference, without wrapper-added inter-sentence silence or a cloned
voice. Check public model metadata, native preprocessing, transcript coverage and
download size before committing. Use fixed existing scripts for a diagnostic
replication; no new confirmatory corpus or main-study inference. A model that
cannot meet these requirements is a documented failed candidate, not a reason to
relax them silently.

Selected candidate after official-source review: MeloTTS Chinese, MIT model/code,
with its Apache-2.0 multilingual BERT text encoder. Pin official revisions and
hashes; approximately 0.9 GB of weights. Use only the named stock ZH speaker and
the existing 10 pairs, three fixed paired seeds 20261114/15/16, speed 1, native
noise settings. Native synthesis must be one full-script infer call with no
wrapper concatenation, inserted silence, or reference-voice input. The ordinary
wrapper's segmentation and 50 ms added chunk padding are excluded. Confirm the
Chinese-only frontend follows the pinned mixed-language Chinese branch exactly;
no non-Chinese input or unknown substitutions. Model internals and checkpoints
stay unchanged. Assess transcript fidelity with existing local ASR, explicitly
as another model's QC signal. Compare aligners and waveform checks across all
60 generated clips, including repeat variation and failures; no humor inference.

## Critical crossroads

### Transcript QC extension (declared before Whisper inference)

The existing CTC recognizer mismatches the target transcript more often on Melo
(23.51% character error) than on Kokoro (12.25%). This is not proof of synthesis
error. Cross-check every original Kokoro clip and all 60 Melo clips with the
official MIT Whisper large-v3-turbo checkpoint, revision
41f01f3fe87f28c78e2fbf8b568835947dd65ed9 (about 1.62 GB). Inference is entirely
local, one whole recording per call, Mandarin transcription, greedy decoding,
no target-text prompt, no timestamps or boundary claims. Keep all transcripts.
Report raw Han character and toned-pinyin edit rates for both recognizers;
traditional/simplified spelling and homophones can inflate character differences.
No clip exclusion or corpus correction follows automatically from this QC.

Stop for user direction before changing the scientific target to waveform quiet
instead of phonetic silence, accepting measurement validity without independent
evidence, lifting human/billing constraints, exceeding the remaining download
allowance, or starting the main study. Routine model/runtime choices and local
diagnostics remain authorized. If evidence shows that the next useful advance
requires one of those decisions, summarize concrete results and the alternatives.
