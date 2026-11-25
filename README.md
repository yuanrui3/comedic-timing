# Comedic timing — local measurement exploration

This is an exploratory research pipeline, not a completed benchmark. The latest research is [the mathematical review](reports/mathematical-research-2026-11-18.md); the latest speech experiment is [the context and model-transfer decision brief](reports/context-transfer-investigation.md), with [the 100-clip transcript and local-audio viewer](reports/context-transfer-clips.html). The [second-aligner and artifact-source investigation](reports/second-aligner-investigation.md), [40 original boundary plots](reports/second-aligner-clips.html), [first investigation of the 34 flags](reports/flagged-followup.md), [pause correction](reports/pause-resolution.md), and [initial exploration](reports/exploration.md) remain available. Read [PLAN.md](PLAN.md) and [LOG.md](LOG.md). Owner-performed human practice is now prepared; billed APIs remain deferred. All model inference is local; audio, weights, caches and portable runtimes are ignored by Git.

## Public repository contents

This repository publishes source, tests, experiment plans, public-domain-derived
stimulus text, research reports/figures, and the synthetic mathematical audit.
Audio, model weights, local runtimes, raw run records, participant responses and
the private practice mapping stay local. Links to `runs/` outside the synthetic
audit, audio playback, and the built practice page therefore require the original
local artifacts or regeneration. GitHub displays the HTML source; local viewers
are opened in a browser after their required media are available.

See [publication scope and third-party materials](docs/publication.md) before
reproducing an experiment. No project-wide reuse license has been selected.
Historical reports contain detector results; participants should finish and lock
their blind annotations before consulting those comparisons.

## Mathematical research update

The [18 November research review and mathematical analysis](reports/mathematical-research-2026-11-18.md)
covers recent boundary-aware alignment, interval-valued outcomes and conformal
inference for grouped data. It derives limits on silent-closure identification,
shared model bias, nuisance removal and calibration with repeated scripts, then
shows how boundary ranges can bound a joke/control contrast without forcing
every clip to one timestamp. The [synthetic numerical audit](runs/mathematical-audit-v1/summary.json)
and [figure](runs/mathematical-audit-v1/mathematical-audit.png) are reproducible on CPU.
These are mathematical examples, not new natural-speech validation. No practice
answers or clip-specific detector timestamps are disclosed in the review; the
active practice page, original metrics and historical flags remain unchanged.

## Human boundary practice

The owner requested a human evaluation and reports fluent/native spoken Mandarin.
The [six-clip practice pack](evaluation/human-practice-v1/START.html) and
[instructions](evaluation/human-practice-v1/README.md) support an initial review
with predicted timestamps hidden. The [protocol](docs/human-boundary-evaluation-plan.md)
preserves uncertainty and includes later hidden repeats. Owner-performed
annotation is now authorized; no human responses have been collected yet. Billed
APIs remain deferred. Historical estimates and flags remain unchanged.

## Context and transfer continuation

Changing distant future context while preserving boundary samples moves Charsiu's
measurable onsets at most 10 ms; one short gap crosses its minimum-duration rule.
Streaming Silero remains unchanged. This does not erase earlier local-muting
anticipation. A distinct stock-voice family, MeloTTS, supplies 60 complete-script
clips with no wrapper-added pauses. MFA versions agree on every ending onset
within 30 ms, yet raw-RMS QC flags 51/60. Mean/linear corrections barely improve
the disagreement, and the gap energy differs from Kokoro's slow baseline drift.

Unprompted local Whisper checks all 40 Kokoro plus 60 Melo recordings. Character
mismatch is 1.42% versus 11.33%; no character mismatch occurs in the immediate
two-character neighborhood on each side of the original Kokoro boundaries.
These are automated QC findings, not independent phonetic timing labels.
All 34 original flags and all original metrics remain unchanged. The scientific
crossroad is whether to proceed with explicitly exploratory model-estimated
timing or require independent boundary evidence before timing claims. Gate 1
remains unaccepted; no main-study run has started.

[Reproduction guide](docs/context-transfer-reproduce.md),
[integrity audit](runs/context-transfer-report-v1/verification.json), and
[cumulative download inventory](runs/context-transfer-report-v1/downloads.json)
record this continuation. It uses existing local runtimes plus pinned Melo,
multilingual BERT and Whisper weights, within the approved 15 GB allowance.

## Second aligner and decoder investigation

Charsiu's Mandarin frame classifier and the newer MFA Mandarin model each agree
with the saved original onset within 30 ms in all 29 disputed clips. Charsiu also
classifies their low-energy onset regions as speech without a transcript. However,
exact-zero ablations expose contextual anticipation in all three phonetic models.
Their agreement is corroboration, not an absolute natural-speech reference.

The five fragmented RMS cases contain native Kokoro baseline wander. Baseline
replay is PCM-identical; conversion does not create it. Removing harmonic
excitation leaves 94–107% of the slow component, while changing only vocoder
stock-voice style leaves 19–51%. Fixed mean/linear frame corrections help interval
coverage but can include quiet speech and do not establish phonetic boundaries.
After controlling for surrounding level, a style swap in the last stage's
widest-kernel block alone leaves 33–39% across all five; reverse swaps amplify
it in all five matched clips. These interventions also change speech substantially.
All 34 original flags and original estimates remain unchanged.

The expanded 288-case recovery set uses six scripts absent from the original 96.
Original MFA passes 273/288, new MFA 222/288, and Charsiu 215/288; the exact edit
types and missing counts are separated in the report. Hard splices remove weak
phone portions and are not natural phonetic labels. A small streaming Silero
probe and fixed chunk-phase ensemble also fail some held-out cases. No method
was promoted by selecting a favorable threshold or fixture subset.

[Reproduction commands and environments](docs/second-aligner-reproduce.md) cover
all new results. The [final integrity audit](runs/second-aligner-investigation-v1/verification.json)
verifies fixture construction, saved hashes, page links and CPU/GPU endpoint parity;
23 software tests passed. New assets remain in the project: pinned Charsiu code/model,
isolated `.runtime/mfa3/`, and a 2.3 MB Silero model using the existing VAD runtime.
`requirements-exploration-lock.txt` records the expanded main environment;
`requirements-mfa3-lock.txt` and `runs/model-locks/mfa3-runtime.json` record the
hybrid native/Python MFA runtime. The retained download/cache inventory is about
2.95 GB against the approved 15 GB allowance; it is not a traffic meter. No system
installation, driver change, model training, human annotation or billed API call.

## Earlier follow-up on the 34 flags

All 29 duration disagreements concentrate at the initial ending phone: the RMS endpoint falls inside MFA's first-phone interval. The other five clips have multiple RMS candidates and dominant 10–18 Hz energy in the modeled gap. These are machine explanations, not independent ground truth; **all 34 flags remain**. No threshold or model was selected just because it agreed better.

Kokoro replay was PCM-identical for all 40 clips, but its latent punctuation durations are not acoustic pauses. A pinned local FSMN neural VAD failed known-gap validation (31/96 within 30 ms), as did fixed 20 Hz filtering (82/96). Zero-dither MFA repeated deterministically but worsened absolute fixture recovery (10/24); retaining original dither with per-clip normalization preserved 96/96 recovery. Original metric outputs and eligibility were preserved.

The new neural probe uses `.venv-vad/`, `funasr-onnx==0.4.11`, CPU ONNX Runtime, and a pinned Apache-2.0 model. `requirements-vad-lock.txt` records dependencies; `runs/model-locks/fsmn_vad.json` records revision and file hashes. Its NumPy requirement conflicts with the main environment, which was left intact. The archive `mandarin_mfa_dither0.zip` is an explicitly labeled CC-BY derivative that changes only metadata; all learned weight bytes were verified identical. It is an experiment, not a replacement model.

Reproduce from existing local assets (choose fresh `--run`/`--out` names if code changes):

```powershell
& '.venv\Scripts\python.exe' -X utf8 -m timing.kokoro_trace
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --run mfa-repeat-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --run mfa-dither0-v1 --dither-zero
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --run mfa-perclip-d0-v1 --dither-zero --per-clip
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --run mfa-perclip-d0-repeat-v1 --dither-zero --per-clip
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --baseline mfa-fixtures-v1 --run mfa-fixtures-d0-v1 --dither-zero
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --baseline mfa-fixtures-v1 --run mfa-fixtures-perclip-d0-v1 --dither-zero --per-clip
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_variants --baseline mfa-fixtures-v1 --run mfa-fixtures-perclip-d1-v1 --per-clip
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_evaluate --mfa mfa-fixtures-d0-v1 --out pause-validation-d0-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_evaluate --mfa mfa-fixtures-perclip-d0-v1 --out pause-validation-perclip-d0-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_evaluate --mfa mfa-fixtures-perclip-d1-v1 --out pause-validation-perclip-d1-v1
& '.venv-vad\Scripts\python.exe' -X utf8 -m timing.fsmn_probe
& '.venv\Scripts\python.exe' -X utf8 -m timing.bandlimited
& '.venv\Scripts\python.exe' -X utf8 -m timing.subsonic_stress
& '.venv\Scripts\python.exe' -X utf8 -m timing.flagged_report
& '.venv\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
```

The fixed experiment plan is `docs/flagged-followup-plan.md`; per-clip findings and report provenance are in `runs/flagged-followup-v1/`. The HTML page loads only local media and requires no server. Rebuild `.venv-vad` separately with the existing portable Python and project-local uv cache; install `requirements-vad-lock.txt` into that environment only. Download the four model files named in its model lock at the recorded Hugging Face commit and verify their SHA256s before inference.

## Corrected pause measurements

CTC character-emission gaps are **not silence durations**. The corrected entry point is `timing.measure_pauses`, which exports explicit HMM silence-phone duration and independent waveform QC. Old `ctc_gap_ms` fields remain historical diagnostics; new CTC output explicitly labels them invalid as primary pauses. On 40 original clips, replacing CTC with phonetic silence reduced mean disagreement against the unchanged legacy RMS from +217.4 to -29.4 ms (MAE 47.9 ms remains). Only 6/40 pass the stricter ambiguity-aware automatic QC; all other records remain flagged, not silently excluded. Gate 1 is not accepted.

The installed phonetic aligner is separate from the GPU environment: `.runtime/mfa/`, MFA 2.2.17, Kaldi 5.5.1068, Python 3.10.22. It runs CPU inference with a CC-BY-4.0 Mandarin acoustic model/dictionary. Exact source releases are in `config/mfa-releases.json`, model hashes in `runs/model-locks/mfa_zh.json`, key executable hashes in `mfa-runtime.json`, and Python versions in `requirements-mfa-lock.txt`. The Conda native package includes a local CUDA runtime dependency; no system driver was installed or changed. `timing.mfa_run` sets PATH only for its subprocess, uses project-local SQLite files, and disables speaker adaptation and fine-tuning. Its runtime log records actual feature settings, including the model's dither=1 default.

Rebuild the current analysis from the existing audio without rerunning synthesis:

```powershell
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_run --run mfa-diagnostic-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_audit --out pause-audit-v3
& '.venv\Scripts\python.exe' -X utf8 -m timing.measure_pauses
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_fixtures
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_run --run mfa-fixtures-v1 --source pause-fixtures-v1 --limit 96
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_evaluate --out pause-validation-v2
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_noise
& '.venv\Scripts\python.exe' -X utf8 -m timing.pause_report
```

The saved validation uses the existing `alignment-v3` artifacts. To independently rerun the GPU aligner with current code, use `timing.align --run pause-fixtures-v1 --analysis alignment-v4`, followed by `timing.pause_evaluate --analysis alignment-v4 --out pause-validation-rerun`; this creates new artifacts. Completed MFA and corrected-metric runs verify checksums and skip work. If their relevant code/configuration changes, choose a new run name. No server is needed for `reports/pause-resolution-audio.html`.

The 96 edited-speech cases comprise 24 absolute hard-splices and 72 incremental insertions. HMM silence passes all within 30 ms, while CTC fails all absolute cases despite passing incremental recovery. This is why a baseline-subtracted insertion test alone was insufficient. Hard-splices are not natural phonetic ground truth. The report retains failed hysteresis/pre-emphasis agreement experiments and controlled-background stress checks rather than selecting a threshold that makes methods agree.

Setup history: the native dependencies were obtained using project-local micromamba 2.10.0, followed by Python wheel installation. Windows path-length failures affected optional Conda test files, so the final environment is a hybrid rather than a clean successful Conda transaction. The native packages used `python=3.10`, `kaldi=5.5.1068`, `openfst=1.8.2`; compatibility pins include MFA 2.2.17, NumPy 1.26.4, statsmodels 0.14.1, SQLAlchemy 2.0.36, praatio 5.1.1, click 8.1.8, rich-click 1.7.4, joblib 1.3.2, pgvector 0.1.8 and setuptools 78.1.1. The saved version/hash snapshots describe the tested environment; a fresh-machine installation has not been tested.

## Environment

Windows, RTX 4090 Laptop GPU (16 GB). The 2026-11-15 follow-up observes NVIDIA driver 610.74 (the earlier setup recorded 581.95); this task did not install or change a driver. The verified project environment uses portable CPython 3.12.16 and PyTorch 2.8.0 with CUDA 12.8, with a fresh GPU matrix-operation check in `runs/environment-followup.json`. `.runtime/uv/uv.exe`, `.venv/`, and caches stay in this folder. Dependency versions are frozen in `requirements-lock.txt`; this is a version snapshot, not a cross-platform hash lock. Model commits and SHA256s are in `runs/model-locks/`.

From PowerShell in the project directory, set local caches before any package operation:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.runtime\python'
$env:TMP = Join-Path (Get-Location) '.cache\tmp'
$env:TEMP = $env:TMP
```

To recreate the installed packages after creating a project virtual environment:

```powershell
& '.\.runtime\uv\uv.exe' pip install --python '.venv\Scripts\python.exe' --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r requirements-lock.txt
```

Downloads are public model files, not inference endpoints. Pin definitions live in `config/models.json`. The original `align_zh` model is a failed vocabulary-coverage probe and is not required for current execution.

```powershell
& '.venv\Scripts\python.exe' -X utf8 -m timing.fetch kokoro_zh align_zh_full surprisal_small surprisal_large
$env:HF_HUB_OFFLINE = '1'
```

## Reproduce each artifact

The commands below run sequentially. Do not launch GPU jobs concurrently on this shared laptop. Source/configuration hashes are immutable per run: if code changes, choose new `--run` and `--analysis` names rather than mixing outputs. Completed artifacts with matching checksums are skipped; interrupted or failed items are retried. Keep the virtual environment frozen during a run. `source_hash` covers all Python modules, so even a report-code change requires a new run ID on resumption.

```powershell
& '.venv\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
& '.venv\Scripts\python.exe' -X utf8 -m timing.corpus
& '.venv\Scripts\python.exe' -X utf8 -m timing.surprisal --run surprisal-v2
& '.venv\Scripts\python.exe' -X utf8 -m timing.synthesize --run kokoro-diagnostic-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.align --run kokoro-diagnostic-v1 --analysis alignment-v2
& '.venv\Scripts\python.exe' -X utf8 -m timing.validate --run kokoro-diagnostic-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.report --run kokoro-diagnostic-v1 --surprisal surprisal-v2
```

`timing.report` computes every aggregate in the exploration report from saved measurements; it does not rerun models. `timing.validate --fixtures-only` reproduces just the synthetic arithmetic checks. Alignment and validation retain failures explicitly. The static `reports/diagnostic-audio.html` opens directly as a local file; no server is required. Audio links depend on this project folder and will not work in a text-only Git checkout.

## Scope and definitions

The ten Mandarin records are newly worded adaptations of public-domain *笑林廣記* entries, not modern performance transcripts. Each record contains source, provenance, setup, joke/control endings and reduced punctuation variants. Han-character count and punctuation positions match automatically. These originals already contain only commas/full stops, so reduced copies are identical and provide no punctuation-variant contrast. Both text copies are LM-scored, but only original text is synthesized in this diagnostic. LM scores condition ending tokens on setup tokens and preserve token offsets and bits; setup and ending are tokenized separately to prevent tokens crossing the marked boundary. They are line-level manipulation checks, not yet all-boundary/word-level coupling analysis.

Two stock Kokoro voices (`zf_001`, `zm_010`) share one model. They do not count as two model families toward the eventual four-to-six-model study. Complete scripts are phonemized and passed to the acoustic model in a single call, with no concatenation or inserted pauses. CTC emission-frame gaps and waveform-energy silence are different quantities; their disagreement must be investigated before drawing humor conclusions. Silence insertion validates changes in measured gaps, not absolute endpoints. Human review, English, a larger retained corpus, extra model families, repeated takes, style conditions and the main analysis remain outstanding.

The implementation includes an exhaustive small-path CTC oracle, zero-versus-missing handling, terminal-padding exclusion, undefined standardized effects for constant differences, and manifest mismatch checks. Scientific validity is not established by passing these software tests.
