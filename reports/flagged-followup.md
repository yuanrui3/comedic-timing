# Investigation of the 34 flagged clips

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
| Original MFA configuration | 24/24 | 72/72 | Passed these edited fixtures only |
| MFA d0 | 10/24 | 71/72 | Sensitivity experiment |
| MFA perclip-d0 | 10/24 | 69/72 | Sensitivity experiment |
| MFA perclip-d1 | 24/24 | 72/72 | Sensitivity experiment |
| FSMN raw posterior ≥0.5 | 8/24 | 23/72 | Reject as precise pause detector |
| Fixed 20 Hz high-pass + RMS | 24/24 | 58/72 | Reject; ambiguous intervals count as failures |

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
| [zh001_zf_001_control](flagged-clips.html#zh001_zf_001_control) | 170 | 215 | /ɥ/; onset +32.5 ms | Retained |
| [zh001_zf_001_joke](flagged-clips.html#zh001_zf_001_joke) | 100 | 240 | /f/; onset +107.5 ms | Retained |
| [zh002_zf_001_control](flagged-clips.html#zh002_zf_001_control) | 160 | 210 | /n/; onset +42.5 ms | Retained |
| [zh002_zf_001_joke](flagged-clips.html#zh002_zf_001_joke) | 170 | 210 | /n/; onset +37.5 ms | Retained |
| [zh003_zf_001_control](flagged-clips.html#zh003_zf_001_control) | 250 | 325 | /tɕ/; onset +42.5 ms | Retained |
| [zh003_zf_001_joke](flagged-clips.html#zh003_zf_001_joke) | 230 | 295 | /tɕ/; onset +37.5 ms | Retained |
| [zh004_zf_001_control](flagged-clips.html#zh004_zf_001_control) | 250 | 305 | /ʂ/; onset +57.5 ms | Retained |
| [zh004_zf_001_joke](flagged-clips.html#zh004_zf_001_joke) | 260 | 310 | /ʂ/; onset +52.5 ms | Retained |
| [zh005_zf_001_control](flagged-clips.html#zh005_zf_001_control) | 110 | 180 | /x/; onset +82.5 ms | Retained |
| [zh006_zf_001_control](flagged-clips.html#zh006_zf_001_control) | 60 | 110 | /ʈʂ/; onset +47.5 ms | Retained |
| [zh006_zf_001_joke](flagged-clips.html#zh006_zf_001_joke) | 30 | 90 | /w/; onset +42.5 ms | Retained |
| [zh007_zf_001_control](flagged-clips.html#zh007_zf_001_control) | 240 | 290 | /n/; onset +42.5 ms | Retained |
| [zh007_zf_001_joke](flagged-clips.html#zh007_zf_001_joke) | 190 | 240 | /n/; onset +47.5 ms | Retained |
| [zh008_zf_001_control](flagged-clips.html#zh008_zf_001_control) | 270 | 320 | /n/; onset +42.5 ms | Retained |
| [zh008_zf_001_joke](flagged-clips.html#zh008_zf_001_joke) | 260 | 315 | /n/; onset +42.5 ms | Retained |
| [zh009_zf_001_control](flagged-clips.html#zh009_zf_001_control) | 100 | 150 | /ts/; onset +37.5 ms | Retained |
| [zh009_zf_001_joke](flagged-clips.html#zh009_zf_001_joke) | 100 | 150 | /ts/; onset +37.5 ms | Retained |
| [zh010_zf_001_joke](flagged-clips.html#zh010_zf_001_joke) | 180 | 220 | /p/; onset +37.5 ms | Retained |
| [zh001_zm_010_control](flagged-clips.html#zh001_zm_010_control) | 180 | ambiguous | peak 16.7 Hz | Retained |
| [zh002_zm_010_joke](flagged-clips.html#zh002_zm_010_joke) | 200 | ambiguous | peak 10.0 Hz | Retained |
| [zh003_zm_010_control](flagged-clips.html#zh003_zm_010_control) | 240 | 290 | /tɕ/; onset +52.5 ms | Retained |
| [zh003_zm_010_joke](flagged-clips.html#zh003_zm_010_joke) | 230 | 290 | /tɕ/; onset +52.5 ms | Retained |
| [zh005_zm_010_control](flagged-clips.html#zh005_zm_010_control) | 140 | 105 | /x/; onset +37.5 ms | Retained |
| [zh005_zm_010_joke](flagged-clips.html#zh005_zm_010_joke) | 170 | ambiguous | peak 17.6 Hz | Retained |
| [zh006_zm_010_control](flagged-clips.html#zh006_zm_010_control) | 50 | 105 | /ʈʂ/; onset +52.5 ms | Retained |
| [zh006_zm_010_joke](flagged-clips.html#zh006_zm_010_joke) | 40 | 95 | /w/; onset +42.5 ms | Retained |
| [zh007_zm_010_control](flagged-clips.html#zh007_zm_010_control) | 280 | 335 | /n/; onset +37.5 ms | Retained |
| [zh007_zm_010_joke](flagged-clips.html#zh007_zm_010_joke) | 290 | 335 | /n/; onset +32.5 ms | Retained |
| [zh008_zm_010_control](flagged-clips.html#zh008_zm_010_control) | 290 | 340 | /n/; onset +42.5 ms | Retained |
| [zh008_zm_010_joke](flagged-clips.html#zh008_zm_010_joke) | 280 | 335 | /n/; onset +47.5 ms | Retained |
| [zh009_zm_010_control](flagged-clips.html#zh009_zm_010_control) | 80 | 120 | /ts/; onset +52.5 ms | Retained |
| [zh009_zm_010_joke](flagged-clips.html#zh009_zm_010_joke) | 70 | 115 | /ts/; onset +57.5 ms | Retained |
| [zh010_zm_010_control](flagged-clips.html#zh010_zm_010_control) | 190 | ambiguous | peak 10.5 Hz | Retained |
| [zh010_zm_010_joke](flagged-clips.html#zh010_zm_010_joke) | 180 | ambiguous | peak 11.1 Hz | Retained |

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
