# Context and transfer reproduction

Use project-local `.venv` and `.venv-vad`; no hosted model API is used. Models are
pinned in `runs/model-locks/{melo_zh,melo-code,melo_bert,whisper_turbo}.json` alongside
the existing Charsiu/MFA/Silero locks. Model files are not redistributed.

Completed source is snapshotted in `runs/context-transfer-report-v1/source`.
Every inference manifest records code and input hashes. To rerun inference after
source/declaration changes, use a separate copy and restore that run's exact
source and declared plan; hashes intentionally reject incompatible resumption.
Future-context preparation used plan SHA256
`3a3a5aded276f36971cf61bc54a208af4c4eb4c10f29a518ca7b815cc92da25b`, archived under
`runs/plan-snapshots/`. Whisper's declaration is copied into its run directory.
The final plan contains the later Melo candidate and ASR extension. Source models
are validated before inference. Preserve original recordings and metrics.

```powershell
# Network downloads only when rebuilding missing official assets; already complete.
& '.venv\Scripts\python.exe' -X utf8 -m timing.fetch_melo metadata
& '.venv\Scripts\python.exe' -X utf8 -m timing.fetch_melo weights
& '.venv\Scripts\python.exe' -X utf8 -m timing.fetch_melo_bert
& '.venv\Scripts\python.exe' -X utf8 -m timing.fetch_whisper

# Future-context experiment (use its exact archived declaration for prepare).
& '.venv\Scripts\python.exe' -X utf8 -m timing.future_context prepare
& '.venv\Scripts\python.exe' -X utf8 -m timing.future_context charsiu
& '.venv-vad\Scripts\python.exe' -X utf8 -m timing.future_context silero
& '.venv\Scripts\python.exe' -X utf8 -m timing.future_context evaluate

# Complete-script second-family diagnostic.
& '.venv\Scripts\python.exe' -X utf8 -m timing.melo_local prepare
& '.venv\Scripts\python.exe' -X utf8 -m timing.melo_local synthesize
& '.venv\Scripts\python.exe' -X utf8 -m timing.align --run melo-diagnostic-v1 --analysis alignment-v2
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa_run --source melo-diagnostic-v1 --run mfa-melo-v1 --limit 60
& '.venv\Scripts\python.exe' -X utf8 -m timing.mfa3_run --source melo-diagnostic-v1 --run mfa3-melo-v1
& '.venv\Scripts\python.exe' -X utf8 -m timing.transfer_charsiu
& '.venv\Scripts\python.exe' -X utf8 -m timing.transfer_analysis

# Unprompted independent ASR and derived checks.
& '.venv\Scripts\python.exe' -X utf8 -m timing.whisper_qc
& '.venv\Scripts\python.exe' -X utf8 -m timing.context_transfer_qc
& '.venv\Scripts\python.exe' -X utf8 -m timing.context_transfer_report
& '.venv\Scripts\python.exe' -X utf8 -m timing.context_transfer_verify
```

All 60 Melo clips and all 100 ASR transcripts must remain in the accounting.
Reports do not use an ASR score to exclude audio. No humorous-timing effect,
population inference, natural-boundary accuracy interval, or Gate 1 acceptance
is computed. Post-hoc high-pass and toneless-pinyin descriptions remain labeled
as such. `context-transfer-qc-v1/source.py` preserves the earlier aggregate lookup
bug; v2 corrects the five-case category and keeps prior ASR numbers unchanged.
