# Comedic timing study — draft protocol

Prepared 2026-11-14 before any speech-model measurement. Status: exploratory development; not a completed preregistration or main run. The owner subsequently requested autonomous exploration and deferral of human annotation and billed API calls.

## Questions and predictions

1. In plain delivery, the mean paired joke-minus-control pre-punchline gap is near zero. Report millisecond estimates and confidence intervals; failure to reject zero does not demonstrate equivalence. A practical equivalence margin must be justified and frozen before a confirmatory run.
2. For models with genuine instruction control, performed delivery increases the joke-minus-control gap relative to plain delivery. Instructions are identical for jokes and controls, with no request for a specific pause duration or location.
3. Pause/next-line token-surprisal association is expected to be weak. The original proposed comparison to a human correlation is not identifiable: Ma et al. report embedding-distance quintile contrasts instead. Report correlations as an extension and a separate, approximate embedding-distance/quintile analysis. An open embedding model cannot reproduce a different proprietary embedding model's values exactly.

## Units, controls and text

- Mandarin first, English second. Target 60 retained pairs per language for the eventual main run; build an excess candidate pool, fix its order before audio generation, and disclose all exclusions. Exploratory material is not the final corpus.
- Store setup and ending as explicit spans with line boundaries and provenance. Public-domain sources or explicit open licences only. Modern retellings, translations and generated controls must be identified as adaptations; do not imply that modern wording appeared in the historical source.
- Human review of 20 proposed splits is deferred at the owner's request. Automated structural validation is not semantic annotation or proof of humor.
- A control shares the exact setup, ending length (Han characters for Chinese; lexical words for English), and punctuation sequence/positions in lexical units. Record English syllable counts where available; matching word counts alone does not match acoustic duration or syntax.
- Store original and reduced-punctuation variants. Convert question/exclamation marks and ellipses to full stops, other boundary punctuation to commas; retain the same transformation for both pair members. Treat the variant as a separate paired condition.
- Score endings conditioned on setup using two local causal LMs per language. Record bits per token, summed bits per line, and length-normalized summaries. Keep pairs with greater ending surprisal in both LMs; report sensitivity to either-model retention. Never filter using pause outcomes.
- Canonical line boundaries are metadata, not extra newlines or spoken setup/punchline labels. Submit the entire script in one model inference. Sentence-by-sentence synthesis or wrapper-added silence cannot test spontaneous boundary timing and is excluded from that endpoint.

## Measurement definitions

Revision after the 2026-11-14 diagnostic: CTC emission-support gaps are rejected as primary pause estimates. Use explicit silence-phone duration from the pinned local phonetic aligner, retaining transcript checks and independent waveform QC. Details and failed alternatives are recorded in `docs/pause-resolution-plan.md`. Residual natural-speech disagreement remains; the edited-speech validation does not clear Gate 1 by itself. No threshold is chosen to minimize detector disagreement, and no constant offset is subtracted. All 40 estimates remain saved with QC flags; do not use a convenient agreeing subset as the study corpus.

- Preserve native waveform and sample rate. Resample a separate copy only as required for alignment, recording its configuration and hashes.
- Primary alignment gap: start of first spoken punchline unit minus end of last spoken setup unit, in milliseconds. Keep negative overlap as a QC flag; never silently clamp or drop it.
- Independently estimate low-energy silence near the boundary using frame RMS. Report threshold, frame/hop length, search interval, nearest-gap selection and failures. Alignment gaps are not intrinsically silence: breath, laughter, quiet consonants and alignment uncertainty can produce disagreement.
- Zero-length gaps are valid observations; missing alignment, unsupported text and truncated/incorrect speech are missing values with reasons, never zero.
- Relative pause: subtract the mean control-boundary gap for the same model, stock voice, language, delivery condition, punctuation variant and boundary mark. Use leave-one-pair-out estimates to avoid self-inclusion; require at least five other pairs or mark unavailable.
- Primary contrast: average takes within script/condition, then subtract the matched control. Pair seeds when supported, retaining the take identifier.
- Coupling: within each voice/language/condition/variant, correlate each eligible boundary's gap with the next line's surprisal. Bootstrap whole scripts, retaining all their boundaries and paired variants. Do not pool distinct languages, tokenizers or voices.
- Secondary post-punchline silence is only identifiable between spoken segments. Terminal file padding is recorded separately and not interpreted as rhetorical pausing. Speaking rate excludes measured silence; report characters/second for Mandarin and words/second for English, without comparing their scales directly.

## Validation and statistics

Context/transfer continuation, 2026-11-15 local: distant-context changes preserve
Charsiu endpoints within 10 ms when measurable, with one minimum-duration status
crossing; the streaming control is unchanged. Sixty whole-script Melo recordings
show another waveform/phonetic discrepancy pattern and stronger automated
transcript concerns than Kokoro. Independent local Whisper adds transcript QC,
not boundary labels. All original estimates and 34 flags remain intact. See
`reports/context-transfer-investigation.md` for the critical choice between
exploratory model-estimated timing and holding claims for independent boundary
evidence. No primary endpoint, eligibility policy or gate is changed here.

Second-aligner exploration, 2026-11-15: both Charsiu and newer MFA corroborate all
29 saved onsets within 30 ms, but ablation shows contextual anticipation inside
known zeros. Native replay/source/style interventions localize the five slow
components to Kokoro output and vocoder conditioning. Neither model agreement
nor a correction that makes intervals unique is an independent truth criterion.
The 288-case held-out expansion limits the earlier 96/96 MFA recovery result;
report both sets and both edit types, with all missing values retained. A new
qualification-before-censoring selector fixes subminimum edge blips without
changing any of the 40 natural estimates. It does not clear any original flag.
See `docs/second-aligner-plan.md`, `docs/second-aligner-reproduce.md` and
`reports/second-aligner-investigation.md`. Gate 1, the Tier 1 pilot and main run
remain unaccepted/unstarted. No human or billed work is required for these saved
diagnostics, and no new human annotation is requested during this exploration.

2026-11-15 follow-up: preserve the 34 original flags. All 29 duration discrepancies localize to RMS endpoints inside the first HMM ending phone; five fragmented intervals contain subsonic energy. These model-based explanations do not validate natural boundary accuracy. The fixed FSMN posterior and 20 Hz filtering alternatives failed edited-speech checks. Do not disable MFA feature dither for determinism: zero-dither variants failed absolute recovery, while per-clip normalization with original dither retained 96/96 passes. The earlier validation is configuration-specific. See `docs/flagged-followup-plan.md` and `reports/flagged-followup.md`.

- First test WAV I/O, injected silences (0, 30, 100, 300, 600, 1000 ms), frame-edge cases, low amplitude and noise. Synthetic tone fixtures validate arithmetic only, not a speech aligner.
- Then insert known silence at independently specified boundaries in real stock-voice clips, rerun alignment from audio and transcript, and require recovery within 30 ms. Do not shift reference timestamps and call that validation. Also report absolute boundary accuracy when an independent reference exists; this human check is currently deferred.
- Report detector agreement using signed bias, MAE, fraction within 30 ms and Bland–Altman limits; correlation alone is insufficient. Automated disagreement is a QC signal, not evidence that either detector is correct.
- Bootstrap 10,000 resamples over scripts with seed 20261114. Report paired mean milliseconds and paired standardized effect dz = mean difference / SD of script differences; constant differences make dz undefined, not infinite evidence. Retain incomplete-pair counts and exclusion reasons.
- Average repeated takes before primary paired inference. Report take variation separately. Main-model comparisons remain stratified; control multiplicity before making claims across many models.
- An exploratory 10-script sample cannot establish a null, equivalence or the behavior of all voice models.

## Execution and review gates

All caches, downloaded runtimes and weights stay under this project. No administrators, drivers, services, paid endpoints, real-person voice cloning, performance-media downloads or pushes. GPU models load sequentially after a free-memory check. Every output uses an immutable configuration/input hash and an atomic manifest; interruption must be safe to resume.

Gate 1 remains a target: 10 scripts and 2 voices, actual speech silence recovery, detector agreement and annotated waveform plots. Human semantic/boundary review is deferred, so automated artifacts remain exploratory. Gate 2 requires a 15-pair pilot across the selected Tier 1 roster, immutable model revisions, measured GPU/runtime projection and approval before the main run. No billed service will be proposed or used during the current exploration. Gate 3 is a report and local static audio comparison page after an approved main run, not a claim supported by diagnostic fixtures.
