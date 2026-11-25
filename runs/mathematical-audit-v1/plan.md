# Mathematical research audit — 2026-11-18

Scope: a literature review through 2026-11-18 and mathematical analysis of the
remaining boundary-identification and nuisance-signal problems. This document
is written before the new numerical results. This work does not change the
active human evaluation, historical flags, measurements, or Gate 1.

All new numerical examples use generated arrays on the CPU. No original audio,
private practice mapping, human responses, or model predictions are inputs.
No model downloads, model training, paid inference, or new annotation is needed.

## Fixed demonstrations

1. Ten independent Gaussian model estimates of a fixed synthetic boundary;
   within-ensemble standard deviation 10 ms, common bias 0–50 ms in 5 ms steps.
   Compare the second/ninth order-statistic interval's coverage of the model
   median with its coverage of the actual synthetic boundary. Use 120,000
   replications, seed 20261118, and compare Monte Carlo against binomial formulas.
2. Ten independent uniform script-level calibration scores, each copied four
   times. Compare naive 40-row split-conformal calibration with the rank
   calculation at the script level. Show the finite-quantile requirement for
   ordinary nonrandomized 95% split conformal with 10, 19, 20, and 40 groups.
   This is a counterexample, not an estimated correlation in our recordings.
3. Mean/linear nuisance projection on 10 ms frames at 16 kHz. Sweep sinusoids
   from 1 to 1,000 Hz and 64 equally spaced phases. Save pooled energy retention
   and phase ranges, with 10, 20, 40, 60, 100, 200, and 500 Hz checkpoints.
   These are pure tones, not simulated Mandarin phones or natural pause labels.
4. Verify nuisance-profile least squares against direct regression and interval
   arithmetic against all rectangle corners. Check the 60 ms indistinguishable
   boundary example's 30 ms minimax point-error bound.

Save source, parameters, exact formulas, numerical checks, summary JSON and
standalone PNG/SVG figures. Distinguish proofs, synthetic illustrations,
previous project observations, literature results and proposed next steps.

## Integrity

Hash the original corrected metrics and the blind practice page before and after.
Do not expose clip-specific predicted timestamps in this report or conversation.
Do not treat model spread, repeatability, synthetic recovery, or human plausible
ranges as calibrated natural-boundary accuracy without their extra assumptions.
