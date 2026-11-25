# Follow-up on 34 flagged clips — 2026-11-15

Freeze the original 40 records and their 34 flags in corrected-pauses-v1.
This follow-up uses local inference and automated tests only. Do not substitute
machine agreement for human ground truth or select a threshold to maximize it.

Before new inference, the endpoint audit shows 29 duration disagreements and
5 ambiguous waveform intervals. In 34/35 unique intervals, HMM onset of the
ending precedes the -35 dB RMS onset; test whether weak phones, feature
normalization, or model behavior account for this pattern.

Experiments, in order:

1. Replay the pinned Kokoro model with saved phonemes, seed, and stock voices.
   Require bit-identical PCM before associating its predicted token durations
   with an existing clip. Save the entire duration trace and punctuation span.
   Those are generator latent allocations, not acoustic silence labels; do not
   apply the English pipeline's undocumented 75 ms timestamp offset to Chinese.
2. Repeat original MFA inference; then set feature dither=0 in a derivative
   acoustic archive's metadata (identical learned weight files). Compare the
   original two-speaker normalization with one utterance per normalization
   group. No adaptation or weight training. Save every outcome, including worse
   agreement. This tests repeatability and dependence on unrelated corpus items.
3. Analyze the five ambiguous intervals and quantify endpoint differences,
   neighboring phone overlap, and waveform energy/spectrum. Any automatic
   explanation remains a hypothesis unless independently established.
4. If a stable acoustic correction is supported, rerun the known-gap fixtures
   and noise cases before promoting it. A change in measurement definition must
   be explicit; sharing endpoints between methods cannot establish independence.

Additional local model checks may follow if these experiments identify a
specific uncertainty. MFA v3 requires a different runtime; avoid silently loading
it into the existing v2 runtime. Qwen alignment's coarse timestamps would not
establish 30 ms accuracy. No study expansion, billed calls, or human annotation.

## Extension before VAD/band-limited results

The five ambiguous clips have dominant 10–18 Hz spectral peaks in their HMM
intervals. Test a fixed 20 Hz, second-order Butterworth high-pass applied forward
and backward to the entire clip, retaining the original -35 dB rule. This is a
subsonic-artifact sensitivity test; it does not change the primary metric.
Validate against all 96 edited-speech cases, and synthetic subsonic contamination.

Download pinned Apache-2.0 funasr/fsmn-vad-onnx revision
f6e9fbb4cefa7397216c763f21307993f147f585. Use its 10 ms raw frame silence
posteriors (class 0), with a fixed 0.5 decision, minimum 20 ms duration, no
utterance-level padding or hangover. Report 0.3/0.7 sensitivity, without choosing
the best agreement. Compare independent intervals inside the same CTC bracket.
25 ms analysis windows use center timestamps; do not calibrate latency using
natural HMM/RMS endpoints. Test absolute and incremental fixtures before deciding
whether the neural detector can support any automated adjudication.
