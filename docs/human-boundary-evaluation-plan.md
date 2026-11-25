# Owner-performed boundary evaluation — proposed protocol v1

Prepared 2026-11-15. This designs a new human review requested by the owner; no
human ratings have been collected. Existing original estimates, flags and gate
status remain unchanged. The procedure is a single-listener reference study,
not a test of funniness or a claim of universal human ground truth.

2026-11-16 tailoring: the owner reports **fluent/native spoken Mandarin**. The
pilot therefore includes word-boundary marking and perceptual judgments. Reading
ability, annotation experience, equipment and prior plot exposure remain to be
recorded; fluency does not by itself establish phonetic-annotation expertise.

## What it should answer

1. In the 29 onset disagreements, does the ending begin with weak speech that
   the waveform threshold misses, or does the phonetic model start too early?
2. In the five fragmented cases, where does speech actually stop and resume,
   and is the intervening sound perceived as speech, breath, another noise, or
   no audible sound?
3. Can the same person make sufficiently consistent judgments to inform the
   existing 30 ms tolerance? If not, preserve an uncertainty region rather than
   forcing a winner.

The first outcome is an independently placed human reference for the two speech
boundaries, with explicit uncertainty. Audibility is a separate outcome. A
listener's apparent silence is not equivalent to a zero-energy waveform.

## Language and experience screen

Record spoken Mandarin ability, Chinese reading ability, phonetic-annotation
experience, whether the listener previously viewed these clips' model markers,
and headphone/output device. Familiarity with the language is necessary for
reliable identification of the intended words. Reading support can include the
Chinese transcript and untimed pinyin; it must not include machine timestamps.

- Fluent/native Mandarin: attempt both boundary marking and listening judgments,
  with training on weak consonants and explicit uncertainty.
- Conversational Mandarin: pilot first. Retain uncertain word identity and do not
  treat confident-looking timestamps as a substitute for correct identification.
- Little/no Mandarin: perform audibility and broad perceived-pause judgments.
  This can address whether the five artifacts are perceptually consequential,
  but is insufficient alone to validate Mandarin word/phone boundaries. A fluent
  listener would still be needed for that part; none is recruited automatically.

No requirement to know IPA. This is not a hearing test; inability to hear a sound
does not establish that it is absent from the signal or inaudible to others.

## Workload and sequence

**Pilot:** six anonymous original clips (three onset disagreements, two artifact
cases, one prior pass), approximately 10–15 minutes. The pilot tests whether the
instructions and tool are usable. Do not show model answers as feedback. Any
change to the boundary definition or workflow creates a protocol revision before
scored annotation starts. Pilot labels stay separate and are never pooled into
scored results. Record previous exposure to these six clips.

**Main session:** all 40 original Kokoro clips plus ten disguised exact repeats:
all five artifact clips and five onset-disagreement clips sampled using a fixed
seed. Thus 50 presentations, not 50 independent recordings. Expect roughly
60–100 minutes, to be revised from pilot timing; use short blocks and breaks.
No Melo clips are included in this first evaluation.

Use a fixed randomized order generated before ratings, subject to at least 12
intervening presentations between exact repeats. A clip's first presentation
must precede its repeat. Use anonymous trial IDs and filenames. Hide original
clip IDs, flag categories, joke/control labels, model names, model/ASR results,
and all predicted boundary positions. Meaning and voice remain audible, so this
is blinding to measurement predictions, not to all stimulus properties.

Show the full intended transcript, with an untimed vertical separator indicating
the target setup/ending transition. Do not show a model-selected time crop by
default: begin with the full original recording and let the listener locate the
transition. Keep full-sentence playback accessible during zoomed inspection.
Do not use the existing reports/plots as the annotation interface: they expose
predictions. Prior exposure cannot be undone and must be disclosed.

Save the first-pass ratings before revealing any model estimates. Hidden repeats
must not display previous answers. When all blind annotations are locked, reveal
comparisons in a separate analysis pass. Revisions after revelation are labeled
adjudication, with the initial labels retained; they are not independent ratings.

## What the listener does for each presentation

1. Play the whole unmodified recording at normal speed. Confirm that the last
   setup word and first ending word are present and intelligible. Record
   `matches`, `mismatch`, or `unsure`; describe any mismatch without repairing the
   transcript. Rate the transition as a perceived gap, continuous speech, or
   uncertain. These listening judgments precede waveform inspection.
2. Zoom around the target words while retaining surrounding context. Replay a
   meaningful speech segment, rather than judging isolated millisecond slices.
   Use both the waveform and spectrogram to place the two boundaries below.
3. For each boundary, mark an **earliest plausible time**, **best estimate**, and
   **latest plausible time**. Use a broad range or `cannot locate` when needed.
   Do not shrink the range to satisfy the 30 ms target. Values are referenced to
   the original file timeline, even when replaying a zoomed segment.
4. Describe the intervening sound: no audible sound, breath, hum/click/other
   non-speech sound, speech continues, or unclear. More than one can apply.
   Record whether the sound obscures or changes understanding of a target word.
5. Give confidence (high / medium / low) and a short reason if uncertain. Save
   and move on. A trial that cannot be resolved after approximately two minutes
   may be saved as uncertain instead of repeatedly searching for a clean answer.

### Boundary convention

**Setup end:** end of the last speech sound belonging to the final setup word.
Include a weak trailing consonant; do not stop merely when its vowel ends.

**Ending start:** beginning of the first speech sound belonging to the first
ending word. Include weak frication or aspiration that belongs to that word;
do not wait for the loud vowel. Breath/noise not belonging to the word gets a
separate label rather than being forced into speech.

Some boundaries are not acoustically identifiable. For example, a word-initial
stop may have a silent closure that cannot be separated from a preceding pause.
Do not silently substitute the audible burst for the start of that phone, or
guess the closure from the transcript. Mark `closure_or_transition_ambiguous`,
optionally mark the burst as an auxiliary observable cue, and leave the intended
phonetic boundary unresolved. This prevents the human task from quietly changing
the primary outcome to audible-vowel onset or waveform quiet.

Continuous speech does not justify forcing a positive pause. If a shared boundary
is clear, the best gap can be zero; if the boundary itself is uncertain, save that
uncertainty rather than a guessed zero. Keep overlap/negative differences visible.
A speech-free interval containing breath is not automatically pure silence.

## Tool and playback specification

Recommended implementation: local Praat with empty annotation tiers, original
WAVs and anonymous trial files. No neural inference or paid service is needed.
Praat's [TextGrid editor](https://www.fon.hum.uva.nl/praat/manual/TextGridEditor.html)
supports synchronized sound inspection and editable point/interval labels.
Use distinct point tiers for setup earliest/best/latest and ending earliest/best/
latest, plus status/notes; do not prepopulate timing proposals.

Start with a 0–8 kHz spectrogram, 5 ms window and 50 dB display range, applied
consistently. These are proposed display settings, not detector thresholds.
Changes to inspect weak speech are allowed but logged; restore the default for
the next trial. Time labels/precision in the tool do not imply equal perceptual
accuracy. Praat's [spectrogram documentation](https://www.fon.hum.uva.nl/praat/manual/Intro_3_2__Configuring_the_spectrogram.html)
explains the time/frequency trade-off. The short window is not a reliable display
of a 10–20 Hz component; the earlier signal analysis remains the evidence for it.

Use headphones in a quiet place, a comfortable level and consistent playback
settings. Record the device and any enhancements/noise suppression. Do not boost
the volume aggressively to chase the low-frequency artifact. Do not filter,
denoise, normalize only the gap, or use style-swapped audio in the primary pass.
Normal-speed playback is the basis for perceptual judgments; a slower playback
can be an auxiliary aid only if explicitly logged and confirmed at normal speed.

## Locked analysis rules

These rules are proposed before any owner annotations and must be frozen before
the scored session. They add human-review fields; they do not overwrite the
historical metrics or erase original flags.

Let setup-end uncertainty be [S_low, S_high] and ending-start uncertainty be
[E_low, E_high]. The implied gap range is:

    [E_low - S_high, E_high - S_low]

It is an elicited plausible range, not a statistical confidence interval. Do not
clamp a negative lower bound or turn an unlocatable boundary into zero. Evaluate
the two endpoints separately as well as the gap, because endpoint errors can
cancel in a duration.

For an endpoint model estimate M and human range [L, U], retain:

- Best-estimate difference M - H_best, signed and absolute.
- Whether M lies in [L, U].
- Distance outside the range: max(L - M, 0, M - U).
- Largest discrepancy across the range: max(abs(M - L), abs(M - U)).

Proposed human-review categories:

- **Compatible at the existing 30 ms tolerance:** both endpoints and the derived
  gap have largest discrepancy at most 30 ms; target words are identifiable;
  confidence is not low; there is no closure/transition ambiguity. This describes
  compatibility with this listener's reference, not guaranteed true accuracy.
- **Inconsistent with this reference:** a relevant endpoint or gap lies more than
  30 ms outside its plausible human range, without invalid/ambiguous annotation.
- **Unresolved:** uncertainty, transcript problems, nonidentifiable closure,
  conflicting evidence, or a remaining case between those two categories.

Flagged clips that lack an individual repeat check remain explicitly unreplicated
human judgments, even when compatible. No detector is chosen by optimizing its
threshold against these 40 recordings. Corrected audio, if later tested, needs a
separate blind listening test of speech preservation; it is not justified solely
because a human ignores the original artifact.

### Consistency checks

Report all ten repeat pairs, endpoint differences, gap differences, overlapping
uncertainty ranges, and categorical agreement. Separate the five artifact repeats
from the five onset repeats. For an individual repeat differing by more than
30 ms at either endpoint, retain unresolved status for that clip until reviewed;
do not average the disagreement away. Similar repeat scores measure within-person
consistency only and can still reflect systematic bias.

The first main presentation is the primary annotation. Repeats are reliability
checks, not extra independent samples or votes. Keep the six earlier automatic
passes in the 40-clip accounting. Report missing/uncertain cases explicitly and
counts by the original 29/5/6 groups after ratings are locked. Do not treat these
diagnostic clips or their shared scripts/voices as independent population samples.

## What one person's evaluation can establish

It can supply a useful human reference, identify speech that a threshold missed,
identify premature model boundaries, and establish whether artifacts are audible
or disruptive for that listener. It may leave some clips unresolved. It cannot
prove that an inaudible low-frequency signal is absent, create an observable
boundary where the acoustics do not identify one, or establish population-level
listener agreement. Formal high-confidence validation would benefit from a
second trained Mandarin listener independently reviewing difficult cases and a
sample of apparently clear cases. No such recruitment is part of this request.

Explicit criteria and consistency checks are central to manual segmentation;
see the [BAS manual-segmentation guidance](https://www.phonetik.uni-muenchen.de/forschung/BITS/TP1/Cookbook/node134.html).
This protocol does not import that project's language-specific label inventory,
pause cutoff, or alignment-assisted labeling procedure.

## Deliverables from a completed evaluation

- Locked anonymized first-pass annotations, uncertainty ranges and comments.
- Separate repeat-consistency report and any post-revelation adjudication.
- A 40-row comparison against each saved detector, retaining the original flags.
- A recommended per-clip action: retain provisional timing, investigate a likely
  detector error, or retain unresolved measurement. No automatic gate acceptance.

## Implemented practice pack, 2026-11-16

The six-clip pilot is prepared in
`evaluation/human-practice-v1/START.html`, with instructions in that folder's
README. It is a standalone local-browser annotation page, requiring no Praat
installation. The underlying boundary definitions and uncertainty fields above
are unchanged. Compared with the originally suggested Praat workflow, its fixed
spectrogram uses an explicit 5 ms Hann window, 1 ms hop and 256-point FFT; all
settings are disclosed and exported. No scored annotations have started. The
earlier protocol is archived in `runs/human-practice-v1/protocol-before-practice.md`.

Selection uses seed 20261116, the declared 3/2/1 categories, sampling within sorted
clip IDs followed by a shuffle. Only anonymous P01–P06 IDs and untimed target text
enter the participant page; the analyst mapping is saved separately. No automatic
timestamps, ASR outputs or original flags enter the client. Original WAV bytes
are embedded without transformation. Initial listening answers are collected
before plots appear and retained if later revised. Blinding remains limited by
the owner's possible prior exposure, as described above.

The page supports local autosave, JSON backup/export and same-pack restore. It
does not send annotations anywhere. All practice exports identify their purpose
as `practice`. Software checks use a separate `software_qa` namespace, banner and
export prefix; their simulated answers must never be treated as owner labels.
The actual browser preview was blocked by the tool's local-file URL restriction.
Code/data checks passed, but live-browser usability remains to be confirmed in
the pilot. No human responses have been collected by the assistant.
