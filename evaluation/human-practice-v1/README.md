# Six-clip Mandarin boundary practice

Open [START.html](START.html) in Edge, Chrome, or another modern desktop browser.
If the link displays HTML source, locate this folder in File Explorer and open
START.html with your browser. The single 11.6 MB file contains the six original
recordings, untimed target text, waveform and spectrogram data. It uses no server,
account, external resource, model inference or upload. No new software is required.

Your reported spoken Mandarin ability is fluent/native. The page asks separately
about reading, annotation experience, playback device and prior plot exposure.
None of those answers has been assumed or filled in for you.

Allow approximately 10–15 minutes for this pilot, with extra time if the tool is
new. These are practice annotations, kept separate from the later scored review.

1. Read the short instructions and fill in your setup. Use headphones, a quiet
   place and comfortable volume. Start at normal playback speed.
2. Listen to each complete recording. The text separator `│` indicates the target
   word transition without giving its time. Answer the two listening questions.
3. Select **Reveal plots and mark boundaries**. For setup end and ending start,
   choose the location status and mark earliest plausible, best and latest times.
   Click **Mark on plot** under a field, then click either plot. Zoom in around
   your last click to refine it. You can also type times in seconds from the
   original recording's beginning. Original-file times persist when zooming.
4. Include weak speech sounds before a loud vowel. If a boundary cannot be
   identified, choose **Cannot locate** and leave its three time fields empty.
   Mark closure/transition ambiguity when appropriate; do not invent a timestamp.
5. Describe intervening sound, confidence and any difficulty. **Save trial**, then
   **Next**. Original first-listening judgments remain recorded if you revise them
   after inspecting plots.
6. Click **Export practice answers**. Keep the downloaded
   `mandarin-practice-<date>.json` file and provide it in this chat for review.
   Partial exports are supported. The page can restore the same pack's JSON file
   through **Resume from an exported JSON backup**.

Browser autosave is a convenience and can depend on local-file/browser settings.
Use exports as backups. No answers are sent to the assistant automatically.
Changing computers or browsers requires importing the exported JSON. Practice
pages do not contain predicted boundaries, detector outputs, flag categories,
source clip IDs or previously recorded human answers.

Orange lines show your setup-end marks; cyan lines show your ending-start marks.
Solid lines are your best estimates; dashed lines show your plausible limits.
No accuracy score appears. A negative gap/range is preserved, not silently forced
to zero. Continuous playback from the visible start retains surrounding context;
pause using the audio controls. Playback seeking is not used to measure a mark.

The waveform is a native-audio min/max display in 1 ms bins; the spectrogram uses
a 5 ms Hann window, 1 ms hop, 256-point FFT, 0–8 kHz display and 50 dB range. It
is an inspection aid, not an exact boundary oracle or a reliable view of subsonic
drift. Audio bytes are identical to the original recordings; displayed rounding
and a three-decimal time field do not imply 1 ms perceptual accuracy.

After the pilot, we will review tool usability, effort, and the frequency of
uncertain judgments before preparing the 40-clip scored session plus ten hidden
repeats. Practice ratings will not be counted as scored human reference data.

Validation: six embedded audio hashes/durations/native sample rates verified;
eleven software-only checks cover initial blinding, range validation, original
time coordinates, saving/navigation, export/import and missing values. The QA
file and QA-tagged test answers are separate from this practice pack. Real browser
preview was blocked by the browser tool's local-file URL policy, so visual layout,
actual playback and browser download behavior have not been verified in a live
browser. No workaround or browser security change was attempted.
