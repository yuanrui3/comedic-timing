# Public source release

Prepared 25 November 2026 at the owner's request to create a public GitHub
repository and push the project. Publication does not change the exploratory
status, measurement definitions, 34 recorded flags, or Gate 1 decision.

## Included

- Project Python/HTML/JavaScript source, software tests and dependency snapshots.
- Experiment plans, methods, research reviews, limitations and decision history.
- Ten Mandarin diagnostic script pairs with source/adaptation provenance.
- Static research figures and the roadmap PDF/PNG.
- The CPU-only synthetic mathematical audit, including its source snapshot,
  numerical results and provenance.
- Existing third-party attribution/license records under `config/` and
  `docs/licenses/`.

## Retained only in the local workspace

- Generated speech, downloaded model weights, model-code checkouts and runtimes.
- Caches, raw experiment runs, log files and run-specific environment records.
- The built practice page, which embeds original audio; private practice mapping,
  practice build/QA artifacts, and participant answer exports.
- Credentials, local settings and local publication tooling.

These files are ignored, not deleted. The public history is initialized from the
reviewed files; it does not include a previous commit containing private artifacts.
Publishing the research reports does expose model analyses of the original
stimuli. Independent annotators should avoid those reports until their labels
are locked; withholding the anonymous practice mapping is not complete protection
against an annotator actively matching transcripts to the public reports.

## Reproduction

This is a source-and-report release, not a self-contained data archive. Existing
report links and commands describe the full local project; excluded media and raw
run JSON files must be regenerated or supplied locally for those commands.
Dependency lock files are version snapshots, not portable binary environments.
The original implementation uses Windows and project-local Python/native runtimes;
a clean installation on another machine has not been verified.

The mathematical audit's numerical outputs are included. Its original runner
also verifies two local study files by hash, so replaying it unchanged requires
those exact local files. The synthetic numerical checks themselves use no audio,
human responses or detector measurements. This integrity precondition has not
been weakened for publication.

The Python software suite can be run from an environment containing the relevant
dependencies with `python -m unittest discover -s tests -v`. Tests involving saved
study artifacts or the browser-practice build may require those local artifacts.
Software checks do not establish accuracy of the scientific measurements.

## Rights and attribution

No project-wide license has been selected for the original source, documentation
or generated study artifacts. This release does not apply a new license to them
or change upstream terms. Third-party materials retain their own notices.

The Mandarin texts are newly worded diagnostic adaptations of historical
*笑林廣記* entries, with sources recorded in
`data/scripts/open_zh_diagnostic.json`; they are not transcribed modern comedy
performances. Contemporary translations/performances are not included.

Model weights and third-party runtime packages are excluded. Their recorded
repositories, revisions and reported licenses in configuration/reproduction
documents identify the dependencies used in the local experiments. Consult the
upstream release terms when acquiring or redistributing any of those assets.
The existing Charsiu and Silero notices are retained in `docs/licenses/`.
Research papers are linked and discussed; their full texts are not redistributed.
