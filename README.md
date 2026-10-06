# GEMSDOE44 — DOE GEMS Prize fault-discovery system (DrivenData #306)

**Live site (start here):** <https://buffedlizard55-lab.github.io/GEMSDOE44/docs/index.html>

---

## 0. STANDING PROMPT — read this at the start of every session

> **Goal.** Place at the top of the leaderboard in the DOE GEMS Prize (DrivenData #306) by
> finding geological **faults missing from the public USGS/INGENIOUS catalogue** in the GeoDAWN
> region of Nevada — the structures that indicate hidden geothermal resources.
>
> **Hard requirement.** This repository must always publish an **easy-to-download, unique
> submission GeoTIFF**: single click on the site → a legal, format-verified file → upload it to
> the DrivenData submission form with a **unique name** and a **short Note** so submissions can
> be told apart later. The download must be the first, most obvious thing on the site, and the
> site must explain exactly how to submit in an executive summary.
>
> **Never copy a previous submission.** Earlier GEMSDOE sites are for learning and education
> only. Every artifact published here must be built by this repository's own pipeline and must
> differ in content from every artifact listed in the GEMSDOE family.
>
> **Method discipline, every session.**
> 1. Generate **3–5 candidate geological hypotheses not yet tried**, each naming: the specific
>    layer(s) involved; the physical signature targeted (edge-detection transform, curvature,
>    lineament, etc.); why it should catch a fault *missing* from the USGS/INGENIOUS catalogue
>    rather than one already in it; and how it differs from everything already implemented in
>    this repository and its predecessors.
> 2. **Rank them by expected DTI improvement per unit of implementation cost.**
> 3. **Validate the top candidate on the spatially-blocked holdout before touching a submission
>    slot.** If it cannot be validated without new external data, name the specific free,
>    official source needed and *check that it is obtainable* before calling the idea viable.
> 4. **Flag every irregularity for review.** No hallucinations. Verify line by line from
>    official, trusted sources and give links for manual review.
> 5. **Work autonomously.** No manual input should be required.
> 6. Run every task through **multiple passes**: implement, then review for bugs and missing
>    requirements, then re-check the whole thing against this prompt. Do not stop after pass 1.
>
> **Core Values — kept central to every decision here.**
>
> **Maximize P(Win).** *"Maximize the Probability of Winning"* is our decision-making framework.
> In every decision we weigh tradeoffs, assess risk, and choose the path that maximizes the
> probability that we win. We set aside our emotions and make tough decisions in order to
> maximize P(Win). It frees us from constraints and clarifies that we must put the outcome
> first.
>
> **Own the Outcome.** We own results end to end — not just our individual slice of the work.
> When problems arise and we have the means to act, we act without waiting for permission or
> assignment. We treat failure and success as signals and use them to improve. We stay
> accountable to the final outcome.

**How the Core Values were applied in this session, concretely.** *Maximize P(Win)* is why
hypothesis H1 was killed after four minutes of measurement instead of being built into a
submission, why the first supervised field was scrapped when its blocked AUC of 0.773 failed to
translate into metric credit, and why field selection was moved onto the official metric itself.
*Own the Outcome* is why every negative result is committed to `registry/` rather than quietly
replaced, and why `registry/irregularities.json` lists eight defects including ones in our own
work.

---

## 1. The one-click submission file

**→ <https://buffedlizard55-lab.github.io/GEMSDOE44/docs/index.html>** — the download is the first
element on the page, with the exact Note to paste and the exact clicks to make.

If the portal ever answers *"Predicted values must be in range [0, 1]"*, the primary file here is
written so that **every** cell is finite and every value is exactly `0.0` or `1.0`, so that
message cannot be triggered by this file. The reason the message normally appears, and the
`-nan.tif` twin for anyone who prefers the official template's convention, are documented in
[`docs/how-to-submit.html`](docs/how-to-submit.html).

### 1.1 Submission record

| field | value |
| --- | --- |
| file | `GEMS44_n-strand-ssmc_20261006T110212Z_4971d359-zeros.tif` (320,989 B) |
| sha256 | `4971d3593c2990bb275b77e5663814d1540af8d6a4de7679dae0dbd7e9605988` |
| zip | `GEMS44_n-strand-ssmc_20261006T110212Z_4971d359-zeros.zip` (210,323 B, inner member byte-identical) |
| predicted cells | 44,090 cells, all exactly 1.0 |
| portal Name / Note | `GEMS44_n-strand-ssmc_20261006T110212Z_4971d359` / "GEMS44 n-strand-ssmc | field = supervised transfer of USGS SGMC strands absent from the catalogue (U) onto the 19 offici…" |
| held-out evidence | candidate 0.401181 vs incumbent 0.165267 (delta +0.235915), 4/4 folds positive |
| format verification | 17 checks, 0 failures — [`registry/verification.json`](registry/verification.json) |
| request checklist | 11 PASS / 1 PARTIAL / 1 PENDING / 0 FAIL — [`registry/request_checklist.json`](registry/request_checklist.json) |
| uniqueness | max Jaccard against any obtainable prior artifact **0.0202**; against the 0.2600 incumbent **0.0121** |

### 1.2 Why the emitter is spaced, not dense

Nineteen already-scored family artifacts were measured for how many of their dots sit 8-adjacent to
another dot. The rank correlation with the live score is **−0.361**: the three best artifacts in the
family (0.2449, 0.2477, 0.2600) have an adjacency fraction of 0.001–0.002, the worst have 0.88–0.99,
and the family's own "DTI-optimal emission" attempt at 335,879 dots scored 0.1002. A dot adjacent to
a dot that already covers the same ground buys almost no new truth coverage while paying the full
false-positive weight. The emitter therefore enforces a minimum separation of 3 px
(`--min-sep`, default 3.0) and the audit reports the verified neighbour count.

---

## 2. What this repository does differently

Three mechanisms, each falsifiable, each carried by a script and a test.

### 2.1 It selects on the metric, not on AUC — because the measurement said so

A clustered field with a spatially blocked **pixel AUC of 0.773** was emitted with an exact
greedy max-coverage emitter and scored with the official metric on a held-out quadrant. It scored
**0.0766** at 44,090 dots. Uniform-random dots at the same mass scored **0.1341**. The real,
live-scored 0.2600 incumbent artifact scored **0.1353** on the same frame, same metric.

An AUC of 0.773 is therefore not evidence that a field will earn metric credit. The reason is the
metric itself: `TP_w` is a **max** over a 300 m disc, so a field earns nothing unless a dot lands
*inside the kernel of a scored truth pixel*. AUC rewards ranking everywhere; the metric pays only
inside a 300 m neighbourhood. **Field and mass are consequently selected by the official metric on
spatially blocked frames.** The failing fold is preserved verbatim in
[`registry/holdout_partial.json`](registry/holdout_partial.json).

### 2.2 It never uses the catalogue as a promotion frame

The competition scores faults the catalogue does **not** contain, so a catalogue-truth proxy
cannot represent the target. This is not a theory: the prior family measured its own catalogue
proxy ranking three artifacts `0.17193 > 0.16635 > 0.16177` while the live board ranked the same
three `0.1922 < 0.2477 < 0.2600` — the **opposite** order. The primary frame here is the one real,
officially published, locally available set of mapped faults the given catalogue lacks:
**62,122 px** of USGS State Geologic Map Compilation faults more than 300 m from the catalogue.

### 2.3 It emits with the metric's own arithmetic

* `FN_w = |G| − TP_w` exactly, so `DTI = TP_w / (0.2·(TP_w + FP_w) + 0.8·|G|)`.
* A binary `{0,1}` file **strictly dominates** any graded version of itself (unit-tested).
* **Marginal rule:** adding a dot of kernel credit `k` pays iff `k > 0.2·DTI`. At a score of
  0.3345 that is **280 m**. Credit is nearly free anywhere inside the kernel and worthless
  outside it — this competition is a *distance/coverage* problem, not a precision problem.
* Since `TP_w` is a max over a disc, expected credit is **submodular** ⇒ fixed-order greedy
  emission carries the standard (1 − 1/e) guarantee. The stopping rule is the closed-form
  break-even condition above, not a tuned knob.

---

## 3. Verified state (2026-10-06)

| item | value | evidence |
| --- | --- | --- |
| Official data acquired | `labels.tif`, `sample_submission.tif`, `training_features.tif` (418,912,844 B, 19 bands), SGMC faults | every sha256 matched its pin — [`registry/data_manifest.json`](registry/data_manifest.json) |
| Metric implementation | exact transcription of the published equations + masking rule, with an independent brute-force transcription | **23 tests pass**, `tests/test_metric.py` |
| Grid contract | EPSG:32611, 100 m, 3730 × 3292, footprint 5,167,373 px, known mask 60,988 px | `registry/population.json` |
| Detectability of catalogue-absent faults | blocked AUC **0.773** (folds 0.762–0.789) | `registry/detectability.json` |
| Hypothesis H1 (anisotropic splay prior) | **FALSIFIED** — offset-angle statistics identical to a matched random control | `registry/population.json` |
| Live leaderboard | rank 1 **0.3345**; the brief's 0.3195 is rank 5 on this read | `registry/leaderboard.json` |
| Format validation | CRS, resolution, shape, transform, dtype, band count, range, finiteness — re-read from disk | `submissions/*-audit.json` |
| Emission fix (spacing) | `min_sep = 3 px` enforced and verified by exact neighbour count; the dense artifact (0.99168 adjacency) was superseded and deleted | `registry/verification.json :: pass_a_contract` |
| Artifact verification | 17 checks green: contract read back from the shipped bytes, zip byte-identity, metric re-derived by an independent padded-shift implementation plus hand-computed toy cases | `registry/verification.json` |
| Request checklist | 11 PASS, 1 PARTIAL (the 0.2778 attribution — IR-44-03), 1 PENDING (PR/merge) | `registry/request_checklist.json` |

---

## 4. Repository map

```
README.md                     this file (standing prompt at the top)
docs/                         GitHub Pages site — index.html leads with the download
  index.html                  executive summary + one-click file
  how-to-submit.html          exact clicks, the exact Note, the range-error fix
  data.html                   auditable data table, grid contract, official links
  method.html                 what was done in order, including the negative results
  validation.html             how the validation frame was chosen, and the probe evidence
  hypotheses.html             the ranked hypotheses, falsified ones kept in
  leaderboard.html            official snapshot + attribution caveat
  limitations.html            what stands in the way + flagged irregularities + next steps
  downloads/                  the submission GeoTIFF, its .zip, its NaN twin, audit sidecars
src/gems44/
  metric.py                   exact official DTI (+ closed forms used by the emitter)
  grid.py                     raster contract: footprint, known mask, band reader
  population.py               measured fault-population statistics
  field.py                    feature spec, greedy marginal-credit emitter
  submit.py                   TIF writer, independent verifier, zip builder
  scripts_common.py           band loading, feature blocks, model training
scripts/
  fetch_official_mirrors.sh   hash-verified fetch, fails closed
  measure_population.py       H1 falsification test + population statistics
  measure_detectability.py    blocked AUC for catalogue-absent faults
  measure_strands.py          geometry of the two fault compilations
  fetch_probes.py             downloads family artifacts whose live scores were reported
  rank_frames.py              ranks candidate local frames against those reported scores
  probe_frame_validity.py     first falsification test of the pre-registered frame
  analyze_halo.py             separates halo concentration from mass as score explanations
  run_selection.py            metric-aligned blocked field + mass selection  <- decisive
  run_holdout.py              exact-operator confirmation run
  build_submission.py         builds and independently verifies the artifact
  build_site.py               renders the site from registry/ only
registry/                     machine-readable evidence; every number on the site comes from here
tests/                        metric, emitter, submission-format and grid tests
data/                         official rasters (gitignored; fetch with the script)
```

---

## 5. Quickstart

```bash
# 1. data (hash-verified; requires `gh` authenticated for github.com)
bash scripts/fetch_official_mirrors.sh

# 2. science
PYTHONPATH=src python scripts/measure_population.py      # falsifies H1, measures the population
PYTHONPATH=src python scripts/measure_detectability.py   # blocked AUC on a catalogue-free frame
PYTHONPATH=src python scripts/run_selection.py           # metric-aligned field + mass selection
PYTHONPATH=src python scripts/run_holdout.py             # exact-operator confirmation

# 3. artifact + site
PYTHONPATH=src python scripts/build_submission.py
PYTHONPATH=src python -m pytest tests -q
python scripts/build_site.py
```

Environment: Python 3.11, `numpy scipy scikit-image scikit-learn rasterio pytest`.
No GPU is required or used.

---

## 6. Limitations in the way (full list: [`docs/limitations.html`](docs/limitations.html))

1. **No organizer score exists for any artifact in this repository.** Every number here is a
   local measurement on a proxy frame. Uploading is the only way to convert a proxy into a fact.
2. **The hidden label set's provenance is unknown.** The organizers' reply in community thread
   11527 could not be read from this sandbox, so no claim is made about it (IR-44-01).
3. **This sandbox reaches only GitHub and PyPI.** Measured 2026-10-06: `earthexplorer.usgs.gov`,
   `prd-tnm.s3.amazonaws.com`, `gdr.openei.org`, `storage.googleapis.com`,
   `raw.githubusercontent.com` and every `*.github.io` host returned HTTP 000. Hypotheses H4 and
   H6 are data-blocked here for that measured reason, not by assumption.
4. **The primary frame is an analogue, not the target.** SGMC faults are pre-Quaternary-inclusive;
   the hidden set need not share their statistics.
5. **Quadrant blocking is coarse.** It removes large-scale leakage but not regional structure that
   persists within a quadrant. Finer blocking is the next methodological upgrade.
6. **Leaderboard aggregation is unresolved** — pooled-over-pixels versus mean-of-per-chunk is an
   unanswered forum question, so a pooled local frame is not guaranteed to map linearly onto the
   board.

**Next steps, in priority order:** upload the artifact and read the board; finer spatial blocking;
H4 coupled ridge–trough matched filter on the derived 1 m lidar scarp raster (already obtainable
from the mirror); H5 cross-compilation disagreement as a gated feature; the two-round emission bar
`0.2·DTI/(1+ρ)`; and an independent scrutineer pass on `src/gems44/metric.py`, because the whole
project reduces to that one operator agreeing with the organizer's scorer.

---

## 7. Sources for manual review

* Problem description, metric and submission format —
  <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
* Leaderboard — <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/>
* Known-fault masking, DrivenData staff statement — <https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516>
* Data provenance question (unreadable from here) — <https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527>
* Reference solution — <https://github.com/drivendataorg/gems-prize-reference-solution>
* Official rules PDF — <https://docs.nlr.gov/docs/fy26osti/96647.pdf> (outside this sandbox's
  network allowlist; not relied on by any quantitative claim here)

Every link above, its fetch status, and what it establishes are in
[`registry/sources.json`](registry/sources.json).

---

## 8. Sibling sessions in this repository

Other Arena sessions worked on the same competition in this same repository and merged their work
into `main` (PR #1 and PR #2). **Nothing of theirs was deleted by the merge of this branch**: their
scripts, tests, registries, artifacts and pages are all still in the tree, and the pages whose paths
collided with this site's generator were preserved verbatim under
[`docs/sibling/`](docs/sibling/):

| what | where |
| --- | --- |
| their README | [`docs/sibling/README_main.md`](docs/sibling/README_main.md) |
| their executive summary / hypotheses / leaderboard pages | [`docs/sibling/`](docs/sibling/) |
| their evidence pages (untouched paths) | [`docs/executive_summary.html`](docs/executive_summary.html), [`docs/irregularities.html`](docs/irregularities.html), [`docs/research.html`](docs/research.html), [`docs/sources.html`](docs/sources.html) |
| their research notes | [`docs/research/`](docs/research/) |
| their artifacts | `docs/downloads/gemsdoe44-h44-*` (this session's submission files sit beside them, different names) |
| their pipeline | `generate_submission.py`, `scripts/{prepare_data,run_checks,generate_h44_6_prune,analyze_spacing_stdlib,audit_tif_stdlib}.py`, `tests/test_submission_and_metric.py` |

Where their analysis and this repository's evidence interact, this repository states the measured
result rather than the assertion: see [`registry/ring_mechanism.json`](registry/ring_mechanism.json)
for the first-hand byte-level test of the 0.2600 → 0.2778 pruning claim, and
`docs/limitations.html` for what remains unverified.
