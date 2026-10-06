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
> **Session amendment (2026-10-06, session 2 — carried forward from here on).** The problem is to
> be split into **two explicitly separate, separately validated stages**, never collapsed into one
> number:
> * **Stage 1 — coarse favouritability.** Strain, heat flow and geochemistry (the official
>   geodetic, conductivity / depth-to-basement and seismicity bands). It decides **only which broad
>   zones are worth searching at all**, and it is validated as a *zone* statement (blocked-cell AUC,
>   base-rate lift, recall of catalogue-absent faults inside the approved area), never with a
>   pixel metric.
> * **Stage 2 — independent fine placement.** A placement model trained and holdout-scored on its
>   own, using only the precision/recall structure the official DTI actually rewards, placing dots
>   **only inside zones Stage 1 approved**.
>
> Both stages' holdout performance must be reported **separately** in the write-up, and the final
> output must be checked against a Stage-1-only emission and a Stage-2-ungated emission at the same
> mass, so it cannot be one stage's footprint wearing a two-stage label. If the coarse stage turns
> out to be nearly uninformative on its own, say so and publish the number (this session: the
> physical-only gate's blocked-cell AUC is 0.529 with a base-rate lift of 1.08×, and that finding is
> printed in `registry/twostage/submission.json :: stage_a_honesty`).
>
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
| file | `GEMS44_n-strand-ssmc_20261006T113858Z_bf3b3914-zeros.tif` (313,219 B) |
| sha256 | `bf3b3914b1730596583bf84483f8bcd67202836944f4044464bfaaeaf67ed913` |
| zip | `GEMS44_n-strand-ssmc_20261006T113858Z_bf3b3914-zeros.zip` (190,160 B, inner member byte-identical) |
| predicted cells | 44,090 cells, all exactly 1.0 |
| field | supervised transfer of USGS SGMC strands the catalogue lacks (`sup_U`), **no** catalogue blend (alpha 1.0) |
| ring | `--ring-px 2`: **0 dots within 200 m** of the known catalogue (0), 3,889 in the 200–300 m band |
| spacing | min separation 3 px verified by exact neighbour count (max neighbours within 2 px = 0); 8-adjacent fraction 0.00098 |
| portal Name / Note | `GEMS44_n-strand-ssmc_20261006T113858Z_bf3b3914` |
| blocked evidence | sup_U at 44,090 on admissible frame P: mean held-out DTI 0.1453 vs incumbent artifact 0.0924, **4/4 folds positive** (sweep, per-fold retrained); the full-grid 0.3639 shown next to the download is in-sample/circular and is labelled so |
| uniqueness | Jaccard vs the 0.2600 incumbent **0.0076** (665 shared dots) |
| format verification | pass A/B/C, see [`registry/verification.json`](registry/verification.json) |
| request checklist | [`registry/request_checklist.json`](registry/request_checklist.json) |

### 1.2 Why the prior best scored 0.2778, measured first-hand

The locally available 0.2778 artifact is a **strict subset** of the live-verified 0.2600 artifact:
6,436 dots deleted, none added, **100% of the deletions inside the 200 m catalogue ring** (the 0–100 m
and 100–200 m bands are emptied to exactly 0.000). Pricing those dots with the exact operator on the
local truth frames:

| frame | 0.2600 anchor | 0.2778 artifact | delta | removed dots' mean credit | net denominator effect |
| --- | --- | --- | --- | --- | --- |
| frame P (beyond 300 m, admissible) | 0.094174 | 0.095386 | **+0.001212** | 0.0135 | +1187.5 |
| frame N (within 300 m, falsified) | 0.175536 | 0.054060 | -0.121476 | 0.4978 | -2220.7 |

So the mechanism is real on the frame that can rank the pair correctly, and the same measurement is
what falsifies frame N: it prefers a dot population that the live board penalised. Is >0.2778
reachable? The marginal rule says a dot pays iff its kernel credit exceeds 0.2·DTI (0.0556 at 0.2778).
Pruning alone therefore cannot get far — it removes cost but also coverage. The two levers that remain
are (a) place dots only where expected credit clears the bar, which is what the ring rule and the
spacing rule do, and (b) improve the field that decides *where* inside the admissible region the dots
go. This artifact takes (a) to its measured limit and (b) to the limit of the evidence available
here; whether that clears 0.2778 is unmeasurable from this sandbox, and the site says so.

### 1.3 Emission rules, each pinned to a measurement

* **Spacing.** `Spearman(reported score, fraction of dots 8-adjacent to another dot) = −0.361` over 19
  live-scored artifacts; the best three sit at 0.001–0.002, the worst at 0.88–0.99.
* **Ring.** Paired at matched mass, excluding the ≤200 m ring changes the admissible frame by
  +0.0109 for the shipped field and is positive for
  every field and mass tested; on the falsified frame it is negative, which is why the frame question
  was settled first.
* **Mass.** Matched to the live-verified incumbent (44,090) so the artifact is a clean A/B on field and
  ring rule alone. Frame P prefers more mass, but frame P's truth is large and diffuse, so that
  preference is not extrapolated.

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

### 2.2 It never uses the catalogue as a promotion frame — and it drops the frame that secretly was one

The competition scores faults the catalogue does **not** contain, so a catalogue-truth proxy
cannot represent the target. This is not a theory: the prior family measured its own catalogue
proxy ranking three artifacts `0.17193 > 0.16635 > 0.16177` while the live board ranked the same
three `0.1922 < 0.2477 < 0.2600` — the **opposite** order. The primary frame here is the one real,
officially published, locally available set of mapped faults the given catalogue lacks:
**62,122 px** of USGS State Geologic Map Compilation faults more than 300 m from the catalogue.

### 2.2b The frame arc, in full

| step | frame used | what happened |
| --- | --- | --- |
| frozen at the start | P (SGMC beyond 300 m) | pre-registered |
| AM-44-01 | N (SGMC within 300 m) | switched after the 19-probe ranking, before any selection result |
| AM-44-03 | N **excluded** | a direct A/B (the 0.2600 and 0.2778 artifacts) is ranked backwards by N; admissibility now requires ranking that pair correctly **and** positive probe excess |
| AM-44-04 | P | field re-picked on P: `sup_U` (+0.0808 mean fold excess over uniform at 61,328; 4/4 folds) beats the catalogue-proximity field, which is *worse than uniform* on P |

Every step is in [`registry/preregistration.json`](registry/preregistration.json); the falsification is
IR-44-15 and the consequence is that the previously shipped artifact
(`..._4971d359-zeros.tif`, frame-N selected, 34.7% of its dots inside the ring) was withdrawn and
replaced by the current one.

**Merge record.** PR #3 (first artifact, verification harness, registries, site) merged as `b9d4ad8`;
PR #5 (frame falsification, field re-pick, ring rule, current artifact) merged as `186bda8`. Sibling
sessions had already landed as PR #1 (`bcad3e2`), PR #2 (`988fd44`) and PR #4 (`3efb86b`), and their
files survive every merge: the check in `scripts/check_request.py` item 13 and the
[`docs/sibling/`](docs/sibling/) directory both assert it.

### 2.2c The final admissibility table (six frames, 19 live-scored artifacts)

| candidate frame | truth px | rho vs reported score | rho of excess over uniform | probes above uniform | ranks the 0.2600/0.2778 pair correctly |
| --- | --- | --- | --- | --- | --- |
| `N_sgmc_within_300m` | 17,493 | +0.335 | +0.396 | 19/19 | no |
| `P_sgmc_beyond_300m` | 62,122 | -0.054 | +0.188 | 8/19 | **yes** |
| `Q_sgmc_200_300m` | 4,155 | +0.468 | +0.382 | 19/19 | no |
| `R_sgmc_beyond_200m` | 66,277 | -0.046 | +0.191 | 8/19 | no |
| `S0_catalogue_nomask` | 60,988 | -0.475 | -0.237 | 19/19 | no |
| `U_sgmc_all_off_catalogue` | 79,615 | -0.393 | +0.209 | 9/19 | no |

Exactly one frame passes: **P (SGMC faults beyond 300 m of the catalogue)**, rho_excess +0.188 and
8/19 probes above their own uniform controls. The frame with the highest excess (N, +0.396) is
*inadmissible* — it ranks the family's live-verified best pair backwards — which is the whole point of
the pair test added in AM-44-03. The shipped artifact and the ring rule are both selected on P; on P
the ring exclusion is positive for every field and mass tested, and `sup_U` beats the incumbent
artifact in 4/4 blocked folds. Source: [`registry/frame_ranking.json`](registry/frame_ranking.json),
[`registry/ring_experiment.json`](registry/ring_experiment.json).

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
| session 3 — two-stage gate + fine placement | site archived at [`docs/sibling/session3/`](docs/sibling/session3/) (incl. the per-stage holdout page); code line `src/gems44/{config,data_io,features,stage1,stage2,emission,emit_submission,mirror_model,metric44}.py`, `scripts/run_all.py` + `scripts/finalize_submission.py`; evidence `evidence/*.json` + `docs/data/*.json`. Pre-registered 4-quadrant holdout: Stage-1 gate AUC 0.864 / conc 2.435 (PASS); Stage-2 arm table A4 0.0230 vs A1 habitat 0.0308 (catalogue-proxy promotion rule NOT met — disclosed). **Final file** `docs/downloads/gemsdoe44-primary.tif` (45,000 dots, sha256 `e3ce40cc…b52cde`, format-audited): the two-stage placement, chosen because it won the off-catalogue mirror test 0.1961 vs 0.1235 (habitat) and beat all four incumbents on the same model — the proxy holdout's opposite verdict is exactly the circularity the separate validations exist to catch. Its DTI variant (no known-fault mask) is `metric44.py` to avoid clobbering this repo's `metric.py` |

Where their analysis and this repository's evidence interact, this repository states the measured
result rather than the assertion: see [`registry/ring_mechanism.json`](registry/ring_mechanism.json)
for the first-hand byte-level test of the 0.2600 → 0.2778 pruning claim, and
`docs/limitations.html` for what remains unverified.

---

## 9. Session 2 (2026-10-06) — the calibrated instrument and the two-stage artifact

This section is additive: everything above it (including the artifact record in §1) is the state of
`main` as merged from the parallel session, and nothing there is replaced. This session adds a
second, independent artifact, the instrument that chose it, and the forensics behind the 0.2778
question. Full write-up: [`docs/research/why_the_family_top_file_scored_highest.md`](docs/research/why_the_family_top_file_scored_highest.md);
machine-readable evidence: [`registry/twostage/`](registry/twostage/).

### 9.1 The submission this session ships

| field | value |
| --- | --- |
| file | `GEMS44_h46-twostageAB_20261006T160000Z_b0cfe956-zeros.tif` (315,429 B) |
| sha256 | `3c04892d0deca00ccbadf77c801130180f05a03a9c0d575e4d8b91127286d06f` |
| zip | `GEMS44_h46-twostageAB_20261006T160000Z_b0cfe956-zeros.zip` (198,695 B, inner member byte-identical) |
| NaN-outside twin | `…-nan.tif` (official template convention) |
| predicted cells | 42,000, every value exactly 1.0; **0 on the masked catalogue, 0 within 200 m of it** |
| portal Name / Note | `GEMS44-H46-TWOSTAGE-AB` / "GEMS44 H46 two-stage A x B \| Stage A coarse physical-zone gate, Stage B curvature/relief placement, exact greedy max-coverage emission, 42,000 dots, 0 within 200 m of the catalogue, 0 on the masked catalogue; holdout 4/4 folds vs the incumbent; UNSCORED" |
| format verification | 10/10 checks re-read from the bytes — `registry/twostage/submission.json :: format.primary.checks` |
| blocked holdout | frame P: **4/4 folds** beat the incumbent at every mass; shipped recipe 0.2267 credit/dot vs the incumbent's 0.1263 |
| uniqueness | max Jaccard against any of the 114 locally available prior artifacts **0.0199** |
| site entry point | <https://buffedlizard55-lab.github.io/GEMSDOE44/docs/index.html> (download is the first element) |

### 9.2 The instrument question — settled, with 21 artifacts

Every candidate statistic was computed for all 21 locally available artifacts carrying an
owner-reported live score (IR-44-08: used as an order, never as a receipt), with this repository's
exact metric:

| instrument | all 21 | the 13 at live ≥ 0.18 | the 6-point pruning ladder |
| --- | ---: | ---: | ---: |
| **frame-P credit per dot** (mass-corrected) | **+0.549** (p = 0.0099) | **+0.945** (p < 0.0001) | **+1.000** |
| frame-P DTI at own mass | +0.017 | −0.308 | −0.543 |
| frame-P excess over uniform | +0.509 | +0.599 | +1.000 |
| frame-N DTI (**the previous promotion frame**) | **−0.500** | **−0.731** | −0.771 |
| frame-N excess (**the previous promotion statistic**) | −0.168 (p = 0.47) | −0.357 (p = 0.23) | −0.371 |
| catalogue DTI with the mask off (contaminated) | −0.803 | −0.907 | — |

Frame P = the 62,122 px of USGS SGMC faults > 300 m from the competition catalogue. The metric's
algebra explains why the mass correction matters: every binary dot raises the denominator by exactly
α = 0.2 regardless of where it lands, so a dot pays iff its own realised credit exceeds 0.2·DTI —
0.052 at a score of 0.26. **The instrument-inflation warning is published with it**: a field fitted
directly on the SGMC off-catalogue population (`gemsdoe29-sgmc-off-catalogue-44k`) scores 0.79
credit/dot on frame P and 0.0512 live, so the frame-P statistic may only be used **out-of-fold**.
Hence the blocked quadrant sweep, and hence the shipped recipe was chosen from the sweep's 0.2267,
not from the shipped file's in-sample 0.4414.

### 9.3 Two stages, separately validated — including the number that undercuts the story

* **Stage A (coarse favouritability, a ZONE statement).** Physical bands (strain, conductivity
  surface, depth to basement, seismicity, gravity, magnetics) plus catalogue fault-corridor density
  at 2 km and 10 km → a gate over 70% of the footprint. Reported twice on purpose: blocked-cell AUC
  **0.942** for the shipped score (self-referential — it is validated against "does this 2 km cell
  contain catalogue faults" and *contains* a smoothed catalogue), versus **0.529** for the
  physical-only variant with a base-rate lift of **1.08×**. The honest reading: coarse physical
  favouritability is nearly uninformative for fault occurrence at this scale, and that is exactly
  why the family's habitat-only emissions scored 0.0041 / 0.1223 / 0.1352.
* **Stage B (fine placement, a PIXEL statement).** 51 transforms of the 19 official bands —
  local relief models, topographic openness/anti-slope, Hessian curvature and ridge response of the
  detrended elevation, potential-field total horizontal gradients, Basin & Range strike-fabric
  alignment, multi-scale step energy. Blocked-quadrant AUC against frame P: `hessian_curv_s3`
  0.7456, `lrm_grad_r15` 0.7340, `lrm_abs_r15` 0.6852, `fabric_align_s5` 0.6755. The classical
  potential-field edge operators are all **below chance on catalogue-absent faults** at 100 m
  (tilt/curvature 0.37, gravity THG 0.46) — a physical statement about what these faults are.
* **Emission**: exact greedy max-coverage over the metric's own kernel (submodular ⇒ (1−1/e)),
  3 px minimum separation, no mass within 200 m of the catalogue. Measured at one field, one mass,
  one scorer: 3 px beats 4/5/6 px, and greedy beats raster-order thinning 0.2267 vs 0.0439 credit
  per dot.
* **Dominance check** (same field, same mass, same scorer): Stage-A-only ranked by the Stage A
  score scores 0.0683 on frame P against the shipped file's 0.3081, Jaccard 0.006 — the file is not
  Stage A's footprint. The ungated Stage-B controls score *higher in-sample* for a mechanical reason
  (a gate only removes candidate pixels), which is stated on the validation page rather than hidden;
  the gate's actual evidence is the out-of-fold sweep, where it wins on **both** frames in 4/4
  folds at every mass.
* **The mass decision**: frame P says more mass keeps paying to 60,000 dots; the family's own live
  A/B pairs say otherwise. The metric makes every A/B pair a model-free measurement: removing a set
  of dots raises a fixed field's score iff their mean credit is below the bar, and the lineage's two
  prunes removed dots at ≤ 0.052–0.055 — while frame P scores those same strata at 0.029–0.089.
  Frame P is therefore 2–3× more generous than the hidden set, and the shipped mass (42,000) sits
  inside the band where the lineage was still gaining live. Flagged as IR-46-03.

### 9.4 The 0.2778 question

`docs/research/why_the_family_top_file_scored_highest.md` answers it end to end: the provenance of
the number (the file is UNSCORED; 0.2778 belongs to a third-party leaderboard row — IR-44-03 / IR-44-10),
the additive budget that makes every dot cost exactly 0.2 of denominator, the stratum decomposition
showing both live prunes removed exactly the below-bar mass, the 6/6 and 21-artifact instrument
calibration, the emission-geometry theory, and the three routes to beat it with what is measurable
here. Five ranked hypotheses live in `registry/twostage/hypotheses.json`: two are validated
(flank excision at the metric's break-even; two-stage gate × curvature placement), one is a
free emission rule already implemented, and two are recorded as **DATA-BLOCKED with the failing
HTTP codes** (USGS ComCat microseismicity — HTTP 000 from this sandbox; 3DEP lidar blocks — HTTP 000,
and the tile-footprint hypothesis was still tested and rejected on measurement).

**Flagged, not hidden**: IR-46-01 (this session's first 24-feature field lost 4/4 folds to the
incumbent at matched mass and is kept in the sweep), IR-46-02 (the sweep's docstring still states a
frame-N promotion gate that nothing passes; frame P decides), IR-46-03 (mass choice, above).
No organizer score exists for anything in this repository; the artifact is a validated candidate.
