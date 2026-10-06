# GEMSDOE44 — two-stage fault placement for the DOE GEMS Prize

**Repo:** `buffedlizard55-lab/GEMSDOE44` · **Competition:** [DOE GEMS Prize — DrivenData #306](https://www.drivendata.org/competitions/306/competition-doe-gems/) · **Site:** [GitHub Pages](https://buffedlizard55-lab.github.io/GEMSDOE44/docs/index.html) (auto-published from `docs/`)

## Executive summary

GEMSDOE44 separates the question our previous attempts conflated:

> **"Where is geology favourable?"** (coarse, zone-scale) — vs — **"where the fault pixel actually is"** (fine, pixel-scale).

It ships:

1. **A one-click, format-guaranteed submission GeoTIFF** (single band, float32,
   EPSG:32611, 100 m, 3730×3292, every cell finite, every value in [0, 1],
   no nodata tag — the exact failure modes behind the portal error
   `"Predicted values must be in range [0, 1]"` are audited byte-by-byte and
   re-verified by re-opening the written file).
2. **Stage 1 — a coarse favourability gate** (2 km superpixels; strain,
   seismicity, heat-flow proxies, radiometrics, catalog uncertainty) that
   decides which broad zones are worth searching. Validated *separately* on a
   pre-registered spatially-blocked 4-fold holdout (zone AUC + catalogue
   concentration ratio).
3. **Stage 2 — an independent fine-scale placement model** (cross-physics
   structure-tensor consensus, strike inheritance, relay-zone and
   hydrothermal-plumbing channels) that places well-separated 0/1 dots **only
   inside Stage-1-approved zones**, using the precision/recall structure the
   official distance-weighted Tversky index (DTI, α=0.2, β=0.8, 300 m kernel)
   actually rewards. Trained and holdout-scored on its own, against five arms
   (random / habitat-only / single-field / H1-only / full model) at matched
   dot budgets.
4. **Both stages' holdout performance reported separately** in
   [`docs/holdout.html`](docs/holdout.html) — collapsing them into one number
   is exactly what hid why the habitat-only attempts failed (0.0041, 0.1223,
   0.1352): DTI rewards precise pixel-level correspondence, and a basin-scale
   favourability statement is a true claim answering a coarser question.
5. **Five new, ranked geological hypotheses** ([`docs/hypotheses.html`](docs/hypotheses.html)),
   each naming its layers, physical signature, why it catches faults the
   USGS/INGENIOUS catalogue lacks, and how it differs from everything the
   group has already run (including the registered-but-unbuilt arms H60/H61).
6. **A live-anchored (mirror) truth model** + **uniqueness report** proving the
   emitted dot set is not a reskin of any previously-scored submission.

Everything is reproducible from this checkout with
`/home/user/venv/bin/python scripts/run_all.py` (evidence JSONs land in
`evidence/`). Every number on the site links to the evidence file that
produced it; every external input carries a sha256 pin and an official source
URL in [`docs/sources.html`](docs/sources.html).

## Core values (focal point for all work in this repo)

**Maximize P(Win).** In every decision we weigh tradeoffs, assess risk, and
choose the path that maximizes the probability of winning. A weekly submission
slot is an experiment, not a lottery ticket: nothing is submitted without
passing the spatially-blocked holdout first.

**Own the Outcome.** We own results end to end. Every claim carries an
evidence class (`verified` / `model` / `claim`), defects in our own artifacts
are measured, published and replaced rather than quietly dropped, and failure
is treated as a signal.

## How to submit (30-second version)

1. Download the primary file from the site (or
   [`docs/downloads/`](docs/downloads/)).
2. On [DrivenData → New submission](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/),
   upload the `.tif` (or the `.zip` containing it).
3. Paste the site's **unique submission name** and the **note** (≤200 chars).
4. Submit. The format receipt in `evidence/audit-*.json` documents the
   guarantees the portal checks (CRS, shape, geotransform, [0,1], no NaN,
   no nodata tag).

## Project brief (verbatim prompt — read at the start of every session)

> Review the repo.
>
> THE FOLLOWING IS THE HIGHEST URGENCY AND MUST BE FOLLOWED!
>
> MUST GENERATE A UNIQUE TIF SUBMISSION FOR THE COMPETITION. DO NOT COPY A
> PREVIOUS SUBMISSION UNLESS IT'S FOR LEARNING AND EDUCATION. BUT WE MUST
> GENERATE A UNIQUE TIF SUBMISSION. The submission must be different than the
> collection of gemsdoe sites below.
>
> There should be an easy to download submission tif file as described by the
> prompt. Read the entire prompt.
>
> Separate "where geology is favorable" from "where the fault pixel actually
> is" — and validate each half separately. Your habitat/favorability-themed
> attempts have been the worst performers in the whole set (0.0041, 0.1223,
> 0.1352) despite resting on sound geology, which is itself informative: DTI
> rewards precise pixel-level correspondence, and a basin-scale
> favorability statement is a true claim that answers a different, coarser
> question. Build this as two explicitly separate, separately-validated
> stages instead of one conflated model: a coarse favorability layer (strain,
> heat flow, geochemistry) that only decides which broad zones are worth
> searching at all, and a second, independent fine-scale placement model —
> trained and holdout-scored on its own, using only the precision/recall
> structure DTI actually rewards — that places points only inside the zones
> the first stage approved. Report both stages' holdout performance
> separately in the write-up, since collapsing them back into one number is
> exactly what's been hiding why the habitat-only attempts failed.
> Normalize, write to the required format, and check the final output isn't
> dominated by one stage's footprint alone before calling it unique.
>
> Before implementing, generate 3–5 candidate geological hypotheses we haven't
> tried yet, each naming: the specific layer(s) involved, the physical
> signature being targeted (e.g., an edge-detection or curvature transform),
> why it should catch a fault missing from the USGS/INGENIOUS catalogue rather
> than one already in it, and how it differs from anything already implemented
> in this repo. Rank them by expected DTI improvement and implementation cost.
> Validate the top candidate on our spatially-blocked holdout set before
> touching a weekly submission slot — do not spend a submission slot on an
> idea that hasn't beaten the current holdout best. If a candidate can't be
> validated without new external data, name the specific free, official source
> needed and check it's obtainable before proposing the idea as viable.
>
> Answer the question using PhD-level experience, knowledge, and judgement:
> study GEMSDOE32 (h33-h33-2-b2, 0.2778) — why and how did it get the highest
> (in-group) score, and can we generate a submission that scores higher than
> the public leader (read live each session; 0.3195 was the stated target,
> 0.3345 was live on 2026-10-06)?
>
> The goal of this project is to place at the top of the leaderboard. The
> site must make the submission file obvious on arrival, explain exactly how
> to submit, keep a current feed of official sources with verified links, and
> remove the need to manually check everything. Work line by line verifying
> from official verified trusted sources, provide links for manual review,
> no manual input, flag irregularities, no hallucinations.
>
> Put this prompt into the repo readme and read it every time we work on the
> project as a starting point.

(Competitions context, data pages, rules and the GEMS-96647 PDF: see
[docs/sources.html](docs/sources.html) for the verified link table.)

## Repo layout

```
src/gems44/        metric.py (official DTI, test-verified), data_io, features,
                   stage1, stage2, emission, mirror_model, emit_submission
scripts/run_all.py end-to-end pipeline (evidence/*.json)
tests/             metric unit tests incl. brute-force cross-check
data/              official rasters (sha256-pinned; NOT in git)
evidence/          every number the site displays
docs/              GitHub Pages site + downloads
```

## Next steps / known limitations (carry into the next session)

1. **The proxy ceiling is honest:** catalogue-based holdout cannot reward
   genuinely new faults. The mirror model (scattered-catalogue truth) is a
   model, not a score; its σ is swept and disclosed.
2. **Distance convention:** the official server-side DTI distance lattice
   (Euclidean vs Manhattan) is not public; we implement the published formulae
   with Euclidean pixel distance and flag it.
3. **Unverified external layers:** 1 m DEM tiles (drainage/chi transforms,
   H60-5) and the MCRE 2020 rupture vector (H60-1) are registered but need
   egress this sandbox lacks; CI steps are specced in
   [docs/hypotheses.html](docs/hypotheses.html).
4. **Band 6 ("tc") semantics conflict** flagged: GEMSDOE25 measured Spearman
   0.9999 vs GeoDAWN radiometric TC, while a GEMSDOE32 note quotes "tilt
   angle". We follow the measurement (radiometric counts) and flag the
   conflict in [docs/irregularities.html](docs/irregularities.html).
5. **Leaderboard drift:** re-read the live leaderboard every session; the
   target moves (0.3195 → 0.3262 → 0.3345 across early October 2026).
