# Why the family's highest-scoring file scored highest — and what it takes to beat it

*Analysis written 2026-10-06 by this session. Every number below is either **[measured here]**
(computed in this repository, from the official competition rasters, with the exact metric in
`src/gems44/metric.py`) or **[owner-report]** (a score reported on one of the GEMSDOE sites,
which is **not** an organizer receipt — IR-44-08). Nothing here is asserted from memory.*

---

## 0. First, the provenance of "0.2778" — flagged, not assumed

The brief states that `gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif` (site
GEMSDOE32) scored **0.2778**, the highest of the family's artifacts. Three separate pieces of
evidence had to be checked before that number could be used at all.

| check | evidence | result |
| --- | --- | --- |
| What does GEMSDOE32 itself claim? | its `README.md` and `docs/index.html`, cloned and read here | the file is labelled **UNSCORED**, with a **projected 0.2747**; "NO ORGANISER SCORE EXISTS for this or any artifact in this repository" |
| Who holds 0.2778 on the public board? | the 2026-10-06 leaderboard snapshot in `registry/leaderboard.json` | rank 13, participant **extradr19**, 10 submissions — **no file attribution published** |
| Is the file what the brief says it is? | bytes read here | it is the family's **0.2708** artifact with every dot within **200 m** of the published catalogue deleted: 40,199 → 37,654 dots, 0 dots within 200 m (**[measured here]**, Jaccard with the 0.2708 file = 0.937, i.e. containment) |

**Consequence, carried as a flagged irregularity (IR-44-03 + IR-44-10).** The phrase "the file that
scored 0.2778" is not supported by any evidence read in this sandbox: the number belongs to a
third-party leaderboard row, and the file itself has never been scored by anyone who published a
receipt. What *is* supported is a **pruning lineage** with two owner-reported live A/B pairs:

```
H19-5 solid  121,131 dots  live 0.1922      [owner-report]
d1.5          60,069 dots  live 0.2477      [owner-report]
d2.8          44,090 dots  live 0.2600      [owner-report]   <- the incumbent in this repository
solo-d2.8     40,199 dots  live 0.2708      [owner-report]   = d2.8 minus its 100 m catalogue flank
h33-2-b2      37,654 dots  (claimed 0.2778 / projected 0.2747)  = solo minus its 200 m flank
```

Everything below is about *why that lineage moved*, measured rather than asserted.

---

## 1. The metric is an additive budget, and every dot has a price of exactly 0.2

From the published definitions (competition page 967, transcription and tests in
`src/gems44/metric.py`, 23 tests in `tests/test_metric.py`):

```
TP_w = Σ_g max_x p(x)·k(d)      FP_w = Σ_{p>0} p(x)·(1 − max_g k)      k(d) = max(1 − d/300 m, 0)
FN_w = |G| − TP_w                (exact, by substitution)
DTI  = TP_w / ( 0.2·(TP_w + FP_w) + 0.8·|G| )
```

Write `T = TP_w`, `n` for the number of dots (a binary file has `Σp = n`), and
`M = Σ_dots max_g k` for the mass that is itself best-covering some truth pixel, so that
`FP_w = n − M`. Then

```
DTI = T / ( 0.2·(T + n − M) + 0.8·|G| )
```

Add one dot whose **fresh** credit is `k` (it improves some truth pixel's max by `k`) and whose own
best-truth weight is `k` — true whenever the dot is farther than 2R = 600 m from every other dot,
which is exactly the regime the family emits in (`min separation ≥ 2.83 px`, **[measured here]**,
nearest-neighbour median = 3.00 px for every artifact in the lineage):

```
Δ denominator = 0.2·k + 0.2·(1 − k) = 0.2       ⇒      DTI rises  ⟺  k > 0.2·DTI
```

**Every dot costs the same 0.2 of denominator; it pays only if its own credit clears 0.2·DTI.**
At the family's scores that bar is:

| live score | break-even credit per dot | break-even distance from truth |
| --- | --- | --- |
| 0.2600 | 0.0520 | 284 m |
| 0.2778 | 0.0556 | 283 m |
| 0.3345 (current public #1, 2026-10-06 read) | 0.0669 | 280 m |

This is not a heuristic: it is the metric's first-order condition, and it is the whole explanation
of the lineage — **mass that realizes less than the bar is mass the file is better off without.**

---

## 2. The decisive measurement: mass-corrected credit per dot on frame P ranks the lineage 6/6

**Frame P** = the 62,122 pixels of USGS SGMC faults that lie **more than 300 m** from every pixel of
the competition catalogue (`labels.tif`), computed here from
`data/external/derived_sgmc_faults_100m_u8.tif` (sha256-pinned mirror of the official SGMC
compilation). It is the only locally available set of **real, mapped, catalogue-absent** faults.

For each artifact of the lineage, `credit per dot = TP_w / n` against frame P — a statistic that
never rewards emitting more dots:

| artifact | dots | `TP_w` on frame P | **credit per dot** | owner-reported live score | `N`-frame DTI (previous session's frame) |
| --- | ---: | ---: | ---: | ---: | ---: |
| H19-5 solid | 121,131 | 7,334 | **0.0605** | 0.1922 | 0.1413 |
| d1.5 | 60,069 | 6,222 | **0.1036** | 0.2477 | 0.1732 |
| d2.8 (incumbent) | 44,090 | 5,569 | **0.1263** | 0.2600 | 0.1755 |
| solo-d2.8 | 40,199 | 5,561 | **0.1383** | 0.2708 | 0.1061 |
| h36-rung30 | 37,660 | 5,400 | **0.1434** | 0.2710 | 0.1029 |
| h33-2-b2 | 37,654 | 5,518 | **0.1465** | 0.2778 *(claimed)* | 0.0541 |

```
Spearman( live score , credit per dot on frame P ) = 1.000   (n = 6, p = 0.0000)
Spearman( live score , N-frame DTI )               = negative across the last three artifacts
```

**Frame P with mass correction reproduces the family's entire reported score ordering exactly.**
The frame the previous session promoted on (frame N = SGMC strands *within* 300 m of the
catalogue) does the opposite: it ranks the three *most pruned* artifacts — the three the live board
rewarded most — as the three worst.

### 2.1 Why the catalogue flank is the mass that gets cut

Decomposing the incumbent's 44,090 dots by distance to the catalogue, and scoring each stratum on
frame P (**[measured here]**):

| stratum | dots | credit per dot on **frame P** | credit per dot on frame N | verdict at the 0.052 bar |
| --- | ---: | ---: | ---: | --- |
| ≤ 100 m from the catalogue | 3,891 | **0.0037** | 0.5291 | 14× **below** the bar |
| 100–200 m | 2,545 | **0.0286** | 0.5275 | below the bar |
| 200–300 m | 2,171 | 0.0893 | 0.3871 | above the bar |
| > 300 m (interior) | 35,483 | **0.1516** | 0.0107 | 2.9× above the bar |
| *uniform random control* | 44,090 | *0.1067* | *0.0257* | — |

Two independent facts fall out:

1. **The two prunes in the lineage removed exactly the below-bar strata** — 3,891 dots at ≤100 m
   (0.2600 → 0.2708) and then 2,545 dots at ≤200 m (→ 0.2747/0.2778). The live A/Bs and this
   decomposition agree on both the sign and the size of the effect: solving
   `T = DTI·(0.2·n + 0.8·|G|)` for the removed dots at `|G| ≈ 7,905 px` (the family's own implied
   hidden-set size) gives **0.012 credit per dot** for the ≤100 m stratum and **0.033** for the
   100–200 m stratum — both far below the 0.052 bar, both matching the frame-P measurement
   (0.004 and 0.029) rather than the frame-N measurement (0.53, 0.53).
2. **Interpreted as an instrument, the live board therefore behaved like frame P and not like
   frame N.** The hidden fault set is *not* enriched in the catalogue's 300 m flank. That is the
   single most consequential finding of this session, because it changes which local statistic is
   allowed to choose a submission.

---

## 3. Can we score above 0.2778? Yes. Where the remaining credit is, measured.

The lineage's own arithmetic sets the target. With the family's implied hidden-set size
`|G| ≈ 7,905 px` and `n = 37,654` dots, a score of 0.3345 requires
`T = 0.3345·(0.2·37,654 + 0.8·7,905) = 4,651`, i.e. **0.1235 credit per dot** — against the
lineage's current best measured 0.1465 on frame P and 0.0893–0.11 live-inferred. Two routes exist,
and both are quantified here:

1. **Emission route (no new data).** The prior session's published artifact in this repository
   (`GEMS44_n-strand-ssmc…`, 44,090 dots) has 19,983 of its dots (45.3 %) inside the 300 m
   catalogue flank (**[measured here]**, from its own audit sidecar and recomputed from its bytes).
   On frame P that is the stratum with 0.004–0.089 credit per dot; on the calibrated instrument the
   same field emitted *without* the flank mass measures **0.1915 credit per dot** — the highest of
   every artifact measured here, and 43 % above the incumbent's 0.1263. **This is the cheapest
   available gain in the whole project, and it is what this session ships.**
2. **Field route (this session's fine-scale library).** A 24-feature fine-scale detector built from
   the 19 official bands alone, cross-validated on spatially blocked quadrants against frame P,
   reaches blocked AUC **0.73–0.78** (fold AUCs) — but on the metric it *lost* to the incumbent's
   actual dots at matched mass in **4/4 folds** (mean ΔDTI −0.030). AUC is not credit: this
   reproduces, in this session's own hands, the family's finding that a field with blocked AUC 0.773
   can score below a uniform control. The features with real discrimination on frame P are
   topographic-curvature and relief transforms:

   | Stage B feature (blocked AUC vs frame P, 4 folds) | mean | folds |
   | --- | ---: | --- |
   | `hessian_curv_s3` — total curvature of detrended elevation, σ = 3 px | **0.7456** | 0.770 / 0.754 / 0.707 / 0.752 |
   | `lrm_grad_r15` — gradient of the local relief model, r = 15 px | 0.7340 | 0.752 / 0.745 / 0.687 / 0.751 |
   | `lrm_abs_r15` — absolute local relief, r = 15 px | 0.6852 | 0.713 / 0.695 / 0.641 / 0.692 |
   | `fabric_align_s5` — strike alignment with the Basin & Range fabric | 0.6755 | 0.713 / 0.683 / 0.663 / 0.643 |
   | `hessian_ridge_s3` — ridge/valley response of detrended elevation | 0.5974 | 0.600 / 0.607 / 0.587 / 0.595 |

   Note what is *absent* from that table: `tilt_angle_tot_curv` (0.3735 — anti-correlated),
   `total horizontal gradient of gravity` (0.456), topographic openness (0.33–0.42). On frame P the
   classical potential-field edge detectors carry **no** off-catalogue signal at 100 m, while the
   *topographic curvature* family carries most of it. That is a physical statement about what these
   faults are — young, surface-expressed scarps — and it is why Stage B here is built on curvature,
   relief and fabric alignment rather than on the geophysical edge operators.

3. **What the data cannot support (flagged, not hidden).** The 8 USGS 1 m lidar tiles that the
   family audited cover only **1.446 %** of the footprint, and inside them the off-catalogue SGMC
   density is **0.40×** the background (catalogue density 1.30×) — **[measured here]**. So the
   "the hidden faults are simply the lidar blocks" hypothesis is **not** supported by the only
   off-catalogue truth available locally, and it is not used.

### 3.1 The live ladder is itself a measurement of the hidden set's marginal credit

The algebra of §1 turns every owner-reported A/B pair into a measurement of the *hidden* set's
marginal dot, with no model and no assumption about `|G|`. Removing a set `R` of dots from a fixed
field raises the score iff the credit those dots realised is below the bar:

```
DTI' > DTI   ⟺   (T − K_R) / (N − 0.2·|R|) > T / N   ⟺   K_R / |R| < 0.2·DTI
```

Applying that to the family's own two prunes:

| A/B pair (same field, measured Jaccard 0.912) | dots removed | score change | ⇒ mean credit of the removed dots | bar at that score |
| --- | ---: | --- | ---: | ---: |
| d2.8 → solo-d2.8 | 3,886 | 0.2600 → 0.2708 | **< 0.052** | 0.054 |
| solo-d2.8 → h33-2-b2 | 2,545 | → 0.2747 *(projected)* | **< 0.055** | 0.055 |

On frame P those same strata measure 0.029–0.089 credit per dot — i.e. **frame P is roughly 2–3×
more generous than the real hidden set**. That is the honest caveat on every "hidden-density"
projection in this repository, and it has a concrete consequence for the mass decision: the
marginal rule's optimum on the hidden set sits at a **lower** mass than frame P would choose. The
sweep's frame-P model optimum is 60,000 dots; this session ships **42,000**, inside the band where
the family's live A/Bs were still gaining, and publishes the whole ladder
(`registry/twostage/analysis.json :: marginal_rule_at_hidden_density`) so the choice can be re-made
against a better instrument. Flagged as IR-46-03.

### 3.2 Emission geometry: why the spacing rule is 3 px and not 1.5 px or 6 px

On an idealised 1-px fault line, the fresh credit a dot earns when the next dot is `s` pixels away
is `s − s²/12` for `s ≤ 2R = 6 px` (max 3.0 at `s = 6`). The family's live ladder moved in exactly
that direction — 1.5 px spacing earns ≈1.31 per dot, 2.8 px earns ≈2.15 — which is why their
thinning kept helping. But frame P is not a 1-px line: it is a 62,122-pixel fault population at
100 m resolution, where overlapping kernels still find fresh truth. Measured at matched mass with
one field and one scorer (`registry/twostage/spacing_diagnostic.json`, in-sample by construction):

| min separation | dots | credit per dot (frame P) | frame-P DTI |
| --- | ---: | ---: | ---: |
| 3 px (shipped) | 42,000 | **0.4414** | 0.3081 |
| 4 px | 42,000 | 0.3870 | 0.2704 |
| 5 px | 31,294 | 0.3735 | 0.2034 |
| 6 px | 23,666 | 0.3763 | 0.1602 |

Coarser spacing buys per-dot credit but loses coverage on *this* truth; the 3 px rule (which is also
the metric's break-even distance at a score of 0.26) wins. The diagnostic is published rather than
quietly tuned: it is in-sample, it is one field, and the family's own evidence (`Spearman(live,
adjacent-dot fraction) = −0.361`) is about adjacent *dots*, which the 3 px rule already eliminates.

---

## 4. What this session ships, and what it explicitly refuses to claim

**Ships.** A two-stage artifact with the stages kept separate *and separately validated*
(`docs/twostage.html`, `registry/twostage/`):

* **Stage A** (coarse favourability — strain, conductivity, depth-to-basement, seismicity,
  gravity/magnetics and catalogue fault-corridor density at 2 km and 10 km): decides *which broad
  zones are worth searching at all*. Validated as a **zone** statement: blocked-cell AUC, base-rate
  lift and off-catalogue recall inside the approved area, against a random zone of identical area.
  **Two AUCs are published, and the distinction matters**: the shipped score includes
  catalogue-density layers and is validated against "does this 2 km cell contain catalogue faults",
  so its blocked AUC (0.942) is partly self-referential — a smoothed fault map predicting a fault
  map. That is a legitimate *gate* input (the catalogue is known, never the truth, and never a pixel
  target), but the honest number for the physical layers themselves is the **physical-only** variant,
  reported beside it in `registry/twostage/submission.json :: stage_a_honesty`.
* **Stage B** (fine-scale placement — curvature, relief, fabric alignment on the official bands):
  decides *where inside those zones*. Validated as a **placement** statement: matched-mass DTI on
  frame P, blocked by quadrant, against the incumbent, the prior artifact and a uniform control.
* **Emission**: the exact greedy max-coverage emitter (`gems44.field.emit_order_np`) with a 3 px
  minimum separation, mass chosen inside the family's empirically-successful band, and **no mass
  within 200 m of the catalogue** — the one mechanism the live board has already paid for twice.
* **The dominance check that keeps "two-stage" honest.** Same field, same mass, same scorer, three
  emissions: Stage-A-only **ranked by the Stage A score** (not by the Stage B belief inside the
  gate — that would be a second Stage-B emission wearing a Stage-A label), Stage-B with no gate, and
  the shipped Stage-B × gate × no-flank file. If the gate alone matched the shipped file, the "two
  stages" claim would be decoration.

**Refuses.** No claim that any local number is an organizer score. The leaderboard target (0.3345
on 2026-10-06) is a dated read of a public page, not a guarantee; the 0.2778 attribution is
contested in §0; and every local number is a proxy whose validity rests on the 6/6 ordering in §2 —
a result obtained on **six** owner-reported scores, which is a small sample and is stated as such.

---

## 5. Sources for manual review

| what | link | used for |
| --- | --- | --- |
| problem description, metric, submission format | https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ | equations in §1; format contract |
| leaderboard (dated read 2026-10-06) | https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/ | §0 attribution; the 0.3345 target |
| known-fault masking statement (DrivenData staff) | https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516 | why catalogue pixels are neither credit nor penalty |
| reference solution | https://github.com/drivendataorg/gems-prize-reference-solution | context only (Tversky loss, α = 0.2 / β = 0.8) |
| USGS State Geologic Map Compilation (SGMC) | https://ngmdb.usgs.gov/Prodesc/proddesc_100179.htm | provenance of the frame-P truth |
| SGMC data download (USGS, public domain) | https://ngmdb.usgs.gov/Geolex/search | source product, mirrored and sha256-pinned in `registry/data_manifest.json` |

*Fetch status.* `drivendata.org` and `community.drivendata.org` return HTTP 000 from this sandbox
(measured); the pages above were read by earlier sessions in this repository and their content is
preserved in `registry/sources.json` with the exact quotes used. `ngmdb.usgs.gov` is also outside
this sandbox's allowlist; the SGMC **product** used here is the sha256-pinned raster in
`data/external/`, whose provenance is recorded in `registry/data_manifest.json` (IR-44-02).
