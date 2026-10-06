# Session-2 Verification Log (2026-10-06): line-by-line audit + H44-6 build

## 1. Environment facts (measured, not assumed)
- Sandbox shell has **no network** (`curl` → HTTP 000), no numpy/scipy/rasterio/GDAL/PIL, `pip install` blocked (PEP 668 + offline). Python 3.11 stdlib only.
- Web verification performed through the fetch tool (13 pages fetched, each cited with read date).
- `data/` absent; `.gitignore` excludes `data/raw/`; single-commit history contains no data → session-1 pipeline irreproducible here (IR-44-03).

## 2. Byte-level verification of the committed 40k primary (stdlib TIFF parser + zlib)
- `docs/downloads/...-507289a8-zeros.tif`: 3292×3730, 12,279,160/12,279,160 finite, min 0.0 / max 1.0, 0 outside [0,1], exactly 40,000 pixels == 1.0, no nodata tag, pixel scale 100m, tiepoint (243350, 4508550), GeoKeys ProjectedCSType=32611 / unit metre / "WGS 84 / UTM zone 11N". SHA-256 `c36d8fea…` matches its receipt.
- NaN twin: 7,111,787 NaN / 5,167,373 finite — footprint identical to GEMSDOE32's receipt (5,167,373). All 40,000 dots inside footprint.
- Spacing: global min nearest-neighbour 2.8284 px; **0 dots < 2.8 px** (the ≥280m claim holds file-wide, stronger than stated); 11,716 dots in [2.8, 3.0) px.
- Zip: single member, inner bytes identical to the TIF (sha match).

## 3. Official-source verification (all fetched 2026-10-06)
- Problem/metric page 967 (2 chunks): DTI α=0.2 β=0.8 R=300m triangular kernel; convention "outside bounds null or nan"; worked example 3.00/1.89/2.00→0.60 reproduced in `run_checks.py` (0.60265).
- Live leaderboard: #1 alexoktaba **0.3345** (23 subs), #2 nchuzhoy 0.3262, #5 DARD 0.3195, #13 0.2778, #16 0.2708, #20 0.2600. Charter target 0.3195 stale (IR-44-02).
- Rules PDF §Preface–§3.2: single 100m GeoTIFF, **3 submissions/week**, AI-use disclosure, finalists' reproducibility, two-round scoring.
- Reference solution: exists, John Lipor (Portland State), U-Net notebook, commit aebe92f 2026-06-16.
- GeoDAWN: Glen & Earney 2024, DOI 10.5066/P93LGLVQ, 149,030 line-km / 51,857 km².
- GDR 1391: DOI 10.15121/1881483, CC-BY-4.0; 2m probes, Qfaults v1/v2, geodetics, wells/springs downloadable (validates H44-3 data-availability premise).
- 3DEP: live; "free of charge and without use restrictions".
- USGS faults: `usgs.github.io/faults/` is **404** → replaced with verified `usgs.gov/programs/earthquake-hazards/faults` (IR-44-10).

## 4. Sibling-site verification (all fetched 2026-10-06)
- GEMSDOE32: audit receipt confirms 37,654 dots, UNSCORED, projection 0.2747; "NO ORGANISER SCORE EXISTS". Charter's "0.2778 scored" premise corrected (IR-44-01).
- GEMSDOE40: H45 Euler arm real (43,038 dots, SI=0, ~400m lattice); 4-score calibration (forward model G≈5667, 0.0890/mass, ±2%) reused for target math: 0.3345 @ 40k needs ≈0.1048/dot (+18%).
- GEMSDOE41: H42 basin-margin×LiDAR method real (40k px, 400m packing, holdout 0.251 vs 0.121 control); "0.2778 remains user-reported without an organizer receipt".

## 5. H44-6 build (new unique submission, deterministic, stdlib-only)
- Rule: greedy maximal independent set at 3.0 px on the 40k anchor, most-isolated-first, ties (row, col). Rationale: DTI takes MAX over each 300m kernel per truth pixel → a second dot inside a covered kernel earns ~0 and pays 0.2.
- Result: 34,546 kept / 5,454 pruned; output min NN exactly 3.0000 px; Jaccard vs parent 0.8637 (<0.94 bar); count 34,546 in no known prior receipt.
- Files: `-zeros.tif` (136,270 B, sha `bc16bea3e530…`), `-nan.tif` (166,625 B, same footprint), `-zeros.zip` (single member), `-audit.json` (145-char note).
- Determinism: re-run reproduces identical sha256. Round-trip audit: 12,279,160 finite [0,1], CRS/grid exact.
- Evidence class: decision-theoretic/geometric UNSCORED candidate; no score claimed or projected.

## 6. Bugs found and fixed by the new gate
- Previous-session portal note measured 206 chars (>200 convention) → shortened to 144 chars; receipt updated; TIF bytes untouched (sha unchanged) (IR-44-07).
- Leaderboard snapshot typo caught in review (mzoorob submissions 9→25).
- `scripts/run_checks.py`: 50 checks, all passing (metric, credit bars at 5 levels, per-file audits, receipts, zips, spacing, subset/Jaccard, count-uniqueness).

## 7. Remaining limitations / next session
- No competition data in checkout → no training, no holdout validation, no off-catalogue measurement. Single remaining blocker unchanged: place DrivenData downloads in `data/` on an equipped machine.
- Cross-repo byte-level Jaccard vs priors unmeasured (no binary downloads in sandbox); uniqueness rests on count + parent-Jaccard + distinct construction.
- Weekly-slot decision (40k vs H44-6 vs H44-2 relay build) needs data-grounded validation; H44-2/H44-3 are the queued physics hypotheses with verified-free data sources.
