"""GEMSDOE44 configuration.

Grid constants are derived from the official competition rasters and cross-checked
against the published audit receipts of the sibling repositories (GEMSDOE32):
single band, float32, EPSG:32611, 100 m, 3730 x 3292.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
EVIDENCE = REPO / "evidence"
DOCS = REPO / "docs"
DOWNLOADS = DOCS / "downloads"

# Official raster paths (verified sha256 in data_io.verify_official_rasters)
FEATURES_TIF = DATA / "training_features.tif"      # 19 bands, nodata = -3.4028234663852886e+38
CATALOG_TIF = DATA / "existing_faults.tif"         # int8: 1 = catalogue fault, 0 = inside, -1 = outside
EXAMPLE_TIF = DATA / "example_submission.tif"      # format template (all NaN outside, 0 inside)

# External official mirrors (group repos, sha256-pinned upstream)
HEATFLOW_CSV = REPO.parent / "ref" / "GEMSDOE28" / "docs" / "data" / "sb_heat_flow_in_footprint.csv"
SLIP_CSV = REPO.parent / "ref" / "GEMSDOE28" / "docs" / "data" / "sb_slip_tendency_in_footprint.csv"
TOPOLOGY_GEOJSON = REPO.parent / "ref" / "GEMSDOE28" / "docs" / "data" / "topology_links.geojson"

# Grid
ROWS, COLS = 3730, 3292
CELL_M = 100.0
CRS = "EPSG:32611"
ORIGIN_E, ORIGIN_N = 243350.0, 4508550.0  # top-left corner (row 0, col 0)

NODE_DATA_SENTINEL = -3.4028234663852886e+38  # float32 min; used by training_features.tif

# 19 official bands, order verified against GEMSDOE25 evidence/band_semantics.json
# (measured Spearman vs GeoDAWN products; tc vs GeoDAWN radiometric TC = 0.9999)
BAND_NAMES = [
    "mag_anom",          # 1
    "rtp",               # 2
    "tmi_hg",            # 3  horizontal gradient of TMI
    "geod_2ndinv",       # 4  second invariant of strain rate tensor
    "iso_grav_anom_slope",  # 5
    "tc",                # 6  radiometric total counts (GEMSDOE25 measurement)
    "geod_shearrate",    # 7
    "geod_dilaterate",   # 8
    "tmi_vg",            # 9  vertical gradient of TMI
    "deq_n100a15",       # 10 earthquake density
    "iso_grav_anom_vg",  # 11
    "det_elev",          # 12 detrended elevation
    "iso_grav_anom",     # 13
    "tmi",               # 14
    "depth_to_base_surf",# 15
    "ieq_n100a15",       # 16 intermediate-depth earthquake density
    "cond_surf",         # 17 surface conductivity
    "iso_grav_anom_hg",  # 18
    "det_elev_slope",    # 19
]

B = {n: i + 1 for i, n in enumerate(BAND_NAMES)}  # name -> band index (1-based)

# Metric (official, drivendata page 967, read 2026-10-06)
ALPHA = 0.2
BETA = 0.8
R_PX = 3  # 300 m support at 100 m resolution

# Pipeline
SEED = 44
N_FOLDS = 4
# Superpixel (2 km x 2 km blocks) for the coarse Stage-1 gate
SUPERPX = 20
# Emission
EXCLUSION_PX = 2     # 200 m minimum spacing between emitted dots
FLANK_B_PX = 2       # catalogue-flank removal radius (200 m); metric-optimal:
                     # hidden truth is off-catalogue by construction
BUDGET_CANDIDATES = [30000, 35000, 40000, 45000]
DEFAULT_BUDGET = 40000
# Stage-1 gate: keep top-q fraction of superpixels by favourability
GATE_Q = 0.35

# Mirror (live-anchored truth) model
MIRROR_SIGMA_PX = (1.5, 1.85, 2.5)
MIRROR_N_SEEDS = 12

# Incumbent submission dot-sets for the uniqueness check (paths relative to REPO.parent/ref)
INCUMBENTS = [
    ("GEMSDOE32", "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif", "0.2778"),
    ("GEMSDOE28", "gems28-h27-4-r1-solo-d2-8-20261003-8acb75e1f2cc-allfinite.tif", "0.2708"),
    ("GEMSDOE25", "gems25-dotted-h19-5-d2-8-20261002-e56ea318af89-zeros.tif", "0.2600"),
    ("GEMSDOE28", "gems27-topo-gap-closure-t-v2-on-d1-5-20261002-5512495c6bd1-allfinite.tif", "0.2449"),
]
