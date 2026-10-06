"""SUPERSEDED - DO NOT RUN.  Kept for provenance only.

This was the first holdout harness.  Its frame ("SGMC absent from the catalogue", i.e. today's
frame P) failed an independent validity test against 19 live-scored artifacts (rho_level -0.054;
registry/frame_ranking.json), and its mass was not density-scaled per fold.  scripts/run_selection.py
replaced it: frames N (primary) and P (secondary), density-matched fold mass, and
scripts/confirm_exact.py re-checks the shipped artifact with the exact 29-offset operator.

Original docstring follows.

Preregistered spatially blocked holdout for the GEMS44 candidate field.

Frames (both 4-quadrant spatially blocked, leave-one-quadrant-out):

  OFF-CATALOGUE (primary)  truth = USGS SGMC fault pixels that are ABSENT from the
      competition catalogue.  This is the only local frame containing real mapped faults
      that the competition's "new faults" resemble: faults a public surface catalogue
      does not carry.  It is used to TRAIN the field and to SCORE the candidate.

  CATALOGUE (sanity only)  truth = a held-out quadrant of the competition catalogue.
      Reported for completeness; it cannot reward off-catalogue skill (this frame is the
      one that provably ranks artifacts in the opposite order to the live board).

Controls, all evaluated on the SAME frames with the SAME metric:
  * incumbent   -- the real, live-scored 0.2600 artifact (dotted-h19-5 d2.8, 44,090 px),
                   fetched from its hash-pinned mirror
  * proximity   -- the standard isotropic prior in this project family: exp(-d_catalogue/1km)
  * uniform     -- random thinning at matched mass

Decision rule, frozen before measurement (see registry/preregistration.json):
  A candidate may spend a submission slot only if it beats the incumbent on the
  OFF-CATALOGUE frame by >= +0.005 pooled, with >= 3 of 4 folds positive, at a mass
  selected leave-one-fold-out.

Writes registry/holdout.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import field as F  # noqa: E402
from gems44 import grid as G  # noqa: E402
from gems44.metric import dti, kernel_offsets  # noqa: E402

BANDS = [
    "magnetic_anomaly", "rtp_magnetic", "tmi_hgrad", "geodetic_2nd_invariant",
    "isostatic_grav_slope", "tilt_angle_tot_curv", "geodetic_shear_rate",
    "geodetic_dilatation_rate", "tmi_vgrad", "dist_to_earthquake", "isostatic_grav_vgrad",
    "detrended_elevation", "isostatic_grav_anomaly", "tmi_total", "depth_to_basement",
    "earthquake_density", "conductivity_surface", "isostatic_grav_hgrad",
    "detrended_elevation_slope",
]
BAND_INDEX = {name: i + 1 for i, name in enumerate(BANDS)}
NEEDED = sorted({b for _, b, _ in F.FEATURE_SPEC})
NAMES = F.feature_names()
MASSES = [4_000, 8_000, 16_000, 24_000, 32_000, 44_090, 60_000]
ROW_BLOCK, PAD = 512, 16
TRAIN_POS = 25_000
TRAIN_NEG = 60_000
CAND_FRACTION = 0.10
RNG = np.random.default_rng(44)

_OFF = kernel_offsets()
_DY = np.array([o[0] for o in _OFF], dtype=np.int64)
_DX = np.array([o[1] for o in _OFF], dtype=np.int64)
_K = np.array([o[2] for o in _OFF], dtype=np.float32)


def emit_order(belief: np.ndarray, candidates: np.ndarray, allowed: np.ndarray, max_dots: int) -> np.ndarray:
    """Exact fixed-order greedy expected-marginal-credit emitter; returns flat indices in
    acceptance order (a nested family, so prefixes are valid sub-emissions)."""
    h, w = belief.shape
    C = np.zeros((h, w), dtype=np.float32)
    b = belief
    order: list[int] = []
    for idx in candidates:
        if len(order) >= max_dots:
            break
        y, x = divmod(int(idx), w)
        if not allowed[y, x]:
            continue
        yy, xx = y + _DY, x + _DX
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy, xx = yy[ok], xx[ok]
        slack = _K[ok] - C[yy, xx]
        np.maximum(slack, 0.0, out=slack)
        gain = float(np.dot(b[yy, xx], slack))
        if gain <= 0.0:
            continue
        order.append(int(idx))
        bits = _K[ok] > C[yy, xx]
        C[yy[bits], xx[bits]] = _K[ok][bits]
    return np.asarray(order, dtype=np.int64)


def raster_from_order(order: np.ndarray, shape: tuple[int, int], mass: int) -> np.ndarray:
    d = np.zeros(shape, dtype=bool)
    sel = order[:mass]
    d.ravel()[sel] = True
    return d


def feature_block(bands: dict[str, np.ndarray], r0: int, r1: int) -> np.ndarray:
    """Design matrix for rows [r0, r1) of the full grid, from cached band arrays."""
    cols = []
    for feat_name, band_name, scale in F.FEATURE_SPEC:
        a = bands[band_name][r0:r1].astype(np.float32)
        a[a <= -1e37] = np.nan
        if scale == 0:
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            cols.append(np.where(np.isfinite(a), a, med))
        elif band_name == "detrended_elevation":
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            filled = np.where(np.isfinite(a), a, med)
            cols.append(np.abs(ndimage.laplace(ndimage.uniform_filter(filled, size=scale))))
        else:
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            filled = np.where(np.isfinite(a), a, med)
            cols.append(ndimage.uniform_filter(filled, size=scale))
    return np.stack(cols, axis=-1)


def load_bands() -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for name in NEEDED:
            bands[name] = src.read(BAND_INDEX[name]).astype(np.float32)
    return bands


def sample_rows(mask: np.ndarray, n: int, rng) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty(0, np.int64)
    take = rng.choice(rows.size, size=min(n, rows.size), replace=False)
    return (rows[take].astype(np.int64) * mask.shape[1] + cols[take].astype(np.int64))


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known

    d_cat = ndimage.distance_transform_edt(~(g.footprint & (labels > 0))).astype(np.float32)
    quad = np.zeros(labels.shape, np.int8)
    quad[: g.height // 2, : g.width // 2] = 0
    quad[: g.height // 2, g.width // 2:] = 1
    quad[g.height // 2:, : g.width // 2] = 2
    quad[g.height // 2:, g.width // 2:] = 3

    inc = None
    p = Path("data/raw/incumbent_d28.tif")
    if p.exists():
        with rasterio.open(p) as src:
            inc = np.isfinite(src.read(1)) & (src.read(1) > 0)

    report: dict = {
        "frames": {"primary": "sgmc_off_catalogue", "sanity": "catalogue_quadrant"},
        "masses": MASSES, "folds": {}, "decision_rule": {
            "beat_incumbent_by": 0.005, "min_positive_folds": 3,
            "mass_selection": "leave-one-fold-out on the off-catalogue frame",
        },
    }
    bands = load_bands()
    print("bands loaded:", len(bands))

    fold_scores: dict[int, dict[int, float]] = {}
    fold_incumbent: dict[int, float] = {}
    fold_baselines: dict[int, dict[str, float]] = {}

    for q in range(4):
        test = quad == q
        train_area = ~test
        truth_all = sgmc & test
        if truth_all.sum() < 200:
            print(f"fold {q}: too few truth px ({int(truth_all.sum())}), skipped")
            continue
        pos_train = sgmc & train_area
        neg_pool = g.footprint & ~g.known & ~sgmc & train_area
        pos_idx = sample_rows(pos_train, TRAIN_POS, RNG)
        neg_idx = sample_rows(neg_pool, TRAIN_NEG, RNG)

        # ---- training matrix, built block-wise then subset
        Xtr, ytr = [], []
        flat_block = ROW_BLOCK * g.width
        for label_val, idx in ((1, pos_idx), (0, neg_idx)):
            order = np.argsort(idx)
            idx = idx[order]
            lo = idx // flat_block
            boundaries = np.searchsorted(lo, np.arange(lo.max() + 2)) if idx.size else np.array([0])
            for b in range(len(boundaries) - 1):
                row_ids = idx[boundaries[b]: boundaries[b + 1]]
                if row_ids.size == 0:
                    continue
                rows = row_ids // g.width
                r0, r1 = int(rows.min()), int(rows.max()) + 1
                fb = feature_block(bands, r0, r1)
                local = (rows - r0) * g.width + (row_ids % g.width)
                Xtr.append(fb.reshape(-1, fb.shape[-1])[local])
                ytr.append(np.full(row_ids.size, label_val, dtype=np.int8))
        X = np.vstack(Xtr)
        y = np.concatenate(ytr)
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0, early_stopping=False,
        )
        model.fit(X, y)
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        del Xtr, X, y

        # ---- predict belief over the test quadrant only
        belief = np.zeros(labels.shape, dtype=np.float32)
        rows_q = np.arange(g.height) if q < 2 else np.arange(g.height // 2, g.height)
        cols_lo, cols_hi = (0, g.width // 2) if q % 2 == 0 else (g.width // 2, g.width)
        r_start, r_end = int(rows_q[0]), int(rows_q[-1]) + 1
        for r0 in range(r_start, r_end, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, r_end)
            fb = feature_block(bands, r0, r1)
            sc = model.predict_proba(fb.reshape(-1, fb.shape[-1]))[:, 1]
            belief[r0:r1, cols_lo:cols_hi] = sc.reshape(r1 - r0, g.width)[:, cols_lo:cols_hi]
            del fb, sc

        allowed = g.footprint & ~g.known & test
        flat = np.where(allowed.ravel(), belief.ravel(), -np.inf)
        n_cand = max(int(allowed.sum() * CAND_FRACTION), 1)
        cand = np.argpartition(-flat, n_cand - 1)[:n_cand]
        cand = cand[np.isfinite(flat[cand])]
        cand = cand[np.argsort(-flat[cand])]
        order = emit_order(belief, cand, allowed, max(MASSES))
        print(f"fold {q}: train AUC(in-sample)={auc:.3f} candidates={cand.size:,} emitted={order.size:,}")

        truth = truth_all
        scores = {}
        for m in MASSES:
            if order.size < m:
                continue
            dots = raster_from_order(order, labels.shape, m)
            scores[m] = dti(dots, truth, valid=g.footprint, known=g.known).score
        fold_scores[q] = scores

        if inc is not None:
            fold_incumbent[q] = dti(inc & test, truth, valid=g.footprint, known=g.known).score

        base = {}
        prox = np.exp(-d_cat / 10.0).astype(np.float32)
        flat_p = np.where(allowed.ravel(), prox.ravel(), -np.inf)
        cp = np.argpartition(-flat_p, n_cand - 1)[:n_cand]
        cp = cp[np.isfinite(flat_p[cp])]
        cp = cp[np.argsort(-flat_p[cp])]
        op = emit_order(prox, cp, allowed, max(MASSES))
        for m in (24_000, 44_090):
            if op.size >= m:
                base[f"proximity@{m}"] = dti(raster_from_order(op, labels.shape, m), truth,
                                             valid=g.footprint, known=g.known).score
        rng = np.random.default_rng(7 + q)
        for m in (24_000, 44_090):
            dd = np.zeros(labels.shape, bool)
            idx = rng.choice(np.nonzero(allowed.ravel())[0], size=m, replace=False)
            dd.ravel()[idx] = True
            base[f"uniform@{m}"] = dti(dd, truth, valid=g.footprint, known=g.known).score
        fold_baselines[q] = base
        report["folds"][str(q)] = {
            "truth_px": int(truth_all.sum()),
            "train_pos": int(pos_idx.size), "train_neg": int(neg_idx.size),
            "train_auc_insample": round(auc, 4),
            "offcatalogue_dti_by_mass": {str(k): round(v, 6) for k, v in scores.items()},
            "incumbent_dti": round(fold_incumbent[q], 6) if inc is not None else None,
            "baselines": {k: round(v, 6) for k, v in base.items()},
        }
        del belief, allowed, flat, cand
        # persist after every fold so a later failure cannot lose completed work
        Path("registry").mkdir(exist_ok=True)
        Path("registry/holdout_partial.json").write_text(
            json.dumps({"folds": report["folds"], "built_utc": "partial"}, indent=2, sort_keys=True) + "\n"
        )

    # ---- leave-one-fold-out mass selection, pooled comparison vs the incumbent
    folds = sorted(fold_scores)
    # only masses that every fold actually reached can take part in selection
    common = [m for m in MASSES if all(m in fold_scores[f] for f in folds)]
    report["masses_common_to_all_folds"] = common
    loo = {}
    for q in folds:
        others = [f for f in folds if f != q]
        means = {m: float(np.mean([fold_scores[f][m] for f in others])) for m in common}
        if not means:
            continue
        best_m = max(means, key=means.get)
        loo[str(q)] = {"chosen_mass": best_m, "heldout_dti": round(fold_scores[q][best_m], 6),
                       "incumbent_heldout_dti": round(fold_incumbent.get(q, float("nan")), 6),
                       "delta": round(fold_scores[q][best_m] - fold_incumbent.get(q, float("nan")), 6)}
    pooled_candidate = float(np.mean([v["heldout_dti"] for v in loo.values()])) if loo else float("nan")
    pooled_incumbent = float(np.mean([v["incumbent_heldout_dti"] for v in loo.values()])) if loo else float("nan")
    n_pos = int(sum(1 for v in loo.values() if v["delta"] > 0))

    report["leave_one_fold_out"] = loo
    report["pooled"] = {
        "candidate_offcatalogue_dti": round(pooled_candidate, 6),
        "incumbent_offcatalogue_dti": round(pooled_incumbent, 6),
        "delta": round(pooled_candidate - pooled_incumbent, 6),
        "positive_folds": n_pos, "n_folds": len(loo),
        "gate_pass": bool((pooled_candidate - pooled_incumbent) >= 0.005 and n_pos >= 3),
    }
    Path("registry").mkdir(exist_ok=True)
    Path("registry/holdout.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== OFF-CATALOGUE frame, per-fold DTI by emitted mass ===")
    header = "fold  truth_px  " + "".join(f"{m:>10d}" for m in MASSES) + f"{'incumbent':>12}"
    print(header)
    for q in folds:
        row = f"{q:>4}  {report['folds'][str(q)]['truth_px']:>8}  "
        row += "".join(f"{fold_scores[q].get(m, float('nan')):>10.4f}" for m in MASSES)
        row += f"{fold_incumbent.get(q, float('nan')):>12.4f}"
        print(row)
    print("\nleave-one-fold-out:", json.dumps(loo, indent=1))
    print("pooled:", json.dumps(report["pooled"], indent=1))
    print("\nbaselines fold0:", json.dumps(fold_baselines.get(0, {}), indent=1))
    print("\nwrote registry/holdout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
