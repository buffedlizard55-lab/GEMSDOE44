"""Decisive measurement: is there any spatially-blocked, measurable signal at fault pixels
that the competition catalogue does NOT contain?

Frames (identical sampling, identical features, identical folds):
  * CATALOGUE : catalogue pixels vs random footprint pixels      (the standard sanity frame)
  * OFF-CAT   : SGMC-only pixels vs random footprint pixels       (stand-in for the hidden
                set: real mapped faults that are absent from the given catalogue)

Both frames exclude the pixels of the other frame, both exclude the known-fault mask from
the positive side, and both use the SAME random-negative pool so the comparison is paired.
Reported per frame: univariate AUC of each of the 19 official bands, plus a spatially
blocked cross-validated AUC of a small gradient-boosted model on all bands + geometry.

This is falsifiable: if OFF-CAT AUC is ~0.5 while CATALOGUE AUC is high, then no field built
from these 19 bands can find the hidden faults, and any submission claiming otherwise is noise.
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
from gems44 import grid as G  # noqa: E402

BAND_NAMES = [
    "magnetic_anomaly", "rtp_magnetic", "tmi_hgrad", "geodetic_2nd_invariant",
    "isostatic_grav_slope", "tilt_angle_tot_curv", "geodetic_shear_rate",
    "geodetic_dilatation_rate", "tmi_vgrad", "dist_to_earthquake", "isostatic_grav_vgrad",
    "detrended_elevation", "isostatic_grav_anomaly", "tmi_total", "depth_to_basement",
    "earthquake_density", "conductivity_surface", "isostatic_grav_hgrad",
    "detrended_elevation_slope",
]
N_POS, N_NEG = 4000, 8000
SEED = 44


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc = sgmc & g.footprint & ~g.known

    d_cat = ndimage.distance_transform_edt(~(g.footprint & (labels > 0)))
    footprint = g.footprint

    rng = np.random.default_rng(SEED)
    # a single shared negative pool, then subsample per frame so pairs are comparable
    neg_all = np.nonzero(footprint & (labels <= 0) & ~sgmc)
    take = rng.choice(neg_all[0].size, size=N_NEG, replace=False)
    neg_rows, neg_cols = neg_all[0][take], neg_all[1][take]

    cat = np.nonzero(footprint & (labels > 0))
    take = rng.choice(cat[0].size, size=min(N_POS, cat[0].size), replace=False)
    cat_rows, cat_cols = cat[0][take], cat[1][take]

    off = np.nonzero(sgmc)
    take = rng.choice(off[0].size, size=min(N_POS, off[0].size), replace=False)
    off_rows, off_cols = off[0][take], off[1][take]

    # features: band value at the pixel, and a 3x3 and 9x9 mean of the same band
    feats: dict[str, np.ndarray] = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for i, name in enumerate(BAND_NAMES, start=1):
            b = src.read(i).astype(np.float32)
            b[b <= G.NODATA / 2.0] = np.nan
            bfilled = np.where(np.isfinite(b), b, np.nan)
            med = float(np.nanmedian(bfilled))
            bf = np.where(np.isfinite(b), b, med)
            small = ndimage.uniform_filter(bf, size=3)
            large = ndimage.uniform_filter(bf, size=9)
            for tag, arr in (("", bf), ("_m3", small), ("_m9", large)):
                feats[name + tag] = np.concatenate(
                    [arr[cat_rows, cat_cols], arr[off_rows, off_cols], arr[neg_rows, neg_cols]]
                )
    feats["dist_to_catalogue_px"] = np.concatenate(
        [d_cat[cat_rows, cat_cols], d_cat[off_rows, off_cols], d_cat[neg_rows, neg_cols]]
    )
    row = np.concatenate([cat_rows, off_rows, neg_rows]).astype(np.float32)
    col = np.concatenate([cat_cols, off_cols, neg_cols]).astype(np.float32)
    quad = ((row > g.height / 2).astype(int) * 2 + (col > g.width / 2).astype(int)).astype(int)
    n_cat, n_off = cat_rows.size, off_rows.size
    n_neg = neg_rows.size

    frame = np.array(["cat"] * n_cat + ["off"] * n_off + ["neg"] * n_neg)
    X = np.column_stack([feats[k] for k in feats]).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"design matrix: {X.shape}, frames: cat={n_cat} off={n_off} neg={n_neg}")

    def auroc(y: np.ndarray, s: np.ndarray) -> float:
        return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")

    report: dict = {
        "n_pos_catalogue": int(n_cat), "n_pos_sgmc_only": int(n_off), "n_neg": int(n_neg),
        "seed": SEED, "univariate_auc": {},
    }
    neg_mask = frame == "neg"

    # ---- univariate AUC per band (raw value + multi-scale means), both frames
    for key in feats:
        a_cat = auroc(np.r_[np.ones(n_cat), np.zeros(n_neg)], np.r_[feats[key][:n_cat], feats[key][n_cat + n_off:]])
        a_off = auroc(np.r_[np.ones(n_off), np.zeros(n_neg)], np.r_[feats[key][n_cat:n_cat + n_off], feats[key][n_cat + n_off:]])
        report["univariate_auc"][key] = {
            "catalogue": round(a_cat, 4),
            "sgmc_only": round(a_off, 4),
            "abs_gap": round(abs(a_cat - 0.5) - abs(a_off - 0.5), 4),
        }

    # ---- spatially blocked CV: hold out one quadrant at a time
    blocked = {"catalogue": [], "sgmc_only": []}
    for frame_name, pos_tag in (("catalogue", "cat"), ("sgmc_only", "off")):
        keep = (frame == pos_tag) | neg_mask                 # rows used by this frame only
        Xf, quadf = X[keep], quad[keep]
        yf = (frame[keep] == pos_tag).astype(int)
        for q in range(4):
            te = quadf == q
            tr = ~te
            if yf[te].sum() < 20 or yf[tr].sum() < 20:
                continue
            m = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_depth=4, random_state=0,
                min_samples_leaf=40,
            )
            m.fit(Xf[tr], yf[tr])
            s = m.predict_proba(Xf[te])[:, 1]
            blocked[frame_name].append(round(auroc(yf[te], s), 4))
    report["blocked_quadrant_auc"] = {
        k: {"folds": v, "mean": round(float(np.mean(v)), 4) if v else None} for k, v in blocked.items()
    }

    # ---- distance-conditioned control: do the SAME features improve with proximity?
    report["auc_vs_distance_to_catalogue"] = {}
    d = feats["dist_to_catalogue_px"]
    key = "detrended_elevation_slope"
    n = frame.size
    for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 10 ** 9)]:
        in_band = (d >= lo) & (d < hi)
        m_pos_cat = in_band & (frame == "cat")
        m_pos_off = in_band & (frame == "off")
        m_neg = in_band & (frame == "neg")
        if m_pos_cat.sum() < 30 or m_neg.sum() < 30:
            continue

        def pair(m_pos):
            idx = m_pos | m_neg
            y = m_pos[idx].astype(int)
            s = feats[key][idx]
            return y, s

        y_cat, s_cat = pair(m_pos_cat)
        y_off, s_off = pair(m_pos_off)
        report["auc_vs_distance_to_catalogue"][f"{lo}-{hi}px"] = {
            "n_cat": int(m_pos_cat.sum()), "n_off": int(m_pos_off.sum()), "n_neg": int(m_neg.sum()),
            "auc_cat_detr_slope": round(auroc(y_cat, s_cat), 4),
            "auc_off_detr_slope": round(auroc(y_off, s_off), 4),
        }

    Path("registry").mkdir(exist_ok=True)
    Path("registry/detectability.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== spatially blocked (quadrant) ROC AUC ===")
    for k, v in report["blocked_quadrant_auc"].items():
        print(f"  {k:12s} folds={v['folds']}  mean={v['mean']}")
    print("\n=== strongest univariate AUCs (|AUC-0.5|) ===")
    ranked = sorted(report["univariate_auc"].items(), key=lambda kv: -abs(kv[1]["catalogue"] - 0.5))
    print(f"  {'feature':32s} {'catalogue':>10s} {'sgmc_only':>10s}")
    for k, v in ranked[:14]:
        print(f"  {k:32s} {v['catalogue']:>10.4f} {v['sgmc_only']:>10.4f}")
    print("\nwrote registry/detectability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
