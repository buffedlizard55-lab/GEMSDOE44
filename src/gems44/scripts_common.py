"""Shared helpers for the GEMS44 scripts: band loading, feature blocks, model training.

Kept deliberately simple and deterministic.  Nothing here reads the competition labels as
a *target* except through the caller's chosen frame; the off-catalogue frame is the one
used for the shipped model.
"""

from __future__ import annotations

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

from . import field as F
from . import grid as G

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
ROW_BLOCK = 512


def load_bands(path: str = "data/raw/training_features.tif") -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    with rasterio.open(path) as src:
        for name in NEEDED:
            bands[name] = src.read(BAND_INDEX[name]).astype(np.float32)
    return bands


def feature_block(bands: dict[str, np.ndarray], r0: int, r1: int) -> np.ndarray:
    cols = []
    for feat_name, band_name, scale in F.FEATURE_SPEC:
        a = bands[band_name][r0:r1].astype(np.float32).copy()
        a[a <= -1e37] = np.nan
        finite = np.isfinite(a)
        med = float(np.median(a[finite])) if finite.any() else 0.0
        filled = np.where(finite, a, med)
        if scale == 0:
            cols.append(filled)
        elif band_name == "detrended_elevation" and scale in (3, 9):
            cols.append(np.abs(ndimage.laplace(ndimage.uniform_filter(filled, size=scale))))
        else:
            cols.append(ndimage.uniform_filter(filled, size=scale))
    return np.stack(cols, axis=-1)


def sample_rows(mask: np.ndarray, n: int, rng) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty(0, np.int64)
    take = rng.choice(rows.size, size=min(n, rows.size), replace=False)
    return rows[take].astype(np.int64) * mask.shape[1] + cols[take].astype(np.int64)


def matrix_at(bands: dict[str, np.ndarray], width: int, idx: np.ndarray) -> np.ndarray:
    """Design matrix for flat indices ``idx``, built one row-block at a time."""
    if idx.size == 0:
        return np.empty((0, len(F.FEATURE_SPEC)), np.float32)
    idx = np.sort(idx)
    out = []
    block = ROW_BLOCK * width
    bounds = np.searchsorted(idx // block, np.arange(idx[-1] // block + 2))
    for b in range(len(bounds) - 1):
        sel = idx[bounds[b]: bounds[b + 1]]
        if sel.size == 0:
            continue
        rows = sel // width
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        fb = feature_block(bands, r0, r1)
        local = (rows - r0) * width + (sel % width)
        out.append(fb.reshape(-1, fb.shape[-1])[local])
    return np.vstack(out)


def train_model(bands: dict[str, np.ndarray], grid, pos_idx: np.ndarray, neg_idx: np.ndarray):
    width = grid.width
    Xp = matrix_at(bands, width, pos_idx)
    Xn = matrix_at(bands, width, neg_idx)
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp), np.int8), np.zeros(len(Xn), np.int8)]
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0, early_stopping=False,
    )
    model.fit(X, y)
    report = {
        "n_pos": int(len(Xp)), "n_neg": int(len(Xn)),
        "n_features": int(X.shape[1]),
        "feature_names": F.feature_names(),
        "insample_auc": float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                y, model.predict_proba(X)[:, 1]
            )
        ),
    }
    return model, report


def predict_field(model, bands: dict[str, np.ndarray], grid) -> np.ndarray:
    out = np.zeros((grid.height, grid.width), dtype=np.float32)
    for r0 in range(0, grid.height, ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, grid.height)
        fb = feature_block(bands, r0, r1)
        sc = model.predict_proba(fb.reshape(-1, fb.shape[-1]))[:, 1]
        out[r0:r1] = sc.reshape(r1 - r0, grid.width)
    return out
