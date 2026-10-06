"""Distance-weighted Tversky index (DTI) - official competition metric.

Implemented verbatim from the published formulae
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/,
read 2026-10-06):

    k(d)      = max(1 - d/R, 0),  R = 3 px (300 m at 100 m resolution)
    TPw       = sum_g  max_{x: d(x,g) <= R} p(x) * k(d(x,g))
    FPw       = sum_{x: p>0} p(x) * (1 - max_g k(d(x,g)))
    FNw       = sum_g  (1 - max_{x: d(x,g) <= R} p(x) * k(d(x,g)))
    DTI       = TPw / (TPw + alpha*FPw + beta*FNw + eps),  alpha=0.2, beta=0.8

Identity used throughout (exact, not an approximation):
    FNw = |G| - TPw
because FNw sums (1 - M_g) over ground-truth pixels with M_g = max p*k.

CONVENTION FLAG (for manual review):
  d(x, g) is Euclidean distance in pixel units. The problem statement says
  "distance to the nearest ground truth pixel ... linear (triangular) kernel
  with 300 m support" but does not state the lattice distance convention.
  For 0/1 dot emissions inside a 3 px kernel the DTI difference between
  Euclidean / Manhattan / Chebyshev is bounded and small; the official
  server-side implementation is not public (the reference solution
  notebook only uses TverskyLoss(alpha=0.2, beta=0.8) for training).
"""
from __future__ import annotations

import numpy as np

from .config import ALPHA, BETA, R_PX

_EPS = 1e-12


def kernel_weights(max_r: int = R_PX) -> tuple[np.ndarray, np.ndarray]:
    """Return (dr, w) for all lattice offsets within Chebyshev radius max_r.

    dr  : (K, 2) int offsets (dr, dc)
    w   : (K,) triangular kernel values k(|dr,dc|_Euclid)
    """
    offs = np.stack(np.meshgrid(np.arange(-max_r, max_r + 1),
                                np.arange(-max_r, max_r + 1),
                                indexing="ij"), axis=-1).reshape(-1, 2)
    dist = np.hypot(offs[:, 0].astype(float), offs[:, 1].astype(float))
    w = np.clip(1.0 - dist / max_r, 0.0, None)
    keep = w > 0
    return offs[keep], w[keep]


def max_kernel_to_truth(truth: np.ndarray, footprint: np.ndarray,
                        max_r: int = R_PX) -> np.ndarray:
    """K(x) = max over truth pixels g of k(d(x, g)), 0 where no truth in R.

    Computed with shifted-mask max over the (2R+1)^2 lattice offsets.
    """
    K = np.zeros_like(truth, dtype=np.float64)
    offs, w = kernel_weights(max_r)
    t = truth.astype(bool)
    H, W = t.shape
    for (dr, dc), wk in zip(offs, w):
        dr, dc = int(dr), int(dc)
        # truth pixel at (r+dr, c+dc) covers x=(r, c); slice without wrap
        r0, r1 = max(0, -dr), min(H, H - dr)
        c0, c1 = max(0, -dc), min(W, W - dc)
        r2, c2 = max(0, dr), max(0, dc)
        sub = t[r0:r1, c0:c1]
        mask = K[r2:r2 + (r1 - r0), c2:c2 + (c1 - c0)]
        upd = np.where(sub, wk, 0.0)
        np.maximum(mask, upd, out=mask)
    K = np.where(footprint, K, 0.0)
    return K


def weighted_true_positive(prediction: np.ndarray, truth: np.ndarray,
                           max_r: int = R_PX) -> float:
    """TPw = sum_g max_{x in N_R(g)} p(x) * k(d(x, g)).

    Sparse-friendly: each predicted pixel updates the truth pixels inside
    its (2R+1)^2 window.
    """
    p = prediction.astype(np.float64)
    rows, cols = np.nonzero(p > 0)
    if rows.size == 0:
        return 0.0
    t = truth.astype(bool)
    # truth pixels only; we accumulate M_g = max p*k over g
    tr_rows, tr_cols = np.nonzero(t)
    M = np.zeros(tr_rows.shape[0], dtype=np.float64)
    offs, w = kernel_weights(max_r)
    H, W = p.shape
    # build truth index lookup (sparse, |G| ~ 6e4); -1 = not a truth pixel
    truth_grid = np.full((H, W), -1, dtype=np.int64)
    truth_grid[tr_rows, tr_cols] = np.arange(tr_rows.size)
    for (dr, dc), wk in zip(offs, w):
        r2 = rows + int(dr)
        c2 = cols + int(dc)
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        if not ok.any():
            continue
        idx = truth_grid[r2[ok], c2[ok]]
        hit = idx >= 0
        if not hit.any():
            continue
        vals = p[rows[ok][hit], cols[ok][hit]] * wk
        np.maximum.at(M, idx[hit], vals)
    tpw = float(M.sum())
    return tpw


def score_components(prediction: np.ndarray, truth: np.ndarray,
                     footprint: np.ndarray, max_r: int = R_PX,
                     K_cache: np.ndarray | None = None,
                     invK_cache: np.ndarray | None = None) -> dict:
    """Full official DTI plus its components.

    prediction : (H, W) float, 0/1 dots (any [0,1] works)
    truth      : (H, W) 0/1 ground truth pixels
    footprint  : (H, W) bool domain mask
    K_cache    : optional precomputed max-kernel-to-truth field (same truth)
    invK_cache : optional precomputed (1 - K) field, same truth (saves a
                 80 MB transient per call when scoring many arms)
    """
    n_truth = int(truth.sum())
    tpw = weighted_true_positive(prediction, truth, max_r)
    fnw = n_truth - tpw
    if K_cache is None:
        K_cache = max_kernel_to_truth(truth, footprint, max_r)
    if invK_cache is None:
        fpw = float((prediction * (1.0 - K_cache)).sum())
    else:
        fpw = float((prediction * invK_cache).sum())
    dti = tpw / (tpw + ALPHA * fpw + BETA * fnw + _EPS)
    return {
        "TPw": tpw,
        "FPw": fpw,
        "FNw": fnw,
        "DTI": float(dti),
        "n_truth": n_truth,
        "n_pred_px": int((prediction > 0).sum()),
        "precision_proxy": tpw / max(fpw + tpw, _EPS),  # informational only
    }


def score(prediction: np.ndarray, truth: np.ndarray,
          footprint: np.ndarray, max_r: int = R_PX) -> float:
    return score_components(prediction, truth, footprint, max_r)["DTI"]
