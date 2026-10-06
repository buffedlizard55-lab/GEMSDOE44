"""Greedy dot emission with an exclusion radius - the DTI-optimal layout.

Rationale (from the official metric):
  * TPw saturates at 1 per truth pixel (max rule), so a second dot on the same
    truth pixel is pure FP cost. 0/1 dots are the optimal emission shape.
  * The optimal layout of a fixed dot budget is therefore a set of well-
    separated dots placed where the score field is highest (greedy best-first
    under a minimum-spacing constraint).
"""
from __future__ import annotations

import numpy as np


def greedy_dots(score: np.ndarray, budget: int, exclusion: int = 2,
                topk_factor: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Select `budget` pixel locations from a score field.

    Greedy best-first: take the top (budget * topk_factor) candidates by
    score, accept a candidate when no already-accepted dot lies within
    Chebyshev distance `exclusion`.  If the candidate window is exhausted
    before the budget is met (happens on spatially smooth / block-constant
    fields where the top-k cluster is suppressed against itself), the
    window is doubled and retried until the budget is met or all valid
    cells have been tried.

    Returns (rows, cols) of accepted dots.
    """
    s = score.ravel()
    n_valid = int((s > 0).sum())
    if n_valid == 0 or budget <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    H, W = score.shape
    rad = exclusion
    k = min(n_valid, budget * topk_factor)
    while True:
        # top-k by argpartition WITHOUT negating the 49 MB field:
        # partition at n-k and take the k elements above it (unsorted)
        part = np.argpartition(s, s.size - k)[s.size - k:]
        part = part[np.argsort(-s[part])]
        rows = (part // W).astype(int)
        cols = (part % W).astype(int)
        suppressed = np.zeros(score.shape, bool)
        acc_r: list[int] = []
        acc_c: list[int] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            if suppressed[r, c]:
                continue
            acc_r.append(r)
            acc_c.append(c)
            r0, r1 = max(r - rad, 0), min(r + rad + 1, H)
            c0, c1 = max(c - rad, 0), min(c + rad + 1, W)
            suppressed[r0:r1, c0:c1] = True
            if len(acc_r) >= budget:
                break
        if len(acc_r) >= budget or k >= n_valid:
            break
        k = min(n_valid, k * 2)
    return (np.asarray(acc_r, dtype=int), np.asarray(acc_c, dtype=int))


def scatter_positions(rows: np.ndarray, cols: np.ndarray, shape,
                      sigma_px: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Mirror model: scatter dot positions with a 2-D Gaussian and snap to grid."""
    r = rows.astype(float) + rng.normal(0.0, sigma_px, rows.size)
    c = cols.astype(float) + rng.normal(0.0, sigma_px, cols.size)
    r = np.clip(np.round(r).astype(int), 0, shape[0] - 1)
    c = np.clip(np.round(c).astype(int), 0, shape[1] - 1)
    return r, c
