"""Live-anchored truth (mirror) model.

The hidden test set is, by construction, faults NOT in the visible catalogue.
Catalogue-based validation cannot see it. The group's own inference
(GEMSDOE25 H28, owner-report-derived): the hidden set is scattered around
structure surfaces at sigma ~ 1.85 px. We implement our own independent
version: scatter the catalogue with 2-D Gaussian noise (several sigmas),
score candidate files against the scattered copies with the official metric.

This is a MODEL, not a score. Results are reported as mean +/- se over
paired seeds, and the sigma is swept to show sensitivity.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .emission import scatter_positions
from .metric import score_components


def mirror_score(prediction: np.ndarray, catalogue: np.ndarray,
                 foot: np.ndarray, sigmas=C.MIRROR_SIGMA_PX,
                 n_seeds: int = C.MIRROR_N_SEEDS, seed0: int = C.SEED):
    H, W = catalogue.shape
    cat_rows, cat_cols = np.nonzero(catalogue > 0)
    out = {}
    for s in sigmas:
        dtis = []
        for k in range(n_seeds):
            rng = np.random.default_rng(seed0 + 1000 * int(s * 100) + k)
            r, c = scatter_positions(cat_rows, cat_cols, (H, W), s, rng)
            T = np.zeros((H, W), np.float32)
            T[r, c] = 1.0
            dtis.append(score_components(prediction, T, foot)["DTI"])
        dtis = np.array(dtis)
        out[f"sigma_{s}"] = {"mean": float(dtis.mean()),
                             "se": float(dtis.std(ddof=1) / np.sqrt(len(dtis))),
                             "n": len(dtis),
                             "min": float(dtis.min()), "max": float(dtis.max())}
    return out
