"""Line-by-line verification of the DTI implementation.

1. Official worked example (drivendata page 967, read 2026-10-06):
   TPw=3.00, FPw=1.89, FNw=2.00  ->  DTI = 0.60
2. Hand-computed micro grids (single truth pixel, known geometry).
3. Identity DTI = TPw / (TPw + 0.2 FPw + 0.8 (|G| - TPw)).
4. Bounds: DTI in [0, 1]; perfect on-truth dots -> DTI = 1; empty -> 0.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems44 import metric44 as M  # noqa: E402


def test_worked_example_components():
    """The published numbers must reproduce the published score."""
    tpw, fpw, fnw = 3.00, 1.89, 2.00
    dti = tpw / (tpw + 0.2 * fpw + 0.8 * fnw + 1e-12)
    assert abs(dti - 0.60) < 0.005  # published: 0.60


def test_single_truth_pixel_exact_hit():
    G = np.zeros((9, 9))
    G[4, 4] = 1.0
    foot = np.ones((9, 9), bool)
    P = np.zeros((9, 9))
    P[4, 4] = 1.0
    s = M.score_components(P, G, foot)
    assert abs(s["TPw"] - 1.0) < 1e-12
    assert abs(s["FNw"]) < 1e-12
    assert abs(s["FPw"]) < 1e-12
    assert abs(s["DTI"] - 1.0) < 1e-9


def test_single_truth_pixel_at_distance_d():
    """One truth pixel, one dot at Euclidean distance d -> DTI = k(d)."""
    for (dr, dc, d, expected_k) in [(0, 1, 1.0, 2.0 / 3.0),
                                    (0, 2, 2.0, 1.0 / 3.0),
                                    (0, 3, 3.0, 0.0),
                                    (1, 1, np.sqrt(2), 1 - np.sqrt(2) / 3)]:
        G = np.zeros((9, 9))
        G[4, 4] = 1.0
        foot = np.ones((9, 9), bool)
        P = np.zeros((9, 9))
        P[4 + dr, 4 + dc] = 1.0
        s = M.score_components(P, G, foot)
        assert abs(s["DTI"] - expected_k) < 1e-9, (dr, dc, s)
        # FN identity
        assert abs(s["FNw"] - (1.0 - s["TPw"])) < 1e-12
        # FP = 1 - k(d) (the dot pays the uncovered kernel share)
        assert abs(s["FPw"] - (1.0 - expected_k)) < 1e-9


def test_saturation_multiple_dots_same_truth():
    """More dots on the same truth pixel must NOT increase TPw (max rule)."""
    G = np.zeros((9, 9))
    G[4, 4] = 1.0
    foot = np.ones((9, 9), bool)
    P = np.zeros((9, 9))
    P[4, 4] = 1.0
    P[4, 5] = 1.0  # second dot at d=1
    s = M.score_components(P, G, foot)
    assert abs(s["TPw"] - 1.0) < 1e-12  # saturated at 1
    assert abs(s["FPw"] - (1.0 - 2.0 / 3.0)) < 1e-9


def test_empty_prediction_is_zero():
    G = np.zeros((9, 9))
    G[4, 4] = 1.0
    foot = np.ones((9, 9), bool)
    P = np.zeros((9, 9))
    s = M.score_components(P, G, foot)
    assert s["DTI"] == 0.0
    assert abs(s["FNw"] - 1.0) < 1e-12


def test_dti_bounded_and_monotone_in_hits():
    rng = np.random.default_rng(0)
    G = (rng.random((60, 60)) < 0.02).astype(float)
    foot = np.ones_like(G, bool)
    # perfect cover
    P1 = G.copy()
    # half of truth covered
    mask = rng.random((60, 60)) < 0.5
    P2 = np.where(mask, G, 0.0)
    d1 = M.score(P1, G, foot)
    d2 = M.score(P2, G, foot)
    assert 0.0 <= d2 < d1 <= 1.0


def test_official_formulas_match_reference_computation():
    """Cross-check score_components against a brute-force O(|G||P|) implementation."""
    rng = np.random.default_rng(1)
    H = W = 40
    G = (rng.random((H, W)) < 0.05).astype(float)
    P = (rng.random((H, W)) < 0.05).astype(float)
    foot = np.ones((H, W), bool)
    R = 3.0

    g = np.argwhere(G > 0)
    p = np.argwhere(P > 0)

    tpw = 0.0
    for (gr, gc) in g:
        best = 0.0
        for (pr, pc) in p:
            d = np.hypot(pr - gr, pc - gc)
            if d <= R:
                best = max(best, P[pr, pc] * (1 - d / R))
        tpw += best
    fpw = 0.0
    for (pr, pc) in p:
        bestk = 0.0
        for (gr, gc) in g:
            d = np.hypot(pr - gr, pc - gc)
            if d <= R:
                bestk = max(bestk, 1 - d / R)
        fpw += P[pr, pc] * (1 - bestk)
    fnw = len(g) - tpw
    ref = tpw / (tpw + 0.2 * fpw + 0.8 * fnw + 1e-12)

    s = M.score_components(P, G, foot)
    assert abs(s["TPw"] - tpw) < 1e-9
    assert abs(s["FPw"] - fpw) < 1e-9
    assert abs(s["FNw"] - fnw) < 1e-9
    assert abs(s["DTI"] - ref) < 1e-9


def test_kernel_support_exactly_3px():
    offs, w = M.kernel_weights(3)
    d = np.hypot(offs[:, 0], offs[:, 1])
    assert np.all(d < 3.0)
    assert (d == 3.0).sum() == 0  # k(3) = 0 excluded
    assert abs(w[np.argmin(d)] - 1.0) < 1e-12
