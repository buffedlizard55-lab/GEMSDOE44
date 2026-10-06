"""Falsification tests for :mod:`gems44.metric`.

These tests deliberately attack the implementation: random grids, soft predictions,
masking, empty sets, and the closed-form reductions.  Any disagreement between the
fast operator and the independent brute-force transcription is a failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from gems44.metric import (
    ALPHA,
    BETA,
    RADIUS_PX,
    binary_credit,
    dti,
    dti_bruteforce,
    dti_closed_form,
    dti_of,
    kernel,
)


def _random_case(seed: int, shape=(24, 26), soft: bool = False):
    rng = np.random.default_rng(seed)
    if soft:
        pred = rng.random(shape).astype(np.float64)
        pred[pred < 0.75] = 0.0
    else:
        pred = (rng.random(shape) < 0.10).astype(np.float64)
    truth = (rng.random(shape) < 0.12).astype(np.int8)
    valid = rng.random(shape) > 0.10
    known = (rng.random(shape) < 0.08)
    pred = np.where(valid & ~known, pred, 0.0)
    truth = np.where(valid & ~known, truth, 0)
    return pred, truth, valid, known


@pytest.mark.parametrize("seed", range(8))
def test_fast_equals_bruteforce_binary(seed):
    pred, truth, valid, known = _random_case(seed)
    a = dti(pred, truth, valid, known)
    b = dti_bruteforce(pred, truth, valid, known)
    assert abs(a.tp_w - b.tp_w) < 1e-9
    assert abs(a.fp_w - b.fp_w) < 1e-9
    assert abs(a.fn_w - b.fn_w) < 1e-9
    assert abs(a.score - b.score) < 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_fast_equals_bruteforce_soft(seed):
    pred, truth, valid, known = _random_case(100 + seed, soft=True)
    a = dti(pred, truth, valid, known)
    b = dti_bruteforce(pred, truth, valid, known)
    assert abs(a.tp_w - b.tp_w) < 1e-9
    assert abs(a.fp_w - b.fp_w) < 1e-9
    assert abs(a.score - b.score) < 1e-12


def test_fn_identity_and_closed_form():
    for seed in range(6):
        pred, truth, valid, known = _random_case(200 + seed)
        r = dti(pred, truth, valid, known)
        assert abs(r.fn_w - (r.n_truth - r.tp_w)) < 1e-9            # FN_w = |G| - TP_w
        assert abs(r.score - dti_closed_form(r.tp_w, r.fp_w, r.n_truth)) < 1e-12


def test_kernel_values():
    assert kernel(0.0) == 1.0
    assert abs(kernel(1.0) - 2.0 / 3.0) < 1e-12          # 100 m -> 2/3
    assert abs(kernel(2.0) - 1.0 / 3.0) < 1e-12          # 200 m -> 1/3
    assert kernel(3.0) == 0.0                            # 300 m -> 0 (k = (1-d/R)+)
    assert kernel(10.0) == 0.0
    assert RADIUS_PX == 3.0 and ALPHA == 0.2 and BETA == 0.8


def test_perfect_prediction_scores_one():
    shape = (20, 20)
    truth = np.zeros(shape, np.int8)
    truth[5, 3:17] = 1
    pred = truth.astype(np.float64)
    r = dti(pred, truth)
    assert abs(r.score - 1.0) < 1e-9


def test_empty_prediction_scores_zero():
    shape = (20, 20)
    truth = np.zeros(shape, np.int8)
    truth[5, 3:17] = 1
    assert dti(np.zeros(shape), truth).score == 0.0


def test_binary_dominates_its_own_scaled_version():
    """Prop. (i): a graded map is strictly dominated by its own thresholded support."""
    shape = (40, 40)
    truth = np.zeros(shape, np.int8)
    truth[8, 5:35] = 1
    truth[30, 5:35] = 1
    rng = np.random.default_rng(7)
    field = rng.random(shape) * 0.4
    field[7:10, 4:36] = 0.9                     # a fuzzy band over the first trace
    base = dti(field, truth).score
    thresh = dti((field >= 0.9).astype(np.float64), truth).score
    assert thresh > base


def test_scaling_a_soft_map_never_helps():
    """DTI(lambda*p) is increasing in lambda, so lambda=1 dominates any lambda<1."""
    shape = (40, 40)
    truth = np.zeros(shape, np.int8)
    truth[8, 5:35] = 1
    rng = np.random.default_rng(11)
    field = rng.random(shape)
    field[7:10, 4:36] = 0.8
    scores = [dti(field * lam, truth).score for lam in (0.2, 0.5, 0.8, 1.0)]
    assert all(b > a for a, b in zip(scores, scores[1:]))


def test_marginal_rule_matches_direct_evaluation():
    """(ii): adding a dot of credit k pays iff k > 0.2 * DTI. Checked by direct re-scoring."""
    shape = (60, 60)
    truth = np.zeros(shape, np.int8)
    truth[30, 5:55] = 1
    dots = np.zeros(shape, np.float64)
    dots[30, np.arange(6, 55, 4)] = 1.0
    s0 = dti(dots, truth).score
    for extra_col, extra_row in [(10, 31), (10, 33), (10, 26)]:
        e = dots.copy()
        e[extra_row, extra_col] = 1.0
        s1 = dti(e, truth).score
        # credit of the added dot = increase in TP_w if it is the best cover of some truth pixel
        added = e - dots
        r0 = dti(dots, truth)
        r1 = dti(e, truth)
        credit = r1.tp_w - r0.tp_w
        predicted_improves = credit > 0.2 * s0
        assert predicted_improves == (s1 > s0)


def test_known_mask_is_neutral():
    """Staff: it should not matter whether known faults are included in predictions."""
    shape = (30, 30)
    truth = np.zeros(shape, np.int8)
    truth[4, 2:28] = 1
    known = np.zeros(shape, bool)
    known[20, 2:28] = True
    pred_a = np.zeros(shape, np.float64)
    pred_a[3:6, 2:28] = 1.0
    pred_b = pred_a.copy()
    pred_b[known] = 1.0                      # also paint the known fault
    assert abs(dti(pred_a, truth, known=known).score - dti(pred_b, truth, known=known).score) < 1e-12
    # ... and a dot placed *on* the known mask earns nothing
    only_known = np.zeros(shape, np.float64)
    only_known[known] = 1.0
    assert dti(only_known, truth, known=known).score == 0.0


def test_binary_credit_matches_dti():
    for seed in range(6):
        pred, truth, valid, known = _random_case(300 + seed)
        s = binary_credit(pred > 0, truth, valid, known)
        r = dti(pred, truth, valid, known)
        assert abs(s["T"] - r.tp_w) < 1e-9
        assert abs(s["F"] - r.fp_w) < 1e-9
        assert s["G"] == r.n_truth
        assert abs(dti_of(s) - r.score) < 1e-12


def test_rejects_out_of_range_and_nonfinite():
    shape = (8, 8)
    truth = np.zeros(shape, np.int8)
    with pytest.raises(ValueError):
        dti(np.full(shape, 1.5), truth)
    with pytest.raises(ValueError):
        dti(np.full(shape, np.nan), truth)
    with pytest.raises(ValueError):
        dti(np.full(shape, -0.1), truth)


def test_nan_outside_footprint_is_allowed():
    """The official sample submission carries NaN outside the footprint: that must not raise."""
    shape = (10, 10)
    truth = np.zeros(shape, np.int8)
    truth[5, 2:8] = 1
    pred = np.full(shape, np.nan)
    pred[5, 2:8] = 1.0
    valid = np.isfinite(pred)          # the footprint is exactly the non-NaN region
    r = dti(pred, truth, valid=valid)
    assert r.score > 0.9
