"""Exact Distance-Weighted Tversky Index (DTI) for the DOE GEMS Prize (DrivenData #306).

Every formula here is transcribed line-by-line from the official problem description
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ , section
"Performance metric"), read 2026-10-06 UTC.  The masking rule is taken from the organizer
statement in community thread 11516.

OFFICIAL DEFINITIONS (verbatim structure)
-----------------------------------------
Let p(x) in [0, 1] be the predicted probability at pixel x and g(x) the ground-truth label.

    k(d) = (1 - d / R)+          R = 300 m  (3 pixels at the 100 m competition grid)

    TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
    FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))]

    DTI(alpha, beta) = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps)
    alpha = 0.2 (false-positive weight), beta = 0.8 (false-negative weight)

MASKING
-------
DrivenData staff (thread 11516, 2026-09-16): "Pixels corresponding to known USGS/INGENIOUS
faults are masked / excluded from evaluation, so they do not count towards penalty terms."
and "for scoring purposes it should not matter whether these known faults are included with
predictions or not."  We therefore zero predictions and truth on the excluded set *before*
applying the kernel.  That is the only reading under which the staff's second sentence is
literally true, so it is the reading implemented here.

IDENTITIES (both re-derived and unit-tested, not assumed)
---------------------------------------------------------
*  FN_w = |G| - TP_w                                    (exact, by substitution)
*  DTI  = TP_w / (alpha*(TP_w + FP_w) + beta*|G| + eps)  (exact, by substitution)
*  for a binary {0,1} prediction, TP_w = sum_{g in G} k(d_pred(g)) * 1[d_pred(g) <= R]
   and FP_w = sum_{x : p(x)=1} (1 - k(d_gt(x))), i.e. two Euclidean distance transforms.

The module is deliberately small and dependency-light so that the exact operator can be
checked against an independent O(|G| * |P|) brute-force transcription in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_PX: float = 3.0
EPS: float = 1e-12
PIXEL_M: float = 100.0


def kernel(d_px: np.ndarray | float, radius_px: float = RADIUS_PX) -> np.ndarray:
    """Triangular kernel k(d) = max(1 - d/R, 0). Distances in pixels (1 px = 100 m)."""
    return np.maximum(1.0 - np.asarray(d_px, dtype=np.float64) / float(radius_px), 0.0)


def kernel_offsets(radius_px: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    """All integer offsets with k > 0, each with its kernel weight.

    R = 3 px -> the 29 integer offsets with k >= 0 collapse to 25 with k > 0, because k is
    exactly 0 at d = R (the four axis offsets at distance 3 fall out; all 25 remaining
    offsets have d <= 2.828 px).  Verified by the tests, not assumed.
    """
    r = int(np.ceil(radius_px))
    out: list[tuple[int, int, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = float(kernel(np.hypot(dy, dx), radius_px))
            if w > 0.0:
                out.append((dy, dx, w))
    return out


_OFFSETS = kernel_offsets()
_OFFSET_ARRAY = np.array([[dy, dx, w] for dy, dx, w in _OFFSETS], dtype=np.float64)


@dataclass(frozen=True)
class DTIConstants:
    """Sufficient statistics of one DTI evaluation."""

    tp_w: float
    fp_w: float
    fn_w: float
    n_truth: int
    score: float

    def ledger(self) -> dict:
        return {
            "TP_w": self.tp_w,
            "FP_w": self.fp_w,
            "FN_w": self.fn_w,
            "G_pixels": self.n_truth,
            "DTI": self.score,
        }


def _scored_domain(pred: np.ndarray, truth: np.ndarray, valid, known) -> tuple[np.ndarray, np.ndarray]:
    """Zero predictions and truth outside the footprint and on the known-fault mask."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("prediction and truth must be equal-shaped 2-D grids")
    if valid is None:
        valid = np.ones(pred.shape, dtype=bool)
    if known is None:
        known = np.zeros(pred.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    known = np.asarray(known, dtype=bool)
    if valid.shape != pred.shape or known.shape != pred.shape:
        raise ValueError("valid/known mask shape mismatch")
    active = valid & ~known
    p = np.where(active, pred, 0.0)
    g = np.where(active, truth > 0, False)
    if not np.isfinite(p).all():
        raise ValueError("predictions inside the scored domain must be finite")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("predictions inside the scored domain must lie in [0, 1]")
    return p, g


def _max_over_disc(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """max over the 29 kernel offsets of field[shifted] * weight. Exact per the definition."""
    n0, n1 = field.shape
    out = np.zeros(field.shape, dtype=np.float64)
    for dy, dx, w in _OFFSETS:
        # shifted[i, j] = field[i + dy, j + dx] * w, with wrapped entries forced to 0.
        shifted = np.roll(np.roll(field, -dy, axis=0), -dx, axis=1) * w
        if dy > 0:            # out[i] = field[i + dy]: valid for i < n0 - dy
            shifted[n0 - dy:, :] = 0.0
        elif dy < 0:          # out[i] = field[i + dy]: valid for i >= -dy
            shifted[:-dy, :] = 0.0
        if dx > 0:
            shifted[:, n1 - dx:] = 0.0
        elif dx < 0:
            shifted[:, :-dx] = 0.0
        np.maximum(out, shifted, out=out)
    return out


def dti(pred: np.ndarray, truth: np.ndarray, valid=None, known=None) -> DTIConstants:
    """Exact DTI for soft or binary predictions, exactly as the organizer defines it."""
    p, g = _scored_domain(pred, truth, valid, known)
    coverage = _max_over_disc(p, _OFFSET_ARRAY[:, 2])          # max_x p(x) k(d(x,g)) for every pixel
    gt_weight = _max_over_disc(g.astype(np.float64), _OFFSET_ARRAY[:, 2])  # max_g k(d(x,g)) for every pixel
    tp_w = float(coverage[g].sum())
    fp_w = float((p * (1.0 - gt_weight))[p > 0].sum())
    fn_w = float(g.sum() - tp_w)
    denom = tp_w + ALPHA * fp_w + BETA * fn_w + EPS
    return DTIConstants(tp_w=tp_w, fp_w=fp_w, fn_w=fn_w, n_truth=int(g.sum()), score=tp_w / denom)


def dti_bruteforce(pred: np.ndarray, truth: np.ndarray, valid=None, known=None) -> DTIConstants:
    """Independent O(|P|*|G|) transcription used only to falsify :func:`dti` in tests."""
    p, g = _scored_domain(pred, truth, valid, known)
    gs = np.argwhere(g)
    ps = np.argwhere(p > 0)
    pv = p[p > 0]
    if gs.size == 0:
        return DTIConstants(0.0, float(pv.sum()), 0.0, 0, 0.0)

    def k_of(d: np.ndarray) -> np.ndarray:
        return np.maximum(1.0 - d / RADIUS_PX, 0.0)

    tp_w = 0.0
    for gy, gx in gs:
        d = np.hypot(ps[:, 0] - gy, ps[:, 1] - gx)
        tp_w += float((pv * k_of(d)).max()) if ps.size else 0.0
    fp_w = 0.0
    for idx, (xy, xv) in enumerate(zip(ps, pv)):
        d = np.hypot(gs[:, 0] - xy[0], gs[:, 1] - xy[1])
        fp_w += float(xv * (1.0 - k_of(d).max()))
    fn_w = float(gs.shape[0]) - tp_w
    denom = tp_w + ALPHA * fp_w + BETA * fn_w + EPS
    return DTIConstants(tp_w, fp_w, fn_w, int(gs.shape[0]), tp_w / denom)


def dti_closed_form(tp_w: float, fp_w: float, n_truth: int) -> float:
    """T / (alpha*(T+F) + beta*G + eps) -- the algebraically reduced official formula."""
    return tp_w / (ALPHA * (tp_w + fp_w) + BETA * n_truth + EPS)


def binary_credit(dots: np.ndarray, truth: np.ndarray, valid=None, known=None) -> dict:
    """Exact sufficient statistics for a binary {0,1} dot file, via two distance transforms.

    Returns T (=TP_w), F (=FP_w), G (=|truth| in the scored domain), n (=number of dots in
    the scored domain) and the two distance maps.  Used by the emitter and the holdout.
    """
    from scipy.ndimage import distance_transform_edt

    p, g = _scored_domain(np.asarray(dots, dtype=np.float64), truth, valid, known)
    dot = p > 0.0
    if not dot.any() or not g.any():
        return {
            "T": 0.0, "F": float(dot.sum()), "G": int(g.sum()), "n": int(dot.sum()),
            "d_pred": np.full(p.shape, np.inf), "d_gt": np.full(p.shape, np.inf),
        }
    d_pred = distance_transform_edt(~dot, sampling=1.0)          # distance from each pixel to nearest dot
    d_gt = distance_transform_edt(~g, sampling=1.0)              # distance from each pixel to nearest truth pixel
    t = float(kernel(d_pred[g]).sum())
    f = float((1.0 - kernel(d_gt[dot])).sum())
    return {"T": t, "F": f, "G": int(g.sum()), "n": int(dot.sum()), "d_pred": d_pred, "d_gt": d_gt}


def dti_of(binary_stats: dict) -> float:
    """DTI from :func:`binary_credit` sufficient statistics."""
    return dti_closed_form(binary_stats["T"], binary_stats["F"], binary_stats["G"])
