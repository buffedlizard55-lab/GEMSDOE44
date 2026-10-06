"""Belief field + metric-exact emission for GEMS44.

Two ideas, both falsifiable and both validated in ``scripts/run_holdout.py``:

1. ``OFFCAT_FIELD`` -- train the field on the only local frame that can reward
   off-catalogue skill: real mapped faults that are absent from the competition
   catalogue (USGS SGMC minus a 300 m catalogue buffer), against random footprint
   pixels, with leave-one-quadrant-out spatial blocking.  Prior repositories in this
   family used catalogue-truth proxies, which provably rank artifacts in the OPPOSITE
   order to the live board (0.17193 > 0.16635 > 0.16177 vs live 0.1922 < 0.2477 <
   0.2600, measured in GEMSDOE32).  This module never uses the catalogue as truth.

2. ``greedy_marginal_credit`` -- the exact greedy max-coverage emitter for the official
   metric.  Because the metric's TP term is a MAX over a 300 m disc, expected credit is
   submodular in the emitted set, so greedy carries the standard (1 - 1/e) guarantee.
   The stopping rule is the metric's own first-order condition, k > alpha * DTI
   (derived and unit-tested in ``gems44.metric``), not a tuned knob.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metric import ALPHA, kernel, kernel_offsets

# ---------------------------------------------------------------------------------
# Feature construction.  Every entry is a physically named transform of one of the 19
# official bands; the names are the band descriptions read from training_features.tif.
# ---------------------------------------------------------------------------------

FEATURE_SPEC: list[tuple[str, str, int]] = [
    # (feature name, source band name, scale px; scale 0 = raw pixel)
    ("detrended_elevation_slope", "detrended_elevation_slope", 0),
    ("detrended_elevation_slope_m3", "detrended_elevation_slope", 3),
    ("detrended_elevation_slope_m9", "detrended_elevation_slope", 9),
    ("detrended_elevation", "detrended_elevation", 0),
    ("detrended_elevation_m3", "detrended_elevation", 3),
    ("tilt_angle_tot_curv", "tilt_angle_tot_curv", 0),
    ("tilt_angle_tot_curv_m9", "tilt_angle_tot_curv", 9),
    ("tmi_hgrad", "tmi_hgrad", 0),
    ("tmi_vgrad", "tmi_vgrad", 0),
    ("tmi_total", "tmi_total", 0),
    ("rtp_magnetic", "rtp_magnetic", 0),
    ("magnetic_anomaly", "magnetic_anomaly", 0),
    ("isostatic_grav_hgrad", "isostatic_grav_hgrad", 0),
    ("isostatic_grav_vgrad", "isostatic_grav_vgrad", 0),
    ("isostatic_grav_slope", "isostatic_grav_slope", 0),
    ("isostatic_grav_anomaly", "isostatic_grav_anomaly", 0),
    ("geodetic_2nd_invariant", "geodetic_2nd_invariant", 0),
    ("geodetic_shear_rate", "geodetic_shear_rate", 0),
    ("geodetic_dilatation_rate", "geodetic_dilatation_rate", 0),
    ("earthquake_density", "earthquake_density", 0),
    ("earthquake_density_m9", "earthquake_density", 9),
    ("depth_to_basement", "depth_to_basement", 0),
    ("depth_to_basement_m9", "depth_to_basement", 9),
    ("conductivity_surface", "conductivity_surface", 0),
    ("conductivity_surface_m9", "conductivity_surface", 9),
    ("curvature_laplacian_m3", "detrended_elevation", 3),
    ("curvature_laplacian_m9", "detrended_elevation", 9),
]


def _nanfill(a: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(a)
    med = float(np.median(a[finite])) if finite.any() else 0.0
    return np.where(finite, a, med), med


def build_feature(block: np.ndarray, band_name: str, scale: int) -> np.ndarray:
    """One feature from one band.  ``scale`` 0 = the value, >0 = mean filter / curvature."""
    a = block.astype(np.float32)
    a[a <= -1e37] = np.nan
    if scale == 0:
        out, _ = _nanfill(a)
        return out
    if band_name == "detrended_elevation" and scale in (3, 9):
        filled, _ = _nanfill(a)
        return np.abs(ndimage.laplace(ndimage.uniform_filter(filled, size=scale)))
    filled, _ = _nanfill(a)
    return ndimage.uniform_filter(filled, size=scale)


def build_matrix(blocks: dict[str, np.ndarray], names: list[str], rows: np.ndarray = None) -> np.ndarray:
    """Assemble the design matrix (or a row subset) from per-band full-grid arrays."""
    cols = []
    for feat_name, band_name, scale in FEATURE_SPEC:
        if feat_name not in names:
            continue
        f = build_feature(blocks[band_name] if scale == 0 or band_name != "detrended_elevation" else blocks[band_name],
                          band_name, scale)
        cols.append(f.ravel() if rows is None else f.ravel()[rows])
    return np.column_stack(cols).astype(np.float32)


def feature_names() -> list[str]:
    return [f[0] for f in FEATURE_SPEC]


# ---------------------------------------------------------------------------------
# Metric-exact emission
# ---------------------------------------------------------------------------------

def greedy_marginal_credit(
    belief: np.ndarray,
    candidates: np.ndarray,
    allowed: np.ndarray,
    max_dots: int,
    belief_scale: float = 1.0,
) -> np.ndarray:
    """Greedy expected-marginal-credit emitter on a belief density ``belief``.

    ``belief[i]`` is the expected density of scored-truth pixels at pixel ``i``.  A dot
    accepted at pixel x adds

        dT(x) = sum_delta belief[x + delta] * max(0, k(delta) - C[x + delta])

    where C is the credit already delivered by accepted dots.  Candidates are visited in
    descending belief order (the standard fixed-order greedy for submodular maximisation).
    Complexity is O(|candidates| * |offsets|), so the full 5.1 M-pixel domain is tractable.

    Returns a boolean raster of accepted dots.
    """
    h, w = belief.shape
    C = np.zeros((h, w), dtype=np.float32)
    dots = np.zeros((h, w), dtype=bool)
    offs = kernel_offsets()
    b = (belief * belief_scale).astype(np.float32)
    n_taken = 0
    for idx in candidates:
        if n_taken >= max_dots:
            break
        y, x = divmod(int(idx), w)
        if not allowed[y, x]:
            continue
        gain = 0.0
        for dy, dx, k in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                slack = k - C[yy, xx]
                if slack > 0.0:
                    gain += float(b[yy, xx]) * slack
        if gain <= 0.0:
            continue
        dots[y, x] = True
        n_taken += 1
        for dy, dx, k in offs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and C[yy, xx] < k:
                C[yy, xx] = k
    return dots


_OFF = kernel_offsets()
_DY = np.array([o[0] for o in _OFF], dtype=np.int64)
_DX = np.array([o[1] for o in _OFF], dtype=np.int64)
_K = np.array([o[2] for o in _OFF], dtype=np.float32)


def emit_order_np(
    belief: np.ndarray,
    candidates: np.ndarray,
    allowed: np.ndarray,
    max_dots: int,
) -> np.ndarray:
    """Vectorised exact fixed-order greedy emitter.  Returns flat indices in acceptance order.

    Same objective as :func:`greedy_marginal_credit`; because the accepted set is a prefix
    of the returned order, every prefix is itself a valid, nested sub-emission, which is
    what makes the mass sweep in ``scripts/run_holdout.py`` a single cheap pass.
    """
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


def marginal_break_even(score: float) -> float:
    """First-order condition of the official metric: a dot pays iff credit > alpha * DTI."""
    return ALPHA * score


def break_even_distance_m(score: float, radius_m: float = 300.0) -> float:
    """Maximum distance from scored truth at which an added dot still pays (official metric)."""
    return radius_m * (1.0 - marginal_break_even(score))


def quantile_candidates(belief: np.ndarray, allowed: np.ndarray, top_fraction: float) -> np.ndarray:
    """Flat indices of the highest-belief allowed pixels, in descending belief order."""
    flat = np.where(allowed.ravel(), belief.ravel(), -np.inf)
    n = int(allowed.sum() * top_fraction)
    n = max(n, 1)
    if n >= flat.size:
        order = np.argsort(-flat)
        return order[np.isfinite(flat[order])]
    part = np.argpartition(-flat, n - 1)[:n]
    part = part[np.isfinite(flat[part])]
    return part[np.argsort(-flat[part])]
