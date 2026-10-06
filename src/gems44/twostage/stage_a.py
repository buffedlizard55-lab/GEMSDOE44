"""Stage A: coarse favourability -- "which broad zones are worth searching at all".

This is deliberately NOT a fault detector.  It is the coarse gate the brief asks
for, and it is validated as a gate, separately from the placement stage:

    Stage A answers:  where, at basin scale, does the heat + permeability +
                      strain argument say faults *should* exist at all?
    Stage B answers:  at pixel scale, where inside those zones is the fault?

The two questions have different truth, different error costs and different
metrics, so they are never collapsed into one number (that collapse is exactly
what hid the failure of the family's habitat-only attempts: 0.0041, 0.1223 and
0.1352 -- three of the worst results in the whole GEMSDOE record -- all of which
were coarse statements asked to answer a pixel-level question).

LAYERS (all from the 19 official bands; no external data, so nothing is
data-blocked and every number is reproducible in this sandbox)
  thermal / permeability proxies   conductivity_surface, depth_to_basement
  strain                            geodetic_2nd_invariant, geodetic_shear_rate,
                                    geodetic_dilatation_rate
  structural density                catalogue fault density at 3 km, 10 km
  seismicity                        earthquake_density
  gravity / magnetics (regional)    isostatic_grav_anomaly, magnetic_anomaly

The gat e is trained as a *zone classifier*: at a coarse scale (mean-pooled by 20
pixels = 2 km), does this 2 km cell contain catalogue faults?  That is the only
honest statement available about "fault-hosting favourability", because the
catalogue is the one complete-at-regional-scale fault map we have that is NOT the
scoring truth.  It is used for a gating decision only -- never as a pixel target.

Validation of Stage A is reported separately as:
  * blocked-quadrant AUC of the zone score
  * base-rate lift: the factor by which the fault density inside the approved
    zone exceeds the footprint background
  * recall of the *off-catalogue* SGMC faults (the frame Stage B is judged on)
    captured inside the approved zone, against the area fraction it occupies
  * the same numbers for a random zone of identical area (the control)

The gate is applied at a chosen area fraction; the fraction is a *coverage*
decision, and the write-up reports the whole lift-vs-fraction curve rather than
one tuned point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import stage_b as B

STAGE_A_BANDS = [
    "conductivity_surface", "depth_to_basement",
    "geodetic_2nd_invariant", "geodetic_shear_rate", "geodetic_dilatation_rate",
    "earthquake_density", "isostatic_grav_anomaly", "magnetic_anomaly",
]

POOL = 20          # 20 px = 2 km coarse cell
POOL_LARGE = 100   # 100 px = 10 km regional cell


@dataclass(frozen=True)
class StageAResult:
    coarsened: np.ndarray       # coarse-cell mean of the score, upsampled to the grid
    score: np.ndarray           # continuous favouritability score (z-scaled, larger = better)
    mask: np.ndarray            # boolean gate at the chosen area fraction
    area_fraction: float
    lift: float                 # fault density inside gate / footprint background
    recall_offcat: float        # fraction of off-catalogue SGMC px inside the gate


def _z(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    m, s = float(np.mean(a)), float(np.std(a)) + 1e-9
    return (a - m) / s


def fault_density(known: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Catalogue fault density in a disc of ``radius`` pixels (count of catalogue px)."""
    # Separable box filter (O(N * k)), NOT a direct 2-D convolution: a (101,101) kernel
    # convolved directly over 12.3 M pixels is ~1.3e11 operations and hangs the sandbox.
    k = (known & valid).astype(np.float32)
    size = 2 * radius + 1
    num = ndimage.uniform_filter(k, size=size, mode="nearest") * float(size * size)
    den = ndimage.uniform_filter(valid.astype(np.float32), size=size, mode="nearest") * float(size * size) + 1e-6
    return (num / den).astype(np.float32)


def stage_a_layers(bands: dict[str, np.ndarray], known: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Build the coarse layers.  Every layer is a named physical quantity."""
    out: dict[str, np.ndarray] = {}
    for b in STAGE_A_BANDS:
        if b not in bands:
            continue
        a = B._fill(bands[b])
        # coarse-scale version: the 2 km cell mean is the scale at which the
        # heat-flow / permeability / strain argument is actually made
        out[f"{b}_coarse"] = ndimage.uniform_filter(a, size=POOL)
    out["catalogue_density_3km"] = fault_density(known, valid, 15)
    out["catalogue_density_10km"] = fault_density(known, valid, 50)
    return out


PHYSICAL_ONLY_WEIGHTS: dict[str, float] = {
    "conductivity_surface_coarse": 0.10,
    "geodetic_2nd_invariant_coarse": 0.10,
    "geodetic_shear_rate_coarse": 0.05,
    "earthquake_density_coarse": 0.08,
    "depth_to_basement_coarse": 0.05,
    "isostatic_grav_anomaly_coarse": 0.04,
    "magnetic_anomaly_coarse": 0.03,
    "geodetic_dilatation_rate_coarse": 0.02,
}

STAGE_A_WEIGHTS: dict[str, float] = {
    "catalogue_density_3km": 0.40,
    "catalogue_density_10km": 0.15,
    "conductivity_surface_coarse": 0.10,
    "geodetic_2nd_invariant_coarse": 0.10,
    "geodetic_shear_rate_coarse": 0.05,
    "earthquake_density_coarse": 0.08,
    "depth_to_basement_coarse": 0.05,
    "isostatic_grav_anomaly_coarse": 0.04,
    "magnetic_anomaly_coarse": 0.03,
}


def stage_a_score(layers: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted sum of z-scored coarse layers.

    The default weights encode the physical argument, not a fit: in an amagmatic
    extensional province a geothermal system needs a heat source at depth
    (conductivity/gravity), a permeable pathway (fault density), and strain that
    keeps that pathway open.  Fault density carries the largest weight because it
    is the only layer that is a direct observation of faulting rather than a
    proxy for the conditions that permit it.

    HONESTY NOTE (why the shipped weights and a physical-only variant are BOTH
    reported).  The catalogue-density layers are a smoothed version of the very
    thing Stage A is validated against ("does this 2 km cell contain catalogue
    faults?"), so the blocked AUC of the shipped score is self-referential: it
    measures how well a smoothed fault map predicts a fault map.  It is a
    legitimate input for a GATE (the catalogue is known, not the truth, and it is
    never used as a pixel target), but it is not evidence that the physical layers
    work.  ``stage_a_score(layers, PHYSICAL_ONLY_WEIGHTS)`` drops the two
    catalogue-density terms, and both AUCs are reported side by side.
    """
    if weights is None:
        weights = STAGE_A_WEIGHTS
    total = np.zeros(next(iter(layers.values())).shape, dtype=np.float32)
    for k, w in weights.items():
        if k in layers:
            total += w * _z(layers[k])
    return total


def apply_gate(score: np.ndarray, valid: np.ndarray, area_fraction: float) -> np.ndarray:
    """Top ``area_fraction`` of the valid footprint by Stage A score.

    Implemented by thresholding the *coarse* score so the accepted zone is
    spatially contiguous rather than speckled: this is a zone statement, and a
    speckled gate would silently become a placement model.
    """
    s = np.where(valid, score, -np.inf)
    thr = float(np.quantile(s[valid], 1.0 - area_fraction))
    return (s >= thr) & valid


def validate_stage_a(
    score: np.ndarray,
    valid: np.ndarray,
    known: np.ndarray,
    offcat: np.ndarray,
    quad: np.ndarray,
    area_fraction: float,
) -> dict:
    """Separately-validated Stage A numbers.  Nothing here is a DTI number."""
    gate = apply_gate(score, valid, area_fraction)
    bg = float(known[valid].mean())
    inside = float(known[gate].mean()) if gate.any() else 0.0
    lift = inside / bg if bg > 0 else float("nan")

    # blocked AUC of the *coarse* score for "this 2 km cell contains catalogue faults"
    cell = ndimage.uniform_filter(known.astype(np.float32), size=POOL, mode="nearest")
    yy = (cell > 0) & valid
    auc_folds = {}
    for q in range(4):
        test = yy & (quad == q)
        if test.sum() == 0:
            continue
        # rank-based AUC of the score restricted to this quadrant's cells
        from sklearn.metrics import roc_auc_score
        sub = valid & (quad == q)
        y = (cell[sub] > 0).astype(np.int8)
        x = score[sub]
        auc_folds[f"quad{q}"] = float(roc_auc_score(y, x)) if y.min() != y.max() else float("nan")

    recall = float(offcat[gate].sum() / max(offcat.sum(), 1))
    return {
        "area_fraction": float(area_fraction),
        "gate_px": int(gate.sum()),
        "catalogue_density_inside_gate": inside,
        "catalogue_density_footprint_background": bg,
        "base_rate_lift": float(lift),
        "offcatalogue_recall_inside_gate": recall,
        "offcatalogue_recall_over_area_fraction": recall / max(area_fraction, 1e-9),
        "blocked_cell_auc": auc_folds,
        "blocked_cell_auc_mean": float(np.nanmean(list(auc_folds.values()))) if auc_folds else float("nan"),
        "note": (
            "Stage A is a ZONE statement. Its numbers are base-rate and recall numbers, "
            "deliberately not DTI numbers: a coarse mask cannot place a pixel, which is "
            "the failure mode of the family's habitat-only emissions (0.0041/0.1223/0.1352)."
        ),
    }
