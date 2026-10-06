"""Stage B feature library: fine-scale fault-placement transforms of the 19 official bands.

DESIGN RULES (why every feature here is named the way it is)
-----------------------------------------------------------
1.  No feature may read the catalogue (`labels.tif`) or the SGMC raster.  The
    catalogue is masked by the official scorer, so a catalogue-trained feature
    optimises the wrong target; SGMC is this project's *validation truth* and
    using it as a feature would leak the frame.  Only the 19 official bands
    enter here.
2.  Every feature is a *fine-scale* transform: it responds to a step, ridge,
    trough or curvature at the 300 m scale the metric actually pays for.  The
    coarse "where is it worth searching at all" statement is Stage A's job and
    is kept physically separate.
3.  The transforms are the standard structural-geology operators, applied to the
    official bands at multiple scales, because the scale of a fault's surface
    expression is not known a priori:
        * LRM  -- local relief model (Hesse/Buchwaldt style): z - mean(z, r).
                  Removes regional slope; a fault scarp appears as a
                  positive/negative paired anomaly.
        * openness -- topographic openness (Yokoyama/Druguet style): the mean
                  horizon angle along rays.  A scarp has a strong
                  positive/negative openness contrast across it.
        * anti-slope -- openness_neg - openness_pos; the classic piedmont-scarp
                  detector (scarps facing away from the slope of the surface).
        * Hessian ridge/valley response -- the eigenvalues of the 2-D Hessian
                  of the smoothed surface.  A fault trace is a ridge or a valley
                  in detrended elevation on one of the scales.
        * total horizontal gradient / analytic signal -- the classical potential
                  field edge detector (Nabighian), applied to gravity and
                  magnetics for the *buried* faults topography cannot see.
        * strike-alignment -- the fraction of the local gradient energy that lies
                  within 30 degrees of the Basin & Range fault fabric
                  (N20E-N20W), evaluated at three scales.  Faults in the Great
                  Basin are not randomly oriented; a lineament that is not
                  fabric-aligned is more often a road, a stream or a fold limb.
4.  Every feature is computed with NaN-safe fills; the nodata sentinel
    -3.4028234663852886e+38 is mapped to NaN and replaced by the block median,
    as in the rest of this repository.

The library is intentionally additive: adding a feature is a new entry in
``STAGE_B_SPEC``, and every entry carries the physical meaning used in the
write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

NODATA = -1e37

# ---------------------------------------------------------------------------------
# band registry: name -> 1-based index in training_features.tif
# ---------------------------------------------------------------------------------
BAND_ORDER = [
    "magnetic_anomaly", "rtp_magnetic", "tmi_hgrad", "geodetic_2nd_invariant",
    "isostatic_grav_slope", "tilt_angle_tot_curv", "geodetic_shear_rate",
    "geodetic_dilatation_rate", "tmi_vgrad", "dist_to_earthquake",
    "isostatic_grav_vgrad", "detrended_elevation", "isostatic_grav_anomaly",
    "tmi_total", "depth_to_basement", "earthquake_density", "conductivity_surface",
    "isostatic_grav_hgrad", "detrended_elevation_slope",
]
BAND_INDEX = {name: i + 1 for i, name in enumerate(BAND_ORDER)}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    scale: int
    family: str          # "topographic" | "potential_field" | "structural"


STAGE_B_SPEC: list[FeatureSpec] = [
    # ---- topographic: local relief models (LRM) ---------------------------------
    FeatureSpec("lrm_r5", "lrm", 5, "topographic"),
    FeatureSpec("lrm_r15", "lrm", 15, "topographic"),
    FeatureSpec("lrm_r40", "lrm", 40, "topographic"),
    FeatureSpec("lrm_abs_r15", "lrm_abs", 15, "topographic"),
    FeatureSpec("lrm_grad_r15", "lrm_grad", 15, "topographic"),
    # ---- topographic: openness and anti-slope ------------------------------------
    FeatureSpec("open_pos_l10", "openness_pos", 10, "topographic"),
    FeatureSpec("open_neg_l10", "openness_neg", 10, "topographic"),
    FeatureSpec("open_antislope_l10", "openness_anti", 10, "topographic"),
    FeatureSpec("open_neg_l30", "openness_neg", 30, "topographic"),
    FeatureSpec("open_antislope_l30", "openness_anti", 30, "topographic"),
    # ---- topographic: Hessian ridge / valley response ----------------------------
    FeatureSpec("hessian_ridge_s3", "hessian_ridge", 3, "topographic"),
    FeatureSpec("hessian_ridge_s8", "hessian_ridge", 8, "topographic"),
    FeatureSpec("hessian_curv_s3", "hessian_curv", 3, "topographic"),
    FeatureSpec("hessian_curv_s8", "hessian_curv", 8, "topographic"),
    # ---- potential field: edge detectors ----------------------------------------
    FeatureSpec("thg_gravity_s3", "total_hgrad", 3, "potential_field"),
    FeatureSpec("thg_gravity_s10", "total_hgrad", 10, "potential_field"),
    FeatureSpec("thg_magnetic_s3", "total_hgrad_mag", 3, "potential_field"),
    FeatureSpec("thg_magnetic_s10", "total_hgrad_mag", 10, "potential_field"),
    FeatureSpec("tilt_tc_s5", "tilt_curv", 5, "potential_field"),
    # ---- structural: fabric alignment + multi-scale step energy ------------------
    FeatureSpec("fabric_align_s5", "fabric_align", 5, "structural"),
    FeatureSpec("fabric_align_s15", "fabric_align", 15, "structural"),
    FeatureSpec("step_energy_s5", "step_energy", 5, "structural"),
    FeatureSpec("step_energy_s15", "step_energy", 15, "structural"),
    FeatureSpec("slope_multiscale_ratio", "slope_ratio", 15, "structural"),
]

# Bands actually read by the library above.
NEEDED_BANDS = [
    "detrended_elevation", "detrended_elevation_slope",
    "isostatic_grav_anomaly", "isostatic_grav_hgrad", "isostatic_grav_vgrad",
    "magnetic_anomaly", "tmi_total", "tmi_hgrad", "tmi_vgrad", "rtp_magnetic",
    "tilt_angle_tot_curv",
]

# Basin & Range fault fabric: N-S to NE-SW normal faults dominate the GeoDAWN
# region.  Two azimuths (0 deg = north, 45 deg = NE) are scored; a lineament is
# "fabric aligned" when its gradient normal is within 30 deg of one of them.
FABRIC_AZIMUTHS_DEG = (0.0, 45.0, 90.0)


def _fill(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a[a <= NODATA] = np.nan
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    med = float(np.median(a[finite]))
    return np.where(finite, a, med).astype(np.float32)


def _rays(length: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Integer ray offsets for the 8 connected directions, 1..length pixels out."""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for dy, dx in dirs:
        ys = np.array([dy * t for t in range(1, length + 1)], dtype=np.int64)
        xs = np.array([dx * t for t in range(1, length + 1)], dtype=np.int64)
        out.append((ys, xs))
    return out


def _shift(a: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """a shifted by (dy, dx) with wrapped entries replaced by ``fill``."""
    h, w = a.shape
    sh = np.roll(np.roll(a, -dy, axis=0), -dx, axis=1)
    if dy > 0:
        sh[h - dy:, :] = fill
    elif dy < 0:
        sh[:-dy, :] = fill
    if dx > 0:
        sh[:, w - dx:] = fill
    elif dx < 0:
        sh[:, :-dx] = fill
    return sh


def openness(z: np.ndarray, length: int, positive: bool = True) -> np.ndarray:
    """Topographic openness (Yokoyama et al. 2002), 8-direction ray version.

        positive openness: pi/2 - mean_theta(atan((max_ray z - z) / dist))
        negative openness: pi/2 - mean_theta(atan((z - min_ray z) / dist))

    Physically: how much sky a point sees (positive) and how enclosed it is
    (negative).  A fault scarp produces a sharp *contrast* between positive and
    negative openness across its trace, which is why the anti-slope difference
    (negative - positive) is the classical scarp operator.

    Distances are in pixels; the arctangent is only ever used comparatively, so
    the 100 m cell size is a common factor and is omitted.  Implemented with a
    running extreme along each ray (no [L, H, W] stack), so it is O(8 * L * H * W)
    time and O(H * W) memory.
    """
    h, w = z.shape
    zf = z.astype(np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        step = float(np.hypot(dy, dx))
        extreme = np.full((h, w), -np.inf if positive else np.inf, dtype=np.float32)
        for t in range(1, length + 1):
            sh = _shift(zf, dy * t, dx * t, -np.inf if positive else np.inf)
            extreme = np.maximum(extreme, sh) if positive else np.minimum(extreme, sh)
            num = (extreme - zf) if positive else (zf - extreme)
            ang = np.arctan2(num, step * t)
            total += np.where(np.isfinite(ang), ang, 0.0)
    return (np.pi / 2.0 - total / 8.0).astype(np.float32)


def _hessian(a: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-smoothed second derivatives of ``a`` (pixels)."""
    dyy = ndimage.gaussian_filter(a, sigma, order=(0, 2), mode="nearest")
    dxx = ndimage.gaussian_filter(a, sigma, order=(2, 0), mode="nearest")
    dxy = ndimage.gaussian_filter(a, sigma, order=(1, 1), mode="nearest")
    return dyy, dxx, dxy


def hessian_response(a: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge/valley (Frangi-style) response and total curvature.

    ``ridge``  = |lambda_min| when lambda_min < 0 <= lambda_max or the mirror
                 (a 1-D ridge or trough: one strongly curved, one flat
                 direction) -- this is the ridge/valley detector.
    ``curv``   = |lambda_min| + |lambda_max| -- overall bending energy, the
                 isotropic curvature operator (part of the free-air/upward
                 continuation family of edge detectors in gravity work).
    """
    dyy, dxx, dxy = _hessian(a.astype(np.float32), sigma)
    tr = dyy + dxx
    det = dyy * dxx - dxy * dxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0)).astype(np.float32)
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    lo = np.minimum(l1, l2)
    hi = np.maximum(l1, l2)
    ridge = np.where((lo < 0.0) & (hi >= 0.0), np.abs(lo), 0.0).astype(np.float32)
    curv = (np.abs(lo) + np.abs(hi)).astype(np.float32)
    return ridge, curv


def total_horizontal_gradient(a: np.ndarray, sigma: float) -> np.ndarray:
    """Total horizontal derivative sqrt((d/dx)^2 + (d/dy)^2) of a smoothed field.

    This is the classical potential-field edge locator: over a vertical contact
    or a fault with a small throw, the horizontal gradient peaks on the edge,
    and the peak location does not move with the field's amplitude, which is why
    it is used instead of the raw derivative.
    """
    s = ndimage.gaussian_filter(a.astype(np.float32), sigma, mode="nearest")
    gy, gx = np.gradient(s)
    return np.hypot(gy, gx).astype(np.float32)


def fabric_alignment(a: np.ndarray, sigma: float) -> np.ndarray:
    """Fraction of the local gradient energy aligned with the regional fault fabric.

    ``a`` should be the detrended elevation.  The gradient direction is the
    direction of steepest slope; a fault scarp's gradient is *perpendicular* to
    its trace, so measuring the alignment of the gradient with the fabric normals
    of (N0E, N45E, N90E) traces is the same as measuring strike alignment.

    Returns cos^2 of the smallest angle to any fabric normal (1 = perfectly
    fabric aligned, 0 = perpendicular to every fabric direction).
    """
    s = ndimage.gaussian_filter(a.astype(np.float32), sigma, mode="nearest")
    gy, gx = np.gradient(s)
    norm = np.hypot(gy, gx) + 1e-9
    out = np.zeros(a.shape, dtype=np.float32)
    for az in FABRIC_AZIMUTHS_DEG:
        th = np.deg2rad(90.0 - az)          # gradient normal of a trace at azimuth az
        c = np.abs(np.cos(th) * gx / norm + np.sin(th) * gy / norm)
        out = np.maximum(out, c.astype(np.float32))
    # cos^2 is the energy fraction; (2c^2-1)^+ is a margin from the 45 deg boundary
    return np.maximum(2.0 * out * out - 1.0, 0.0).astype(np.float32)


def step_energy(a: np.ndarray, sigma: float) -> np.ndarray:
    """Multi-scale step energy: |laplacian| of the smoothed field, edge-normalised.

    A fault scarp is a step; the Laplacian of the surface is a quadrature-matched
    filter for a step at the scale sigma, and dividing by the local gradient
    magnitude makes the response depend on the *sharpness* of the break rather
    than on how steep the whole hillslope is.
    """
    s = ndimage.gaussian_filter(a.astype(np.float32), sigma, mode="nearest")
    lap = np.abs(ndimage.laplace(s))
    g = total_horizontal_gradient(s, max(0.5, sigma / 2.0)) + 1e-6
    return (lap / g).astype(np.float32)


def build_stage_b_feature(name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Dispatch one named feature.  ``bands`` are raw float32 arrays with nodata."""
    z = _fill(bands["detrended_elevation"])
    zs = _fill(bands["detrended_elevation_slope"])
    grav = _fill(bands["isostatic_grav_anomaly"])
    mag = _fill(bands["tmi_total"])
    for spec in STAGE_B_SPEC:
        if spec.name != name:
            continue
        k, s = spec.kind, spec.scale
        if k == "lrm":
            return (z - ndimage.uniform_filter(z, size=2 * s + 1)).astype(np.float32)
        if k == "lrm_abs":
            return np.abs(z - ndimage.uniform_filter(z, size=2 * s + 1)).astype(np.float32)
        if k == "lrm_grad":
            lrm = z - ndimage.uniform_filter(z, size=2 * s + 1)
            return total_horizontal_gradient(lrm, 2.0)
        if k == "openness_pos":
            return openness(z, s, positive=True)
        if k == "openness_neg":
            return openness(z, s, positive=False)
        if k == "openness_anti":
            return openness(z, s, positive=False) - openness(z, s, positive=True)
        if k == "hessian_ridge":
            return hessian_response(z, float(s))[0]
        if k == "hessian_curv":
            return hessian_response(z, float(s))[1]
        if k == "total_hgrad":
            return total_horizontal_gradient(grav, float(s))
        if k == "total_hgrad_mag":
            return total_horizontal_gradient(mag, float(s))
        if k == "tilt_curv":
            tc = _fill(bands["tilt_angle_tot_curv"])
            return np.abs(ndimage.gaussian_filter(tc, float(s), mode="nearest")).astype(np.float32)
        if k == "fabric_align":
            return fabric_alignment(z, float(s)) * np.abs(ndimage.sobel(ndimage.gaussian_filter(z, float(s)), axis=0)).astype(np.float32)
        if k == "step_energy":
            return step_energy(z, float(s))
        if k == "slope_ratio":
            a = ndimage.uniform_filter(zs, size=s)
            b = ndimage.uniform_filter(zs, size=1) + 1e-6
            return (a / b).astype(np.float32)
        raise KeyError(f"unknown feature kind {k}")
    raise KeyError(f"unknown Stage B feature {name}")


def feature_names() -> list[str]:
    return [f.name for f in STAGE_B_SPEC]
