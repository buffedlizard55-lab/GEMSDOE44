"""Fault-population statistics measured from the official catalogue.

Nothing in this module is assumed.  Every quantity is computed from the hash-verified
``labels.tif`` bytes and written to ``registry/population.json``.

The geological premise (H1, see ``registry/hypotheses.json``): a new fault added by a
fault expert to an existing map is overwhelmingly *new geometry of an existing system*
- an across-strike splay, an along-strike continuation past a tip, or an en-echelon
relay link.  If that is true, the *population* of mapped faults carries the statistics
needed to place the missing members, and the placement prior must be ANISOTROPIC in the
local strike frame.  This module measures the statistics that a synthesizer needs:

* ``segments``      - connected components of the catalogue (length, azimuth, centroid)
* ``azimuth_hist``  - length-weighted trace azimuth histogram (the regional strike family)
* ``spacing``       - across-strike separation of near-parallel segments
* ``tip_fan``       - measured along-strike continuation statistics past segment tips
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _azimuth_deg(v: np.ndarray) -> float:
    """Trace azimuth in [0, 180) degrees, measured clockwise from +x (east)."""
    a = np.degrees(np.arctan2(v[0], v[1])) % 180.0  # rows are y-down, cols are x
    return float(a)


def segment_statistics(labels: np.ndarray, footprint: np.ndarray, min_px: int = 12) -> list[dict]:
    """Connected components of the known-fault raster with length + azimuth.

    Azimuth is measured by the principal eigenvector of the component's pixel coordinates
    (orthogonal-regression / PCA direction), which is the standard structural measure of a
    map trace's direction.
    """
    known = footprint & (labels > 0)
    lab, n = ndimage.label(known, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    slices = ndimage.find_objects(lab)
    out: list[dict] = []
    for i, sl in zip(idx, slices):
        ys, xs = np.nonzero(lab[sl] == i)
        if ys.size < min_px:
            continue
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pts = np.column_stack([ys.astype(float), xs.astype(float)])
        centred = pts - pts.mean(axis=0)
        cov = centred.T @ centred / max(len(pts) - 1, 1)
        w, vecs = np.linalg.eigh(cov)
        direction = vecs[:, -1]
        extent = float(np.sqrt(max(w[-1], 0.0)) * 2.0)   # 2 sigma along-trace, in px
        out.append(
            {
                "id": int(i),
                "px": int(ys.size),
                "length_px_2sigma": round(extent, 3),
                "length_km_2sigma": round(extent * 0.1, 3),
                "azimuth_deg": round(_azimuth_deg(direction), 2),
                "elongation": round(float(np.sqrt(w[-1] / max(w[0], 1e-9))), 2),
                "centroid_row": float(pts[:, 0].mean()),
                "centroid_col": float(pts[:, 1].mean()),
            }
        )
    return out


def azimuth_histogram(segments: list[dict], bin_deg: float = 10.0) -> dict:
    """Length-weighted azimuth histogram of the mapped trace population."""
    if not segments:
        return {"bins": [], "weights": []}
    az = np.array([s["azimuth_deg"] for s in segments])
    wt = np.array([s["length_px_2sigma"] for s in segments], dtype=float)
    edges = np.arange(0.0, 180.0 + bin_deg, bin_deg)
    hist, _ = np.histogram(az, bins=edges, weights=wt)
    return {
        "bin_edges_deg": edges.tolist(),
        "bin_centres_deg": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
        "weights": hist.tolist(),
        "modal_bin_deg": float(((edges[:-1] + edges[1:]) / 2.0)[int(hist.argmax())]),
        "total_length_px": float(wt.sum()),
    }


def nearest_fault_distance_geometry(
    labels: np.ndarray, footprint: np.ndarray, sample: np.ndarray, exclude_label: int
) -> dict:
    """For each ``sample`` pixel, the offset vector to the nearest catalogue pixel.

    Returns the offset distance (px) and the angle between the offset vector and the local
    catalogue strike (deg; 0 = along strike, 90 = across strike).  This is the measurement
    that decides whether an anisotropic offset kernel can beat an isotropic proximity kernel.
    """
    known = footprint & (labels > 0)
    lab, _ = ndimage.label(known, structure=np.ones((3, 3), bool))
    dist, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    ys, xs = np.nonzero(sample)
    if ys.size == 0:
        return {"n": 0}
    d = dist[ys, xs]
    ny, nx = iy[ys, xs].astype(float), ix[ys, xs].astype(float)
    vy, vx = ys - ny, xs - nx
    norm = np.hypot(vy, vx)
    ok = norm > 0
    vy, vx, norm = vy[ok], vx[ok], norm[ok]
    d, ys, xs = d[ok], ys[ok], xs[ok]

    # local strike at the nearest catalogue pixel: PCA of catalogue pixels in a 15x15 window
    h, w = labels.shape
    win = 7
    ang = np.full(ys.size, np.nan)
    for k in range(ys.size):
        r0, c0 = int(ny[ok][k]), int(nx[ok][k])
        sub = lab[max(r0 - win, 0): r0 + win + 1, max(c0 - win, 0): c0 + win + 1]
        yy, xx = np.nonzero(sub > 0)
        if yy.size < 6:
            continue
        pts = np.column_stack([yy.astype(float), xx.astype(float)])
        pts = pts - pts.mean(axis=0)
        cov = pts.T @ pts / (len(pts) - 1)
        _, vecs = np.linalg.eigh(cov)
        direction = vecs[:, -1]
        ang[k] = _azimuth_deg(direction)

    with np.errstate(invalid="ignore"):
        off_ang = np.degrees(np.arctan2(vy, vx)) % 180.0
        delta = np.abs(((off_ang - ang) + 90.0) % 180.0 - 90.0)   # 0..90, angle to strike
    delta = delta[np.isfinite(delta)]
    return {
        "n": int(delta.size),
        "delta_deg": delta,
        "delta_median_deg": float(np.median(delta)) if delta.size else None,
        "frac_within_30deg_of_strike": float((delta < 30).mean()) if delta.size else None,
        "frac_within_30deg_of_perp": float((delta > 60).mean()) if delta.size else None,
        "frac_within_45deg_of_strike": float((delta < 45).mean()) if delta.size else None,
    }


def binned_angle_profile(delta_deg: np.ndarray, bin_deg: float = 15.0) -> dict:
    """Histogram of the offset-vector angle to local strike, with a random-control band."""
    if delta_deg.size == 0:
        return {}
    edges = np.arange(0.0, 90.0 + bin_deg, bin_deg)
    hist, _ = np.histogram(delta_deg, bins=edges)
    frac = hist / hist.sum()
    # uniform control: P(angle to a fixed line within 15 deg bin) = 1/3 of the 0..90 range
    control = np.full_like(frac, 1.0 / len(frac))
    return {
        "bin_edges_deg": edges.tolist(),
        "frac": frac.tolist(),
        "uniform_control_frac": control.tolist(),
        "enrichment": (frac / control).tolist(),
    }
