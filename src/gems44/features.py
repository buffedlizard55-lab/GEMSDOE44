"""Feature engineering for the two-stage system.

Stage-2 channels (per pixel):
  RAW  (19) : robust z-scores of the official bands (depth_to_base sign-flipped)
  H1   (4)  : cross-physics structure-tensor consensus on 4 independent
              physics fields (topography, magnetics, gravity, radiometrics)
              at two coherence scales
  H2   (1)  : strike inheritance from the 345 NBMG Qfaults gap links
  H3   (2)  : relay-zone + endpoint-uncertainty surfaces (catalog gap geometry)
  H4   (1)  : hydrothermal-plumbing composite (conductivity x shallow
              conductive base x high heat-flow well density)
  H5   (1)  : seismicity lineament coherence x magnetic edge witness

Stage-1 channels (per 2 km superpixel): block statistics of the bands plus
relay/gap counts and heat-flow well counts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import config as C


# ----------------------------------------------------------------------------
# small numerics
# ----------------------------------------------------------------------------

def _fill_nan(a: np.ndarray, foot: np.ndarray) -> np.ndarray:
    a = a.copy()
    bad = np.isnan(a) & foot
    if bad.any():
        m = np.nanmedian(a[foot & ~np.isnan(a)])
        a[bad] = m if np.isfinite(m) else 0.0
    a[~foot] = np.nan
    return a


def zscore_robust(a: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """(x - median) / (p90 - p10) over valid footprint cells, clipped to [-4, 4]."""
    out = np.full_like(a, 0.0, dtype=np.float32)
    v = a[foot & np.isfinite(a)]
    if v.size == 0:
        return out
    med = np.median(v)
    lo, hi = np.percentile(v, [10, 90])
    sd = max(hi - lo, 1e-9)
    # in-place ops: keep transients to a single full-grid buffer (~49 MB)
    np.subtract(a, med, out=out)
    np.divide(out, sd, out=out)
    np.clip(out, -4.0, 4.0, out=out)
    out[~foot] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def structure_tensor(f: np.ndarray, sigma_grad: float, sigma_c: float):
    """Second-moment (structure) tensor of field f.

    Returns (coherence, cos2theta, sin2theta) where theta is the lineament
    orientation (mod pi).  Coherence = (l1-l2)/(l1+l2).

    NaN cells (outside the survey footprint) are filled with the finite
    median BEFORE filtering: a Gaussian kernel would otherwise propagate
    NaN into the whole array.
    """
    f = np.asarray(f, np.float32)
    if np.isnan(f).any():
        f = f.copy()
        med = np.nanmedian(f[np.isfinite(f)])
        f[~np.isfinite(f)] = med if np.isfinite(med) else 0.0
    g = ndimage.gaussian_filter(f, sigma_grad)
    fx = ndimage.sobel(g, axis=1)
    del g
    fy = ndimage.sobel(ndimage.gaussian_filter(f, sigma_grad), axis=0)
    # second moments, reusing two scratch buffers to cap transients
    tmp = np.empty_like(fx)
    tmp2 = np.empty_like(fx)
    np.multiply(fx, fx, out=tmp)
    a = ndimage.uniform_filter(tmp, sigma_c)
    np.multiply(fx, fy, out=tmp)
    b = ndimage.uniform_filter(tmp, sigma_c)
    np.multiply(fy, fy, out=tmp)
    c = ndimage.uniform_filter(tmp, sigma_c)
    del fx, fy
    d = a - c
    tr = a + c
    np.multiply(d, 0.5, out=tmp)
    np.square(tmp, out=tmp)
    np.multiply(b, b, out=tmp2)
    np.add(tmp, tmp2, out=tmp)
    np.maximum(tmp, 0.0, out=tmp)
    det = np.sqrt(tmp, out=tmp)
    del a, c
    # coh = (l1-l2)/(l1+l2) = 2*det / tr
    np.multiply(det, 2.0, out=tmp2)
    np.add(tr, 1e-12, out=tr)
    coh = np.divide(tmp2, tr, out=tmp2)
    del det, tr
    # orientation: ang2 = 2*theta = arctan2(2b, a-c); return cos/sin(2*theta)
    np.multiply(b, 2.0, out=tmp)
    ang2 = np.arctan2(tmp, d, out=tmp)
    del d, b
    mask = coh > 0.05
    np.cos(ang2, out=tmp2)
    np.multiply(tmp2, mask, out=tmp2)
    cos2 = tmp2
    np.sin(ang2, out=tmp)
    np.multiply(tmp, mask, out=tmp)
    sin2 = tmp
    return coh.astype(np.float32), cos2, sin2


def ang_agreement(cos2a: np.ndarray, sin2a: np.ndarray,
                  cos2b: np.ndarray, sin2b: np.ndarray) -> np.ndarray:
    """cos(2*dtheta) in [-1, 1] between two orientation fields (mod pi)."""
    return (cos2a * cos2b + sin2a * sin2b).astype(np.float32)


# ----------------------------------------------------------------------------
# H1: cross-physics lineament consensus
# ----------------------------------------------------------------------------

H1_FIELDS = ("det_elev", "rtp", "iso_grav_anom", "tc")


def build_h1(grid, foot: np.ndarray):
    """Return dict of channels: pair_hi, pair_lo, cmax2_hi, cmax2_lo (float32 HxW).

    Per field, per coherence scale s:
        R_s = robust z-score of (lambda1 * coherence)  [gradient energy x
              linearity] - sparse: near zero on flat areas, large on strong
              lineaments.
    Cross-physics consensus:
        pair_s  = max over the 6 field pairs of R_i * R_j * max(0, cos2(dtheta))
        cmax2_s = second largest coherence among the 4 fields at scale s

    A pixel only scores high when two INDEPENDENT physics fields both show a
    strong lineament in the same orientation - the discriminator single-field
    ridge detectors lack.
    """
    names = list(H1_FIELDS)
    out = {}
    # energy proxy |grad| per field (scale-independent, computed once)
    import scipy.ndimage as ndi
    E_all = {}
    for nm in names:
        f = grid.band_array(nm)
        f = np.where(np.isfinite(f), f, np.nanmedian(f[np.isfinite(f)])).astype(np.float32)
        gx = ndi.sobel(ndi.gaussian_filter(f, 1.5), axis=1)
        gy = ndi.sobel(ndi.gaussian_filter(f, 1.5), axis=0)
        E_all[nm] = np.hypot(gx, gy).astype(np.float32)
        del f, gx, gy
    # Build scale-by-scale: holding both scales' tensors at once cost ~1.2 GB
    # extra, which pushed the process over the 3.7 GB cgroup cap.
    buf = np.zeros(foot.shape, np.float32)
    buf2 = np.zeros(foot.shape, np.float32)
    for s in (2, 4):
        tensors = {
            name: structure_tensor(_fill_nan(grid.band_array(name), foot),
                                   sigma_grad=1.5, sigma_c=float(s))
            for name in names
        }
        best_pair = np.zeros(foot.shape, np.float32)
        # positive-part z-score energy (pairs only use max(Rz, 0)); in-place
        Rz = {}
        for nm in names:
            r = zscore_robust(E_all[nm] * tensors[nm][0], foot)
            np.maximum(r, 0.0, out=r)
            Rz[nm] = r
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c2_i, s2_i = tensors[names[i]][1], tensors[names[i]][2]
                c2_j, s2_j = tensors[names[j]][1], tensors[names[j]][2]
                # agree = max(0, cos2i*cos2j + sin2i*sin2j), in two buffers
                np.multiply(c2_i, c2_j, out=buf)
                np.multiply(s2_i, s2_j, out=buf2)
                np.add(buf, buf2, out=buf)
                np.maximum(buf, 0.0, out=buf)
                np.multiply(Rz[names[i]], Rz[names[j]], out=buf2)
                np.multiply(buf2, buf, out=buf2)
                np.maximum(best_pair, buf2, out=best_pair)
        # second-largest of 4 coherences via pairwise tournament:
        # second = max(min(m1,m2), max(l1,l2)) with m/l = max/min of pairs.
        # (The original np.partition(cohs, 2, axis=0)[1] returned the
        # 2nd-SMALLEST coherence - an un-ordered index below kth - a bug;
        # this implements the documented intent.)  No 196 MB stack/partition;
        # best_pair is scratch once the pair channel is materialised.
        pair_out = np.where(foot, best_pair, 0.0).astype(np.float32)
        c0, c1, c2, c3 = (tensors[nm][0] for nm in names)
        np.maximum(c0, c1, out=buf)                       # m1
        np.maximum(c2, c3, out=buf2)                      # m2
        np.minimum(buf, buf2, out=best_pair)              # min(m1, m2)
        np.minimum(c0, c1, out=buf)                       # l1
        np.minimum(c2, c3, out=buf2)                       # l2
        np.maximum(buf, buf2, out=buf)                     # max(l1, l2)
        np.maximum(best_pair, buf, out=best_pair)          # second largest
        suf = "hi" if s == 2 else "lo"
        out[f"pair_{suf}"] = pair_out
        np.maximum(best_pair, 0.0, out=best_pair)
        out[f"cmax2_{suf}"] = np.where(foot, best_pair, 0.0).astype(np.float32)
        del tensors, Rz, best_pair, pair_out
    return out


# ----------------------------------------------------------------------------
# H2/H3: NBMG Qfaults gap links (official vector mirror, 345 features)
# ----------------------------------------------------------------------------

@dataclass
class GapLinks:
    mid_row: np.ndarray = field(default_factory=lambda: np.array([]))
    mid_col: np.ndarray = field(default_factory=lambda: np.array([]))
    end_rows: np.ndarray = field(default_factory=lambda: np.array([]))
    end_cols: np.ndarray = field(default_factory=lambda: np.array([]))
    cos2: np.ndarray = field(default_factory=lambda: np.array([]))
    sin2: np.ndarray = field(default_factory=lambda: np.array([]))
    weight: np.ndarray = field(default_factory=lambda: np.array([]))
    relay_weight: np.ndarray = field(default_factory=lambda: np.array([]))
    n: int = 0
    n_relay: int = 0


def lonlat_to_rc(lon: np.ndarray, lat: np.ndarray):
    """Approximate lon/lat -> row/col on the UTM11 grid (small region, ~100 km).

    Uses the grid's own origin: convert one reference point by pyproj and
    offset in meters.  Error < 1 px across the study area.
    """
    import pyproj
    wgs84 = pyproj.CRS.from_epsg(4326)
    utm11 = pyproj.CRS.from_epsg(32611)
    tr = pyproj.Transformer.from_crs(wgs84, utm11, always_xy=True)
    e, n = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    col = (e - C.ORIGIN_E) / C.CELL_M
    row = (C.ORIGIN_N - n) / C.CELL_M
    return row, col


def load_gap_links() -> GapLinks:
    gj = json.load(open(C.TOPOLOGY_GEOJSON))
    rows, cols = [], []
    er, ec = [], []
    c2, s2, w, rw = [], [], [], []
    for f in gj["features"]:
        g = f.get("geometry") or {}
        coords = g.get("coordinates") if g.get("type") == "LineString" else None
        if not coords or len(coords) < 2:
            continue
        (lon1, lat1), (lon2, lat2) = coords[0], coords[-1]
        r1, c1 = lonlat_to_rc(np.array([lon1]), np.array([lat1]))
        r2, c2v = lonlat_to_rc(np.array([lon2]), np.array([lat2]))
        r1, c1, r2, c2v = int(round(r1[0])), int(round(c1[0])), int(round(r2[0])), int(round(c2v[0]))
        p = f["properties"]
        gap_km = float(p.get("gap_km") or 0.0)
        kin = bool(p.get("kinematic_compat"))
        scompat = float(p.get("strike_compat") or 0.0)
        strike = p.get("strike")
        if strike in (None, "") or float(strike) == 0.0:
            # fall back to the gap-bridge azimuth
            dx = c2v - c1
            dy = -(r2 - r1)
            strike = np.degrees(np.arctan2(dx, dy)) % 180.0
        th = np.radians(float(strike))
        # endpoint uncertainty weight: weaker catalog classes flag more uncertainty
        def fw(ft):
            if ft == "Inferred":
                return 1.0
            if ft == "Moderately Constrained":
                return 0.6
            return 0.3
        wend = max(fw(p.get("ftype_src")), fw(p.get("ftype_tgt")))
        er += [r1, r2]
        ec += [c1, c2v]
        if gap_km < 3.0 and (kin or scompat > 0.5):
            mr = (r1 + r2) / 2.0
            mc = (c1 + c2v) / 2.0
            rows.append(mr)
            cols.append(mc)
            c2.append(np.cos(2 * th))
            s2.append(np.sin(2 * th))
            w.append(scompat if scompat > 0 else 0.5)
            rw.append(1.0 if kin else 0.5)
    gl = GapLinks(
        mid_row=np.asarray(rows), mid_col=np.asarray(cols),
        end_rows=np.asarray(er), end_cols=np.asarray(ec),
        cos2=np.asarray(c2), sin2=np.asarray(s2),
        weight=np.asarray(w), relay_weight=np.asarray(rw),
        n=len(rows),
    )
    gl.n_relay = int((gl.relay_weight >= 1.0).sum())
    return gl


def _stamp_gaussians(h: int, w: int, rows: np.ndarray, cols: np.ndarray,
                     weights: np.ndarray, sigma: float) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    if rows.size == 0:
        return out
    rad = int(np.ceil(3 * sigma))
    lo_r = int(max(rows.min() - rad, 0))
    hi_r = int(min(rows.max() + rad, h - 1))
    lo_c = int(max(cols.min() - rad, 0))
    hi_c = int(min(cols.max() + rad, w - 1))
    for r, c, wt in zip(rows, cols, weights):
        r, c, wt = int(round(r)), int(round(c)), float(wt)
        if not (0 <= r < h and 0 <= c < w) or wt <= 0:
            continue
        rr0, rr1 = max(r - rad, 0), min(r + rad + 1, h)
        cc0, cc1 = max(c - rad, 0), min(c + rad + 1, w)
        gr = np.arange(rr0, rr1) - r
        gc = np.arange(cc0, cc1) - c
        d2 = gr[:, None] ** 2 + gc[None, :] ** 2
        out[rr0:rr1, cc0:cc1] += wt * np.exp(-d2 / (2.0 * sigma * sigma))
    return out


def build_h2h3(foot: np.ndarray, gl: GapLinks):
    """H2 strike inheritance (distance-decayed cos2 agreement with gap-link
    strike) and H3 relay / endpoint-uncertainty surfaces."""
    h, w = foot.shape
    relay = _stamp_gaussians(h, w, gl.mid_row, gl.mid_col, gl.relay_weight, sigma=2.0)
    endpt = _stamp_gaussians(h, w, gl.end_rows, gl.end_cols,
                             np.ones(gl.end_rows.shape, np.float32), sigma=2.0)
    dmap = np.full((h, w), 120.0, np.float32)
    c2f = np.zeros((h, w), np.float32)
    s2f = np.zeros((h, w), np.float32)
    if gl.n:
        lab = np.zeros((h, w), np.int64)
        for i in range(gl.n):
            r = int(round(gl.mid_row[i]))
            c = int(round(gl.mid_col[i]))
            if 0 <= r < h and 0 <= c < w:
                lab[r, c] = 1
        dmap = np.clip(ndimage.distance_transform_edt(lab == 0), 0, 120).astype(np.float32)
        # nearest-link orientation inside a 60 px window around each link
        # (290 links x 121x121 windows - cheap)
        dbest = np.full((h, w), np.inf, np.float32)
        RAD = 60
        for i in range(gl.n):
            r = int(round(gl.mid_row[i]))
            c = int(round(gl.mid_col[i]))
            r0, r1 = max(r - RAD, 0), min(r + RAD + 1, h)
            c0, c1 = max(c - RAD, 0), min(c + RAD + 1, w)
            if r0 >= r1 or c0 >= c1:
                continue
            gr = np.arange(r0, r1) - r
            gc = np.arange(c0, c1) - c
            d2 = gr[:, None] ** 2 + gc[None, :] ** 2
            sel = d2 < dbest[r0:r1, c0:c1]
            if not sel.any():
                continue
            dbest[r0:r1, c0:c1][sel] = d2[sel]
            c2f[r0:r1, c0:c1][sel] = gl.cos2[i]
            s2f[r0:r1, c0:c1][sel] = gl.sin2[i]
        zero = dmap > 60
        c2f[zero] = 0.0
        s2f[zero] = 0.0
    return {
        "relay": np.where(foot, relay, 0.0).astype(np.float32),
        "endpt": np.where(foot, endpt, 0.0).astype(np.float32),
        "inherit_c2": np.where(foot, c2f, 0.0).astype(np.float32),
        "inherit_s2": np.where(foot, s2f, 0.0).astype(np.float32),
        "d_relay": np.where(foot, dmap, 120.0).astype(np.float32),
    }


def h2_inheritance_channel(h2h3: dict, pair_or: np.ndarray,
                           cos2_or: np.ndarray, sin2_or: np.ndarray) -> np.ndarray:
    """H2 channel: cos2 agreement between the pixel's own lineament orientation
    (from the H1 consensus orientation) and the nearest gap-link strike,
    decayed by distance."""
    agree = h2h3["inherit_c2"] * cos2_or + h2h3["inherit_s2"] * sin2_or
    decay = np.exp(-h2h3["d_relay"] / 20.0)
    return np.clip(agree, -1, 1) * decay


# ----------------------------------------------------------------------------
# H4 / H5
# ----------------------------------------------------------------------------

def build_h4h5(grid, foot: np.ndarray,
               well_rows: np.ndarray, well_cols: np.ndarray,
               well_hf: np.ndarray):
    cond = zscore_robust(_fill_nan(grid.band_array("cond_surf"), foot), foot)
    depth = zscore_robust(_fill_nan(grid.band_array("depth_to_base_surf"), foot), foot)
    # shallow conductive base = LOW depth -> negate (in-place: no transients)
    np.clip(cond, 0.0, None, out=cond)
    np.negative(depth, out=depth)
    np.clip(depth, 0.0, None, out=depth)
    h4 = np.multiply(cond, depth, out=cond)
    del depth
    # high heat-flow wells within 5 km
    hf_map = np.zeros(foot.shape, np.float32)
    if well_rows.size:
        hot = well_hf >= 1500.0
        hf_map = _stamp_gaussians(foot.shape[0], foot.shape[1],
                                  well_rows[hot], well_cols[hot],
                                  np.ones(int(hot.sum()), np.float32), sigma=25.0)
        hf_map[~(hf_map > 0.05)] = 0.0
        hf_map[hf_map > 0.0] = 1.0
        h4 *= hf_map
        del hf_map
    # H5: seismicity structure-tensor coherence x magnetic edge witness
    deq = _fill_nan(grid.band_array("deq_n100a15"), foot)
    coh_deq, _, _ = structure_tensor(deq, sigma_grad=1.5, sigma_c=4.0)
    del deq
    edge = zscore_robust(_fill_nan(grid.band_array("tmi_hg"), foot), foot)
    np.clip(edge, 0.0, None, out=edge)
    h5 = np.multiply(coh_deq, edge, out=edge)
    del coh_deq
    return {
        "h4_conduit": np.where(foot, h4, 0.0).astype(np.float32),
        "h5_seis_edge": np.where(foot, h5, 0.0).astype(np.float32),
    }


# ----------------------------------------------------------------------------
# full stage-2 channel stack
# ----------------------------------------------------------------------------

RAW_SIGN = {  # band name -> sign (+1 keep, -1 flip)
    **{n: 1 for n in C.BAND_NAMES},
    "depth_to_base_surf": -1,
}


def build_stage2_channels(grid, foot: np.ndarray,
                          well_rows: np.ndarray, well_cols: np.ndarray,
                          well_hf: np.ndarray) -> tuple[dict, dict]:
    """Return (channels dict name->(H,W) float32, meta).

    Bands are streamed from disk one at a time (float32) to stay inside the
    sandbox RAM budget.
    """
    ch = {}
    # Order matters for RAM: every heavy transient (h1 tensors, h2/h45
    # structure tensors) is built BEFORE the 19 z-band stack (933 MB) is
    # accumulated in `ch`, so transients never stack on the full dict.
    h1 = build_h1(grid, foot)
    ch.update(h1)
    gl = load_gap_links()
    h2h3 = build_h2h3(foot, gl)
    ch["relay"] = h2h3["relay"]
    ch["endpt"] = h2h3["endpt"]
    # H2 needs the pixel's own lineament orientation: use the detrended-surface
    # tensor (sigma_c=4) as the proxy orientation field (documented choice).
    f0 = _fill_nan(grid.band_array("det_elev"), foot)
    _, c2o, s2o = structure_tensor(f0, 1.5, 4.0)
    ch["h2_inherit"] = h2_inheritance_channel(h2h3, None, c2o, s2o)
    del f0, c2o, s2o
    h45 = build_h4h5(grid, foot, well_rows, well_cols, well_hf)
    ch.update(h45)
    for name in C.BAND_NAMES:
        a = _fill_nan(grid.band_array(name), foot)
        z = zscore_robust(a, foot)
        del a
        if RAW_SIGN[name] < 0:
            np.negative(z, out=z)
        ch[f"z_{name}"] = z
    meta = {"n_gap_links": gl.n, "n_relay": gl.n_relay}
    return ch, meta


def assemble_X(channels: dict, names: list[str]) -> np.ndarray:
    """(C, H, W) -> (H*W, C) float32."""
    C_ = len(names)
    out = np.empty((channels[names[0]].shape[0] * channels[names[0]].shape[1], C_),
                   dtype=np.float32)
    for i, nm in enumerate(names):
        out[:, i] = channels[nm].ravel()
    return out


def channel_names() -> list[str]:
    names = [f"z_{n}" for n in C.BAND_NAMES]
    names += ["pair_hi", "pair_lo", "cmax2_hi", "cmax2_lo",
              "relay", "endpt", "h2_inherit", "h4_conduit", "h5_seis_edge"]
    return names
