#!/usr/bin/env python3
"""Does excluding the <=200 m catalogue ring help THIS repository's own emission?

Registered as AM-44-03 before this ran.  Two things are measured, both at matched mass so that
placement, not mass, is the only difference:

  A. the ring rule on the shipped field  (blend_U = sup_U**0.5 * prox10**0.5)
  B. the same rule on the analytic proximity field (prox10), as a field-independent control

Every variant is scored with the exact official operator on all four local truth frames:
N (SGMC within 300 m of the catalogue), P (beyond 300 m), Q (200-300 m), R (beyond 200 m),
plus the uniform control at the same mass, per frame.

Writes registry/ring_experiment.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage

sys.path.insert(0, "src")
from gems44 import field as F          # noqa: E402
from gems44 import grid as G           # noqa: E402
from gems44 import metric as M         # noqa: E402

MASSES = [44_090, 61_328, 80_000]              # 34,546 = the sibling session's mass; 80k probes the cap
CAND_FRACTION = 0.25
MIN_SEP = 3.0
RING_PX = 2.0                                   # <= 200 m


def main() -> int:
    g = G.load_grid()
    sgmc = G.read_band("data/external/derived_sgmc_faults_100m_u8.tif", 1) > 0
    d_cat = ndimage.distance_transform_edt(~g.known)
    frames = {
        "N_sgmc_within_300m": sgmc & (d_cat <= 3.0),
        "P_sgmc_beyond_300m": sgmc & (d_cat > 3.0),
        "Q_sgmc_200_300m": sgmc & (d_cat > RING_PX) & (d_cat <= 3.0),
        "R_sgmc_beyond_200m": sgmc & (d_cat > RING_PX),
    }
    base_allowed = g.footprint & ~g.known
    masks = {"ring_included": base_allowed,
             "ring_excluded": base_allowed & (d_cat > RING_PX)}

    # fields
    prox10 = np.exp(-d_cat / 10.0).astype(np.float32)
    belief_u = Path("data/interim/belief_U.npy")
    fields = {"prox10": prox10}
    if belief_u.exists():
        bu = np.load(belief_u).astype(np.float32)
        fields["sup_U"] = bu                      # AM-44-04: the frame-P winner, no catalogue blend
        fields["blend_U"] = (np.sqrt(np.maximum(bu, 0.0) * prox10)).astype(np.float32)
    else:
        print("WARNING: data/interim/belief_U.npy absent; supervised field skipped")

    rng = np.random.default_rng(44)
    allowed_idx = np.nonzero(base_allowed.ravel())[0]
    inc = G.read_band("data/raw/incumbent_d28.tif", 1) > 0
    out: dict = {"written_utc": "2026-10-06", "ring_px": RING_PX, "min_sep_px": MIN_SEP,
                 "masses": MASSES, "design": "AM-44-03; each variant scored with the exact operator"}
    ref = {}
    for fname, truth in frames.items():
        rec = {"truth_px": int(truth.sum()),
               "incumbent_at_44090": round(M.dti(inc, truth, valid=g.footprint, known=g.known).score, 6)}
        rec["uniform"] = {}
        for mass in MASSES:
            u = np.zeros(base_allowed.shape, bool)
            u.ravel()[rng.choice(allowed_idx, size=min(mass, allowed_idx.size), replace=False)] = True
            rec["uniform"][str(mass)] = round(M.dti(u, truth, valid=g.footprint, known=g.known).score, 6)
        ref[fname] = rec
    out["references"] = ref

    results: dict = {}
    for fname, belief in fields.items():
        for mname, allowed in masks.items():
            cand = F.quantile_candidates(belief, allowed, CAND_FRACTION)
            biggest = max(MASSES)
            order = F.emit_order_np(belief, cand, allowed, biggest, min_sep=MIN_SEP)
            print(f"{fname} / {mname}: emitter placed {order.size:,} of {biggest:,} requested",
                  flush=True)
            for mass in MASSES:
                dots = np.zeros(allowed.shape, bool)
                k = min(mass, order.size)
                dots.ravel()[order[:k]] = True
                key = f"{fname}|{mname}|{mass}"
                rec = {"field": fname, "mask": mname, "requested_mass": mass, "emitted": int(k),
                       "dots_in_ring": int((dots & (d_cat <= RING_PX)).sum()),
                       "inner_ring_frac": round(float((dots & (d_cat <= RING_PX)).sum()) / max(k, 1), 5)}
                for fr, truth in frames.items():
                    rec[fr] = round(M.dti(dots, truth, valid=g.footprint, known=g.known).score, 6)
                results[key] = rec
                print("   ", {k2: rec[k2] for k2 in ("emitted", "inner_ring_frac")},
                      {fr: rec[fr] for fr in frames}, flush=True)
    out["variants"] = results

    # paired comparison: ring-excluded minus ring-included at the same field and mass
    paired = {}
    for key in results:
        fld, m, mass = key.split("|")
        if m != "ring_included":
            continue
        other = f"{fld}|ring_excluded|{mass}"
        if other in results:
            paired[f"{fld}|{mass}"] = {fr: round(results[other][fr] - results[key][fr], 6)
                                       for fr in frames}
    out["paired_delta_ring_excluded_minus_included"] = paired
    # excess over the uniform control at the same mass, per frame (the statistic AM-44-04 uses)
    excess = {}
    for key, rec in results.items():
        mass = rec["requested_mass"]
        excess[key] = {fr: round(rec[fr] - ref[fr]["uniform"][str(mass)], 6) for fr in frames}
    out["excess_over_uniform_at_44090_reference"] = excess
    Path("registry/ring_experiment.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("\npaired deltas (ring-excluded minus ring-included):")
    for k, v in paired.items():
        print("  ", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
