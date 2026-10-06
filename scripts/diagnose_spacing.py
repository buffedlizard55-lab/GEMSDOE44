"""Emission-geometry diagnostic: does the minimum dot separation matter, at matched mass?

The metric's algebra is exact and simple: every binary dot raises the denominator by
alpha = 0.2 regardless of where it lands, and raises the numerator by whatever fresh kernel
credit it is the first to reach.  Spacing therefore trades two things: dots closer together
than 2R share kernel credit (the second dot buys little) but cover the same ground, while dots
further apart cover more ground with less overlap.  On an idealised 1-px fault line the credit
of a dot at spacing s (s <= 2R) is  s - s^2/12, which is increasing in s -- so coarser spacing
buys MORE fresh credit per dot, and the family's live ladder (1.5 px -> 0.2477, 2.8 px -> 0.2600,
both from the SAME field) is consistent with it.

This script measures it on frame P with everything else held fixed: one belief field, one
candidate mask, one mass, one scorer.  Only min_sep changes.  In-sample by construction (the
belief is the shipped one) -- it is a geometry comparison, not a generalisation claim.

Writes registry/twostage/spacing_diagnostic.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, "src")
from gems44 import field as F
from gems44.metric import binary_credit, dti_of
from gems44.twostage import frames as FR
from gems44.twostage import stage_a as A
from gems44.twostage import stage_b as SB

GATE_FRACTION = 0.70
NO_HALO_PX = 2.0
HIDDEN_G = 7905.0
MASSES = (28_000, 42_000)
SEPS = (3.0, 4.0, 5.0, 6.0)


def main() -> int:
    Fs = FR.load_frames()
    belief = np.load("data/interim/belief_twostage.npy")
    bands = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for n in A.STAGE_A_BANDS:
            bands[n] = src.read(SB.BAND_INDEX[n]).astype(np.float32)
    layers = A.stage_a_layers(bands, Fs.known, Fs.valid)
    score_A = A.stage_a_score(layers)
    gate = A.apply_gate(score_A, Fs.valid, GATE_FRACTION)
    del bands, layers

    allowed = Fs.valid & ~Fs.known
    allowed_final = allowed & (Fs.d_known > NO_HALO_PX) & gate
    print(f"allowed_final {int(allowed_final.sum()):,} px", flush=True)

    cand = F.quantile_candidates(belief, allowed_final, 0.25)
    print(f"candidates {cand.size:,}", flush=True)

    rows = []
    for mass in MASSES:
        for sep in SEPS:
            order = F.emit_order_np(belief, cand, allowed_final, mass, min_sep=sep)
            d = np.zeros(Fs.valid.shape, dtype=bool)
            d.ravel()[order] = True
            n = int(d.sum())
            st = binary_credit(d.astype(np.float64), Fs.P, valid=Fs.valid, known=Fs.known)
            dti = dti_of(st)
            c = st["T"] / max(n, 1)
            hid = st["T"] / (0.2 * n + 0.8 * HIDDEN_G)
            rows.append({"mass_requested": mass, "min_sep_px": sep, "dots": n,
                         "P_T": st["T"], "P_F": st["F"], "P_DTI": dti,
                         "credit_per_dot": c, "dti_at_hidden_density": hid})
            print(f"  mass {mass:6d} sep {sep:.1f}: n={n:6d} c/dot={c:.4f} P_DTI={dti:.4f} "
                  f"hidden={hid:.4f}", flush=True)
    out = {
        "note": ("same belief field, same candidate mask, same scorer; only the minimum dot "
                 "separation changes.  Frame P (SGMC beyond 300 m of the catalogue), in-sample. "
                 "credit_per_dot is the mass-corrected instrument that reproduced the family's "
                 "reported live ordering 6/6 (scripts/calibrate_instrument.py)."),
        "hidden_g_assumed_px": HIDDEN_G,
        "gate_fraction": GATE_FRACTION,
        "no_halo_px": NO_HALO_PX,
        "rows": rows,
    }
    Path("registry/twostage/spacing_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
