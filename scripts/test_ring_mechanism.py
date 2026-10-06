#!/usr/bin/env python3
"""Decisive local test of the '0.2600 -> 0.2778 came from pruning the <=200 m catalogue ring' claim.

The two files exist locally, so the relation can be read byte by byte:

    A = gems32-probe-S1-ANCHOR-identical-to-live-02600.tif   44,090 dots, the live 0.2600 anchor
    B = gemsdoe32-h33-h33-2-b2-...-e5eb6e7e-zeros.tif        37,654 dots, the 0.2778 artifact (owner-reported)
    B is a strict subset of A; A \\ B = 6,436 dots, and every removed dot lies within 200 m of the
    catalogue (bands 0-100 m and 100-200 m are emptied to exactly 0.000).

This script asks the only question that matters for the mechanism: were those 6,436 removed dots
net-negative *on real truth*?  It scores A, B and the removal set on the two local off-catalogue
truth frames (N and P) with the exact official operator, and decomposes the change into
kernel credit earned versus 0.2-per-dot false-positive mass.

Writes registry/ring_mechanism.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as ndimage

sys.path.insert(0, "src")
from gems44 import grid as G          # noqa: E402
from gems44 import metric as M        # noqa: E402

PRIOR = Path("/home/user/prior/GEMSDOE32/docs/downloads")
ANCHOR = PRIOR / "gems32-probe-S1-ANCHOR-identical-to-live-02600.tif"
SHIPPED_02778 = PRIOR / "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif"


def dots_of(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        v = src.read(1)
    return np.isfinite(v) & (v > 0)


def main() -> int:
    g = G.load_grid()
    out: dict = {"written_utc": "2026-10-06"}
    d_cat = ndimage.distance_transform_edt(~g.known)
    sgmc = G.read_band("data/external/derived_sgmc_faults_100m_u8.tif", 1) > 0
    frames = {"N": sgmc & (d_cat <= 3.0), "P": sgmc & (d_cat > 3.0)}

    A = dots_of(ANCHOR)
    B = dots_of(SHIPPED_02778)
    removed = A & ~B
    assert int((B & ~A).sum()) == 0, "B must be a subset of A"
    out["subset_check"] = {"A_dots": int(A.sum()), "B_dots": int(B.sum()),
                           "removed_dots": int(removed.sum()), "added_dots": int((B & ~A).sum()),
                           "removed_frac_inside_200m": round(float((removed & (d_cat <= 2.0)).sum())
                                                             / max(int(removed.sum()), 1), 6)}

    for name, frame in frames.items():
        rec: dict = {"truth_px": int(frame.sum())}
        for label, dots in (("anchor_0.2600", A), ("artifact_0.2778", B)):
            r = M.dti(dots, frame, valid=g.footprint, known=g.known)
            rec[label] = {"dti": round(r.score, 6), "TP_w": round(r.tp_w, 3), "FP_w": round(r.fp_w, 3),
                          "dots_in_frame_domain": int((dots & ~g.known).sum())}
        # what the removed dots were doing: credit they earn, cost they pay
        r_rem = M.dti(removed, frame, valid=g.footprint, known=g.known)
        rec["removed_set"] = {
            "dots": int(removed.sum()),
            "dti_of_removed_alone": round(r_rem.score, 6),
            "TP_w_earned_by_removed": round(r_rem.tp_w, 3),
            "FP_w_of_removed": round(r_rem.fp_w, 3),
            "mean_credit_per_removed_dot": round(r_rem.tp_w / max(int(removed.sum()), 1), 4),
            "fp_cost_at_alpha_0.2": round(0.2 * r_rem.fp_w, 3),
            "net_effect_on_denominator": round(0.2 * r_rem.fp_w - r_rem.tp_w, 3),
        }
        rec["observed_delta_dti_pruned_minus_anchor"] = round(rec["artifact_0.2778"]["dti"]
                                                             - rec["anchor_0.2600"]["dti"], 6)
        out[f"frame_{name}"] = rec

    # my own artifact: the same profile question
    sub = json.loads(Path("registry/submission.json").read_text())
    mine = dots_of(Path("submissions") / f"{sub['slug']}-zeros.tif")
    prof = {}
    for lo, hi, label in ((0.0, 1.0, "0-100m"), (1.0, 2.0, "100-200m"), (2.0, 3.0, "200-300m"),
                          (3.0, np.inf, "beyond300m")):
        m = mine & (d_cat > lo) & (d_cat <= hi)
        prof[label] = {"dots": int(m.sum()), "frac": round(int(m.sum()) / max(int(mine.sum()), 1), 5)}
    out["shipped_artifact_profile"] = prof
    for name, frame in frames.items():
        r = M.dti(mine, frame, valid=g.footprint, known=g.known)
        inner = mine & (d_cat <= 2.0)
        r2 = M.dti(inner, frame, valid=g.footprint, known=g.known)
        out[f"shipped_frame_{name}"] = {
            "dti": round(r.score, 6), "inner_ring_dots": int(inner.sum()),
            "inner_ring_TP_w": round(r2.tp_w, 3), "inner_ring_mean_credit": round(r2.tp_w / max(int(inner.sum()), 1), 4)}

    Path("registry/ring_mechanism.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
