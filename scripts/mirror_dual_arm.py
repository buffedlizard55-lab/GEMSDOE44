#!/usr/bin/env python3
"""Disclosed follow-up: mirror-model (live-anchored truth) comparison of the
two validated placements at matched budget, because the catalogue-proxy
holdout rewards finding the MAPPED catalogue (the Stage-1 gate was trained
on it) while the hidden truth is OFF-catalogue.

Arms:
  A1_habitat : greedy dots from the Stage-1 favourability field
  A4_full    : greedy dots from the full-data Stage-2 model probability
Both restricted to gate(best_q) & ~flank(B=2) & footprint, 200 m exclusion,
same N (from budget_sweep.json). Scored with the official DTI against
scattered-catalogue truths (sigma sweep, paired seeds) - a MODEL, not a score.

Writes evidence/mirror_dual_arm.json. Run AFTER scripts/run_all.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems44 import config as C  # noqa: E402
from gems44 import data_io, features, stage1, stage2  # noqa: E402
from gems44.emission import greedy_dots  # noqa: E402
from gems44.metric import score_components  # noqa: E402
from gems44.mirror_model import mirror_score  # noqa: E402


def main() -> None:
    ev = json.load(open(C.EVIDENCE / "budget_sweep.json"))
    best_N = int(ev["chosen_N"])
    qsel = json.load(open(C.EVIDENCE / "gate_q_selection.json"))
    best_q = float(qsel["chosen_q"])

    grid = data_io.load_grid()
    wx, wy, whf = data_io.load_heatflow_wells()
    wr, wc, whf = data_io.wells_to_grid(wx, wy, whf)
    foot, cat, H, W = grid.footprint, grid.catalogue, grid.rows, grid.cols

    gl = features.load_gap_links()
    relay_ch = features.build_h2h3(foot, gl)
    s1 = stage1.build_stage1_features(
        grid, foot, relay_ch["relay"], relay_ch["endpt"], wr, wc, whf, cat)
    s1h = stage1.stage1_holdout(s1, H, W, cat, foot, q=C.GATE_Q)
    fav = s1h["favourability"]
    hh, ww = (H + C.SUPERPX - 1) // C.SUPERPX, (W + C.SUPERPX - 1) // C.SUPERPX
    fav_pixel = np.repeat(np.repeat(fav.reshape(hh, ww), C.SUPERPX, axis=0)[:H, :],
                          C.SUPERPX, axis=1)[:, :W].astype(np.float32)
    thr = np.quantile(fav, 1.0 - best_q) if best_q < 1 else 0.0
    gate = (fav_pixel >= thr) & foot
    flank = stage2.flank_mask(cat)
    domain = gate & ~flank & foot

    channels, _ = features.build_stage2_channels(grid, foot, wr, wc, whf)
    names = features.channel_names()
    foot_idx = np.argwhere(foot).astype(np.int32)
    y_all = cat[foot_idx[:, 0], foot_idx[:, 1]].astype(int)
    pos_all = np.where(y_all == 1)[0]
    neg_all = np.where(y_all == 0)[0]
    rng_sub = np.random.default_rng(C.SEED)
    neg_sel = rng_sub.choice(len(neg_all), size=10 * len(pos_all), replace=False)
    tr_idx = np.concatenate([pos_all, neg_all[neg_sel]])
    rng_sub.shuffle(tr_idx)
    Xtr = stage2.gather_pixels(channels, names, foot_idx[tr_idx])
    clf_all = stage2.train_stage2_model(Xtr, y_all[tr_idx], seed=C.SEED)
    del Xtr
    pred = np.zeros((H, W), np.float32)
    all_rows = np.unique(foot_idx[:, 0])
    for i in range(0, len(all_rows), 96):
        stage2.predict_rows(clf_all, channels, names, foot_idx, pred,
                            all_rows[i:i + 96])

    arms = {
        "A1_habitat": np.where(domain, fav_pixel, 0.0).astype(np.float32),
        "A4_full": np.where(domain, pred, 0.0).astype(np.float32),
    }
    out = {
        "note": "disclosed model-based arm comparison (catalogue-proxy holdout "
                "favours the gate field by construction; the mirror is the "
                "less-circular off-catalogue test). A MODEL, not a score.",
        "best_N": best_N, "best_q": best_q,
    }
    for arm, field in arms.items():
        rows, cols = greedy_dots(field, best_N, C.EXCLUSION_PX)
        P = np.zeros((H, W), np.float32)
        P[rows, cols] = 1.0
        # catalogue-proxy DTI (same footing as the holdout, full catalogue)
        prox = score_components(P, cat, foot)
        out[arm] = {
            "dots": int(P.sum()),
            "proxy_dti_full_catalogue": prox["DTI"],
            "mirror": mirror_score(P, cat, foot),
        }
    C.EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open(C.EVIDENCE / "mirror_dual_arm.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
