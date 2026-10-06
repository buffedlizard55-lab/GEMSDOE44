#!/usr/bin/env python3
"""GEMSDOE44 pipeline: verify data -> build stages -> blocked holdout ->
gate + budget selection (pre-registered rules) -> unique submission TIF ->
mirror-model + uniqueness reports.

Writes evidence/*.json for every step.  Run:
    /home/user/venv/bin/python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems44 import config as C  # noqa: E402
from gems44 import data_io, features, stage1, stage2  # noqa: E402
from gems44.emit_submission import (audit_file, uniqueness_report,  # noqa: E402
                                    write_submission, write_zip)
from gems44.metric import max_kernel_to_truth, score_components  # noqa: E402
from gems44.emission import greedy_dots  # noqa: E402
from gems44.mirror_model import mirror_score  # noqa: E402

C.EVIDENCE.mkdir(parents=True, exist_ok=True)
C.DOWNLOADS.mkdir(parents=True, exist_ok=True)


def dump(name: str, obj) -> None:
    p = C.EVIDENCE / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"  -> {p.name}")


def main() -> None:
    t0 = time.time()
    print("== 0. verify official rasters")
    ver = data_io.verify_official_rasters()
    for v in ver:
        assert v["sha256_ok"], f"sha256 mismatch: {v['file']}"
    dump("verify_rasters.json", ver)

    print("== 1. load grid")
    grid = data_io.load_grid()
    dump("grid_stats.json", {
        "rows": grid.rows, "cols": grid.cols, "crs": grid.crs,
        "transform": list(grid.transform),
        "n_footprint": grid.n_footprint, "n_catalogue": grid.n_catalogue,
        "n_outside": int(grid.catalog_outside.sum()),
    })
    foot = grid.footprint
    cat = grid.catalogue
    H, W = grid.rows, grid.cols

    print("== 2. external official mirrors (heat-flow wells, gap links)")
    wx, wy, whf = data_io.load_heatflow_wells()
    wr, wc, whf = data_io.wells_to_grid(wx, wy, whf)
    gl = features.load_gap_links()
    import pyproj
    tr = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(32611), pyproj.CRS.from_epsg(4326), always_xy=True)
    ll = tr.transform(wx, wy)
    dump("external_mirrors.json", {
        "heatflow_wells_in_grid": int(wr.size),
        "heatflow_lat_range": [float(np.min(ll[1])), float(np.max(ll[1]))] if wr.size else None,
        "heatflow_hf_range_mWm2": [float(whf.min()), float(whf.max())] if wr.size else None,
        "hf_wells_ge_1500": int((whf >= 1500).sum()) if wr.size else 0,
        "gap_links_total": gl.n,
        "gap_links_relay_class": gl.n_relay,
        "gap_links_source": ("NBMG Qfaults-INGENIOUS v2 vector mirror, "
                             "official endpoint recorded per-feature in "
                             "nbmg_source_layer_url (345 features)"),
    })

    # NOTE: Stage 1 runs BEFORE the 410 MB stage-2 channel stack is built -
    # its block-stat transients (~350 MB) must not stack on top of it
    # (sandbox has 3.8 GB RAM).
    print("== 3. stage 1: coarse favourability gate + separate holdout")
    relay_ch = features.build_h2h3(foot, gl)
    s1 = stage1.build_stage1_features(
        grid, foot, relay_ch["relay"], relay_ch["endpt"],
        wr, wc, whf, cat)
    s1h = stage1.stage1_holdout(s1, H, W, cat, foot, q=C.GATE_Q)
    dump("stage1_holdout.json", {
        "preregistration": "GEMSDOE44-PREREG-1 stage 1: 4-quadrant block CV; "
                           "gate validated by zone AUC + catalogue concentration "
                           "ratio at top-q (q=0.35); proxy = visible catalogue "
                           "(necessary-condition test only)",
        "q": C.GATE_Q, "folds": s1h["folds"],
        "pooled": {
            "auc_mean": float(np.nanmean([f["auc"] for f in s1h["folds"].values()])),
            "conc_mean": float(np.nanmean([f["concentration_ratio"] for f in s1h["folds"].values()])),
        },
    })
    fav = s1h["favourability"]

    def fav_to_pixel(favb: np.ndarray) -> np.ndarray:
        hh = (H + C.SUPERPX - 1) // C.SUPERPX
        ww = (W + C.SUPERPX - 1) // C.SUPERPX
        b2 = favb.reshape(hh, ww)
        out = np.repeat(b2, C.SUPERPX, axis=0)[:H, :]
        out = np.repeat(out, C.SUPERPX, axis=1)[:, :W]
        return out

    fav_pixel = fav_to_pixel(fav)

    def gate_pixel_for(q: float) -> np.ndarray:
        if q >= 1.0:
            return foot.copy()
        thr = np.quantile(fav, 1.0 - q)
        return (fav_pixel >= thr) & foot

    gate_pixel = gate_pixel_for(C.GATE_Q)
    dump("gate_selection_plan.json", {
        "rule": "primary q=0.35 for the arm table; q selection on the A4-full "
                "arm only over q in {0.25, 0.35, 0.5, 1.0}, highest pooled "
                "proxy DTI wins, ties -> larger q (disclosed: selection on the "
                "same holdout)",
        "q_primary": C.GATE_Q, "q_candidates": [0.25, 0.35, 0.5, 1.0],
    })

    print("== 4. stage-2 channel stack (structure tensors etc.)")
    channels, meta = features.build_stage2_channels(
        grid, foot, wr, wc, whf)
    names = features.channel_names()
    print("   channels:", len(names), meta)
    # feature rows are gathered from the channel dict per tile (no 580 MB
    # flat matrix: the sandbox has 3.8 GB RAM)
    foot_idx = np.argwhere(foot).astype(np.int32)
    relay_ch = None  # free the stage-1 stamp arrays
    import gc
    gc.collect()

    print("== 5. stage 2: separate blocked holdout (5 arms)")
    s2h = stage2.stage2_holdout(channels, foot_idx, cat, foot, gate_pixel,
                                budget_fold=10000, fav_pixel=fav_pixel.astype(np.float32))
    arms = s2h["arms"]
    a4 = arms["A4_full"]["dti"]
    a0, a1, a2, a3 = (arms[a]["dti"] for a in
                      ["A0_random", "A1_habitat", "A2_singlefield", "A3_h1only"])
    beats = {
        "A4>A0": int((a4 > a0).sum()), "A4>A1": int((a4 > a1).sum()),
        "A4>A2": int((a4 > a2).sum()), "A4>A3": int((a4 > a3).sum()),
    }
    conc = [f["concentration_ratio"] for f in s1h["folds"].values()]
    promotion = {
        "A4_beats_A0_A1_A2_ge3folds": beats["A4>A0"] >= 3 and beats["A4>A1"] >= 3
                                       and beats["A4>A2"] >= 3,
        "A4_beats_A3_ge2folds": beats["A4>A3"] >= 2,
        "stage1_conc_ge1.3_ge3folds": int(np.nanmean([c >= 1.3 for c in conc]) * 4) >= 3,
    }
    promoted = all(promotion.values())
    dump("stage2_holdout.json", {
        "preregistration": "GEMSDOE44-PREREG-1 stage 2: 4-quadrant block CV; "
                           "each arm emits 10,000 dots (200 m exclusion) inside "
                           "the held-out quadrant under gate q=0.35 and the "
                           "B=2 catalogue-flank mask; scored with the official "
                           "DTI vs the fold's catalogue",
        "budget_fold": 10000,
        "per_fold": s2h["folds"],
        "arm_means": {k: {"mean_dti": float(v["dti"].mean()),
                          "per_fold": [float(x) for x in v["dti"]],
                          "px": int(v["px"][0])}
                      for k, v in arms.items()},
        "beats_counts": beats,
        "promotion_rule": promotion,
        "PROMOTED": bool(promoted),
    })

    print("== 6. gate-q selection (A4 arm only)")
    q_scores = {}
    for q in [0.25, 0.5, 1.0]:
        gp = gate_pixel_for(q)
        r = stage2.stage2_holdout(channels, foot_idx, cat, foot, gp,
                                  budget_fold=10000, emit_arms=["A4_full"],
                                  fav_pixel=fav_pixel)
        q_scores[q] = float(r["arms"]["A4_full"]["dti"].mean())
    q_scores[0.35] = float(a4.mean())
    best_q = max(q_scores, key=lambda q: (q_scores[q], q))
    dump("gate_q_selection.json", {
        "q_scores_A4_mean_proxyDTI": {str(k): v for k, v in q_scores.items()},
        "chosen_q": best_q,
        "disclosure": "selection made on the same spatially-blocked holdout "
                      "(mild optimism, disclosed per pre-registration)",
    })
    gate_pixel = gate_pixel_for(best_q)

    print("== 7. budget sweep (pooled, full-data model, per-fold emission)")
    # full-data model on a 1:10 negative subsample (positives: all catalogue px)
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
    quads = stage2.quadrant_mask(H, W)
    flank = stage2.flank_mask(cat)
    # full prediction (tiled)
    pred = np.zeros((H, W), np.float32)
    all_rows = np.unique(foot_idx[:, 0])
    for i in range(0, len(all_rows), 96):
        stage2.predict_rows(clf_all, channels, names, foot_idx, pred,
                            all_rows[i:i + 96])
    K_all = max_kernel_to_truth(cat.astype(np.float32), foot)
    budget_scores = {}
    for N in C.BUDGET_CANDIDATES:
        P = np.zeros((H, W), np.float32)
        for f in range(4):
            field = np.where(gate_pixel & ~flank & quads[f] & foot, pred, 0.0).astype(np.float32)
            r, c = greedy_dots(field, N // 4, C.EXCLUSION_PX)
            P[r, c] = 1.0
        budget_scores[N] = score_components(P, cat, foot, K_cache=K_all)["DTI"]
    best_N = max(budget_scores, key=lambda n: (budget_scores[n], -n))
    dump("budget_sweep.json", {
        "note": "pooled proxy DTI; full-data model (optimistic, relative "
                "comparison only; disclosed)",
        "scores": {str(k): v for k, v in budget_scores.items()},
        "chosen_N": best_N,
    })

    print(f"== 8. final emission (N={best_N}, q={best_q})")
    field = np.where(gate_pixel & ~flank & foot, pred, 0.0).astype(np.float32)
    rows, cols = greedy_dots(field, best_N, C.EXCLUSION_PX)
    K = max_kernel_to_truth(cat.astype(np.float32), foot)
    P = np.zeros((H, W), np.float32)
    P[rows, cols] = 1.0
    prox = score_components(P, cat, foot, K_cache=K)
    dump("final_proxy_score.json", {
        "note": "full-data model vs FULL catalogue (train-set overlap; "
                "optimistic; NOT a score estimate)",
        **prox,
    })

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    import hashlib
    digest = hashlib.sha256(f"GEMS44-FINAL-{rows.tobytes()}{cols.tobytes()}".encode()).hexdigest()[:8]
    tif_name = f"gemsdoe44-two-stage-{ts}-{digest}-zeros.tif"
    tif_path = C.DOWNLOADS / tif_name
    receipt = write_submission(rows, cols, 1.0, tif_path)
    write_zip(tif_path, tif_path.with_suffix(".zip"))
    dump(f"audit-{tif_name}.json", receipt)
    assert receipt["checks"]["all_checks_passed"], "format audit FAILED"
    # stable alias for the site (one-click download link stays constant)
    import shutil
    stable = C.DOWNLOADS / "gemsdoe44-primary.tif"
    shutil.copyfile(tif_path, stable)
    write_zip(stable, C.DOWNLOADS / "gemsdoe44-primary.zip")
    dump("audit-primary-alias.json", {
        "alias_of": tif_name, "sha256": receipt["sha256"],
        "audit": f"audit-{tif_name}.json",
    })

    print("== 9. dominance check (no single-stage footprint)")
    prov = dominance_report(rows, cols, fav, s1, pred, foot, best_q, channels)
    dump("dominance_check.json", prov)

    print("== 10. mirror model (live-anchored truth model)")
    mirror = {}
    mirror["final"] = mirror_score(P, cat, foot)
    for repo, fname, score in C.INCUMBENTS:
        p = C.REPO.parent / "ref" / repo / "docs" / "downloads" / fname
        if p.exists():
            import rasterio
            with rasterio.open(p) as ds:
                A = ds.read(1)
            mirror[fname] = mirror_score(A, cat, foot)
    dump("mirror_model.json", {
        "description": "catalogue scattered with 2-D Gaussian (independent "
                       "re-implementation of the live-anchored truth idea; "
                       "sigma swept 1.5-2.5 px); official DTI over paired "
                       "seeds; a MODEL, not a score",
        "results": mirror,
    })

    print("== 11. uniqueness vs previously-scored submissions")
    uniq = uniqueness_report(rows, cols, C.INCUMBENTS)
    dump("uniqueness_report.json", uniq)

    print("== 12. summary")
    summary = {
        "wall_time_s": round(time.time() - t0, 1),
        "stage1": {"q": best_q, "pooled_conc": float(np.nanmean(conc)),
                   "folds": s1h["folds"]},
        "stage2": {"promoted": bool(promoted), "beats": beats,
                   "A4_mean": float(a4.mean()),
                   "A0_mean": float(a0.mean()),
                   "A1_mean": float(a1.mean()),
                   "A2_mean": float(a2.mean()),
                   "A3_mean": float(a3.mean())},
        "budget": best_N,
        "final": {
            "tif": tif_name,
            "sha256": receipt["sha256"],
            "dots": int((P > 0).sum()),
            "proxy_dti_full_catalogue": prox["DTI"],
        },
        "mirror": mirror,
        "uniqueness": uniq,
        "dominance": prov,
    }
    dump("pipeline_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ["stage2", "budget", "final"] if k in summary}, indent=1, default=str))


def dominance_report(rows, cols, fav, s1, pred, foot, q, channels):
    """Where did the emitted dots come from? Neither stage may dominate.

    Stage-1 provenance: every dot must sit inside the Stage-1 gate (the
    coarse favourability zone).  Stage-2 provenance: dots should be enriched
    for the fine-scale structure channels (H1 cross-physics lineament
    consensus, gap-link relay) relative to the gate-zone baseline - i.e.
    the fine placement model, not the habitat alone, chose the pixels.
    """
    ids = s1["ids"]
    sup_ids = ids[rows, cols]
    fav_at = fav[sup_ids]
    thr = np.quantile(fav, 1.0 - q) if q < 1 else 0.0
    deciles = np.digitize(fav_at, np.quantile(fav, np.linspace(0.1, 0.9, 9)))

    pred_dots = pred[rows, cols]
    pair_dots = np.maximum(channels["pair_hi"][rows, cols],
                           channels["pair_lo"][rows, cols])
    relay_dots = channels["relay"][rows, cols]

    pred_med_dots = float(np.median(pred_dots))
    pair_p90_foot = float(np.percentile(
        np.maximum(channels["pair_hi"][foot], channels["pair_lo"][foot]), 90))
    relay_nonzero_dots = float((relay_dots > 0).mean())
    pair_at_p90 = float((pair_dots >= pair_p90_foot).mean())
    s2_backed = float(((pair_dots >= pair_p90_foot) | (relay_dots > 0)).mean())
    return {
        "n_dots": int(len(rows)),
        "fav_decile_hist": [int((deciles == i + 1).sum()) for i in range(9)],
        "fav_decile_top_share": float((deciles == 9).mean()),
        "fav_in_gate_share": float((fav_at >= thr).mean()),
        "stage1_dominance_check": "PASS" if float((fav_at >= thr).mean()) == 1.0
                                  else "FAIL",
        "stage2_pred_median_at_dots": pred_med_dots,
        "stage2_pair_at_footprint_p90_share": pair_at_p90,
        "stage2_relay_nonzero_share": relay_nonzero_dots,
        "stage2_backed_share": s2_backed,
        "stage2_dominance_check": "PASS" if s2_backed >= 0.5 else "FAIL",
        "note": "stage-1 check: dots outside the gate are impossible by "
                "construction (gate is a hard constraint at emission); "
                "stage-2 check: >=50% of dots sit on H1 consensus at/above "
                "footprint p90 or on a gap-link relay stamp - if this fails, "
                "the fine placement model added nothing over the habitat",
    }


if __name__ == "__main__":
    main()
