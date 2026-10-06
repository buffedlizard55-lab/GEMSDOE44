"""Metric-aligned, spatially blocked field + mass selection.  THE decisive experiment.

Why this design
---------------
Fold 0 of the first holdout (registry/holdout_partial.json, kept for the record) produced the
result that changed the design: a supervised off-catalogue classifier reached a blocked pixel
AUC of 0.773, yet when its belief field was EMITTED and scored with the official metric on a
held-out quadrant it scored 0.0766, while uniform-random dots scored 0.1341 and the live-scored
0.2600 incumbent scored 0.1353.  A classifier's AUC is therefore NOT the right objective here:
the metric pays only for dots that land within 300 m of scored truth.  The objective must be the
metric itself, on a spatially blocked frame whose positives the catalogue does not already carry.

Leakage discipline
------------------
Every field that is learned is re-fitted inside each fold, using only the three training
quadrants, and is used to predict only the held-out quadrant.  Fields that are not learned
(distance transforms, concealment ranking) carry no fold information by construction.  Nothing
in this file reads the held-out quadrant before it is scored.

FRAME P (primary)  truth = SGMC fault pixels more than 300 m from the competition catalogue.
                   Real, officially published mapped faults that the given catalogue does not
                   carry - the closest locally available analogue of the hidden label set.
                   Because the positives are by construction far from the catalogue, this frame
                   cannot be won by copying the catalogue.
FRAME S (sanity)   truth = the catalogue itself.  Reported only, never used for selection.

Scoring uses gems44.metric.binary_credit (exact for binary {0,1} files: two Euclidean distance
transforms), which is what makes a 4-fold x 8-field x 6-mass sweep tractable on 2 vCPU.  The
final artifact is re-confirmed with the 29-offset operator in scripts/run_holdout.py.

Writes registry/selection.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import field as F  # noqa: E402
from gems44 import grid as G  # noqa: E402
from gems44.metric import binary_credit, dti_of  # noqa: E402
from gems44.scripts_common import (  # noqa: E402
    ROW_BLOCK, feature_block, load_bands, matrix_at, sample_rows,
)

MASSES = [4_000, 8_000, 14_000, 21_000, 30_000, 44_090]
CAND_FRACTION = 0.06
TRAIN_POS, TRAIN_NEG = 25_000, 60_000
SEED = 44


def emit(belief: np.ndarray, allowed: np.ndarray, max_dots: int) -> np.ndarray:
    cand = F.quantile_candidates(belief, allowed, CAND_FRACTION)
    return F.emit_order_np(belief, cand, allowed, max_dots)


def score(order: np.ndarray, shape, mass: int, truth, footprint, known) -> float:
    d = np.zeros(shape, dtype=bool)
    d.ravel()[order[:mass]] = True
    return float(dti_of(binary_credit(d, truth, valid=footprint, known=known)))


def main() -> int:
    rng = np.random.default_rng(SEED)
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known

    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    frame_p = sgmc & (d_cat > 3.0)
    print(f"FRAME P truth px: {int(frame_p.sum()):,}   frame S truth px: {int(cat.sum()):,}")

    quad = np.zeros(labels.shape, np.int8)
    quad[: g.height // 2, g.width // 2:] = 1
    quad[g.height // 2:, : g.width // 2] = 2
    quad[g.height // 2:, g.width // 2:] = 3

    bands = load_bands()
    print("building the leak-free belief family...")
    shared: dict[str, np.ndarray] = {}
    for sigma in (3.0, 10.0, 30.0):
        shared[f"prox{int(sigma)}"] = np.exp(-d_cat / sigma).astype(np.float32)
    d2b = bands["depth_to_basement"].astype(np.float64)
    fin = np.isfinite(d2b)
    d2b = np.where(fin, d2b, float(np.median(d2b[fin])))
    rank = np.empty(d2b.size, np.float64)
    rank[np.argsort(d2b, axis=None)] = np.arange(d2b.size, dtype=np.float64)
    shared["prox10_conceal"] = (shared["prox10"] * (0.35 + 1.65 * (rank / d2b.size).reshape(d2b.shape).astype(np.float32))).astype(np.float32)
    del d2b, rank, fin
    family = list(shared) + ["sup_offcat", "blend25", "blend50", "blend75"]

    inc = None
    ip = Path("data/raw/incumbent_d28.tif")
    if ip.exists():
        with rasterio.open(ip) as src:
            inc = np.isfinite(src.read(1)) & (src.read(1) > 0)

    report: dict = {
        "frames": {"P": "sgmc_faults_beyond_300m_of_catalogue", "S": "catalogue (sanity only)",
                   "P_truth_px": int(frame_p.sum()), "S_truth_px": int(cat.sum())},
        "masses": MASSES, "candidate_fraction": CAND_FRACTION, "fields": family,
        "leakage_discipline": "every learned field re-fitted inside each fold on the three training quadrants only",
        "fold_results": {},
    }

    for q in range(4):
        test = quad == q
        truth = frame_p & test
        if truth.sum() < 200:
            continue
        allowed = g.footprint & ~g.known & test
        print(f"fold {q}: truth={int(truth.sum()):,} allowed={int(allowed.sum()):,}")

        # ---- leak-free supervised field for this fold
        pos_idx = sample_rows(frame_p & ~test, TRAIN_POS, rng)
        neg_idx = sample_rows(g.footprint & ~g.known & ~sgmc & ~test, TRAIN_NEG, rng)
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xp = matrix_at(bands, g.width, pos_idx)
        Xn = matrix_at(bands, g.width, neg_idx)
        X = np.vstack([Xp, Xn])
        y = np.r_[np.ones(len(Xp), np.int8), np.zeros(len(Xn), np.int8)]
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0, early_stopping=False,
        )
        model.fit(X, y)
        del Xp, Xn, X
        belief = np.zeros(labels.shape, np.float32)
        c0, c1 = (0, g.width // 2) if q % 2 == 0 else (g.width // 2, g.width)
        r0g, r1g = (0, g.height // 2) if q < 2 else (g.height // 2, g.height)
        for r0 in range(r0g, r1g, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, r1g)
            fb = feature_block(bands, r0, r1)
            sc = model.predict_proba(fb.reshape(-1, fb.shape[-1]))[:, 1]
            belief[r0:r1, c0:c1] = sc.reshape(r1 - r0, g.width)[:, c0:c1]
            del fb, sc
        fold_fields = dict(shared)
        fold_fields["sup_offcat"] = belief
        for a in (0.25, 0.5, 0.75):
            fold_fields[f"blend{int(a*100)}"] = ((belief ** a) * (shared["prox10"] ** (1 - a))).astype(np.float32)

        row: dict = {"truth_px": int(truth.sum())}
        for fname in family:
            b = np.where(test, fold_fields[fname], 0.0).astype(np.float32)
            order = emit(b, allowed, max(MASSES))
            row[fname] = {str(m): round(score(order, labels.shape, m, truth, g.footprint, g.known), 6)
                          for m in MASSES if order.size >= m}
            print(f"   {fname:16s} emitted={order.size:>7,} " +
                  " ".join(f"{row[fname].get(str(m), float('nan')):.4f}" for m in MASSES))
        if inc is not None:
            row["incumbent"] = round(float(dti_of(binary_credit(inc & test, truth, valid=g.footprint, known=g.known))), 6)
        rr = np.random.default_rng(7 + q)
        row["uniform"] = {}
        for m in MASSES:
            d = np.zeros(labels.shape, bool)
            idx = rr.choice(np.nonzero(allowed.ravel())[0], size=min(m, int(allowed.sum())), replace=False)
            d.ravel()[idx] = True
            row["uniform"][str(m)] = round(float(dti_of(binary_credit(d, truth, valid=g.footprint, known=g.known))), 6)
        report["fold_results"][str(q)] = row
        print(f"fold {q} done: incumbent={row.get('incumbent')} uniform@44090={row['uniform'].get('44090')}")
        Path("registry").mkdir(exist_ok=True)
        Path("registry/selection_partial.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        del belief, fold_fields, allowed

    folds = sorted(report["fold_results"], key=int)
    combos = []
    for fname in family:
        for m in MASSES:
            vals = [report["fold_results"][f].get(fname, {}).get(str(m)) for f in folds]
            if any(v is None for v in vals):
                continue
            combos.append((fname, m, [float(v) for v in vals]))
    loo = []
    for i, q in enumerate(folds):
        others = [j for j in range(len(folds)) if j != i]
        best = max(combos, key=lambda c: float(np.mean([c[2][j] for j in others])))
        loo.append({"fold": int(q), "chosen_field": best[0], "chosen_mass": best[1],
                    "heldout_dti": round(best[2][i], 6),
                    "incumbent_heldout_dti": report["fold_results"][q].get("incumbent"),
                    "delta_vs_incumbent": round(best[2][i] - report["fold_results"][q]["incumbent"], 6),
                    "uniform_heldout_dti": report["fold_results"][q]["uniform"][str(best[1])]})
    deltas = [x["delta_vs_incumbent"] for x in loo]
    report["leave_one_fold_out"] = loo
    report["pooled"] = {
        "candidate_heldout_dti": round(float(np.mean([x["heldout_dti"] for x in loo])), 6),
        "incumbent_heldout_dti": round(float(np.mean([x["incumbent_heldout_dti"] for x in loo])), 6),
        "delta": round(float(np.mean(deltas)), 6),
        "positive_folds": int(sum(1 for d in deltas if d > 0)), "n_folds": len(loo),
        "gate_pass": bool(np.mean(deltas) >= 0.005 and sum(1 for d in deltas if d > 0) >= 3),
    }
    best_all = max(combos, key=lambda c: float(np.mean(c[2])))
    report["best_mean_over_all_folds"] = {
        "field": best_all[0], "mass": best_all[1], "mean_dti": round(float(np.mean(best_all[2])), 6),
        "note": "descriptive only - the promotion decision uses leave_one_fold_out, not this row",
    }
    Path("registry/selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== FRAME P, mean DTI over the 4 blocked folds ===")
    print(f"{'field':16s}" + "".join(f"{m:>10d}" for m in MASSES))
    for fname in family:
        cells = []
        for m in MASSES:
            vals = [report["fold_results"][f].get(fname, {}).get(str(m)) for f in folds]
            cells.append(f"{np.mean([v for v in vals if v is not None]):>10.4f}" if any(v is not None for v in vals) else f"{'-':>10s}")
        print(f"{fname:16s}" + "".join(cells))
    print(f"{'uniform':16s}" + "".join(f"{np.mean([report['fold_results'][f]['uniform'][str(m)] for f in folds]):>10.4f}" for m in MASSES))
    print(f"{'incumbent':16s}" + f"{np.mean([report['fold_results'][f]['incumbent'] for f in folds]):>10.4f}")
    print("\n" + json.dumps(report["pooled"], indent=1))
    print("best mean:", json.dumps(report["best_mean_over_all_folds"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
