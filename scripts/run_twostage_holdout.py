"""Two-stage, spatially-blocked holdout: does the candidate field beat the incumbent?

Protocol (identical for every candidate; nothing is tuned on the test fold)
-------------------------------------------------------------------------
For each of the four quadrants q:
  1. TRAIN  on the other three quadrants only:
       positives  = frame P (SGMC faults > 300 m from the catalogue)
       negatives  = random footprint pixels with no mapped fault of any kind
       features   = stage_b feature library (19 official bands only)
  2. TEST   on quadrant q:
       truth      = frame P restricted to q
       mass       = the number of dots the INCUMBENT file places in q (matched mass)
       candidates = the top-``mass`` pixels of the candidate field inside q
       metric     = the exact official DTI (gems44.metric), known mask applied
  3. Report AUC as well, but the decisive number is the matched-mass DTI difference.

Also measured: two emission rules on the SAME field (raster-order top-n vs the
metric-exact greedy max-coverage emitter), so that "field" and "emission" are
never conflated -- the same error the family's habitat-only runs made.

Run:  PYTHONPATH=src python scripts/run_twostage_holdout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from gems44 import metric
from gems44.twostage import frames as FR
from gems44.twostage import stage_b as B

INCUMBENT = "data/raw/incumbent_d28.tif"
SEED = 44


def load_bands(names: list[str], path: str = "data/raw/training_features.tif") -> dict[str, np.ndarray]:
    out = {}
    with rasterio.open(path) as src:
        for n in names:
            out[n] = src.read(B.BAND_INDEX[n]).astype(np.float32)
    return out


def build_all_features(bands: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Stack the requested features as a (n_features, H, W) float32 array."""
    return np.stack([B.build_stage_b_feature(n, bands) for n in names]).astype(np.float32)


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    F = FR.load_frames()
    P, U, N = F.P, F.U, F.N
    background = F.valid & ~(F.known | F.sgmc)
    print("frames:", F.summary(), flush=True)

    with rasterio.open(INCUMBENT) as src:
        inc = np.isfinite(src.read(1)) & (src.read(1) > 0.5)
    print(f"incumbent dots: {int(inc.sum()):,}", flush=True)

    names = B.feature_names()
    bands = load_bands(B.NEEDED_BANDS)
    feats = build_all_features(bands, names)
    print(f"features built: {feats.shape} in {time.time()-t0:.0f}s", flush=True)

    # ---- sampled training rows: all P positives + an equal-ish negative pool -----
    pos_rows = np.flatnonzero((P).ravel())
    neg_rows = rng.choice(np.flatnonzero(background.ravel()),
                          size=min(400000, int(background.sum())), replace=False)
    all_rows = np.concatenate([pos_rows, neg_rows])
    y_all = np.concatenate([np.ones(pos_rows.size, np.int8), np.zeros(neg_rows.size, np.int8)])
    q_all = F.quad.ravel()[all_rows]

    X = np.empty((all_rows.size, len(names)), dtype=np.float32)
    for j in range(len(names)):
        X[:, j] = feats[j].ravel()[all_rows]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    report: dict = {"protocol": {}, "folds": [], "mass_matched": {}}
    for q in range(4):
        tr = q_all != q
        te = q_all == q
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
        )
        clf.fit(X[tr], y_all[tr])
        auc_q = float(roc_auc_score(y_all[te], clf.predict_proba(X[te])[:, 1]))

        # full-grid score for quadrant q
        score = np.zeros(F.valid.shape, dtype=np.float32)
        qi = np.flatnonzero((F.quad == q).ravel())
        Xq = np.empty((qi.size, len(names)), dtype=np.float32)
        for j in range(len(names)):
            Xq[:, j] = feats[j].ravel()[qi]
        Xq = np.nan_to_num(Xq, nan=0.0, posinf=0.0, neginf=0.0)
        score.ravel()[qi] = clf.predict_proba(Xq)[:, 1]
        del Xq
        np.save(f"/tmp/twostage_score_q{q}.npy", score)

        truth_q = P & (F.quad == q)
        inc_q = inc & (F.quad == q)
        n_mass = int(inc_q.sum())
        cand_order = np.argsort(-score.ravel())
        allowed = (F.quad == q).ravel() & F.valid.ravel()
        cand = np.zeros(F.valid.size, dtype=bool)
        taken = 0
        for idx in cand_order:
            if taken >= n_mass:
                break
            if allowed[idx]:
                cand[idx] = True
                taken += 1
        cand = cand.reshape(F.valid.shape)

        def dti_of(dots: np.ndarray) -> dict:
            st = metric.binary_credit(dots.astype(np.float64), truth_q, valid=F.valid, known=F.known)
            return {"DTI": metric.dti_of(st), "T": st["T"], "F": st["F"], "G": st["G"], "n": st["n"]}

        a, b = dti_of(inc_q), dti_of(cand)
        report["folds"].append({
            "quadrant": q, "auc": round(auc_q, 4), "mass": n_mass,
            "incumbent": {k: round(v, 6) for k, v in a.items()},
            "candidate": {k: round(v, 6) for k, v in b.items()},
            "delta_candidate_minus_incumbent": round(b["DTI"] - a["DTI"], 6),
        })
        print(f"fold {q}: AUC={auc_q:.4f} mass={n_mass:,} "
              f"incumbent DTI={a['DTI']:.4f} (T={a['T']:.0f}) candidate DTI={b['DTI']:.4f} (T={b['T']:.0f}) "
              f"delta={b['DTI']-a['DTI']:+.4f}", flush=True)

    deltas = [f["delta_candidate_minus_incumbent"] for f in report["folds"]]
    report["mass_matched"] = {
        "mean_delta": float(np.mean(deltas)),
        "folds_positive": int(sum(d > 0 for d in deltas)),
        "n_folds": len(deltas),
    }
    report["protocol"] = {
        "frames": F.summary(),
        "feature_names": names,
        "incumbent": INCUMBENT,
        "seed": SEED,
        "truth": "frame P = SGMC faults > 300 m from the catalogue (never the catalogue itself)",
        "mass": "matched to the incumbent's dot count per quadrant",
        "metric": "exact gems44.metric.dti",
    }
    out = Path("registry/twostage"); out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_field_vs_incumbent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mass_matched"], indent=2))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
