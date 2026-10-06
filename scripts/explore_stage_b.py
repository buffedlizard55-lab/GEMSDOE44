"""Exploratory, blocked measurement of every Stage B feature against the P frame.

Run:  PYTHONPATH=src python scripts/explore_stage_b.py
Writes: registry/twostage/stage_b_feature_auc.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from gems44.twostage import frames as FR, stage_b as B
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(44)
t0 = time.time()
F = FR.load_frames()
print(F.summary(), flush=True)
P, U, N = F.P, F.U, F.N
bg = F.valid & ~(F.known | F.sgmc)
print("frame P", P.sum(), "background candidates", bg.sum())

n_pos = min(40000, int(P.sum()))
pos_rows = rng.choice(np.flatnonzero(P.ravel()), size=n_pos, replace=False)
neg_rows = rng.choice(np.flatnonzero(bg.ravel()), size=min(250000, int(bg.sum())), replace=False)
rows = np.concatenate([pos_rows, neg_rows])
lab = np.concatenate([np.ones(len(pos_rows), np.int8), np.zeros(len(neg_rows), np.int8)])
quad_rows = F.quad.ravel()[rows]

import rasterio
bands = {}
with rasterio.open("data/raw/training_features.tif") as src:
    for name in B.NEEDED_BANDS:
        bands[name] = src.read(B.BAND_INDEX[name]).astype(np.float32)

results = {}
for spec in B.STAGE_B_SPEC:
    t1 = time.time()
    try:
        f = B.build_stage_b_feature(spec.name, bands)
    except Exception as e:
        print("ERR", spec.name, e); continue
    x = f.ravel()[rows]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    aucs = []
    for q in range(4):
        m = quad_rows == q
        if m.sum() == 0 or lab[m].min() == lab[m].max():
            continue
        aucs.append(float(roc_auc_score(lab[m], x[m])))
    results[spec.name] = {
        "family": spec.family, "scale": spec.scale,
        "auc_folds": [round(a, 4) for a in aucs],
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_min": round(float(np.min(aucs)), 4),
        "sign_consistency": int(sum(a > 0.5 for a in aucs)),
        "seconds": round(time.time() - t1, 1),
    }
    print(f"{spec.name:26s} mean_auc={results[spec.name]['auc_mean']:.4f} folds={results[spec.name]['auc_folds']} ({time.time()-t1:.0f}s)", flush=True)
    del f

out = Path("registry/twostage"); out.mkdir(parents=True, exist_ok=True)
(out / "stage_b_feature_auc.json").write_text(json.dumps({
    "frame": "P = SGMC faults >300 m from catalogue, blocked 4-quadrant folds",
    "n_pos": int(n_pos), "n_neg": int(len(neg_rows)), "seed": 44,
    "features": dict(sorted(results.items(), key=lambda kv: -kv[1]["auc_mean"])),
}, indent=2) + "\n")
print("TOTAL", round(time.time() - t0, 1), "s")
