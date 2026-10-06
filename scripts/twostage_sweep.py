"""Two-stage blocked sweep: field variants x emission variants, scored with the exact metric.

PRE-REGISTERED PROTOCOL (fixed before the sweep results are read, recorded in
registry/twostage/preregistration.json)

  Engine      : HistGradientBoostingClassifier on the cached Stage B features
  Training    : positives = frame U (all SGMC faults the catalogue lacks, 79,615 px)
                negatives = footprint pixels with no mapped fault of any kind
  Blocking    : leave-one-quadrant-out; the model never sees the held-out quadrant
  Mass        : density matched, mass_fold = M_global * allowed_q / allowed_total
  Metric      : exact official DTI (gems44.metric), known-fault mask applied
  Primary     : frame N (SGMC strands within 300 m of the catalogue, 17,493 px)
                -- the only frame measured to rank the family's reported live scores
                with a positive sign (rho_excess = +0.396, n = 19, GEMSDOE44
                scripts/rank_frames.py)
  Robustness  : frame P (SGMC strands beyond 300 m, 62,122 px) must not regress
  Controls    : incumbent d2.8 artifact, the prior session's shipped artifact,
                and a uniform-random emission at the same mass

  GATE (promotion of a variant): mean frame-N delta vs the INCUMBENT >= +0.005
                AND at least 3 of 4 folds positive AND frame-P mean delta >= 0.

Run:  PYTHONPATH=src python scripts/twostage_sweep.py
Writes: registry/twostage/sweep.json (and .partial.json after every fold)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
from gems44 import field as F
from gems44.metric import binary_credit, dti_of
from gems44.twostage import frames as FR
from gems44.twostage import stage_a as A

FEATDIR = Path("data/interim/feats")
INCUMBENT = "data/raw/incumbent_d28.tif"
PRIOR = "submissions/GEMS44_n-strand-ssmc_20261006T110212Z_4971d359-zeros.tif"
SEED = 44
MASSES = [36_000, 42_000, 48_000, 60_000]
CAND_FRACTION = 0.25
MIN_SEP = 3.0
TRAIN_POS, TRAIN_NEG = 60_000, 160_000
BLURS = [0.0, 1.85]


def load_feature(name: str) -> np.memmap:
    return np.load(FEATDIR / f"{name}.npy", mmap_mode="r")


def feature_names() -> list[str]:
    """The cached feature order, from the manifest written by twostage_features.py."""
    man = json.loads(Path("registry/twostage/feature_manifest.json").read_text())
    return list(man["features"].keys())


def sample(rows: np.ndarray, feats: list[str], shape) -> np.ndarray:
    X = np.empty((rows.size, len(feats)), dtype=np.float32)
    for j, name in enumerate(feats):
        X[:, j] = load_feature(name).reshape(-1)[rows]
    return X


def emit(belief: np.ndarray, allowed: np.ndarray, mass: int, min_sep: float = MIN_SEP) -> np.ndarray:
    cand = F.quantile_candidates(belief, allowed, CAND_FRACTION)
    order = F.emit_order_np(belief, cand, allowed, mass, min_sep=min_sep)
    dots = np.zeros(belief.shape, dtype=bool)
    dots.ravel()[order] = True
    return dots


def score(dots: np.ndarray, truth: np.ndarray, valid, known) -> dict:
    st = binary_credit(dots.astype(np.float64), truth, valid=valid, known=known)
    return {"DTI": dti_of(st), "T": st["T"], "F": st["F"], "G": st["G"], "n": st["n"]}


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    Fs = FR.load_frames()
    P, U, N = Fs.P, Fs.U, Fs.N
    background = Fs.valid & ~(Fs.known | Fs.sgmc)
    quad = Fs.quad

    with rasterio.open(INCUMBENT) as src:
        inc = np.isfinite(src.read(1)) & (src.read(1) > 0.5)
    with rasterio.open(PRIOR) as src:
        prior = src.read(1) > 0.5

    from gems44.twostage import stage_b as SB
    bands = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for n in A.STAGE_A_BANDS:
            bands[n] = src.read(SB.BAND_INDEX[n]).astype(np.float32)
    layers = A.stage_a_layers(bands, Fs.known, Fs.valid)
    score_A = A.stage_a_score(layers)
    gate_70 = A.apply_gate(score_A, Fs.valid, 0.70)

    feats = feature_names()
    print(f"{len(feats)} features; frames {Fs.summary()}", flush=True)

    allowed_all = Fs.valid & ~Fs.known
    n_allowed_total = int(allowed_all.sum())

    pos_rows_all = np.flatnonzero(U.ravel())
    neg_pool = np.flatnonzero(background.ravel())

    out_partial = Path("registry/twostage/sweep.partial.json")
    out_final = Path("registry/twostage/sweep.json")
    report: dict = {"protocol": {"masses": MASSES, "min_sep": MIN_SEP, "cand_fraction": CAND_FRACTION,
                                 "blurs": BLURS, "train_pos": TRAIN_POS, "train_neg": TRAIN_NEG,
                                 "seed": SEED, "features": feats,
                                 "frames": Fs.summary(),
                                 "gate": "mean frame-N delta vs incumbent >= +0.005, >=3/4 folds positive, frame-P mean delta >= 0"},
                    "folds": []}

    for q in range(4):
        tf = time.time()
        tr = quad.ravel()[pos_rows_all] != q
        tr_pos = np.flatnonzero(tr)
        take_pos = rng.choice(tr_pos, size=min(TRAIN_POS, tr_pos.size), replace=False)
        neg_cand = np.flatnonzero(quad.ravel()[neg_pool] != q)
        take_neg = rng.choice(neg_cand, size=min(TRAIN_NEG, neg_cand.size), replace=False)
        rows = np.concatenate([pos_rows_all[take_pos], neg_pool[take_neg]])
        y = np.r_[np.ones(take_pos.size, np.int8), np.zeros(take_neg.size, np.int8)]
        X = sample(rows, feats, Fs.valid.shape)
        del rows
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                            min_samples_leaf=40, l2_regularization=1.0,
                                            random_state=SEED, early_stopping=False)
        clf.fit(X, y)
        del X
        print(f"fold {q}: trained on {take_pos.size}+{take_neg.size} rows ({time.time()-tf:.0f}s)", flush=True)

        te = np.flatnonzero((quad == q).ravel())
        belief = np.zeros(Fs.valid.size, dtype=np.float32)
        for i in range(0, te.size, 500_000):
            idx = te[i:i + 500_000]
            Xt = np.empty((idx.size, len(feats)), dtype=np.float32)
            for j, name in enumerate(feats):
                Xt[:, j] = load_feature(name).reshape(-1)[idx]
            belief[idx] = clf.predict_proba(Xt)[:, 1]
            del Xt
        belief = belief.reshape(Fs.valid.shape)
        np.save(f"/tmp/belief_q{q}.npy", belief)
        del te

        truth_N, truth_P = N & (quad == q), P & (quad == q)
        inc_q, prior_q = inc & (quad == q), prior & (quad == q)
        allowed_q = allowed_all & (quad == q)
        n_q = int(allowed_q.sum())
        print(f"fold {q}: predicting done, allowed {n_q:,} ({time.time()-tf:.0f}s)", flush=True)

        fold = {"quadrant": q, "allowed_px": n_q, "mass": {}, "belief": {}}
        fold["reference"] = {
            "incumbent": score(inc_q, truth_N, Fs.valid, Fs.known),
            "incumbent_frameP": score(inc_q, truth_P, Fs.valid, Fs.known),
            "prior_artifact": score(prior_q, truth_N, Fs.valid, Fs.known),
            "prior_artifact_frameP": score(prior_q, truth_P, Fs.valid, Fs.known),
        }
        for M in MASSES:
            m_q = int(round(M * n_q / n_allowed_total))
            if m_q <= 0:
                continue
            entry: dict = {}
            for blur in BLURS:
                b = ndimage.gaussian_filter(belief, blur) if blur > 0 else belief
                tag = f"greedy_blur{blur:g}"
                d = emit(b, allowed_q, m_q)
                entry[tag] = {"N": score(d, truth_N, Fs.valid, Fs.known),
                              "P": score(d, truth_P, Fs.valid, Fs.known)}
                if blur == 0.0:
                    d_gate = emit(b, allowed_q & gate_70, m_q)
                    entry["greedy_gatedA70"] = {"N": score(d_gate, truth_N, Fs.valid, Fs.known),
                                                "P": score(d_gate, truth_P, Fs.valid, Fs.known)}
                    # the family's measured winning move: no mass within 200 m of the catalogue
                    allowed_q_nohalo = allowed_q & (Fs.d_known > 2.0)
                    d_nh = emit(b, allowed_q_nohalo, m_q)
                    entry["greedy_nohalo2"] = {"N": score(d_nh, truth_N, Fs.valid, Fs.known),
                                               "P": score(d_nh, truth_P, Fs.valid, Fs.known)}
                    d_nhg = emit(b, allowed_q_nohalo & gate_70, m_q)
                    entry["greedy_nohalo2_gatedA70"] = {"N": score(d_nhg, truth_N, Fs.valid, Fs.known),
                                                        "P": score(d_nhg, truth_P, Fs.valid, Fs.known)}
                    del d_nh, d_nhg, allowed_q_nohalo
                    # soft gate: Stage A multiplies the belief (rank-normalised into [0.6, 1.4])
                    a_rank = np.argsort(np.argsort(np.where(Fs.valid, score_A, -np.inf).ravel()))
                    a_w = (0.6 + 0.8 * a_rank / max(a_rank.size - 1, 1)).reshape(Fs.valid.shape).astype(np.float32)
                    d_soft = emit(b * a_w, allowed_q, m_q)
                    entry["greedy_softA"] = {"N": score(d_soft, truth_N, Fs.valid, Fs.known),
                                             "P": score(d_soft, truth_P, Fs.valid, Fs.known)}
                    del a_rank, a_w, d_soft
                # top-n control on the same belief (one mass is enough to make the point)
                if blur == 1.85 and M == 42_000:
                    flat = np.where(allowed_q.ravel(), b.ravel(), -np.inf)
                    top = np.argpartition(-flat, m_q - 1)[:m_q]
                    d_top = np.zeros(Fs.valid.size, dtype=bool)
                    d_top[top] = True
                    d_top = d_top.reshape(Fs.valid.shape)
                    entry["topn"] = {"N": score(d_top, truth_N, Fs.valid, Fs.known),
                                     "P": score(d_top, truth_P, Fs.valid, Fs.known)}
                    del flat, top, d_top
            # Stage A only emission (the control the final file must NOT be dominated by)
            d_a = emit(np.where(allowed_q, score_A, -np.inf), allowed_q, m_q)
            entry["stageA_only"] = {"N": score(d_a, truth_N, Fs.valid, Fs.known),
                                    "P": score(d_a, truth_P, Fs.valid, Fs.known)}
            # uniform control
            ur = np.flatnonzero(allowed_q.ravel())
            pick = rng.choice(ur, size=m_q, replace=False)
            d_u = np.zeros(Fs.valid.size, dtype=bool)
            d_u[pick] = True
            d_u = d_u.reshape(Fs.valid.shape)
            entry["uniform"] = {"N": score(d_u, truth_N, Fs.valid, Fs.known),
                                "P": score(d_u, truth_P, Fs.valid, Fs.known)}
            del d_u, d_a
            fold["mass"][str(M)] = {"fold_mass": m_q, "variants": entry}
            print(f"  fold {q} mass {M}: " + " ".join(
                f"{k}:N={v['N']['DTI']:.4f},P={v['P']['DTI']:.4f}" for k, v in entry.items()), flush=True)
        report["folds"].append(fold)
        out_partial.write_text(json.dumps(report, indent=2) + "\n")

    # ---- aggregate: mean fold delta vs incumbent at each mass x variant -------------
    agg: dict = {}
    for M in MASSES:
        key = str(M)
        variants = set()
        for fold in report["folds"]:
            if key in fold["mass"]:
                variants |= set(fold["mass"][key]["variants"].keys())
        for v in sorted(variants):
            dN, dP, wins = [], [], 0
            for fold in report["folds"]:
                if key not in fold["mass"]:
                    continue
                inc_n = fold["reference"]["incumbent"]["DTI"]
                inc_p = fold["reference"]["incumbent_frameP"]["DTI"]
                val = fold["mass"][key]["variants"].get(v)
                if val is None:
                    continue
                dN.append(val["N"]["DTI"] - inc_n)
                dP.append(val["P"]["DTI"] - inc_p)
                wins += int(val["N"]["DTI"] > inc_n)
            agg[f"{v}@{M}"] = {"mean_delta_N": float(np.mean(dN)), "mean_delta_P": float(np.mean(dP)),
                               "folds_positive_N": wins, "n_folds": len(dN)}
    report["aggregate"] = dict(sorted(agg.items(), key=lambda kv: -kv[1]["mean_delta_N"]))
    best = next(iter(report["aggregate"]))
    b = report["aggregate"][best]
    report["promotion"] = {
        "best_variant": best,
        "gate_pass": bool(b["mean_delta_N"] >= 0.005 and b["folds_positive_N"] >= 3 and b["mean_delta_P"] >= 0.0),
    }
    out_final.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["promotion"], indent=2))
    print(json.dumps({k: report["aggregate"][k] for k in list(report["aggregate"])[:10]}, indent=2))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
