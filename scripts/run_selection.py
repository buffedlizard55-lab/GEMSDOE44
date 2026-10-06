"""Metric-aligned, spatially blocked field + emission selection.  THE decisive experiment.

What changed and why (protocol amendment, recorded in registry/preregistration.json)
------------------------------------------------------------------------------------
The first version of this script promoted on frame P (SGMC faults >300 m from the catalogue).
Before reading any of its results, a fabricated-control test was run on 19 family artifacts whose
live scores were reported (scripts/rank_frames.py): frame P reproduces the live order with
rho_level = -0.054 and rho_excess = +0.19, i.e. it is useless-to-inverted.  Frame N (SGMC
strands within 300 m of the catalogue, 17,493 px) is the only frame with a positive sign
(rho_level +0.335, rho_excess +0.396) and all 19 artifacts beat a uniform control on it.
The gate below therefore applies to frame N, and frame P is still reported as the falsified case.

Blocking and mass scaling
-------------------------
Four quadrants, leave-one-quadrant-out.  A learned field is re-fitted inside each fold on the
three training quadrants and used only on the held-out one.  Mass is scaled to the fold's share of
the allowed area, so that the fold density equals the density of a global submission of that mass:
    mass_fold = M_global * allowed_q / allowed_total
This makes a fold DTI an unbiased estimate of the global DTI under spatial stationarity, because
`DTI = T/(0.2(T+F)+0.8G)` is invariant when the numerator and both denominator terms scale with
the region's share of truth and dots.

Writes registry/selection.json (and registry/selection_partial.json after every fold).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import field as F  # noqa: E402
from gems44 import grid as G  # noqa: E402
from gems44.metric import binary_credit, dti_of  # noqa: E402
from gems44.scripts_common import (  # noqa: E402
    ROW_BLOCK, feature_block, load_bands, matrix_at, sample_rows,
)

GLOBAL_MASSES = [30_000, 44_090, 61_328]
CAND_FRACTION = 0.06
TRAIN_POS, TRAIN_NEG = 8_000, 30_000
SEED = 44
MIN_SEP = 3.0  # H7: no dot within 3 px (300 m) of another; see field.emit_order_np


def emit(belief: np.ndarray, allowed: np.ndarray, max_dots: int) -> np.ndarray:
    cand = F.quantile_candidates(belief, allowed, CAND_FRACTION)
    return F.emit_order_np(belief, cand, allowed, max_dots, min_sep=MIN_SEP)


def score(order: np.ndarray, shape, mass: int, truth, valid, known) -> float:
    d = np.zeros(shape, dtype=bool)
    if mass > 0:
        d.ravel()[order[:mass]] = True
    return float(dti_of(binary_credit(d, truth, valid=valid, known=known)))


def train(bands, width, pos_mask, neg_mask, rng):
    pos = sample_rows(pos_mask, TRAIN_POS, rng)
    neg = sample_rows(neg_mask, TRAIN_NEG, rng)
    if pos.size < 200 or neg.size < 200:
        return None
    X = np.vstack([matrix_at(bands, width, pos), matrix_at(bands, width, neg)])
    y = np.r_[np.ones(pos.size, np.int8), np.zeros(neg.size, np.int8)]
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       random_state=0, early_stopping=False)
    m.fit(X, y)
    del X
    return m


def predict_region(model, bands, width, r0, r1, c0, c1) -> np.ndarray:
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
    for rb in range(r0, r1, ROW_BLOCK):
        r1b = min(rb + ROW_BLOCK, r1)
        fb = feature_block(bands, rb, r1b)
        sc = model.predict_proba(fb.reshape(-1, fb.shape[-1]))[:, 1]
        out[rb - r0:r1b - r0] = sc.reshape(r1b - rb, width)[:, c0:c1]
        del fb, sc
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known
    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat).astype(np.float32)
    frame = {
        "N": sgmc & (d_cat <= 3.0),
        "P": sgmc & (d_cat > 3.0),
    }
    allowed_all = g.footprint & ~g.known
    print(f"frame N truth {int(frame['N'].sum()):,} px | frame P truth {int(frame['P'].sum()):,} px "
          f"| allowed {int(allowed_all.sum()):,} px")

    quad = np.zeros(labels.shape, np.int8)
    quad[: g.height // 2, g.width // 2:] = 1
    quad[g.height // 2:, : g.width // 2] = 2
    quad[g.height // 2:, g.width // 2:] = 3

    bands = load_bands()
    print("analytic shared fields...")
    shared = {f"prox{s}": np.exp(-d_cat / s).astype(np.float32) for s in (3, 10, 30)}
    d2b = bands["depth_to_basement"].astype(np.float64)
    fin = np.isfinite(d2b)
    med = float(np.median(d2b[fin]))
    d2b = np.where(fin, d2b, med)
    rank = np.empty(d2b.size, np.float64)
    rank[np.argsort(d2b, axis=None)] = np.arange(d2b.size, dtype=np.float64)
    shared["prox10_conceal"] = (shared["prox10"] *
                                (0.35 + 1.65 * (rank / d2b.size).reshape(d2b.shape).astype(np.float32))
                                ).astype(np.float32)
    del d2b, rank, fin

    inc = None
    if Path("data/raw/incumbent_d28.tif").exists():
        with rasterio.open("data/raw/incumbent_d28.tif") as src:
            v = src.read(1)
        inc = np.isfinite(v) & (v > 0)

    # Trimmed family: prox3/prox30/prox10_conceal proved identical or uninformative on fold 0 of
    # the preceding run (prox3 == prox10 == prox30 exactly, since both order candidates by the same
    # monotone distance), and blend_N was dominated by blend_U on frame N.  The controls that matter
    # are kept: the pure proximity stencil (prox10), the two supervised fields, and the blend.
    family = ["prox10", "sup_N", "sup_U", "blend_U"]
    report: dict = {
        "design": {
            "frames": {"N": "sgmc strands within 300 m of the catalogue (PRIMARY)",
                       "P": "sgmc strands beyond 300 m of the catalogue (falsified frame, secondary)"},
            "frame_truth_px": {k: int(v.sum()) for k, v in frame.items()},
            "global_masses": GLOBAL_MASSES,
            "mass_scaling": "mass_fold = M_global * allowed_q / allowed_total",
            "candidate_fraction": CAND_FRACTION, "min_sep_px": MIN_SEP,
            "fields": family,
            "leakage_discipline": "learned fields re-fitted per fold on the three training quadrants only",
            "gate": "frame N: mean fold delta vs incumbent >= +0.005 and >= 3 of 4 folds positive",
            "amendment": "primary frame changed from P to N by scripts/rank_frames.py before any sweep "
                         "result was read; recorded in registry/preregistration.json and IR-44-09",
        },
        "fold_results": {},
    }
    total_allowed = int(allowed_all.sum())

    for q in range(4):
      try:
        r0g, r1g = (0, g.height // 2) if q < 2 else (g.height // 2, g.height)
        c0, c1 = (0, g.width // 2) if q % 2 == 0 else (g.width // 2, g.width)
        test = quad == q
        allowed = allowed_all & test
        n_allowed = int(allowed.sum())
        masses = {M: max(int(round(M * n_allowed / total_allowed)), 1) for M in GLOBAL_MASSES}
        max_mass = max(masses.values())
        row: dict = {"allowed_px": n_allowed, "fold_masses": {str(k): v for k, v in masses.items()},
                     "frame_truth_px": {k: int((v & test).sum()) for k, v in frame.items()}}
        print(f"\nfold {q}: allowed={n_allowed:,} masses={masses}")

        train_bg = allowed_all & ~test & ~sgmc
        models = {
            "sup_N": train(bands, g.width, frame["N"] & ~test, train_bg, rng),
            "sup_U": train(bands, g.width, sgmc & ~test, train_bg, rng),
        }
        print(f"   [fold {q}] models fitted", flush=True)
        fields = {k: v[r0g:r1g, c0:c1] for k, v in shared.items()}
        for name, model in models.items():
            if model is None:
                continue
            fields[name] = predict_region(model, bands, g.width, r0g, r1g, c0, c1)
        if "sup_N" in fields and "sup_U" in fields:
            for nm, key in (("blend_U", "sup_U"),):
                fields[nm] = (np.sqrt(np.maximum(fields[key], 0.0) * fields["prox10"])).astype(np.float32)

        for fname in family:
            if fname not in fields:
                continue
            b = np.zeros(labels.shape, np.float32)
            b[r0g:r1g, c0:c1] = fields[fname]
            order = emit(np.where(test, b, 0.0).astype(np.float32), allowed, max_mass)
            row[fname] = {str(M): round(score(order, labels.shape, masses[M], frame["N"] & test,
                                              g.footprint, g.known), 6) for M in GLOBAL_MASSES}
            row[fname + "__frameP"] = {str(M): round(score(order, labels.shape, masses[M],
                                                            frame["P"] & test, g.footprint, g.known), 6)
                                       for M in GLOBAL_MASSES}
            print(f"   {fname:16s} N: " + " ".join(f"{M//1000}k={row[fname][str(M)]:.4f}" for M in GLOBAL_MASSES)
                  + "  | P@44090=" + f"{row[fname + '__frameP']['44090']:.4f}")

        for fname, cmd in (("incumbent", None), ("uniform", None)):
            if fname == "incumbent":
                if inc is None:
                    continue
                dots = inc & test
            else:
                rr = np.random.default_rng(7 + q)
                idx = rr.choice(np.nonzero(allowed.ravel())[0], size=min(max_mass, n_allowed), replace=False)
                dots = np.zeros(labels.shape, bool)
                dots.ravel()[idx] = True
            row[fname] = {}
            row[fname + "__frameP"] = {}
            for M, m in masses.items():
                if fname == "uniform":
                    st = binary_credit(dots, frame["N"] & test, valid=g.footprint, known=g.known)
                    sp = binary_credit(dots, frame["P"] & test, valid=g.footprint, known=g.known)
                    row[fname][str(M)] = round(dti_of(st), 6)
                    row[fname + "__frameP"][str(M)] = round(dti_of(sp), 6)
                else:
                    # the incumbent is compared at the SAME dot density; a raster-order prefix of
                    # its dots would sample only the top rows, so draw a uniform random subset
                    dots_idx = np.flatnonzero(dots.ravel())
                    si = np.random.default_rng(1000 + q).choice(
                        dots_idx, size=min(m, dots_idx.size), replace=False)
                    row[fname][str(M)] = round(score(si, labels.shape, si.size,
                                                     frame["N"] & test, g.footprint, g.known), 6)
                    row[fname + "__frameP"][str(M)] = round(score(si, labels.shape, si.size,
                                                                  frame["P"] & test, g.footprint, g.known), 6)
            print(f"   {fname:16s} N: " + " ".join(f"{M//1000}k={row[fname][str(M)]:.4f}" for M in GLOBAL_MASSES))

        report["fold_results"][str(q)] = row
        Path("registry").mkdir(exist_ok=True)
        Path("registry/selection_partial.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        del fields, models
      except Exception as exc:  # noqa: BLE001
        print(f"fold {q} FAILED: {type(exc).__name__}: {exc}")
        report["fold_results"][str(q)] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- leave-one-quadrant-out choice on frame N
    folds = sorted(report["fold_results"], key=int)
    combos = []
    for fname in family:
        for M in GLOBAL_MASSES:
            vals = [report["fold_results"][f].get(fname, {}).get(str(M)) for f in folds]
            if any(v is None for v in vals):
                continue
            combos.append((fname, M, [float(v) for v in vals]))
    loo = []
    for i, q in enumerate(folds):
        others = [j for j in range(len(folds)) if j != i]
        best = max(combos, key=lambda c: float(np.mean([c[2][j] for j in others])))
        inc_i = report["fold_results"][q]["incumbent"][str(best[1])]
        loo.append({"fold": int(q), "chosen_field": best[0], "chosen_mass": best[1],
                    "heldout_dti": round(best[2][i], 6), "incumbent_heldout_dti": inc_i,
                    "delta_vs_incumbent": round(best[2][i] - inc_i, 6),
                    "uniform_heldout_dti": report["fold_results"][q]["uniform"][str(best[1])]})
    d = [x["delta_vs_incumbent"] for x in loo]
    best_all = max(combos, key=lambda c: float(np.mean(c[2])))
    report["leave_one_fold_out"] = loo
    report["pooled"] = {
        "candidate_heldout_dti": round(float(np.mean([x["heldout_dti"] for x in loo])), 6),
        "incumbent_heldout_dti": round(float(np.mean([x["incumbent_heldout_dti"] for x in loo])), 6),
        "delta": round(float(np.mean(d)), 6), "positive_folds": int(sum(1 for x in d if x > 0)),
        "n_folds": len(loo), "gate_pass": bool(np.mean(d) >= 0.005 and sum(1 for x in d if x > 0) >= 3),
    }
    report["best_mean_over_all_folds"] = {
        "field": best_all[0], "mass": best_all[1], "mean_dti": round(float(np.mean(best_all[2])), 6),
        "note": "descriptive; the promotion decision uses leave_one_fold_out",
    }
    Path("registry/selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== FRAME N, mean fold DTI (density-matched mass) ===")
    print(f"{'field':18s}" + "".join(f"{M:>9d}" for M in GLOBAL_MASSES))
    for fname in family:
        cells = []
        for M in GLOBAL_MASSES:
            v = [report["fold_results"][f].get(fname, {}).get(str(M)) for f in folds]
            cells.append(f"{np.mean([x for x in v if x is not None]):>9.4f}" if any(x is not None for x in v) else f"{'-':>9s}")
        print(f"{fname:18s}" + "".join(cells))
    for ctrl in ("incumbent", "uniform"):
        if all(str(M) in report["fold_results"][f].get(ctrl, {}) for f in folds for M in GLOBAL_MASSES):
            print(f"{ctrl:18s}" + "".join(f"{np.mean([report['fold_results'][f][ctrl][str(M)] for f in folds]):>9.4f}"
                                          for M in GLOBAL_MASSES))
    print("\n" + json.dumps({"pooled": report["pooled"],
                             "best_mean": report["best_mean_over_all_folds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
