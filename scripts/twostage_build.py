"""Build the two-stage Stage A x Stage B submission artifact, with every diagnostic.

WHAT THIS WRITES
----------------
  submissions/<slug>-zeros.tif      every cell finite, values exactly {0.0, 1.0}, no nodata tag
                                    (cannot trigger the portal's "[0, 1]" rejection)
  submissions/<slug>-nan.tif        identical inside the footprint, NaN outside, nodata=nan
  submissions/<slug>-zeros.zip      the .zip the portal also accepts
  submissions/<slug>-audit.json     the emission + format diagnostics, re-read from the bytes
  registry/twostage/submission.json the machine-readable record the site renders

TWO STAGES, EXPLICITLY SEPARATE
-------------------------------
  Stage A  coarse favourability from coarse physical layers (strain, conductivity, depth to
           basement, seismicity, gravity/magnetics) + fault-corridor density at 2 km and 10 km.
           Output: a ZONE gate.  Its own validation lives in registry/twostage/stage_a.json.
  Stage B  fine-scale placement: gradient-boosted detector on the fine-scale curvature / relief /
           fabric-alignment features, trained on SGMC faults the catalogue lacks, emitted with
           the metric-exact greedy max-coverage emitter.

  The final file is the Stage-B emission restricted to (i) the Stage-A zone and (ii) ground more
  than 200 m from the published catalogue.  Two dominance checks are reported: the final file is
  compared against a Stage-A-only emission and against an ungated Stage-B emission at the same
  mass, so it cannot be "one stage's footprint" wearing a two-stage label.

Run:  PYTHONPATH=src python scripts/twostage_build.py --mass 42000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
from gems44 import field as F
from gems44 import submit as S
from gems44.metric import binary_credit, dti_of, kernel_offsets
from gems44.twostage import frames as FR
from gems44.twostage import stage_a as A
from gems44.twostage import stage_b as SB

FEATDIR = Path("data/interim/feats")
SEED = 44
MIN_SEP = 3.0
NO_HALO_PX = 2.0          # 200 m: no mass within this distance of the published catalogue
GATE_FRACTION = 0.70      # Stage A zone area fraction
HIDDEN_G_DEFAULT = 7905   # family-implied hidden truth size (px); used for the mass diagnostic


def load_feature_names() -> list[str]:
    man = json.loads(Path("registry/twostage/feature_manifest.json").read_text())
    return list(man["features"].keys())


def sample_matrix(rows: np.ndarray, feats: list[str]) -> np.ndarray:
    X = np.empty((rows.size, len(feats)), dtype=np.float32)
    for j, name in enumerate(feats):
        X[:, j] = np.load(FEATDIR / f"{name}.npy", mmap_mode="r").reshape(-1)[rows]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def predict_full(clf, feats: list[str], shape) -> np.ndarray:
    n = int(np.prod(shape))
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, 500_000):
        idx = np.arange(i, min(i + 500_000, n))
        X = sample_matrix(idx, feats)
        out[idx] = clf.predict_proba(X)[:, 1]
        del X
    return out.reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=int, default=42_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", type=str, default="h46-twostageAB")
    args = ap.parse_args()

    t_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rng = np.random.default_rng(args.seed)

    Fs = FR.load_frames()
    P, U, N = Fs.P, Fs.U, Fs.N
    background = Fs.valid & ~(Fs.known | Fs.sgmc)
    feats = load_feature_names()
    print(f"{len(feats)} features; frames {Fs.summary()}", flush=True)

    # ---------------- Stage B: train on all quadrants (fold discipline is in the sweep) --------
    pos_rows = np.flatnonzero(U.ravel())
    pos_rows = rng.choice(pos_rows, size=min(60_000, pos_rows.size), replace=False)
    neg_rows = rng.choice(np.flatnonzero(background.ravel()), size=160_000, replace=False)
    rows = np.concatenate([pos_rows, neg_rows])
    y = np.r_[np.ones(pos_rows.size, np.int8), np.zeros(neg_rows.size, np.int8)]
    X = sample_matrix(rows, feats)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                        min_samples_leaf=40, l2_regularization=1.0,
                                        random_state=SEED, early_stopping=False)
    clf.fit(X, y)
    del X, rows
    belief = predict_full(clf, feats, Fs.valid.shape)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    np.save("data/interim/belief_twostage.npy", belief)
    print(f"Stage B belief: max={belief.max():.4f} mean_footprint={belief[Fs.valid].mean():.5f}", flush=True)

    # ---------------- Stage A: coarse gate -----------------------------------------------------
    bands = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for n in A.STAGE_A_BANDS:
            bands[n] = src.read(SB.BAND_INDEX[n]).astype(np.float32)
    layers = A.stage_a_layers(bands, Fs.known, Fs.valid)
    score_A = A.stage_a_score(layers)
    score_A_phys = A.stage_a_score(layers, A.PHYSICAL_ONLY_WEIGHTS)
    gate = A.apply_gate(score_A, Fs.valid, GATE_FRACTION)
    print(f"Stage A gate: {int(gate.sum()):,} px = {gate.sum()/Fs.valid.sum():.3%} of footprint", flush=True)
    del bands, layers

    # ---------------- emission -----------------------------------------------------------------
    allowed = Fs.valid & ~Fs.known
    allowed_nohalo = allowed & (Fs.d_known > NO_HALO_PX)
    allowed_final = allowed_nohalo & gate
    for name, mask in (("allowed (all off-catalogue)", allowed), ("no-halo", allowed_nohalo),
                       ("no-halo x Stage A gate (SHIPPED)", allowed_final)):
        c = F.quantile_candidates(belief, mask, 0.25)
        print(f"  candidate list  {name:34s} {c.size:,} px", flush=True)

    def emit(mask: np.ndarray, mass: int, b: np.ndarray | None = None) -> np.ndarray:
        b = belief if b is None else b
        cand = F.quantile_candidates(b, mask, 0.25)
        order = F.emit_order_np(b, cand, mask, mass, min_sep=MIN_SEP)
        d = np.zeros(Fs.valid.shape, dtype=bool)
        d.ravel()[order] = True
        return d

    print("emitting final artifact...", flush=True)
    dots = emit(allowed_final, args.mass)
    print(f"final dots: {int(dots.sum()):,}", flush=True)

    # ---------------- controls and dominance checks ---------------------------------------------
    def score(d, truth):
        st = binary_credit(d.astype(np.float64), truth, valid=Fs.valid, known=Fs.known)
        return {"DTI": dti_of(st), "T": st["T"], "F": st["F"], "G": st["G"], "n": st["n"]}

    checks = {}
    for label, truth in (("frame_P_sgmc_beyond_300m", P), ("frame_N_sgmc_within_300m", N),
                         ("frame_U_sgmc_all_off_catalogue", U)):
        checks[label] = {"final": score(dots, truth)}

    with rasterio.open("data/raw/incumbent_d28.tif") as src:
        inc = np.isfinite(src.read(1)) & (src.read(1) > 0.5)
    prior_path = "submissions/GEMS44_n-strand-ssmc_20261006T110212Z_4971d359-zeros.tif"
    prior = None
    if Path(prior_path).exists():
        with rasterio.open(prior_path) as src:
            prior = src.read(1) > 0.5
    for label, truth in (("frame_P_sgmc_beyond_300m", P), ("frame_N_sgmc_within_300m", N)):
        checks[label]["incumbent_d2_8"] = score(inc, truth)
        if prior is not None:
            checks[label]["prior_artifact"] = score(prior, truth)

    # dominance check: same mass, each stage alone.
    # NOTE the Stage-A-only control ranks by the Stage A SCORE, not by the Stage-B belief inside
    # the gate.  Ranking a gate-only emission by the fine-scale model would be a second Stage-B
    # emission wearing a Stage-A label, which is exactly the sort of quiet conflation this whole
    # two-stage split exists to prevent.
    dots_stageA = emit(allowed_nohalo & gate, args.mass,
                       b=np.where(allowed_nohalo & gate, score_A, -np.inf))
    dots_stageB_ungated = emit(allowed, args.mass)                    # no gate, no flank rule
    dots_stageB_nohalo_nogate = emit(allowed_nohalo, args.mass)       # flank rule, no gate
    dom = {
        "stage_A_only_at_same_mass": {f: score(dots_stageA, t) for f, t in (("P", P), ("N", N))},
        "stage_B_ungated_at_same_mass": {f: score(dots_stageB_ungated, t) for f, t in (("P", P), ("N", N))},
        "stage_B_nohalo_nogate_at_same_mass": {f: score(dots_stageB_nohalo_nogate, t)
                                               for f, t in (("P", P), ("N", N))},
        "final": {f: score(dots, t) for f, t in (("P", P), ("N", N))},
        "jaccard_final_vs_stageA_only": float((dots & dots_stageA).sum() / max((dots | dots_stageA).sum(), 1)),
        "jaccard_final_vs_stageB_ungated": float((dots & dots_stageB_ungated).sum() / max((dots | dots_stageB_ungated).sum(), 1)),
    }

    # uniqueness vs every prior artifact available locally
    uniqueness = {}
    for cand_path in sorted(Path("/tmp").glob("GEMSDOE*/docs/downloads/*.tif")) + \
                     sorted(Path("docs/downloads").glob("*.tif")):
        try:
            with rasterio.open(cand_path) as src:
                if (src.height, src.width) != Fs.valid.shape:
                    continue
                a = src.read(1)
        except Exception:
            continue
        d = np.isfinite(a) & (a > 0.5)
        if d.sum() == 0:
            continue
        j = float((dots & d).sum() / max((dots | d).sum(), 1))
        uniqueness[cand_path.name[:70]] = {"jaccard": j, "shared": int((dots & d).sum()),
                                           "mass": int(d.sum())}
        del a, d
    worst = max(uniqueness.values(), key=lambda v: v["jaccard"]) if uniqueness else {"jaccard": 0.0}

    # ---------------- write the artifact --------------------------------------------------------
    with rasterio.open("data/raw/sample_submission.tif") as src:
        contract = S.Contract(src.height, src.width, src.transform, str(src.crs), Fs.valid)
    slug = f"GEMS44_{args.tag}_{t_utc}_{hashlib.sha256(dots.tobytes()).hexdigest()[:8]}"
    out = Path("submissions"); out.mkdir(exist_ok=True)
    primary = S.write_twin(out / f"{slug}-zeros.tif", dots, contract, outside="zeros",
                           name=f"{slug} two-stage Stage A x Stage B")
    nan_twin = S.write_twin(out / f"{slug}-nan.tif", dots, contract, outside="nan")
    zpath = S.make_zip(out / f"{slug}-zeros.tif", out / f"{slug}-zeros.zip")

    ks = float(sum(w for _, _, w in kernel_offsets()))
    c_per_dot_P = checks["frame_P_sgmc_beyond_300m"]["final"]["T"] / max(int(dots.sum()), 1)
    audit = {
        "slug": slug, "built_utc": t_utc, "tag": args.tag, "seed": args.seed,
        "mass": int(dots.sum()), "requested_mass": args.mass,
        "min_sep_px": MIN_SEP, "no_halo_px": NO_HALO_PX, "stage_a_gate_fraction": GATE_FRACTION,
        "stage_a": A.validate_stage_a(score_A, Fs.valid, Fs.known, Fs.U, Fs.quad, GATE_FRACTION),
        "stage_a_honesty": {
            "shipped_weights": A.STAGE_A_WEIGHTS,
            "physical_only_weights": A.PHYSICAL_ONLY_WEIGHTS,
            "physical_only_validation": A.validate_stage_a(score_A_phys, Fs.valid, Fs.known, Fs.U,
                                                           Fs.quad, GATE_FRACTION),
            "note": ("the shipped score includes catalogue-density layers and is validated against "
                     "'does this 2 km cell contain catalogue faults', so its blocked AUC is "
                     "self-referential (a smoothed fault map predicting a fault map).  It is a "
                     "legitimate GATE input -- the catalogue is known, and it is never a pixel "
                     "target -- but the physical-only AUC is the honest number for the physical "
                     "layers themselves."),
        },
        "stage_b": {"features": feats, "train_positives": int(pos_rows.size),
                    "train_negatives": int(neg_rows.size), "target": "frame U (SGMC faults the catalogue lacks)",
                    "model": "HistGradientBoostingClassifier(300 iters, lr 0.06, 31 leaves)"},
        "frames": Fs.summary(),
        "scores": checks,
        "stage_dominance_check": dom,
        "calibrated_instrument": {
            "frame_P_credit_per_dot": float(c_per_dot_P),
            "kernel_sum_offsets": ks,
            "hidden_g_assumed_px": HIDDEN_G_DEFAULT,
            "dti_at_hidden_density": float(
                checks["frame_P_sgmc_beyond_300m"]["final"]["T"]
                / (0.2 * int(dots.sum()) + 0.8 * HIDDEN_G_DEFAULT)),
            "note": ("credit per dot on frame P is the statistic that reproduced the family's reported "
                     "live ordering 6/6 (scripts/calibrate_instrument.py); the hidden-density DTI uses "
                     "the family-implied hidden truth size and is a scale check, not a prediction"),
        },
        "uniqueness": {"max_jaccard_vs_any_local_artifact": float(worst["jaccard"]),
                       "n_artifacts_compared": len(uniqueness),
                       "per_artifact": dict(sorted(uniqueness.items(), key=lambda kv: -kv[1]["jaccard"])[:12])},
        "emission_diagnostics": {
            "dots": int(dots.sum()),
            "dots_on_known_mask": int((dots & Fs.known).sum()),
            "within_200m_of_catalogue": int((dots & (Fs.d_known <= 2)).sum()),
            "within_300m_of_catalogue": int((dots & (Fs.d_known <= 3)).sum()),
            "inside_stage_a_gate": int((dots & gate).sum()),
            "frac_inside_stage_a_gate": float((dots & gate).sum() / max(int(dots.sum()), 1)),
        },
        "format": {"primary": primary, "nan_twin": nan_twin, "zip": zpath},
    }
    (out / f"{slug}-audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    reg = Path("registry/twostage"); reg.mkdir(parents=True, exist_ok=True)
    (reg / "submission.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")

    print(json.dumps({k: audit[k] for k in ("slug", "mass", "calibrated_instrument",
                                           "emission_diagnostics", "uniqueness")}, indent=2)[:2500])
    print("format checks:", json.dumps(primary["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
