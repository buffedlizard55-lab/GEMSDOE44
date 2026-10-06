"""Instrument calibration: which local statistic actually ranks the family's reported live scores?

WHY
---
Every local number in this project is a proxy.  The family reports live scores for many
artifacts; the artifacts themselves are public.  Any statistic that is going to *choose*
what we submit must first be shown to reproduce those reported scores.  This script does
that, on as many artifacts as are locally available, and reports the rank correlation of
every candidate instrument.

INSTRUMENTS TESTED
------------------
  P_credit_per_dot  mean realised credit (TP_w / n) of the artifact's own dots against
                    frame P truth (SGMC faults further than 300 m from the competition
                    catalogue).  Mass-corrected by construction: it never rewards emitting
                    more dots.
  P_dti             official DTI against frame P at the artifact's own mass
  N_dti             official DTI against frame N (SGMC strands within 300 m of the
                    catalogue) -- the frame the previous session promoted on
  N_excess          N_dti minus a uniform-random control at the same mass (the previous
                    session's promotion statistic)
  P_excess          P_dti minus the same control
  catalogue_dti     official DTI against the raw catalogue with the mask DISABLED
                    (contaminated: this is what the model may have been trained on)

Everything is computed with the repository's exact metric implementation and the
known-fault mask applied exactly as the official scorer applies it.

Run:  PYTHONPATH=src python scripts/calibrate_instrument.py
Writes: registry/twostage/instrument_calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from gems44.metric import binary_credit, dti_of, kernel_offsets
from gems44.twostage import frames as FR

SEED = 44
KERNEL_SUM = float(sum(w for _, _, w in kernel_offsets()))

# (label, path, owner-reported live score).  Scores are from the owner's own sites, as
# transcribed in the brief and in the family registries; they are NOT organizer receipts
# (IR-44-08) and are used here as an ORDER only.
CANDIDATES: list[tuple[str, str, float | None]] = []


def add(label: str, path: str, score: float | None) -> None:
    if Path(path).exists():
        CANDIDATES.append((label, path, score))


def discover() -> None:
    """Every artifact the sandbox has locally, with its owner-reported score."""
    roots = {
        "/tmp/GEMSDOE24/docs/downloads": None, "/tmp/GEMSDOE25/docs/downloads": None,
        "/tmp/GEMSDOE27/docs/downloads": None, "/tmp/GEMSDOE28/docs/downloads": None,
        "/tmp/GEMSDOE31/docs/downloads": None, "/tmp/GEMSDOE32/docs/downloads": None,
        "/tmp/GEMSDOE33/docs/downloads": None, "/tmp/GEMSDOE36/docs/downloads": None,
        "/tmp/GEMSDOE23/docs/downloads": None, "/tmp/GEMSDOE29/docs/downloads": None,
        "/tmp/GEMSDOE30/docs/downloads": None, "/tmp/GEMSDOE10/docs/downloads": None,
        "/tmp/12GEMSDOE/docs/downloads": None, "/tmp/13GEMSDOE/docs/downloads": None,
        "/tmp/17GEMSDOE/docs/downloads": None, "/tmp/20GEMSDOE/docs/downloads": None,
        "/tmp/19GEMSDOE/docs/downloads": None, "/tmp/GEMSDOE21/docs/downloads": None,
        "/tmp/GEMSDOE37/docs/downloads": None, "/tmp/GEMSDOE38/docs/downloads": None,
        "/tmp/GEMSDOE40/docs/downloads": None, "/tmp/GEMSDOE39/docs/downloads": None,
        "/tmp/GEMSDOE34/docs/downloads": None, "/tmp/GEMSDOE35/docs/downloads": None,
        "/tmp/GEMSDOE41/docs/downloads": None,
    }
    reported = {
        # fragment -> owner-reported live score (0.0 = reported, score not stated)
        "h25-1-dotted-h19-5-d2-8": 0.2600, "dotted-h19-5-d2-8": 0.2600,
        "h25-1-dotted-h19-5-d1-5": 0.2477, "reference-h19-5": 0.1894,
        "h24-0-mirror-h19-5": 0.1922,
        "h27-4-r1-solo-d2-8": 0.2708, "h27-4-solo-d28": 0.2708, "h27-4-solo-d2-8": 0.2708,
        "h36-1-rung30-blind-r1": 0.2710, "h32-1-prethin-tip-euler": 0.2649,
        "h33-h33-2-b2": 0.2778, "h32d-submodular-multipysics": None,
        "topo-gap-closure-t-v2-on-d1-5": 0.2449, "topo-gap-closure-t-v2-on-d2-8": None,
        "all-increments-d2-8-h27-4-r1": None, "h28-1-edge-coherence-plus": None,
        "gemsdoe31-add-arm-d28": None, "gemsdoe31-reference-d28-offcat-44090": None,
        "h33d-analog-tip-stepover-r30": 0.2632, "anderson-geothermal-pinn": 0.2750,
        "h30-arrangement-matched-habitat": 0.1352, "repo-c0-habitat-emission": 0.0041,
        "sgmc-off-catalogue-44k": 0.0512, "efd28-repro": 0.2600,
        "d28-poisson300m-offcat-44090": 0.2600, "r7-nms3-dem10-scarp": 0.1294,
        "r13-lattice-s5_v2": 0.0904, "h16-continuation": 0.0461,
        "h20-dem10-scarp-thin": 0.0921, "h25-ctx-ridge": 0.1280, "h28-dotted-ridge": 0.1839,
        "h20-1-sarnnpu-powerlaw": 0.1890, "h20-5-continuous-pu-proxy": 0.1859,
        "h19-4-reference": 0.1894, "h23-a-dti-optimal-emission": 0.1002,
        "h16-1-topo-geophys-baseline-ridges": 0.1855, "h18-3a-topo-geophys-x-complexity": 0.0976,
        "h19-5-powerlaw-budget-multiline-corroborated": 0.1922,
        "gemsdoe2-dual-family-union": 0.1560, "pindrop-v4-nodes": 0.1193,
        "dilcond-oof-v1": 0.1223, "dem10-scarp": None,
    }
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.tif")):
            if "nan" in f.name and (p / f.name.replace("-nan", "-zeros")).exists():
                continue
            score = None
            for frag, s in reported.items():
                if frag in f.name:
                    score = s
                    break
            add(f.name[:52], str(f), score)


def uniform_dti_analytic(mass: int, truth_px: int, n_foot: int, kernel_sum: float) -> float:
    """Exact-in-expectation DTI of a uniform-random dot file (thin approximation).

    For randomly placed dots the expected credit per dot is c0 = truth_px * kernel_sum / n_foot
    (linearity of the kernel sums; verified against a Monte-Carlo control in
    ``scripts/verify_uniform_control.py``).  Then T = c0 * mass, M = T (the same quantity),
    FP = mass - T, so DTI = T / (0.2*mass + 0.8*truth_px).
    """
    c0 = truth_px * kernel_sum / n_foot
    return float(c0 * mass / (0.2 * mass + 0.8 * truth_px))


def uniform_dti_montecarlo(mass: int, truth: np.ndarray, valid: np.ndarray, known: np.ndarray,
                           rng, n_draw: int = 4) -> float:
    """Monte-Carlo check of the analytic control (few draws; exact metric each time)."""
    flat = np.flatnonzero((valid & ~known).ravel())
    vals = []
    for _ in range(n_draw):
        pick = rng.choice(flat, size=min(mass, flat.size), replace=False)
        d = np.zeros(valid.size, dtype=bool)
        d[pick] = True
        st = binary_credit(d.reshape(valid.shape).astype(float), truth, valid=valid, known=known)
        vals.append(dti_of(st))
    return float(np.mean(vals))


def main() -> int:
    discover()
    add("INCUMBENT d2.8 (live 0.2600)", "data/raw/incumbent_d28.tif", 0.2600)
    add("GEMS44 prior artifact (unscored)", "submissions/GEMS44_n-strand-ssmc_20261006T110212Z_4971d359-zeros.tif", None)
    print(f"{len(CANDIDATES)} local artifacts found")

    F = FR.load_frames()
    P, N, U = F.P, F.N, F.U
    rng = np.random.default_rng(SEED)
    allowed = F.valid & ~F.known

    rows = []
    seen: set[tuple[int, float]] = set()
    for label, path, score in CANDIDATES:
        try:
            with rasterio.open(path) as src:
                if (src.height, src.width) != F.valid.shape:
                    print(f"  skip (shape {src.height}x{src.width} != grid): {label[:40]}", flush=True)
                    continue
                a = src.read(1)
        except Exception as exc:                       # unreadable file: skip, never crash
            print(f"  skip ({type(exc).__name__}): {label[:40]}", flush=True)
            continue
        d = np.isfinite(a) & (a > 0.5)
        if d.sum() == 0:
            continue
        mass = int(d.sum())
        if mass == 0:
            continue
        if (mass, float(d.sum())) in seen:
            continue
        stP = binary_credit(d.astype(float), P, valid=F.valid, known=F.known)
        stN = binary_credit(d.astype(float), N, valid=F.valid, known=F.known)
        stC = binary_credit(d.astype(float), F.known, valid=F.valid, known=None)
        uP_dti = uniform_dti_analytic(mass, int(P.sum()), int(F.valid.sum()), KERNEL_SUM)
        uN_dti = uniform_dti_analytic(mass, int(N.sum()), int(F.valid.sum()), KERNEL_SUM)
        key = (mass, round(float(stP["T"] / mass), 6))
        if key in seen:                                # nan/zeros twins of the same bytes
            continue
        seen.add(key)
        rows.append({
            "label": label, "path": path, "reported_score": score, "mass": mass,
            "score_provenance": ("owner-reported on a family site (IR-44-08)"
                                 + ("; attribution of 0.2778 to this file is unsupported (IR-44-03)"
                                    if score == 0.2778 else "")),
            "P_credit_per_dot": float(stP["T"] / mass),
            "P_dti": dti_of(stP), "P_excess": dti_of(stP) - uP_dti, "uniform_P_dti": uP_dti,
            "N_dti": dti_of(stN), "N_excess": dti_of(stN) - uN_dti, "uniform_N_dti": uN_dti,
            "catalogue_dti_nomask": dti_of(stC),
        })
        print(f"{label[:44]:44s} mass={mass:7d} c/dot_P={rows[-1]['P_credit_per_dot']:.4f} "
              f"P={rows[-1]['P_dti']:.4f} N={rows[-1]['N_dti']:.4f} score={score}", flush=True)

    # ---- analytic control vs Monte Carlo (documented check, printed and stored) -----
    mc_checks = {}
    for label, mass in [(rows[0]["label"], rows[0]["mass"])] if rows else []:
        for frame_name, truth in (("P", P), ("N", N)):
            mc = uniform_dti_montecarlo(mass, truth, F.valid, F.known, np.random.default_rng(SEED))
            an = uniform_dti_analytic(mass, int(truth.sum()), int(F.valid.sum()), KERNEL_SUM)
            mc_checks[f"{label}|frame_{frame_name}"] = {"monte_carlo_4draws": mc, "analytic": an,
                                                        "abs_diff": abs(mc - an)}
            print(f"uniform control check {frame_name}: MC={mc:.4f} analytic={an:.4f} diff={abs(mc-an):.4f}",
                  flush=True)
    scored = [r for r in rows if r["reported_score"] is not None]
    instruments = ["P_credit_per_dot", "P_dti", "P_excess", "N_dti", "N_excess", "catalogue_dti_nomask"]
    calib = {}
    for ins in instruments:
        x = [r[ins] for r in scored]
        y = [r["reported_score"] for r in scored]
        rho = spearmanr(x, y)
        calib[ins] = {"spearman": float(rho.statistic), "p_value": float(rho.pvalue), "n": len(scored)}
    contested = [r for r in scored if r["reported_score"] == 0.2778]
    scored_conservative = [r for r in scored if r not in contested]
    calib_conservative = {}
    for ins in instruments:
        if len(scored_conservative) >= 3:
            rho = spearmanr([r[ins] for r in scored_conservative],
                            [r["reported_score"] for r in scored_conservative])
            calib_conservative[ins] = {"spearman": float(rho.statistic),
                                       "p_value": float(rho.pvalue), "n": len(scored_conservative)}

    unscored_ranked = sorted([r for r in rows if r["reported_score"] is None],
                             key=lambda r: -r["P_credit_per_dot"])[:10]

    lineage = [r for r in scored if r["label"].split()[0][:3] in ("h25", "d1.", "d2.", "gems", "h27", "h36", "h33")
               and ("d2-8" in r["label"] or "d1-5" in r["label"] or "reference-h19-5" in r["label"]
                    or "solo" in r["label"] or "rung30" in r["label"] or "h33-2-b2" in r["label"])]
    lin_calib = {}
    if len(lineage) >= 3:
        for ins in instruments:
            x = [r[ins] for r in lineage]
            y = [r["reported_score"] for r in lineage]
            rho = spearmanr(x, y)
            lin_calib[ins] = {"spearman": float(rho.statistic), "p_value": float(rho.pvalue), "n": len(lineage)}

    out = Path("registry/twostage")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": ("owner-reported live scores are not organizer receipts (IR-44-08); they are used as an "
                 "ORDER.  Every metric number here is this repository's exact implementation of the "
                 "published DTI, with the official known-fault masking."),
        "uniform_control_check": mc_checks,
        "kernel_sum_offsets": KERNEL_SUM,
        "n_artifacts": len(rows),
        "n_with_reported_score": len(scored),
        "instrument_rank_correlation_all": calib,
        "instrument_rank_correlation_excluding_contested_0p2778": calib_conservative,
        "unscored_local_candidates_by_frame_P_credit_per_dot": [
            {"label": r["label"], "mass": r["mass"], "P_credit_per_dot": r["P_credit_per_dot"],
             "P_dti": r["P_dti"], "N_dti": r["N_dti"]} for r in unscored_ranked],
        "lineage_labels": [r["label"] for r in lineage],
        "instrument_rank_correlation_lineage": lin_calib,
        "per_artifact": rows,
    }
    (out / "instrument_calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nAll artifacts (n=%d):" % len(scored))
    for k, v in sorted(calib.items(), key=lambda kv: -abs(kv[1]["spearman"])):
        print(f"  {k:24s} rho={v['spearman']:+.3f} p={v['p_value']:.4f}")
    print("Lineage subset (n=%d):" % len(lineage))
    for k, v in sorted(lin_calib.items(), key=lambda kv: -abs(kv[1]["spearman"])):
        print(f"  {k:24s} rho={v['spearman']:+.3f} p={v['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
