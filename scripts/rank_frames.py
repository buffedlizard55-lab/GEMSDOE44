"""Rank candidate LOCAL frames by how well they reproduce the reported live-score order.

19 family artifacts are on disk with reported live scores spanning 0.0461 -> 0.2600 (probe set
fetched by scripts/fetch_probes.py).  All of them are strictly binary {0,1}, so the exact binary
operator applies: two Euclidean distance transforms per (artifact, frame).

For each candidate frame we report
  rho_level  : Spearman(frame DTI, reported score)
  rho_excess : Spearman(frame DTI - uniform control at the same mass, reported score)
The excess removes the mass effect, which matters because probe masses span 44,090 -> 335,879.

A frame is only usable for promotion if rho is high AND the sign is positive: the frame must put
the artifacts that the live board scores higher, higher.

Writes registry/frame_ranking.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import grid as G  # noqa: E402
from gems44.metric import binary_credit, dti_of  # noqa: E402


ANCHOR = Path("/home/user/prior/GEMSDOE32/docs/downloads/"
              "gems32-probe-S1-ANCHOR-identical-to-live-02600.tif")
SHIPPED_02778 = Path("/home/user/prior/GEMSDOE32/docs/downloads/"
                     "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif")


def _dots(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        v = src.read(1)
    return np.isfinite(v) & (v > 0)


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    cat = g.footprint & (labels > 0)
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known
    d_cat = ndimage.distance_transform_edt(~cat)

    frames = {
        "N_sgmc_within_300m": sgmc & (d_cat <= 3.0),
        "P_sgmc_beyond_300m": sgmc & (d_cat > 3.0),
        "Q_sgmc_200_300m": sgmc & (d_cat > 2.0) & (d_cat <= 3.0),
        "R_sgmc_beyond_200m": sgmc & (d_cat > 2.0),
        "U_sgmc_all_off_catalogue": sgmc.copy(),
        "S0_catalogue_nomask": cat,
    }
    manifest = json.loads(Path("registry/probe_manifest.json").read_text())["probes"]
    probes = [(Path(m["file"]).name, m["reported_score"]) for m in manifest if m["status"] != "NOT FOUND"]
    probes.sort(key=lambda t: t[1])

    rng = np.random.default_rng(44)
    allowed = g.footprint & ~g.known
    allowed_idx = np.nonzero(allowed.ravel())[0]
    rows = []
    for name, score in probes:
        with rasterio.open(Path("data/probes") / name) as src:
            dots = src.read(1) > 0.5
        n = int(dots.sum())
        idx = rng.choice(allowed_idx, size=min(n, allowed_idx.size), replace=False)
        u = np.zeros(g.footprint.shape, bool)
        u.ravel()[idx] = True
        row = {"file": name, "reported_score": score, "mass": n}
        for fname, truth in frames.items():
            # The catalogue frame must be scored WITHOUT the known mask, otherwise its truth is
            # deleted by that mask and every artifact scores exactly 0 (which is what happened on
            # the first run and is itself the proof that catalogue frames are degenerate).
            kn = None if fname.startswith("S0") else g.known
            st = binary_credit(dots, truth, valid=g.footprint, known=kn)
            su = binary_credit(u, truth, valid=g.footprint, known=kn)
            row[f"dti_{fname}"] = round(dti_of(st), 6)
            row[f"uniform_{fname}"] = round(dti_of(su), 6)
        rows.append(row)
        print(f"{name[:46]:48s} live={score:.4f} n={n:>7,} " +
              " ".join(f"{k.split('_',1)[1]}={row['dti_'+k]:.4f}" for k in frames))

    reported = np.array([r["reported_score"] for r in rows])
    verdict = {}
    for fname in frames:
        lvl = np.array([r[f"dti_{fname}"] for r in rows])
        exc = np.array([r[f"dti_{fname}"] - r[f"uniform_{fname}"] for r in rows])
        verdict[fname] = {
            "truth_px": int(frames[fname].sum()),
            "rho_level": round(float(spearmanr(reported, lvl).statistic), 3),
            "rho_excess": round(float(spearmanr(reported, exc).statistic), 3),
            "mean_dti": round(float(lvl.mean()), 6),
            "probes_beating_uniform": int((exc > 0).sum()),
            "n_probes": len(rows),
        }
    report = {
        "purpose": "choose the promotion frame by evidence, not by preference",
        "caveat": "reported scores are owner-reported, not organizer receipts (IR-44-08); n=%d. "
                  "S0 is scored with known=None by necessity: with the official mask its truth set is empty." % len(rows),
        "frames": {k: int(v.sum()) for k, v in frames.items()},
        "verdict": verdict, "rows": rows,
    }
    # Direct A/B test: the two artifacts whose live scores are known for the same dot family
    # (the 0.2600 anchor and the 0.2778 artifact, which is that anchor minus the <=200 m ring).
    # A frame that ranks the pair backwards cannot be used to select anything.
    pair = {}
    for fname, truth in frames.items():
        kn = None if fname.startswith("S0") else g.known
        a = dti_of(binary_credit(_dots(ANCHOR), truth, valid=g.footprint, known=kn))
        b = dti_of(binary_credit(_dots(SHIPPED_02778), truth, valid=g.footprint, known=kn))
        pair[fname] = {"anchor_0.2600": round(a, 6), "artifact_0.2778": round(b, 6),
                       "delta_pruned_minus_anchor": round(b - a, 6), "ranks_pair_correctly": bool(b > a)}
    report["direct_ab_pair_test"] = pair
    for fname, rec in pair.items():
        verdict[fname]["ranks_known_pair_correctly"] = rec["ranks_pair_correctly"]

    ranked = sorted(verdict.items(), key=lambda kv: -kv[1]["rho_excess"])
    report["best_frame_by_rho_excess"] = ranked[0][0]
    report["positive_rho_frames"] = [k for k, v in verdict.items() if v["rho_excess"] > 0.4]
    Path("registry").mkdir(exist_ok=True)
    Path("registry/frame_ranking.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps(verdict, indent=1))
    print("best by excess:", ranked[0][0], "| frames with rho_excess > 0.4:", report["positive_rho_frames"])
    print("direct A/B (0.2778 minus 0.2600 anchor, must be positive for an admissible frame):")
    for fname, rec in pair.items():
        print(f"   {fname:26s} delta={rec['delta_pruned_minus_anchor']:+.6f} "
              f"{'OK' if rec['ranks_pair_correctly'] else 'BACKWARDS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
