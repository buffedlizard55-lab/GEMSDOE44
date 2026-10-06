"""Can any LOCAL frame rank artifacts the way the live board does?  A falsification test.

The whole project reduces to the answer.  We have four artifacts from the wider family whose
score was REPORTED (owner-reported, not organizer-authenticated - IR-44-08) and whose bytes we
hold, spanning 0.1922 -> 0.2600.  If a local frame cannot rank those four in the same order as
the reported scores, that frame is not a usable promotion frame, and no hypothesis may be
selected on it.  Two frames are pitted against each other:

  S  the catalogue (60,988 px)                       - already known to be INVERTED vs the board
  P  SGMC >300 m from the catalogue (62,122 px)      - the frame this session proposed
  N  SGMC <=300 m from the catalogue (21,471 px)     - a third, deliberately different frame

Scoring is the exact official operator (src/gems44/metric.py).  A uniform control at the same
mass is scored on every frame, and the rank correlation against the reported scores is printed.

Writes registry/frame_validity.json.
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
from gems44.metric import binary_credit, dti, dti_of  # noqa: E402

PROBES = [
    ("gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif", 0.1922,
     "19GEMSDOE", "owner-reported; not authenticated"),
    ("gems27-topo-gap-closure-t-v2-on-d1-5-20261002-5512495c6bd1-nan.tif", 0.2449,
     "GEMSDOE27", "owner-reported; not authenticated"),
    ("gems24-h25-1-dotted-h19-5-d1-5-20261002-989f59505db1-nan.tif", 0.2477,
     "GEMSDOE24", "owner-reported; not authenticated"),
    ("gems24-h25-1-dotted-h19-5-d2-8-20261002-e56ea318af89-nan.tif", 0.2600,
     "GEMSDOE25/GEMSDOE24", "owner-reported; owner page conflicted (IR-44-03 class)"),
]


def read_probe(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
    a[~np.isfinite(a)] = 0.0
    return a


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known
    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat)

    frames = {
        "S_catalogue": cat,
        "P_sgmc_far": sgmc & (d_cat > 3.0),
        "N_sgmc_near": sgmc & (d_cat <= 3.0),
    }
    report: dict = {
        "purpose": "falsification test of local promotion frames against owner-reported live scores",
        "caveat": "the live scores are owner-reported and not organizer-authenticated (IR-44-08); "
                  "the test asks only for the ORDER, which is the property a promotion frame needs",
        "frames": {k: int(v.sum()) for k, v in frames.items()},
        "rows": [],
    }

    rng = np.random.default_rng(44)
    allowed = g.footprint & ~g.known
    for name, live, site, note in PROBES:
        p = read_probe(Path("data/probes") / name)
        pos = p > 0
        vals = np.unique(p[pos])
        binary = bool(np.array_equal(vals, np.array([1.0]))) if vals.size == 1 else bool(vals.size == 2 and vals[0] == 0.0 and vals[1] == 1.0)
        row = {"artifact": name, "site": site, "live_score_reported": live, "note": note,
               "dots_or_mass": int(pos.sum()), "binary": binary}
        for fname, truth in frames.items():
            if binary:
                st = binary_credit(pos, truth, valid=g.footprint, known=g.known)
                sc = dti_of(st)
            else:
                sc = dti(p, truth, valid=g.footprint, known=g.known).score
            row[f"dti_{fname}"] = round(float(sc), 6)
        idx = rng.choice(np.nonzero(allowed.ravel())[0], size=min(int(pos.sum()), int(allowed.sum())), replace=False)
        u = np.zeros(g.footprint.shape, bool)
        u.ravel()[idx] = True
        for fname, truth in frames.items():
            st = binary_credit(u, truth, valid=g.footprint, known=g.known)
            row[f"uniform_{fname}"] = round(dti_of(st), 6)
        report["rows"].append(row)
        print(f"{name[:52]:54s} live={live:.4f} n={row['dots_or_mass']:>6,} " +
              " ".join(f"{k.split('_',1)[1]}={row['dti_'+k]:.4f}" for k in frames))

    live = np.array([r["live_score_reported"] for r in report["rows"]])
    verdict = {}
    for fname in frames:
        loc = np.array([r[f"dti_{fname}"] for r in report["rows"]])
        rho = float(spearmanr(live, loc).statistic)
        verdict[fname] = {
            "spearman_vs_live": round(rho, 3),
            "same_order": bool(rho == 1.0),
            "beats_its_own_uniform_control_on_all": bool(
                all(r[f"dti_{fname}"] > r[f"uniform_{fname}"] for r in report["rows"])),
            "spread": round(float(loc.max() - loc.min()), 6),
        }
    report["verdict"] = verdict
    n_ok = sum(1 for v in verdict.values() if v["same_order"])
    report["conclusion"] = (
        f"{n_ok} of {len(verdict)} frames reproduce the reported score order. "
        + ("At least one frame is usable for promotion, subject to its own weaknesses."
           if n_ok else
           "NO local frame reproduces the reported order: no hypothesis may be promoted on a local "
           "frame in this session. Selection must fall back to mechanism-based reasoning plus "
           "explicitly declared risk (registered as an irregularity).")
    )
    Path("registry").mkdir(exist_ok=True)
    Path("registry/frame_validity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps({"verdict": verdict, "conclusion": report["conclusion"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
