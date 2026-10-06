"""Is the live score explained by WHERE an artifact puts its dots relative to the known catalogue?

This is the non-circular version of the frame question: it uses no external fault compilation,
no metric simulation and no SGMC raster - only the catalogue that the metric masks, and the 19
artifacts whose live scores were reported.  If "more mass in the catalogue's 300 m halo" goes
with a higher live score across those 19, then the hidden truth is catalogue-adjacent and the
dot allocation is the lever.

Reported per artifact: mass, dot counts and fractions within r px of the catalogue for
r in {1, 2, 3, 5, 10, 30}, mean distance to the catalogue, and the own-mask fraction.
Then Spearman correlations against the reported score, and a rank-regression that separates
halo concentration from mass, because mass alone also moves the metric.

Writes registry/halo_analysis.json.
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

RADII = (1, 2, 3, 5, 10, 30)


def rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat)

    manifest = json.loads(Path("registry/probe_manifest.json").read_text())["probes"]
    probes = [(Path(m["file"]).name, m["reported_score"]) for m in manifest if m["status"] != "NOT FOUND"]
    probes.sort(key=lambda t: t[1])

    rows = []
    for name, score in probes:
        with rasterio.open(Path("data/probes") / name) as src:
            dots = src.read(1) > 0.5
        allowed = g.footprint & ~g.known
        n = int(dots.sum())
        row = {"file": name, "reported_score": score, "mass": n,
               "on_known_mask": int((dots & g.known).sum()),
               "mean_dist_to_catalogue_px": round(float(d_cat[dots].mean()), 3) if n else None}
        for r in RADII:
            c = int((dots & (d_cat <= r)).sum())
            row[f"n_within_{r}px"] = c
            row[f"frac_within_{r}px"] = round(c / max(n, 1), 5)
        big = (dots & ~allowed).sum()
        row["outside_scored_domain"] = int(big)
        rows.append(row)
        print(f"{name[:44]:46s} live={score:.4f} n={n:>7,} frac<=3px={row['frac_within_3px']:.3f} "
              f"mean_d={row['mean_dist_to_catalogue_px']}")

    live = np.array([r["reported_score"] for r in rows])
    cors = {}
    for key in ["frac_within_1px", "frac_within_2px", "frac_within_3px", "frac_within_5px",
                "frac_within_10px", "frac_within_30px", "mean_dist_to_catalogue_px", "mass",
                "on_known_mask", "n_within_3px"]:
        v = np.array([r[key] if r[key] is not None else 0.0 for r in rows], dtype=float)
        cors[key] = round(float(spearmanr(live, v).statistic), 3)

    # rank-regression: live_rank ~ a*frac3_rank + b*log_mass_rank
    X = np.c_[np.ones(len(rows)), rank(np.array([r["frac_within_3px"] for r in rows])),
              rank(np.log(np.array([r["mass"] for r in rows], dtype=float)))]
    y = rank(live)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    report = {
        "purpose": "separate halo concentration from mass as explanations of the reported live score",
        "caveat": "n=19 owner-reported scores; correlations are descriptive, not causal proof",
        "spearman_vs_live": cors,
        "rank_regression": {
            "model": "live_rank ~ 1 + frac_within_3px_rank + log_mass_rank",
            "coef_frac3": round(float(coef[1]), 3), "coef_log_mass": round(float(coef[2]), 3),
            "r2": round(float(1.0 - resid.var() / y.var()), 3),
            "spearman_frac3_after_mass_removed": round(float(spearmanr(live, X[:, 1] - coef[2] * X[:, 2]).statistic), 3),
        },
        "rows": rows,
    }
    Path("registry").mkdir(exist_ok=True)
    Path("registry/halo_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps({"spearman_vs_live": cors, "rank_regression": report["rank_regression"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
