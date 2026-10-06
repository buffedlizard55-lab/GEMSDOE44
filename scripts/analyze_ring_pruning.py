#!/usr/bin/env python3
"""First-hand, byte-level test of the 'prune the 100-200 m catalogue ring' mechanism.

The claim under test (asserted in a sibling session's analysis and in the GEMSDOE32 study
corpus, not by the organizers):

    "0.2708 -> 0.2778 came from deleting 2,545 dots, all of them in the ring
     100 m < d(catalogue) <= 200 m; pruning them cost almost no TP_w but removed
     0.2 * 2,545 = 509 units of false-positive mass."

What this script can verify from files that exist locally:
  A. the exact dot count and the catalogue-distance profile of the 0.2778 artifact itself;
  B. whether any pair of locally available artifacts is related by pure ring pruning
     (subset relation, everything removed inside the ring, nothing added);
  C. the *family-level* generality of the rule: for the 19 artifacts with reported live
     scores, the rank correlation between 'fraction of dots inside the ring' and the score.

It cannot verify the 0.2708 -> 0.2778 step directly, because the 0.2708 artifact (H27-4) is
not present in this sandbox; that limitation is recorded in the output.

Writes registry/ring_profile.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as ndimage
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from gems44 import grid as G  # noqa: E402

PRIOR = Path("/home/user/prior/GEMSDOE32/docs/downloads")
BANDS = [(0.0, 0.0, "on the known catalogue (d = 0)"),
         (0.0, 1.0, "0 < d <= 100 m"),
         (1.0, 2.0, "100 m < d <= 200 m  <- the ring the claim is about"),
         (2.0, 3.0, "200 m < d <= 300 m"),
         (3.0, np.inf, "beyond 300 m")]


def dot_file(p: Path) -> np.ndarray | None:
    with rasterio.open(p) as src:
        v = src.read(1)
    d = np.isfinite(v) & (v > 0)
    return d if d.any() else None


def band_counts(dots: np.ndarray, d_cat: np.ndarray) -> dict:
    n = int(dots.sum())
    out = {}
    for lo, hi, label in BANDS:
        m = dots & (d_cat > lo) & (d_cat <= hi) if lo > 0 or hi < np.inf else dots & (d_cat == 0)
        if lo == 0.0 and hi == 0.0:
            m = dots & (d_cat <= 0.0)
        out[label] = {"dots": int(m.sum()), "frac": round(int(m.sum()) / max(n, 1), 5)}
    out["total_dots"] = n
    return out


def main() -> int:
    g = G.load_grid()
    d_cat = ndimage.distance_transform_edt(~g.known)
    res: dict = {"written_utc": "2026-10-06", "bands": [b[2] for b in BANDS],
                 "cannot_verify": "the 0.2708 artifact (H27-4) is not present in this sandbox, so the "
                                  "step 0.2708 -> 0.2778 cannot be reproduced byte for byte here"}

    # A + B: the locally available family artifacts
    wanted = {
        "0.2778_artifact": "gemsdoe32-h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif",
        "0.2778_plus_h33_1": "gemsdoe32-h33-h33-2b2-plus-h33-1-20261004T220000Z-31588dc7-zeros.tif",
        "0.2600_repro_44090": "gemsdoe32-d28-ref02600-repro-44090-20261004T183200Z-426073b6-zeros.tif",
        "0.2600_anchor_probe": "gems32-probe-S1-ANCHOR-identical-to-live-02600.tif",
        "lazygreedy_maxcov_44090": "gems32-lazygreedy-maxcov-44090-supconfined-zeros.tif",
        "h32d_submodular_46090": "gemsdoe32-h32d-submodular-multipysics-46090-20261004T183200Z-4de30601-zeros.tif",
    }
    arts: dict[str, np.ndarray] = {}
    for name, fn in wanted.items():
        p = PRIOR / fn
        if not p.exists():
            res.setdefault("missing_files", []).append(fn)
            continue
        d = dot_file(p)
        if d is None:
            continue
        arts[name] = d
        res.setdefault("artifacts", {})[name] = {**band_counts(d, d_cat), "file": fn}

    # pairwise containment among the locally available artifacts
    pairs = {}
    names = sorted(arts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = arts[a], arts[b]
            inter = int((A & B).sum())
            only_a, only_b = int((A & ~B).sum()), int((B & ~A).sum())
            if inter == 0:
                continue
            rec = {"shared": inter, "only_A": only_a, "only_B": only_b,
                   "jaccard": round(inter / max(int((A | B).sum()), 1), 4)}
            if only_a == 0 or only_b == 0:
                small, big = (a, b) if only_a == 0 else (b, a)
                removed = arts[big] & ~arts[small]
                rec["subset_relation"] = f"{small} is a subset of {big}"
                rec["removed_band_profile"] = band_counts(removed, d_cat)
            pairs[f"{a} vs {b}"] = rec
    res["pairwise"] = pairs

    # C: family-level test on the 19 artifacts with reported live scores
    man = json.loads(Path("registry/probe_manifest.json").read_text())
    probes = man["probes"] if isinstance(man, dict) and "probes" in man else man
    rows = []
    for p in probes:
        fn = p.get("path") or p.get("file")
        score = p.get("reported_score")
        if not fn or score is None:
            continue
        path = Path(fn)
        if not path.exists():
            continue
        d = dot_file(path)
        if d is None:
            continue
        bc = band_counts(d, d_cat)
        rows.append({"probe": path.name[:44], "score": float(score),
                     "ring_frac": bc[BANDS[2][2]]["frac"], "inner_frac": bc[BANDS[1][2]]["frac"],
                     "up_to_300m_frac": round(bc[BANDS[1][2]]["frac"] + bc[BANDS[2][2]]["frac"]
                                              + bc[BANDS[3][2]]["frac"] + bc[BANDS[0][2]]["frac"], 5),
                     "beyond_300m_frac": bc[BANDS[4][2]]["frac"]})
    if len(rows) >= 5:
        sc = np.array([r["score"] for r in rows])
        res["family_level"] = {
            "n": len(rows),
            "rho_ring_frac_vs_score": round(float(spearmanr([r["ring_frac"] for r in rows], sc).statistic), 4),
            "rho_inner_frac_vs_score": round(float(spearmanr([r["inner_frac"] for r in rows], sc).statistic), 4),
            "rho_beyond300m_frac_vs_score": round(float(spearmanr([r["beyond_300m_frac"] for r in rows], sc).statistic), 4),
            "rows": sorted(rows, key=lambda r: -r["score"]),
        }

    Path("registry/ring_profile.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairwise", "family_level", "artifacts")}, indent=1))
    for name, rec in (res.get("artifacts") or {}).items():
        print(f"{name:26s} dots={rec['total_dots']:>6d} "
              + " ".join(f"{v['frac']:.3f}" for k, v in rec.items()
                         if k != 'total_dots' and isinstance(v, dict)))
    for k, v in (res.get("pairwise") or {}).items():
        print(k, "->", {kk: vv for kk, vv in v.items() if kk != "removed_band_profile"})
    fl = res.get("family_level")
    if fl:
        print(f"family-level (n={fl['n']}): rho(ring frac, score)={fl['rho_ring_frac_vs_score']} "
              f"rho(beyond-300m frac, score)={fl['rho_beyond300m_frac_vs_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
