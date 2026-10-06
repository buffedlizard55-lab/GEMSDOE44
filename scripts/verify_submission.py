#!/usr/bin/env python3
"""Verification pass 1 and 2 for the shipped artifact.

Pass A  contract, read back from the shipped bytes (not from in-memory state)
Pass B  the metric re-derived by an independent route (union-of-shifts, no FFT, no np.roll
        wraparound) plus hand-checked toy cases and masking-semantics tests
Pass C  uniqueness against the 19 prior GEMSDOE artifacts in data/probes/ and the incumbent

Writes registry/verification.json and exits non-zero if anything fails.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as ndimage

sys.path.insert(0, "src")
from gems44 import grid as G          # noqa: E402
from gems44 import metric as M        # noqa: E402
from gems44 import submit as S        # noqa: E402

FAIL: list[str] = []


def check(name: str, ok: bool, detail) -> dict:
    if not ok:
        FAIL.append(name)
    return {"ok": bool(ok), "detail": detail}


def independent_credit(source: np.ndarray, target_shape: tuple[int, int], radius_px: float) -> np.ndarray:
    """max over kernel offsets of source[x + offset] * k(offset), with padded (never wrapped) shifts.

    This is a second, independent implementation of the two max terms in the official metric:
    it never uses np.roll, an Euclidean distance transform, or any closed form.
    """
    h, w = target_shape
    r = int(np.ceil(radius_px))
    pad = np.zeros((h + 2 * r, w + 2 * r), dtype=np.float64)
    pad[r:r + h, r:r + w] = source.astype(np.float64)
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            wgt = max(1.0 - float(np.hypot(dy, dx)) / radius_px, 0.0)
            if wgt <= 0.0:
                continue
            shifted = pad[r - dy:r - dy + h, r - dx:r - dx + w]
            np.maximum(out, shifted * wgt, out=out)
    return out


def independent_dti(dots: np.ndarray, truth: np.ndarray, valid: np.ndarray, known: np.ndarray) -> float:
    """DTI written straight from the three official sums, using padded shifts only."""
    truth = truth & valid & ~known
    pred = (dots & valid & ~known).astype(np.float64)
    g = float(truth.sum())
    tp_w = float(independent_credit(pred, truth.shape, M.RADIUS_PX)[truth].sum())
    fp_w = float((1.0 - independent_credit(truth.astype(np.float64), truth.shape, M.RADIUS_PX))[pred > 0].sum())
    fn_w = g - tp_w
    return tp_w / (0.2 * (tp_w + fp_w) + 0.8 * g + M.EPS)


def main() -> int:
    out: dict = {"pass_a_contract": {}, "pass_b_metric": {}, "pass_c_uniqueness": {}}
    sub = json.loads(Path("registry/submission.json").read_text())
    slug = sub["slug"]
    primary = Path("submissions") / f"{slug}-zeros.tif"
    twin = Path("submissions") / f"{slug}-nan.tif"
    zpath = Path("submissions") / f"{slug}-zeros.zip"

    # ---------------- Pass A: contract, read back from disk ----------------
    g = G.load_grid()
    with rasterio.open("data/raw/sample_submission.tif") as s:
        ref = {"crs": s.crs.to_string(), "transform": tuple(s.transform)[:6],
               "shape": (s.height, s.width), "dtype": s.dtypes[0], "count": s.count}

    files = {}
    for label, p in (("primary", primary), ("nan_twin", twin)):
        with rasterio.open(p) as src:
            v = src.read(1)
            files[label] = {
                "count": src.count, "dtype": src.dtypes[0], "crs": src.crs.to_string(),
                "transform": tuple(src.transform)[:6], "shape": (src.height, src.width),
                "nodata": src.nodata,
                "finite_min": float(np.nanmin(v)), "finite_max": float(np.nanmax(v)),
                "n_finite": int(np.isfinite(v).sum()), "n_nan": int(np.isnan(v).sum()),
                "n_gt_1": int((np.nan_to_num(v, nan=-1.0) > 1.0).sum()),
                "n_lt_0": int((np.nan_to_num(v, nan=1.0) < 0.0).sum()),
                "n_nan_inside_footprint": int((np.isnan(v) & g.footprint).sum()),
                "unique_values": sorted(np.unique(np.nan_to_num(v, nan=-1.0)).tolist())[:5],
            }
    # NaN outside the footprint is the twin's deliberate difference; everything else must match.
    for label in files:
        f = files[label]
        out["pass_a_contract"][f"{label}_contract"] = check(
            f"{label} contract",
            f["count"] == 1 and f["dtype"] == "float32" and f["crs"] == ref["crs"]
            and f["transform"] == ref["transform"] and f["shape"] == ref["shape"]
            and f["n_lt_0"] == 0 and f["n_gt_1"] == 0
            and f["n_nan_inside_footprint"] == 0,
            {"errors_vs_reference": {k: f[k] for k in ("count", "dtype", "crs", "transform", "shape")},
             "reference": ref, "range": [f["finite_min"], f["finite_max"]],
             "outside_range": [f["n_lt_0"], f["n_gt_1"]],
             "nan_inside_footprint": f["n_nan_inside_footprint"]},
        )
    pv = files["primary"]
    out["pass_a_contract"]["primary_is_binary_and_mass_matched"] = check(
        "primary binary mass match",
        pv["unique_values"] == [0.0, 1.0] and int(np.isclose(pv["finite_max"], 1.0)) == 1,
        {"unique_values": pv["unique_values"], "requested_mass": sub["mass"]},
    )
    with rasterio.open(primary) as src:
        dots = src.read(1) > 0.5
    out["pass_a_contract"]["dots_equal_mass"] = check(
        "dots == mass", int(dots.sum()) == int(sub["mass"]), {"dots": int(dots.sum()), "mass": sub["mass"]})
    out["pass_a_contract"]["no_dots_on_known_mask"] = check(
        "no dots on known mask", int((dots & g.known).sum()) == 0, {"dots_on_known": int((dots & g.known).sum())})
    out["pass_a_contract"]["verified_mass_report_field"] = check(
        "audit report agrees with the file",
        int(sub["emission_diagnostics"]["dots"]) == int(dots.sum()),
        {"report": sub["emission_diagnostics"]["dots"], "file": int(dots.sum())})

    # the zip must contain exactly the loose file, byte for byte
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        inner = z.read(names[0]) if len(names) == 1 else b""
    loose = primary.read_bytes()
    out["pass_a_contract"]["zip_bytes_match"] = check(
        "zip member == loose TIF",
        names == [primary.name] and inner == loose,
        {"members": names, "zip_bytes": len(inner), "loose_bytes": len(loose),
         "identical": inner == loose})
    out["pass_a_contract"]["published_copies_match"] = check(
        "docs/downloads copies == submissions copies",
        all((Path("docs/downloads") / p.name).exists()
            and (Path("docs/downloads") / p.name).read_bytes() == p.read_bytes()
            for p in (primary, twin, zpath)),
        {"published": [p.name for p in sorted(Path("docs/downloads").glob("GEMS44_*"))]})

    # ---------------- Pass B: independent metric ----------------
    with rasterio.open("data/raw/labels.tif") as src:
        lab = src.read(1)
    sgmc = G.read_band("data/external/derived_sgmc_faults_100m_u8.tif", 1) > 0
    d_cat = ndimage.distance_transform_edt(~g.known)
    frames = {"N": sgmc & (d_cat <= 3.0), "P": sgmc & (d_cat > 3.0)}
    for k, truth in frames.items():
        fast = M.dti(dots, truth, valid=g.footprint, known=g.known).score
        brute = M.dti_bruteforce(dots, truth, valid=g.footprint, known=g.known).score
        indep = independent_dti(dots, truth, g.footprint, g.known)
        out["pass_b_metric"][f"frame_{k}"] = check(
            f"frame {k}: exact == brute == independent",
            abs(fast - brute) < 1e-12 and abs(fast - indep) < 1e-12,
            {"exact": fast, "bruteforce": brute, "independent_union_of_shifts": indep,
             "max_abs_diff": max(abs(fast - brute), abs(fast - indep))})

    # toy cases, hand-computed
    t = np.zeros((9, 9), bool)
    t[2:5, 2:5] = True                      # 9-pixel block; centre pixel 2 px from the edge
    p = np.zeros((9, 9), bool)
    p[3, 3] = True                          # one dot at the block centre
    valid = np.ones((9, 9), bool)
    known = np.zeros((9, 9), bool)
    toy_dti = M.dti(p, t, valid=valid, known=known).score
    # hand computation: the dot sits at the centre of a 3x3 block, so the kernel is graded —
    # centre 1.0, edge 1 - 1/3, corner 1 - sqrt(2)/3.  FP_w = 0 (the dot's nearest truth is
    # itself, max_g k = 1), FN_w = G - TP_w.
    toy_tp = 1.0 + 4.0 * (1.0 - 1.0 / 3.0) + 4.0 * (1.0 - np.sqrt(2.0) / 3.0)
    toy_expected = toy_tp / (0.2 * toy_tp + 0.8 * 9.0)
    out["pass_b_metric"]["toy_single_dot"] = check(
        "toy: single dot inside a 3x3 truth block", abs(toy_dti - toy_expected) < 1e-12,
        {"got": toy_dti, "expected": toy_expected, "TP_w_hand": toy_tp})
    # the triangular kernel vanishes AT d = R: k(d) = (1 - d/R)+, so a dot 3 px away earns
    # exactly 0 while a dot 2 px away earns k = 1/3.  This is the boundary the live score
    # profile is sensitive to, so it is asserted explicitly.
    t2 = np.zeros((9, 9), bool)
    t2[4, 4] = True
    p2 = np.zeros((9, 9), bool)
    p2[4, 2] = True                          # dx = 2 -> k = 1/3
    d2 = M.dti(p2, t2, valid=valid, known=known).score
    p3 = np.zeros((9, 9), bool)
    p3[4, 1] = True                          # dx = 3 -> k = 0 exactly
    d3 = M.dti(p3, t2, valid=valid, known=known).score
    d2_expected = (1.0 / 3.0) / (0.2 * (1.0 / 3.0 + 2.0 / 3.0) + 0.8 * 1.0)
    out["pass_b_metric"]["kernel_vanishes_at_300m"] = check(
        "k(2 px) = 1/3 credits, k(3 px) = 0 does not",
        abs(d2 - d2_expected) < 1e-12 and d3 == 0.0,
        {"dx2_dti": d2, "dx2_expected": d2_expected, "dx3_dti": d3})
    # masking: a dot painted on the known mask must not enter prediction (it is excluded),
    # and truth covered by the known mask must not enter the denominator
    p4 = np.zeros((9, 9), bool)
    p4[0, 0] = True
    known4 = np.zeros((9, 9), bool)
    known4[0, 0] = True
    t4 = np.zeros((9, 9), bool)
    t4[8, 8] = True
    masked_dti = M.dti(p4, t4, valid=valid, known=known4).score
    out["pass_b_metric"]["known_mask_excluded"] = check(
        "dot on the known mask contributes nothing", masked_dti == 0.0, {"dti": masked_dti})
    # empty prediction and perfect prediction
    out["pass_b_metric"]["degenerate_cases"] = check(
        "empty prediction -> 0.0; perfect prediction -> 1.0",
        M.dti(np.zeros((9, 9), bool), t, valid=valid, known=known).score == 0.0
        and abs(M.dti(t.copy(), t, valid=valid, known=known).score - 1.0) < 1e-12,
        {"empty": M.dti(np.zeros((9, 9), bool), t, valid=valid, known=known).score,
         "perfect": M.dti(t.copy(), t, valid=valid, known=known).score})
    # a truth pixel at the very corner must not be credited by a wrapped dot from the far edge
    t5 = np.zeros((9, 9), bool)
    t5[0, 0] = True
    p5 = np.zeros((9, 9), bool)
    p5[8, 8] = True
    out["pass_b_metric"]["no_wraparound"] = check(
        "corner truth is not credited by a far-corner dot",
        M.dti(p5, t5, valid=valid, known=known).score == 0.0,
        {"dti": M.dti(p5, t5, valid=valid, known=known).score})

    # ---------------- Pass C: uniqueness ----------------
    inc = G.read_band("data/raw/incumbent_d28.tif", 1) > 0
    def jac(a, b):
        return float((a & b).sum()) / max(float((a | b).sum()), 1.0)
    prior = {}
    for p in sorted(Path("data/probes").glob("*.tif")):
        v = G.read_band(str(p), 1)
        d = np.isfinite(v) & (v > 0)
        if d.any():
            prior[p.stem] = {"dots": int(d.sum()), "jaccard_vs_ours": round(jac(dots, d), 4),
                                 "ours_inside_theirs": round(float((dots & d).sum()) / int(dots.sum()), 4)}
    out["pass_c_uniqueness"]["vs_incumbent"] = check(
        "not the incumbent", jac(dots, inc) < 0.10,
        {"jaccard": round(jac(dots, inc), 4), "shared": int((dots & inc).sum()),
         "incumbent_dots": int(inc.sum())})
    worst = max(prior.items(), key=lambda kv: kv[1]["jaccard_vs_ours"]) if prior else None
    out["pass_c_uniqueness"]["vs_prior_gemsdoe_artifacts"] = check(
        "distinct from all 19 prior artifacts", bool(prior) and worst[1]["jaccard_vs_ours"] < 0.10,
        {"n_compared": len(prior), "max_jaccard": worst[1]["jaccard_vs_ours"] if worst else None,
         "max_jaccard_artifact": worst[0] if worst else None, "per_artifact": prior})

    Path("registry/verification.json").write_text(
        json.dumps({"artifact": slug, **out, "n_failures": len(FAIL), "failures": FAIL}, indent=1,
                   sort_keys=True) + "\n")
    for block, items in out.items():
        for name, r in items.items():
            print(f"{'PASS' if r['ok'] else 'FAIL':4s} {block} :: {name}")
    print(f"\n{len(FAIL)} failures" + (f": {FAIL}" if FAIL else " — all checks green"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
