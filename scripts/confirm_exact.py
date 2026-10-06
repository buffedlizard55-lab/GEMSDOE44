"""Confirm the shipped artifact with the EXACT 29-offset metric operator (not the fast path).

The selection run scores with ``gems44.metric.binary_credit``, which computes the same sufficient
statistics as the official definition from two Euclidean distance transforms.  This script
re-derives the same numbers with ``gems44.metric.dti`` - the literal transcription of the published
formulas, a 29-offset max over the triangular kernel - for

  * the shipped artifact,
  * the 0.2600 incumbent control,
  * a uniform control of the same mass,

on both frames.  Agreement between the two operators to < 1e-9 is the last check before upload.

Reads registry/submission.json (written by scripts/build_submission.py) or the newest
submissions/*-audit.json; writes registry/exact_confirmation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import grid as G  # noqa: E402
from gems44.metric import binary_credit, dti, dti_of  # noqa: E402


def newest_audit() -> dict | None:
    subs = sorted(Path("submissions").glob("*-audit.json"))
    if not subs:
        return None
    return json.loads(subs[-1].read_text())


def dots_from(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
    return np.isfinite(a) & (a > 0.5)


def main() -> int:
    audit = newest_audit()
    if audit is None:
        print("no submissions/*-audit.json yet")
        return 1
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known
    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat)
    frames = {"N_sgmc_within_300m": sgmc & (d_cat <= 3.0), "P_sgmc_beyond_300m": sgmc & (d_cat > 3.0)}

    dots = dots_from(audit["primary"]["path"])
    with rasterio.open("data/raw/incumbent_d28.tif") as src:
        v = src.read(1)
    inc = np.isfinite(v) & (v > 0)
    rng = np.random.default_rng(44)
    allowed = np.nonzero((g.footprint & ~g.known).ravel())[0]
    idx = rng.choice(allowed, size=int(dots.sum()), replace=False)
    uni = np.zeros(g.footprint.shape, bool)
    uni.ravel()[idx] = True

    report = {"artifact": audit["slug"], "mass": int(dots.sum()),
              "operator_note": "exact = gems44.metric.dti (25 positive-weight kernel offsets, R=3px); fast = binary_credit (closed form on two Euclidean distance transforms)",
              "frames": {}, "agreement_all_within_1e-9": True}
    for fname, truth in frames.items():
        row = {"truth_px": int(truth.sum())}
        for label, d in (("artifact", dots), ("incumbent", inc), ("uniform_same_mass", uni)):
            exact = dti(d, truth, valid=g.footprint, known=g.known).score
            fast = dti_of(binary_credit(d, truth, valid=g.footprint, known=g.known))
            row[label] = {"exact": round(float(exact), 9), "fast": round(float(fast), 9),
                          "abs_diff": abs(float(exact) - float(fast))}
            report["agreement_all_within_1e-9"] &= bool(abs(exact - fast) < 1e-9)
        row["delta_artifact_minus_incumbent_exact"] = round(
            row["artifact"]["exact"] - row["incumbent"]["exact"], 6)
        report["frames"][fname] = row
        print(f"{fname:24s} artifact={row['artifact']['exact']:.6f} "
              f"incumbent={row['incumbent']['exact']:.6f} uniform={row['uniform_same_mass']['exact']:.6f} "
              f"delta={row['delta_artifact_minus_incumbent_exact']:+.6f}")

    Path("registry").mkdir(exist_ok=True)
    Path("registry/exact_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("operators agree within 1e-9:", report["agreement_all_within_1e-9"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
