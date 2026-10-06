"""Measure the strand structure of the two compilations and the incumbent's dot allocation.

Answers, with numbers rather than assumptions:
  A. does the USGS SGMC raster cover the competition catalogue, or is it disjoint line work?
  B. where does the live-scored 0.2600 incumbent actually spend its 44,090 dots: on the catalogue
     (deleted by the mask), in the catalogue's 300 m halo, on SGMC strands, or on nothing?
  C. how do the incumbent's sufficient statistics decompose per frame?

The answers decide the field design, because the metric pays only for dots inside the 300 m
kernel of scored truth, and dots on the known mask are removed before scoring.

Writes registry/strands.json.
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
from gems44.metric import binary_credit  # noqa: E402

PX = 100.0


def pct(a: np.ndarray, qs=(0, 10, 25, 50, 75, 90, 99, 100)) -> dict:
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs} if a.size else {}


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    cat = g.footprint & (labels > 0)
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    with rasterio.open("data/raw/incumbent_d28.tif") as src:
        inc = src.read(1)
    inc = np.isfinite(inc) & (inc > 0)

    sgmc &= g.footprint
    d_cat = ndimage.distance_transform_edt(~cat)
    d_sgmc = ndimage.distance_transform_edt(~sgmc)

    report: dict = {"pixel_m": PX}

    # ---- A. relation between the two compilations
    report["A_comparison"] = {
        "catalogue_px": int(cat.sum()),
        "sgmc_in_footprint_px": int(sgmc.sum()),
        "sgmc_on_known_mask_px": int((sgmc & g.known).sum()),
        "sgmc_off_known_px": int((sgmc & ~g.known).sum()),
        "catalogue_px_within_1px_of_sgmc": int((cat & (d_sgmc <= 1)).sum()),
        "catalogue_px_within_3px_of_sgmc": int((cat & (d_sgmc <= 3)).sum()),
        "cat_to_sgmc_distance_px": pct(d_sgmc[cat]),
        "sgmc_off_known_to_cat_distance_px_bins": {
            "0-1": int((sgmc & ~g.known & (d_cat <= 1)).sum()),
            "1-3": int((sgmc & ~g.known & (d_cat > 1) & (d_cat <= 3)).sum()),
            "3-10": int((sgmc & ~g.known & (d_cat > 3) & (d_cat <= 10)).sum()),
            "10-30": int((sgmc & ~g.known & (d_cat > 10) & (d_cat <= 30)).sum()),
            ">30": int((sgmc & ~g.known & (d_cat > 30)).sum()),
        },
    }

    # ---- B. incumbent allocation
    n = int(inc.sum())
    b = {
        "on_known_mask_deleted": int((inc & g.known).sum()),
        "within_1px_of_catalogue": int((inc & (d_cat <= 1)).sum()),
        "1_to_3px": int((inc & (d_cat > 1) & (d_cat <= 3)).sum()),
        "3_to_10px": int((inc & (d_cat > 3) & (d_cat <= 10)).sum()),
        "10_to_30px": int((inc & (d_cat > 10) & (d_cat <= 30)).sum()),
        "beyond_30px": int((inc & (d_cat > 30)).sum()),
        "exactly_on_sgmc": int((inc & sgmc).sum()),
        "within_1px_of_sgmc": int((inc & (d_sgmc <= 1)).sum()),
        "within_3px_of_sgmc": int((inc & (d_sgmc <= 3)).sum()),
        "beyond_3px_of_sgmc": int((inc & (d_sgmc > 3)).sum()),
    }
    b["n_dots"] = n
    b["fraction_beyond_3px_of_both"] = round(
        float((inc & (d_sgmc > 3) & (d_cat > 3)).sum() / max(n, 1)), 4)
    report["B_incumbent_allocation"] = b

    # ---- C. sufficient statistics per frame for the incumbent vs uniform at the same mass
    frames = {
        "N_sgmc_within_300m_of_catalogue": sgmc & ~g.known & (d_cat <= 3),
        "P_sgmc_beyond_300m_of_catalogue": sgmc & ~g.known & (d_cat > 3),
    }
    rng = np.random.default_rng(44)
    allowed = g.footprint & ~g.known
    idx = rng.choice(np.nonzero(allowed.ravel())[0], size=n, replace=False)
    u = np.zeros(g.shape if hasattr(g, "shape") else inc.shape, bool)
    u.ravel()[idx] = True
    report["C_per_frame"] = {}
    for name, truth in frames.items():
        si = binary_credit(inc, truth, valid=g.footprint, known=g.known)
        su = binary_credit(u, truth, valid=g.footprint, known=g.known)
        report["C_per_frame"][name] = {
            "truth_px": int(truth.sum()),
            "incumbent": {"T": round(si["T"], 1), "F": round(si["F"], 1), "G": si["G"],
                          "n": si["n"],
                          "DTI": round(float(si["T"] / (0.2 * (si["T"] + si["F"]) + 0.8 * si["G"] + 1e-12)), 6)},
            "uniform_same_mass": {
                "T": round(su["T"], 1), "F": round(su["F"], 1),
                "DTI": round(float(su["T"] / (0.2 * (su["T"] + su["F"]) + 0.8 * su["G"] + 1e-12)), 6)},
        }

    # ---- D. strand geometry of the SGMC residual
    lab, nseg = ndimage.label(sgmc & ~g.known, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes > 0]
    report["D_sgmc_geometry"] = {
        "components": int(nseg),
        "px_ge_4": int((sizes >= 4).sum()),
        "px_ge_12": int((sizes >= 12).sum()),
        "largest_component_px": int(sizes.max()) if sizes.size else 0,
        "median_component_px": float(np.median(sizes)) if sizes.size else 0.0,
    }

    Path("registry").mkdir(exist_ok=True)
    Path("registry/strands.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
