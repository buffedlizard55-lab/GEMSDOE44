"""Measure the fault-population statistics that a placement prior needs.

Writes ``registry/population.json``.  Everything here is computed from hash-verified
official bytes (``data/raw/labels.tif``) and the SGMC external raster
(``data/external/derived_sgmc_faults_100m_u8.tif``).

The decisive measurement is `offset_geometry`: for real mapped faults that are NOT in the
competition catalogue (SGMC-only pixels, our closest available analogue of the hidden
truth), is the offset vector to the nearest catalogue pixel preferentially ACROSS strike
rather than uniformly distributed in direction?  If yes, an anisotropic, strike-frame
placement prior is justified by data.  If no, the isotropic proximity prior everyone has
used cannot be beaten this way and the hypothesis must be dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gems44 import grid as G
from gems44 import population as P

OUT = Path("registry/population.json")


def main() -> int:
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    print("grid:", json.dumps(g.summary()))

    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc = sgmc & g.footprint & ~g.known
    print(f"SGMC pixels inside footprint and off the catalogue: {int(sgmc.sum()):,}")

    segments = P.segment_statistics(labels, g.footprint)
    az = P.azimuth_histogram(segments)
    print(f"segments (>=12 px): {len(segments)}  modal azimuth bin: {az['modal_bin_deg']} deg")

    # --- offset geometry against the catalogue, SGMC-off-catalogue as stand-in truth
    geom_sgmc = P.nearest_fault_distance_geometry(labels, g.footprint, sgmc, exclude_label=1)
    delta_sgmc = geom_sgmc.pop("delta_deg")

    # --- random control: footprint pixels at a matched distance from the catalogue
    dist_to_cat = np.zeros_like(labels, dtype=float)
    from scipy import ndimage
    d = ndimage.distance_transform_edt(~(g.footprint & (labels > 0)))
    dist_to_cat[...] = d
    rng = np.random.default_rng(44)
    cand = np.nonzero(
        g.footprint & ~g.known & ~sgmc & (dist_to_cat >= 1.5) & (dist_to_cat <= 9.0)
    )
    m = cand[0].size
    take = rng.choice(m, size=min(6000, m), replace=False)
    control = np.zeros_like(labels, dtype=bool)
    control[cand[0][take], cand[1][take]] = True

    # matched distance window on the SGMC side
    ys, xs = np.nonzero(sgmc & (dist_to_cat >= 1.5) & (dist_to_cat <= 9.0))
    sgmc_matched = np.zeros_like(labels, dtype=bool)
    if ys.size > 6000:
        take = rng.choice(ys.size, size=6000, replace=False)
        ys, xs = ys[take], xs[take]
    sgmc_matched[ys, xs] = True

    geom_sgmc_m = P.nearest_fault_distance_geometry(labels, g.footprint, sgmc_matched, 1)
    delta_sgmc_m = geom_sgmc_m.pop("delta_deg")
    geom_ctrl = P.nearest_fault_distance_geometry(labels, g.footprint, control, 0)
    delta_ctrl = geom_ctrl.pop("delta_deg")

    profile_sgmc = P.binned_angle_profile(delta_sgmc_m)
    profile_ctrl = P.binned_angle_profile(delta_ctrl)

    payload = {
        "source": {
            "labels": "data/raw/labels.tif sha256 7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
            "sgmc": "data/external/derived_sgmc_faults_100m_u8.tif",
            "note": "measured, not assumed; regenerate with scripts/measure_population.py",
        },
        "grid": g.summary(),
        "sgmc": {
            "off_catalogue_px": int(sgmc.sum()),
            "off_catalogue_km": round(float(sgmc.sum()) * 0.1, 1),
        },
        "segments": {
            "n": len(segments),
            "median_length_km": float(np.median([s["length_km_2sigma"] for s in segments])),
            "p90_length_km": float(np.percentile([s["length_km_2sigma"] for s in segments], 90)),
            "median_elongation": float(np.median([s["elongation"] for s in segments])),
            "sample": sorted(segments, key=lambda s: -s["px"])[:12],
        },
        "azimuth": az,
        "offset_geometry_all_sgmc": geom_sgmc,
        "offset_geometry_sgmc_matched": geom_sgmc_m,
        "offset_geometry_random_control_matched": geom_ctrl,
        "angle_profile_sgmc_matched": profile_sgmc,
        "angle_profile_random_control_matched": profile_ctrl,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")

    print("\n--- offset-vector angle to local catalogue strike (0=along, 90=across) ---")
    for name, prof in (("SGMC-only (stand-in hidden)", profile_sgmc), ("random control", profile_ctrl)):
        print(f"{name:32s} enrichment per 15 deg bin: "
              + " ".join(f"{e:5.2f}" for e in prof.get("enrichment", [])))
    print("frac within 30 deg of STRIKE : SGMC %.3f | control %.3f"
          % (geom_sgmc_m["frac_within_30deg_of_strike"], geom_ctrl["frac_within_30deg_of_strike"]))
    print("frac within 30 deg of ACROSS : SGMC %.3f | control %.3f"
          % (geom_sgmc_m["frac_within_30deg_of_perp"], geom_ctrl["frac_within_30deg_of_perp"]))
    print("median angle to strike       : SGMC %.1f | control %.1f"
          % (geom_sgmc_m["delta_median_deg"], geom_ctrl["delta_median_deg"]))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
