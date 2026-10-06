"""Final submission writer + independent byte-level format audit.

Guarantees (re-verified by re-opening the written file from disk):
  * single band, float32, 3730 x 3292, EPSG:32611, 100 m, official geotransform
  * every cell finite; every value in [0, 1]; zero NaN; NO nodata tag
    (the portal's "Predicted values must be in range [0, 1]" rejects both
    out-of-range values AND out-of-range nodata sentinels)
  * matches the official example_submission.tif CRS/shape/geotransform
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from . import config as C


def write_submission(rows: np.ndarray, cols: np.ndarray, value: float,
                     path: str | Path) -> dict:
    out = np.zeros((C.ROWS, C.COLS), dtype=np.float32)
    out[rows, cols] = value
    transform = from_origin(C.ORIGIN_E, C.ORIGIN_N, C.CELL_M, C.CELL_M)
    prof = {
        "driver": "GTiff",
        "height": C.ROWS,
        "width": C.COLS,
        "count": 1,
        "dtype": "float32",
        "crs": C.CRS,
        "transform": transform,
        # deliberately no nodata
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(out, 1)
    return audit_file(path)


def audit_file(path: str | Path) -> dict:
    """Re-open from disk and verify every format guarantee."""
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        meta = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": ds.count,
            "dtype": str(ds.dtypes[0]),
            "shape": [ds.height, ds.width],
            "crs": str(ds.crs),
            "transform": list(ds.transform),
            "nodata": ds.nodata,
        }
        finite = np.isfinite(arr)
        checks = {
            "single_band": ds.count == 1,
            "dtype_float32": str(ds.dtypes[0]) == "float32",
            "dimensions_3730x3292": [ds.height, ds.width] == [C.ROWS, C.COLS],
            "crs_epsg_32611": "32611" in str(ds.crs),
            "official_transform": (
                abs(ds.transform.a - 100.0) < 1e-6
                and abs(ds.transform.e + 100.0) < 1e-6
                and abs(ds.transform.c - C.ORIGIN_E) < 1e-6
                and abs(ds.transform.f - C.ORIGIN_N) < 1e-6
            ),
            "all_finite": bool(finite.all()),
            "no_nan": int(np.isnan(arr).sum()) == 0,
            "no_inf": int(np.isinf(arr).sum()) == 0,
            "no_nodata_tag": ds.nodata is None,
            "range_0_1_inclusive": bool((arr.min() >= 0.0) and (arr.max() <= 1.0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "emitted_positive_pixels": int((arr > 0).sum()),
        }
        checks["all_checks_passed"] = all(
            v for k, v in checks.items() if isinstance(v, bool))
    return meta | {"checks": checks}


def write_zip(tif_path: str | Path, zip_path: str | Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tif_path, arcname=Path(tif_path).name)


def uniqueness_report(mine_r, mine_c, incumbents: list[tuple[str, str, str]]):
    """Pixel overlap of our dot set with previously-scored incumbent dot sets.

    incumbents: list of (ref_repo, filename, live_score) under REPO.parent/ref.
    Random-overlap expectation for two sets of sizes n1, n2 inside the
    footprint F is n1*n2/F.
    """
    from . import config as Cfg
    F = 5167373  # footprint cells (verified from existing_faults.tif)
    M = np.zeros((Cfg.ROWS, Cfg.COLS), bool)
    M[mine_r, mine_c] = True
    rep = []
    for repo, fname, score in incumbents:
        p = Cfg.REPO.parent / "ref" / repo / "docs" / "downloads" / fname
        if not p.exists():
            rep.append({"file": fname, "status": "missing"})
            continue
        with rasterio.open(p) as ds:
            arr = ds.read(1)
        other = (arr > 0).astype(bool)
        inter = int((M & other).sum())
        n1, n2 = int(M.sum()), int(other.sum())
        expected = n1 * n2 / F
        rep.append({
            "file": fname,
            "live_score": score,
            "n_theirs": n2,
            "overlap_px": inter,
            "overlap_frac_of_mine": inter / max(n1, 1),
            "random_expectation_px": expected,
            "excess_vs_random": inter - expected,
        })
    return rep
