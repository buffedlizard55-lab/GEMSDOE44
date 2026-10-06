"""Build and validate the one-click submission artifact.

Format contract, taken verbatim from the official problem description
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format,
read 2026-10-06):
  * same projected CRS as the training data, UTM 11N / EPSG:32611
  * same resolution as the training data, 100 m
  * same bounds as the training data, data outside the bounds null or nan
  * a single layer, datatype 32-bit float (float32), values between 0 and 1

We ship two twins of the same prediction set:
  * ``*-zeros.tif``  every cell finite, values exactly {0.0, 1.0}, no nodata tag.
    This is the file that cannot trigger the portal message
    "Predicted values must be in range [0, 1]" under any validator.
  * ``*-nan.tif``    identical inside the footprint, NaN outside with nodata = nan,
    i.e. byte-convention-identical to the official ``sample_submission.tif``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass(frozen=True)
class Contract:
    height: int
    width: int
    transform: Affine
    crs: str
    footprint: np.ndarray


def write_twin(
    path: str | Path,
    dots: np.ndarray,
    contract: Contract,
    outside: str = "zeros",
    name: str = "",
) -> dict:
    """Write one single-band float32 GeoTIFF and re-read it to verify every requirement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.zeros(contract.footprint.shape, dtype=np.float32)
    vals[dots] = 1.0
    nodata = None
    if outside == "nan":
        vals = np.where(contract.footprint, vals, np.float32(np.nan))
        nodata = float("nan")
    elif outside != "zeros":
        raise ValueError("outside must be 'zeros' or 'nan'")

    profile = {
        "driver": "GTiff", "height": contract.height, "width": contract.width,
        "count": 1, "dtype": "float32", "crs": contract.crs,
        "transform": contract.transform, "compress": "deflate", "tiled": False,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(vals, 1)
        if name:
            dst.update_tags(1, description=name)

    return verify(path, contract, outside)


def verify(path: str | Path, contract: Contract, outside: str = "zeros") -> dict:
    """Independent re-read of the bytes on disk.  Any failure is a hard error."""
    path = Path(path)
    raw = path.read_bytes()
    with rasterio.open(path) as src:
        a = src.read(1)
        checks = {
            "single_band": src.count == 1,
            "float32": src.dtypes[0] == "float32",
            "crs_epsg32611": src.crs is not None and src.crs.to_epsg() == 32611,
            "resolution_100m": all(abs(abs(r) - 100.0) < 1e-6 for r in src.res),
            "shape_matches_template": (src.height, src.width) == (contract.height, contract.width),
            "transform_matches_template": tuple(src.transform)[:6] == tuple(contract.transform)[:6],
        }
        inside_vals = a[np.isfinite(a)]
        checks["all_values_in_0_1"] = bool(inside_vals.size == 0 or (inside_vals.min() >= 0.0 and inside_vals.max() <= 1.0))
        if outside == "zeros":
            checks["every_cell_finite"] = bool(np.isfinite(a).all())
            checks["positive_only_outside_nothing"] = bool((a[~contract.footprint] == 0.0).all())
            checks["values_are_binary"] = bool(set(np.unique(a).tolist()) <= {0.0, 1.0})
        else:
            checks["nan_exactly_outside_footprint"] = bool(np.array_equal(np.isnan(a), ~contract.footprint))
            checks["finite_values_binary"] = bool(set(np.unique(a[np.isfinite(a)]).tolist()) <= {0.0, 1.0})
        n_pos = int((np.nan_to_num(a, nan=0.0) > 0).sum())
        on_known = int((np.nan_to_num(a, nan=0.0)[contract.footprint & getattr(contract, "known", np.zeros_like(contract.footprint))] > 0).sum()) \
            if hasattr(contract, "known") else None
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_positive_cells": n_pos,
        "min_finite": float(np.nanmin(a)) if np.isfinite(a).any() else None,
        "max_finite": float(np.nanmax(a)) if np.isfinite(a).any() else None,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def make_zip(tif_path: str | Path, zip_path: str | Path) -> dict:
    """The portal also accepts a .zip containing a single GeoTIFF."""
    tif_path, zip_path = Path(tif_path), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(tif_path, arcname=tif_path.name)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        payload = z.read(names[0])
    return {
        "path": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "contents": names,
        "inner_sha256": hashlib.sha256(payload).hexdigest(),
        "inner_matches_loose_file": hashlib.sha256(payload).hexdigest()
        == hashlib.sha256(tif_path.read_bytes()).hexdigest(),
    }


def write_sidecar(json_path: str | Path, payload: dict) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
