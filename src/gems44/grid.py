"""Grid + raster contract for the GEMS competition (DrivenData #306).

All numbers below were read from the hash-verified official bytes, not assumed.
See ``registry/data_manifest.json`` for the sha256 pins and
``registry/grid_receipt.json`` for the machine-readable read-out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

NODATA = -3.4028234663852886e38  # official training_features.tif nodata
KNOWN_NODATA = -1                # labels.tif nodata (int8)
RADIUS_M = 300.0
PIXEL_M = 100.0


@dataclass(frozen=True)
class Grid:
    """The competition grid, read from the official template."""

    height: int
    width: int
    transform: tuple
    crs: str
    footprint: np.ndarray      # bool, True where the submission may carry a value
    known: np.ndarray          # bool, True on the scored-out known-fault mask
    template_nodata: float

    @property
    def scored(self) -> np.ndarray:
        return self.footprint & ~self.known

    def summary(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
            "transform": [float(v) for v in self.transform],
            "footprint_px": int(self.footprint.sum()),
            "known_fault_px": int(self.known.sum()),
            "scored_domain_px": int(self.scored.sum()),
            "template_nodata": self.template_nodata,
        }


def load_grid(data_dir: str | Path = "data/raw") -> Grid:
    """Build the grid contract from labels.tif and sample_submission.tif."""
    data_dir = Path(data_dir)
    with rasterio.open(data_dir / "labels.tif") as src:
        labels = src.read(1)
        h, w, transform, crs = src.height, src.width, tuple(src.transform)[:6], str(src.crs)
    with rasterio.open(data_dir / "sample_submission.tif") as src:
        template = src.read(1)
        template_nodata = src.nodata
    if labels.shape != template.shape:
        raise ValueError("labels.tif and sample_submission.tif disagree on shape")
    footprint = labels != KNOWN_NODATA
    if not np.array_equal(footprint, np.isfinite(template)):
        raise ValueError("footprint from labels.tif disagrees with the NaN region of sample_submission.tif")
    known = footprint & (labels > 0)
    return Grid(h, w, transform, crs, footprint, known, template_nodata)


def read_band(path: str | Path, index: int) -> np.ndarray:
    """Read one band of the official feature stack as float32 with nodata -> NaN."""
    with rasterio.open(path) as src:
        band = src.read(index).astype(np.float32)
    band[band <= NODATA / 2.0] = np.nan
    return band


def read_labels(data_dir: str | Path = "data/raw") -> np.ndarray:
    with rasterio.open(Path(data_dir) / "labels.tif") as src:
        return src.read(1)


def write_receipt(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
