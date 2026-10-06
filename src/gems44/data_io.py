"""Load and verify the official competition rasters and external official mirrors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from . import config as C


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Pins from the GEMSDOE repo data/bridge/manifest.json (mirrors of the official
# DrivenData data tab; each pin also matches the dropbox mirror URL listed there).
OFFICIAL_PINS = {
    "training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "example_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
}


@dataclass
class Grid:
    """Verified competition grid + official rasters as numpy arrays.

    The 19-band feature stack (~1.9 GB float64) is NOT held resident; use
    band_array(name) for one band at a time (float32, sentinel -> NaN).
    """
    rows: int
    cols: int
    transform: tuple
    crs: str
    footprint: np.ndarray          # bool, True inside the survey footprint
    catalogue: np.ndarray          # 0/1 float: catalogue fault pixels
    catalog_outside: np.ndarray    # bool: outside-footprint cells (value -1)

    @property
    def n_footprint(self) -> int:
        return int(self.footprint.sum())

    @property
    def n_catalogue(self) -> int:
        return int(self.catalogue.sum())

    def band_array(self, name: str) -> np.ndarray:
        import rasterio
        from . import config as C
        with rasterio.open(C.FEATURES_TIF) as ds:
            a = ds.read(C.B[name]).astype(np.float32)
        a[np.isclose(a, C.NODE_DATA_SENTINEL, rtol=0.0, atol=1.0)] = np.nan
        return a


def verify_official_rasters() -> list[dict]:
    """Re-open all three official rasters and verify pins + geometry."""
    out = []
    for name, pin in OFFICIAL_PINS.items():
        p = C.DATA / name
        actual = sha256_of(p)
        with rasterio.open(p) as ds:
            out.append({
                "file": name,
                "sha256_ok": actual == pin,
                "sha256": actual,
                "bands": ds.count,
                "shape": [ds.height, ds.width],
                "dtype": str(ds.dtypes[0]),
                "crs": str(ds.crs),
                "transform": list(ds.transform),
            })
    return out


def load_grid() -> Grid:
    with rasterio.open(C.CATALOG_TIF) as ds:
        cat_raw = ds.read(1)
        transform = tuple(ds.transform)
        crs = str(ds.crs)
        H, W = ds.height, ds.width
    outside = cat_raw == -1
    footprint = ~outside
    catalogue = (cat_raw == 1).astype(np.float32)

    with rasterio.open(C.FEATURES_TIF) as ds:
        assert (ds.height, ds.width) == (H, W)
        assert str(ds.crs) == crs
        assert tuple(ds.transform) == transform
        assert ds.count == len(C.BAND_NAMES)

    return Grid(H, W, transform, crs, footprint, catalogue, outside)


def load_example_template() -> tuple[dict, np.ndarray]:
    """Return (meta, array) of the official sample submission (format template)."""
    with rasterio.open(C.EXAMPLE_TIF) as ds:
        arr = ds.read(1).astype(np.float64)
        meta = {
            "crs": str(ds.crs),
            "transform": tuple(ds.transform),
            "shape": [ds.height, ds.width],
            "dtype": str(ds.dtypes[0]),
            "nodata": ds.nodata,
        }
    return meta, arr


def load_heatflow_wells() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """USGS Great Basin heat-flow well points (mirrored from GEMSDOE28 docs/data).

    The CSV carries the original USAEAC_83_117 (Albers, centre -117,
    std parallels 29.5/45.5, false easting/northing 0) coordinates in the
    x_eac117 / y_eac117 columns. We reproject to WGS84 then to EPSG:32611
    with pyproj (the sidecar JSON documents this source CRS verbatim).

    Returns (x_32611, y_32611, hf_meas) for wells inside the grid bbox.
    """
    import csv
    import pyproj

    albers = pyproj.CRS.from_proj4(
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-117 "
        "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    )
    wgs84 = pyproj.CRS.from_epsg(4326)
    utm11 = pyproj.CRS.from_epsg(32611)
    tr_a2g = pyproj.Transformer.from_crs(albers, wgs84, always_xy=True)
    tr_g2u = pyproj.Transformer.from_crs(wgs84, utm11, always_xy=True)

    xs, ys, hfs = [], [], []
    with open(C.HEATFLOW_CSV, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row.get("kind") != "Point":
                continue
            try:
                x0 = float(row["x_eac117"])
                y0 = float(row["y_eac117"])
                hf = float(row["hf_meas"])
            except (KeyError, TypeError, ValueError):
                continue
            lon, lat = tr_a2g.transform(x0, y0)
            if not (-121.5 < lon < -114.0 and 36.5 < lat < 42.0):
                continue
            x, y = tr_g2u.transform(lon, lat)
            xs.append(x)
            ys.append(y)
            hfs.append(hf)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    hfs = np.asarray(hfs, dtype=np.float64)
    # keep wells inside the competition grid
    e_min, n_max = C.ORIGIN_E, C.ORIGIN_N
    e_max = C.ORIGIN_E + C.COLS * C.CELL_M
    n_min = C.ORIGIN_N - C.ROWS * C.CELL_M
    keep = (xs >= e_min) & (xs <= e_max) & (ys >= n_min) & (ys <= n_max) & np.isfinite(hfs)
    return xs[keep], ys[keep], hfs[keep]


def wells_to_grid(xs: np.ndarray, ys: np.ndarray,
                  values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices (row, col) of wells on the competition grid + their values."""
    c = (xs - C.ORIGIN_E) / C.CELL_M
    r = (C.ORIGIN_N - ys) / C.CELL_M
    cols = np.clip(np.round(c).astype(int), 0, C.COLS - 1)
    rows = np.clip(np.round(r).astype(int), 0, C.ROWS - 1)
    return rows, cols, values
