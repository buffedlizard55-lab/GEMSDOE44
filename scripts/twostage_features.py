"""Cache every Stage B feature as a full-grid float32 .npy (memory-cheap, reproducible).

Feature set = this session's fine-scale library (`gems44.twostage.stage_b.STAGE_B_SPEC`)
unioned with the prior session's multi-scale band transforms
(`gems44.field.FEATURE_SPEC`), de-duplicated.  Both are transforms of the 19 official
bands only -- the catalogue and the SGMC raster are never read here.

Run:  PYTHONPATH=src python scripts/twostage_features.py
Writes: data/interim/feats/<name>.npy  (gitignored) and registry/twostage/feature_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, "src")
from gems44 import field as F
from gems44.twostage import stage_b as B

OUT = Path("data/interim/feats")
MANIFEST = Path("registry/twostage/feature_manifest.json")


def prior_style_features(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The prior session's transforms, recomputed here (same definitions)."""
    from scipy import ndimage

    def fill(a):
        a = a.astype(np.float32)
        a[a <= -1e37] = np.nan
        fin = np.isfinite(a)
        med = float(np.median(a[fin])) if fin.any() else 0.0
        return np.where(fin, a, med).astype(np.float32)

    out: dict[str, np.ndarray] = {}
    for name, band, scale in F.FEATURE_SPEC:
        a = fill(bands[band])
        if scale == 0:
            out[name] = a
        elif band == "detrended_elevation" and scale in (3, 9):
            out[name] = np.abs(ndimage.laplace(ndimage.uniform_filter(a, size=scale))).astype(np.float32)
        else:
            out[name] = ndimage.uniform_filter(a, size=scale).astype(np.float32)
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    need = sorted(set(B.NEEDED_BANDS) | {b for _, b, _ in F.FEATURE_SPEC})
    bands: dict[str, np.ndarray] = {}
    with rasterio.open("data/raw/training_features.tif") as src:
        for n in need:
            bands[n] = src.read(B.BAND_INDEX[n]).astype(np.float32)

    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "data/raw/training_features.tif (19 official bands; no catalogue, no SGMC)",
                "features": {}}

    def persist(name: str, a: np.ndarray) -> None:
        """Write one feature through to disk and free it immediately (3 GB sandbox)."""
        a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.save(OUT / f"{name}.npy", a)
        manifest["features"][name] = {
            "path": str(OUT / f"{name}.npy"),
            "dtype": "float32",
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }
        del a

    seen = set()
    for name in B.feature_names():
        persist(name, B.build_stage_b_feature(name, bands))
        seen.add(name)
        print(f"stage_b {name:26s} {time.time()-t0:7.0f}s", flush=True)
    for name, a in prior_style_features(bands).items():
        if name not in seen:
            persist(name, a)
            print(f"prior   {name:26s} {time.time()-t0:7.0f}s", flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"cached {len(manifest['features'])} features in {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
