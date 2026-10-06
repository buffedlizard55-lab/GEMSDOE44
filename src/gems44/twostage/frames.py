"""Evaluation frames for the two-stage GEMS44 system.

WHY THIS MODULE EXISTS
----------------------
The competition hides the truth.  Every local number is a *proxy*.  The prior
sessions in this family established, by measurement, which proxies are usable:

  * catalogue truth (`labels.tif`) is what the competition MASK OUT, and training
    on it makes a model better at the wrong thing.  19GEMSDOE measured
    Spearman rho = +0.11 (p = 0.70, n = 15 scored artifacts) between
    catalogue-truth "known_dense" DTI and the live leaderboard, and the
    family's own three-artifact experiment ranked 0.17193 > 0.16635 > 0.16177
    on the catalogue proxy while the live board ranked the same files
    0.1922 < 0.2477 < 0.2600 (measured in GEMSDOE32, registry/frame_validity).

  * the USGS State Geologic Map Compilation (SGMC) fault layer contains real,
    mapped faults that the given catalogue does not contain.  Restricted to
    pixels further than 300 m (one DTI kernel radius) from the given catalogue
    it is the only local set of *catalogue-absent, really-existing* faults
    available, and it is the frame on which live leaderboard order is at least
    partially recoverable: 19GEMSDOE reported Spearman rho = +0.52 (p = 0.048)
    against live scores for exactly this external `sgmc_gap` frame.

This module builds those frames and the spatially blocked folds used by both
stages.  It never trains on the catalogue and never scores a candidate on the
pixels the truth set is known to contain.

FRAME DEFINITIONS (all boolean rasters, competition grid, EPSG:32611, 100 m)
  valid            labels.tif != -1 (the submission footprint, 5,167,373 px)
  known            labels.tif  > 0  (USGS/INGENIOUS catalogue, 60,988 px, MASKED
                   by the official scorer, therefore never truth, never penalty)
  sgmc             derived SGMC fault raster > 0  (83,593 px total)
  P  (primary)     sgmc & ~known & valid & d(known) > 300 m   -- 62,122 px
  U  (union)       sgmc & ~known & valid                      -- 79,615 px
  N  (near)        sgmc & ~known & valid & d(known) <= 300 m  -- 17,493 px

P is the primary validation frame: faults that exist in an official compilation
and are demonstrably absent from the competition catalogue by more than one
kernel radius.  N is carried as a control: a detector that is only good at the
catalogue flank is not a detector of catalogue-absent faults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..grid import KNOWN_NODATA

RADIUS_PX = 3.0
PIXEL_M = 100.0


@dataclass(frozen=True)
class Frames:
    """Every mask needed to build and score a two-stage candidate."""

    valid: np.ndarray          # submission footprint
    known: np.ndarray          # catalogue (masked out by the official scorer)
    sgmc: np.ndarray           # all SGMC faults on the grid
    d_known: np.ndarray        # Euclidean distance (px) to the nearest catalogue pixel
    quad: np.ndarray           # quadrant id 0..3 for spatially blocked folds

    # ---- derived truth frames -------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """Primary: SGMC faults further than 300 m from the catalogue."""
        return self.sgmc & ~self.known & self.valid & (self.d_known > RADIUS_PX)

    @property
    def U(self) -> np.ndarray:
        """Union: every SGMC fault the catalogue lacks."""
        return self.sgmc & ~self.known & self.valid

    @property
    def N(self) -> np.ndarray:
        """Near: SGMC faults inside the catalogue's 300 m halo (control frame)."""
        return self.sgmc & ~self.known & self.valid & (self.d_known <= RADIUS_PX)

    def summary(self) -> dict:
        return {
            "valid_px": int(self.valid.sum()),
            "known_catalogue_px": int(self.known.sum()),
            "sgmc_all_px": int(self.sgmc.sum()),
            "frame_P_sgmc_beyond_300m_px": int(self.P.sum()),
            "frame_U_sgmc_off_catalogue_px": int(self.U.sum()),
            "frame_N_sgmc_within_300m_px": int(self.N.sum()),
            "pixel_m": PIXEL_M,
            "kernel_radius_px": RADIUS_PX,
        }


def load_frames(data_dir: str = "data/raw", sgmc_path: str = "data/external/derived_sgmc_faults_100m_u8.tif",
                height: int = 3730, width: int = 3292) -> Frames:
    """Read the grid contract rasters and build every frame + the 2x2 blocked folds."""
    import rasterio

    with rasterio.open(f"{data_dir}/labels.tif") as src:
        labels = src.read(1)
        if (src.height, src.width) != (height, width):
            raise ValueError("labels.tif does not match the competition grid")
    with rasterio.open(sgmc_path) as src:
        sgmc = src.read(1) > 0

    valid = labels != KNOWN_NODATA
    known = labels > 0
    if sgmc.shape != labels.shape:
        raise ValueError("SGMC raster shape mismatch")

    # 2x2 spatially blocked quadrants: large enough that a held-out quadrant is
    # > 1,200 km^2, so regional structure cannot leak between train and test.
    yy, xx = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
    quad = (yy >= labels.shape[0] // 2).astype(np.int8) * 2 + (xx >= labels.shape[1] // 2).astype(np.int8)

    d_known = distance_transform_edt(~known)
    return Frames(valid=valid, known=known, sgmc=sgmc, d_known=d_known, quad=quad)
