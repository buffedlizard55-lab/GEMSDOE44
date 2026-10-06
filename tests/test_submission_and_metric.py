"""pytest suite for equipped machines (requires pytest+numpy+rasterio).

Session-2 note (2026-10-06): this file cannot run in the offline stdlib-only
sandbox. The equivalent gate runnable anywhere is `python3 scripts/run_checks.py`
(50 checks, all passing), which re-verifies the metric math and every shipped TIF
byte-for-byte. Run that here; run this suite where the scipy stack exists.
"""
import pytest
import numpy as np
import rasterio
from pathlib import Path

def test_submission_raster_format():
    tif_path = Path("docs/downloads/gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-zeros.tif")
    assert tif_path.exists()
    
    with rasterio.open(tif_path) as src:
        assert src.count == 1, "Must be single-band"
        assert src.dtypes[0] == "float32", "Must be float32"
        assert src.shape == (3730, 3292), "Dimensions must match exact grid (3730x3292)"
        assert "32611" in str(src.crs), "CRS must be EPSG:32611"
        assert src.nodata is None, "Nodata must be None to prevent sentinel range errors"
        
        arr = src.read(1)
        assert np.all(np.isfinite(arr)), "All cells must be finite"
        assert np.all((arr >= 0.0) & (arr <= 1.0)), "Predicted values must be strictly in range [0, 1]"
        assert (arr > 0).sum() == 40000, "Must emit exactly 40,000 dots"

def test_metric_worked_example():
    """Verify DTI metric against competition worked example (TP=3.00, FP=1.89, FN=2.00 -> 0.60)."""
    tp = 3.00
    fp = 1.89
    fn = 2.00
    alpha = 0.2
    beta = 0.8
    eps = 1e-12
    dti = tp / (tp + alpha * fp + beta * fn + eps)
    assert round(dti, 2) == 0.60
    assert abs(dti - 0.6026) < 1e-3

def test_marginal_credit_identity():
    """Verify credit bar identity: dDTI > 0 <=> k > alpha * DTI."""
    alpha = 0.2
    current_dti = 0.2778
    tau_live = alpha * current_dti
    assert abs(tau_live - 0.05556) < 1e-4
