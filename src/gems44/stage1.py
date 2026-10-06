"""Stage 1 - coarse favourability gate (2 km superpixels).

Answers the coarse question: which broad zones are worth searching at all?
Validated separately on the spatially-blocked holdout (quadrant folds):
  * zone-level AUC
  * concentration ratio of catalogue pixels in the top-q fraction of zones
    (vs the q baseline)
The validation proxy is the visible catalogue - a NECESSARY-condition test
(a gate that cannot concentrate known faults cannot be trusted to
concentrate new ones); like all catalogue-based validation it cannot reward
genuinely new faults (IR-44-PROXY-01).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config as C


def _pad_to_multiple(a: np.ndarray, sup: int) -> np.ndarray:
    H, W = a.shape
    ph = (H + sup - 1) // sup * sup
    pw = (W + sup - 1) // sup * sup
    out = np.full((ph, pw), np.nan, dtype=np.float64)
    out[:H, :W] = a
    return out


def _blocks(a: np.ndarray, sup: int):
    """Return (Hb, Wb, sup, sup) block view of NaN-padded array."""
    p = _pad_to_multiple(a, sup)
    return p.reshape(p.shape[0] // sup, sup, p.shape[1] // sup, sup)


def _block_stat(a: np.ndarray, sup: int, mode: str):
    """Fast per-block stats. NaN handling:

    p90  : NaN -> -inf then np.percentile (exact when a block has >10% valid
           cells - true for every block of this footprint; all-NaN blocks
           give -inf and are zeroed by the z-score step)
    p1   : NaN -> +inf, same argument
    median: NaN -> band global median (coarse stage-1 feature; bias negligible)
    sum  : nansum
    """
    b = _blocks(a, sup)
    if mode == "sum":
        return np.nan_to_num(np.nansum(b, axis=(1, 3)), nan=0.0)
    if mode == "p90":
        with np.errstate(invalid="ignore"):
            return np.percentile(np.where(np.isnan(b), -np.inf, b), 90,
                                 axis=(1, 3))
    if mode == "p1":
        with np.errstate(invalid="ignore"):
            return np.percentile(np.where(np.isnan(b), np.inf, b), 10,
                                 axis=(1, 3))
    if mode == "median":
        fill = np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0
        return np.percentile(np.where(np.isnan(b), fill, b), 50, axis=(1, 3))
    raise ValueError(mode)


def superpixel_ids(H: int, W: int, sup: int = C.SUPERPX):
    hh, ww = (H + sup - 1) // sup, (W + sup - 1) // sup
    rr = (np.arange(H) // sup)[:, None]
    cc = (np.arange(W) // sup)[None, :]
    return (rr * ww + cc).astype(np.int64)


def build_stage1_features(grid, foot: np.ndarray,
                          relay: np.ndarray, endpt: np.ndarray,
                          well_rows: np.ndarray, well_cols: np.ndarray,
                          well_hf: np.ndarray, catalogue: np.ndarray):
    """Per-superpixel favourability features (11 channels).

    Bands are streamed from disk one at a time (float32).
    """
    H, W = foot.shape
    sup = C.SUPERPX
    ids = superpixel_ids(H, W, sup)
    hh, ww = (H + sup - 1) // sup, (W + sup - 1) // sup
    n_sup = hh * ww

    def stat(name: str, mode: str) -> np.ndarray:
        # float32 throughout: halves the transient block arrays (2 x 246 MB ->
        # 2 x 123 MB) and coarse block stats do not need float64 precision
        a = grid.band_array(name)
        a = np.where(np.isfinite(a), a, np.nan)
        s = _block_stat(a, sup, mode)
        return s.astype(np.float64)

    hf = np.zeros(n_sup, np.float32)
    if well_rows.size:
        hot = well_hf >= 1500.0
        if hot.any():
            wf = ids[well_rows[hot], well_cols[hot]]
            hf = np.bincount(wf, minlength=n_sup).astype(np.float32)
    cat = np.bincount(
        ids[catalogue.astype(bool)].ravel(), minlength=n_sup).astype(np.float32)

    iso_p90 = stat("iso_grav_anom", "p90").ravel()
    iso_p1 = stat("iso_grav_anom", "p1").ravel()
    cols_arr = [
        stat("geod_shearrate", "p90"),
        stat("geod_dilaterate", "p90"),
        stat("geod_2ndinv", "p90"),
        np.log1p(stat("deq_n100a15", "sum") + stat("ieq_n100a15", "sum")),
        stat("cond_surf", "p90"),
        stat("tc", "p90"),
        iso_p90 - iso_p1,
        -stat("depth_to_base_surf", "median"),
        np.log1p(_block_stat(np.where(foot, relay, 0.0), sup, "sum")).ravel(),
        np.log1p(_block_stat(np.where(foot, endpt, 0.0), sup, "sum")).ravel(),
        np.log1p(hf),
    ]
    X = np.column_stack([np.asarray(a).ravel() for a in cols_arr]).astype(np.float64)
    # robust z-score per channel
    for i in range(X.shape[1]):
        v = X[:, i]
        fin = np.isfinite(v)
        if fin.sum() < 10:
            X[:, i] = 0.0
            continue
        med = np.median(v[fin])
        lo, hi = np.percentile(v[fin], [10, 90])
        sd = max(hi - lo, 1e-9)
        X[:, i] = np.clip((v - med) / sd, -4, 4)
        X[~np.isfinite(X[:, i]), i] = 0.0
    y = (cat >= 3).astype(int)
    return {"ids": ids, "n_sup": n_sup, "blocks": (hh, ww), "X": X,
            "y": y, "cat_count": cat}


def quadrant_of_block_ids(ids: np.ndarray, blocks: tuple[int, int]) -> np.ndarray:
    hh, ww = blocks
    cper = ww
    r = ids // cper
    c = ids % cper
    return (r >= hh // 2).astype(int) * 2 + (c >= ww // 2).astype(int)


def fit_gate(X: np.ndarray, y: np.ndarray, fit_idx: np.ndarray,
             seed: int = C.SEED):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=seed,
    )
    clf.fit(np.where(np.isfinite(X), X, 0.0)[fit_idx], y[fit_idx])
    return clf


def gate_favourability(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(np.where(np.isfinite(X), X, 0.0))[:, 1]


def stage1_holdout(s1: dict, H: int, W: int, catalogue: np.ndarray,
                   foot: np.ndarray, q: float = C.GATE_Q) -> dict:
    """4-quadrant block CV of the gate (separate, coarse validation)."""
    # fold indices are 1-D superpixel ids 0..n_sup-1 (s1["ids"] is 2-D per
    # pixel - never index feature rows with it)
    n_sup = s1["n_sup"]
    X, y = s1["X"], s1["y"]
    qid = quadrant_of_block_ids(np.arange(n_sup, dtype=np.int64), s1["blocks"])
    cat = s1["cat_count"]

    folds = {}
    for f in range(4):
        fit_idx = np.where(qid != f)[0]
        eval_idx = np.where(qid == f)[0]
        clf = fit_gate(X, y, fit_idx, seed=C.SEED + f)
        proba = gate_favourability(clf, X[eval_idx])
        if 0 < y[eval_idx].sum() < len(y[eval_idx]):
            auc = float(roc_auc_score(y[eval_idx], proba))
        else:
            auc = float("nan")
        order = np.argsort(-proba)
        k = max(1, int(q * len(order)))
        top_global = eval_idx[order[:k]]
        px_top = float(cat[top_global].sum())
        px_tot = float(cat[eval_idx].sum())
        conc = (px_top / px_tot) / q if px_tot > 0 else float("nan")
        folds[f] = {"auc": auc, "concentration_ratio": conc,
                    "px_top": px_top, "px_total": px_tot,
                    "n_blocks": int(len(eval_idx)),
                    "pos_blocks": int(y[eval_idx].sum())}
    clf_all = fit_gate(X, y, np.arange(len(X)), seed=C.SEED)
    fav = gate_favourability(clf_all, X)
    return {"folds": folds, "favourability": fav}
