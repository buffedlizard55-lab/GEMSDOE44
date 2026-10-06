"""Stage 2 - fine-scale placement model (separate from Stage 1).

A gradient-boosted pixel classifier trained on the catalogue proxy, with
spatially-blocked (quadrant) holdout scoring under the official metric.
Placement = greedy best-first dot emission with a 200 m exclusion radius,
restricted to Stage-1-approved zones (the gate), with the metric-optimal
catalogue-flank (B=2 px) removal applied to the candidate field BEFORE
emission so the budget is not wasted.

Arms (all emit the same dot budget in the held-out quadrant):
  A0 random | A1 habitat-only (Stage-1 field) | A2 single-field ridge
  A3 H1 multi-physics coherence only | A4 full model
Promotion rule (pre-registered GEMSDOE44-PREREG-1):
  A4 > A0, A1, A2 in >= 3/4 folds  AND  A4 > A3 in >= 2/4 folds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from .emission import greedy_dots
from .metric import max_kernel_to_truth, score_components
from .features import channel_names


def quadrant_mask(H: int, W: int) -> list[np.ndarray]:
    mids = [H // 2, W // 2]
    masks = []
    for f in range(4):
        m = np.zeros((H, W), bool)
        r0, r1 = (0, mids[0]) if (f // 2 == 0) else (mids[0], H)
        c0, c1 = (0, mids[1]) if (f % 2 == 0) else (mids[1], W)
        m[r0:r1, c0:c1] = True
        masks.append(m)
    return masks


def flank_mask(catalogue: np.ndarray, b: int = C.FLANK_B_PX) -> np.ndarray:
    """Pixels within `b` px (Chebyshev) of the catalogue."""
    return ndimage.binary_dilation(catalogue.astype(bool), iterations=b)


def train_stage2_model(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = C.SEED):
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def gather_pixels(channels: dict, names: list[str], idx_rc: np.ndarray) -> np.ndarray:
    """(N,2) pixel coords -> (N, C) float32 feature rows."""
    X = np.empty((len(idx_rc), len(names)), dtype=np.float32)
    for i, nm in enumerate(names):
        X[:, i] = channels[nm][idx_rc[:, 0], idx_rc[:, 1]]
    return X


def sample_training_pixels(channels: dict, names: list[str],
                           catalogue: np.ndarray, foot: np.ndarray,
                           quadrant: np.ndarray, neg_ratio: int = 5,
                           seed: int = C.SEED):
    """Positive catalogue pixels + `neg_ratio`x random negatives,
    all inside `quadrant & foot`."""
    domain = foot & quadrant & (catalogue == 0)
    pos_idx = np.argwhere((catalogue > 0) & quadrant & foot)
    rng = np.random.default_rng(seed)
    dom_idx = np.argwhere(domain)
    n_neg = min(len(dom_idx), neg_ratio * max(len(pos_idx), 1))
    neg_sel = rng.choice(len(dom_idx), size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, dom_idx[neg_sel]], axis=0)
    rng2 = np.random.default_rng(seed + 1)
    perm = rng2.permutation(len(idx))
    idx = idx[perm]
    y = catalogue[idx[:, 0], idx[:, 1]].astype(int)
    return gather_pixels(channels, names, idx), y


def predict_rows(clf, channels: dict, names: list[str], foot_idx: np.ndarray,
                 pred: np.ndarray, row_band: np.ndarray):
    """Fill pred for the footprint pixels whose row is in `row_band`."""
    m = np.isin(foot_idx[:, 0], row_band)
    if not m.any():
        return
    X = gather_pixels(channels, names, foot_idx[m])
    p = clf.predict_proba(X)[:, 1]
    pred[foot_idx[m, 0], foot_idx[m, 1]] = p
    del X


def arm_score_fields(channels: dict, gate_pixel: np.ndarray,
                     flank: np.ndarray, quadrant: np.ndarray,
                     foot: np.ndarray, score_A4: np.ndarray,
                     fav_pixel: np.ndarray | None = None):
    """Build the five arm score fields (all masked to gate & ~flank & quadrant).

    A1 = habitat-only: the Stage-1 favourability field itself as the
    placement score (the coarse question answering a fine question).
    """
    base_mask = gate_pixel & ~flank & quadrant & foot
    rng = np.random.default_rng(C.SEED)
    random = rng.random(foot.shape, dtype=np.float32)
    if fav_pixel is None:
        a1 = np.ones(foot.shape, np.float32)
    else:
        a1 = fav_pixel.astype(np.float32, copy=True)
    a2 = np.abs(channels["z_det_elev_slope"],
                out=np.empty(foot.shape, np.float32))
    a3 = np.maximum(channels["pair_hi"], channels["pair_lo"])
    # each arm gets its own writable float32 copy; masked in place (no
    # np.where + astype transients)
    out = {}
    for k, f in (("A0_random", random), ("A1_habitat", a1),
                 ("A2_singlefield", a2), ("A3_h1only", a3),
                 ("A4_full", score_A4)):
        g = np.array(f, dtype=np.float32, copy=True)
        np.nan_to_num(g, copy=False)
        g[~base_mask] = 0.0
        out[k] = g
    return out


def stage2_holdout(channels: dict, foot_idx: np.ndarray,
                   catalogue: np.ndarray, foot: np.ndarray,
                   gate_pixel: np.ndarray,
                   budget_fold: int = 10000,
                   emit_arms=None, fav_pixel: np.ndarray | None = None) -> dict:
    """4-quadrant block CV with per-arm proxy DTI under the official metric.

    channels : dict name->(H,W) float32 (all channel_names())
    foot_idx : (n_footprint_px, 2) pixel coordinates
    """
    H, W = catalogue.shape
    quads = quadrant_mask(H, W)
    flank = flank_mask(catalogue)
    names = channel_names()
    results = {a: {"dti": [], "px": []} for a in
               (emit_arms or ["A0_random", "A1_habitat", "A2_singlefield",
                              "A3_h1only", "A4_full"])}
    fold_details = []
    all_rows = np.unique(foot_idx[:, 0])
    for f in range(4):
        tr_q = np.zeros((H, W), bool)
        for g in range(4):
            if g != f:
                tr_q |= quads[g]
        Xtr, y = sample_training_pixels(channels, names, catalogue, foot,
                                        tr_q, seed=C.SEED + f)
        clf = train_stage2_model(Xtr, y, seed=C.SEED + f)
        # predict on the held-out quadrant, tiled (96 px row bands)
        ev = quads[f] & foot
        pred = np.zeros((H, W), np.float32)
        ev_rows = all_rows[ev[all_rows, :].any(axis=1)]
        for i in range(0, len(ev_rows), 96):
            predict_rows(clf, channels, names, foot_idx, pred, ev_rows[i:i + 96])
        pred[~ev] = 0.0
        fields = arm_score_fields(channels, gate_pixel, flank, quads[f],
                                  foot, pred, fav_pixel=fav_pixel)
        fold_res = {"fold": f}
        truth_f = (catalogue * quads[f]).astype(np.float32)
        K_f = max_kernel_to_truth(truth_f, foot)
        # 1 - K computed once per fold (80 MB) instead of per arm
        invK_f = 1.0 - K_f
        for arm in results:
            r, c = greedy_dots(fields[arm], budget_fold, C.EXCLUSION_PX)
            P = np.zeros((H, W), np.float32)
            P[r, c] = 1.0
            s = score_components(P, truth_f, foot, K_cache=K_f,
                                 invK_cache=invK_f)
            results[arm]["dti"].append(s["DTI"])
            results[arm]["px"].append(int(P.sum()))
            fold_res[arm] = s["DTI"]
        fold_details.append(fold_res)
        del fields, pred, invK_f
    return {"folds": fold_details, "arms": {k: {"dti": np.array(v["dti"]),
                                                "px": np.array(v["px"])}
                                            for k, v in results.items()}}
