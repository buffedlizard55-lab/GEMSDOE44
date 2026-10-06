"""Build the one-click submission artifact for GEMSDOE44.

Pipeline (every step auditable, none of it hard-coded in HTML)

1. field    gradient-boosted classifier on the 19 official bands + named multi-scale transforms,
            trained to predict the SGMC strands that the competition catalogue does not carry.
            --target N  : positives = strands within 300 m of the catalogue (frame N, 17,493 px)
            --target U  : positives = all off-catalogue strands (79,615 px)
            --target NU : positives = both, as one class
            Training for the shipped artifact uses ALL four quadrants; the spatially blocked
            version of the same estimator is what registry/selection.json scores.
2. emission exact fixed-order greedy max-coverage emitter (gems44.field.emit_order_np) over the
            highest-belief allowed pixels.  By construction it never accepts a dot that adds no
            new kernel credit, so no two dots are 8-adjacent - which is what the 19 reported
            scores in registry/probe_sgmc_alignment.json say separates good artifacts from bad.
            Catalogue pixels are excluded: the organizer deletes them from every metric term.
3. mass     from --mass (default: registry/selection.json's LOO choice, then 44,090).
4. artifact single-band float32 GeoTIFF {0,1}, EPSG:32611, 100 m, same bounds as the template,
            a NaN-outside twin, a .zip, and an audit sidecar with the emission diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import field as F  # noqa: E402
from gems44 import grid as G  # noqa: E402
from gems44 import submit as S  # noqa: E402
from gems44.metric import binary_credit, dti_of  # noqa: E402
from gems44.scripts_common import (  # noqa: E402
    load_bands, predict_field, sample_rows, train_model,
)

DEFAULT_MASS = 44_090
CAND_FRACTION = 0.25  # large enough that a 3 px separation can still place ~44k dots
                      # (with 0.06 the emitter saturated at ~10-20k and the mass was not reached)


def min_separation_ok(dots: np.ndarray, min_sep: float) -> dict:
    """Exact check that no dot lies within ``min_sep`` px of another dot.

    A distance transform cannot be used directly (every dot is at distance 0 from itself), so
    convolve with the disc of radius ceil(min_sep)-1 and subtract the dot itself: if the maximum
    neighbour count is 0, no pair is closer than min_sep.
    """
    r = int(np.ceil(min_sep)) - 1
    if r <= 0 or not dots.any():
        return {"radius_px_checked": 0, "max_neighbours_within_radius": 0, "ok": True}
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((yy * yy + xx * xx) <= r * r).astype(np.float32)
    counts = ndimage.convolve(dots.astype(np.float32), disc, mode="constant") - 1.0
    m = float(counts[dots].max()) if dots.any() else 0.0
    return {"radius_px_checked": r, "max_neighbours_within_radius": int(round(m)), "ok": bool(m <= 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["N", "U", "NU"], default="N",
                    help="which strand population the field predicts")
    ap.add_argument("--mass", type=int, default=None)
    ap.add_argument("--policy", type=str, default="n-strand-ssmc")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--outdir", type=str, default="submissions")
    ap.add_argument("--belief", type=str, default=None, help="optional .npy belief field to reuse")
    ap.add_argument("--cand-fraction", type=float, default=CAND_FRACTION)
    ap.add_argument("--min-sep", type=float, default=3.0,
                    help="H7 minimum Euclidean separation between accepted dots, in pixels "
                         "(3 px = the metric's break-even distance at a score of 0.26)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="geometric blend weight on the supervised field: belief = "
                         "sup**alpha * prox10**(1-alpha); alpha=1 -> supervised only, 0 -> catalogue "
                         "proximity only (the frame-N winner but not a fault prediction)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known
    cat = g.footprint & (labels > 0)
    d_cat = ndimage.distance_transform_edt(~cat)

    positives = {
        "N": sgmc & (d_cat <= 3.0),
        "U": sgmc,
        "NU": sgmc,
    }[args.target]

    # ---- mass
    mass = args.mass
    mass_source = "command line"
    if mass is None:
        sp = Path("registry/selection.json")
        if sp.exists():
            s = json.loads(sp.read_text())
            gates = s.get("pooled", {})
            if gates.get("gate_pass") and s.get("best_mean_over_all_folds", {}).get("mass"):
                mass = int(s["best_mean_over_all_folds"]["mass"])
                mass_source = "registry/selection.json best_mean_over_all_folds (gate PASS)"
        if mass is None:
            mass = DEFAULT_MASS
            mass_source = "DEFAULT_MASS (no gate-passing selection)"

    # ---- field
    cache = Path(args.belief) if args.belief else Path(f"data/interim/belief_{args.target}.npy")
    if cache.exists():
        belief = np.load(cache)
        train_report = {"loaded_from": str(cache)}
        print(f"belief loaded from {cache}")
    else:
        bands = load_bands()
        pos = sample_rows(positives, 60_000, rng)
        neg = sample_rows(g.footprint & ~g.known & ~sgmc, 160_000, rng)
        model, train_report = train_model(bands, g, pos, neg)
        belief = predict_field(model, bands, g)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, belief)
        print("trained field:", json.dumps({k: v for k, v in train_report.items() if k != "feature_names"}))
    belief = np.asarray(belief, dtype=np.float32)
    if args.alpha < 1.0:
        prox10 = np.exp(-d_cat / 10.0).astype(np.float32)
        belief = (np.maximum(belief, 0.0) ** args.alpha) * (prox10 ** (1.0 - args.alpha))
        belief = belief.astype(np.float32)
        print(f"blended with catalogue proximity at alpha={args.alpha}")
    print(f"belief field: max={belief.max():.4f} mean={belief[g.footprint].mean():.5f}")

    # ---- emission
    allowed = g.footprint & ~g.known
    order = F.emit_order_np(belief, F.quantile_candidates(belief, allowed, args.cand_fraction), allowed, mass,
                            min_sep=args.min_sep)
    if order.size < mass:
        print(f"WARNING: emitter reached only {order.size:,} dots of the requested {mass:,}")
        mass = int(order.size)
    dots = np.zeros(labels.shape, dtype=bool)
    dots.ravel()[order[:mass]] = True

    on_known = int((dots & g.known).sum())
    if on_known:
        print(f"WARNING: {on_known} dots on the known mask - removing")
        dots &= ~g.known
    n_dots = int(dots.sum())

    # ---- emission diagnostics (the properties the 19 reported scores singled out)
    adj = int((dots & (ndimage.uniform_filter(dots.astype(np.float32), size=3) * 9.0 >= 2.0)).sum())
    nb3 = int((dots & (ndimage.uniform_filter(dots.astype(np.float32), size=7) * 49.0 >= 2.0)).sum())
    diag = {
        "dots": n_dots,
        "dots_on_known_mask": on_known,
        "dots_8_adjacent_to_another": adj,
        "dots_within_300m_of_another_dot": nb3,
        "frac_8_adjacent": round(adj / max(n_dots, 1), 5),
        "frac_within_300m_of_another": round(nb3 / max(n_dots, 1), 5),
        "min_sep_enforced_px": args.min_sep,
        "min_sep_verified": min_separation_ok(dots, args.min_sep),
        "inside_footprint": int((dots & g.footprint).sum()),
        "within_300m_of_catalogue": int((dots & (d_cat <= 3)).sum()),
        "within_300m_of_sgmc": int((dots & (ndimage.distance_transform_edt(~sgmc) <= 3)).sum()),
    }

    # ---- frame scores of the emitted set (diagnostic only; frame N is the field's own target,
    #      so its score is circular for THIS artifact and is labelled as such)
    frames = {"N": sgmc & (d_cat <= 3.0), "P": sgmc & (d_cat > 3.0)}
    diag["frame_scores_circular_for_N_trained_field"] = {
        k: round(dti_of(binary_credit(dots, v, valid=g.footprint, known=g.known)), 6)
        for k, v in frames.items()}

    # ---- uniqueness against the incumbent control
    ip = Path("data/raw/incumbent_d28.tif")
    if ip.exists():
        with rasterio.open(ip) as src:
            v = src.read(1)
        inc = np.isfinite(v) & (v > 0)
        diag["incumbent_global_frames_same_operator"] = {
            k: round(dti_of(binary_credit(inc, vv, valid=g.footprint, known=g.known)), 6)
            for k, vv in frames.items()}
        diag["incumbent_mass"] = int(inc.sum())
        inter = int((dots & inc).sum())
        diag["uniqueness_vs_incumbent"] = {
            "incumbent_dots": int(inc.sum()), "shared_dots": inter,
            "jaccard": round(inter / max(int((dots | inc).sum()), 1), 4),
        }

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    contract = S.Contract(g.height, g.width, rasterio.transform.Affine(*g.transform), g.crs, g.footprint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / "_pending.tif"
    v0 = S.write_twin(tmp, dots, contract, outside="zeros", name=f"GEMS44 {args.policy} {stamp}")
    slug = f"GEMS44_{args.policy}_{stamp}_{v0['sha256'][:8]}"
    primary = outdir / f"{slug}-zeros.tif"
    tmp.rename(primary)
    v0 = S.verify(primary, contract, "zeros")
    twin = outdir / f"{slug}-nan.tif"
    v1 = S.write_twin(twin, dots, contract, outside="nan", name=f"GEMS44 {args.policy} {stamp}")
    zinfo = S.make_zip(primary, outdir / f"{slug}-zeros.zip")

    note = (f"GEMS44 {args.policy} | field = supervised transfer of USGS SGMC strands absent from the "
            f"catalogue ({args.target}) onto the 19 official bands; exact greedy max-coverage emission "
            f"{n_dots} dots; 0 on the known mask; unique file")[:200]

    payload = {
        "slug": slug, "policy": args.policy, "target": args.target, "alpha": args.alpha,
        "min_sep_px": args.min_sep, "cand_fraction": args.cand_fraction,
        "built_utc": stamp, "seed": args.seed,
        "mass": int(n_dots), "mass_source": mass_source, "emission_diagnostics": diag,
        "first_order_condition": {
            "alpha": 0.2,
            "note": "a dot pays iff its kernel credit k > 0.2 * DTI; k = 1 - d/300 m",
            "break_even_distance_m_at_DTI_0.2600": round(300 * (1 - 0.2 * 0.2600), 1),
            "break_even_distance_m_at_DTI_0.3345": round(300 * (1 - 0.2 * 0.3345), 1),
        },
        "primary": v0, "nan_twin": v1, "zip": zinfo,
        "note_for_portal": note, "unique_name_for_portal": slug,
        "format_contract_source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        "train_report": train_report,
        "evidence_used": ["registry/preregistration.json (AM-44-02 field choice)",
                          "registry/selection.json", "registry/frame_ranking.json",
                          "registry/probe_sgmc_alignment.json",
                          "registry/exact_confirmation.json (written by scripts/confirm_exact.py)"],
        "known_irregularity": "IR-44-03: the 0.2778 attribution is confirmed by thread 11525 only "
                              "through the artifact's file name, not yet by an independent DTI reproduction",
    }
    S.write_sidecar(outdir / f"{slug}-audit.json", payload)

    # Publish the artifact: machine-readable registry record + the files the site links to.
    reg = Path("registry/submission.json")
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    pub = Path("docs/downloads")
    pub.mkdir(parents=True, exist_ok=True)
    for old in pub.glob("GEMS44_*"):
        old.unlink()
    copied = []
    for src in (primary, twin, outdir / f"{slug}-zeros.zip"):
        dst = pub / src.name
        shutil.copy2(src, dst)
        copied.append(f"{dst} ({dst.stat().st_size} B)")
    payload["published_to"] = copied
    reg.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("registry:", reg)
    for c in copied:
        print("published:", c)
    print(json.dumps({"slug": slug, "mass": mass, "dots": n_dots, "diag": diag}, indent=1))
    print("primary :", v0["path"], v0["bytes"], "B", v0["sha256"][:16], "checks_pass=", v0["all_checks_pass"])
    print("nan twin:", v1["path"], v1["bytes"], "B", "checks_pass=", v1["all_checks_pass"])
    print("zip     :", zinfo["path"], zinfo["bytes"], "B inner_matches=", zinfo["inner_matches_loose_file"])
    print("note    :", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
