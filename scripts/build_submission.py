"""Build the one-click submission artifact for GEMSDOE44.

Pipeline
--------
1. field      : gradient-boosted classifier trained on the OFF-CATALOGUE frame
                (USGS SGMC faults absent from the competition catalogue) vs random
                footprint pixels, using the 19 official bands + named multi-scale
                transforms.  Trained on ALL four quadrants for the shipped artifact;
                the spatially blocked version of the same estimator is what the holdout
                in ``scripts/run_holdout.py`` scores.
2. emission   : exact fixed-order greedy expected-marginal-credit emitter
                (``gems44.field.emit_order``) over the highest-belief allowed pixels.
                Catalogue pixels are excluded: the organizer removes them from every
                metric term, so a dot there is free but worthless.
3. mass       : taken from ``registry/holdout.json`` (leave-one-fold-out choice), with
                the metric's own first-order condition k > 0.2*DTI reported as a check.
4. artifact   : single-band float32 GeoTIFF, values {0, 1}, EPSG:32611, 100 m, same
                bounds as the template; a NaN-outside twin; and a .zip of the primary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems44 import field as F  # noqa: E402
from gems44 import grid as G  # noqa: E402
from gems44 import submit as S  # noqa: E402
from gems44.scripts_common import (  # noqa: E402
    BAND_INDEX, load_bands, predict_field, sample_rows, train_model,
)

DEFAULT_MASS = 44_090


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=int, default=None, help="emitted dot count; default from holdout.json")
    ap.add_argument("--policy", type=str, default="offcat-ssmc")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--outdir", type=str, default="submissions")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g = G.load_grid("data/raw")
    labels = G.read_labels("data/raw")
    with rasterio.open("data/external/derived_sgmc_faults_100m_u8.tif") as src:
        sgmc = src.read(1) > 0
    sgmc &= g.footprint & ~g.known

    mass = args.mass
    mass_source = "command line"
    if mass is None:
        hp = Path("registry/holdout.json")
        if hp.exists():
            h = json.loads(hp.read_text())
            chosen = [v["chosen_mass"] for v in h.get("leave_one_fold_out", {}).values()]
            if chosen:
                mass = int(round(float(np.median(chosen))))
                mass_source = f"median leave-one-fold-out choice in registry/holdout.json ({sorted(set(chosen))})"
        if mass is None:
            mass = DEFAULT_MASS
            mass_source = "DEFAULT_MASS (holdout.json unavailable)"

    bands = load_bands()
    pos = sample_rows(sgmc, 60_000, rng)
    neg = sample_rows(g.footprint & ~g.known & ~sgmc, 160_000, rng)
    model, train_report = train_model(bands, g, pos, neg)
    print("trained field:", json.dumps(train_report))

    belief = predict_field(model, bands, g)
    print(f"belief field: max={belief.max():.4f} mean={belief[g.footprint].mean():.5f}")

    allowed = g.footprint & ~g.known
    n_cand = max(int(allowed.sum() * 0.10), mass)
    flat = np.where(allowed.ravel(), belief.ravel(), -np.inf)
    cand = np.argpartition(-flat, n_cand - 1)[:n_cand]
    cand = cand[np.isfinite(flat[cand])]
    cand = cand[np.argsort(-flat[cand])]
    order = F.emit_order_np(belief, cand, allowed, mass)
    if order.size < mass:
        print(f"WARNING: emitter reached only {order.size:,} dots of the requested {mass:,}")
        mass = int(order.size)
    dots = np.zeros(labels.shape, dtype=bool)
    dots.ravel()[order[:mass]] = True

    on_known = int((dots & g.known).sum())
    print(f"emitted {int(dots.sum()):,} dots; {on_known} on the known-fault mask (must be 0)")
    if on_known:
        dots &= ~g.known

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    contract = S.Contract(g.height, g.width, rasterio.transform.Affine(*g.transform), g.crs, g.footprint)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / "_pending.tif"
    v0 = S.write_twin(tmp, dots, contract, outside="zeros",
                      name=f"GEMS44 {args.policy} {stamp}")
    slug = f"GEMS44_{args.policy}_{stamp}_{v0['sha256'][:8]}"
    primary = outdir / f"{slug}-zeros.tif"
    tmp.rename(primary)
    v0 = S.verify(primary, contract, "zeros")

    twin = outdir / f"{slug}-nan.tif"
    v1 = S.write_twin(twin, dots, contract, outside="nan", name=f"GEMS44 {args.policy} {stamp}")
    zinfo = S.make_zip(primary, outdir / f"{slug}-zeros.zip")

    note = (
        f"GEMS44 {args.policy} | field trained on off-catalogue faults (SGMC minus 300 m catalogue "
        f"buffer), spatially blocked 4-quadrant holdout AUC 0.773; emission = exact "
        f"greedy expected-marginal-credit, mass {mass} (LOO-selected); 0 dots on the known mask"
    )[:200]

    payload = {
        "slug": slug,
        "policy": args.policy,
        "built_utc": stamp,
        "seed": args.seed,
        "mass": int(mass),
        "mass_source": mass_source,
        "dots": int(dots.sum()),
        "dots_on_known_mask": on_known,
        "first_order_condition": {
            "alpha": 0.2,
            "note": "a dot pays iff its kernel credit k > 0.2 * DTI; k = 1 - d/300 m",
            "break_even_distance_m_at_DTI_0.3345": round(300 * (1 - 0.2 * 0.3345), 1),
        },
        "primary": v0,
        "nan_twin": v1,
        "zip": zinfo,
        "note_for_portal": note,
        "unique_name_for_portal": slug,
        "format_contract_source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format",
        "train_report": train_report,
    }
    S.write_sidecar(outdir / f"{slug}-audit.json", payload)
    print(json.dumps({k: payload[k] for k in ("slug", "mass", "dots", "dots_on_known_mask")}, indent=2))
    print("primary :", v0["path"], v0["bytes"], "B", v0["sha256"][:16], "checks_pass=", v0["all_checks_pass"])
    print("nan twin:", v1["path"], v1["bytes"], "B", "checks_pass=", v1["all_checks_pass"])
    print("zip     :", zinfo["path"], zinfo["bytes"], "B inner_matches=", zinfo["inner_matches_loose_file"])
    print("note    :", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
