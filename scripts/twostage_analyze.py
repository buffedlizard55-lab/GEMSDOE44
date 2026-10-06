"""Aggregate the two-stage sweep, calibrate the choice, and print the decision table.

Reads  registry/twostage/sweep.json
Writes registry/twostage/analysis.json

Three statistics per (variant, mass):
  mean_delta_P     mean fold delta of the official DTI against the incumbent, on frame P
                   (SGMC faults > 300 m from the catalogue) -- the primary instrument
  credit_per_dot_P mean TP_w per emitted dot on frame P -- the mass-corrected statistic that
                   reproduced the family's reported live ordering 6/6 (calibrate_instrument.py)
  dti_hidden_density  the same T projected to the family-implied hidden truth size
                   (|G| = 7,905 px, scaled per fold by the fold's area share).  A SCALE CHECK
                   only: it assumes the field's credit scales with truth density.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HIDDEN_G = 7905.0
OUT = Path("registry/twostage")


def main() -> int:
    sweep = json.loads((OUT / "sweep.json").read_text())
    folds = sweep["folds"]
    allowed_total = sum(f["allowed_px"] for f in folds)
    variants = sorted({v for f in folds for m in f["mass"].values() for v in m["variants"]})
    masses = sorted({int(k) for f in folds for k in f["mass"]})

    rows = []
    for M in masses:
        for v in variants:
            dP, dN, cpd, hid, winsP, winsN = [], [], [], [], 0, 0
            for f in folds:
                key = str(M)
                if key not in f["mass"]:
                    continue
                val = f["mass"][key]["variants"].get(v)
                if val is None:
                    continue
                inc = f["reference"]["incumbent"]
                incP = f["reference"]["incumbent_frameP"]
                dP.append(val["P"]["DTI"] - incP["DTI"])
                dN.append(val["N"]["DTI"] - inc["DTI"])
                winsP += int(val["P"]["DTI"] > incP["DTI"])
                winsN += int(val["N"]["DTI"] > inc["DTI"])
                n_fold = val["P"]["n"]
                cpd.append(val["P"]["T"] / max(n_fold, 1))
                share = f["allowed_px"] / allowed_total
                g_hid = HIDDEN_G * share
                r = g_hid / max(val["P"]["G"], 1)
                hid.append((val["P"]["T"] * r) / (0.2 * n_fold + 0.8 * g_hid))
            rows.append({
                "variant": v, "mass_global": M,
                "mean_delta_P": float(np.mean(dP)), "folds_positive_P": winsP, "n_folds": len(dP),
                "mean_delta_N": float(np.mean(dN)), "folds_positive_N": winsN,
                "credit_per_dot_P": float(np.mean(cpd)),
                "dti_hidden_density": float(np.mean(hid)),
            })
    rows.sort(key=lambda r: -r["mean_delta_P"])
    for r in rows:
        print(f"{r['variant']:26s} M={r['mass_global']:6d}  dP={r['mean_delta_P']:+.4f} ({r['folds_positive_P']}/4)"
              f"  dN={r['mean_delta_N']:+.4f}  c/dot={r['credit_per_dot_P']:.4f}"
              f"  dti@hidden={r['dti_hidden_density']:.4f}")
    # ---- the metric's marginal rule at hidden density, for the shipping decision ------------
    marginal = {}
    for var in sorted({r["variant"] for r in rows}):
        seq = [r for r in rows if r["variant"] == var]
        seq.sort(key=lambda r: r["mass_global"])
        for a, b in zip(seq, seq[1:]):
            mc, bars, n_folds = [], [], 0
            for f in folds:
                ka, kb = str(a["mass_global"]), str(b["mass_global"])
                va = f["mass"].get(ka, {}).get("variants", {}).get(var)
                vb = f["mass"].get(kb, {}).get("variants", {}).get(var)
                if va is None or vb is None:
                    continue
                share = f["allowed_px"] / allowed_total
                g_hid = HIDDEN_G * share
                r = g_hid / max(vb["P"]["G"], 1)
                dn = vb["P"]["n"] - va["P"]["n"]
                if dn <= 0:
                    continue
                dti_hid_b = (r * vb["P"]["T"]) / (0.2 * vb["P"]["n"] + 0.8 * g_hid)
                mc.append(r * (vb["P"]["T"] - va["P"]["T"]) / dn)
                bars.append(0.2 * dti_hid_b)
                n_folds += 1
            if not mc:
                continue
            marginal[f"{var}|{a['mass_global']}->{b['mass_global']}"] = {
                "marginal_credit_at_hidden_density": float(np.mean(mc)),
                "bar_0p2_dti": float(np.mean(bars)),
                "safety_factor": float(np.mean(mc) / max(np.mean(bars), 1e-9)),
                "pays": bool(np.mean(mc) > np.mean(bars)),
                "n_folds": n_folds,
            }
    payload = {
        "marginal_rule_at_hidden_density": marginal,
        "note": ("frame P = SGMC faults > 300 m from the catalogue; the mass-corrected statistic on this "
                 "frame is the instrument that reproduced the family's reported live ordering 6/6.  All "
                 "local numbers are proxies; no organizer score exists for any candidate here."),
        "hidden_g_assumed_px": HIDDEN_G,
        "allowed_total_px": allowed_total,
        "rows": rows,
        "by_hidden_density": sorted(rows, key=lambda r: -r["dti_hidden_density"])[:12],
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nSorted by the hidden-density scale check:")
    for r in payload["by_hidden_density"][:8]:
        print(f"  {r['variant']:26s} M={r['mass_global']:6d} dti@hidden={r['dti_hidden_density']:.4f}"
              f" dP={r['mean_delta_P']:+.4f} c/dot={r['credit_per_dot_P']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
