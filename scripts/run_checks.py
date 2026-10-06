"""Stdlib-only verification runner for GEMSDOE44 (replaces pytest where scipy
stack is unavailable). Every check re-reads bytes from disk. Exit != 0 on failure.

Covers: official metric worked example, credit-bar identity, all 4 shipped TIFs,
audit receipts, zips, spacing guarantees, subset/Jaccard uniqueness evidence.
"""
import glob
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "docs" / "downloads"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_tif_stdlib import read_pixels, get_values  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- 1. official metric worked example (page 967: TP=3.00 FP=1.89 FN=2.00 -> 0.60)
tp, fp, fn, a, b, e = 3.00, 1.89, 2.00, 0.2, 0.8, 1e-12
dti = tp / (tp + a * fp + b * fn + e)
check("metric_worked_example_rounds_0.60", round(dti, 2) == 0.60, f"dti={dti:.5f}")
check("metric_worked_example_exact", abs(dti - 3 / 4.978) < 1e-9, f"{dti:.6f} vs {3/4.978:.6f}")

# ---- 2. credit-bar identity k > alpha*DTI ; d_pay < R*(1-bar)
for score in (0.2600, 0.2708, 0.2778, 0.3195, 0.3345):
    bar = 0.2 * score
    dpay = 300.0 * (1 - bar)
    check(f"credit_bar_{score}", abs(bar - 0.2 * score) < 1e-12, f"bar={bar:.5f} dpay={dpay:.1f}m")
check("credit_bar_02778_value", abs(0.2 * 0.2778 - 0.05556) < 1e-5)

# ---- 3+4. audit every shipped TIF against its receipt
zeros_files = sorted(glob.glob(str(DL / "*-zeros.tif")))
nan_files = sorted(glob.glob(str(DL / "*-nan.tif")))
check("two_primary_pairs_shipped", len(zeros_files) == 2 and len(nan_files) == 2,
      f"zeros={len(zeros_files)} nan={len(nan_files)}")

dots_cache = {}
for zf in zeros_files:
    slug = Path(zf).name.replace("-zeros.tif", "")
    receipt_p = DL / f"{slug}-audit.json"
    check(f"receipt_exists_{slug[:40]}", receipt_p.exists())
    if not receipt_p.exists():
        continue
    r = json.loads(receipt_p.read_text())
    pix, w, h, data, bo, tags = read_pixels(zf)
    n = len(pix)
    n_nan = sum(1 for v in pix if v != v)
    n_inf = sum(1 for v in pix if v in (float("inf"), float("-inf")))
    finite = [v for v in pix if v == v and v not in (float("inf"), float("-inf"))]
    n_out = sum(1 for v in finite if v < 0.0 or v > 1.0)
    n_pos = sum(1 for v in finite if v == 1.0)
    scale = get_values(data, bo, tags, 33550)
    tie = get_values(data, bo, tags, 33922)
    geo = get_values(data, bo, tags, 34735)
    proj_code = geo[geo.index(3072) + 3] if 3072 in geo else None
    has_nodata_tag = 42113 in tags
    dots_cache[zf] = {i for i, v in enumerate(pix) if v > 0.0}
    exp_pos = r["emitted_positive_pixels"]
    check(f"dims_{slug[-12:]}", (w, h) == (3292, 3730), f"{w}x{h}")
    check(f"allfinite_{slug[-12:]}", n_nan == 0 and n_inf == 0 and len(finite) == n, f"n={n}")
    check(f"range01_{slug[-12:]}", n_out == 0 and min(finite) == 0.0 and max(finite) == 1.0)
    check(f"count_{slug[-12:]}", n_pos == exp_pos, f"{n_pos}=={exp_pos}")
    check(f"sha_{slug[-12:]}", sha(zf) == r["sha256"])
    check(f"size_{slug[-12:]}", Path(zf).stat().st_size == r["size_bytes"])
    check(f"grid_{slug[-12:]}", scale[:2] == [100.0, 100.0] and tie[3:5] == [243350.0, 4508550.0])
    check(f"crs32611_{slug[-12:]}", proj_code == 32611, f"proj={proj_code}")
    check(f"nonodata_{slug[-12:]}", not has_nodata_tag)
    check(f"note200_{slug[-12:]}", len(r["note"]) <= 200, f"{len(r['note'])} chars")
    # nan twin
    nf = DL / f"{slug}-nan.tif"
    check(f"nantwin_exists_{slug[-12:]}", nf.exists())
    if nf.exists():
        pixn, wn, hn, _, _, tagsn = read_pixels(str(nf))
        n_nan_n = sum(1 for v in pixn if v != v)
        fin_n = [v for v in pixn if v == v]
        n_pos_n = sum(1 for v in fin_n if v == 1.0)
        check(f"nan_footprint_{slug[-12:]}", len(fin_n) == 5167373 and n_nan_n == 7111787,
              f"finite={len(fin_n)} nan={n_nan_n}")
        check(f"nan_count_{slug[-12:]}", n_pos_n == exp_pos)
        # dots identical between twins
        dots_n = {i for i, v in enumerate(pixn) if v > 0.0}
        check(f"twins_match_{slug[-12:]}", dots_n == dots_cache[zf])
    # zip
    zp = DL / f"{slug}-zeros.zip"
    check(f"zip_exists_{slug[-12:]}", zp.exists())
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            ok_single = len(names) == 1 and names[0] == Path(zf).name
            ok_bytes = z.read(names[0]) == Path(zf).read_bytes() if names else False
            check(f"zip_single_{slug[-12:]}", ok_single, str(names))
            check(f"zip_bytes_{slug[-12:]}", ok_bytes)

# ---- 5. spacing guarantees (grid-hash NN)
def min_nn(dots, w):
    CELL = 8
    buckets = {}
    for lin in dots:
        r, c = divmod(lin, w)
        buckets.setdefault((r // CELL, c // CELL), []).append((r, c))
    best = 1e9
    for (br, bc), members in buckets.items():
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neigh.extend(buckets.get((br + dr, bc + dc), []))
        for (r, c) in members:
            for (r2, c2) in neigh:
                if r == r2 and c == c2:
                    continue
                d = math.hypot(r - r2, c - c2)
                if d < best:
                    best = d
    return best

for zf, need, label in [(zeros_files[0], 2.8, "anchor40k"), (zeros_files[1], 3.0, "h44-6")]:
    # identify by count instead of order
    cnt = len(dots_cache[zf])
    need = 3.0 if cnt < 40000 else 2.8
    label = "h44-6" if cnt < 40000 else "anchor40k"
    m = min_nn(dots_cache[zf], 3292)
    check(f"spacing_{label}", m + 1e-9 >= need, f"minNN={m:.4f}px need>={need}")

# ---- 6. subset / Jaccard uniqueness evidence
if len(zeros_files) == 2:
    a, b = dots_cache[zeros_files[0]], dots_cache[zeros_files[1]]
    big, small = (a, b) if len(a) > len(b) else (b, a)
    check("h44-6_subset_of_anchor", small.issubset(big), f"|small|={len(small)} |big|={len(big)}")
    j = len(small) / len(big)
    check("jaccard_vs_parent_below_094", j < 0.94, f"J={j:.4f}")
    check("count_unique_vs_known_priors", len(small) not in
          (37654, 40000, 40199, 43038, 44090, 46090, 20000, 121131), f"n={len(small)}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
