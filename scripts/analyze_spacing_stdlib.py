"""Measure dot-spacing distribution of the committed GEMSDOE44 primary (stdlib-only).

Uses a grid-hash so 40k dots x neighbor lookup stays fast in pure Python.
Also verifies zip-inner bytes equal the zeros.tif and decodes full GeoKeys.
"""
import hashlib
import math
import zipfile
from pathlib import Path
from audit_tif_stdlib import read_pixels, get_values

Z = Path("docs/downloads/gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-zeros.tif")
N = Path("docs/downloads/gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-nan.tif")

pix, w, h, data, bo, tags = read_pixels(Z)
dots = [i for i, v in enumerate(pix) if v > 0.0]
print("dots:", len(dots), "grid:", w, "x", h)
assert len(dots) == 40000

# footprint from nan twin
pixn, wn, hn, datan, bon, tagsn = read_pixels(N)
foot = [i for i, v in enumerate(pixn) if v == v]
print("footprint cells:", len(foot))
assert len(foot) == 5167373
outsiders = [i for i in dots if pixn[i] != pixn[i]]
print("dots outside footprint (must be 0):", len(outsiders))

# grid hash cell size 8px; nearest-neighbor distance for each dot
CELL = 8
buckets = {}
for idx, lin in enumerate(dots):
    r, c = divmod(lin, w)
    key = (r // CELL, c // CELL)
    buckets.setdefault(key, []).append((r, c, idx))

violations_28 = 0
violations_30 = 0
hist = [0] * 9  # bins: <1.5,<2.0,<2.5,<2.83,<3.0,<3.5,<4,<5,>=5
pairs_checked = 0
for (br, bc), members in buckets.items():
    neigh = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            neigh.extend(buckets.get((br + dr, bc + dc), []))
    for (r, c, idx) in members:
        best = 1e9
        for (r2, c2, idx2) in neigh:
            if idx2 == idx:
                continue
            d = math.hypot(r - r2, c - c2)
            pairs_checked += 1
            if d < best:
                best = d
        if best < 2.8:
            violations_28 += 1
        if best < 3.0:
            violations_30 += 1
        b = 0 if best < 1.5 else 1 if best < 2.0 else 2 if best < 2.5 else 3 if best < 2.83 else 4 if best < 3.0 else 5 if best < 3.5 else 6 if best < 4.0 else 7 if best < 5.0 else 8
        hist[b] += 1

print("nearest-neighbor histogram (px): <1.5,<2.0,<2.5,<2.83,<3.0,<3.5,<4,<5,>=5:")
print(hist)
print("dots with NN < 2.8px:", violations_28)
print("dots with NN < 3.0px:", violations_30)

# zip inner hash
zpath = Path("docs/downloads/gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-zeros.zip")
with zipfile.ZipFile(zpath) as z:
    inner = z.read(z.namelist()[0])
print("zip inner sha256 == zeros sha256:", hashlib.sha256(inner).hexdigest() == hashlib.sha256(Z.read_bytes()).hexdigest())

# full geokeys
geo = get_values(data, bo, tags, 34735)
print("geokey dir:", geo)
asc = get_values(data, bo, tags, 34737) if 34737 in tags else None
print("geoascii:", repr(asc)[:200])
