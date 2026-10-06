"""Generate the H44-6 unique submission: kernel-matched D3.0 redundancy prune.

PASS 1 implementation (2026-10-06, session 2). Stdlib-only; fully reproducible:
  inputs  = committed anchor TIFs in docs/downloads/ (verified by audit script)
  rule    = greedy maximal independent set at 3.0 px (300 m = metric kernel R),
            priority = most-isolated dots first, ties broken by (row, col)
  outputs = new zeros.tif + nan.tif + zip + audit.json in docs/downloads/

Why 3.0 px: the competition DTI takes a MAX over each 300 m kernel per truth
pixel, so a second dot inside an already-covered kernel earns ~0 marginal
credit while paying 0.2 mass cost (cost bar k > 0.2*DTI). The anchor file was
measured (analyze_spacing_stdlib.py) to have 11,716 dots with a neighbour in
[2.8, 3.0) px; those are the least-valuable mass under the metric.

Evidence class: decision-theoretic / geometric. UNSCORED candidate. No new
physics claimed; no score claimed or projected.
"""
import hashlib
import json
import math
import struct
import sys
import zipfile
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_tif_stdlib import parse_tiff, get_values  # noqa: E402

ANCHOR_Z = "gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-zeros.tif"
ANCHOR_N = "gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-nan.tif"
TIMESTAMP = "20261006T180000Z"  # fixed -> deterministic filename across re-runs
RADIUS_PX = 3.0
DL = Path(__file__).resolve().parents[1] / "docs" / "downloads"

TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def read_float_pixels(path):
    data, bo, tags = parse_tiff(path)
    assert bo == "<", "writer supports little-endian anchor only"
    assert sys.byteorder == "little", "writer requires little-endian host"
    import zlib as _z
    w = get_values(data, bo, tags, 256)[0]
    h = get_values(data, bo, tags, 257)[0]
    comp = get_values(data, bo, tags, 259)[0]
    assert comp == 8
    rps = get_values(data, bo, tags, 278)[0]
    offs = get_values(data, bo, tags, 273)
    bcs = get_values(data, bo, tags, 279)
    pix = array("f")
    for off, bc in zip(offs, bcs):
        a = array("f")
        a.frombytes(_z.decompress(data[off:off + bc]))
        pix.extend(a)
    assert len(pix) == w * h
    return pix, w, h, rps, data, bo, tags


def raw_tag_payloads(data, bo, tags):
    """Return {tag: (type, count, payload_bytes)} for every tag."""
    out = {}
    for tag, (typ, cnt, val) in tags.items():
        out[tag] = (typ, cnt, bytes(val))
    return out


def write_geotiff(path, w, h, pix, rps, template_payloads, nodata_ascii=None):
    """Write a little-endian deflate float32 GeoTIFF reusing template tag payloads."""
    import zlib as _z
    assert len(pix) == w * h
    payloads = dict(template_payloads)
    if nodata_ascii is not None:
        payloads[42113] = (2, len(nodata_ascii) + 1, nodata_ascii.encode("ascii") + b"\x00")
    # build strips
    nstrips = (h + rps - 1) // rps
    strips = []
    for s in range(nstrips):
        r0 = s * rps
        r1 = min(r0 + rps, h)
        seg = array("f", pix[r0 * w:r1 * w])
        strips.append(_z.compress(seg.tobytes(), 6))
    payloads[273] = (4, nstrips, b"")  # offsets, filled below
    payloads[279] = (4, nstrips, b"")  # bytecounts, filled below
    payloads[278] = (3, 1, struct.pack("<H", rps) if rps < 65536 else struct.pack("<I", rps))
    if rps >= 65536:
        payloads[278] = (4, 1, struct.pack("<I", rps))
    else:
        payloads[278] = (3, 1, struct.pack("<H", rps))
    tag_ids = sorted(payloads.keys())
    n = len(tag_ids)
    ifd_size = 2 + 12 * n + 4
    # assign out-of-line offsets
    blob = bytearray()
    entries = []
    for tag in tag_ids:
        typ, cnt, val = payloads[tag]
        if tag in (273, 279):
            continue  # handled after layout known
        total = TYPE_SIZES[typ] * cnt
        if total <= 4:
            entries.append((tag, typ, cnt, val + b"\x00" * (4 - total)))
        else:
            entries.append((tag, typ, cnt, ("blob", len(blob), total)))
            blob.extend(val)
            if len(val) != total:
                raise AssertionError(f"tag {tag}: payload {len(val)} != {total}")
    # strip chatter: offsets/bytecounts payloads
    nstrips_bytes_off = 8 + ifd_size + len(blob)
    offsets = []
    acc = nstrips_bytes_off + 8 * nstrips  # after the two LONG arrays
    # LONG arrays stored out-of-line right after other blob data
    arr_off = 8 + ifd_size + len(blob)
    blob_off_arr = len(blob)
    bytecounts = [len(s) for s in strips]
    for bc in bytecounts:
        offsets.append(acc)
        acc += bc
    off_bytes = struct.pack("<%dI" % nstrips, *offsets)
    bc_bytes = struct.pack("<%dI" % nstrips, *bytecounts)
    blob.extend(off_bytes)
    blob.extend(bc_bytes)
    entries.append((273, 4, nstrips, ("abs", arr_off)))
    entries.append((279, 4, nstrips, ("abs", arr_off + 4 * nstrips)))
    entries.sort(key=lambda e: e[0])
    # emit
    out = bytearray()
    out.extend(b"II\x2a\x00\x08\x00\x00\x00")
    out.extend(struct.pack("<H", n))
    for tag, typ, cnt, ref in entries:
        if isinstance(ref, bytes):
            out.extend(struct.pack("<HHI", tag, typ, cnt))
            out.extend(ref)
        else:
            kind, a, *_ = ref
            file_off = (8 + ifd_size + a) if kind == "blob" else a
            out.extend(struct.pack("<HHI", tag, typ, cnt))
            out.extend(struct.pack("<I", file_off))
    out.extend(struct.pack("<I", 0))
    assert len(out) == 8 + ifd_size
    out.extend(blob)
    assert len(out) == nstrips_bytes_off + 8 * nstrips
    for s in strips:
        out.extend(s)
    Path(path).write_bytes(bytes(out))


def packbits_big(mask):
    nbytes = (len(mask) + 7) // 8
    buf = bytearray(nbytes)
    for i, b in enumerate(mask):
        if b:
            buf[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(buf)


def main():
    pix, w, h, rps, data, bo, tags = read_float_pixels(DL / ANCHOR_Z)
    pixn, wn, hn, _, _, _, _ = read_float_pixels(DL / ANCHOR_N)
    assert (w, h) == (wn, hn) == (3292, 3730)
    foot = bytearray(1 if v == v else 0 for v in pixn)
    assert sum(foot) == 5167373
    dots = [i for i, v in enumerate(pix) if v > 0.0]
    assert len(dots) == 40000
    assert all(foot[i] for i in dots)

    # full-set nearest-neighbour distances via grid hash
    CELL = 8
    buckets = {}
    for idx, lin in enumerate(dots):
        r, c = divmod(lin, w)
        buckets.setdefault((r // CELL, c // CELL), []).append((r, c, idx))
    nn = [1e9] * len(dots)
    for (br, bc), members in buckets.items():
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neigh.extend(buckets.get((br + dr, bc + dc), []))
        for (r, c, idx) in members:
            best = 1e9
            for (r2, c2, j) in neigh:
                if j == idx:
                    continue
                d = math.hypot(r - r2, c - c2)
                if d < best:
                    best = d
            nn[idx] = best
    print(f"anchor dots=40000 minNN={min(nn):.4f} maxNN={max(nn):.4f}")

    # greedy MIS: most-isolated first, ties by (row, col)
    order = sorted(range(len(dots)), key=lambda i: (-nn[i], dots[i] // w, dots[i] % w))
    KCELL = 4
    kept_buckets = {}
    kept = []

    def has_kept_near(r, c):
        kr, kc = r // KCELL, c // KCELL
        R = int(math.ceil(RADIUS_PX / KCELL)) + 1
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                for (r2, c2) in kept_buckets.get((kr + dr, kc + dc), ()):
                    if math.hypot(r - r2, c - c2) < RADIUS_PX:
                        return True
        return False

    for i in order:
        lin = dots[i]
        r, c = divmod(lin, w)
        if not has_kept_near(r, c):
            kept.append(lin)
            kept_buckets.setdefault((r // KCELL, c // KCELL), []).append((r, c))
    kept_set = set(kept)
    print(f"kept={len(kept)} pruned={40000 - len(kept)}")

    # verify min spacing of output
    kb = {}
    for lin in kept:
        r, c = divmod(lin, w)
        kb.setdefault((r // CELL, c // CELL), []).append((r, c))
    min_out = 1e9
    for (br, bc), members in kb.items():
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neigh.extend(kb.get((br + dr, bc + dc), []))
        for (r, c) in members:
            for (r2, c2) in neigh:
                if r == r2 and c == c2:
                    continue
                d = math.hypot(r - r2, c - c2)
                if d < min_out:
                    min_out = d
    print(f"output minNN={min_out:.4f} (must be >= {RADIUS_PX})")
    assert min_out >= RADIUS_PX - 1e-9

    mask = bytearray(len(pix))
    for lin in kept:
        mask[lin] = 1
    digest = hashlib.sha256(packbits_big(mask)).hexdigest()[:8]
    slug = f"gemsdoe44-h44-6-d30-prune-{len(kept)}-{TIMESTAMP}-{digest}"
    print("slug:", slug)

    arr = array("f", [0.0]) * (w * h)
    for lin in kept:
        arr[lin] = 1.0
    template = raw_tag_payloads(data, bo, tags)
    # drop stale strip tags from template (recomputed in writer)
    template.pop(273, None)
    template.pop(279, None)

    z_path = DL / f"{slug}-zeros.tif"
    n_path = DL / f"{slug}-nan.tif"
    write_geotiff(z_path, w, h, arr, rps, template, None)
    arrn = array("f", [float("nan")]) * (w * h)
    for i in range(len(pix)):
        if foot[i]:
            arrn[i] = arr[i]
    write_geotiff(n_path, w, h, arrn, rps, template, "nan")

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    z_sha, n_sha = sha(z_path), sha(n_path)
    zp = DL / f"{slug}-zeros.zip"
    # Fixed ZipInfo date_time => deterministic zip bytes across re-runs.
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        info = zipfile.ZipInfo(z_path.name, date_time=(2026, 10, 6, 18, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, z_path.read_bytes())
    receipt = {
        "submission_name": f"GEMSDOE44-H44-6-D30-PRUNE-{len(kept)}",
        "generator": "scripts/generate_h44_6_prune.py (stdlib-only, deterministic)",
        "parent_anchor": ANCHOR_Z,
        "parent_sha256": hashlib.sha256((DL / ANCHOR_Z).read_bytes()).hexdigest(),
        "rule": f"greedy maximal independent set at {RADIUS_PX} px, priority=most-isolated-first, ties=(row,col)",
        "radius_px": RADIUS_PX,
        "anchor_dots": 40000,
        "kept_dots": len(kept),
        "pruned_dots": 40000 - len(kept),
        "anchor_min_nn_px": min(nn),
        "output_min_nn_px": min_out,
        "filename": z_path.name,
        "sha256": z_sha,
        "size_bytes": z_path.stat().st_size,
        "nan_filename": n_path.name,
        "nan_sha256": n_sha,
        "nan_size_bytes": n_path.stat().st_size,
        "zip_filename": zp.name,
        "zip_sha256": sha(zp),
        "zip_size_bytes": zp.stat().st_size,
        "shape": [h, w],
        "crs": "EPSG:32611",
        "transform": [100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0],
        "dtype": "float32",
        "nodata": None,
        "emitted_positive_pixels": len(kept),
        "in_footprint_finite_pixels": 5167373,
        "in_footprint_min": 0.0,
        "in_footprint_max": 1.0,
        "full_grid_finite_pixels": w * h,
        "evidence_class": "decision-theoretic/geometric; UNSCORED candidate; no score claimed or projected",
        "note": "",
    }
    note = (f"GEMSDOE44 H44-6-D30 | 300m kernel-matched redundancy prune of 40k anchor: "
            f"{len(kept)} dots, NN>=3.0px, in-footprint [0,1] safe; UNSCORED | sha {z_sha[:8]}")
    receipt["note"] = note
    receipt["note_chars"] = len(note)
    with open(DL / f"{slug}-audit.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print("wrote", z_path.name, z_sha[:12], z_path.stat().st_size)
    print("wrote", n_path.name, n_sha[:12], n_path.stat().st_size)
    print("note", f"({len(note)} chars):", note)


if __name__ == "__main__":
    main()
