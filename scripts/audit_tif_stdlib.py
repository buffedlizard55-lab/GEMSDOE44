"""Stdlib-only GeoTIFF auditor for GEMSDOE44 (no numpy/rasterio/GDAL needed).

Parses TIFF structure with struct, decompresses strips with zlib, interprets
float32 pixels with the array module. Verifies every competition format rule.
"""
import struct
import zlib
import json
import hashlib
import zipfile
from array import array
from pathlib import Path

TAG_NAMES = {
    256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample",
    259: "Compression", 262: "PhotometricInterpretation",
    273: "StripOffsets", 277: "SamplesPerPixel", 278: "RowsPerStrip",
    279: "StripByteCounts", 284: "PlanarConfiguration",
    339: "SampleFormat", 33550: "ModelPixelScaleTag",
    33922: "ModelTiepointTag", 34735: "GeoKeyDirectoryTag",
    34737: "GeoAsciiParamsTag", 42113: "GDAL_NODATA",
}

TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def parse_tiff(path):
    data = Path(path).read_bytes()
    bo = "<" if data[:2] == b"II" else ">"
    assert struct.unpack(bo + "H", data[2:4])[0] == 42, "not a TIFF"
    ifd_off = struct.unpack(bo + "I", data[4:8])[0]
    n = struct.unpack(bo + "H", data[ifd_off:ifd_off + 2])[0]
    tags = {}
    for i in range(n):
        e = ifd_off + 2 + i * 12
        tag, typ, cnt = struct.unpack(bo + "HHI", data[e:e + 8])
        raw = data[e + 8:e + 12]
        total = TYPE_SIZES[typ] * cnt
        if total <= 4:
            val = raw[:total]
        else:
            off = struct.unpack(bo + "I", raw)[0]
            val = data[off:off + total]
        tags[tag] = (typ, cnt, val)
    return data, bo, tags


def get_values(data, bo, tags, tag):
    typ, cnt, val = tags[tag]
    fmt = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d"}.get(typ)
    if typ == 2:
        return val.rstrip(b"\x00").decode("ascii", "replace")
    if typ == 5:  # RATIONAL pairs
        out = []
        for i in range(cnt):
            num, den = struct.unpack(bo + "II", val[i * 8:(i + 1) * 8])
            out.append(num / den if den else float("nan"))
        return out
    return list(struct.unpack(bo + fmt * cnt, val))


def read_pixels(path):
    data, bo, tags = parse_tiff(path)
    w = get_values(data, bo, tags, 256)[0]
    h = get_values(data, bo, tags, 257)[0]
    bps = get_values(data, bo, tags, 258)[0]
    comp = get_values(data, bo, tags, 259)[0]
    spp = get_values(data, bo, tags, 277)[0] if 277 in tags else 1
    rps = get_values(data, bo, tags, 278)[0]
    offsets = get_values(data, bo, tags, 273)
    bytecounts = get_values(data, bo, tags, 279)
    sfmt = get_values(data, bo, tags, 339)[0] if 339 in tags else 1
    assert bps == 32 and sfmt == 3, f"expected float32, got bps={bps} sfmt={sfmt}"
    assert spp == 1, f"expected single band, got {spp}"
    assert comp in (1, 8), f"unexpected compression {comp}"
    pix = array("f")
    for off, bc in zip(offsets, bytecounts):
        chunk = data[off:off + bc]
        if comp == 8:
            chunk = zlib.decompress(chunk)
        a = array("f")
        a.frombytes(chunk)
        pix.extend(a)
    assert len(pix) == w * h, f"pixel count {len(pix)} != {w*h}"
    return pix, w, h, data, bo, tags


def audit(path):
    p = Path(path)
    pix, w, h, data, bo, tags = read_pixels(p)
    n = len(pix)
    n_nan = sum(1 for v in pix if v != v)
    n_posinf = sum(1 for v in pix if v == float("inf"))
    n_neginf = sum(1 for v in pix if v == float("-inf"))
    finite = [v for v in pix if v == v and v not in (float("inf"), float("-inf"))]
    nf = len(finite)
    mn = min(finite) if finite else None
    mx = max(finite) if finite else None
    n_out = sum(1 for v in finite if v < 0.0 or v > 1.0)
    n_pos = sum(1 for v in finite if v > 0.0)
    n_one = sum(1 for v in finite if v == 1.0)
    distinct = sorted(set(finite))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    scale = get_values(data, bo, tags, 33550) if 33550 in tags else None
    tie = get_values(data, bo, tags, 33922) if 33922 in tags else None
    nodata = get_values(data, bo, tags, 42113) if 42113 in tags else None
    geo = get_values(data, bo, tags, 34735) if 34735 in tags else None
    return {
        "file": p.name, "size_bytes": p.stat().st_size, "sha256": sha,
        "width": w, "height": h, "total_cells": n,
        "n_nan": n_nan, "n_posinf": n_posinf, "n_neginf": n_neginf,
        "n_finite": nf, "min_finite": mn, "max_finite": mx,
        "n_outside_0_1": n_out, "n_positive": n_pos, "n_equal_1": n_one,
        "n_distinct_finite": len(distinct),
        "distinct_min10": distinct[:10],
        "pixel_scale": scale, "tiepoint": tie,
        "nodata_tag": nodata, "geokey_dir_len": len(geo) if geo else 0,
        "geokey_dir_head": geo[:16] if geo else None,
    }


if __name__ == "__main__":
    import sys
    results = []
    for arg in sys.argv[1:]:
        r = audit(arg)
        results.append(r)
        print(json.dumps(r, indent=2))
    # zip check
    zpath = Path("docs/downloads/gemsdoe44-h44-multiphysics-euler-margin-40k-20261006T120000Z-507289a8-zeros.zip")
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            print(json.dumps({"zip": zpath.name, "members": names,
                              "member_sizes": [z.getinfo(n).file_size for n in names]}, indent=2))
            inner = z.read(names[0])
            print(json.dumps({"zip_inner_sha256": hashlib.sha256(inner).hexdigest()}, indent=2))
