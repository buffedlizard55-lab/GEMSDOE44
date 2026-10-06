#!/usr/bin/env bash
# Fetch every official competition raster into data/ and FAIL CLOSED on any digest mismatch.
#
# Why a mirror: the DrivenData data tab is login-walled (verified 2026-10-06: an unauthenticated
# request redirects to the login page).  The bytes used here come from public GitHub repositories
# owned by the same account that owns this repository, and every byte is pinned by sha256 in
# registry/data_manifest.json.  A pin proves reproducibility, NOT organizer authentication --
# registered as IR-44-02 in registry/irregularities.json.
#
# Requires: `gh` authenticated for github.com (or GH_TOKEN in the environment).
# drivendata.org is never contacted by this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw}"
MANIFEST="$ROOT/registry/data_manifest.json"
mkdir -p "$DEST" "$ROOT/data/external"

python3 - "$MANIFEST" "$DEST" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path

manifest, dest = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads(manifest.read_text())
fail = []

def gh(repo, ref, path, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
                        "-H", "Accept: application/vnd.github.raw"], stdout=fh, check=True)

for f in spec["files"]:
    target = dest.parent / f["dest"]
    if target.exists():
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got == f["sha256"] and target.stat().st_size == f["bytes"]:
            print(f"PRESENT {f['id']:32s} {f['bytes']:>12,} bytes {got[:16]}...")
            continue
    if "parts" in f:
        chunks = []
        for i, p in enumerate(f["parts"]):
            c = dest / f"_part-{i:03d}"
            gh(f["repo"], f["ref"], p, c)
            chunks.append(c)
        with target.open("wb") as out:
            for c in chunks:
                out.write(c.read_bytes())
        for c in chunks:
            c.unlink()
    else:
        gh(f["repo"], f["ref"], f["path"], target)
    got = hashlib.sha256(target.read_bytes()).hexdigest()
    n = os.path.getsize(target)
    ok = (got == f["sha256"]) and (n == f["bytes"])
    print(("PASS " if ok else "FAIL ") + f"{f['id']:32s} {n:>12,} bytes {got[:16]}...")
    if not ok:
        fail.append(f["id"])

print(json.dumps({"verified": len(spec["files"]) - len(fail), "failed": fail}, indent=1))
sys.exit(1 if fail else 0)
PY
