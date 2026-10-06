#!/usr/bin/env bash
# Fetch the official competition rasters into data/.
#
# Primary path (unrestricted machine): the DrivenData data tab
#   https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# requires a (free) DrivenData login. Download training_features.tif,
# labels.tif, sample_submission.tif and 1m_DEM_links.csv into data/ and
# rename:  labels.tif -> existing_faults.tif, sample_submission.tif ->
# example_submission.tif, training_features.tif stays as-is.
#
# Fallback path (egress-restricted sandbox): the group's sha256-pinned
# GitHub mirror of that exact tab (each pin names the official dropbox URL):
#   https://github.com/buffedlizard55-lab/GEMSDOE  (data/bridge/)
# This sandbox uses that fallback; verification (sha256 + geometry) is done
# by scripts/run_all.py step 0 and recorded in evidence/verify_rasters.json.
#
# Expected sha256 (from data/bridge/manifest.json):
#   training_features.tif   4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5
#   existing_faults.tif     7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093
#   example_submission.tif  2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
python3 - <<'EOF'
import hashlib, pathlib
pins = {
    "data/training_features.tif": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "data/existing_faults.tif": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "data/example_submission.tif": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
}
ok = True
for p, want in pins.items():
    f = pathlib.Path(p)
    if not f.exists():
        print(f"MISSING {p}"); ok = False; continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    status = "OK " if h == want else "BAD"
    if h != want: ok = False
    print(f"{status} {p} {h[:16]}…")
raise SystemExit(0 if ok else 1)
EOF
