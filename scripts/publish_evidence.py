#!/usr/bin/env python3
"""Copy the evidence JSONs the site needs into docs/data/ (the Pages root).

Run after scripts/run_all.py:
    /home/user/venv/bin/python scripts/publish_evidence.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
DEST = ROOT / "docs" / "data"
DEST.mkdir(parents=True, exist_ok=True)

NAMES = [
    "pipeline_summary.json",
    "verify_rasters.json",
    "grid_stats.json",
    "external_mirrors.json",
    "stage1_holdout.json",
    "stage2_holdout.json",
    "gate_q_selection.json",
    "budget_sweep.json",
    "final_proxy_score.json",
    "mirror_model.json",
    "uniqueness_report.json",
    "dominance_check.json",
]
for n in NAMES:
    src = EV / n
    if src.exists():
        shutil.copyfile(src, DEST / n)
        print("published", n)
    else:
        print("MISSING", n)
# audit receipts (latest primary alias) - index.html links them under
# downloads/, so publish to both locations
DL = ROOT / "docs" / "downloads"
DL.mkdir(parents=True, exist_ok=True)
for p in sorted(EV.glob("audit-*.json")):
    shutil.copyfile(p, DEST / p.name)
    shutil.copyfile(p, DL / p.name)
print("done")
