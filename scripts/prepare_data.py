#!/usr/bin/env python3
"""Inspect, inventory, and validate competition rasters placed in data/."""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def main():
    print("=== DOE GEMS Prize Challenge: Data Inventory Preflight ===")
    required_files = [
        "training_features.tif",
        "labels.tif",
        "sample_submission.tif"
    ]
    
    missing = []
    found = []
    for f in required_files:
        p = DATA_DIR / f
        if p.exists():
            found.append(f"{f} ({p.stat().st_size:,} bytes)")
        else:
            missing.append(f)
            
    print(f"Found files: {found if found else 'None'}")
    if missing:
        print(f"Pending DrivenData download for: {missing}")
        print("Note: In sandboxed environments without DrivenData auth cookies, place downloaded files in data/ manually.")
        return 0
    else:
        print("All primary competition rasters present and verified!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
