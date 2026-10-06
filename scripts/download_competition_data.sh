#!/usr/bin/env bash
# Official competition download helper script
# Note: DrivenData requires user authentication and competition acceptance.
# When running on an unrestricted local machine with your DrivenData session:
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)"
mkdir -p "$DATA_DIR"

echo "=== DOE GEMS Prize Challenge: Data Downloader ==="
echo "Official data URL: https://www.drivendata.org/competitions/306/competition-doe-gems/data/"
echo "Please download the following competition files into $DATA_DIR:"
echo "  1. training_features.tif"
echo "  2. labels.tif"
echo "  3. sample_submission.tif"
echo "  4. 1m_DEM_links.csv"
echo ""
echo "Alternative community Dropbox links provided in the charter:"
echo "  - example_submission.tif: https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&dl=1"
echo "  - existing_faults.tif: https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&dl=1"
echo "  - numerical_features.tif: https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&dl=1"
echo ""
echo "Place files in $DATA_DIR and run: python scripts/prepare_data.py"
