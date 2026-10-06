"""Make `gems44` importable from tests/ without an install step (both
test lineages in this repo import the package as `gems44.*`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
