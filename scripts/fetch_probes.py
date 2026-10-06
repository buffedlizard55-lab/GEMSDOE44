"""Fetch a spread of family artifacts whose scores were reported, to rank the LOCAL frames.

Every entry below names a site mirror and a fragment of the artifact name taken from
prior/GEMSDOE32/docs/score-ledger.csv.  The score is the owner-reported live score and is used
ONLY as an ordering to be reproduced - never as a value to be fitted.  Reproducing the order is
what a local promotion frame must do before this project is allowed to select on it.

Writes registry/probe_manifest.json and data/probes/*.tif (gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TARGETS = [
    ("6GEMSDOE", "hgb88-topk03", 0.0286),
    ("GEMSDOE10", "h16-continuation", 0.0461),
    ("13GEMSDOE", "lattice-s5", 0.0904),
    ("GEMSDOE10", "h20-dem10-scarp-thin", 0.0921),
    ("16GEMSDOE", "complexity-prior", 0.0976),
    ("GEMSDOE22", "dti-optimal-emission-6pct", 0.1002),
    ("GEMSDOE3", "pindrop-v4-nodes", 0.1193),
    ("GEMSDOE26", "dilcond-oof-v1", 0.1223),
    ("GEMSDOE10", "ctx-ridge", 0.1280),
    ("12GEMSDOE", "nms3-dem10-scarp", 0.1294),
    ("GEMSDOE23", "arrangement-matched-habitat", 0.1352),
    ("7GEMSDOE", "ridge-top2pct", 0.1461),
    ("GEMSDOE2", "dual-family-union", 0.1560),
    ("GEMSDOE", "gems-submission-20260925T001403Z-7f00890a", 0.1563),
    ("GEMSDOE10", "h28-dotted-ridge", 0.1839),
    ("16GEMSDOE", "baseline-ridges", 0.1855),
    ("20GEMSDOE", "sarnnpu-powerlaw", 0.1890),
    ("19GEMSDOE", "h19-4-multiline", 0.1894),
    ("19GEMSDOE", "powerlaw-budget-multiline", 0.1922),
    ("GEMSDOE27", "topo-gap-closure-t-v2-on-d1-5", 0.2449),
    ("GEMSDOE24", "dotted-h19-5-d1-5", 0.2477),
    ("GEMSDOE24", "dotted-h19-5-d2-8", 0.2600),
]


def gh_json(args: list[str]) -> object:
    return json.loads(subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout)


def gh_raw(repo: str, path: str, out: Path) -> None:
    with out.open("wb") as fh:
        subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}", "-H", "Accept: application/vnd.github.raw"],
                       stdout=fh, check=True)


def find(repo: str, frag: str) -> tuple[str, int] | None:
    for d in ("docs/downloads", "docs", "data/bridge"):
        try:
            items = gh_json([f"repos/{repo}/contents/{d}"])
        except subprocess.CalledProcessError:
            continue
        if not isinstance(items, list):
            continue
        cands = [i for i in items if isinstance(i, dict) and i.get("name", "").endswith(".tif")
                 and frag in i["name"] and "allfinite" not in i["name"]]
        cands.sort(key=lambda i: (not i["name"].endswith("-nan.tif"), i["size"]))
        if cands:
            return f"{d}/{cands[0]['name']}", cands[0]["size"]
    return None


def main() -> int:
    out_dir = Path("data/probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for site, frag, score in TARGETS:
        repo = f"buffedlizard55-lab/{site}"
        local = next((p for p in out_dir.glob("*.tif") if frag in p.name), None)
        if local is not None:
            manifest.append({"repo": repo, "path": str(local), "file": local.name, "fragment": frag, "reported_score": score,
                             "bytes": local.stat().st_size, "status": "already local"})
            print(f"LOCAL  {frag:38s} {local.name}")
            continue
        hit = find(repo, frag)
        if hit is None:
            manifest.append({"repo": repo, "fragment": frag, "reported_score": score, "status": "NOT FOUND"})
            print(f"MISS   {frag:38s} ({repo})")
            continue
        path, size = hit
        dest = out_dir / Path(path).name
        gh_raw(repo, path, dest)
        manifest.append({"repo": repo, "path": path, "file": dest.name, "fragment": frag, "reported_score": score,
                         "bytes": dest.stat().st_size, "status": "fetched", "mirror_size": size})
        print(f"FETCH  {frag:38s} {dest.name} {dest.stat().st_size:,} B  <- {path}")
    manifest.sort(key=lambda m: m["reported_score"])
    Path("registry").mkdir(exist_ok=True)
    Path("registry/probe_manifest.json").write_text(json.dumps(
        {"note": "reported scores are owner-reported live scores (not organizer receipts, IR-44-08); "
                 "they are used as an ORDER only", "probes": manifest}, indent=2, sort_keys=True) + "\n")
    print(f"\n{sum(1 for m in manifest if m['status'] != 'NOT FOUND')}/{len(TARGETS)} probes available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
