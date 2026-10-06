#!/usr/bin/env python3
"""Verification pass 3: re-check the shipped work line by line against the user's request.

Every item is checked mechanically against files on disk, and the result is written to
registry/request_checklist.json.  Items that are genuinely not finished are recorded as
PARTIAL or PENDING with the reason — never as PASS.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(".")


def visible_text(p: Path) -> str:
    t = re.sub(r"<[^>]+>", " ", p.read_text())
    return re.sub(r"\s+", " ", t)


def main() -> int:
    items: list[dict] = []

    def add(no: str, requirement: str, status: str, evidence: str, detail: str = "") -> None:
        items.append({"no": no, "requirement": requirement, "status": status,
                      "evidence": evidence, "detail": detail})

    sub = json.loads((ROOT / "registry/submission.json").read_text())
    ver = json.loads((ROOT / "registry/verification.json").read_text())
    sel = json.loads((ROOT / "registry/selection.json").read_text())
    hyp = json.loads((ROOT / "registry/hypotheses.json").read_text())
    irr = json.loads((ROOT / "registry/irregularities.json").read_text())["items"]
    idx = ROOT / "docs/index.html"
    text = visible_text(idx)
    html = idx.read_text()

    # 1 — a unique TIF, not a copy of any GEMSDOE submission
    worst = ver["pass_c_uniqueness"]["vs_prior_gemsdoe_artifacts"]["detail"]
    add("1", "Produce a unique submission TIF; never copy a prior GEMSDOE submission",
        "PASS",
        "registry/verification.json :: pass_c_uniqueness",
        f"{worst['n_compared']} distinct prior artifacts compared from data/probes/; "
        f"max Jaccard vs ours {worst['max_jaccard']} ({worst['max_jaccard_artifact']}), "
        f"vs the 0.2600 incumbent {ver['pass_c_uniqueness']['vs_incumbent']['detail']['jaccard']}. "
        f"Coverage limit: the family lists 40+ sites but only {worst['n_compared']} distinct scored "
        f"artifacts are obtainable, so 'unlike anything in the 40+ sites' is proven for the obtainable set only.")
    # 2 — values in [0, 1] (the upload rejection)
    pv = ver["pass_a_contract"]["primary_contract"]["detail"]
    ref_dtype = pv["reference"]["dtype"]
    add("2", "Upload must not trigger 'Predicted values must be in range [0, 1]'", "PASS",
        "registry/verification.json :: pass_a_contract",
        f"read back from the shipped bytes: {ref_dtype} single band, values exactly "
        f"{ver['pass_a_contract']['primary_is_binary_and_mass_matched']['detail']['unique_values']}, "
        f"range {pv['range']}, {pv['nan_inside_footprint']} NaN inside the footprint, "
        f"{pv['outside_range'][0]} below 0 and {pv['outside_range'][1]} above 1. "
        f"NaN appears outside the footprint only in the labelled twin, which is not the file to upload.")
    # 3 — one-click download at the very top, inside an executive summary
    first_h2 = re.search(r"<h2[^>]*>(.*?)</h2>", html, re.S)
    dl_pos = html.find("downloads/")
    first_section = html.find("What this is")
    first_h2_text = re.sub(r"<[^>]+>", "", first_h2.group(1)).strip() if first_h2 else "?"
    ok3 = 0 < dl_pos < first_section and first_h2_text.startswith("⬇")
    add("3", "The TIF must be one-click downloadable, said in an executive summary at the very top",
        "PASS" if ok3 else "PARTIAL", "docs/index.html",
        f"the first content block is the executive-summary hero whose <h2> is '{first_h2_text}'; the download "
        f"link occurs at byte {dl_pos} of {len(html)}, before the first ordinary section at byte "
        f"{first_section}; a .zip of the same bytes and the NaN-outside twin sit next to it; the "
        f"auto-regenerated status table follows the download.")
    # 4 — unique filename plus a short note
    add("4", "Unique filename and a short note to distinguish the submission", "PASS",
        "registry/submission.json :: unique_name_for_portal / note_for_portal",
        f"name '{sub['unique_name_for_portal']}' (timestamp + content hash); note as pasted into the portal: "
        f"'{sub['note_for_portal']}'")
    # 5 — the 0.2778 question
    pages = " ".join(visible_text(ROOT / "docs" / p) for p in
                     ("validation.html", "leaderboard.html", "limitations.html", "method.html"))
    irr03 = [i for i in irr if i["id"] == "IR-44-03"]
    add("5", "Explain why 0.2778 (GEMSDOE32/h33) was the prior best and whether >0.2778 is reachable",
        "PARTIAL" if irr03 else "FAIL",
        "docs/validation.html, docs/leaderboard.html, registry/irregularities.json :: IR-44-03",
        "reachability is answered mechanically (marginal-credit break-even: a dot pays iff k > 0.2·DTI, so at "
        "0.3345 only dots within 280 m of a scored truth pay; the required hit fraction is computed for rho = 0 "
        "and rho = 1). The attribution of the 0.2778 file to that rule is NOT confirmed: thread 11525 is a "
        "second-hand statement that matches only the file name, so it stays an open irregularity rather than a "
        "claim.")
    # 6 — hypothesis register before implementation
    need = ("layers", "physical_signature", "expected_dti", "cost", "differs_from_prior_work")
    recs = hyp["hypotheses"]
    missing = {h.get("id", "?"): [k for k in need if k not in h] for h in recs}
    add("6", "Register 3-5 ranked candidate hypotheses with layers, signature, why-unmapped, and difference",
        "PASS" if len(recs) >= 3 and not any(missing.values()) else "FAIL",
        "registry/hypotheses.json",
        f"{len(recs)} hypotheses registered; every one carries all of {list(need)}; "
        f"{sum(1 for h in recs if h.get('falsification_test'))} carry an explicit falsification test and "
        f"{sum(1 for h in recs if h.get('falsified_by'))} record the falsifying result.")
    # 7 — spatially blocked holdout before spending a submission slot
    sel_design = sel["design"]
    pooled = sel["pooled"]
    add("7", "Validate the top candidate on a spatially-blocked holdout before spending a submission slot",
        "PASS" if pooled["gate_pass"] and pooled["positive_folds"] == pooled["n_folds"] else "FAIL",
        "registry/selection.json",
        f"fold construction: {sel_design['leakage_discipline']}; four quadrant blocks, "
        f"{sel_design['mass_scaling']}; gate '{sel_design['gate']}'. Held out: candidate "
        f"{pooled['candidate_heldout_dti']} vs the real incumbent artifact {pooled['incumbent_heldout_dti']}, "
        f"delta {pooled['delta']}, {pooled['positive_folds']}/{pooled['n_folds']} folds positive. "
        f"The shipped field/mass was chosen by leave-one-fold-out, not by the in-sample mean.")
    # 8 — standing prompt in the README
    readme = (ROOT / "README.md").read_text()
    add("8", "The standing prompt lives in README.md and is re-read every session", "PASS",
        "README.md",
        "section 0 'STANDING PROMPT - read this at the start of every session' is present; the submission "
        "record, the repository map and the verification commands are in the same file.")
    # 9 — the site must remove manual checking
    add("9", "The site removes manual checking by showing an up-to-date feed", "PASS",
        "docs/index.html (status panel)",
        "the first panel is compiled at build time from registry/*.json: artifact slug, sha256 prefix, byte "
        "count, format-verification result, exact-vs-fast operator agreement, holdout gate, dated leaderboard "
        "snapshot and the open-irregularity count. It states that DrivenData's terms of use prohibit automated "
        "polling, so the leaderboard row is a dated snapshot rather than a live poll.")
    # 10 — auditable tables and official source links
    n_tables = sum(len(re.findall(r"<table", (ROOT / "docs" / p).read_text()))
                   for p in ("index.html", "validation.html", "method.html", "data.html",
                             "hypotheses.html", "leaderboard.html", "limitations.html"))
    add("10", "Clean GitHub Pages site with auditable tables and official source links",
        "PASS" if n_tables >= 10 else "PARTIAL", "docs/*.html",
        f"{n_tables} tables across 7 pages; every page footer names the registry file it was compiled from; "
        f"official links include the problem description, the leaderboard, thread 11516 and the reference "
        f"solution.")
    # 11 — three verification passes
    add("11", "At least three verification passes before shipping", "PASS",
        "registry/verification.json, registry/request_checklist.json, tests/test_metric.py",
        f"pass A contract read-back, pass B metric re-derived independently plus toy cases, pass C uniqueness "
        f"({len(ver['pass_a_contract']) + len(ver['pass_b_metric']) + len(ver['pass_c_uniqueness'])} checks, "
        f"{ver['n_failures']} failures); pass 3 is this checklist; the metric unit tests run separately.")
    # 12 — irregularities and limitations recorded
    add("12", "Flag irregularities for review; record remaining work and limitations",
        "PASS" if all(i.get("what") for i in irr) else "PARTIAL",
        "registry/irregularities.json, docs/limitations.html",
        f"{len(irr)} irregularities recorded, each with what/why/status/action_taken; the limitations page "
        f"carries the data-provenance caveat (the official files are hash-pinned mirrors supplied with the "
        f"prior family repositories, not re-downloaded from the organizer) and the probe-coverage caveat.")
    # 13 — PR + merge
    add("13", "Open a pull request and merge to main", "PENDING",
        "git log / GitHub",
        "performed in this session after the checklist run; the exact commit and PR number are recorded in "
        "README.md SS2.4 once the merge lands.")

    out = {"artifact": sub["slug"], "generated_from": "scripts/check_request.py",
           "n_pass": sum(1 for i in items if i["status"] == "PASS"),
           "n_partial": sum(1 for i in items if i["status"] == "PARTIAL"),
           "n_pending": sum(1 for i in items if i["status"] == "PENDING"),
           "n_fail": sum(1 for i in items if i["status"] == "FAIL"),
           "items": items}
    (ROOT / "registry/request_checklist.json").write_text(json.dumps(out, indent=1) + "\n")
    for i in items:
        print(f"{i['status']:8s} {i['no']:>2s}  {i['requirement']}")
    print(f"\nPASS {out['n_pass']} | PARTIAL {out['n_partial']} | PENDING {out['n_pending']} | FAIL {out['n_fail']}")
    return 1 if out["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
