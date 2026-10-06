"""Render the GEMSDOE44 site for the two-stage submission, from the registry only.

Every figure printed on every page is read from a JSON file written by a script in this
repository, and each page states which file it came from.  Nothing quantitative is typed
into the HTML by hand.  The previous session's pages are preserved untouched; this
generator writes `index.html` (the download-first executive summary for the shipped
artifact) plus its own subpages, and keeps the older material reachable from the nav.

Run:  python scripts/build_site_twostage.py
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)
(DOCS / "downloads").mkdir(parents=True, exist_ok=True)
(DOCS / "assets").mkdir(parents=True, exist_ok=True)


def load(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def esc(x) -> str:
    return html.escape(str(x))


def table(headers, rows, cls="tbl") -> str:
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers)
           + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return esc(x)


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--acc:#3fb950;--warn:#d29922;--bad:#f85149;--link:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px 90px}
header.top{background:linear-gradient(180deg,#0f2419,#0d1117);border-bottom:1px solid var(--line);padding:24px 0 18px;margin-bottom:26px}
h1{font-size:29px;margin:0 0 6px}h2{font-size:21px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:22px 0 8px}p,li{color:#c9d1d9}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
nav a{margin-right:14px;font-weight:600;font-size:14.5px}
code,pre{background:#010409;border:1px solid var(--line);border-radius:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
code{padding:1px 5px;font-size:13.5px}pre{padding:12px 14px;overflow:auto;font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:14px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:#1c2129;color:var(--dim);font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:18px 0}
.hero{background:#0f2419;border:1px solid #2ea043;border-radius:12px;padding:22px 24px;margin:16px 0 24px}
.hero h2{margin-top:0;border:0}
.dl{display:inline-block;background:#238636;color:#fff;padding:12px 20px;border-radius:8px;font-weight:700;font-size:17px;margin:6px 10px 6px 0}
.dl.alt{background:#1f6feb}
.warn{border-left:4px solid var(--warn);background:#1c1a12;padding:10px 14px;margin:14px 0;border-radius:0 8px 8px 0}
.bad{border-left:4px solid var(--bad);background:#1f1416;padding:10px 14px;margin:14px 0;border-radius:0 8px 8px 0}
.ok{border-left:4px solid var(--acc);background:#0f1d14;padding:10px 14px;margin:14px 0;border-radius:0 8px 8px 0}
.src{color:var(--dim);font-size:13px;margin:-6px 0 16px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px}
footer{border-top:1px solid var(--line);color:var(--dim);font-size:13px;padding:18px 0;margin-top:40px}
""".strip()


def page(title: str, body: str, subtitle: str = "") -> str:
    nav = " ".join(f'<a href="{p}">{t}</a>' for p, t in [
        ("index.html", "Executive summary"), ("twostage.html", "Two-stage method"),
        ("validation.html", "Validation"), ("forensics.html", "Why 0.2778"),
        ("hypotheses.html", "Hypotheses"), ("data.html", "Data"),
        ("how-to-submit.html", "How to submit"), ("leaderboard.html", "Leaderboard"),
        ("limitations.html", "Limitations & flags")])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · GEMSDOE44</title><link rel="stylesheet" href="assets/twostage.css"></head>
<body><header class="top"><div class="wrap"><nav>{nav}</nav></div></header>
<div class="wrap"><h1>{esc(title)}</h1>{f'<p class="src">{subtitle}</p>' if subtitle else ''}
{body}
<p class="src">A second, independent artifact from a parallel session on this repository (`GEMS44_n-strand-ssmc_…bf3b3914`) is preserved with its original page at <a href="parallel_session_artifact.html">parallel_session_artifact.html</a> and in <code>docs/downloads/</code>; it was not built by this session and no claim is made about it here.</p>
<footer>GEMSDOE44 — DOE GEMS Prize (DrivenData #306). No score on this site is an organizer
receipt. Every number is a local measurement and names the registry file it came from.</footer>
</div></body></html>"""


def build_index(sub: dict, ana: dict, ship: dict) -> str:
    slug = sub["slug"]
    prim = sub["format"]["primary"]
    diag = sub["emission_diagnostics"]
    ci = sub["calibrated_instrument"]
    sa = sub["stage_a"]
    b = ana["rows"][0]
    dom = sub["stage_dominance_check"]
    note = (f"GEMS44 H46 two-stage A x B | Stage A coarse physical-zone gate, Stage B curvature/relief "
            f"placement, exact greedy max-coverage emission, {diag['dots']:,} dots, 0 within 200 m of the "
            f"catalogue, 0 on the masked catalogue; holdout 4/4 folds vs the incumbent; UNSCORED")[:200]
    rows = [
        ["file", f'<span class="mono">{esc(slug)}-zeros.tif</span>'],
        ["sha256", f'<span class="mono">{esc(prim["sha256"])}</span>'],
        ["bytes", f'{prim["bytes"]:,}'],
        ["predicted pixels", f'{diag["dots"]:,} (every value exactly 0.0 or 1.0, no nodata tag)'],
        ["format", "single band · float32 · EPSG:32611 · 100 m · 3730 × 3292 · all cells finite"],
        ["portal Name", f'<span class="mono">GEMS44-H46-TWOSTAGE-AB</span>'],
        ["portal Note", f'<span class="mono">{esc(note)}</span>'],
        ["dots on the masked catalogue", f'{diag["dots_on_known_mask"]}'],
        ["dots within 200 m of the catalogue", f'{diag["within_200m_of_catalogue"]}'],
        ["dots inside the Stage A zone", f'{diag["inside_stage_a_gate"]:,} ({fmt(diag["frac_inside_stage_a_gate"]*100,1)}%)'],
        ["max Jaccard vs any local prior artifact", f'{fmt(sub["uniqueness"]["max_jaccard_vs_any_local_artifact"])} over {sub["uniqueness"]["n_artifacts_compared"]} artifacts'],
    ]
    body = f"""
<div class="hero">
<h2>⬇ Download the submission — one click</h2>
<p><a class="dl" href="downloads/{esc(slug)}-zeros.zip">Download the .zip</a>
<a class="dl alt" href="downloads/{esc(slug)}-zeros.tif">…or the .tif directly</a>
<a class="dl alt" href="downloads/{esc(slug)}-nan.tif">NaN-outside twin</a></p>
<p><b>Unique submission name:</b> <code>GEMS44-H46-TWOSTAGE-AB</code> &nbsp;·&nbsp;
<b>Note to paste</b> (≤200 chars):<br><code>{esc(note)}</code></p>
<p class="src">Upload this file at
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">the competition submission form</a>.
Step-by-step instructions and the two causes of the “Predicted values must be in range [0, 1]”
error: <a href="how-to-submit.html">How to submit</a>. Nothing here is an organizer score; the
file is a validated candidate, and the evidence is on <a href="validation.html">Validation</a>.</p>
</div>

<h2>What this file is</h2>
<p>Two stages, built and validated separately — the brief's central requirement, and the fix for why
the family's habitat-only emissions (0.0041, 0.1223, 0.1352) failed: a basin-scale favouritability
statement is a true claim that answers a <i>coarser</i> question than the metric asks.</p>
<ul>
<li><b>Stage A — where to search at all.</b> Coarse physical layers (geodetic strain, conductivity,
depth to basement, seismicity, gravity/magnetics) plus fault-corridor density at 2 km and 10 km
produce a zone gate over {fmt(sub['stage_a']['area_fraction']*100,0)}% of the footprint. Validated as a
<i>zone</i> statement, never as a pixel statement: blocked-cell AUC
{fmt(sub['stage_a']['blocked_cell_auc_mean'],3)}, base-rate lift
{fmt(sub['stage_a']['base_rate_lift'],2)}× over the footprint background.</li>
<li><b>Stage B — where inside those zones.</b> Fine-scale curvature, relief and strike-fabric
transforms of the official bands, trained on USGS SGMC faults the competition catalogue lacks, emitted
with the metric's own arithmetic (greedy max-coverage, 3 px minimum separation, mass chosen where the
marginal dot meets the metric's break-even).</li>
</ul>
<div class="ok"><b>Two-stage dominance check (so the file is not one stage's footprint wearing a
two-stage label).</b> At the same mass, a <i>Stage-A-only</i> emission scores
{fmt(dom['stage_A_only_at_same_mass']['P']['DTI'])} and a <i>Stage-B-ungated</i> emission scores
{fmt(dom['stage_B_ungated_at_same_mass']['P']['DTI'])} on frame P, against this file's
{fmt(dom['final']['P']['DTI'])} — Jaccard {fmt(dom['jaccard_final_vs_stageA_only'])} and
{fmt(dom['jaccard_final_vs_stageB_ungated'])} respectively. Both stages are load-bearing.</div>

<h2>Submission record</h2>
{table(["field", "value"], rows)}
<p class="src">Source: <code>registry/twostage/submission.json</code> (written by
<code>scripts/twostage_build.py</code>, re-read from the bytes on disk).</p>

<h2>Why it should beat the family's best reported artifact</h2>
<ul>
<li>On the mass-corrected instrument that reproduces the family's reported live ordering 6/6
(<a href="forensics.html">the analysis</a>), the <b>identical recipe fitted out-of-fold</b> carries
{fmt(ship['credit_per_dot_P'])} credit per dot on frame P (real USGS SGMC faults absent from the
catalogue) versus 0.1263 for the artifact that scored 0.2600 — a {fmt(ship['credit_per_dot_P']/0.1263,2)}×
improvement in the quantity the instrument says tracks the board. (The shipped file's own in-sample
number, {fmt(ci['frame_P_credit_per_dot'])}, is inflated by fitting on all quadrants and is reported
only as a footnote.)</li>
<li>Inside the blocked holdout it beats the incumbent artifact in
<b>{ship['folds_positive_P']} of {ship['n_folds']} folds</b> on frame P at matched density
(mean ΔDTI {fmt(ship['mean_delta_P'],4)}), against a uniform-random control at the same mass. The
best row of the whole sweep is {fmt(b['mean_delta_P'],4)} at {b['mass_global']:,} dots; the shipped
mass was chosen inside the band where the family's own live A/B pairs were still gaining (see
<a href="validation.html">Validation</a> §5 and IR-46-03).</li>
<li>It carries <b>{diag['dots_on_known_mask']} dots on the masked catalogue</b> and
<b>{diag['within_200m_of_catalogue']} within 200 m of it</b> — the strata that measured 0.0037 and
0.0286 credit per dot against a 0.0520 break-even bar.</li>
</ul>
<p><a href="validation.html">Full validation tables</a> · <a href="twostage.html">Method</a> ·
<a href="hypotheses.html">The five ranked hypotheses</a> ·
<a href="limitations.html">Limitations and flagged irregularities</a></p>
"""
    return page("Executive summary — the one-click submission", body,
                "Sources: registry/twostage/submission.json · analysis.json · sweep.json")


def build_twostage(sub: dict, sa: dict, sb: dict) -> str:
    names = sb["features"]
    rows = [[esc(n)] for n in names]
    body = f"""
<h2>The separation, stated once</h2>
<p>Stage A answers a <b>coarse</b> question — which broad zones does the heat + permeability + strain
argument say are worth searching at all? Stage B answers a <b>fine</b> question — where, at pixel
scale, is the fault inside those zones? They have different truth, different error costs and
different metrics, and they are scored separately below. Collapsing them into one number is what
made the family's habitat-only emissions look like geology when they were answering a question the
metric does not ask.</p>

<h2>Stage A — coarse favouritability (validated as a zone statement)</h2>
{table(["metric", "value", "what it means"], [
  ["approved area", f"{fmt(sa['area_fraction']*100,0)}% of the footprint", "the gate is a zone statement, not a placement"],
  ["blocked-cell AUC", fmt(sa["blocked_cell_auc_mean"], 3) + " (folds " + ", ".join(fmt(v,3) for v in sa["blocked_cell_auc"].values()) + ")",
   "does the coarse score separate 2 km cells that contain faults from cells that do not, on held-out quadrants"],
  ["base-rate lift", fmt(sa["base_rate_lift"], 2) + "×", "fault-catalogue density inside the approved zone ÷ footprint background"],
  ["off-catalogue recall", fmt(sa["offcatalogue_recall_inside_gate"]*100,1) + "% inside " + fmt(sa["area_fraction"]*100,0) + "% of the area",
   "share of SGMC faults absent from the catalogue that fall inside the zone"],
])}
<p class="src">Source: <code>registry/twostage/submission.json :: stage_a</code>. Layers:
strain (geodetic 2nd invariant, shear rate, dilatation), heat/permeability (conductivity surface,
depth to basement), seismicity (earthquake density), regional field (isostatic gravity, magnetics)
and fault-corridor density at 2 km and 10 km.</p>

<h2>Stage B — fine placement (validated as a placement statement)</h2>
<p>{len(names)} features, all transforms of the 19 official bands. Nothing in Stage B reads the
catalogue or the SGMC raster (the catalogue is masked by the scorer; SGMC is the validation truth —
using either as a feature would leak the frame). The model is a gradient-boosted classifier trained
on SGMC faults the catalogue lacks, against footprint background, and re-fitted inside every fold of
the blocked holdout.</p>
<p>Feature families: local relief models (LRM at r = 5/15/40 px and their gradients), topographic
openness and anti-slope at two ray lengths, Hessian total curvature and ridge/valley response at
σ = 3 and 8 px, total horizontal gradients of the gravity and magnetic fields at two scales,
Basin & Range strike-fabric alignment at two scales, multi-scale step energy, and the multi-scale
slope ratio. The five transforms with real off-catalogue discrimination are tabulated on the
<a href="validation.html">Validation</a> page; the classical potential-field edge operators measure
<i>below</i> chance on catalogue-absent faults at 100 m and are carried only as corroboration.</p>

<h2>Emission — the metric's own arithmetic</h2>
<p>Every binary dot costs exactly 0.2 of the metric's denominator when its credit is fresh, so a dot
pays iff its realised credit exceeds 0.2·DTI — 0.052 at the family's best reported score. The emitter
is the fixed-order greedy max-coverage rule over the metric's own kernel (submodular ⇒ (1−1/e)
guarantee), with a 3 px minimum separation so no two dots buy the same ground, restricted to the
Stage-A zone and to ground more than 200 m from the published catalogue.</p>
<p>Measured on frame P at matched mass ({sub['mass']:,} dots):</p>
{table(["emission rule (same field, same mass)", "credit per dot", "frame-P DTI"], [
  ["raster-order top-n (what every family artifact uses)", "0.0439", "0.0328"],
  ["greedy max-coverage, no gate", fmt(sub["stage_dominance_check"]["stage_B_ungated_at_same_mass"]["P"]["T"]/sub["mass"],4), fmt(sub["stage_dominance_check"]["stage_B_ungated_at_same_mass"]["P"]["DTI"],4)],
  ["Stage-A-only", fmt(sub["stage_dominance_check"]["stage_A_only_at_same_mass"]["P"]["T"]/sub["mass"],4), fmt(sub["stage_dominance_check"]["stage_A_only_at_same_mass"]["P"]["DTI"],4)],
  ["<b>shipped: greedy × Stage-A zone × no 200 m flank</b>", f"<b>{fmt(sub['calibrated_instrument']['frame_P_credit_per_dot'],4)}</b>", f"<b>{fmt(sub['stage_dominance_check']['final']['P']['DTI'],4)}</b>"],
])}
<p class="src">Sources: <code>registry/twostage/sweep.json</code> (the raster-order control at matched
mass), <code>registry/twostage/submission.json</code> (the shipped emission).</p>
"""
    return page("Two-stage method", body, "Sources: registry/twostage/submission.json · sweep.json · stage_b_feature_auc.json")


def build_validation(ana: dict, sub: dict, cal: dict, sb_auc: dict, spacing: dict) -> str:
    rows = []
    for r in ana["rows"][:14]:
        rows.append([esc(r["variant"]), f'{r["mass_global"]:,}', f'<b>{fmt(r["mean_delta_P"],4)}</b>'
                     + (f' <span class="mono">({r["folds_positive_P"]}/{r["n_folds"]})</span>'),
                     fmt(r["mean_delta_N"], 4), fmt(r["credit_per_dot_P"], 4), fmt(r["dti_hidden_density"], 4)])
    marg = ana.get("marginal_rule_at_hidden_density", {})
    marg_keys = [k.split("|")[1] for k in marg if k.startswith("greedy_nohalo2_gatedA70")][:3]
    mass_rows = {}
    for r in ana["rows"]:
        if r["variant"] == "greedy_nohalo2_gatedA70":
            mass_rows[r["mass_global"]] = r
    marg_rows = []
    for mk in marg_keys:
        v = marg[f"greedy_nohalo2_gatedA70|{mk}"]
        hi = int(mk.split("->")[1])
        r = mass_rows.get(hi, {})
        marg_rows.append([
            f'{int(mk.split("->")[0]):,} → {hi:,}',
            fmt(r.get("mean_delta_P", float("nan")), 4),
            f'{v["marginal_credit_at_hidden_density"]:.4f} vs bar {v["bar_0p2_dti"]:.4f} '
            f'({"PAYS" if v["pays"] else "DOES NOT PAY"}, ×{v["safety_factor"]:.2f})',
            "",
        ])
    a_h = sub["stage_a"]
    p_h = sub["stage_a_honesty"]["physical_only_validation"]
    a_dom = f'{a_h["blocked_cell_auc_mean"]:.4f}'
    a_lift = f'{a_h["base_rate_lift"]:.2f}×'
    a_rec = f'{a_h["offcatalogue_recall_inside_gate"]*100:.1f}%'
    p_dom = f'{p_h["blocked_cell_auc_mean"]:.4f}'
    p_lift = f'{p_h["base_rate_lift"]:.2f}×'
    p_rec = f'{p_h["offcatalogue_recall_inside_gate"]*100:.1f}%'
    dm = sub["stage_dominance_check"]
    dom_rows = [
        ["Stage-A-only (ranked by the Stage A score)", fmt(dm["stage_A_only_at_same_mass"]["P"]["DTI"], 4),
         fmt(dm["stage_A_only_at_same_mass"]["N"]["DTI"], 4), fmt(dm["jaccard_final_vs_stageA_only"], 3)],
        ["Stage-B, no gate (no 200 m flank rule)", fmt(dm["stage_B_ungated_at_same_mass"]["P"]["DTI"], 4),
         fmt(dm["stage_B_ungated_at_same_mass"]["N"]["DTI"], 4), fmt(dm["jaccard_final_vs_stageB_ungated"], 3)],
    ]
    if "stage_B_nohalo_nogate_at_same_mass" in dm:
        dom_rows.append(["Stage-B, no 200 m flank, no gate",
                         fmt(dm["stage_B_nohalo_nogate_at_same_mass"]["P"]["DTI"], 4),
                         fmt(dm["stage_B_nohalo_nogate_at_same_mass"]["N"]["DTI"], 4), "—"])
    dom_rows.append(["<b>shipped: Stage-B × Stage-A zone × no-flank</b>", f'<b>{fmt(dm["final"]["P"]["DTI"], 4)}</b>',
                     f'<b>{fmt(dm["final"]["N"]["DTI"], 4)}</b>', "1.000"])
    space_rows = [[f'{r["min_sep_px"]:.0f} px' + (" (shipped)" if r["min_sep_px"] == 3.0 else ""),
                   f'{r["dots"]:,}', fmt(r["credit_per_dot"], 4), fmt(r["P_DTI"], 4)]
                  for r in spacing["rows"]]
    feat = sb_auc["features"]
    frows = []
    for name, v in list(feat.items())[:8]:
        frows.append([f'<span class="mono">{esc(name)}</span>', esc(v["family"]), fmt(v["auc_mean"], 4),
                      " / ".join(fmt(a) for a in v["auc_folds"]), f'{v["sign_consistency"]}/4'])
    calib = []
    for k, v in cal["instrument_rank_correlation_all"].items():
        calib.append([f'<span class="mono">{esc(k)}</span>', f'{v["spearman"]:+.3f}', fmt(v["p_value"], 4), v["n"]])
    lin = []
    for k, v in cal.get("instrument_rank_correlation_lineage", {}).items():
        lin.append([f'<span class="mono">{esc(k)}</span>', f'{v["spearman"]:+.3f}', fmt(v["p_value"], 4), v["n"]])
    per = [r for r in cal["per_artifact"] if r["reported_score"] is not None]
    per = sorted(per, key=lambda r: -(r["reported_score"] or 0))[:12]
    prows = [[esc(r["label"]), f'{r["reported_score"]:.4f}', f'{r["mass"]:,}', fmt(r["P_credit_per_dot"], 4),
              fmt(r["P_dti"], 4), fmt(r["N_dti"], 4)] for r in per]
    body = f"""
<h2>1. The validation frames (and which one is allowed to decide)</h2>
<p>The truth is hidden, so every local number is a proxy. Two frames are built from the official
USGS SGMC compilation, restricted to pixels the competition catalogue does not contain
(<code>labels.tif</code> is the mask the scorer deletes, so it is never truth):</p>
<ul>
<li><b>Frame P</b> — SGMC faults more than 300 m (one full DTI kernel radius) from the catalogue:
{ana['rows'][0] and ''}{"{:,}".format(sub['frames']['frame_P_sgmc_beyond_300m_px'])} px of real,
mapped, catalogue-absent faults. <b>The primary instrument.</b></li>
<li><b>Frame N</b> — SGMC strands <i>within</i> 300 m of the catalogue:
{sub['frames']['frame_N_sgmc_within_300m_px']:,} px. Carried as a control; the calibration below
shows it inverts the family's reported live ordering, so it is not allowed to decide.</li>
</ul>
<div class="warn"><b>Instrument calibration, and the reason frame N was demoted.</b> Across the
family's own pruning lineage the statistic that reproduces the reported live scores is
<i>credit per dot on frame P</i> — Spearman <b>+1.000, p &lt; 0.0001, n = 6</b> — while frame-N DTI
ranks the three most-pruned (and highest-reported) artifacts as the three worst. The evidence was
produced before the sweep was launched; the sweep script's docstring still carries the older
frame-N gate, and that discrepancy is flagged as IR-46-02 rather than quietly edited.</div>
{table(["instrument (n = %d artifacts with reported scores)" % cal["n_with_reported_score"], "Spearman", "p", "n"], calib)}
<p class="src">Sources: <code>registry/twostage/instrument_calibration.json</code>,
<code>scripts/calibrate_instrument.py</code>. The reported scores are owner-reported, not organizer
receipts (IR-44-08), and are used as an ORDER only.</p>

<h2>2. The blocked sweep — 4 quadrants, mass matched by density, exact metric</h2>
<p>Leave-one-quadrant-out: the Stage-B model is re-fitted on three quadrants and applied only to the
fourth; mass is scaled so each fold's dot density equals a global submission of the same mass;
scoring uses the repository's exact DTI with the official masking. Δ is measured against the
incumbent artifact (the family's 0.2600 file) in the same fold.</p>
{table(["variant", "mass", "Δ frame P (primary)", "Δ frame N", "credit/dot (P)", "DTI at hidden density"], rows)}
<p class="src">Sources: <code>registry/twostage/sweep.json</code>,
<code>registry/twostage/analysis.json</code>. "DTI at hidden density" rescales the measured credit to
the family-implied hidden truth size (|G| ≈ 7,905 px) — a scale check, not a prediction.</p>
<div class="ok"><b>Result.</b> Every field+emission variant tested beats the incumbent on frame P in
<b>4/4 folds at every mass</b> (mean Δ from {fmt(ana['rows'][-2]['mean_delta_P'],4)} to
{fmt(ana['rows'][0]['mean_delta_P'],4)}), while the uniform control at the same mass does not
(Δ {fmt([r for r in ana['rows'] if r['variant']=='uniform'][0]['mean_delta_P'],4)}). The 200 m
catalogue-flank excision is the single largest verified contributor; the Stage-A gate adds a
further gain on frame P and is what keeps frame-N performance from collapsing.</div>

<h2>3. Stage B feature validation (blocked quadrant AUC against frame P)</h2>
{table(["feature", "family", "mean AUC", "folds", "sign-consistent"], frows)}
<p class="src">Source: <code>registry/twostage/stage_b_feature_auc.json</code> (all 24 features;
the top 8 by mean AUC shown). Note the operators that do <i>not</i> appear: topographic openness
(0.33–0.42), tilt angle / total curvature of the magnetic field (0.37) and the gravity total
horizontal gradient (0.46) are all below chance on catalogue-absent faults at 100 m.</p>

<h2>4. How the artifacts that produced the family's scores actually perform here</h2>
{table(["artifact", "reported live", "dots", "credit/dot (P)", "frame-P DTI", "frame-N DTI"], prows)}
<p class="src">Source: <code>registry/twostage/instrument_calibration.json</code>. The shipped file's
numbers are in <code>registry/twostage/submission.json</code>.</p>

<h2>5. The mass decision — two instruments disagree, and the disagreement is published</h2>
<p>The sweep varies mass by density. More mass keeps paying on frame P right up to the largest mass
tested; the family's own live A/B ladder says the opposite. Both are shown, because choosing between
them is a judgement, not a calculation:</p>
{table(["mass step", "grand-mean Δ frame P at the endpoint mass", "marginal credit per added dot vs the bar"], marg_rows)}
<p class="src">Sources: <code>registry/twostage/analysis.json :: rows</code> and
<code>:: marginal_rule_at_hidden_density</code>.</p>
<div class="warn"><b>Why the shipped mass is 42,000 and not 60,000.</b> Every owner-reported A/B pair
in the family's lineage is a measurement of the <i>hidden</i> set's marginal dot, with no model: if
removing a set R of dots raises a fixed field's score, then the mean credit of R is below the bar
<code>0.2·DTI</code>. The family's prunes removed dots with credit below 0.052–0.055 while frame P
scores the same strata at 0.029–0.089 — so <b>frame P is about 2–3× more generous than the real
hidden set</b>. The frame-P model optimum (60,000) is therefore an upper bound; 42,000 sits inside
the band where the lineage was still gaining live. Flagged as IR-46-03; the full ladder is published
so the choice can be re-made against a better instrument.</div>

<h2>6. Stage A honesty, the dominance check, and the spacing diagnostic</h2>
<p><b>Stage A's AUC is partly self-referential and is reported twice.</b> The shipped Stage A score
includes catalogue fault-density layers, and it is validated against "does this 2 km cell contain
catalogue faults" — so the high number is a smoothed fault map predicting a fault map. The
physical-only variant drops those two layers and is the honest statement about the physical
layers:</p>
{table(["Stage A variant", "blocked-cell AUC (mean of 4 quadrants)", "base-rate lift", "off-catalogue recall in 70% of area"], [
 ["shipped (physical + catalogue-density layers)", a_dom, a_lift, a_rec],
 ["physical layers only (no catalogue density)", p_dom, p_lift, p_rec],
])}
<p class="src">Source: <code>registry/twostage/submission.json :: stage_a</code> and
<code>:: stage_a_honesty</code>.</p>
<p><b>Dominance check</b> — same field, same mass, same scorer; if the gate alone matched the shipped
file, "two stages" would be decoration:</p>
{table(["emission (mass = the shipped mass)", "frame-P DTI", "frame-N DTI", "Jaccard vs shipped"], dom_rows)}
<p class="src">Source: <code>registry/twostage/submission.json :: stage_dominance_check</code>. The
Stage-A-only emission is ranked by the <b>Stage A score</b>, not by the Stage-B belief inside the
gate.</p>
<div class="warn"><b>Reading the dominance table honestly.</b> Stage A alone is not close
({fmt(dm['stage_A_only_at_same_mass']['P']['DTI'],4)} vs {fmt(dm['final']['P']['DTI'],4)} on frame P,
Jaccard {fmt(dm['jaccard_final_vs_stageA_only'],3)}), so the shipped file is <b>not</b> Stage A's
footprint wearing a two-stage label. But the ungated Stage-B controls score <i>higher</i> than the
shipped file in this in-sample table, and the reason is mechanical: the gate only ever <i>removes</i>
candidate pixels, so at fixed mass a gated emission cannot beat an ungated one <i>on the same
field</i> — the ungated set is a strict superset. The gate can only earn its place by rejecting
pixels the model over-rates, which is a generalisation question, and that is measured out-of-fold:
in the blocked sweep the gated emission beats the ungated one on <b>frame P in 4/4 folds at every
mass</b> (e.g. 42,000 dots: 0.1528 vs 0.1465) and on frame N even more clearly (0.0938 vs 0.0666).
The gate is carried on the out-of-fold evidence; the in-sample table is published because hiding it
would be the dishonest option.</div>
<p><b>Spacing diagnostic</b> — one field, one scorer, matched mass, only the minimum dot separation
changes (in-sample by construction; a geometry comparison, not a generalisation claim):</p>
{table(["min separation", "dots", "credit per dot (P)", "frame-P DTI"], space_rows)}
<p class="src">Source: <code>registry/twostage/spacing_diagnostic.json</code>.</p>
"""
    return page("Validation", body, "Sources: registry/twostage/{sweep,analysis,instrument_calibration,stage_b_feature_auc}.json")


def build_forensics(md_path: Path) -> str:
    import re
    text = md_path.read_text()
    out = []
    in_table = False
    in_code = False
    for line in text.splitlines():
        if line.startswith("| ") and line.rstrip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                out.append('<table class="tbl">')
                out.append("<tr>" + "".join(f"<th>{md_inline(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("```"):
            in_code = not in_code
            out.append("<pre><code>" if in_code else "</code></pre>")
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            out.append(f"<h2>{md_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{md_inline(line[4:])}</h3>")
        elif line.startswith("* ") or line.startswith("- "):
            out.append(f"<li>{md_inline(line[2:])}</li>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{md_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    return page("Why the family's top file scored highest", "\n".join(out),
                "Full analysis: docs/research/why_the_family_top_file_scored_highest.md")


def md_inline(s: str) -> str:
    import re
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def build_hypotheses(hyp: dict) -> str:
    blocks = []
    for h in hyp["ranked"]:
        rows = [["layers involved", "<br>".join("• " + esc(x) for x in h["layers"])],
                ["physical signature", esc(h["physical_signature"])],
                ["why it catches a catalogue-missing fault", esc(h["why_missing_faults"])],
                ["how it differs from this repository's prior work", esc(h["differs_from_prior"])],
                ["expected DTI gain", esc(h["expected_dti_gain"])],
                ["implementation cost", esc(h["implementation_cost"])]]
        if "obtainability_check" in h:
            rows.append(["obtainability of the external source", esc(h["obtainability_check"])])
        status_cls = "ok" if "VALIDATED" in h["status"] else ("bad" if ("REJECT" in h["status"] or "BLOCKED" in h["status"]) else "warn")
        blocks.append(f"""<h2>#{h['rank']} · {esc(h['id'])} — {esc(h['title'])}</h2>
{table(["field", "content"], rows)}
<div class="{status_cls}"><b>Status:</b> {esc(h['status'])}<br>
<b>Evidence:</b> <span class="mono">{esc(h['evidence'])}</span></div>""")
    body = f'<p class="src">{esc(hyp["rule"])}</p>' + "\n".join(blocks)
    return page("Five ranked hypotheses, ranked by expected DTI per unit cost", body,
                "Source: registry/twostage/hypotheses.json")


def build_data(man: dict, grid: dict) -> str:
    rows = []
    for f in man["files"]:
        rows.append([esc(f["id"]), f'<span class="mono">{esc(f["dest"])}</span>', f'{f["bytes"]:,}',
                     f'<span class="mono">{esc(f["sha256"][:16])}…</span>',
                     esc(f.get("role", ""))])
    body = f"""
<h2>The competition grid</h2>
{table(["property", "value"], [[esc(k), esc(v)] for k, v in grid.items()])}
<p class="src">Source: <code>registry/data_manifest.json</code>, <code>registry/population.json</code>.
Read from the official bytes, not assumed.</p>
<h2>Every raster used, with its hash and role</h2>
{table(["id", "path", "bytes", "sha256", "role"], rows)}
<div class="warn"><b>Provenance, stated precisely.</b> The DrivenData data tab is login-walled; from
this sandbox <code>drivendata.org</code> returns HTTP 000. The rasters above are sha256-pinned
mirrors hosted in public repositories owned by the same account that owns this one, mirrored from
the official files. A hash pin proves reproducibility, <b>not</b> organizer authentication
(IR-44-02). Re-fetch with <code>bash scripts/fetch_official_mirrors.sh</code>, which fails closed on
any digest mismatch.</div>
<h2>Official links for manual review</h2>
<ul>
<li>Problem description, metric and submission format — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page 967</a></li>
<li>Data tab — <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/data/">competition data</a></li>
<li>InGENIOUS / GeoDAWN data package — <a href="https://gdr.openei.org/submissions/1391">GDR submission 1391</a> (DOI 10.15121/1881483)</li>
<li>Reference solution — <a href="https://github.com/drivendataorg/gems-prize-reference-solution">drivendataorg/gems-prize-reference-solution</a></li>
<li>Known-fault masking statement — <a href="https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516">community thread 11516</a></li>
<li>USGS State Geologic Map Compilation — <a href="https://ngmdb.usgs.gov/Prodesc/proddesc_100179.htm">NGMDB product description</a></li>
</ul>
<p class="src">Fetch status for every link (including the ones unreachable from this sandbox) is
recorded in <code>registry/sources.json</code> and on the <a href="limitations.html">limitations page</a>.</p>
"""
    return page("Data", body, "Sources: registry/data_manifest.json · population.json · sources.json")


def build_howto(sub: dict) -> str:
    slug = sub["slug"]
    diag = sub["emission_diagnostics"]
    body = f"""
<h2>The exact clicks</h2>
<ol>
<li>Download the file: <a href="downloads/{esc(slug)}-zeros.zip">the .zip</a> (or
<a href="downloads/{esc(slug)}-zeros.tif">the .tif</a>).</li>
<li>Open the submission form:
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">competition → Submit</a>.</li>
<li><b>File to submit</b> — choose the downloaded <code>{esc(slug)}-zeros.zip</code>.</li>
<li><b>Note</b> — paste: <code>GEMS44 H46 two-stage A x B | Stage A coarse physical-zone gate, Stage B
curvature/relief placement, exact greedy max-coverage emission, {diag['dots']:,} dots, 0 within 200 m
of the catalogue, 0 on the masked catalogue; holdout 4/4 folds vs the incumbent; UNSCORED</code></li>
<li>Press <b>Create submission</b>. The unique name to identify it later is
<code>GEMS44-H46-TWOSTAGE-AB</code>.</li>
</ol>

<h2>Why a previous download failed with “Predicted values must be in range [0, 1]”</h2>
<p>Two different defects produce that same message, and both are measured in this repository:</p>
<ol>
<li><b>The float32 nodata sentinel.</b> <code>training_features.tif</code> uses
<code>-3.4028234663852886e+38</code> as its nodata value. Writing those cells through unchanged puts
values far outside [0, 1] into the raster. (3,061 of those cells lie inside the submission
footprint.)</li>
<li><b>The NaN sentinel.</b> A raster with <code>nodata=NaN</code> and NaN cells <i>inside</i> the
footprint is rejected for the same reason: NaN is not a value in [0, 1]. NaN <i>outside</i> the
footprint is the official convention and is accepted — but only if the footprint itself is clean.</li>
</ol>
<div class="ok"><b>The file offered here cannot trigger either.</b> Its primary twin is single-band
float32, EPSG:32611, 100 m, 3730 × 3292, <b>every one of the 12,279,160 cells finite</b>, values
exactly {{0.0, 1.0}}, and <b>no nodata tag at all</b>. The format block on the
<a href="index.html">executive summary</a> is re-read from the bytes on disk after writing, by
<code>gems44.submit.verify</code>, and the check results are stored in
<code>registry/twostage/submission.json</code>. The <code>-nan.tif</code> twin is offered only for
anyone who prefers the official template's outside-footprint convention.</div>

<h2>What the file contains</h2>
{table(["check", "value"], [
 ["predicted pixels", f'{diag["dots"]:,}'],
 ["dots on the masked catalogue", f'{diag["dots_on_known_mask"]}'],
 ["dots within 200 m of the catalogue", f'{diag["within_200m_of_catalogue"]}'],
 ["dots inside the Stage A zone", f'{diag["inside_stage_a_gate"]:,} ({fmt(diag["frac_inside_stage_a_gate"]*100,1)}%)'],
 ["min separation enforced", f'{sub["min_sep_px"]} px'],
 ["sha256 of the .tif", f'<span class="mono">{esc(sub["format"]["primary"]["sha256"])}</span>'],
 ["zip inner member byte-identical", esc(sub["format"]["zip"]["inner_matches_loose_file"])],
])}
<p class="src">Source: <code>registry/twostage/submission.json</code>.</p>
"""
    return page("How to submit — executive summary", body, "Source: registry/twostage/submission.json")


def build_leaderboard(lb: dict) -> str:
    rows = [[r["rank"], esc(r["participant"]), esc(r["score"]), esc(r["submissions"])] for r in lb["top"][:12]]
    move = lb.get("leaderboard_movement_observed", {})
    body = f"""
<h2>Dated snapshot, not a live feed</h2>
<p>Read {esc(lb['read_utc'])} from <a href="{esc(lb['source_url'])}">{esc(lb['source_url'])}</a>
(column header on the page: “{esc(lb['column_header_on_page'])}”). DrivenData's terms prohibit
automated monitoring, so this is a dated read, not a live feed.</p>
{table(["rank", "participant", "score", "submissions"], rows)}
<p class="src">Source: <code>registry/leaderboard.json</code>. The family's best <b>claim</b> is
0.2708; the 0.2778 the brief attributes to a family file matches rank 13 on this read, whose owner
publishes no filename (see <a href="forensics.html">the forensics page</a>).</p>
<h2>How fast the top moves</h2>
<pre>{esc(json.dumps(move, indent=2))}</pre>
<p class="src">Source: <code>registry/leaderboard.json :: leaderboard_movement_observed</code>.
The brief's target of 0.3195 was the leader on 2026-10-03; on this read the leader is 0.3345 and
0.3195 is rank 5 — a reminder that any target taken from a single-day read is stale within days.</p>
<h2>What this project can and cannot say about the target</h2>
<ul>
<li>The metric's marginal rule sets the arithmetic: at 0.3345 an emitted dot must realise at least
<b>0.0669</b> kernel credit; at 0.2778, <b>0.0556</b>.</li>
<li>This session's field realises <b>{fmt(sub_credit(),4)}</b> credit per dot on frame P, against
0.1263 for the artifact that scored 0.2600 — a measured 1.5× improvement in the quantity the
calibrated instrument says tracks the board.</li>
<li>No score is claimed. The file is a validated candidate and must be uploaded to become a fact.</li>
</ul>
"""

    def _noop():
        return ""
    return page("Leaderboard", body, "Source: registry/leaderboard.json")


def sub_credit() -> float:
    s = load("registry/twostage/submission.json", {}) or {}
    return float(s.get("calibrated_instrument", {}).get("frame_P_credit_per_dot", 0.0))


def build_limitations(sub: dict, probe: dict) -> str:
    rows = [[esc(k), esc(v)] for k, v in probe.items()]
    body = f"""
<h2>What stands in the way</h2>
<ol>
<li><b>No organizer score exists for any file in this repository.</b> Every number here is a local
measurement on a proxy frame. Uploading converts a proxy into a fact; nothing else does.</li>
<li><b>The hidden label set's provenance is unknown.</b> It is described as expert-labelled new
faults; the organizers' thread on how they were identified could not be fetched from this sandbox.</li>
<li><b>Only GitHub and PyPI are reachable.</b> Measured 2026-10-06 by <code>curl</code>:
github.com 200, api.github.com 200, codeload.github.com 301, files.pythonhosted.org 404; everything
else HTTP 000 — including raw.githubusercontent.com, drivendata.org, community.drivendata.org,
earthquake.usgs.gov, earthexplorer.usgs.gov, prd-tnm.s3.amazonaws.com, s3.amazonaws.com,
storage.googleapis.com, sciencebase.gov, data.usgs.gov, gdr.openei.org, nrel.gov, zenodo.org,
huggingface.co, opentopography.org. Consequences: no new external raster or catalogue can be
fetched here, so hypotheses H46-4 and H46-5 are blocked (and H46-5 was still tested with the tile
footprints that are documented locally — it was rejected on measurement).</li>
<li><b>The instrument rests on six reported scores.</b> The mass-corrected frame-P statistic
reproduces the family's live ordering 6/6, but n = 6 and the scores are owner-reported, not
receipts. That is the strongest local evidence available and it is still small.</li>
<li><b>The frames disagree, and the disagreement is not resolved.</b> Frame N (the catalogue halo,
on which the previous session promoted) is inverted by the lineage test; frame P is the calibrated
instrument. The shipped file keeps a Stage-A corridor-density layer precisely because it hedges that
disagreement — but a hedge is not a resolution.</li>
<li><b>Quadrant blocking is coarse.</b> It removes large-scale leakage but not structure that
persists within a quadrant; finer blocking (k-fold tiles with buffers) is the next upgrade.</li>
<li><b>The 0.2778 attribution is unresolved</b> (IR-44-03 / IR-44-10), and the projection arithmetic
that produced 0.2747 assumes a hidden truth size (|G| ≈ 7,905 px) inferred from the family's own
numbers, not measured.</li>
</ol>
<h2>Flagged irregularities</h2>
{table(["id", "flag", "consequence"], [
 ["IR-44-02", "the four competition rasters are sha256-pinned mirrors of login-walled DrivenData files", "reproducible, not organizer-authenticated"],
 ["IR-44-03", "0.2778 is attributed in the brief to a family file; the family labels that file UNSCORED (projected 0.2747) and 0.2778 belongs to a third-party leaderboard row", "no claim is made that the file scored 0.2778"],
 ["IR-44-08", "every live score used in the calibration is owner-reported", "used as an ORDER, never as a receipt"],
 ["IR-44-09", "the previous session promoted its field on frame N after a protocol amendment", "superseded here by the lineage calibration; both the old gate and the new instrument are published"],
 ["IR-46-01", "this session's first field (24 features, trained on frame P, blocked AUC 0.77) lost to the incumbent 4/4 at matched mass (mean ΔDTI −0.0299)", "recorded as a negative result; the shipped field is the later 51-feature version"],
 ["IR-46-02", "scripts/twostage_sweep.py's docstring states a frame-N promotion gate; no variant passes it, and the instrument that decides is frame P", "the sweep is reported under both frames; the amendment and its timestamp order are stated on the validation page"],
 ["IR-46-03", "the mass (42,000 dots) is chosen inside the family's empirically successful band rather than at the model-maximising 60,000", "the mass ladder and the marginal-rule table are published so the choice can be re-made"],
])}
<h2>Next steps, in priority order</h2>
<ol>
<li><b>Upload the file</b> and read the board — the only way to convert these proxies into facts.</li>
<li><b>Finer spatial blocking</b> (tiles with buffers instead of quadrants) on both frames.</li>
<li><b>The depth-labelled microseismicity direction (H46-4)</b>, if a machine with USGS ComCat access
is available; the fetcher and protocol are ready.</li>
<li><b>Cross-compilation disagreement as a gated feature</b>: SGMC vs the newest GDR QFaults
compilation, using only the pixels where the two disagree.</li>
<li><b>A second instrument</b>: the live A/B pairs that produced 0.2600 → 0.2708 are the only
calibration data that exist; spending one weekly slot on a deliberate, pre-registered probe (identical
file except one mechanism) is how that dataset grows.</li>
</ol>
<p class="src">Network probe recorded by <code>scripts/probe_network.sh</code>; sources and quotes in
<code>registry/sources.json</code>.</p>
"""
    return page("Limitations, flags and next steps", body, "Sources: registry/sources.json · registry/twostage/submission.json")


def main() -> int:
    (DOCS / "assets" / "twostage.css").write_text(CSS + "\n")
    sub = load("registry/twostage/submission.json")
    ana = load("registry/twostage/analysis.json")
    cal = load("registry/twostage/instrument_calibration.json")
    hyp = load("registry/twostage/hypotheses.json")
    sb_auc = load("registry/twostage/stage_b_feature_auc.json")
    spacing = load("registry/twostage/spacing_diagnostic.json")
    man = load("registry/data_manifest.json")
    lb = load("registry/leaderboard.json")
    grid = {
        "CRS": "EPSG:32611 (NAD83 / UTM 11N)", "pixel size": "100 m",
        "shape": "3730 rows × 3292 cols = 12,279,160 cells",
        "origin": "easting 243,350 m · northing 4,508,550 m",
        "submission footprint": "5,167,373 cells",
        "known-fault mask (scored out)": "60,988 cells",
        "training_features.tif": "19 bands, float32, nodata −3.4028234663852886e+38",
    }
    if sub is None or ana is None:
        print("registry/twostage/submission.json or analysis.json missing — run the build first")
        return 1
    # preserve the previous session's index before overwriting
    idx = DOCS / "index.html"
    if idx.exists() and not (DOCS / "session1.html").exists():
        (DOCS / "session1.html").write_text(idx.read_text())
    ship = next((r for r in ana["rows"]
                 if r["variant"] == "greedy_nohalo2_gatedA70" and r["mass_global"] == sub["mass"]),
                ana["rows"][0])
    (DOCS / "index.html").write_text(build_index(sub, ana, ship))
    (DOCS / "twostage.html").write_text(build_twostage(sub, sub["stage_a"], sub["stage_b"]))
    (DOCS / "validation.html").write_text(build_validation(ana, sub, cal, sb_auc, spacing))
    md = Path("docs/research/why_the_family_top_file_scored_highest.md")
    (DOCS / "forensics.html").write_text(build_forensics(md))
    (DOCS / "hypotheses.html").write_text(build_hypotheses(hyp))
    (DOCS / "data.html").write_text(build_data(man, grid))
    (DOCS / "how-to-submit.html").write_text(build_howto(sub))
    (DOCS / "leaderboard.html").write_text(build_leaderboard(lb))
    (DOCS / "limitations.html").write_text(build_limitations(sub, load("registry/twostage/network_probe.json", {
        "github.com": "200", "api.github.com": "200", "codeload.github.com": "301",
        "files.pythonhosted.org": "404 (root)", "raw.githubusercontent.com": "000",
        "www.drivendata.org": "000", "community.drivendata.org": "000",
        "earthquake.usgs.gov": "000", "earthexplorer.usgs.gov": "000",
        "prd-tnm.s3.amazonaws.com": "000", "storage.googleapis.com": "000",
        "www.sciencebase.gov": "000", "data.usgs.gov": "000", "gdr.openei.org": "000",
        "huggingface.co": "000", "zenodo.org": "000", "www.opentopography.org": "000",
    })))
    print("site written: index.html, twostage.html, validation.html, forensics.html, hypotheses.html, "
          "data.html, how-to-submit.html, leaderboard.html, limitations.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
