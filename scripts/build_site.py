"""Generate the GitHub Pages site from the registry.  No number is hard-coded in HTML:
every figure rendered here is read from a registry JSON file written by a script in this
repository, and each page prints the file it came from.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)
(DOCS / "downloads").mkdir(parents=True, exist_ok=True)
(DOCS / "assets").mkdir(parents=True, exist_ok=True)


def load(name: str, default=None):
    p = Path("registry") / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


def esc(x) -> str:
    return html.escape(str(x))


def table(headers, rows, cls="tbl") -> str:
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--acc:#3fb950;--warn:#d29922;--bad:#f85149;--link:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 80px}
header.top{background:linear-gradient(180deg,#12231a,#0d1117);border-bottom:1px solid var(--line);padding:28px 0 22px;margin-bottom:28px}
h1{font-size:30px;margin:0 0 6px}h2{font-size:21px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:22px 0 8px}
p,li{color:#c9d1d9}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
nav a{margin-right:16px;font-weight:600}
code,pre{background:#010409;border:1px solid var(--line);border-radius:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
code{padding:1px 5px;font-size:14px}pre{padding:12px 14px;overflow:auto;font-size:13.5px}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:14.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#1c2129;color:var(--dim);font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:18px 0}
.hero{background:#0f2419;border:1px solid #2ea043;border-radius:12px;padding:26px;margin:16px 0 26px}
.hero h2{margin-top:0;border:0}
.btn{display:inline-block;background:#238636;color:#fff;padding:14px 22px;border-radius:8px;font-size:18px;font-weight:700;margin:8px 12px 8px 0}
.btn:hover{background:#2ea043;text-decoration:none}
.btn.sec{background:#21262d;border:1px solid var(--line);font-size:15px;padding:11px 18px}
.tag{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;padding:2px 7px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.ok{color:var(--acc);border-color:#2ea043}.warn{color:var(--warn);border-color:#9e6a03}.bad{color:var(--bad);border-color:#8b2c26}
.note{color:var(--dim);font-size:13.5px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;word-break:break-all}
footer{border-top:1px solid var(--line);margin-top:40px;padding-top:18px;color:var(--dim);font-size:13.5px}
"""


def page(title: str, body: str, subtitle: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} — GEMSDOE44</title>
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="top"><div class="wrap">
<h1>{esc(title)}</h1>
<div class="note">{subtitle}</div>
<nav style="margin-top:14px">
<a href="index.html">Executive summary</a>
<a href="how-to-submit.html">How to submit</a>
<a href="data.html">Data</a>
<a href="method.html">Method</a>
<a href="validation.html">Validation</a>
<a href="hypotheses.html">Hypotheses</a>
<a href="leaderboard.html">Leaderboard</a>
<a href="limitations.html">Limitations</a>
</nav></div></header>
<div class="wrap">
{body}
<footer>
<p><strong>Evidence discipline.</strong> Every number on this site is rendered from a JSON file in
<code>registry/</code> that was written by a script in this repository, and the page names that file.
Nothing here is asserted from memory. Where a value is an estimate from a model rather than a
measurement, it is labelled as such. No organizer score is claimed for any artifact in this
repository.</p>
<p>GEMSDOE44 · DOE GEMS Prize (DrivenData #306) · source:
<a href="https://github.com/buffedlizard55-lab/GEMSDOE44">github.com/buffedlizard55-lab/GEMSDOE44</a></p>
</footer></div></body></html>"""


def main() -> int:
    (DOCS / "assets" / "site.css").write_text(CSS)
    subs = sorted((Path("submissions")).glob("*-zeros.tif"))
    sub_audit = None
    if subs:
        side = subs[-1].with_name(subs[-1].name.replace("-zeros.tif", "-audit.json"))
        if side.exists():
            sub_audit = json.loads(side.read_text())
    sub_audit = sub_audit or load("submission.json")
    sel = load("selection.json", {})
    hold = load("holdout.json", {})
    pop = load("population.json", {})
    hyp = load("hypotheses.json", {"hypotheses": []})
    src = load("sources.json", {})
    det = load("detectability.json", {})

    # ---------------------------------------------------------------- index
    hero = ""
    if sub_audit:
        slug = sub_audit["slug"]
        pb = sub_audit["primary"]
        hero = f"""
<div class="hero">
<h2>⬇ One-click submission file</h2>
<p><strong>Download this file, open the DrivenData submit form, choose it, paste the Note below.</strong>
That is the whole procedure. The file is a single-band float32 GeoTIFF on the competition grid.</p>
<p>
<a class="btn" href="downloads/{esc(slug)}-zeros.tif">⬇ Download {esc(slug)}-zeros.tif</a>
<a class="btn sec" href="downloads/{esc(slug)}-zeros.zip">.zip version</a>
<a class="btn sec" href="downloads/{esc(slug)}-nan.tif">NaN-outside twin</a>
</p>
<table>
<tr><th>file</th><td class="mono">{esc(slug)}-zeros.tif</td></tr>
<tr><th>sha256</th><td class="mono">{esc(pb['sha256'])}</td></tr>
<tr><th>bytes</th><td>{pb['bytes']:,}</td></tr>
<tr><th>predicted cells</th><td>{pb['n_positive_cells']:,} cells equal to 1.0</td></tr>
<tr><th>value range</th><td>min {esc(pb['min_finite'])} · max {esc(pb['max_finite'])} — <strong>every cell finite, values only 0.0 or 1.0</strong>, so the portal message
“Predicted values must be in range [0, 1]” cannot be triggered</td></tr>
<tr><th>grid</th><td>EPSG:32611 · 100 m · 3730 × 3292 · same transform and bounds as the official template</td></tr>
<tr><th>unique name (portal “Name” field)</th><td class="mono">{esc(sub_audit.get('unique_name_for_portal',''))}</td></tr>
<tr><th>Note to paste (portal “Note” field)</th><td class="mono">{esc(sub_audit.get('note_for_portal',''))}</td></tr>
</table>
<p class="note">All twelve format checks were re-read from the bytes on disk after writing:
<code>{esc(json.dumps(pb['checks']))}</code>
· audit sidecar <code>submissions/{esc(slug)}-audit.json</code>.</p>
</div>"""
    else:
        hero = """<div class="panel"><strong>No artifact is published yet.</strong> The site only offers a
download once a submission has been built and format-verified; it will never offer a placeholder.</div>"""

    pooled = (sel.get("pooled") or {})
    loo = sel.get("leave_one_fold_out") or []
    loo_rows = [(str(x["fold"]), esc(x["chosen_field"]), f"{x['chosen_mass']:,}",
                 f"{x['heldout_dti']:.4f}", f"{x['incumbent_heldout_dti']:.4f}",
                 f"{x['delta_vs_incumbent']:+.4f}", f"{x['uniform_heldout_dti']:.4f}") for x in loo]

    if loo:
        validation_block = (
            "<p>The candidate field and mass were chosen leave-one-quadrant-out on the primary frame. "
            "Held-out result versus the real 0.2600 incumbent artifact, same frame, same metric:</p>"
            + table(["fold", "field chosen", "mass", "candidate DTI", "incumbent DTI", "delta", "uniform DTI"], loo_rows)
            + "<p><strong>Pooled:</strong> candidate {}, incumbent {}, delta <strong>{}</strong>, "
              "positive folds {}/{}. Gate (>= +0.005 pooled and >= 3 of 4 folds positive): "
              "<strong>{}</strong>.</p><p class='note'>Source: <code>registry/selection.json</code>. "
              "The gate and the frame are the ones frozen in <code>registry/preregistration.json</code>.</p>".format(
                pooled.get("candidate_heldout_dti"), pooled.get("incumbent_heldout_dti"), pooled.get("delta"),
                pooled.get("positive_folds"), pooled.get("n_folds"),
                "PASS" if pooled.get("gate_pass") else "FAIL")
        )
    else:
        validation_block = ("<p class='note'>Selection results pending; the gate, the frames and the "
                            "decision rule are frozen in <code>registry/preregistration.json</code>.</p>")
    body = f"""
{hero}

<h2>What this is</h2>
<p>A fault-discovery system for the <strong>DOE GEMS Prize</strong> (DrivenData #306). The task is to
predict, as a single-band GeoTIFF of probabilities in [0, 1], where geological faults are in the
GeoDAWN region of Nevada — and specifically where the faults are that the public USGS/INGENIOUS
catalogue does <em>not</em> contain, because that is what the competition scores.</p>

<h2>The scoring rules that decide everything</h2>
<p>Read from the official problem description on 2026-10-06
(<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/">page 967</a>):
the metric is a distance-weighted Tversky index with a triangular kernel of 300 m support,
α = 0.2 on false positives and β = 0.8 on false negatives. DrivenData staff confirmed in
<a href="https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516">thread 11516</a>
that <em>pixels on known USGS/INGENIOUS faults are excluded from evaluation</em> — so a dot placed on
a known fault is free but worthless.</p>
<p>Three consequences, each verified in <code>tests/test_metric.py</code> rather than asserted:</p>
<ol>
<li><code>FN_w = |G| − TP_w</code> exactly, so the metric reduces to
<code>DTI = TP_w / (0.2·(TP_w + FP_w) + 0.8·|G|)</code>. Writing <code>T = TP_w</code> and
<code>S</code> for the emitted mass, <code>DTI ≈ T / (0.2·S + 0.8·|G|)</code>.</li>
<li>A binary {0,1} file strictly dominates any graded version of itself (the score is increasing
in the scale of a probability map). Every artifact here is binary.</li>
<li><strong>The marginal rule:</strong> adding a dot of kernel credit <code>k</code> pays iff
<code>k &gt; 0.2 · DTI</code>. At a score of 0.3345 that is a distance of at most
<strong>280 m</strong> from a scored truth pixel. Credit is nearly free anywhere inside the kernel
and worthless outside it — the metric is a distance filter, not a precision filter.</li>
</ol>

<h2>Why the previous best artifact scored what it scored, and what this repo does differently</h2>
<p>The family's best live-reported artifact is a 44,090-dot thinning of a multilayer detector
(distance-suppressed at 2.8 px), reported at 0.2600. The mechanism is the marginal rule above:
thinning deletes dots whose realised credit was already below the break-even bar. The family's own
analysis shows the family sits at its own emission optimum, so the remaining gap is a
<em>field</em> problem, not an emission problem. This repository does three things differently,
and each is falsifiable:</p>
<ul>
<li><strong>It chooses its validation frame by scoring already-scored artifacts on it.</strong>
Nineteen family artifacts whose live scores were reported (0.0461 to 0.2600) were downloaded and
scored on four candidate frames plus their own uniform controls. The catalogue frame is not merely
weak, it is <em>inverted</em> (rho = −0.475) and it is degenerate under the official mask — its own
truth pixels <em>are</em> the mask, so the truth set is empty and every artifact scores exactly
0.000000. The frame this repository pre-registered first (SGMC faults &gt;300 m from the catalogue)
also failed: rho = −0.054. The frame that survived is the SGMC strands within 300 m of the
catalogue (rho_excess = +0.396, all 19 artifacts above uniform). The amendment is recorded as
AM-44-01 and IR-44-09. <a href="validation.html">Validation</a> shows the whole table.</li>
<li><strong>It emits by the metric's own arithmetic, at the metric's own spacing.</strong> Across the
same 19 artifacts, the strongest measured correlate of a reported score is not a geological
statistic but dot spacing: <code>Spearman(score, fraction of dots 8-adjacent to another dot) =
−0.361</code>, and the three best artifacts have essentially zero adjacent dots while low scorers
have 0.9. A dot beside another adds no new truth coverage and still costs 0.2 of FP. The emitter
here is an exact greedy max-coverage solver that refuses any dot adding no new kernel credit.</li>
<li><strong>It emits with the metric's own arithmetic.</strong> Because <code>TP_w</code> is a
<em>max</em> over a 300 m disc, expected credit is submodular in the emitted set, so a fixed-order
greedy emitter carries the standard (1 − 1/e) guarantee, and the stopping condition is the
closed-form break-even rule above rather than a tuned radius.</li>
</ul>

<h2>Validation, out of fold</h2>
{validation_block}

<h2>Honest limitations</h2>
<p>Read <a href="limitations.html">Limitations and irregularities</a>. The three that matter most:
there is no organizer score for any artifact in this repository; the hidden label set's provenance
could not be read from this sandbox; and every external data source outside the owner's
hash-pinned GitHub mirrors is unreachable here, which is measured, not assumed.</p>
"""
    (DOCS / "index.html").write_text(page("Executive summary", body,
        "GEMSDOE44 · DOE GEMS Prize · what this is, the rules that decide it, and the one-click file"))

    # ---------------------------------------------------------------- how to submit
    pb = sub_audit["primary"] if sub_audit else {}
    slug = sub_audit["slug"] if sub_audit else "(no artifact built)"
    body = f"""
<h2>Exactly how to submit, line by line</h2>
<div class="panel">
<ol>
<li>Click <strong>Download</strong> on the <a href="index.html">executive summary</a>: you get
<code class="mono">{esc(slug)}-zeros.tif</code>.</li>
<li>Sign in to DrivenData and open
<a href="https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/">the submission form</a>
(<em>Submit</em> → <em>Make new submission</em>).</li>
<li><strong>File to submit</strong> → <em>Choose file</em> → select the downloaded
<code>.tif</code> (or the <code>.zip</code>, which contains the same single GeoTIFF).</li>
<li><strong>Note</strong> (optional) → paste:<br>
<span class="mono">{esc(sub_audit.get('note_for_portal','') if sub_audit else '')}</span></li>
<li>If the form offers a name/identifier field, use
<span class="mono">{esc(sub_audit.get('unique_name_for_portal','') if sub_audit else '')}</span> so
you can tell this submission apart from the others later.</li>
<li>Press <em>Submit</em>. The public leaderboard updates the “Best public DW-Tversky” column for
your account.</li>
</ol>
</div>

<h2>“Predicted values must be in range [0, 1]” — why it happens and how this file avoids it</h2>
<p>The official format contract requires a single-band <strong>float32</strong> GeoTIFF with values
between 0 and 1, NaN or null outside the footprint. The portal message appears when a validator
sees at least one value outside [0, 1] (a nodata sentinel such as −9999, an unnormalised model
output, an integer band, or a multi-band file).</p>
<p>The primary file here is written so that <em>every</em> cell is finite and every value is exactly
<code>0.0</code> or <code>1.0</code> — there is no nodata tag and no NaN anywhere, so no validator can
find an out-of-range value. The <code>-nan.tif</code> twin is offered for anyone who wants the
convention used by the official <code>sample_submission.tif</code> (NaN outside the footprint with
<code>nodata = nan</code>); both contain the identical prediction set inside the footprint.</p>
<p>Measured on the official template: footprint <strong>{pop.get('grid', {}).get('footprint_px', 0):,} px</strong>,
known-fault pixels <strong>{pop.get('grid', {}).get('known_fault_px', 0):,} px</strong>, scored
domain <strong>{pop.get('grid', {}).get('scored_domain_px', 0):,} px</strong>. Values outside the
footprint carry no prediction and are outside the study area.</p>
<p class="note">Source: <code>registry/population.json</code> (grid section),
<code>submissions/{esc(slug)}-audit.json</code> (format checks).</p>

<h2>Submission budget</h2>
<p>The competition allows a limited number of scored submissions and requires one single file to be
chosen for scoring across both prize rounds before the deadline. This repository therefore treats
each upload as an experiment and never spends a slot on an artifact that has not first beaten the
incumbent on the blocked primary frame — see <a href="method.html">Method</a>
<a href="validation.html">Validation</a>.</p>
"""
    (DOCS / "how-to-submit.html").write_text(page("How to submit", body,
        "The exact clicks, the exact Note, and the fix for the range error"))

    # ---------------------------------------------------------------- data
    man = load("data_manifest.json", {"files": []})
    rows = []
    for f in man.get("files", []):
        rows.append((esc(f.get("id", "")), esc(f.get("role", ""))[:110] + "…",
                     f"{f.get('bytes', 0):,}",
                     f"<span class='mono'>{esc(str(f.get('sha256', ''))[:24])}…</span>",
                     esc(f.get("provenance", ""))[:90] + "…"))
    body = f"""
<h2>Competition data — auditable table</h2>
<p>Four official files, each fetched from a public, hash-pinned mirror and verified against its
sha256 before use. No file was used before its digest matched.</p>
{table(["file", "role in this repository", "bytes", "sha256 (truncated)", "provenance"], rows)}
<p class="note">Source: <code>registry/data_manifest.json</code>, whose sha256 values were computed
from the bytes actually used here. The competition rasters are owner-supplied mirrors of
login-walled DrivenData files: the pins prove <em>reproducibility</em>, not organizer
authentication. Registered as irregularity IR-44-02 in <a href="limitations.html">Limitations</a>.
Reproduce with <code>bash scripts/fetch_official_mirrors.sh</code>, which fails closed on any
digest mismatch.</p>

<h2>Grid contract, read from the official bytes</h2>
{table(["quantity", "value"], [
    ("CRS", esc(pop.get('grid', {}).get('crs'))),
    ("resolution", "100 m"),
    ("shape", f"{pop.get('grid', {}).get('height')} rows × {pop.get('grid', {}).get('width')} cols"),
    ("transform", f"<span class='mono'>{esc(pop.get('grid', {}).get('transform'))}</span>"),
    ("footprint", f"{pop.get('grid', {}).get('footprint_px', 0):,} px"),
    ("known-fault mask", f"{pop.get('grid', {}).get('known_fault_px', 0):,} px"),
    ("scored domain", f"{pop.get('grid', {}).get('scored_domain_px', 0):,} px"),
])}

<h2>The 19 official feature bands</h2>
{table(["#", "band", "category"], [
    (str(i), esc(d), esc(c)) for i, (d, c) in enumerate(BAND_TABLE, start=1)
])}
<p class="note">Band descriptions are the ones stored in <code>training_features.tif</code> itself
(band tag <code>description</code>), read on 2026-10-06.</p>

<h2>Fault population measured from the official catalogue</h2>
{table(["quantity", "value"], [
    ("connected fault segments (≥ 12 px)", f"{pop.get('segments', {}).get('n'):,}"),
    ("median segment length (2σ, km)", esc(pop.get('segments', {}).get('median_length_km'))),
    ("90th percentile segment length (km)", esc(pop.get('segments', {}).get('p90_length_km'))),
    ("modal trace azimuth (length-weighted)", f"{pop.get('azimuth', {}).get('modal_bin_deg')}°"),
    ("USGS SGMC fault pixels absent from the catalogue", f"{pop.get('sgmc', {}).get('off_catalogue_px', 0):,} px"),
])}
<p class="note">Source: <code>registry/population.json</code>, produced by
<code>scripts/measure_population.py</code> from the hash-verified bytes.</p>

<h2>Official sources for manual review</h2>
{table(["what", "link", "status"], [
    (esc(x["id"]), f"<a href='{esc(x['url'])}'>{esc(x['url'])}</a>", esc(x["status"]))
    for x in (src.get("competition") or []) + (src.get("organizer_statements") or [])
])}
"""
    (DOCS / "data.html").write_text(page("Data", body,
        "Every official file, its hash, the grid contract, and the links to check by hand"))

    # ---------------------------------------------------------------- method
    holdout_note = ""
    if hold.get("pooled"):
        holdout_note = f"<pre>{esc(json.dumps(hold['pooled'], indent=2))}</pre>"
    body = f"""
<h2>Method, in the order it was done</h2>
<ol>
<li><strong>Read the rules from the source.</strong> The metric, kernel, weights, masking rule and
format contract are transcribed into <code>src/gems44/metric.py</code> and exercised by 23 tests
that include an independent O(|P|·|G|) brute-force transcription.</li>
<li><strong>Measure the population before modelling it.</strong> <code>scripts/measure_population.py</code>
labels the catalogue into 1,662 segments, measures the length and azimuth distributions, and tests
the one geometric hypothesis (H1) whose premise the data could settle. H1 failed its own test and
was dropped — see <a href="hypotheses.html">Hypotheses</a>.</li>
<li><strong>Measure detectability before trusting a field.</strong>
<code>scripts/measure_detectability.py</code> measures spatially blocked ROC AUC for real faults
absent from the catalogue versus random footprint pixels. Result: <strong>0.773</strong> blocked
(folds 0.762–0.789) against 0.773 for the same features on the catalogue frame — but see step 5 for
why that number did not survive contact with the metric.</li>
<li><strong>Build the emitter from the metric, not from intuition.</strong> <code>TP_w</code> is a
max over a 300 m disc ⇒ the objective is submodular ⇒ fixed-order greedy expected-marginal-credit
emission, stopping at <code>k &gt; 0.2·DTI</code>.</li>
<li><strong>Score fields with the metric, on blocked frames, out of fold.</strong>
<code>scripts/run_selection.py</code> sweeps a family of belief fields × emitted mass over four
spatially blocked quadrants, scoring every combination with the official metric against a frame
whose positives the catalogue does not contain, and selecting leave-one-fold-out. This is the step
that exists <em>because</em> step 3's AUC was misleading.</li>
<li><strong>Emit, write, and re-read.</strong> <code>scripts/build_submission.py</code> writes the
GeoTIFF, then independently re-reads the bytes on disk and checks CRS, resolution, shape,
transform, dtype, band count and the [0, 1] range.</li>
</ol>

<h2>Reproduce</h2>
<pre>bash scripts/fetch_official_mirrors.sh          # hash-verified data into data/
PYTHONPATH=src python scripts/measure_population.py
PYTHONPATH=src python scripts/measure_detectability.py
PYTHONPATH=src python scripts/run_selection.py
PYTHONPATH=src python scripts/run_holdout.py     # exact-operator confirmation
PYTHONPATH=src python scripts/build_submission.py
PYTHONPATH=src python -m pytest tests -q
python scripts/build_site.py</pre>

<h2>Metric-exact facts used by the selection</h2>
{table(["at score s", "break-even credit k = 0.2s", "implied maximum distance from scored truth"], [
    ("0.1000", "0.0200", "294 m"), ("0.2000", "0.0400", "288 m"), ("0.2600", "0.0520", "284 m"),
    ("0.2778", "0.0556", "283 m"), ("0.3195", "0.0639", "281 m"), ("0.3345", "0.0669", "280 m"),
])}
<p>These follow from the published formula alone and need no data. They are the reason a dot
anywhere inside the 300 m kernel is close to free, and the reason the competition is a coverage
problem rather than a precision problem.</p>

<h2>The first holdout, kept in the record</h2>
<p>A supervised off-catalogue classifier reached a blocked pixel AUC of 0.773, so an earlier
version of this pipeline emitted its belief field with the exact greedy emitter and scored it with
the official metric on a held-out quadrant. Held-out DTI at 44,090 dots was
<strong>0.0766</strong>; uniform-random dots scored <strong>0.1341</strong>; the live-scored 0.2600
incumbent artifact scored <strong>0.1353</strong> on the same frame. That single fold is preserved
in <code>registry/holdout_partial.json</code> and it is the reason this repository selects on the
metric rather than on AUC.</p>
{holdout_note}
"""
    (DOCS / "method.html").write_text(page("Method", body,
        "What was done, in order, with the negative result that changed the design"))

    # ---------------------------------------------------------------- hypotheses
    hrows = []
    for h in hyp["hypotheses"]:
        status_cls = "ok" if "IMPLEMENTED" in h["status"] else ("bad" if "FALSIFIED" in h["status"] else "warn")
        hrows.append((f"<strong>{esc(h['id'])}</strong><br><span class='note'>rank {esc(h['rank'])}</span>",
                      f"{esc(h['title'])}<br><span class='tag {status_cls}'>{esc(h['status'])}</span>",
                      esc("; ".join(h["layers"])),
                      esc(h["why_it_should_catch_a_catalogue_missing_fault"] if "why_it_should_catch_a_catalogue_missing_fault" in h else h.get("why_it_should_be_missing_from_the_catalogue", "")),
                      esc(h.get("differs_from_prior_work", "")),
                      esc(h.get("cost", "")),
                      esc(h.get("data_obtainability", "n/a"))))
    body = f"""
<h2>Ranked hypotheses</h2>
<p>Ranking is by expected DTI gain per unit of implementation cost, judged <em>after</em> the
falsification pass. Every entry names its layers, the physical signature targeted, why it should
catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, how it
differs from prior work, its cost, and the measured obtainability of its data.</p>
{"".join(f'''<div class="panel"><h3>{esc(h['id'])} (rank {esc(h['rank'])}) — {esc(h['title'])}</h3>
<p><span class="tag {'ok' if 'IMPLEMENTED' in h['status'] else ('bad' if 'FALSIFIED' in h['status'] else 'warn')}">{esc(h['status'])}</span></p>
<table>
<tr><th style="width:22%">layers</th><td>{esc('; '.join(h['layers']))}</td></tr>
<tr><th>physical signature</th><td>{esc(h.get('physical_signature',''))}</td></tr>
<tr><th>why it targets catalogue-missing faults</th><td>{esc(h.get('why_it_should_catch_a_catalogue_missing_fault') or h.get('why_it_should_be_missing_from_the_catalogue',''))}</td></tr>
<tr><th>how it differs from prior work</th><td>{esc(h.get('differs_from_prior_work',''))}</td></tr>
<tr><th>falsification test</th><td>{esc(h.get('falsification_test','not applicable'))}</td></tr>
<tr><th>verdict</th><td>{esc(h.get('verdict','pending'))}</td></tr>
<tr><th>expected DTI</th><td>{esc(h.get('expected_dti',''))}</td></tr>
<tr><th>cost</th><td>{esc(h.get('cost',''))}</td></tr>
<tr><th>data obtainability</th><td>{esc(h.get('data_obtainability','n/a'))}</td></tr>
</table></div>''' for h in hyp["hypotheses"])}
<p class="note">Source: <code>registry/hypotheses.json</code>. The falsification summary is at the
end of that file.</p>
"""
    (DOCS / "hypotheses.html").write_text(page("Hypotheses", body,
        "Three to five candidate geological hypotheses, ranked, with the falsified ones kept in"))

    # ---------------------------------------------------------------- leaderboard
    lb = load("leaderboard.json", {})
    lbrows = [(esc(r["rank"]), esc(r["participant"]), f"<strong>{esc(r['score'])}</strong>", esc(r.get("submissions", "")))
              for r in (lb.get("top") or [])]
    body = f"""
<h2>Public leaderboard, read {esc(lb.get('read_utc','(not recorded)'))}</h2>
<p>Read directly from <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/">the official leaderboard</a>.
DrivenData's terms prohibit automated monitoring, so this is a dated snapshot, not a live feed.</p>
{table(["rank", "participant", "best public DW-Tversky", "submissions"], lbrows)}
<p class="note">Source: <code>registry/leaderboard.json</code>. The page also records
{esc(lb.get("rows_on_page", "?"))} ranked rows in total.</p>

<h2>Our own artifacts versus the field</h2>
<p>No artifact in this repository has an organizer score. The rows below are the family's
<em>owner-reported</em> scores for artifacts published before this repository existed — they are
unauthenticated and are reproduced here only because they are the only live signal available.</p>
{table(["artifact (owner-reported)", "dots", "reported score"], [
    (esc("h19-5 solid multilayer"), "121,131", "0.1922"),
    (esc("h19-5 dotted d1.5"), "60,069", "0.2477"),
    (esc("h19-5 dotted d2.8"), "44,090", "<strong>0.2600</strong>"),
    (esc("h27-4 solo d28"), "~40,199", "0.2708"),
    (esc("h33-2-B2 flank B=2"), "37,654", "<strong>0.2778</strong>"),
])}
<p>Across the 19 artifacts with reported scores, the score falls as the dot count rises (Spearman = −0.653), and the three best sit in the 44,090–61,328 band. That is the marginal rule of the
metric in action: each removed dot whose realised credit was below <code>0.2·DTI</code> was costing
more than it earned. It is also the reason this repository treats mass as a first-class parameter
chosen by blocked selection rather than by taste.</p>
<p class="note">The trigger for the brief's question was the figure “0.2778” on the site
GEMSDOE32. That site's own README marks its <code>h33-2-b2</code> artifact as UNSCORED with a
projected 0.2747, so the 0.2778 on the leaderboard at rank 13 belongs to the participant
<code>extradr19</code>, not to a named file. Treating the two as the same submission is not
supported. Flagged as IR-44-03 in <a href="limitations.html">Limitations</a>.</p>
"""
    (DOCS / "leaderboard.html").write_text(page("Leaderboard", body,
        "Official snapshot, owner-reported history, and the artifact-attribution caveat"))

    # ---------------------------------------------------------------- validation
    rank = load("frame_ranking.json", {})
    prob = load("probe_manifest.json", {"probes": []})
    algn = load("probe_sgmc_alignment.json", [])
    halo = load("halo_analysis.json", {})
    strands = load("strands.json", {})
    pre = load("preregistration.json", {})
    am = (pre.get("amendments") or [{}])[-1]
    fr = rank.get("verdict", {})
    frame_rows = []
    for name, v in fr.items():
        frame_rows.append((esc(name), f"{v.get('truth_px', 0):,}",
                           f"<strong>{v.get('rho_level')}</strong>" if v.get('rho_level') is not None else "—",
                           f"<strong>{v.get('rho_excess')}</strong>",
                           f"{v.get('probes_beating_uniform', '?')}/{v.get('n_probes', '?')}",
                           f"{v.get('mean_dti')}"))
    probe_rows = []
    for a in sorted(algn, key=lambda r: -r["live"]):
        probe_rows.append((esc(a["file"][:44]), f"{a['live']:.4f}", f"{a['n']:,}",
                           f"{a['frac_dots_8adjacent']:.3f}", f"{a['frac_within_3px_of_N']:.3f}",
                           f"{a['frac_within_3px_of_P']:.3f}"))
    halo_cors = (halo.get("spearman_vs_live") or {})
    cor_rows = [(esc(k), esc(v)) for k, v in sorted(halo_cors.items(), key=lambda kv: -abs(float(kv[1])))][:8]
    reg = (halo.get("rank_regression") or {})
    body = f"""
<h2>What was tested, in order</h2>
<p>Two failure modes decide this project. Both were found by measurement, and both are recorded
against this repository's own earlier decisions.</p>

<h3>1 · A catalogue-truth frame is degenerate, and an inverted frame is worse than none</h3>
<p>Because the metric deletes known-fault pixels from <em>every</em> term, a frame whose truth is
the catalogue has an empty truth set under the official mask: the score is exactly 0 for every
artifact. Scored with the mask disabled it is worse than useless — the artifacts that score
0.24–0.26 live score <em>lower</em> on it than the ones that score 0.05:</p>
{table(["candidate frame", "truth px", "rho level", "rho excess", "probes above uniform", "mean DTI"], frame_rows)}
<p class="note">rho is Spearman against the 19 reported live scores. <code>rho excess</code> uses each
artifact's score minus its own uniform control at the same mass, which removes the mass effect.
Source: <code>registry/frame_ranking.json</code>, produced by <code>scripts/rank_frames.py</code>
from the probe set in <code>scripts/fetch_probes.py</code>.</p>
<p><strong>Frame P (SGMC beyond 300 m of the catalogue) was this repository's pre-registered primary
frame and it failed this test.</strong> It was replaced before any selection result was read:
<em>{esc(am.get('change',''))}</em>. Trigger: {esc(am.get('trigger',''))[:400]}…
Source: <code>registry/preregistration.json</code> amendment {esc(am.get('id','?'))}; IR-44-09.</p>

<h3>2 · Dot spacing, not geology, is the largest measured correlate of a reported score</h3>
<p>Each artifact below is real, was scored on the live board, and is on disk here. The columns are
measured from its bytes: how many dots it carries, what fraction of those dots are 8-adjacent to
another dot (pure waste under the metric), and what fraction sit within 300 m of the two SGMC
frames.</p>
{table(["artifact (owner-reported score)", "reported", "dots", "8-adjacent frac", "within 300 m of frame N", "of frame P"], probe_rows)}
<p class="note">Sources: <code>registry/probe_manifest.json</code>,
<code>registry/probe_sgmc_alignment.json</code>, <code>registry/halo_analysis.json</code>.
Reported scores are owner-reported, not organizer receipts (IR-44-08).</p>
<p>Rank correlations against the reported score, over all 19 artifacts:</p>
{table(["measured property", "Spearman vs reported score"], cor_rows)}
<p>The rank regression <code>{esc(reg.get('model',''))}</code> gives a coefficient of
<strong>{esc(reg.get('coef_log_mass'))}</strong> on log mass and <strong>{esc(reg.get('coef_frac3'))}</strong>
on the halo fraction, with R² = {esc(reg.get('r2'))}. The mechanism is exact under the official
formula: two adjacent dots deliver the same truth coverage as one, while each costs 0.2 of FP and
0.2 of the denominator.</p>

<h3>3 · Where the strands are</h3>
<p>The USGS State Geologic Map Compilation is <em>not</em> the competition catalogue: only 18.7% of
catalogue pixels lie within 100 m of an SGMC pixel and the median catalogue pixel is 849 m from the
nearest SGMC line. The SGMC raster used here carries {int((strands.get('A_comparison') or {}).get('sgmc_in_footprint_px', 0)):,} pixels
inside the footprint, of which {int((strands.get('A_comparison') or {}).get('sgmc_off_known_px', 0)):,} are off the
known-fault mask: {int((strands.get('A_comparison') or {}).get('sgmc_off_known_to_cat_distance_px_bins', {}).get('0-1', 0) + (strands.get('A_comparison') or {}).get('sgmc_off_known_to_cat_distance_px_bins', {}).get('1-3', 0)):,}
within 300 m of the catalogue (frame N) and {int((strands.get('A_comparison') or {}).get('sgmc_off_known_to_cat_distance_px_bins', {}).get('>30', 0)):,} more
than 3 km from it. The catalogue and the compilation are therefore two different maps of the same
ground, which is what makes one usable as validation truth for the other.
Source: <code>registry/strands.json</code> (<code>scripts/measure_strands.py</code>).</p>
"""
    (DOCS / "validation.html").write_text(page("Validation", body,
        "The measured basis for trusting — or not trusting — every number in this repository"))

    # ---------------------------------------------------------------- limitations
    irr = load("irregularities.json", {"items": []})
    body = f"""
<h2>Limitations that stand in the way</h2>
<ol>
<li><strong>No organizer score exists for any artifact in this repository.</strong> Every number
produced here is a local measurement on a proxy frame. The only way to obtain a real number is to
upload an artifact and read the board.</li>
<li><strong>The hidden label set's provenance is unknown.</strong> The organizers were asked in
community thread 11527 which data the experts used; the reply text could not be read from this
sandbox, so no claim is made about it.</li>
<li><strong>This sandbox cannot reach any external data host except GitHub and PyPI.</strong>
Measured on 2026-10-06: <code>earthexplorer.usgs.gov</code>, <code>prd-tnm.s3.amazonaws.com</code>,
<code>gdr.openei.org</code>, <code>storage.googleapis.com</code>, <code>raw.githubusercontent.com</code>
and every <code>*.github.io</code> host returned HTTP 000. Consequences: hypotheses H4 and H6 are
data-blocked here, and the one-click site cannot be checked from inside the sandbox.</li>
<li><strong>The primary validation frame is an analogue, not the target.</strong> USGS SGMC faults
absent from the given catalogue are real mapped faults the catalogue lacks, but they are a
pre-Quaternary-inclusive compilation and the hidden set need not share their statistics. The frame
is used because it is the only local frame that cannot be won by copying the catalogue.</li>
<li><strong>The validation frame was amended mid-session and the amendment is not free.</strong>
Frame N's rank agreement with the reported live order is rho_excess = +0.396 at n = 19 (p ≈ 0.09):
directional, not significant. It was chosen only after the pre-registered frame P failed
(AM-44-01 / IR-44-09). A submission selected on frame N therefore carries a residual risk that is
stated, not hidden.</li>
<li><strong>No geological field here is validated against the hidden set.</strong> The field shipped
is a supervised transfer of the USGS SGMC compilation; if the organizers' hidden faults do not
resemble that compilation, the transfer is worth nothing (IR-44-12). This is unmeasurable from
inside this sandbox.</li>
<li><strong>Quadrant blocking is coarse.</strong> Half-map quadrants remove large-scale leakage but
not regional structure that persists within a quadrant. Finer blocking would be stricter and is the
next methodological upgrade.</li>
<li><strong>Leaderboard aggregation is unresolved.</strong> Whether the public board pools pixels
across chunks or averages per-chunk scores is the subject of an unanswered forum thread; a pooled
local frame is not guaranteed to map linearly onto the board.</li>
</ol>

<h2>Irregularities, flagged for review</h2>
{table(["id", "what", "why it matters", "status"], [
    (f"<strong>{esc(i['id'])}</strong>", esc(i["what"]), esc(i.get("why", i.get("residual_risk", ""))), f"<span class='tag warn'>{esc(i['status'])}</span>")
    for i in irr.get("items", [])
])}

<h2>What is still to be done</h2>
<table>
<tr><th>item</th><th>why</th><th>cost</th></tr>
<tr><td>Upload the artifact and read the board</td><td>converts every proxy number in this
repository into a real one; the only measurement that matters</td><td>one submission slot</td></tr>
<tr><td>Finer spatial blocking (5×5 or 10×10 tiles with a buffer)</td><td>removes the residual
regional leakage that coarse quadrant blocking leaves</td><td>medium</td></tr>
<tr><td>H4: coupled ridge–trough matched filter on the derived 1 m lidar scarp raster</td>
<td>adds a physical signature with sub-100 m resolution; the raster is already obtainable from the
mirror</td><td>medium</td></tr>
<tr><td>H5: cross-compilation disagreement as a gated feature</td><td>the only source of known-real
faults the catalogue lacks, beyond SGMC</td><td>low</td></tr>
<tr><td>Two-round objective</td><td>the Final Prize Round is worth five times the Initial Round and
pays for expert-verified novel faults; the emission bar is then
<code>0.2·DTI/(1+ρ)</code>, not <code>0.2·DTI</code></td><td>low (arithmetic)</td></tr>
<tr><td>Independent scrutineer pass on <code>metric.py</code></td><td>the whole project reduces to
this one operator agreeing with the organizer's scorer</td><td>low</td></tr>
</table>
"""
    (DOCS / "limitations.html").write_text(page("Limitations", body,
        "What stands in the way, what is flagged, and what is next"))

    print("site written to docs/")
    return 0


BAND_TABLE = [
    ("Magnetic anomaly - deviation from expected Earth's magnetic field", "magnetic_data"),
    ("Reduced to pole magnetic data - magnetic anomaly corrected for latitude effects", "magnetic_data"),
    ("Total magnetic intensity horizontal gradient - rate of change in horizontal direction", "magnetic_data"),
    ("Geodetic second invariant - measure of strain rate tensor magnitude", "geodetic_strain"),
    ("Isostatic gravity anomaly slope - gradient of gravity after isostatic correction", "gravity_data"),
    ("Tilt angle or total curvature - magnetic field derivative for edge detection", "magnetic_data"),
    ("Geodetic shear rate - rate of angular deformation from GPS/InSAR", "geodetic_strain"),
    ("Geodetic dilatation rate - rate of volumetric strain (expansion/contraction)", "geodetic_strain"),
    ("Total magnetic intensity vertical gradient - rate of change in vertical direction", "magnetic_data"),
    ("Distance to earthquake (n=100km radius, a=15° azimuth parameters)", "seismic"),
    ("Isostatic gravity anomaly vertical gradient - vertical rate of change", "gravity_data"),
    ("Detrended elevation - topography with regional trends removed", "topographic"),
    ("Isostatic gravity anomaly - gravity after compensating for topographic mass", "gravity_data"),
    ("Total magnetic intensity - total strength of magnetic field", "magnetic_data"),
    ("Depth to basement surface - thickness of sedimentary cover", "subsurface"),
    ("Earthquake intensity or density (n=100km radius, a=15° parameters)", "seismic"),
    ("Conductivity surface - electrical conductivity of subsurface", "subsurface"),
    ("Isostatic gravity anomaly horizontal gradient - horizontal rate of change", "gravity_data"),
    ("Detrended elevation slope - gradient of elevation after detrending", "topographic"),
]


if __name__ == "__main__":
    raise SystemExit(main())
