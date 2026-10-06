"""GEMSDOE44 - fault discovery system for the DOE GEMS Prize (DrivenData #306).

Two independent build lineages coexist in this package (merged 2026-10-06):

  * Two-stage line (session 3): coarse favourability gate (stage1, which 2 km
    zones are worth searching) + fine-scale placement model (stage2, where the
    fault pixels actually are). Both trained and holdout-scored separately on a
    pre-registered spatially-blocked 4-fold design (GEMSDOE44-PREREG-1).
    Modules: config, data_io, features, emission, emit_submission, stage1,
    stage2, mirror_model, metric44 (no-catalogue-mask DTI variant).

  * Spaced exact-greedy line (session 4): grid, population, lineaments, field,
    submit, scripts_common, metric (official DTI WITH the USGS/INGENIOUS
    known-fault masking rule from community thread 11516; brute-force cross-checks).

NOTE FOR REVIEW: the two metric variants differ by the known-fault masking
convention (metric.py masks; metric44.py does not) - see
docs/irregularities.html and the session log in README.md.
"""
__all__ = [
    # spaced exact-greedy line
    "metric", "grid", "population", "lineaments", "field", "submit",
    # two-stage line
    "config", "data_io", "features", "emission", "emit_submission",
    "metric44", "mirror_model", "stage1", "stage2",
]
