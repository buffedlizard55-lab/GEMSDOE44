"""GEMSDOE44 - two-stage fault placement system for the DOE GEMS Prize (DrivenData #306).

Stage 1: coarse favourability gate (which 2 km zones are worth searching).
Stage 2: fine-scale placement model (where the fault pixels actually are).

Both stages are trained and holdout-scored separately on a pre-registered
spatially-blocked 4-fold design (GEMSDOE44-PREREG-1).
"""
