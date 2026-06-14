"""
config.py — Global configuration for the simulation suite.

Centralises all tunable parameters, paths, and physical constants so that
no magic numbers appear in the model modules.  Import this module everywhere;
never hard-code values in src/.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SRC_DIR = ROOT / "src"

# ---------------------------------------------------------------------------
# Lattice parameters (Phase 2 — src/lattice_bands.py)
# ---------------------------------------------------------------------------
LATTICE: dict = {
    "type": "square",          # 'square' | 'triangular' | 'honeycomb'
    "Nx": 32,                  # sites along x
    "Ny": 32,                  # sites along y
    "t": 1.0,                  # nearest-neighbour hopping (energy unit)
    "t_prime": 0.0,            # next-nearest hopping
    "mu": 0.0,                 # chemical potential
}

# ---------------------------------------------------------------------------
# BdG / pairing parameters (Phase 3 — src/pairing_bdg.py)
# ---------------------------------------------------------------------------
BDG: dict = {
    "Delta_0": 0.1,            # initial gap guess (units of t)
    "T": 0.0,                  # temperature (units of t/k_B)
    "max_iter": 200,           # self-consistency iterations
    "tol": 1e-8,               # convergence criterion on |ΔΔ|
    "symmetry": "s-wave",      # 's-wave' | 'd-wave' | 'p-wave'
}

# ---------------------------------------------------------------------------
# Mediator / RPA parameters (Phase 4 — src/mediator_rpa.py)
# ---------------------------------------------------------------------------
RPA: dict = {
    "omega_D": 1.0,            # mediator energy scale (Debye-like cutoff)
    "lambda_ep": 0.3,          # dimensionless coupling
    "n_matsubara": 256,        # Matsubara frequency grid points
}

# ---------------------------------------------------------------------------
# Channel / null-model parameters (Phase 5 — src/channels.py, null_models.py)
# ---------------------------------------------------------------------------
CHANNELS: dict = {
    "active": ["singlet", "triplet"],   # pairing channels to evaluate
}

NULL_MODELS: dict = {
    "n_random": 500,           # realisations for random-matrix null model
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Two-scale / crossover parameters (Phase 6 — src/two_scale.py)
# ---------------------------------------------------------------------------
TWO_SCALE: dict = {
    "xi_coherence": 10.0,      # coherence length in lattice units
    "xi_correlation": 3.0,     # correlation length in lattice units
    "crossover_exponent": 2,   # k exponent (cf. memory-burden analogy)
}

# ---------------------------------------------------------------------------
# QE scaffold (Phase 7 — src/qe_scaffold.py)
# ---------------------------------------------------------------------------
QE: dict = {
    "pseudopotential_dir": DATA_DIR / "pseudopotentials",
    "ecutwfc": 40.0,           # plane-wave cutoff in Ry (placeholder)
    "k_mesh": (4, 4, 1),       # Monkhorst-Pack grid
}

# ---------------------------------------------------------------------------
# Figure / output settings (Phase 8 — src/figures.py)
# ---------------------------------------------------------------------------
FIGURES: dict = {
    "dpi": 160,
    "format": ["png", "pdf"],
    "style": "seaborn-v0_8-whitegrid",
    "font_family": "serif",
}

# ---------------------------------------------------------------------------
# Diagnostic threshold (analogous to delta_diag in PBH module)
# ---------------------------------------------------------------------------
CORR_DIAG: float = 1e-3       # minimum |C(r)| considered non-trivial
