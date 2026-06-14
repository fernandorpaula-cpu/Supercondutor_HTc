"""
src/channels.py — Effective d-wave pairing vertex and coherence diagnostics
under pressure for a single-band square-lattice cuprate model.

Physical picture
----------------
In the t-t'-J model the d_{x²-y²} pairing vertex receives two contributions:

    V_d_eff(P) = V_hop(P)  +  V_exch(P)

1.  Hopping-only (kinematic) contribution — from t' renormalisation of the
    particle-hole bubble.  The FS topology change driven by |t'/t| is the
    dominant kinematic mechanism discussed in the BCS weak-coupling analysis.

        λ_hop(P) = (1/N) Σ_k  f_d(k)² * |t'(P)| / t(P)  * w_FS(k)

    where f_d(k) = cos kx − cos ky is the d_{x²-y²} form factor and
    w_FS(k) is a Fermi-surface weight (see below).

2.  Exchange-enhanced contribution — from interlayer superexchange J_perp,
    which renormalises the singlet pairing amplitude:

        λ_exch(P) = α_exch * J_perp(P) / t(P)  *  N0_hat(P)

    where N0_hat is the dimensionless single-particle DOS at the Fermi level
    (in units of 1/t) and α_exch [ASSUMED] ~ 1 captures geometry factors.

3.  Total vertex: V_d_eff = λ_hop + λ_exch.

Coherence diagnostic
--------------------
C_coh(P) is a dimensionless proxy for phase coherence, modelled as a smooth
dome that peaks at an optimal pressure P_opt and decays on both sides while
staying above a positive floor C_floor > 0:

    C_coh(P) = C_floor + (C_peak − C_floor) * sech²((P − P_opt)/σ_P)

The dome shape captures the competition between rising Δ (favouring coherence)
and pair-breaking from disorder/fluctuations at very high P.

[ASSUMED]: P_opt, σ_P, C_peak, C_floor — calibrate to Tc(P) data.

Compressibility factor
----------------------
F_comp(P) = V(P)/V₀ — dimensionless volume ratio from Birch-Murnaghan EOS.
Used to track how much the lattice has compressed at each pressure.

All λ values are dimensionless projections (not Tc estimators).
They compare channels and pressure trends, not absolute coupling strengths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.lattice_bands import (
    P_GRID_GPA,
    build_kgrid,
    dispersion_square,
    fermi_function,
    parameter_table,
    t_of_P,
    tprime_of_P,
    t_perp_of_P,
    J_perp_of_P,
    volume_ratio,
)

# ---------------------------------------------------------------------------
# Channel model parameters — all [ASSUMED] pending experimental calibration
# ---------------------------------------------------------------------------

# k-grid resolution for FS integrals
NK: int = 128

# Fermi smearing for FS weights [eV] — acts as Lorentzian half-width
ETA_FS_EV: float = 0.015  # ~ 170 K; broader than Tc, narrower than bandwidth

# Geometry prefactor for exchange contribution — [ASSUMED] = 1.0
ALPHA_EXCH: float = 1.0

# Coherence dome parameters — [ASSUMED], calibrate to Tc(P) experiment
P_OPT_GPA: float = 15.0      # optimal pressure for coherence peak [GPa]
SIGMA_P_GPA: float = 10.0    # half-width of dome [GPa]
C_PEAK: float = 1.0           # peak value (normalised to 1 by convention)
C_FLOOR: float = 0.05         # positive floor — coherence never fully lost


# ---------------------------------------------------------------------------
# Fermi-surface weight
# ---------------------------------------------------------------------------

def _fs_weight(eps_k: NDArray, eta: float = ETA_FS_EV) -> NDArray:
    """
    Lorentzian Fermi-surface weight: w(k) ∝ η / (ε_k² + η²).

    Normalised so that Σ_k w(k) = 1.
    Approximates δ(ε_k) on a finite k-grid.
    """
    w = eta / (eps_k**2 + eta**2)
    w /= w.sum()
    return w


# ---------------------------------------------------------------------------
# Form factor
# ---------------------------------------------------------------------------

def d_wave_form_factor(kx: NDArray, ky: NDArray) -> NDArray:
    """d_{x²-y²} form factor: f_d(k) = cos kx − cos ky."""
    return np.cos(kx) - np.cos(ky)


# ---------------------------------------------------------------------------
# Channel eigenvalues
# ---------------------------------------------------------------------------

def lambda_hop(
    P_GPa: float | NDArray,
    Nk: int = NK,
) -> float | NDArray:
    """
    Hopping-only d-wave channel eigenvalue at pressure P [GPa].

    λ_hop(P) = |t'(P)/t(P)| * <f_d²>_FS

    where <...>_FS = Σ_k f_d(k)² w_FS(k) is the FS-averaged square of the
    d-wave form factor, weighted by the Lorentzian DOS.

    [ASSUMED]: linear coupling of |t'/t| to channel strength (weak-coupling).
    """
    scalar = np.ndim(P_GPa) == 0
    P_arr = np.atleast_1d(np.asarray(P_GPa, dtype=float))
    result = np.empty_like(P_arr)

    kx, ky = build_kgrid(Nk, Nk)

    for i, P in enumerate(P_arr):
        t = t_of_P(P)
        tp = tprime_of_P(P)
        mu = -0.80 * t  # constant filling
        eps = dispersion_square(kx, ky, t, tp, mu)
        w = _fs_weight(eps)
        fd = d_wave_form_factor(kx, ky)
        result[i] = abs(tp / t) * float(np.sum(fd**2 * w))

    return float(result[0]) if scalar else result


def lambda_exch(
    P_GPa: float | NDArray,
    Nk: int = NK,
) -> float | NDArray:
    """
    Exchange-enhanced d-wave contribution at pressure P [GPa].

    λ_exch(P) = α_exch * (J_perp(P) / t(P)) * N̂₀(P)

    N̂₀(P) = t(P) * Σ_k w_FS(k)   [dimensionless DOS at FS, in units of 1/t]

    The sum Σ_k w_FS(k) = 1 by normalisation; N̂₀ = t * (1/η) * (η/π) * ...
    Simplified: N̂₀(P) ≡ t(P) / (π * η)  (Lorentzian single-pole estimate).

    [ASSUMED]: ALPHA_EXCH = 1; ignores vertex corrections and retardation.
    """
    scalar = np.ndim(P_GPa) == 0
    P_arr = np.atleast_1d(np.asarray(P_GPa, dtype=float))

    t = t_of_P(P_arr)
    J = J_perp_of_P(P_arr)
    # DOS estimate: N̂₀ = t / (π η), dimensionless
    N0_hat = t / (np.pi * ETA_FS_EV)
    result = ALPHA_EXCH * (J / t) * N0_hat

    return float(result[0]) if scalar else result


def V_d_eff(
    P_GPa: float | NDArray,
    Nk: int = NK,
) -> float | NDArray:
    """
    Total effective d-wave pairing vertex V_d_eff(P) = λ_hop + λ_exch.
    """
    return lambda_hop(P_GPa, Nk) + lambda_exch(P_GPa)


def exchange_minus_hopping(
    P_GPa: float | NDArray,
    Nk: int = NK,
) -> float | NDArray:
    """
    Difference λ_exch(P) − λ_hop(P): positive when exchange dominates.
    """
    return lambda_exch(P_GPa) - lambda_hop(P_GPa, Nk)


# ---------------------------------------------------------------------------
# Coherence dome
# ---------------------------------------------------------------------------

def C_coh(P_GPa: float | NDArray) -> float | NDArray:
    """
    Smooth coherence dome C_coh(P), always above C_FLOOR > 0.

    C_coh(P) = C_floor + (C_peak − C_floor) * sech²((P − P_opt) / σ_P)

    [ASSUMED] shape and location — calibrate to Tc(P) data.
    """
    x = (np.asarray(P_GPa, dtype=float) - P_OPT_GPA) / SIGMA_P_GPA
    return C_FLOOR + (C_PEAK - C_FLOOR) / np.cosh(x) ** 2


# ---------------------------------------------------------------------------
# Compressibility factor
# ---------------------------------------------------------------------------

def F_comp(P_GPa: float | NDArray) -> float | NDArray:
    """
    Compressibility factor F_comp(P) = V(P)/V₀ from Birch-Murnaghan EOS.

    Ranges from 1.0 at P=0 to ~0.75 at P=30 GPa (Hg1212 estimate).
    """
    return volume_ratio(np.asarray(P_GPa, dtype=float))


# ---------------------------------------------------------------------------
# Full channel table
# ---------------------------------------------------------------------------

def channel_table(P_grid: NDArray | None = None, Nk: int = NK) -> dict[str, NDArray]:
    """
    Compute all channel quantities on P_grid [GPa].

    Returns dict with keys:
        P, t, tprime, t_perp, J_perp,
        lambda_hop, lambda_exch, V_d_eff,
        exch_minus_hop, C_coh, F_comp
    """
    if P_grid is None:
        P_grid = P_GRID_GPA
    P = np.asarray(P_grid, dtype=float)

    band = parameter_table(P)
    lhop = lambda_hop(P, Nk)
    lexch = lambda_exch(P)
    vd = lhop + lexch

    return {
        "P":              P,
        "t":              band["t"],
        "tprime":         band["tprime"],
        "t_perp":         band["t_perp"],
        "J_perp":         band["J_perp"],
        "lambda_hop":     lhop,
        "lambda_exch":    lexch,
        "V_d_eff":        vd,
        "exch_minus_hop": lexch - lhop,
        "C_coh":          C_coh(P),
        "F_comp":         F_comp(P),
    }
