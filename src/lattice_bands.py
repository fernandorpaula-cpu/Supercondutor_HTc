"""
src/lattice_bands.py — Tight-binding band structure and pressure-dependent
hopping parameters for a single-band square-lattice cuprate model.

Material target: Hg1212 (HgBa₂CaCu₂O₆₊δ), bilayer structure.

Physical model
--------------
Dispersion:
    ε(k) = -2t(cos kx + cos ky) - 4t' cos kx cos ky - μ

where t, t', μ are pressure-dependent via a Birch-Murnaghan compression.

Pressure dependence of hopping parameters follows the Harrison rule
    t ∝ d^(-n),  n ≈ 3.5 for Cu-O (pd-σ bond),
applied to the ab-plane bond length compression d(P)/d₀ = [V(P)/V₀]^(1/3) / √2
(square lattice — only one in-plane direction matters for 2D).

Parameter provenance
--------------------
t₀ = 0.43 eV          — DFT estimate, Markiewicz et al. PRB 72 054519 (2005)
t₀'/t₀ = -0.40        — ARPES + DFT for Hg family; Pavarini et al. PRL (2001)
t_perp₀ = 0.09 eV     — bilayer splitting; Chakravarty et al. (1993) estimate
J_perp₀ = 0.010 eV    — interlayer superexchange; order-of-magnitude estimate
B₀ = 100 GPa           — bulk modulus Hg1212; approximate, Loureiro et al. (2001)
B₀' = 5.0             — pressure derivative; typical for cuprates
μ₀/t₀ = -0.80         — optimal doping, FS consistent with ARPES

ALL PARAMETERS LABELED [LIT], [EST], or [ASSUMED] below.
None of these values are first-principles results computed here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Pressure-independent reference parameters
# — source labels: [LIT] = published literature, [EST] = order-of-magnitude
#   estimate, [ASSUMED] = working assumption pending calibration
# ---------------------------------------------------------------------------

# Nearest-neighbour hopping [eV] — [LIT] Markiewicz et al. PRB 72 054519 (2005)
T0_EV: float = 0.43

# t'/t ratio at zero pressure — [LIT] Pavarini et al. PRL 87 047003 (2001)
TPRIME_RATIO_0: float = -0.40

# Interlayer single-particle hopping [eV] — [EST] bilayer Hg1212
T_PERP_0_EV: float = 0.09

# Interlayer superexchange [eV] — [EST] << in-plane J ~ 0.13 eV
J_PERP_0_EV: float = 0.010

# Chemical potential / t₀ at optimal doping — [ASSUMED] FS shape from ARPES
MU_OVER_T0: float = -0.80

# Birch-Murnaghan equation of state — [LIT] approximate Hg1212
B0_GPA: float = 100.0     # bulk modulus [GPa]
B0_PRIME: float = 5.0     # dP/dB (dimensionless)

# Harrison exponent for pd-σ hopping t ∝ d^{-n}
# n ≈ 7/2 for Cu-O bond — [LIT] Harrison, "Electronic Structure" (1980)
HARRISON_N: float = 3.5

# t_perp decays faster than in-plane (c-axis more compressible) — [ASSUMED]
HARRISON_N_PERP: float = 5.0

# Pressure grid [GPa] for all outputs
P_GRID_GPA: NDArray = np.linspace(0.0, 30.0, 61)


# ---------------------------------------------------------------------------
# Equation of state
# ---------------------------------------------------------------------------

def _bm3_pressure(x: float, B0: float, B0p: float) -> float:
    """
    Third-order Birch-Murnaghan pressure as a function of x = V/V₀.

    P(x) = (3B₀/2) * (x^{-7/3} - x^{-5/3})
            * [1 + (3/4)(B₀'-4)(x^{-2/3} - 1)]
    """
    t1 = x ** (-7.0 / 3.0) - x ** (-5.0 / 3.0)
    t2 = 1.0 + (3.0 / 4.0) * (B0p - 4.0) * (x ** (-2.0 / 3.0) - 1.0)
    return 1.5 * B0 * t1 * t2


def volume_ratio(P_GPa: float | NDArray) -> float | NDArray:
    """
    Return V(P)/V₀ by inverting the 3rd-order Birch-Murnaghan EOS.

    V/V₀ = 1 at P = 0 by construction.
    """
    scalar = np.ndim(P_GPa) == 0
    P_arr = np.atleast_1d(np.asarray(P_GPa, dtype=float))
    out = np.empty_like(P_arr)

    for i, P in enumerate(P_arr):
        if P <= 0.0:
            out[i] = 1.0
        else:
            # V/V₀ ∈ (0.5, 1.0) is a safe bracket for P < 200 GPa
            try:
                out[i] = brentq(
                    lambda x: _bm3_pressure(x, B0_GPA, B0_PRIME) - P,
                    0.50, 1.0 - 1e-9,
                    xtol=1e-10, rtol=1e-10,
                )
            except ValueError:
                out[i] = np.nan
                warnings.warn(f"EOS inversion failed at P={P:.1f} GPa", stacklevel=2)

    return out[0] if scalar else out


# ---------------------------------------------------------------------------
# Pressure-dependent hopping parameters
# ---------------------------------------------------------------------------

def t_of_P(P_GPa: float | NDArray) -> float | NDArray:
    """
    Nearest-neighbour hopping t(P) [eV].

    Harrison scaling: t ∝ d^{-n} ∝ (V/V₀)^{n/3} for 2D ab-plane bond.
    Bond length along a-direction: d ∝ (V/V₀)^{1/3} (isotropic approx.)

    Source: [LIT] Harrison (1980), [ASSUMED] isotropic compression.
    """
    vr = volume_ratio(P_GPa)
    return T0_EV * vr ** (-HARRISON_N / 3.0)


def tprime_of_P(P_GPa: float | NDArray) -> float | NDArray:
    """
    Next-nearest hopping t'(P) [eV].

    |t'/t| increases slightly with pressure as the FS becomes more rounded.
    Phenomenological linear correction: r(P) = r₀ * (1 + 0.004·P [GPa]).
    Coefficient [ASSUMED] — needs ARPES calibration under pressure.
    """
    ratio = TPRIME_RATIO_0 * (1.0 + 0.004 * np.asarray(P_GPa, dtype=float))
    return ratio * t_of_P(P_GPa)


def t_perp_of_P(P_GPa: float | NDArray) -> float | NDArray:
    """
    Interlayer hopping t_perp(P) [eV].

    The c-axis is more compressible than ab; use Harrison exponent n_perp > n.
    t_perp ∝ (V/V₀)^{n_perp/3}  — [ASSUMED] isotropic volume; [EST] n_perp=5.
    """
    vr = volume_ratio(P_GPa)
    return T_PERP_0_EV * vr ** (-HARRISON_N_PERP / 3.0)


def J_perp_of_P(P_GPa: float | NDArray) -> float | NDArray:
    """
    Interlayer superexchange J_perp(P) [eV].

    J_perp ∝ t_perp² / U,  U pressure-independent at this level.
    Normalised to J_perp₀ at P=0:
        J_perp(P) = J_perp₀ * (t_perp(P) / t_perp₀)²

    Source: [EST] — see interlayer exchange literature for YBCO/Bi2212.
    """
    ratio = t_perp_of_P(P_GPa) / T_PERP_0_EV
    return J_PERP_0_EV * ratio ** 2


def mu_of_P(P_GPa: float | NDArray) -> float | NDArray:
    """
    Chemical potential μ(P) [eV], tracking optimal doping under pressure.

    Kept at fixed μ/t(P) = MU_OVER_T0 — constant filling approximation.
    [ASSUMED]: doping level fixed; real experiments may show charge transfer.
    """
    return MU_OVER_T0 * t_of_P(P_GPa)


# ---------------------------------------------------------------------------
# k-mesh
# ---------------------------------------------------------------------------

def build_kgrid(Nx: int, Ny: int) -> tuple[NDArray, NDArray]:
    """
    Return (kx, ky) 2D arrays for an Nx×Ny uniform grid over [-π, π).

    Shape of each output: (Nx, Ny).
    """
    kx_1d = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    ky_1d = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="ij")
    return kx, ky


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def dispersion_square(
    kx: NDArray,
    ky: NDArray,
    t: float,
    t_prime: float = 0.0,
    mu: float = 0.0,
) -> NDArray:
    """
    Single-band square-lattice dispersion.

    ε(k) = -2t(cos kx + cos ky) - 4t' cos kx cos ky - μ

    Args:
        kx, ky:  k-grid arrays (any broadcast-compatible shape).
        t:       nearest-neighbour hopping [eV].
        t_prime: next-nearest hopping [eV].
        mu:      chemical potential [eV].

    Returns:
        Array of band energies, same shape as kx.
    """
    return (
        -2.0 * t * (np.cos(kx) + np.cos(ky))
        - 4.0 * t_prime * np.cos(kx) * np.cos(ky)
        - mu
    )


def density_of_states(
    energies: NDArray,
    n_bins: int = 400,
    eta: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """
    DOS via histogram of band energies.

    Args:
        energies: flat array of ε(k) values.
        n_bins:   number of energy bins.
        eta:      optional Lorentzian broadening width [eV]; 0 = histogram only.

    Returns:
        (E_centres, dos) — both 1D, normalised so ∫ dos dE = 1.
    """
    e_flat = energies.ravel()
    counts, edges = np.histogram(e_flat, bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    if eta > 0.0:
        # Lorentzian broadening: convolve histogram with L(E, eta)
        dE = centres[1] - centres[0]
        kernel_x = np.arange(-(n_bins // 2), n_bins // 2) * dE
        kernel = (eta / np.pi) / (kernel_x**2 + eta**2)
        kernel /= kernel.sum()
        counts = np.convolve(counts, kernel, mode="same")

    return centres, counts


def fermi_function(eps: NDArray, T_eV: float = 0.026) -> NDArray:
    """Fermi-Dirac distribution; T_eV = k_B T in eV (default: room T)."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(np.clip(eps / T_eV, -500, 500)))


# ---------------------------------------------------------------------------
# Full parameter table over pressure grid
# ---------------------------------------------------------------------------

def parameter_table(P_grid: NDArray | None = None) -> dict[str, NDArray]:
    """
    Compute all pressure-dependent parameters on P_grid [GPa].

    Returns dict with keys:
        P, VrV0, t, tprime, t_perp, J_perp, mu
    """
    if P_grid is None:
        P_grid = P_GRID_GPA
    P = np.asarray(P_grid, dtype=float)
    return {
        "P":      P,
        "VrV0":   volume_ratio(P),
        "t":      t_of_P(P),
        "tprime": tprime_of_P(P),
        "t_perp": t_perp_of_P(P),
        "J_perp": J_perp_of_P(P),
        "mu":     mu_of_P(P),
    }
