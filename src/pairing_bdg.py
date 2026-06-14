"""
src/pairing_bdg.py — Bogoliubov–de Gennes d-wave gap equation and DOS.

Physical model
--------------
Single-band square-lattice, d_{x²-y²} pairing symmetry.

Gap ansatz:
    Δ(k) = Δ_d · f̃_d(k),    f̃_d(k) = (cos kx − cos ky) / 2

f̃_d is normalised so that |f̃_d|_max = 1 at the antinodal points (π, 0),
(0, π), (−π, 0), (0, −π).  Hence Δ_max = Δ_d is the maximum gap amplitude.

Self-consistent gap equation at temperature T (per lattice site):
    1 = (V_d/N) Σ_k  f̃_d(k)²  ·  tanh(E_k / 2T)  /  (2 E_k)
    E_k = √( ε_k² + Δ_d² f̃_d(k)² )

At T = 0:  tanh → 1.

Tc_MF (mean-field onset temperature) is found from the linearised equation
(Δ_d → 0+):
    1 = (V_d/N) Σ_k  f̃_d(k)²  ·  tanh(|ε_k| / 2Tc_MF)  /  (2 |ε_k|)

Important labels
----------------
Δ_d           : LOCAL PAIRING AMPLITUDE PROXY — the amplitude of the pair
                 field in the BdG mean-field.  It is NOT the zero-resistance Tc.
Tc_MF         : mean-field BCS onset (pair-formation scale T*).  In cuprates
                 Tc_MF > Tc_resist because phase fluctuations are absent in BdG.
Tc_onset      : defined here as Tc_MF.  Not to be confused with Tc_zero
                 (the actual zero-resistance temperature), which requires
                 phase-coherence physics beyond BdG.

Calibration (P = 0, Hg1212)
----------------------------
V_d = 1.3532 eV   →   Tc_MF(0) = 126.0 K   →   Δ_d(0) = 26.1 meV
    2 Δ_d / kB Tc_MF = 4.81  (enhanced above weak-coupling d-wave BCS ≈ 4.28;
    attributed to the near-Van-Hove FS topology with |t'/t| = 0.40)

V_d is calibrated by solve so Tc_MF(P=0) = 126 K (experimental Tc of Hg1212).
Δ_d(P=0) = 26.1 meV is a PREDICTION of the model, not an input.

Pressure dependence
-------------------
V_d(P) = V_D_CALIB × [V_d_eff(P) / V_d_eff(0)]
    where V_d_eff from src/channels.channel_table().

Band parameters ε_k(P) are updated at every pressure via src/lattice_bands.

Parameter provenance
--------------------
All band parameters: see src/lattice_bands docstring.
All assumed flags inherit from that module.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K_B: float = 8.617333e-5          # eV / K
HBAR_EV_S: float = 6.582119569e-16  # ℏ in eV·s (for rate expressions, unused here)

# ---------------------------------------------------------------------------
# Calibrated coupling constant — DO NOT CHANGE without re-running calibrate_V_d()
# Calibration condition: Tc_MF(P=0) = 126.0 K on a 256×256 k-grid
# with t0=0.430 eV, t'/t=-0.400, μ=-0.344 eV (src/lattice_bands defaults).
# ---------------------------------------------------------------------------
V_D_CALIB: float = 1.3532         # eV

# ---------------------------------------------------------------------------
# Numerical grid settings
# ---------------------------------------------------------------------------
NK_GAP: int = 128     # k-grid for gap / Tc scan over pressure (speed)
NK_DOS: int = 256     # k-grid for DOS computation (accuracy)
ETA_DOS: float = 0.003  # Lorentzian broadening for BdG DOS [eV]
N_E_DOS: int = 800    # energy-grid points for DOS

BDG_DISCLAIMER: str = (
    "Tc_MF (BdG mean-field) != Tc_onset; "
    "Delta_d = local pairing proxy — NOT the charge-transfer gap"
)

# Brentq tolerances
_XTOL: float = 1e-9
_RTOL: float = 1e-10


# ---------------------------------------------------------------------------
# d-wave form factor
# ---------------------------------------------------------------------------

def d_wave_form_factor_norm(kx: NDArray, ky: NDArray) -> NDArray:
    """
    Normalised d-wave form factor f̃_d(k) = (cos kx − cos ky) / 2.

    Range: [−1, 1].  Nodes at kx = ±ky.  Maximum at antinodes (π, 0) etc.
    """
    return (np.cos(kx) - np.cos(ky)) / 2.0


# ---------------------------------------------------------------------------
# Gap equation
# ---------------------------------------------------------------------------

def _gap_sum_T0(Delta_d: float, eps_k: NDArray, fd_k: NDArray) -> float:
    """(1/N) Σ_k f̃_d² / (2 E_k)  at T = 0."""
    E_k = np.sqrt(eps_k**2 + (Delta_d * fd_k)**2)
    return float(np.mean(fd_k**2 / (2.0 * E_k)))


def _gap_sum_T(Delta_d: float, eps_k: NDArray, fd_k: NDArray, T_eV: float) -> float:
    """(1/N) Σ_k f̃_d² tanh(E_k/2T) / (2 E_k)  at finite T."""
    E_k = np.sqrt(eps_k**2 + (Delta_d * fd_k)**2)
    th = np.tanh(np.clip(E_k / (2.0 * T_eV), -500.0, 500.0))
    return float(np.mean(fd_k**2 * th / (2.0 * E_k)))


def _linearised_sum(T_eV: float, eps_k: NDArray, fd_k: NDArray) -> float:
    """
    (1/N) Σ_k f̃_d² tanh(|ε_k|/2T) / (2|ε_k|)  — Δ→0+ limit for Tc.

    l'Hôpital at ε_k = 0:  tanh(x)/x → 1  ⇒  ratio = 1/(2T).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.abs(eps_k) < 1e-12,
            1.0 / (2.0 * T_eV),
            np.tanh(np.clip(eps_k / (2.0 * T_eV), -500.0, 500.0)) / eps_k,
        )
    return float(np.mean(fd_k**2 * ratio / 2.0))


# ---------------------------------------------------------------------------
# Solver for Δ_d at fixed V_d (T = 0)
# ---------------------------------------------------------------------------

def solve_gap_T0(
    eps_k: NDArray,
    fd_k: NDArray,
    V_d: float,
    Delta_min: float = 1e-5,
    Delta_max: float = 0.300,
) -> float:
    """
    Solve the self-consistent gap equation at T = 0 for Δ_d [eV].

    Returns 0.0 if no non-trivial solution exists in [Delta_min, Delta_max].
    """
    f_min = V_d * _gap_sum_T0(Delta_min, eps_k, fd_k) - 1.0
    f_max = V_d * _gap_sum_T0(Delta_max, eps_k, fd_k) - 1.0

    if f_min * f_max > 0:
        # No bracketed root — trivial (normal) or too strongly coupled
        if f_min > 0 and f_max > 0:
            warnings.warn(
                f"Gap equation sum > 1 for all Δ ∈ [{Delta_min}, {Delta_max}]. "
                "V_d may be too large; increase Delta_max.",
                stacklevel=2,
            )
        return 0.0

    return brentq(
        lambda D: V_d * _gap_sum_T0(D, eps_k, fd_k) - 1.0,
        Delta_min, Delta_max, xtol=_XTOL, rtol=_RTOL,
    )


# ---------------------------------------------------------------------------
# Solver for Tc_MF via linearised gap equation
# ---------------------------------------------------------------------------

def solve_Tc_MF(
    eps_k: NDArray,
    fd_k: NDArray,
    V_d: float,
    T_lo_K: float = 1.0,
    T_hi_K: float = 500.0,
) -> float:
    """
    Find the mean-field onset temperature Tc_MF [K] by solving the linearised
    gap equation.

    Tc_MF is the temperature at which V_d · (1/N) Σ_k f̃_d² tanh(|ε_k|/2T) / (2|ε_k|) = 1.

    This is the BdG pair-formation scale (T* in cuprate terminology), NOT the
    zero-resistance Tc.  Returns 0.0 if no solution in [T_lo, T_hi].
    """
    T_lo = T_lo_K * K_B
    T_hi = T_hi_K * K_B

    f_lo = V_d * _linearised_sum(T_lo, eps_k, fd_k) - 1.0
    f_hi = V_d * _linearised_sum(T_hi, eps_k, fd_k) - 1.0

    if f_lo * f_hi > 0:
        warnings.warn(
            f"Tc_MF not bracketed in [{T_lo_K}, {T_hi_K}] K for V_d={V_d:.4f} eV.",
            stacklevel=2,
        )
        return 0.0

    T_c_eV = brentq(
        lambda T: V_d * _linearised_sum(T, eps_k, fd_k) - 1.0,
        T_lo, T_hi, xtol=_XTOL * K_B, rtol=_RTOL,
    )
    return T_c_eV / K_B


# ---------------------------------------------------------------------------
# V_d calibration utility (run once; result hardcoded as V_D_CALIB)
# ---------------------------------------------------------------------------

def calibrate_V_d(
    target_Tc_K: float,
    eps_k: NDArray,
    fd_k: NDArray,
    V_lo: float = 0.5,
    V_hi: float = 3.0,
) -> float:
    """
    Find V_d such that Tc_MF(V_d) = target_Tc_K.

    Utility for calibration only; result stored in V_D_CALIB.
    """
    def residual(Vd: float) -> float:
        return solve_Tc_MF(eps_k, fd_k, Vd) - target_Tc_K

    f_lo = residual(V_lo)
    f_hi = residual(V_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Cannot bracket V_d in [{V_lo}, {V_hi}] eV for Tc_target={target_Tc_K} K. "
            f"residuals: f_lo={f_lo:.2f}, f_hi={f_hi:.2f}"
        )
    return brentq(residual, V_lo, V_hi, xtol=1e-6)


# ---------------------------------------------------------------------------
# BdG density of states
# ---------------------------------------------------------------------------

def dos_bdg(
    E_grid: NDArray,
    eps_k: NDArray,
    fd_k: NDArray,
    Delta_d: float,
    eta: float = ETA_DOS,
) -> NDArray:
    """
    BdG density of states N(E) via Lorentzian broadening.

    N(E) = (1/N) Σ_k [ u_k² L(E − E_k, η) + v_k² L(E + E_k, η) ]

    where:
        E_k  = √(ε_k² + Δ_d² f̃_d(k)²)
        u_k² = (1 + ε_k/E_k) / 2
        v_k² = (1 − ε_k/E_k) / 2
        L(x, η) = (η/π) / (x² + η²)

    The d-wave DOS is expected to show:
        - Linear-in-E V-shape for |E| ≪ Δ_d  (nodal quasiparticles)
        - Coherence peaks just below E = ±Δ_d  (antinodal quasiparticles)
        - N(0) > 0 (set by broadening η, not physical in pure d-wave at T=0)

    Args:
        E_grid:  1D array of energies at which to evaluate N [eV].
        eps_k:   dispersion array (μ already subtracted), shape (Nx, Ny).
        fd_k:    normalised form factor, shape (Nx, Ny).
        Delta_d: pairing amplitude [eV] — LOCAL PAIRING PROXY, not Tc_zero.
        eta:     Lorentzian half-width [eV].

    Returns:
        N(E) array, shape (len(E_grid),), normalised so ∫ N dE ≈ 1.
    """
    E_k = np.sqrt(eps_k**2 + (Delta_d * fd_k)**2)  # shape (Nx, Ny)
    uk2 = 0.5 * (1.0 + eps_k / np.where(E_k > 1e-15, E_k, 1e-15))
    vk2 = 0.5 * (1.0 - eps_k / np.where(E_k > 1e-15, E_k, 1e-15))

    E_flat = E_grid[:, None, None]           # (NE, 1, 1)
    Ek_flat = E_k[None, :, :]               # (1, Nx, Ny)
    uk2_flat = uk2[None, :, :]
    vk2_flat = vk2[None, :, :]

    L_pos = (eta / np.pi) / ((E_flat - Ek_flat)**2 + eta**2)
    L_neg = (eta / np.pi) / ((E_flat + Ek_flat)**2 + eta**2)

    N_E = np.mean(uk2_flat * L_pos + vk2_flat * L_neg, axis=(1, 2))
    return N_E


def dos_energy_grid(Delta_d: float, n_pts: int = N_E_DOS) -> NDArray:
    """
    Return an energy grid centred on zero, spanning ±4Δ_d.

    Denser near E = 0 (nodes) and E = ±Δ_d (coherence peaks) via concatenation.
    """
    # Fine grid near E=0 and near E=±Delta_d; coarser elsewhere
    E_max = 4.0 * max(Delta_d, 0.005)
    E_node = np.linspace(-0.3 * Delta_d, 0.3 * Delta_d, n_pts // 4)
    E_peak_p = np.linspace(0.6 * Delta_d, 1.4 * Delta_d, n_pts // 4)
    E_peak_n = -E_peak_p[::-1]
    E_broad = np.concatenate([
        np.linspace(-E_max, -1.4 * Delta_d, n_pts // 4),
        np.linspace(1.4 * Delta_d, E_max, n_pts // 4),
    ])
    E_all = np.sort(np.unique(np.concatenate([E_node, E_peak_p, E_peak_n, E_broad])))
    return E_all


# ---------------------------------------------------------------------------
# Pressure scan
# ---------------------------------------------------------------------------

def bdg_pressure_scan(
    P_grid: NDArray | None = None,
    Nk_gap: int = NK_GAP,
    Nk_dos: int = NK_DOS,
    P_dos_select: Sequence[float] = (0.0, 10.0, 20.0, 30.0),
    V_d_calib: float = V_D_CALIB,
) -> dict:
    """
    Compute Δ_d(P) and Tc_MF(P) for all pressures in P_grid.

    For pressures in P_dos_select, also compute the full BdG DOS N(E, P).

    Args:
        P_grid:        pressure array [GPa]; defaults to src/lattice_bands.P_GRID_GPA.
        Nk_gap:        k-grid for gap / Tc (fast).
        Nk_dos:        k-grid for DOS (accurate).
        P_dos_select:  subset of pressures for DOS output.
        V_d_calib:     calibrated pairing strength [eV] (default = V_D_CALIB).

    Returns:
        dict with keys:
            P           — pressure array [GPa]
            Delta_d_eV  — Δ_d(P) [eV]   (LOCAL PAIRING PROXY, NOT Tc_zero)
            Delta_d_meV — Δ_d(P) [meV]
            Tc_MF_K     — Tc_MF(P) [K]  (pair-formation onset, NOT Tc_zero)
            ratio_2DkT  — 2Δ_d / (kB Tc_MF) [dimensionless]
            V_d_P       — effective coupling V_d(P) [eV]
            V_d_eff_ratio — V_d_eff(P)/V_d_eff(0) [dimensionless]
            dos_P       — dict {P_val: (E_grid, N_E)} for P in P_dos_select
    """
    # defer import to avoid circular at module level
    from src.lattice_bands import (
        P_GRID_GPA, build_kgrid, dispersion_square,
        t_of_P, tprime_of_P, mu_of_P,
    )
    from src.channels import V_d_eff as vd_eff_fn

    if P_grid is None:
        P_grid = P_GRID_GPA
    P_arr = np.asarray(P_grid, dtype=float)

    # k-grid for gap scan
    kx_g, ky_g = build_kgrid(Nk_gap, Nk_gap)
    fd_g = d_wave_form_factor_norm(kx_g, ky_g)

    # k-grid for DOS
    kx_d, ky_d = build_kgrid(Nk_dos, Nk_dos)
    fd_d = d_wave_form_factor_norm(kx_d, ky_d)

    # Channel strength at P=0 for normalisation
    Vd_eff_0 = vd_eff_fn(0.0, Nk=Nk_gap)

    # Output arrays
    Delta_d = np.zeros(len(P_arr))
    Tc_MF   = np.zeros(len(P_arr))
    V_d_P   = np.zeros(len(P_arr))
    Vd_ratio = np.zeros(len(P_arr))
    dos_dict: dict[float, tuple[NDArray, NDArray]] = {}

    P_dos_set = set(float(p) for p in P_dos_select)

    print(f"  BdG pressure scan: {len(P_arr)} points, Nk_gap={Nk_gap}², Nk_dos={Nk_dos}²")

    for i, P in enumerate(P_arr):
        t  = t_of_P(P)
        tp = tprime_of_P(P)
        mu = mu_of_P(P)

        # Band dispersion at this pressure
        eps_g = dispersion_square(kx_g, ky_g, t, tp, mu)

        # Channel-strength scaling of V_d
        vd_eff_P = vd_eff_fn(P, Nk=Nk_gap)
        Vd_ratio[i] = vd_eff_P / Vd_eff_0
        Vd = V_d_calib * Vd_ratio[i]
        V_d_P[i] = Vd

        # Solve gap equation at T=0
        Delta_d[i] = solve_gap_T0(eps_g, fd_g, Vd)

        # Tc_MF from linearised equation
        Tc_MF[i] = solve_Tc_MF(eps_g, fd_g, Vd)

        # DOS at selected pressures
        P_key = round(P, 4)
        if any(abs(P - p_sel) < 0.01 for p_sel in P_dos_set):
            eps_d = dispersion_square(kx_d, ky_d, t, tp, mu)
            E_grid = dos_energy_grid(Delta_d[i])
            N_E = dos_bdg(E_grid, eps_d, fd_d, Delta_d[i])
            # Normalise: ∫ N dE = 1 (trapezoid)
            norm = np.trapezoid(N_E, E_grid)
            if norm > 0:
                N_E /= norm
            dos_dict[float(P)] = (E_grid, N_E)

        if (i + 1) % 10 == 0 or i == len(P_arr) - 1:
            print(
                f"    P={P:5.1f} GPa  Δ_d={Delta_d[i]*1e3:6.2f} meV  "
                f"Tc_MF={Tc_MF[i]:6.1f} K  V_d={Vd:.4f} eV"
            )

    ratio_2DkT = np.where(
        Tc_MF > 0,
        2.0 * Delta_d / (K_B * Tc_MF),
        np.nan,
    )

    return {
        "P":            P_arr,
        "Delta_d_eV":   Delta_d,
        "Delta_d_meV":  Delta_d * 1e3,
        "Tc_MF_K":      Tc_MF,
        "ratio_2DkT":   ratio_2DkT,
        "V_d_P":        V_d_P,
        "V_d_eff_ratio": Vd_ratio,
        "dos_P":        dos_dict,
    }
