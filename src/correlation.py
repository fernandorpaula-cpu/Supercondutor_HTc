"""
src/correlation.py — Correlation and exchange proxies under pressure.

MANDATORY LABELS (do not remove):
    A. "gap Hubbard-HF (banda única) ≠ gap de transferência de carga real do cuprato"
    B. "m(P) é proxy de meia-banda (half-filling); não é magnetização do cuprato dopado"
    C. "Z_BR é aproximação de Gutzwiller; validade limitada perto do limite de Mott"
    D. "J_Emery pode super-prever o realce de J sob pressão — verificar [ASSUMED] em Δ_pd(P)"

Physical models
--------------
1. Hubbard-HF proxy (single-band, half-filling approximation)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Self-consistent mean-field gap Δ_HF and staggered magnetization m:

       2 / U_CORR = (1/N) Σ_k  1 / sqrt(ε_k² + Δ_HF²)
       m = (2/N) Σ_k  Δ_HF / (2 sqrt(ε_k² + Δ_HF²))

   where ε_k = −2t(cos kx + cos ky) − 4t' cos kx cos ky  at μ = 0.

   The half-filling AFM Slater gap opens for ANY U > 0 on the square lattice
   (perfect nesting at Q = (π,π)).

   !! LABEL A !!  Δ_HF is a SINGLE-BAND proxy.  The real cuprate has a
   charge-transfer gap Δ_CT ~ 1.5–2 eV set by the Cu d – O p splitting and
   NOT by U alone.  Δ_HF overestimates the gap because it omits d-p mixing.

   !! LABEL B !!  m(P) is evaluated at HALF-FILLING with μ = 0.  Real
   Hg1212 at optimal doping (δ ≈ 0.15) has no long-range magnetic order.
   This is a correlation-strength proxy, not a physical magnetization.

2. Brinkman-Rice quasiparticle weight (Gutzwiller approximation)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Z_BR = max(0, 1 − (U_CORR / U_c)²)
   where U_c(P) = 2|⟨T⟩₀(P)|  (twice the absolute half-filling kinetic energy).

   Z_BR → 0 as U → U_c(P): Mott transition within Gutzwiller.

   !! LABEL C !!  Z_BR is Gutzwiller (single Slater determinant with
   constrained double occupancy).  It neglects vertex corrections, frequency
   dependence of Σ, and orbital degeneracy.  More accurate slave-boson or
   DMFT calculations give similar qualitative trends but different numbers.

3. Emery three-band model proxies
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Cu-O hopping: t_pd(P) = T_PD_0 × (V/V₀)^{−N_PD/3}
   O-O hopping:  t_pp(P) = T_PP_0 × (V/V₀)^{−N_PP/3}
   Charge-transfer energy: Δ_pd(P) = Δ_PD_0 × (V/V₀)^{ALPHA_DELTA_PD}

   Note: Δ_pd DECREASES slowly under pressure (oxygen 2p rises relative to Cu 3d
   as the bond compresses).  The exponent ALPHA_DELTA_PD = 0.3 is [ASSUMED].

4. Superexchange
   ~~~~~~~~~~~~~
   Single-band Hubbard:
       J_Hub(P) = 4 t(P)² / U_CORR       [grows as t²]

   Emery / Zhang-Rice singlet (Hybertsen et al. 1990):
       J_Emery(P) = 4 t_pd(P)^4 / Δ_pd(P)^2
                    × (1/U_D + 2/(2 Δ_pd(P) + U_P))

   Over-prediction criterion:
       J_Emery(P)/J_Emery(0) > J_Hub(P)/J_Hub(0)

   This ALWAYS holds for P > 0 because:
   • t_pd^4 ∝ (V/V₀)^{−14/3} grows faster than t² ∝ (V/V₀)^{−7/3}
   • Δ_pd^2 ∝ (V/V₀)^{+0.6} is a small opposing factor

   !! LABEL D !!  J_Emery therefore OVER-PREDICTS the pressure enhancement
   of J relative to the simple Hubbard model.  Both models treat U_D and U_P
   as pressure-independent, which is itself an approximation.

Parameter provenance
--------------------
U_CORR_EV    = 1.5 eV  — [ASSUMED] Mott physics U (≠ RPA U = 0.3 eV)
T_PD_0_EV    = 1.30 eV — [LIT] Hybertsen et al., PRB 39 9028 (1990)
T_PP_0_EV    = 0.65 eV — [EST] O-O hopping order of magnitude
DELTA_PD_0_EV= 3.6 eV  — [LIT] Emery (1987); Zaanen-Sawatzky-Allen (1985)
U_D_EV       = 8.0 eV  — [LIT] Cu 3d on-site Coulomb, typical cuprate
U_P_EV       = 4.0 eV  — [EST] O 2p on-site Coulomb
HARRISON_N_PD= 3.5     — [LIT] pd-σ exponent, Harrison (1980)
HARRISON_N_PP= 2.5     — [ASSUMED] pp bond shorter range than pd
ALPHA_DELTA_PD=0.3     — [ASSUMED] dΔ_pd/d(ln V); small positive value
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .lattice_bands import (
    build_kgrid, dispersion_square,
    t_of_P, tprime_of_P, volume_ratio,
    fermi_function, P_GRID_GPA,
)

# ---------------------------------------------------------------------------
# Mandatory labels
# ---------------------------------------------------------------------------

LABEL_A: str = "gap Hubbard-HF (banda única) ≠ gap de transferência de carga real do cuprato"
LABEL_B: str = "m(P) é proxy de meia-banda (half-filling); não é magnetização do cuprato dopado"
LABEL_C: str = "Z_BR é aproximação de Gutzwiller; validade limitada perto do limite de Mott"
LABEL_D: str = "J_Emery pode super-prever o realce de J sob pressão — verificar [ASSUMED] em Δ_pd(P)"

# ---------------------------------------------------------------------------
# Parameters (all labeled per provenance)
# ---------------------------------------------------------------------------

# Correlation Hubbard U — larger than RPA U (mediator_rpa uses 0.30 eV)
# [ASSUMED] typical Mott/Hubbard scale for cuprates
U_CORR_EV: float = 1.5

# Emery model reference values
T_PD_0_EV: float    = 1.30   # [LIT] Hybertsen et al. PRB 39 9028 (1990)
T_PP_0_EV: float    = 0.65   # [EST]
DELTA_PD_0_EV: float = 3.6   # [LIT] Emery (1987); Zaanen, Sawatzky & Allen (1985)
U_D_EV: float       = 8.0    # [LIT] Cu 3d on-site Coulomb
U_P_EV: float       = 4.0    # [EST] O 2p on-site Coulomb

# Harrison exponents for Emery parameters
HARRISON_N_PD: float = 3.5   # [LIT] pd-σ bond
HARRISON_N_PP: float = 2.5   # [ASSUMED] pp bond shorter range

# Pressure exponent for Δ_pd: Δ_pd(P) = Δ_pd₀ × (V/V₀)^α
# α > 0 → Δ_pd decreases under pressure (O 2p rises relative to Cu 3d)
ALPHA_DELTA_PD: float = 0.3  # [ASSUMED]

# k-grid size for HF and BR calculations
NK_CORR: int = 48

# Threshold for reporting Emery over-prediction
OVERPRED_RATIO_THRESHOLD: float = 1.05  # flag when J_Emery/J_Hub enhancement > this

# ---------------------------------------------------------------------------
# Half-filling dispersion proxy (μ = 0)
# ---------------------------------------------------------------------------

def eps_half_filling(kx: NDArray, ky: NDArray, P_GPa: float) -> NDArray:
    """
    Dispersion for the HF proxy at μ = 0.

    Uses pressure-dependent t(P) and t'(P) from lattice_bands but sets μ = 0.
    This gives a half-filling proxy: filling = 1 exactly only for t' = 0;
    with t' ≠ 0 there is a small deviation, acceptable for a proxy calculation.

    [ASSUMED] μ = 0 approximation for half-filling.
    """
    t  = float(t_of_P(P_GPa))
    tp = float(tprime_of_P(P_GPa))
    return dispersion_square(kx, ky, t, tp, mu=0.0)


# ---------------------------------------------------------------------------
# 1. Hubbard-HF gap and magnetization
# ---------------------------------------------------------------------------

def _hf_lhs(Delta: float, eps_k: NDArray) -> float:
    """
    Left-hand side of the HF gap equation (should equal 2/U_CORR).

    LHS(Δ) = (1/N) Σ_k 1 / sqrt(ε_k² + Δ²)

    For Δ = 0 this diverges (log VHS for square lattice → always has solution).
    """
    if Delta < 1e-12:
        # Regularise: sum 1/|ε| → large but finite for discretised grid
        eps_safe = np.where(np.abs(eps_k) < 1e-10, 1e-10, np.abs(eps_k))
        return float(np.mean(1.0 / eps_safe))
    return float(np.mean(1.0 / np.sqrt(eps_k**2 + Delta**2)))


def hubbard_hf(
    eps_k: NDArray,
    U_corr: float = U_CORR_EV,
) -> tuple[float, float]:
    """
    Solve the HF mean-field gap equation at half-filling.

    2/U_corr = (1/N) Σ_k 1/sqrt(ε_k² + Δ_HF²)

    Staggered magnetization: m = (2 Δ_HF / U_corr) / 2 = Δ_HF / U_corr

    Returns:
        (Delta_HF [eV], m [dimensionless, 0 ≤ m ≤ 1])

    !! LABEL A, B !!  See module docstring.
    """
    target = 2.0 / U_corr

    # LHS is monotonically decreasing from ∞ (at Δ=0) to 0 (at Δ→∞).
    # Always has a solution for any U > 0 on the square lattice.
    f0 = _hf_lhs(0.0, eps_k)
    if f0 < target:
        # No ordered state (shouldn't happen for square lattice, but guard)
        return 0.0, 0.0

    # Upper bracket: LHS(Δ_max) < target
    Delta_max = U_corr  # large enough for any t-scale
    while _hf_lhs(Delta_max, eps_k) > target:
        Delta_max *= 2.0
        if Delta_max > 1e3:
            break

    try:
        Delta_HF = brentq(
            lambda D: _hf_lhs(D, eps_k) - target,
            1e-6, Delta_max,
            xtol=1e-8, rtol=1e-8,
        )
    except ValueError:
        Delta_HF = 0.0

    m = Delta_HF / U_corr  # staggered magnetization proxy
    return Delta_HF, m


# ---------------------------------------------------------------------------
# 2. Brinkman-Rice quasiparticle weight
# ---------------------------------------------------------------------------

def kinetic_energy_half(eps_k: NDArray) -> float:
    """
    Mean kinetic energy per site at T=0 half-filling (μ = 0, occupied: ε_k < 0).

    ⟨T⟩₀ = (1/N) Σ_{k: ε_k < 0} ε_k  [eV]  (negative number)
    """
    occ = eps_k[eps_k < 0.0]
    if len(occ) == 0:
        return 0.0
    return float(np.sum(occ)) / eps_k.size


def brinkman_rice_Z(eps_k: NDArray, U_corr: float = U_CORR_EV) -> float:
    """
    Brinkman-Rice quasiparticle weight in the Gutzwiller approximation.

    Z_BR = max(0, 1 − (U_corr / U_c)²)
    where U_c = 2 |⟨T⟩₀|  (critical U for Mott transition).

    Returns:
        Z ∈ [0, 1].  Z = 0 implies a Mott insulating proxy.

    !! LABEL C !!  Gutzwiller approximation; not DMFT.
    """
    T0 = kinetic_energy_half(eps_k)
    U_c = 2.0 * abs(T0)
    if U_c < 1e-10:
        return 0.0
    ratio_sq = (U_corr / U_c) ** 2
    return float(max(0.0, 1.0 - ratio_sq))


# ---------------------------------------------------------------------------
# 3. Emery model proxies
# ---------------------------------------------------------------------------

def emery_t_pd(P_GPa: float | NDArray) -> float | NDArray:
    """
    Cu-O hopping t_pd(P) [eV] via Harrison scaling.

    t_pd(P) = T_PD_0 × (V(P)/V₀)^{−N_PD/3}

    Source: [LIT] Hybertsen et al. PRB 39 9028 (1990) for t_pd₀.
    Pressure scaling: [ASSUMED] Harrison exponent n_pd = 3.5.
    """
    vr = volume_ratio(P_GPa)
    return T_PD_0_EV * vr ** (-HARRISON_N_PD / 3.0)


def emery_t_pp(P_GPa: float | NDArray) -> float | NDArray:
    """
    O-O hopping t_pp(P) [eV] via Harrison scaling.

    t_pp(P) = T_PP_0 × (V(P)/V₀)^{−N_PP/3}

    Source: [EST] shorter range than pd; n_pp = 2.5 [ASSUMED].
    """
    vr = volume_ratio(P_GPa)
    return T_PP_0_EV * vr ** (-HARRISON_N_PP / 3.0)


def emery_delta_pd(P_GPa: float | NDArray) -> float | NDArray:
    """
    Charge-transfer energy Δ_pd(P) [eV].

    Δ_pd(P) = DELTA_PD_0 × (V(P)/V₀)^{ALPHA_DELTA_PD}

    Δ_pd decreases slowly under pressure (α > 0, V/V₀ < 1).
    The rate is [ASSUMED]; experimental data are scarce.

    !! NOTE !!  Δ_pd is an IONIC energy, not a hopping integral.
    Its pressure dependence is much weaker than t_pd's.
    """
    vr = volume_ratio(P_GPa)
    return DELTA_PD_0_EV * vr ** ALPHA_DELTA_PD


# ---------------------------------------------------------------------------
# 4. Superexchange
# ---------------------------------------------------------------------------

def J_hubbard(P_GPa: float | NDArray, U_corr: float = U_CORR_EV) -> float | NDArray:
    """
    Hubbard superexchange J_Hub(P) = 4 t(P)² / U_corr  [eV].

    Uses the single-band in-plane hopping t(P) from lattice_bands.
    Grows monotonically with P as t increases.
    """
    t = t_of_P(P_GPa)
    return 4.0 * t**2 / U_corr


def J_emery(
    P_GPa: float | NDArray,
    U_d: float = U_D_EV,
    U_p: float = U_P_EV,
) -> float | NDArray:
    """
    Emery / Zhang-Rice singlet superexchange J_Emery(P) [eV].

    J_Emery = 4 t_pd(P)^4 / Δ_pd(P)^2
              × ( 1/U_d  +  2/(2 Δ_pd(P) + U_p) )

    Source: Hybertsen, Schlüter & Christensen PRB 39 9028 (1990),
            Zhang & Rice PRB 37 3759 (1988).

    !! LABEL D !!  J_Emery grows as t_pd^4 ∝ (V/V₀)^{−14/3} under pressure,
    while J_Hub grows as t^2 ∝ (V/V₀)^{−7/3}.  The Emery model therefore
    OVER-PREDICTS the pressure enhancement of J compared to the simple
    Hubbard model.  This is because t_pd^4 has a steeper volume exponent
    and Δ_pd^2 only partially compensates.  Both U_d and U_p are taken
    as pressure-independent here — an additional [ASSUMED] simplification.
    """
    tpd = emery_t_pd(P_GPa)
    Dpd = emery_delta_pd(P_GPa)
    coupling = 1.0 / U_d + 2.0 / (2.0 * Dpd + U_p)
    return 4.0 * tpd**4 / Dpd**2 * coupling


def overprediction_report(
    P: NDArray,
    J_hub: NDArray,
    J_em: NDArray,
) -> dict:
    """
    Identify where J_Emery over-predicts the pressure enhancement vs J_Hub.

    Over-prediction criterion:
        r_Emery(P) = J_Emery(P)/J_Emery(0) > r_Hub(P) = J_Hub(P)/J_Hub(0)

    Returns dict with:
        'ratio_Hub':   J_Hub/J_Hub(0) array
        'ratio_Emery': J_Emery/J_Emery(0) array
        'overpred':    boolean array, True where Emery enhances more than Hub
        'P_first_overpred': first P where flag is True [GPa], or NaN
        'max_excess':  max(r_Emery - r_Hub)
    """
    r_hub = J_hub / J_hub[0]
    r_em  = J_em  / J_em[0]
    flag = r_em > r_hub * OVERPRED_RATIO_THRESHOLD
    first = float(P[flag][0]) if np.any(flag) else float("nan")
    return {
        "ratio_Hub":          r_hub,
        "ratio_Emery":        r_em,
        "overpred":           flag,
        "P_first_overpred":   first,
        "max_excess":         float((r_em - r_hub).max()),
    }


# ---------------------------------------------------------------------------
# Full pressure scan
# ---------------------------------------------------------------------------

@dataclass
class CorrelationScanResult:
    """Container for the full correlation proxy scan."""
    P:          NDArray
    # HF proxy
    Delta_HF:   NDArray   # Hubbard-HF gap [eV] (half-filling proxy)
    m_HF:       NDArray   # staggered magnetization proxy [dimensionless]
    # Brinkman-Rice
    Z_BR:       NDArray   # quasiparticle weight [dimensionless]
    U_c:        NDArray   # critical Brinkman-Rice U [eV]
    # Emery parameters
    t_pd:       NDArray   # Cu-O hopping [eV]
    t_pp:       NDArray   # O-O hopping [eV]
    Delta_pd:   NDArray   # charge-transfer energy [eV]
    # Superexchange
    J_Hub:      NDArray   # 4t²/U [eV]
    J_Em:       NDArray   # Emery J [eV]
    enh_Hub:    NDArray   # J_Hub(P)/J_Hub(0)
    enh_Emery:  NDArray   # J_Emery(P)/J_Emery(0)
    overpred:   NDArray   # bool: Emery over-predicts vs Hub
    # Single-band t(P)
    t:          NDArray   # in-plane hopping [eV]
    # Metadata
    U_corr:     float = U_CORR_EV
    Nk:         int = NK_CORR


def correlation_scan(
    P_grid: NDArray = P_GRID_GPA,
    Nk:     int     = NK_CORR,
    U_corr: float   = U_CORR_EV,
) -> CorrelationScanResult:
    """
    Compute all correlation proxies over a pressure grid.

    At each pressure P:
      1. Build ε_k at μ=0 (half-filling proxy).
      2. Solve HF gap equation → Δ_HF(P), m(P).
      3. Compute kinetic energy → Z_BR(P), U_c(P).
      4. Compute Emery parameters: t_pd, t_pp, Δ_pd.
      5. Compute J_Hub and J_Emery.
      6. Flag over-prediction.

    Returns CorrelationScanResult.
    """
    P = np.asarray(P_grid, dtype=float)
    n = len(P)
    kx, ky = build_kgrid(Nk, Nk)

    Delta_HF = np.zeros(n)
    m_HF     = np.zeros(n)
    Z_BR     = np.zeros(n)
    U_c_arr  = np.zeros(n)
    t_arr    = np.zeros(n)
    tpd_arr  = np.zeros(n)
    tpp_arr  = np.zeros(n)
    Dpd_arr  = np.zeros(n)
    JHub_arr = np.zeros(n)
    JEm_arr  = np.zeros(n)

    for i, Pi in enumerate(P):
        eps_k = eps_half_filling(kx, ky, Pi)

        # HF gap
        dHF, m = hubbard_hf(eps_k, U_corr)
        Delta_HF[i] = dHF
        m_HF[i]     = m

        # Brinkman-Rice
        T0 = kinetic_energy_half(eps_k)
        Uc = 2.0 * abs(T0)
        U_c_arr[i] = Uc
        Z_BR[i]    = float(max(0.0, 1.0 - (U_corr / Uc)**2)) if Uc > 1e-10 else 0.0

        # Single-band t(P)
        t_arr[i] = float(t_of_P(Pi))

        # Emery
        tpd_arr[i] = float(emery_t_pd(Pi))
        tpp_arr[i] = float(emery_t_pp(Pi))
        Dpd_arr[i] = float(emery_delta_pd(Pi))
        JHub_arr[i] = float(J_hubbard(Pi, U_corr))
        JEm_arr[i]  = float(J_emery(Pi))

    # Enhancements and over-prediction
    enh_Hub   = JHub_arr / JHub_arr[0]
    enh_Emery = JEm_arr  / JEm_arr[0]
    overpred  = enh_Emery > enh_Hub * OVERPRED_RATIO_THRESHOLD

    return CorrelationScanResult(
        P         = P,
        Delta_HF  = Delta_HF,
        m_HF      = m_HF,
        Z_BR      = Z_BR,
        U_c       = U_c_arr,
        t         = t_arr,
        t_pd      = tpd_arr,
        t_pp      = tpp_arr,
        Delta_pd  = Dpd_arr,
        J_Hub     = JHub_arr,
        J_Em      = JEm_arr,
        enh_Hub   = enh_Hub,
        enh_Emery = enh_Emery,
        overpred  = overpred,
        U_corr    = U_corr,
        Nk        = Nk,
    )
