"""
src/mediator_rpa.py — Spin-fluctuation pairing mediator via RPA susceptibility.

LABEL (mandatory — do not remove):
    "mediador paramagnon identificado em nível de modelo RPA;
     hipótese de consistência, não prova experimental"

Physical model
--------------
The bare (Lindhard) static susceptibility is computed for a square-lattice
tight-binding band under pressure and dressed by the RPA:

    χ₀(q) = -(1/N) Σ_k [f(ε_k) - f(ε_{k+q})] / (ε_{k+q} - ε_k)

    χ_RPA(q) = χ₀(q) / [1 - U χ₀(q)]

In the t-t' cuprate band, χ₀ peaks near q = (π, π) due to Fermi-surface
nesting.  The RPA enhances this peak: χ_RPA(Q_AFM) >> χ_RPA(0).

Singlet spin-fluctuation vertex (Bickers-Scalapino-White, PRL 1987):
    V_sing(k-k') = (3/2) U² χ_RPA(k-k')

This interaction is effectively attractive in the d_{x²-y²} channel because
g_d(k) and g_d(k+Q) have opposite signs, producing a positive eigenvalue:
    λ_d = -<g_d|V_sing|g_d>_FS / <g_d²>_FS > 0

while the s-wave eigenvalue is negative (repulsive for uniform s).

Paramagnon energy scale:
    ω_sf(P) = 1 / χ_RPA(Q_AFM, P)  [eV]

Dimensionless mediator coupling:
    λ_med(P) = N(0,P) × (3/2) U² × χ_RPA(Q_AFM, P)  [dimensionless]

Parameter provenance
--------------------
U_HUB = 0.30 eV      — [ASSUMED] sub-critical Hubbard U (Stoner < 1 everywhere)
NK_CHI = 32           — [ASSUMED] k/q-grid size for chi0 calculation
T_RPA_EV = 0.005 eV  — [ASSUMED] finite-T regularisation for Lindhard integral
ETA_FS_EV = 0.015 eV — [ASSUMED] Lorentzian FS broadening for eigenvalue integrals

Stoner criterion
----------------
The Stoner parameter S(P) = U × χ₀(Q_AFM, P) must satisfy S < 1 for the
paramagnon description to be valid.  The code warns if S > STONER_THRESHOLD.
S → 1 signals an antiferromagnetic instability; the present model applies only
in the paramagnon (disordered) phase.

Interpretation constraints (mandatory)
---------------------------------------
1.  The RPA is a weak-coupling approximation.  It is NOT a controlled
    expansion for U/t ~ 0.7 (our case); the paramagnon description is a
    hypothesis, not a proof.
2.  λ_d computed here is a d-wave pairing EIGENVALUE ESTIMATE from the
    spin-fluctuation vertex, NOT the same as λ_hop + λ_exch in channels.py
    (which is a kinematic/exchange decomposition at a different level).
3.  ω_sf is the characteristic paramagnon energy: it sets the mediator
    frequency scale, not the superconducting Tc directly.
4.  The pressure dependence of λ_d and ω_sf is phenomenological: the model
    captures trends (t'/t and FS topology evolution) but does not account for
    retardation, vertex corrections, or self-energy renormalisation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .lattice_bands import (
    build_kgrid, dispersion_square,
    t_of_P, tprime_of_P, mu_of_P,
    fermi_function, P_GRID_GPA,
)

# ---------------------------------------------------------------------------
# Constants  (all labeled per provenance scheme)
# ---------------------------------------------------------------------------

LABEL_RPA: str = (
    "mediador paramagnon identificado em nível de modelo RPA; "
    "hipótese de consistência, não prova experimental"
)

# Hubbard U [eV] — [ASSUMED] sub-critical for Stoner everywhere in P range
U_HUB: float = 0.30

# k/q-grid for chi0 susceptibility calculation
# Larger Nk = more accurate but O(Nk⁴) cost
NK_CHI: int = 32

# Small finite temperature for Lindhard regularisation [eV] — [ASSUMED]
T_RPA_EV: float = 0.005

# Lorentzian FS broadening for eigenvalue projections [eV] — [ASSUMED]
ETA_FS_EV: float = 0.015

# Warn if Stoner parameter U chi0(Q_AFM) exceeds this
STONER_THRESHOLD: float = 0.95

# AFM wavevector indices on the NK_CHI grid: q = (π, π) at shift Nk//2
# (roll-based convention: chi0[Nk//2, Nk//2] = chi0 at q=(π,π))
def _q_afm_idx(Nk: int) -> tuple[int, int]:
    return (Nk // 2, Nk // 2)


# ---------------------------------------------------------------------------
# Form factors
# ---------------------------------------------------------------------------

def d_wave_form(kx: NDArray, ky: NDArray) -> NDArray:
    """d_{x²-y²} form factor g_d(k) = cos kx − cos ky."""
    return np.cos(kx) - np.cos(ky)


def s_wave_form(kx: NDArray, ky: NDArray) -> NDArray:
    """Uniform s-wave form factor g_s(k) = 1."""
    return np.ones_like(kx)


# ---------------------------------------------------------------------------
# Lindhard / RPA susceptibility
# ---------------------------------------------------------------------------

def chi0_static(eps_k: NDArray, T_eV: float = T_RPA_EV) -> NDArray:
    """
    Static Lindhard susceptibility χ₀(q) on the q-grid defined by the k-grid.

    χ₀[iqx, iqy] = -(1/N) Σ_k [f(ε_k) - f(ε_{k+q})] / (ε_{k+q} - ε_k)

    where q corresponds to a shift of (iqx, iqy) grid steps:
        q_x = iqx × 2π / Nk   (q=0 at iqx=0, q=π at iqx=Nk//2)

    The L'Hôpital limit at ε_{k+q} → ε_k is handled by replacing the
    integrand with f(ε_k)[1-f(ε_k)] / T_eV.

    Returns χ₀ in units of 1/eV.
    """
    Nk = eps_k.shape[0]
    chi0 = np.zeros((Nk, Nk), dtype=float)
    f_k = fermi_function(eps_k, T_eV)

    for iqx in range(Nk):
        for iqy in range(Nk):
            eps_q = np.roll(np.roll(eps_k, -iqx, axis=0), -iqy, axis=1)
            f_q = np.roll(np.roll(f_k, -iqx, axis=0), -iqy, axis=1)
            denom = eps_q - eps_k
            small = np.abs(denom) < 1e-9
            # L'Hôpital: ∂f/∂ε evaluated at ε_k
            lhop = f_k * (1.0 - f_k) / T_eV
            integrand = np.where(
                small,
                lhop,
                (f_k - f_q) / np.where(small, 1.0, denom),
            )
            # No leading minus: (f_k - f_q)/(eps_q - eps_k) is already positive
            # at q→0 (L'Hôpital gives f(1-f)/T > 0) and at nesting q=(π,π).
            chi0[iqx, iqy] = np.mean(integrand)

    return chi0


def chi_rpa_from_chi0(chi0: NDArray, U: float = U_HUB) -> NDArray:
    """
    RPA susceptibility χ_RPA(q) = χ₀(q) / [1 - U χ₀(q)].

    Warns if the Stoner criterion U max(χ₀) ≥ STONER_THRESHOLD is approached.

    Returns χ_RPA in units of 1/eV.
    """
    stoner = U * chi0
    s_max = float(stoner.max())
    if s_max >= STONER_THRESHOLD:
        warnings.warn(
            f"Stoner parameter U χ₀_max = {s_max:.3f} ≥ {STONER_THRESHOLD}. "
            "Approaching antiferromagnetic instability; RPA paramagnon picture "
            "may not be reliable.",
            stacklevel=2,
        )
    denom = 1.0 - stoner
    # Guard against division by zero (clamping)
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return chi0 / denom_safe


def V_singlet(chi_rpa: NDArray, U: float = U_HUB) -> NDArray:
    """
    Singlet spin-fluctuation pairing vertex V_sing(q) = (3/2) U² χ_RPA(q).

    Units: eV (since [U²] = eV², [χ_RPA] = 1/eV).

    See: Bickers, Scalapino & White, PRL 62, 961 (1989).

    Interpretation: V_sing > 0 everywhere, but acts as an ATTRACTIVE
    potential in the d-wave channel because g_d(k) g_d(k+Q) < 0 at Q=(π,π).
    """
    return 1.5 * U**2 * chi_rpa


# ---------------------------------------------------------------------------
# Fermi-surface eigenvalue projections
# ---------------------------------------------------------------------------

def _fs_lorentz(eps_k: NDArray, eta: float) -> NDArray:
    """Lorentzian FS weight w(k) = η/(π(ε²+η²)) [1/eV]."""
    return eta / (np.pi * (eps_k**2 + eta**2))


def N0_at_EF(eps_k: NDArray, eta: float = ETA_FS_EV) -> float:
    """
    DOS at Fermi level N(0) = (1/N) Σ_k L(ε_k, η)  [1/eV].

    Uses Lorentzian broadening rather than histogram for smooth derivatives.
    """
    return float(np.mean(_fs_lorentz(eps_k, eta)))


def lambda_channel(
    eps_k: NDArray,
    g_k: NDArray,
    V_sing: NDArray,
    eta: float = ETA_FS_EV,
) -> float:
    """
    Pairing eigenvalue for channel with form factor g(k) via FFT convolution.

    λ_g = -Σ_{k,k'} w_k g_k V(k-k') w_{k'} g_{k'} / Σ_k w_k g_k²

    V(k-k') = V_sing at shift index (k-k') mod N  (roll convention).

    The double sum is evaluated via:
        Σ_k (wg)_k [V ⊛ (wg)]_k = (1/N²) Σ_q FFT(V)(q) |FFT(wg)(q)|²

    where ⊛ denotes circular convolution (IFFT[FFT(V)·FFT(wg)]).

    Returns dimensionless λ_g.  λ_d > 0 (attractive) and λ_s < 0 (repulsive)
    for the spin-fluctuation vertex.
    """
    w = _fs_lorentz(eps_k, eta)
    wg = w * g_k
    N = float(eps_k.size)  # total k-points for normalisation

    # Circular convolution: (V ⊛ wg)[k] = Σ_{k'} V[(k-k') % N] wg[k']
    conv = np.fft.ifft2(np.fft.fft2(V_sing) * np.fft.fft2(wg)).real

    # The inner sum over k' produces a factor of N; divide by N so that
    # the result equals the mean <g V g>_FS rather than N × mean.
    # Verified: for uniform V0, this gives λ_s = −N(0) V0 (standard BCS).
    numerator = float(np.sum(wg * conv)) / N
    denominator = float(np.sum(w * g_k**2))

    if abs(denominator) < 1e-14:
        return 0.0
    return -numerator / denominator


# ---------------------------------------------------------------------------
# Paramagnon energy and mediator coupling
# ---------------------------------------------------------------------------

def omega_sf(chi_rpa: NDArray, Nk: int) -> float:
    """
    Paramagnon characteristic energy ω_sf(P) = 1 / χ_RPA(Q_AFM)  [eV].

    Derivation: in the Ornstein-Zernike form χ_RPA(q) ~ χ_Q / (1 + ξ²δq²),
    the inverse peak height sets the spin-fluctuation energy scale.
    ω_sf → 0 as χ_RPA(Q_AFM) → ∞ (i.e., as the Stoner instability is
    approached and antiferromagnetic order sets in).

    Returns ω_sf in eV.
    """
    iq, jq = _q_afm_idx(Nk)
    chi_Q = float(chi_rpa[iq, jq])
    if chi_Q < 1e-12:
        return np.inf
    return 1.0 / chi_Q


def lambda_mediator(N0: float, chi_rpa: NDArray, U: float, Nk: int) -> float:
    """
    Dimensionless spin-fluctuation coupling at Q_AFM.

    λ_med = N(0) × (3/2) U² × χ_RPA(Q_AFM)  [dimensionless]

    This is the McMillan-like coupling strength contributed by the peak of the
    spin-fluctuation spectrum.  It differs from λ_d (which integrates over all
    q with d-wave form factors) but tracks the same physical enhancement.
    """
    iq, jq = _q_afm_idx(Nk)
    chi_Q = float(chi_rpa[iq, jq])
    return N0 * 1.5 * U**2 * chi_Q


# ---------------------------------------------------------------------------
# Stoner parameter diagnostic
# ---------------------------------------------------------------------------

def stoner_param(chi0: NDArray, U: float = U_HUB) -> float:
    """S = U × max(χ₀)  [dimensionless].  Must be < 1 for paramagnon picture."""
    return float(U * chi0.max())


# ---------------------------------------------------------------------------
# Full pressure scan
# ---------------------------------------------------------------------------

@dataclass
class RPAScanResult:
    """Container for the full RPA pressure scan output."""
    P:          NDArray
    Stoner:     NDArray   # U χ₀(Q_AFM) at each P
    lambda_d:   NDArray   # d-wave eigenvalue
    lambda_s:   NDArray   # s-wave eigenvalue
    ratio_d_s:  NDArray   # λ_d / |λ_s| — channel selectivity
    omega_sf:   NDArray   # paramagnon energy ω_sf [eV]
    lambda_med: NDArray   # dimensionless peak coupling
    chi0_P0:    NDArray   # bare χ₀ map at P=0 (Nk × Nk)
    chi_rpa_P0: NDArray   # RPA χ map at P=0 (Nk × Nk)
    Nk:         int = NK_CHI
    U:          float = U_HUB
    label:      str = LABEL_RPA


def rpa_pressure_scan(
    P_grid: NDArray = P_GRID_GPA,
    Nk_chi: int = NK_CHI,
    T_rpa: float = T_RPA_EV,
    U: float = U_HUB,
    eta_fs: float = ETA_FS_EV,
) -> RPAScanResult:
    """
    Compute RPA susceptibility diagnostics over a pressure grid.

    At each pressure:
      1. Build ε(k,P) from lattice_bands parameters.
      2. Compute χ₀(q, P) via Lindhard sum on Nk_chi × Nk_chi grid.
      3. Dress to χ_RPA = χ₀ / (1 - U χ₀).
      4. Extract V_sing(q) = (3/2) U² χ_RPA.
      5. Project onto d-wave and s-wave: λ_d, λ_s.
      6. Extract ω_sf = 1 / χ_RPA(Q_AFM) and λ_med = N(0) V_peak.

    Returns RPAScanResult with all arrays.
    """
    P = np.asarray(P_grid, dtype=float)
    n = len(P)
    kx, ky = build_kgrid(Nk_chi, Nk_chi)
    g_d = d_wave_form(kx, ky)
    g_s = s_wave_form(kx, ky)

    Stoner     = np.zeros(n)
    ld         = np.zeros(n)
    ls         = np.zeros(n)
    osf        = np.zeros(n)
    lmed       = np.zeros(n)

    chi0_P0    = None
    chi_rpa_P0 = None

    for i, Pi in enumerate(P):
        t  = float(t_of_P(Pi))
        tp = float(tprime_of_P(Pi))
        mu = float(mu_of_P(Pi))

        eps_k = dispersion_square(kx, ky, t, tp, mu)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi0    = chi0_static(eps_k, T_eV=T_rpa)
            chi_rpa = chi_rpa_from_chi0(chi0, U)

        iq, jq = _q_afm_idx(Nk_chi)
        Stoner[i] = U * float(chi0[iq, jq])

        V_sing = V_singlet(chi_rpa, U)
        N0     = N0_at_EF(eps_k, eta_fs)

        ld[i]   = lambda_channel(eps_k, g_d, V_sing, eta_fs)
        ls[i]   = lambda_channel(eps_k, g_s, V_sing, eta_fs)
        osf[i]  = omega_sf(chi_rpa, Nk_chi)
        lmed[i] = lambda_mediator(N0, chi_rpa, U, Nk_chi)

        if i == 0:
            chi0_P0    = chi0.copy()
            chi_rpa_P0 = chi_rpa.copy()

    denom_s = np.abs(ls)
    ratio = np.where(denom_s > 1e-12, ld / denom_s, np.nan)

    return RPAScanResult(
        P          = P,
        Stoner     = Stoner,
        lambda_d   = ld,
        lambda_s   = ls,
        ratio_d_s  = ratio,
        omega_sf   = osf,
        lambda_med = lmed,
        chi0_P0    = chi0_P0,
        chi_rpa_P0 = chi_rpa_P0,
        Nk         = Nk_chi,
        U          = U,
    )
