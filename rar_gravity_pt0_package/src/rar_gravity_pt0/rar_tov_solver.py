r"""
RAR / TOV structure solver.

We integrate the Tolman-Oppenheimer-Volkoff equations for a static,
spherically symmetric, self-gravitating fermion gas whose local
thermodynamic state is fixed by the relativistic equilibrium
(Tolman-Klein) conditions of the RAR model.

Metric:   ds^2 = e^{nu(r)} c^2 dt^2 - e^{lambda(r)} dr^2 - r^2 dOmega^2 .

Mass function and metric:
    dM/dr   = 4 pi r^2 rho(r)                      ,  rho = rho_E / c^2
    e^{-lambda} = 1 - 2 G M / (r c^2)
    dnu/dr  = (2 G / c^2) [ M + 4 pi r^3 P / c^2 ]
              / [ r^2 ( 1 - 2 G M / (r c^2) ) ]
    dP/dr   = -(rho_E + P) (1/2) dnu/dr            (automatically satisfied
                                                    by the equilibrium below)

Equilibrium conditions (RAR).  Only differences of nu matter, so we
integrate with nu(0) = 0 and rigidly shift nu afterwards to match the
exterior Schwarzschild solution.  With nu0 = nu(0) = 0:

    beta(r)  = beta0 * exp(-nu(r)/2)                      (Tolman)
    theta(r) = (1 + theta0 beta0)/beta0 - 1/beta(r)       (Klein)
    W(r)     = (1 + W0    beta0)/beta0 - 1/beta(r)         (cutoff energy
                                                            also redshifts)

Note  W(r) - theta(r) = W0 - theta0 = const  (constant cutoff depth).

The free central parameters of a configuration are (theta0, beta0, W0)
plus the fermion mass m.  Integration proceeds outward until the cutoff
parameter W(r) -> 0 (the gas truncates, density -> 0): that radius is the
surface R of the configuration.

UNITS: SI throughout.  Radii [m], masses [kg], densities [kg/m^3],
pressures [Pa], energy densities [J/m^3]; nu, lambda dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import C_LIGHT, G_NEWTON, G_SPIN
from .eos_fermion_cutoff import eos_state


@dataclass(frozen=True)
class CentralParams:
    """Central (r=0) RAR parameters that fully define a configuration."""
    theta0: float          # central degeneracy
    beta0: float           # central temperature parameter kT0/(mc^2)
    W0: float              # central cutoff parameter eps_c0/(kT0)
    m_kg: float            # fermion mass [kg]
    g: int = G_SPIN

    def __post_init__(self):
        if self.beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        if self.W0 <= 0:
            raise ValueError("W0 must be > 0 (need a finite cutoff)")


@dataclass
class Profile:
    """Radial profile of a solved configuration (all SI / dimensionless)."""
    r_m: np.ndarray
    mass_kg: np.ndarray
    rho_kg_m3: np.ndarray
    pressure_pa: np.ndarray
    nu_metric: np.ndarray
    lambda_metric: np.ndarray
    theta: np.ndarray
    beta: np.ndarray
    W: np.ndarray
    surface_radius_m: float
    total_mass_kg: float

    def enclosed_mass_kg(self, r_query_m):
        """Interpolated enclosed gravitating mass M(<r) [kg]."""
        r_query_m = np.atleast_1d(np.asarray(r_query_m, dtype=float))
        m = np.interp(r_query_m, self.r_m, self.mass_kg,
                      left=0.0, right=self.total_mass_kg)
        return m if m.size > 1 else float(m[0])


def _local_params(nu: float, cp: CentralParams):
    """Map nu(r) -> (theta, beta, W) via the equilibrium conditions."""
    beta = cp.beta0 * np.exp(-nu / 2.0)
    theta = (1.0 + cp.theta0 * cp.beta0) / cp.beta0 - 1.0 / beta
    W = (1.0 + cp.W0 * cp.beta0) / cp.beta0 - 1.0 / beta
    return theta, beta, W


def _state_at(nu: float, cp: CentralParams):
    """EOS (rho_E, rho, P) at metric potential nu."""
    theta, beta, W = _local_params(nu, cp)
    if W <= 0.0:
        return 0.0, 0.0, 0.0, theta, beta, W
    pt = eos_state(theta, W, beta, cp.m_kg, cp.g)
    return (pt.energy_density_j_m3, pt.mass_density_kg_m3,
            pt.pressure_pa, theta, beta, W)


def solve_profile(cp: CentralParams,
                  r_start_m: float = 1.0e6,
                  r_max_m: float = 1.0e21,
                  n_eval: int = 2000,
                  rtol: float = 1e-7,
                  atol_mass_kg: float = 1.0e18) -> Profile:
    """Integrate the RAR/TOV system outward from the centre.

    Parameters
    ----------
    cp        : central parameters.
    r_start_m : small starting radius to avoid the r=0 coordinate
                singularity (the central region M ~ (4/3)pi r^3 rho_c is
                seeded analytically).
    r_max_m   : hard upper bound on integration radius [m].
    n_eval    : number of log-spaced output samples.

    The integration stops when the cutoff W(r) reaches ~0 (surface).
    """
    rho_E_c, rho_c, P_c, *_ = _state_at(0.0, cp)
    if rho_c <= 0:
        raise ValueError("central density is zero; check central params")

    # Seed central values analytically (uniform-density core of radius r_start).
    M0 = (4.0 / 3.0) * np.pi * r_start_m**3 * rho_c
    nu0 = 0.0

    def rhs(r, y):
        M, nu = y
        rho_E, rho, P, theta, beta, W = _state_at(nu, cp)
        # guard the metric near the horizon (should not happen for dilute gas)
        f = 1.0 - 2.0 * G_NEWTON * M / (r * C_LIGHT**2)
        f = max(f, 1.0e-12)
        dMdr = 4.0 * np.pi * r**2 * rho
        dnudr = (2.0 * G_NEWTON / C_LIGHT**2) * (
            (M + 4.0 * np.pi * r**3 * P / C_LIGHT**2) / (r**2 * f)
        )
        return [dMdr, dnudr]

    def surface_event(r, y):
        _, _, _, _, _, W = _state_at(y[1], cp)
        return W - 1.0e-8
    surface_event.terminal = True
    surface_event.direction = -1.0

    r_eval = np.geomspace(r_start_m, r_max_m, n_eval)
    sol = solve_ivp(rhs, (r_start_m, r_max_m), [M0, nu0],
                    method="LSODA", t_eval=r_eval,
                    events=surface_event, rtol=rtol, atol=[atol_mass_kg, 1e-10],
                    max_step=r_max_m)

    r = sol.t
    M = sol.y[0]
    nu_raw = sol.y[1]

    # recompute local thermodynamics on the grid
    rho = np.empty_like(r)
    P = np.empty_like(r)
    theta = np.empty_like(r)
    beta = np.empty_like(r)
    W = np.empty_like(r)
    for i, (ri, nui) in enumerate(zip(r, nu_raw)):
        _, rho[i], P[i], theta[i], beta[i], W[i] = _state_at(nui, cp)

    surface_R = float(r[-1])
    total_M = float(M[-1])

    # lambda from the mass function
    lam = -np.log(np.clip(1.0 - 2.0 * G_NEWTON * M / (r * C_LIGHT**2),
                          1.0e-12, None))

    # rigidly shift nu so that e^{nu(R)} matches exterior Schwarzschild:
    #   e^{nu(R)} = 1 - 2 G M_tot / (R c^2)
    nu_surface_target = np.log(max(1.0 - 2.0 * G_NEWTON * total_M
                                   / (surface_R * C_LIGHT**2), 1.0e-12))
    nu = nu_raw + (nu_surface_target - nu_raw[-1])

    return Profile(
        r_m=r, mass_kg=M, rho_kg_m3=rho, pressure_pa=P,
        nu_metric=nu, lambda_metric=lam, theta=theta, beta=beta, W=W,
        surface_radius_m=surface_R, total_mass_kg=total_M,
    )


def core_radius_m(profile: Profile) -> float:
    """Estimate the dense-core radius as the first density e-folding
    (rho drops below rho_c / e).  Returns the surface radius if the gas is
    single-scale (no core-halo separation)."""
    rho = profile.rho_kg_m3
    r = profile.r_m
    if rho[0] <= 0:
        return r[0]
    threshold = rho[0] / np.e
    below = np.where(rho < threshold)[0]
    if below.size == 0:
        return profile.surface_radius_m
    return float(r[below[0]])
