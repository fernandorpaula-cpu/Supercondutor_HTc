r"""
Relativistic Fermi-Dirac equation of state WITH energy cutoff (RAR model).

Physics
-------
The Ruffini-Arguelles-Rueda (RAR) model describes self-gravitating
fermions at finite temperature in general relativity.  To regularise the
otherwise infinite-mass isothermal halo, the phase-space distribution is
*truncated* at an escape (cutoff) energy eps_c (the "fermionic King"
distribution):

    f(eps) = [ 1 - exp((eps - eps_c)/kT) ] / [ exp((eps - mu)/kT) + 1 ],
             for eps <= eps_c ,   and   f = 0 for eps > eps_c .

Here `eps` is the *kinetic* energy of a particle,
    eps(p) = sqrt(p^2 c^2 + m^2 c^4) - m c^2 ,
`mu` is the chemical potential (rest mass subtracted) and `T` the
temperature.  g = 2 is the spin degeneracy.

Dimensionless local variables (functions of radius in the full problem):
    theta = mu  / (kT)      degeneracy parameter
    W     = eps_c / (kT)    cutoff   parameter
    beta  = kT / (m c^2)    temperature parameter

Thermodynamic densities (SI):
    n       = C (mc)^3              * I_n     [1/m^3]
    rho_E   = C (mc)^3 (mc^2)       * I_rho   [J/m^3]   (energy density)
    rho     = rho_E / c^2                     [kg/m^3]  (gravitating mass density)
    P       = C (mc)^3 (mc^2) / 3   * I_P     [Pa]
with C = g / (2 pi^2 hbar^3).

The dimensionless truncated Fermi integrals, using x = p/(mc):
    I_n   = \int_0^{x_c}  f(x) x^2 dx
    I_rho = \int_0^{x_c}  sqrt(1+x^2) f(x) x^2 dx
    I_P   = \int_0^{x_c}  x^4 / sqrt(1+x^2) f(x) dx
where  u(x) = (sqrt(1+x^2) - 1)/beta = eps/kT ,
       f(x) = (1 - exp(u - W)) / (exp(u - theta) + 1) ,  u <= W ,
       x_c  = sqrt((1 + W*beta)^2 - 1)   (so that u(x_c) = W).

UNITS: every returned quantity is SI and named accordingly.  Inputs
theta, W, beta are dimensionless.  `m_kg` is the fermion mass in kg.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .constants import C_LIGHT, G_SPIN, eos_prefactor


@dataclass(frozen=True)
class EOSPoint:
    """Local thermodynamic state at one point of the star (all SI)."""
    energy_density_j_m3: float   # rho_E   [J/m^3]
    mass_density_kg_m3: float    # rho     [kg/m^3]   ( = rho_E / c^2 )
    pressure_pa: float           # P       [Pa]
    number_density_per_m3: float # n       [1/m^3]


def _x_cutoff(beta: float, W: float) -> float:
    """Upper momentum bound x_c such that the kinetic energy equals eps_c."""
    if W <= 0.0 or beta <= 0.0:
        return 0.0
    val = (1.0 + W * beta) ** 2 - 1.0
    return float(np.sqrt(val)) if val > 0.0 else 0.0


def _u_of_x(x: float, beta: float) -> float:
    """Dimensionless kinetic energy u = eps/kT."""
    return (np.sqrt(1.0 + x * x) - 1.0) / beta


def occupation(x, theta: float, W: float, beta: float):
    """Truncated Fermi-Dirac occupation f(x), vectorised, numerically safe.

    Returns 0 outside [0, x_c].  The exponentials are clipped to avoid
    overflow in the degenerate (theta >> 1) regime.
    """
    x = np.asarray(x, dtype=float)
    u = _u_of_x(x, beta)
    # cutoff numerator: (1 - e^{u-W}); fermi denominator: (e^{u-theta}+1)
    a = np.clip(u - W, -700.0, 700.0)
    b = np.clip(u - theta, -700.0, 700.0)
    num = 1.0 - np.exp(a)
    den = np.exp(b) + 1.0
    f = num / den
    f = np.where(u <= W, f, 0.0)
    return np.clip(f, 0.0, 1.0)


def _integral(kind: str, theta: float, W: float, beta: float) -> float:
    """Compute one of the dimensionless truncated Fermi integrals."""
    x_c = _x_cutoff(beta, W)
    if x_c <= 0.0:
        return 0.0

    if kind == "n":
        integrand = lambda x: occupation(x, theta, W, beta) * x * x
    elif kind == "rho":
        integrand = lambda x: occupation(x, theta, W, beta) * np.sqrt(1.0 + x * x) * x * x
    elif kind == "P":
        integrand = lambda x: occupation(x, theta, W, beta) * x**4 / np.sqrt(1.0 + x * x)
    else:  # pragma: no cover
        raise ValueError(f"unknown integral kind {kind!r}")

    # Split the interval to help the quadrature resolve the Fermi edge
    # near x ~ x_theta where the occupation drops from ~1 to ~0.
    x_theta = 0.0
    if theta > 0.0:
        arg = (1.0 + theta * beta) ** 2 - 1.0
        if arg > 0.0:
            x_theta = min(np.sqrt(arg), x_c)
    points = sorted({0.0, x_theta, x_c})
    total = 0.0
    for lo, hi in zip(points[:-1], points[1:]):
        if hi > lo:
            val, _ = quad(lambda x: float(integrand(x)), lo, hi, limit=200)
            total += val
    return total


def eos_state(theta: float, W: float, beta: float, m_kg: float,
              g: int = G_SPIN) -> EOSPoint:
    """Evaluate the EOS at one (theta, W, beta) point.

    Parameters
    ----------
    theta, W, beta : dimensionless degeneracy / cutoff / temperature params.
    m_kg           : fermion mass [kg].
    g              : spin degeneracy (default 2).

    Returns
    -------
    EOSPoint with SI quantities.
    """
    mc = m_kg * C_LIGHT            # [kg m/s]
    mc2 = m_kg * C_LIGHT**2        # [J]
    C = eos_prefactor(g)          # [1/(J^3 s^3)]
    scale_n = C * mc**3           # [1/m^3]

    I_n = _integral("n", theta, W, beta)
    I_rho = _integral("rho", theta, W, beta)
    I_P = _integral("P", theta, W, beta)

    n = scale_n * I_n                      # [1/m^3]
    rho_E = scale_n * mc2 * I_rho          # [J/m^3]
    P = scale_n * mc2 * I_P / 3.0          # [Pa]
    rho = rho_E / C_LIGHT**2               # [kg/m^3]

    return EOSPoint(
        energy_density_j_m3=rho_E,
        mass_density_kg_m3=rho,
        pressure_pa=P,
        number_density_per_m3=n,
    )


def central_degeneracy_from_density(rho_c_kg_m3: float, beta0: float, W0: float,
                                    m_kg: float, g: int = G_SPIN,
                                    theta_bracket=(-50.0, 200.0),
                                    tol: float = 1e-6) -> float:
    """Invert the EOS at the centre: find theta0 reproducing a target central
    mass density rho_c (kg/m^3) at fixed (beta0, W0).

    Useful when Crespi's Table 4 specifies a central density rather than a
    central degeneracy.  Monotonic in theta -> simple bisection.
    """
    from scipy.optimize import brentq

    def resid(theta):
        return eos_state(theta, W0, beta0, m_kg, g).mass_density_kg_m3 - rho_c_kg_m3

    lo, hi = theta_bracket
    flo, fhi = resid(lo), resid(hi)
    if flo * fhi > 0:
        raise ValueError(
            "central density not bracketed by theta range "
            f"[{lo},{hi}] -> residuals ({flo:.3e},{fhi:.3e}); "
            "adjust beta0/W0 or the bracket."
        )
    return float(brentq(resid, lo, hi, xtol=tol))
