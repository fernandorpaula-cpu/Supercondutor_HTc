r"""
Shooting solver for RAR configurations.

The RAR/TOV integrator (`rar_tov_solver.solve_profile`) takes the central
parameters (theta0, beta0, W0, m) and returns a full profile.  For the
PT0 reproduction we must instead hit *physical targets* (e.g. a total /
core mass of ~ 4e6 M_sun for Sgr A*).  This module performs the outer
shooting: it varies one central parameter until a chosen scalar
observable of the profile matches a target value.

This is a genuine boundary-value solve (Newton/bisection on the central
parameter), NOT an interpolation between the 56 keV and 300 keV cases.
The prompt explicitly forbids the interpolation shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .rar_tov_solver import CentralParams, Profile, solve_profile, core_radius_m


@dataclass
class ShootingResult:
    central_params: CentralParams
    profile: Profile
    shot_parameter: str
    shot_value: float
    target_name: str
    target_value: float
    achieved_value: float
    rel_error: float
    converged: bool
    n_iter: int


def _profile_observable(profile: Profile, name: str) -> float:
    """Scalar observables usable as shooting targets."""
    if name == "total_mass_kg":
        return profile.total_mass_kg
    if name == "core_mass_kg":
        return float(profile.enclosed_mass_kg(core_radius_m(profile)))
    if name == "surface_radius_m":
        return profile.surface_radius_m
    raise ValueError(f"unknown shooting observable {name!r}")


def shoot_central_theta(beta0: float, W0: float, m_kg: float,
                        target_name: str, target_value: float,
                        theta_bracket=(5.0, 80.0),
                        solver_kwargs: dict | None = None,
                        xtol: float = 1e-4,
                        max_iter: int = 100) -> ShootingResult:
    """Vary the central degeneracy theta0 until `target_name` matches
    `target_value`.

    theta0 controls how degenerate (and therefore how massive/compact) the
    fermion core is, so the chosen observable is monotonic in theta0 over a
    sensible range, making bisection robust.
    """
    solver_kwargs = solver_kwargs or {}
    n_calls = {"n": 0}

    def make_profile(theta0: float) -> Profile:
        cp = CentralParams(theta0=theta0, beta0=beta0, W0=W0, m_kg=m_kg)
        return solve_profile(cp, **solver_kwargs)

    def resid(theta0: float) -> float:
        n_calls["n"] += 1
        prof = make_profile(theta0)
        return _profile_observable(prof, target_name) - target_value

    lo, hi = theta_bracket
    flo, fhi = resid(lo), resid(hi)
    if flo * fhi > 0:
        # not bracketed: return the closest endpoint, flagged not-converged
        theta_best = lo if abs(flo) < abs(fhi) else hi
        prof = make_profile(theta_best)
        achieved = _profile_observable(prof, target_name)
        return ShootingResult(
            central_params=CentralParams(theta_best, beta0, W0, m_kg),
            profile=prof, shot_parameter="theta0", shot_value=theta_best,
            target_name=target_name, target_value=target_value,
            achieved_value=achieved,
            rel_error=abs(achieved - target_value) / abs(target_value),
            converged=False, n_iter=n_calls["n"],
        )

    theta_sol = brentq(resid, lo, hi, xtol=xtol, maxiter=max_iter)
    prof = make_profile(theta_sol)
    achieved = _profile_observable(prof, target_name)
    return ShootingResult(
        central_params=CentralParams(theta_sol, beta0, W0, m_kg),
        profile=prof, shot_parameter="theta0", shot_value=theta_sol,
        target_name=target_name, target_value=target_value,
        achieved_value=achieved,
        rel_error=abs(achieved - target_value) / abs(target_value),
        converged=True, n_iter=n_calls["n"],
    )
