r"""
Shooting solver for RAR configurations.

The RAR/TOV integrator (`rar_tov_solver.solve_profile`) takes the central
parameters (theta0, beta0, W0, m) and returns a full profile.  For the
PT0 reproduction we must instead hit *physical targets* (e.g. a total /
core mass of ~ 4e6 M_sun for Sgr A*).  This module performs the outer
shooting: it varies ONE central parameter (``theta0`` or ``beta0``) until
a chosen scalar observable of the profile matches a target value.

This is a genuine boundary-value solve (bisection on the central
parameter), NOT an interpolation between the 56 keV and 300 keV cases.
The prompt explicitly forbids the interpolation shortcut.

Which knob to vary
------------------
* ``theta0`` (central degeneracy): controls the core mass on the rising,
  NON-degenerate branch, but the core mass *saturates* once the core is
  fully degenerate -> use when the target sits below that plateau.
* ``beta0``  (central temperature parameter kT0/mc^2): moves the core
  along the degenerate branch up to the relativistic turning point
  (an OV-like maximum mass).  Use when ``theta0`` saturates below target.
  NOTE: beyond the turning point the core mass *decreases*; if the target
  exceeds the branch maximum the solve cannot bracket it and is honestly
  reported as ``converged=False`` (we return the closest config, never a
  fabricated success).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .rar_tov_solver import CentralParams, Profile, solve_profile, core_radius_m

SHOOTABLE = ("theta0", "beta0")


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
    branch_max: float | None = None   # branch maximum found (if target > max)


def _profile_observable(profile: Profile, name: str) -> float:
    """Scalar observables usable as shooting targets."""
    if name == "total_mass_kg":
        return profile.total_mass_kg
    if name == "core_mass_kg":
        return float(profile.enclosed_mass_kg(core_radius_m(profile)))
    if name == "surface_radius_m":
        return profile.surface_radius_m
    raise ValueError(f"unknown shooting observable {name!r}")


def _with_param(base: CentralParams, parameter: str, value: float) -> CentralParams:
    if parameter not in SHOOTABLE:
        raise ValueError(f"parameter must be one of {SHOOTABLE}, got {parameter!r}")
    return replace(base, **{parameter: value})


def shoot_parameter(base: CentralParams, parameter: str,
                    target_name: str, target_value: float,
                    bracket: tuple[float, float],
                    solver_kwargs: dict | None = None,
                    xtol: float | None = None,
                    max_iter: int = 100) -> ShootingResult:
    """Vary ``parameter`` (theta0 or beta0) of ``base`` until
    ``target_name`` matches ``target_value``.

    Robust to a non-monotonic (turning-point) response: if the target is
    not bracketed, we locate the branch maximum of the observable over the
    bracket and report ``converged=False`` with that maximum, so a target
    above the physical ceiling is flagged rather than silently faked.
    """
    solver_kwargs = solver_kwargs or {}
    n_calls = {"n": 0}
    lo, hi = bracket
    if xtol is None:
        xtol = 1e-4 * max(abs(lo), abs(hi), 1.0)

    def make_profile(val: float) -> Profile:
        return solve_profile(_with_param(base, parameter, val), **solver_kwargs)

    def obs(val: float) -> float:
        n_calls["n"] += 1
        return _profile_observable(make_profile(val), target_name)

    def resid(val: float) -> float:
        return obs(val) - target_value

    flo, fhi = resid(lo), resid(hi)

    if flo * fhi <= 0:
        # standard bracketed root
        sol = brentq(resid, lo, hi, xtol=xtol, maxiter=max_iter)
        prof = make_profile(sol)
        achieved = _profile_observable(prof, target_name)
        return ShootingResult(
            central_params=_with_param(base, parameter, sol),
            profile=prof, shot_parameter=parameter, shot_value=sol,
            target_name=target_name, target_value=target_value,
            achieved_value=achieved,
            rel_error=abs(achieved - target_value) / abs(target_value),
            converged=True, n_iter=n_calls["n"])

    # not bracketed: find the branch MAXIMUM of the observable to report the
    # physical ceiling honestly (maximise obs == minimise -obs).
    res = minimize_scalar(lambda v: -obs(v), bounds=(lo, hi), method="bounded",
                          options={"xatol": xtol})
    v_max = float(res.x)
    branch_max = -float(res.fun)
    prof = make_profile(v_max)
    achieved = _profile_observable(prof, target_name)
    return ShootingResult(
        central_params=_with_param(base, parameter, v_max),
        profile=prof, shot_parameter=parameter, shot_value=v_max,
        target_name=target_name, target_value=target_value,
        achieved_value=achieved,
        rel_error=abs(achieved - target_value) / abs(target_value),
        converged=False, n_iter=n_calls["n"], branch_max=branch_max)


def shoot_central_theta(beta0: float, W0: float, m_kg: float,
                        target_name: str, target_value: float,
                        theta_bracket=(5.0, 80.0),
                        solver_kwargs: dict | None = None,
                        xtol: float = 1e-4,
                        max_iter: int = 100) -> ShootingResult:
    """Backward-compatible wrapper: shoot on the central degeneracy theta0."""
    base = CentralParams(theta0=float(np.mean(theta_bracket)),
                         beta0=beta0, W0=W0, m_kg=m_kg)
    return shoot_parameter(base, "theta0", target_name, target_value,
                           bracket=tuple(theta_bracket),
                           solver_kwargs=solver_kwargs, xtol=xtol,
                           max_iter=max_iter)
