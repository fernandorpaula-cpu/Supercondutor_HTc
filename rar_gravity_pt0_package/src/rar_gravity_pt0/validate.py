r"""
PT0 validation: run each Crespi case, compute observables, compare to
targets, assign per-observable errors and an overall GO/NO-GO decision.

Error metric (per the prompt):
    err = |model - target| / |target|

Decision bands (per observable, and aggregated):
    err <= 0.01  -> EXCELLENT
    err <= 0.03  -> GO            (sufficient for a quantitative paper)
    err <= 0.10  -> BORDERLINE    (requires audit)
    err  > 0.10  -> NO-GO         (not good enough for a quantitative paper)

A target value of `null`/missing is reported as NO-TARGET and excluded
from the aggregate decision (it cannot help or hurt), but it is flagged
loudly so that a partial targets file cannot silently produce a GO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import (KEV, fermion_mass_kg, msun_to_kg, mpc_to_m, pc_to_m,
                        AU, YEAR)
from .orbit_s2 import S2Orbit, s2_observables
from .rar_tov_solver import CentralParams, core_radius_m, solve_profile
from .shooting import shoot_central_theta

EXCELLENT, GO, BORDERLINE, NOGO, NOTARGET = (
    "EXCELLENT", "GO", "BORDERLINE", "NO-GO", "NO-TARGET")


def classify(err: float | None) -> str:
    if err is None:
        return NOTARGET
    if err <= 0.01:
        return EXCELLENT
    if err <= 0.03:
        return GO
    if err <= 0.10:
        return BORDERLINE
    return NOGO


@dataclass
class ObsComparison:
    name: str
    unit: str
    model: float
    target: float | None
    rel_error: float | None
    verdict: str


@dataclass
class CaseResult:
    case_id: str
    mc2_keV: float
    converged: bool
    central_params: dict
    comparisons: list[ObsComparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    profile: Any = None
    verdict: str = NOTARGET


def _rel_error(model: float, target: float | None) -> float | None:
    if target is None or target == 0:
        return None
    return abs(model - target) / abs(target)


def load_targets(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_central_params(case: dict) -> tuple[CentralParams, float, list[str]]:
    """Construct central parameters for a case from the YAML spec.

    Required: mc2_keV, beta0, W0.  theta0 may be given directly, or the
    case may request shooting (handled by the caller) by omitting theta0.
    """
    notes: list[str] = []
    mc2_keV = float(case["mc2_keV"])
    m_kg = fermion_mass_kg(mc2_keV)
    beta0 = float(case["beta0"])
    W0 = float(case["W0"])
    theta0 = case.get("theta0", None)
    if theta0 is None:
        theta0 = 30.0  # placeholder; caller will shoot
        notes.append("theta0 not given -> shooting on central degeneracy")
    cp = CentralParams(theta0=float(theta0), beta0=beta0, W0=W0, m_kg=m_kg)
    return cp, mc2_keV, notes


def run_case(case_id: str, case: dict, solver_kwargs: dict | None = None) -> CaseResult:
    solver_kwargs = solver_kwargs or {}
    cp, mc2_keV, notes = _build_central_params(case)

    targets = case.get("targets", {}) or {}

    # --- solve (with optional shooting on total/core mass) ---
    converged = True
    shoot_target = case.get("shoot", None)
    if shoot_target is not None:
        tname = shoot_target["observable"]          # e.g. core_mass_kg
        # target value in M_sun in the YAML -> kg
        tval = msun_to_kg(float(shoot_target["target_msun"]))
        res = shoot_central_theta(
            beta0=cp.beta0, W0=cp.W0, m_kg=cp.m_kg,
            target_name=tname, target_value=tval,
            theta_bracket=tuple(shoot_target.get("theta_bracket", (5.0, 80.0))),
            solver_kwargs=solver_kwargs,
        )
        cp = res.central_params
        profile = res.profile
        converged = res.converged
        notes.append(
            f"shoot {tname}: theta0={cp.theta0:.4f}, "
            f"achieved={res.achieved_value:.4e} kg, "
            f"rel_err={res.rel_error:.3e}, converged={res.converged}")
    else:
        profile = solve_profile(cp, **solver_kwargs)

    # --- observables ---
    orbit = S2Orbit.default()
    obs = s2_observables(profile, orbit)

    model_vals = {
        "core_mass_msun": obs.extended_mass.core_mass_msun,
        "core_radius_pc": profile_core_radius_pc(profile),
        "total_mass_msun": profile.total_mass_kg / msun_to_kg(1.0),
        "extended_mass_within_s2_msun": obs.extended_mass.extended_mass_msun,
        "s2_precession_arcmin_per_orbit": obs.precession.delta_phi_total_arcmin,
    }
    units = {
        "core_mass_msun": "M_sun",
        "core_radius_pc": "pc",
        "total_mass_msun": "M_sun",
        "extended_mass_within_s2_msun": "M_sun",
        "s2_precession_arcmin_per_orbit": "arcmin/orbit",
    }

    comparisons: list[ObsComparison] = []
    for key, model in model_vals.items():
        target = targets.get(key, None)
        target = None if target is None else float(target)
        err = _rel_error(model, target)
        comparisons.append(ObsComparison(
            name=key, unit=units[key], model=model, target=target,
            rel_error=err, verdict=classify(err)))

    # aggregate verdict = worst verdict among observables that HAVE a target
    graded = [c for c in comparisons if c.rel_error is not None]
    if not graded:
        verdict = NOTARGET
        notes.append("no numeric targets present -> cannot decide GO/NO-GO")
    else:
        order = {EXCELLENT: 0, GO: 1, BORDERLINE: 2, NOGO: 3}
        verdict = max((c.verdict for c in graded), key=lambda v: order[v])

    return CaseResult(
        case_id=case_id, mc2_keV=mc2_keV, converged=converged,
        central_params=dict(theta0=cp.theta0, beta0=cp.beta0,
                            W0=cp.W0, m_kg=cp.m_kg),
        comparisons=comparisons, notes=notes, profile=profile, verdict=verdict)


def profile_core_radius_pc(profile) -> float:
    return core_radius_m(profile) / pc_to_m(1.0)


def aggregate_decision(results: list[CaseResult]) -> str:
    """Overall GO/NO-GO across all cases (worst case wins)."""
    order = {EXCELLENT: 0, GO: 1, BORDERLINE: 2, NOGO: 3, NOTARGET: -1}
    graded = [r.verdict for r in results if r.verdict != NOTARGET]
    if not graded:
        return NOTARGET
    return max(graded, key=lambda v: order[v])
