"""
src/null_models.py — Statistical null models for Tc_zero(P) and Wtr(P).

Purpose
-------
Fit three phenomenological curves (linear, quadratic, saturating) to
experimental Tc_zero(P) and Wtr(P) from HG1212_DATA.  Compute RMSE, MAE,
bootstrap 95 % CI, and design-matrix conditioning index ρ for each model.

MANDATORY RULE (print at every summary):
    "qualidade de interpolação não implica mecanismo físico"
    (quality of interpolation does not imply a physical mechanism)

These null models are benchmarks only.  A low RMSE proves only that the
functional form can describe the pressure dependence — it says nothing about
the underlying physics.  In particular:
  - A saturating model does NOT imply "saturation of coherence at high P".
  - A quadratic model does NOT imply "quadratic coupling to the lattice".
  - C_coh(P) computed here is distinct from the dome-shaped C_coh in channels.py.

Models implemented
------------------
  linear:     f(P; a, b)      = a + b*P                  (2 params)
  quadratic:  f(P; a, b, c)   = a + b*P + c*P²           (3 params)
  saturating: f(P; a, b, Pc)  = a + b*(1 - exp(-P/Pc))   (3 params)

Degenerate fits
---------------
The saturating model on Wtr(P) is expected to be degenerate: the optimizer
finds b → ∞ and Pc → ∞ such that b*(1-exp(-P/Pc)) ≈ b*P/Pc ≈ linear.
This is detected by a conditioning index ρ > DEGENERACY_THRESHOLD and
reported honestly in the summary table.

Bootstrap
---------
sigma_T = 1.5 K (user-specified).
n_boot and seed come from config.NULL_MODELS.
Each bootstrap replicate perturbs the observed data with N(0, σ_T²), refits,
and stores the model curve on a fine pressure grid.  The 95 % CI is the
[2.5, 97.5] percentile band of the resampled curves.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.linalg import svd

from .two_scale import HG1212_DATA, wtr_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMA_T: float = 1.5          # bootstrap noise [K]
DEGENERACY_THRESHOLD: float = 1e4  # ρ above this → flag as degenerate
RULE: str = (
    "qualidade de interpolação não implica mecanismo físico"
)

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def model_linear(P: NDArray, a: float, b: float) -> NDArray:
    return a + b * np.asarray(P, float)


def model_quadratic(P: NDArray, a: float, b: float, c: float) -> NDArray:
    P = np.asarray(P, float)
    return a + b * P + c * P ** 2


def model_saturating(P: NDArray, a: float, b: float, Pc: float) -> NDArray:
    P = np.asarray(P, float)
    Pc = max(abs(Pc), 1e-6)
    return a + b * (1.0 - np.exp(-P / Pc))


_MODEL_FNS: dict[str, Callable] = {
    "linear": model_linear,
    "quadratic": model_quadratic,
    "saturating": model_saturating,
}

_P0: dict[str, list[float]] = {
    "linear":     [120.0, 0.5],
    "quadratic":  [120.0, 2.0, -0.05],
    "saturating": [120.0, 15.0, 5.0],
}

_P0_WTR: dict[str, list[float]] = {
    "linear":     [3.0, 0.4],
    "quadratic":  [3.0, 0.3, 0.005],
    "saturating": [3.0, 10.0, 8.0],
}

# ---------------------------------------------------------------------------
# Design-matrix conditioning index
# ---------------------------------------------------------------------------

def conditioning_index(P: NDArray, model_name: str, popt: NDArray) -> float:
    """
    Condition number of the Jacobian evaluated at popt.

    For linear/quadratic models this equals the condition number of the
    Vandermonde design matrix (exact).  For the saturating model the Jacobian
    is computed numerically via finite differences.
    """
    P = np.asarray(P, float)
    fn = _MODEL_FNS[model_name]
    eps = 1e-5
    J = np.zeros((len(P), len(popt)))
    f0 = fn(P, *popt)
    for j, p in enumerate(popt):
        dp = np.zeros_like(popt)
        dp[j] = eps * (abs(p) + 1e-10)
        J[:, j] = (fn(P, *(popt + dp)) - f0) / dp[j]
    _, s, _ = svd(J, full_matrices=False)
    if s[-1] < 1e-15:
        return np.inf
    return float(s[0] / s[-1])


# ---------------------------------------------------------------------------
# Single-model fit
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    model_name: str
    observable: str              # 'Tc_zero' or 'Wtr'
    popt: NDArray
    pcov: NDArray
    rmse: float
    mae: float
    rho: float                   # conditioning index
    degenerate: bool
    residuals: NDArray           # observed - predicted, per pressure point
    predicted: NDArray


def fit_one(
    P: NDArray,
    y_obs: NDArray,
    model_name: str,
    observable: str,
    p0_override: list[float] | None = None,
) -> FitResult:
    fn = _MODEL_FNS[model_name]
    p0 = p0_override or (_P0_WTR if observable == "Wtr" else _P0)[model_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov = curve_fit(fn, P, y_obs, p0=p0, maxfev=10_000)
        except RuntimeError:
            popt = np.array(p0, float)
            pcov = np.full((len(p0), len(p0)), np.nan)

    predicted = fn(P, *popt)
    resid = y_obs - predicted
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    rho = conditioning_index(P, model_name, popt)
    degenerate = rho > DEGENERACY_THRESHOLD or not np.isfinite(rho)

    return FitResult(
        model_name=model_name,
        observable=observable,
        popt=popt,
        pcov=pcov,
        rmse=rmse,
        mae=mae,
        rho=rho,
        degenerate=degenerate,
        residuals=resid,
        predicted=predicted,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    model_name: str
    observable: str
    P_fine: NDArray
    ci_lo: NDArray
    ci_hi: NDArray
    n_boot: int
    sigma_T: float = SIGMA_T


def bootstrap_fit(
    P: NDArray,
    y_obs: NDArray,
    model_name: str,
    observable: str,
    n_boot: int = 500,
    seed: int = 42,
    sigma_T: float = SIGMA_T,
    P_fine: NDArray | None = None,
) -> BootstrapResult:
    if P_fine is None:
        P_fine = np.linspace(P[0], P[-1], 300)

    fn = _MODEL_FNS[model_name]
    p0 = (_P0_WTR if observable == "Wtr" else _P0)[model_name]
    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        y_perturbed = y_obs + rng.normal(0.0, sigma_T, size=len(y_obs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = curve_fit(fn, P, y_perturbed, p0=p0, maxfev=10_000)
                curves.append(fn(P_fine, *popt))
            except RuntimeError:
                pass

    curves = np.array(curves)
    ci_lo = np.percentile(curves, 2.5, axis=0)
    ci_hi = np.percentile(curves, 97.5, axis=0)

    return BootstrapResult(
        model_name=model_name,
        observable=observable,
        P_fine=P_fine,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_boot=len(curves),
        sigma_T=sigma_T,
    )


# ---------------------------------------------------------------------------
# Full null-model analysis
# ---------------------------------------------------------------------------

@dataclass
class NullModelAnalysis:
    observable: str
    P: NDArray
    y_obs: NDArray
    fits: dict[str, FitResult] = field(default_factory=dict)
    boots: dict[str, BootstrapResult] = field(default_factory=dict)
    best_model: str = ""

    def best_rmse(self) -> float:
        return min(f.rmse for f in self.fits.values())


def run_null_analysis(
    data: dict = HG1212_DATA,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, NullModelAnalysis]:
    """
    Fit all null models to Tc_zero(P) and Wtr(P) from data.

    Returns dict with keys 'Tc_zero' and 'Wtr'.
    """
    P = data["P_GPa"]
    observables = {
        "Tc_zero": data["Tc_zero_K"],
        "Wtr":     wtr_data(data),
    }

    P_fine = np.linspace(P[0], P[-1], 300)
    results: dict[str, NullModelAnalysis] = {}

    for obs_name, y_obs in observables.items():
        ana = NullModelAnalysis(observable=obs_name, P=P, y_obs=y_obs)

        for mname in ("linear", "quadratic", "saturating"):
            fr = fit_one(P, y_obs, mname, obs_name)
            ana.fits[mname] = fr

            br = bootstrap_fit(
                P, y_obs, mname, obs_name,
                n_boot=n_boot, seed=seed, P_fine=P_fine,
            )
            ana.boots[mname] = br

        ana.best_model = min(ana.fits, key=lambda k: ana.fits[k].rmse)
        results[obs_name] = ana

    return results


# ---------------------------------------------------------------------------
# Constant-gap baseline (kept from original scaffold)
# ---------------------------------------------------------------------------

def constant_gap(
    Nx: int,
    Ny: int,
    Delta_0: float = 0.1,
) -> NDArray:
    """Return a uniform real gap field — the simplest mean-field baseline."""
    return np.full((Nx, Ny), Delta_0, dtype=complex)
