"""
src/two_scale.py — Two-scale model: local pairing vs global phase coherence.

Physical model
--------------
The superconducting transition in a cuprate under pressure is governed by two
distinct scales:

    Tc_onset(P)  — temperature at which local Cooper pairs first form.
                   Tracks the pairing amplitude Δ_d(P); onset of diamagnetic
                   fluctuations and beginning of resistance drop.

    Tc_zero(P)   — temperature at which long-range phase coherence is
                   established and resistance reaches zero.  Requires both
                   local pairing AND global phase locking.

    Wtr(P) = Tc_onset(P) − Tc_zero(P)  — transition width.  Measures the
                   strength of phase fluctuations / decoherence that separates
                   pair formation from global coherence.

The two-scale factorisation:
    Tc_zero_model(P) = C_coh(P) × Tc_onset_model(P)

where C_coh(P) ∈ (0, 1) is the global coherence factor.  For C_coh → 1 the
two scales merge (BCS limit); for C_coh → 0 coherence is fully suppressed
(incoherent Cooper pairs, no zero-resistance).

Calibration — Hg1212
---------------------
The model is calibrated against APPROXIMATE literature data for
HgBa₂CaCu₂O₆₊δ (Hg1212), optimally doped.

!! IMPORTANT — DATA STATUS !!
The numerical values in HG1212_DATA are APPROXIMATE ORDER-OF-MAGNITUDE
estimates derived from the qualitative shape described in published literature
(Gao et al. PRB 50 4260, 1994; Monteverde et al. PRB 69 214502, 2004).
They are NOT an accurate digitisation of any specific figure.
Replace HG1212_DATA with actual digitised measurements before drawing
quantitative conclusions.

Hg1223 data is PLACEHOLDER ONLY — see HG1223_DATA below.

Interpretation constraints (mandatory — do not change without justification)
---------------------------------------------------------------------------
1. Tc_onset tracks LOCAL pairing (the BdG Δ_d proxy).
2. Tc_zero requires GLOBAL coherence in addition to pairing.
3. Wtr measures broadening / coherence loss; it is NOT a sample-quality
   artefact alone — it reflects genuine phase-fluctuation physics.
4. This model does NOT predict absolute Tc from first principles.
   It is a diagnostic decomposition, not a theory of superconductivity.
5. C_coh(P) calibrated here (monotonically decreasing for Hg1212) is
   DISTINCT from the dome-shaped C_coh in src/channels.py.  The channels.py
   C_coh is a general phenomenological placeholder; the value calibrated here
   reflects the measured Tc_zero / Tc_onset ratio for this material.
"""

from __future__ import annotations

INTERPRETATION_BLOCK: str = (
    "Two-scale model: Tc_onset = C_coh(P) x Tc_MF(BdG). "
    "Tc_MF != Tc_onset; Delta_d = local pairing proxy (NOT gap CT). "
    "C_coh reflects measured Tc_zero/Tc_onset ratio [APPROXIMATE data]."
)

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

# ---------------------------------------------------------------------------
# Experimental reference data — Hg1212
# ---------------------------------------------------------------------------

# !! PLACEHOLDER DATA — see module docstring !!
# Source: [LIT] approximate shape from Gao et al. (1994) and Monteverde et al.
# (2004) for optimally doped Hg1212. Wtr values [ASSUMED] for polycrystalline
# sample; replace with actual digitised measurements.
HG1212_DATA: dict = {
    "label":       "Hg1212 (HgBa₂CaCu₂O₆₊δ), optimally doped",
    "status":      "APPROXIMATE — replace with actual digitised data",
    "source":      "[LIT] Gao et al. PRB 50 4260 (1994); Monteverde et al. PRB 69 214502 (2004)",
    "P_GPa":       np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30], dtype=float),
    "Tc_onset_K":  np.array([126, 131, 137, 142, 147, 151, 154, 152, 149, 145, 140], dtype=float),
    "Tc_zero_K":   np.array([122, 127, 132, 136, 140, 143, 144, 141, 136, 131, 126], dtype=float),
    # Wtr derived: Wtr = Tc_onset - Tc_zero
    # [4, 4, 5, 6, 7, 8, 10, 11, 13, 14, 14]  K
}
# Sanity check on embedded data
assert np.all(HG1212_DATA["Tc_onset_K"] > HG1212_DATA["Tc_zero_K"]), \
    "Tc_onset must exceed Tc_zero everywhere"

# !! PLACEHOLDER — Hg1223 data not yet digitised !!
HG1223_DATA: dict = {
    "label":   "Hg1223 (HgBa₂Ca₂Cu₃O₈₊δ), optimally doped",
    "status":  "PLACEHOLDER — no data embedded; do not use for quantitative analysis",
    "source":  "[PLACEHOLDER] see Chu et al. (1993) and related references",
    "P_GPa":   None,
    "Tc_onset_K": None,
    "Tc_zero_K":  None,
}


# ---------------------------------------------------------------------------
# Derived experimental quantities
# ---------------------------------------------------------------------------

def wtr_data(data: dict = HG1212_DATA) -> NDArray:
    """Transition width Wtr(P) = Tc_onset(P) − Tc_zero(P) [K]."""
    return data["Tc_onset_K"] - data["Tc_zero_K"]


def kappa_data(data: dict = HG1212_DATA) -> NDArray:
    """Empirical coherence factor κ(P) = Tc_zero(P) / Tc_onset(P)."""
    return data["Tc_zero_K"] / data["Tc_onset_K"]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def Tc_onset_model(P_GPa: float | NDArray, coeffs: NDArray) -> float | NDArray:
    """
    Phenomenological Tc_onset_model(P) from polynomial fit.

    Tc_onset_model(P) = sum_n coeffs[n] * P^(deg-n)   (numpy.polyval convention)

    Args:
        P_GPa:  pressure [GPa].
        coeffs: polynomial coefficients from numpy.polyfit or calibrate_model().

    Interpretation: Tc_onset tracks LOCAL pairing amplitude Δ_d(P).
    It is NOT Tc_MF from BdG — BdG Tc_MF misses charge transfer and retardation
    under pressure, causing a spurious drop.  This fit is purely phenomenological.
    """
    return np.polyval(coeffs, np.asarray(P_GPa, dtype=float))


def C_coh_model(P_GPa: float | NDArray, coeffs: NDArray) -> float | NDArray:
    """
    Global coherence factor C_coh(P) from polynomial fit to κ_data.

    C_coh_model(P) = polyval(coeffs, P),  clipped to [0, 1].

    Interpretation:
      - C_coh < 1 implies Tc_zero < Tc_onset (pair decoherence).
      - C_coh decreases with P for Hg1212 (Wtr widens under pressure).
      - DISTINCT from the dome-shaped C_coh in src/channels.py.
        The channels.py C_coh is a general placeholder; this value is
        calibrated to κ_exp = Tc_zero_exp / Tc_onset_exp.
    """
    val = np.polyval(coeffs, np.asarray(P_GPa, dtype=float))
    return np.clip(val, 0.0, 1.0)


def Tc_zero_model_fn(
    P_GPa: float | NDArray,
    coeffs_onset: NDArray,
    coeffs_coh: NDArray,
) -> float | NDArray:
    """
    Tc_zero_model(P) = C_coh(P) × Tc_onset_model(P).

    Interpretation: Tc_zero requires both local pairing (Tc_onset) AND
    global phase coherence (C_coh).  Tc_zero is NOT equal to Δ_d(P)/kB;
    it is the temperature of ZERO RESISTANCE, which is lower.
    """
    return C_coh_model(P_GPa, coeffs_coh) * Tc_onset_model(P_GPa, coeffs_onset)


def Wtr_model_fn(
    P_GPa: float | NDArray,
    coeffs_onset: NDArray,
    coeffs_coh: NDArray,
) -> float | NDArray:
    """
    Wtr_model(P) = Tc_onset_model(P) − Tc_zero_model(P)
                 = (1 − C_coh(P)) × Tc_onset_model(P).

    Wtr measures decoherence: Wtr → 0 in BCS limit (C_coh → 1),
    Wtr → Tc_onset in fully incoherent limit.
    """
    return (1.0 - C_coh_model(P_GPa, coeffs_coh)) * Tc_onset_model(P_GPa, coeffs_onset)


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def rmse(predicted: NDArray, observed: NDArray) -> float:
    """Root-mean-square error [same units as inputs]."""
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def rmse_Tc_zero(
    coeffs_onset: NDArray,
    coeffs_coh: NDArray,
    data: dict,
    P_min: float = 0.0,
) -> dict[str, float]:
    """
    Compute RMSE for Tc_zero prediction vs data, split by pressure range.

    Returns dict with keys 'all', 'high_P' (P >= P_min).
    """
    P   = data["Tc_onset_K"] * 0 + data["P_GPa"]   # ensure array
    P   = data["P_GPa"]
    Tz_pred = Tc_zero_model_fn(P, coeffs_onset, coeffs_coh)
    Tz_obs  = data["Tc_zero_K"]

    mask_high = P >= P_min
    return {
        "all":    rmse(Tz_pred, Tz_obs),
        "high_P": rmse(Tz_pred[mask_high], Tz_obs[mask_high]),
        "onset":  rmse(Tc_onset_model(P, coeffs_onset), data["Tc_onset_K"]),
        "P_min":  P_min,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Container for calibration output."""
    coeffs_onset:  NDArray                    # polynomial coefficients for Tc_onset
    coeffs_coh:    NDArray                    # polynomial coefficients for C_coh
    deg_onset:     int = 2
    deg_coh:       int = 2
    rmse_all:      float = 0.0               # RMSE Tc_zero, all P
    rmse_high_P:   float = 0.0              # RMSE Tc_zero, P >= P_threshold
    rmse_onset:    float = 0.0              # RMSE Tc_onset fit
    P_threshold:   float = 12.0             # GPa
    data_label:    str = ""
    data_status:   str = ""


def calibrate_model(
    data: dict = HG1212_DATA,
    deg_onset: int = 2,
    deg_coh: int = 2,
    P_threshold: float = 12.0,
) -> CalibrationResult:
    """
    Calibrate the two-scale model against experimental data.

    Step 1: Fit Tc_onset_model(P) = polynomial(deg_onset) to Tc_onset_data.
    Step 2: Compute κ_data = Tc_zero_data / Tc_onset_data.
    Step 3: Fit C_coh_model(P) = polynomial(deg_coh) to κ_data.
    Step 4: Tc_zero_model = C_coh × Tc_onset_model.
    Step 5: Compute RMSE for Tc_zero prediction (all P and P >= P_threshold).

    Returns CalibrationResult with all fit parameters and RMSE values.
    """
    if data.get("status", "").startswith("PLACEHOLDER"):
        raise ValueError(
            f"Data for '{data.get('label', '?')}' is a placeholder. "
            "Embed actual measurements before calibrating."
        )

    P         = data["P_GPa"]
    Tc_on     = data["Tc_onset_K"]
    Tc_z      = data["Tc_zero_K"]
    kappa     = Tc_z / Tc_on

    # Step 1 — Tc_onset polynomial fit
    coeffs_on  = np.polyfit(P, Tc_on, deg_onset)

    # Step 2/3 — C_coh polynomial fit to κ
    coeffs_coh = np.polyfit(P, kappa, deg_coh)

    # Step 4/5 — RMSE
    errs = rmse_Tc_zero(coeffs_on, coeffs_coh, data, P_min=P_threshold)

    return CalibrationResult(
        coeffs_onset  = coeffs_on,
        coeffs_coh    = coeffs_coh,
        deg_onset     = deg_onset,
        deg_coh       = deg_coh,
        rmse_all      = errs["all"],
        rmse_high_P   = errs["high_P"],
        rmse_onset    = errs["onset"],
        P_threshold   = P_threshold,
        data_label    = data.get("label", ""),
        data_status   = data.get("status", ""),
    )


# ---------------------------------------------------------------------------
# Full diagnostic table over continuous pressure grid
# ---------------------------------------------------------------------------

def two_scale_table(
    P_grid: NDArray,
    cal: CalibrationResult,
    Delta_d_meV: NDArray | None = None,
    Tc_MF_K:     NDArray | None = None,
) -> dict[str, NDArray]:
    """
    Evaluate all two-scale diagnostics on a fine pressure grid.

    Args:
        P_grid:       pressure array [GPa].
        cal:          CalibrationResult from calibrate_model().
        Delta_d_meV:  optional BdG pairing amplitude at each P [meV]
                      (local pairing proxy, NOT Tc_zero).
        Tc_MF_K:      optional BdG mean-field Tc at each P [K]
                      (pair-formation scale, NOT Tc_zero).

    Returns dict with:
        P, Tc_onset_model, C_coh, Tc_zero_model, Wtr_model,
        Delta_d_meV (if provided), Tc_MF_K (if provided).
    """
    P  = np.asarray(P_grid, dtype=float)
    Ton = Tc_onset_model(P, cal.coeffs_onset)
    Cc  = C_coh_model(P, cal.coeffs_coh)
    Tz  = Cc * Ton
    Wtr = Ton - Tz

    out: dict[str, NDArray] = {
        "P":               P,
        "Tc_onset_model":  Ton,
        "C_coh":           Cc,
        "Tc_zero_model":   Tz,
        "Wtr_model":       Wtr,
    }
    if Delta_d_meV is not None:
        out["Delta_d_meV"] = np.asarray(Delta_d_meV, dtype=float)
    if Tc_MF_K is not None:
        out["Tc_MF_K"] = np.asarray(Tc_MF_K, dtype=float)
    return out
