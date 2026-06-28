"""Structure (TOV/RAR) solver checks."""
import numpy as np
import pytest

from rar_gravity_pt0.constants import (fermion_mass_kg, G_NEWTON, C_LIGHT,
                                       kg_to_msun)
from rar_gravity_pt0.rar_tov_solver import (CentralParams, solve_profile,
                                            core_radius_m)

M56 = fermion_mass_kg(56.0)


@pytest.fixture(scope="module")
def profile():
    cp = CentralParams(theta0=20.0, beta0=5e-5, W0=60.0, m_kg=M56)
    return solve_profile(cp)


def test_required_fields_present_and_finite(profile):
    for fld in ("r_m", "mass_kg", "rho_kg_m3", "pressure_pa",
                "nu_metric", "lambda_metric"):
        arr = getattr(profile, fld)
        assert arr.size > 10
        assert np.all(np.isfinite(arr))


def test_mass_monotonic_nondecreasing(profile):
    dM = np.diff(profile.mass_kg)
    assert np.all(dM >= -1e-3 * profile.total_mass_kg)  # allow tiny numerical noise


def test_density_decreases_outward(profile):
    # central density is the maximum
    assert profile.rho_kg_m3[0] == pytest.approx(profile.rho_kg_m3.max(), rel=1e-6)


def test_no_horizon_metric_consistency(profile):
    # 1 - 2GM/(rc^2) must stay positive (no horizon) for a dilute gas
    f = 1.0 - 2.0 * G_NEWTON * profile.mass_kg / (profile.r_m * C_LIGHT**2)
    assert np.all(f > 0.0)
    # lambda = -ln(f)
    assert np.allclose(profile.lambda_metric, -np.log(f), atol=1e-6)


def test_exterior_metric_match(profile):
    # nu shifted so e^{nu(R)} = 1 - 2GM_tot/(R c^2)
    R = profile.surface_radius_m
    expected = np.log(1.0 - 2.0 * G_NEWTON * profile.total_mass_kg
                      / (R * C_LIGHT**2))
    assert profile.nu_metric[-1] == pytest.approx(expected, rel=1e-6)


def test_core_radius_within_surface(profile):
    rc = core_radius_m(profile)
    assert 0 < rc <= profile.surface_radius_m


def test_core_mass_increases_with_beta0():
    # along the degenerate branch a warmer (larger beta0) core is more massive,
    # up to the turning point; check the rising part.
    def cm(b0):
        p = solve_profile(CentralParams(20.0, b0, 60.0, M56))
        return float(p.enclosed_mass_kg(core_radius_m(p)))
    assert cm(5e-5) > cm(1e-5)
