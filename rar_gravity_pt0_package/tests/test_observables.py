"""Observable checks: extended mass, precession, S2 bundling, validation."""
import numpy as np
import pytest

from rar_gravity_pt0.constants import fermion_mass_kg, AU
from rar_gravity_pt0.rar_tov_solver import CentralParams, solve_profile
from rar_gravity_pt0.mass_profile import extended_mass_within
from rar_gravity_pt0.precession import (schwarzschild_precession_rad,
                                        precession_per_orbit)
from rar_gravity_pt0.orbit_s2 import S2Orbit, s2_observables
from rar_gravity_pt0.validate import classify, _rel_error

M56 = fermion_mass_kg(56.0)


@pytest.fixture(scope="module")
def profile():
    return solve_profile(CentralParams(22.0, 5e-5, 60.0, M56))


def test_extended_mass_nonnegative_and_le_enclosed(profile):
    orbit = S2Orbit.default()
    em = extended_mass_within(profile, orbit.r_apo_m)
    assert em.extended_mass_kg >= 0
    assert em.extended_mass_kg <= em.enclosed_mass_kg
    assert em.core_mass_kg <= em.enclosed_mass_kg


def test_schwarzschild_precession_known_formula():
    # 6 pi G M /(c^2 a (1-e^2)); check positivity & scaling with M
    a = 1020 * AU
    p1 = schwarzschild_precession_rad(4e6 * 1.98847e30, a, 0.88)
    p2 = schwarzschild_precession_rad(8e6 * 1.98847e30, a, 0.88)
    assert p1 > 0
    assert p2 == pytest.approx(2 * p1, rel=1e-12)


def test_mass_precession_is_retrograde(profile):
    orbit = S2Orbit.default()
    m_core = float(profile.enclosed_mass_kg(1e-4 * 3.0857e16))
    prec = precession_per_orbit(profile, orbit.a_m, orbit.e, m_core)
    # extended distributed mass -> retrograde (<=0) contribution
    assert prec.delta_phi_mass_rad <= 0.0
    # GR part is prograde
    assert prec.delta_phi_gr_rad > 0.0


def test_s2_observables_bundle(profile):
    obs = s2_observables(profile)
    assert np.isfinite(obs.extended_mass.extended_mass_msun)
    assert np.isfinite(obs.precession.delta_phi_total_arcmin)


def test_classify_bands():
    assert classify(0.005) == "EXCELLENT"
    assert classify(0.02) == "GO"
    assert classify(0.05) == "BORDERLINE"
    assert classify(0.2) == "NO-GO"
    assert classify(None) == "NO-TARGET"


def test_rel_error_formula():
    assert _rel_error(110.0, 100.0) == pytest.approx(0.10)
    assert _rel_error(100.0, None) is None
    assert _rel_error(100.0, 0.0) is None
