"""EOS physics & unit checks for the truncated Fermi-Dirac equation of state."""
import numpy as np
import pytest

from rar_gravity_pt0.constants import (fermion_mass_kg, C_LIGHT, HBAR,
                                       G_SPIN, eos_prefactor)
from rar_gravity_pt0.eos_fermion_cutoff import (eos_state, occupation,
                                                _x_cutoff, _integral)


M56 = fermion_mass_kg(56.0)


def test_occupation_bounds_and_cutoff():
    x = np.linspace(0, 50, 500)
    f = occupation(x, theta=30.0, W=60.0, beta=1e-5)
    assert np.all(f >= 0.0) and np.all(f <= 1.0)
    # beyond the cutoff momentum the occupation must vanish
    x_c = _x_cutoff(1e-5, 60.0)
    assert np.all(occupation(x_c * 1.5, 30.0, 60.0, 1e-5) == 0.0)


def test_positive_state_and_units():
    pt = eos_state(theta=30.0, W=60.0, beta=1e-5, m_kg=M56)
    assert pt.energy_density_j_m3 > 0
    assert pt.mass_density_kg_m3 > 0
    assert pt.pressure_pa > 0
    assert pt.number_density_per_m3 > 0
    # rho = rho_E / c^2 must hold exactly (unit consistency)
    assert pt.mass_density_kg_m3 == pytest.approx(
        pt.energy_density_j_m3 / C_LIGHT**2, rel=1e-12)


def test_pressure_positive_and_subrelativistic_ordering():
    # For a cold, non-relativistic gas P << rho_E (P/rho_E ~ <v^2>/3 c^2 small)
    pt = eos_state(theta=30.0, W=40.0, beta=1e-5, m_kg=M56)
    assert pt.pressure_pa < pt.energy_density_j_m3


def test_number_density_matches_prefactor_scaling():
    # n should scale as (m c)^3 ; doubling m at fixed dimensionless params
    # scales I_* identically, so n ~ m^3.
    pt1 = eos_state(20.0, 30.0, 1e-4, M56)
    pt2 = eos_state(20.0, 30.0, 1e-4, 2 * M56)
    assert pt2.number_density_per_m3 / pt1.number_density_per_m3 == pytest.approx(8.0, rel=1e-6)


def test_integrals_zero_when_no_cutoff():
    assert _integral("n", 30.0, 0.0, 1e-5) == 0.0
    assert _x_cutoff(1e-5, 0.0) == 0.0


def test_monotonic_density_in_theta():
    # more degenerate -> denser (at fixed beta, W)
    lo = eos_state(10.0, 60.0, 1e-5, M56).mass_density_kg_m3
    hi = eos_state(40.0, 60.0, 1e-5, M56).mass_density_kg_m3
    assert hi > lo
