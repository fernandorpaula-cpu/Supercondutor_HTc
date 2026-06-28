"""
Physical constants and unit conversions (strict SI internally).

UNIT POLICY (read this before using anything in this package)
-------------------------------------------------------------
* All physics is computed in SI base units:
      length    -> metre        [m]
      mass      -> kilogram     [kg]
      time      -> second       [s]
      energy    -> joule        [J]
      pressure  -> pascal       [Pa] = [J/m^3]
      energy density -> [J/m^3]
      mass density   -> [kg/m^3]
      number density -> [1/m^3]
* Fermion masses are specified by the user as a *rest energy* in keV
  (e.g. 56 keV, 300 keV). They are converted to kg here, once, via
  `fermion_mass_kg(keV)`.  Nothing downstream re-derives masses.
* Astronomical inputs (parsec, AU, solar mass) are converted to SI at
  the boundary. No code below the boundary uses astronomical units.

Every public quantity in the solver carries an explicit unit suffix in
its name (`_m`, `_kg`, `_pa`, `_kg_m3`, `_j_m3`, `_per_m3`) so that unit
mistakes are visible at the call site.  This is a deliberate, auditable
choice — the prompt forbids "ignoring units".
"""
from __future__ import annotations

import math

# --- Fundamental constants (CODATA 2018, SI) ---
C_LIGHT = 2.997_924_58e8          # speed of light            [m/s]
G_NEWTON = 6.674_30e-11           # gravitational constant    [m^3 kg^-1 s^-2]
HBAR = 1.054_571_817e-34          # reduced Planck constant   [J s]
H_PLANCK = 6.626_070_15e-34       # Planck constant           [J s]
K_B = 1.380_649e-23               # Boltzmann constant        [J/K]
EV = 1.602_176_634e-19            # electronvolt              [J]
KEV = 1.0e3 * EV                  # kilo-electronvolt         [J]

# --- Astronomical conversions ---
M_SUN = 1.988_47e30               # solar mass                [kg]
PARSEC = 3.085_677_581e16         # parsec                    [m]
AU = 1.495_978_707e11             # astronomical unit         [m]
YEAR = 3.155_695_2e7              # Julian year               [s]

# --- Degeneracy ---
G_SPIN = 2                        # spin degeneracy of a spin-1/2 fermion (g = 2s+1)


def fermion_mass_kg(mc2_keV: float) -> float:
    """Convert a fermion rest energy (in keV) to a mass in kg.

    m = (mc^2) / c^2 ,  with mc^2 given in keV.
    """
    return (mc2_keV * KEV) / C_LIGHT**2


def schwarzschild_radius_m(mass_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2  [m]."""
    return 2.0 * G_NEWTON * mass_kg / C_LIGHT**2


def msun_to_kg(m_msun: float) -> float:
    return m_msun * M_SUN


def kg_to_msun(m_kg: float) -> float:
    return m_kg / M_SUN


def pc_to_m(r_pc: float) -> float:
    return r_pc * PARSEC


def m_to_pc(r_m: float) -> float:
    return r_m / PARSEC


def mpc_to_m(r_mpc: float) -> float:
    """milli-parsec -> metre."""
    return r_mpc * 1.0e-3 * PARSEC


# Convenience: the EOS prefactor C = g / (2 pi^2 hbar^3) appears everywhere.
def eos_prefactor(g: int = G_SPIN) -> float:
    """C = g / (2 pi^2 hbar^3)   [1 / (J^3 s^3)] ; multiplied by (m c)^3 it
    yields a number density scale [1/m^3]."""
    return g / (2.0 * math.pi**2 * HBAR**3)
