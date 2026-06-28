r"""
Extended-mass observable relevant to the S2 orbit.

GRAVITY-2024 constrains the *extended* (non point-like, diffuse) mass
enclosed within the S2 orbit.  In the RAR picture the central object is a
dense fermion core that mimics the ~4e6 M_sun compact source; on top of it
there is a diffuse fermion distribution whose enclosed mass within S2 is
the quantity confronted with the GRAVITY limit (~1200 M_sun at 1 sigma,
compatible with zero).

We define the extended mass within a radius r as the gravitating mass
enclosed at r MINUS the dense-core mass:

    M_ext(<r) = M(<r) - M_core ,

with M_core = M(< r_core) and r_core the first density e-folding (see
`rar_tov_solver.core_radius_m`).  The relevant radius for S2 is its
apoapsis r_apo = a (1 + e).

UNITS: SI.  Masses [kg], radii [m]; helper returns also in M_sun for
reporting.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import kg_to_msun
from .rar_tov_solver import Profile, core_radius_m


@dataclass(frozen=True)
class ExtendedMass:
    r_apo_m: float
    core_radius_m: float
    core_mass_kg: float
    enclosed_mass_kg: float        # total M(< r_apo)
    extended_mass_kg: float        # M(< r_apo) - M_core
    core_mass_msun: float
    extended_mass_msun: float


def extended_mass_within(profile: Profile, r_apo_m: float) -> ExtendedMass:
    """Compute the extended (diffuse) mass enclosed within radius r_apo."""
    r_core = core_radius_m(profile)
    m_core = float(profile.enclosed_mass_kg(r_core))
    m_enc = float(profile.enclosed_mass_kg(r_apo_m))
    m_ext = max(m_enc - m_core, 0.0)
    return ExtendedMass(
        r_apo_m=r_apo_m,
        core_radius_m=r_core,
        core_mass_kg=m_core,
        enclosed_mass_kg=m_enc,
        extended_mass_kg=m_ext,
        core_mass_msun=kg_to_msun(m_core),
        extended_mass_msun=kg_to_msun(m_ext),
    )
