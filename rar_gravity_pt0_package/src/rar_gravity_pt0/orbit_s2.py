r"""
S2 orbit description and the observables tied to it.

S2 (a.k.a. S0-2) is the benchmark star whose orbit around Sgr A* anchors
the GRAVITY measurements.  This module bundles its orbital elements and
ties together the two PT0 observables computed on a solved profile:

    * extended mass within the S2 orbit  (mass_profile)
    * precession per S2 orbit            (precession)

Default orbital elements (GRAVITY Collaboration values; flagged [LIT]):
    a  = 0.1255 arcsec  ->  ~ 1020 AU  semi-major axis
    e  = 0.8843          eccentricity
    P  = 16.05 yr        period
These are stored in SI internally.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import AU, YEAR
from .mass_profile import ExtendedMass, extended_mass_within
from .precession import Precession, precession_per_orbit
from .rar_tov_solver import Profile, core_radius_m


@dataclass(frozen=True)
class S2Orbit:
    a_m: float
    e: float
    period_s: float

    @classmethod
    def default(cls) -> "S2Orbit":
        # a ~ 1020 AU, e = 0.8843, P = 16.05 yr   [LIT: GRAVITY]
        return cls(a_m=1020.0 * AU, e=0.8843, period_s=16.05 * YEAR)

    @property
    def r_peri_m(self) -> float:
        return self.a_m * (1.0 - self.e)

    @property
    def r_apo_m(self) -> float:
        return self.a_m * (1.0 + self.e)


@dataclass(frozen=True)
class S2Observables:
    extended_mass: ExtendedMass
    precession: Precession


def s2_observables(profile: Profile, orbit: S2Orbit | None = None) -> S2Observables:
    """Compute the S2-relevant observables for a solved RAR profile."""
    orbit = orbit or S2Orbit.default()
    ext = extended_mass_within(profile, orbit.r_apo_m)
    m_core_kg = float(profile.enclosed_mass_kg(core_radius_m(profile)))
    prec = precession_per_orbit(profile, orbit.a_m, orbit.e, m_core_kg)
    return S2Observables(extended_mass=ext, precession=prec)
