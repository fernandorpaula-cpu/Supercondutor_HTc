"""
rar_gravity_pt0 — physical, auditable PT0 solver for the RAR fermionic
dark-matter model confronted with GRAVITY-2024 extended-mass constraints
on Sgr A*.

Pipeline:
    eos_fermion_cutoff  -> truncated relativistic Fermi-Dirac EOS
    rar_tov_solver      -> TOV/RAR radial structure
    shooting            -> boundary-value solve on central parameters
    mass_profile        -> extended mass within S2
    precession          -> per-orbit S2 precession
    orbit_s2            -> S2 orbit bundling of observables
    validate / report   -> Crespi comparison, GO/NO-GO, figures
"""
from .constants import fermion_mass_kg  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "constants",
    "eos_fermion_cutoff",
    "rar_tov_solver",
    "shooting",
    "mass_profile",
    "precession",
    "orbit_s2",
    "validate",
    "report",
]
