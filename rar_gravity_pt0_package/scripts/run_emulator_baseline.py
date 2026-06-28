#!/usr/bin/env python3
"""
PT0 baseline sanity run (Step 1 of the mandated order).

Exercises the whole stack on a small, fast configuration so that a fresh
checkout can confirm the package is wired correctly BEFORE the heavy
validation run:

  * evaluate the truncated Fermi-Dirac EOS at the centre,
  * integrate one RAR/TOV profile,
  * compute the S2 observables,
  * print a compact summary.

It does NOT compare against Crespi targets (that is run_pt0_validate.py).
Exit code is 0 on success.
"""
import _bootstrap  # noqa: F401  (path setup)

import numpy as np

from rar_gravity_pt0.constants import (fermion_mass_kg, kg_to_msun, m_to_pc)
from rar_gravity_pt0.eos_fermion_cutoff import eos_state
from rar_gravity_pt0.rar_tov_solver import (CentralParams, core_radius_m,
                                            solve_profile)
from rar_gravity_pt0.orbit_s2 import s2_observables


def main() -> int:
    print("=" * 64)
    print("RAR-GRAVITY PT0 — emulator baseline")
    print("=" * 64)

    mc2_keV = 56.0
    m_kg = fermion_mass_kg(mc2_keV)
    print(f"fermion: mc^2 = {mc2_keV} keV  ->  m = {m_kg:.4e} kg")

    # 1) EOS at the centre
    theta0, beta0, W0 = 30.0, 1.0e-5, 60.0
    pt = eos_state(theta0, W0, beta0, m_kg)
    print("\n[EOS @ centre]")
    print(f"  theta0={theta0}  beta0={beta0:.1e}  W0={W0}")
    print(f"  rho   = {pt.mass_density_kg_m3:.4e} kg/m^3")
    print(f"  P     = {pt.pressure_pa:.4e} Pa")
    print(f"  n     = {pt.number_density_per_m3:.4e} 1/m^3")
    print(f"  rho_E = {pt.energy_density_j_m3:.4e} J/m^3")
    assert pt.mass_density_kg_m3 > 0, "EOS produced non-positive density"

    # 2) one TOV/RAR profile
    cp = CentralParams(theta0=theta0, beta0=beta0, W0=W0, m_kg=m_kg)
    prof = solve_profile(cp)
    r_core = core_radius_m(prof)
    print("\n[RAR/TOV profile]")
    print(f"  surface R   = {m_to_pc(prof.surface_radius_m):.4e} pc")
    print(f"  total mass  = {kg_to_msun(prof.total_mass_kg):.4e} M_sun")
    print(f"  core radius = {m_to_pc(r_core):.4e} pc")
    print(f"  core mass   = {kg_to_msun(float(prof.enclosed_mass_kg(r_core))):.4e} M_sun")
    print(f"  grid points = {prof.r_m.size}")
    for fld in ("r_m", "mass_kg", "rho_kg_m3", "pressure_pa",
                "nu_metric", "lambda_metric"):
        arr = getattr(prof, fld)
        assert np.all(np.isfinite(arr)), f"non-finite values in {fld}"
    print("  required fields present & finite: "
          "r_m, mass_kg, rho_kg_m3, pressure_pa, nu_metric, lambda_metric  [OK]")

    # 3) observables
    obs = s2_observables(prof)
    print("\n[S2 observables]")
    print(f"  extended mass within S2 = "
          f"{obs.extended_mass.extended_mass_msun:.4e} M_sun")
    print(f"  S2 precession           = "
          f"{obs.precession.delta_phi_total_arcmin:.4e} arcmin/orbit")
    print(f"    (GR part   = {obs.precession.delta_phi_gr_rad:.4e} rad)")
    print(f"    (mass part = {obs.precession.delta_phi_mass_rad:.4e} rad)")

    print("\nBASELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
