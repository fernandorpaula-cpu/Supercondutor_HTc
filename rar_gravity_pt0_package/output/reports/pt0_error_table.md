# PT0 error table vs Crespi targets

## Case `case_56keV`  (mc^2 = 56 keV)

- converged: **True**
- central params: theta0=21.5203, beta0=5.000e-05, W0=60.000

| observable | unit | model | target | rel.err | verdict |
|---|---|---|---|---|---|
| core_mass_msun | M_sun | 4.0750e+06 |    ---    |    ---    | NO-TARGET |
| core_radius_pc | pc | 2.4637e-04 |    ---    |    ---    | NO-TARGET |
| total_mass_msun | M_sun | 5.4434e+13 |    ---    |    ---    | NO-TARGET |
| extended_mass_within_s2_msun | M_sun | 3.9071e+06 |    ---    |    ---    | NO-TARGET |
| s2_precession_arcmin_per_orbit | arcmin/orbit | -54.4776 |    ---    |    ---    | NO-TARGET |

Notes:
  - theta0 not given -> shooting on central degeneracy
  - shoot core_mass_kg: theta0=21.5203, achieved=8.1030e+36 kg, rel_err=3.432e-10, converged=True
  - no numeric targets present -> cannot decide GO/NO-GO

**Case verdict: NO-TARGET**

## Case `case_300keV`  (mc^2 = 300 keV)

- converged: **False**
- central params: theta0=40.0000, beta0=8.000e-03, W0=60.000

| observable | unit | model | target | rel.err | verdict |
|---|---|---|---|---|---|
| core_mass_msun | M_sun | 2.9548e+06 |    ---    |    ---    | NO-TARGET |
| core_radius_pc | pc | 1.2035e-06 |    ---    |    ---    | NO-TARGET |
| total_mass_msun | M_sun | 7.0295e+06 |    ---    |    ---    | NO-TARGET |
| extended_mass_within_s2_msun | M_sun | 4.0747e+06 |    ---    |    ---    | NO-TARGET |
| s2_precession_arcmin_per_orbit | arcmin/orbit | -50.4015 |    ---    |    ---    | NO-TARGET |

Notes:
  - theta0 not given -> shooting on central degeneracy
  - shoot core_mass_kg: theta0=40.0000, achieved=5.8755e+36 kg, rel_err=2.749e-01, converged=False
  - no numeric targets present -> cannot decide GO/NO-GO

**Case verdict: NO-TARGET**

## OVERALL DECISION: NO-TARGET
