# PT0 error table vs Crespi targets

## Case `case_56keV`  (mc^2 = 56 keV)

- converged: **True**
- central params: theta0=21.5199, beta0=5.000e-05, W0=60.000

| observable | unit | model | target | rel.err | verdict |
|---|---|---|---|---|---|
| core_mass_msun | M_sun | 4.0749e+06 |    ---    |    ---    | NO-TARGET |
| core_radius_pc | pc | 2.4637e-04 |    ---    |    ---    | NO-TARGET |
| total_mass_msun | M_sun | 5.4437e+13 |    ---    |    ---    | NO-TARGET |
| extended_mass_within_s2_msun | M_sun | 3.9071e+06 |    ---    |    ---    | NO-TARGET |
| s2_precession_arcmin_per_orbit | arcmin/orbit | -54.4790 |    ---    |    ---    | NO-TARGET |

Notes:
  - theta0 not given -> shooting on central degeneracy
  - shoot theta0 -> core_mass_kg: theta0=2.1520e+01, achieved=8.1028e+36 kg, rel_err=2.208e-05, converged=True
  - no numeric targets present -> cannot decide GO/NO-GO

**Case verdict: NO-TARGET**

## Case `case_300keV`  (mc^2 = 300 keV)

- converged: **False**
- central params: theta0=40.0000, beta0=4.979e-03, W0=60.000

| observable | unit | model | target | rel.err | verdict |
|---|---|---|---|---|---|
| core_mass_msun | M_sun | 3.1649e+06 |    ---    |    ---    | NO-TARGET |
| core_radius_pc | pc | 1.6425e-06 |    ---    |    ---    | NO-TARGET |
| total_mass_msun | M_sun | 6.7847e+06 |    ---    |    ---    | NO-TARGET |
| extended_mass_within_s2_msun | M_sun | 3.6198e+06 |    ---    |    ---    | NO-TARGET |
| s2_precession_arcmin_per_orbit | arcmin/orbit | -49.7972 |    ---    |    ---    | NO-TARGET |

Notes:
  - shoot beta0 -> core_mass_kg: beta0=4.9793e-03, achieved=6.2933e+36 kg, rel_err=2.233e-01, converged=False; target ABOVE branch maximum 6.2933e+36 kg (3.165e+06 Msun)
  - no numeric targets present -> cannot decide GO/NO-GO

**Case verdict: NO-TARGET**

## OVERALL DECISION: NO-TARGET
