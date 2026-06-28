# RAR-GRAVITY PT0 — physical solver

Physical, auditable **PT0** ("ponto zero") solver for the Ruffini–Argüelles–Rueda
(RAR) fermionic dark-matter model in Sgr A\*, confronted with the
**GRAVITY-2024** constraint on the extended (diffuse) mass within the S2 orbit.

The goal of PT0 is **reproduction, not discovery**: rebuild numerically the
56 keV and 300 keV RAR configurations and compare against the Crespi et al.
Table 4 reference values before making any new scientific claim. No paper is
written here; no formal exclusion is declared.

## Layout

```
rar_gravity_pt0_package/
├── data/
│   └── crespi_table4_targets.yaml     # reference targets (TEMPLATE — see notes)
├── src/rar_gravity_pt0/
│   ├── constants.py                   # SI constants + UNIT POLICY
│   ├── eos_fermion_cutoff.py          # truncated relativistic Fermi-Dirac EOS
│   ├── rar_tov_solver.py              # TOV/RAR radial structure
│   ├── shooting.py                    # boundary-value solve on central params
│   ├── mass_profile.py                # extended mass within S2
│   ├── precession.py                  # per-orbit S2 precession (GR + mass)
│   ├── orbit_s2.py                    # S2 orbit + observable bundling
│   ├── validate.py                    # Crespi comparison + GO/NO-GO logic
│   └── report.py                      # error table, GO/NO-GO log, figures
├── scripts/
│   ├── run_emulator_baseline.py       # Step 1: sanity run
│   └── run_pt0_validate.py            # Step 5: full validation --case all
├── tests/                             # 22 unit + integration tests
└── output/                           # logs/ figures/ reports/ (generated)
```

## Quick start

```bash
pip install -r requirements.txt

# Step 1 — baseline sanity (EOS + one TOV profile + observables)
python scripts/run_emulator_baseline.py

# tests
pytest -q

# Step 5 — full validation, both cases, writes output/
python scripts/run_pt0_validate.py --case all
```

Outputs:
- `output/reports/pt0_error_table.md` — model-vs-target error table
- `output/logs/pt0_go_no_go.txt` — the GO/NO-GO decision log
- `output/figures/*.png` — density/mass profiles and metric potentials

## Physics summary

* **EOS** — truncated ("fermionic King") Fermi–Dirac distribution with an
  escape-energy cutoff; relativistic dispersion `ε=√(p²c²+m²c⁴)−mc²`; returns
  energy density, mass density, pressure and number density in SI.
* **Structure** — TOV equations with the RAR Tolman–Klein equilibrium
  conditions `β(r)=β₀e^{−ν/2}`, `θ(r)`, `W(r)`; integrated outward to the
  truncation surface.
* **Shooting** — varies the central degeneracy `θ₀` to hit a physical target
  (e.g. ~4.075×10⁶ M☉ core), a genuine BVP solve — **not** interpolation
  between 56 and 300 keV.
* **Observables** — extended mass `M(<r_apo)−M_core` for S2, and per-orbit
  precession split into Schwarzschild (prograde) and extended-mass
  (retrograde) parts.

## Units

Strict SI internally; every public quantity carries an explicit unit suffix
(`_m`, `_kg`, `_pa`, `_kg_m3`, `_j_m3`, `_per_m3`). See `constants.py`.

## Status

See **`DELIVERY_NOTES.md`** for what is implemented, what is calibrated,
the honest current results, and the **action required** (real Crespi Table 4
numbers) before the GO/NO-GO is scientifically binding.
