#!/usr/bin/env python
"""
run_all.py — Full pipeline for the Hg-1212/Hg-1223 cuprate pressure study.

Execution order:
  1. Two-scale calibration  (src/two_scale.py)
  2. Null models            (src/null_models.py)
  3. BdG gap + DOS          (src/pairing_bdg.py)
  4. Channel vertex         (src/channels.py)
  5. RPA mediator           (src/mediator_rpa.py)
  6. Correlation proxies    (src/correlation.py)
  7. All figures            (src/figures.py)
  8. Auto-audit report      (outputs/auto_auditoria.md)

Outputs (all in outputs/):
  *.csv   — tabular data for each module
  *.png   — 300 dpi raster figures
  *.pdf   — vector figures
  auto_auditoria.md — full audit report

Exit 0 on success.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED: int = 42
rng = np.random.default_rng(SEED)

t0 = time.time()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_log_lines: list[str] = []


def section(title: str) -> None:
    sep = "─" * 72
    msg = f"\n{sep}\n  {title}\n{sep}"
    print(msg)
    _log_lines.append(msg)


def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(str(msg))


def ok(label: str) -> None:
    msg = f"  [PASS] {label}"
    print(msg)
    _log_lines.append(msg)


def fail(label: str) -> None:
    msg = f"  [FAIL] {label}"
    print(msg)
    _log_lines.append(msg)


_acceptance: list[tuple[str, bool]] = []


def check(label: str, cond: bool) -> None:
    _acceptance.append((label, cond))
    (ok if cond else fail)(label)


# ---------------------------------------------------------------------------
# Package versions
# ---------------------------------------------------------------------------

def _pkg_version(pkg: str) -> str:
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        try:
            return importlib.metadata.version(pkg)  # type: ignore[attr-defined]
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# 1. Two-scale calibration
# ---------------------------------------------------------------------------

section("1. Two-scale calibration  (src/two_scale.py)")

from src.two_scale import (
    HG1212_DATA, HG1223_DATA,
    calibrate_model, two_scale_table,
    Tc_onset_model, C_coh_model, Tc_zero_model_fn,
    INTERPRETATION_BLOCK,
)
from src.lattice_bands import P_GRID_GPA

calib = calibrate_model(HG1212_DATA)
ts_table = two_scale_table(P_GRID_GPA, calib)

log(f"  Data: {HG1212_DATA['label']}")
log(f"  Status: {HG1212_DATA['status']}")
log(f"  RMSE onset  = {calib.rmse_onset:.3f} K")
log(f"  RMSE all    = {calib.rmse_all:.3f} K")
log(f"  RMSE high-P = {calib.rmse_high_P:.3f} K")
log()
log(f"  {INTERPRETATION_BLOCK}")

check("Two-scale calibration RMSE_onset < 5 K", calib.rmse_onset < 5.0)
check("Hg1223 data remains PLACEHOLDER (ValueError guard)",
      "PLACEHOLDER" in HG1223_DATA.get("status", ""))

import csv

with open(OUTPUT_DIR / "two_scale_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    keys = list(ts_table.keys())
    w.writerow(keys)
    for row in zip(*[ts_table[k] for k in keys]):
        w.writerow([f"{v:.4f}" for v in row])
log(f"  CSV → {OUTPUT_DIR / 'two_scale_table.csv'}")

# ---------------------------------------------------------------------------
# 2. Null models
# ---------------------------------------------------------------------------

section("2. Null models  (src/null_models.py)")

from src.null_models import run_null_analysis, RULE, SIGMA_T

null_results = run_null_analysis(HG1212_DATA, n_boot=500, seed=SEED)

for obs, nr in null_results.items():
    best = nr.best_model
    fit  = nr.fits[best]
    log(f"  {obs}: best={best}  RMSE={fit.rmse:.3f} K  rho={fit.rho:.3e}"
        + ("  [DEGEN]" if fit.degenerate else ""))

log(f"\n  RULE: {RULE}")

check("Tc_zero best model = quadratic", null_results["Tc_zero"].best_model == "quadratic")
check("Wtr saturating fit degenerate (rho > 1e6)",
      bool(null_results["Wtr"].fits["saturating"].degenerate))

res_path = OUTPUT_DIR / "null_models_residuals.csv"
with open(res_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["observable", "model", "P_GPa", "residual"])
    for obs, nr in null_results.items():
        for mname, fit in nr.fits.items():
            for P_val, res in zip(nr.P, fit.residuals):
                w.writerow([obs, mname, f"{P_val:.2f}", f"{res:.4f}"])
log(f"  CSV → {res_path}")

# ---------------------------------------------------------------------------
# 3. BdG gap + DOS
# ---------------------------------------------------------------------------

section("3. BdG gap + DOS  (src/pairing_bdg.py)")

from src.pairing_bdg import (
    bdg_pressure_scan, V_D_CALIB, NK_GAP, NK_DOS,
    BDG_DISCLAIMER,
)

bdg = bdg_pressure_scan(P_GRID_GPA)

Delta0 = bdg["Delta_d_meV"][0]
Tc0    = bdg["Tc_MF_K"][0]
ratio0 = bdg["ratio_2DkT"][0]
log(f"  P=0:  Delta_d = {Delta0:.1f} meV  Tc_MF = {Tc0:.1f} K  2D/kTc = {ratio0:.2f}")
log(f"  {BDG_DISCLAIMER}")

check("Delta_d(P=0) in [20, 45] meV", 20 <= Delta0 <= 45)
check("2Delta/kBTc ratio in [3, 6]", 3.0 <= ratio0 <= 6.0)

bdg_path = OUTPUT_DIR / "bdg_table.csv"
with open(bdg_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P_GPa", "Delta_d_meV", "Tc_MF_K", "ratio_2DkT", "V_d_P_eV"])
    for i, P_val in enumerate(bdg["P"]):
        w.writerow([f"{P_val:.1f}", f"{bdg['Delta_d_meV'][i]:.4f}",
                    f"{bdg['Tc_MF_K'][i]:.4f}", f"{bdg['ratio_2DkT'][i]:.4f}",
                    f"{bdg['V_d_P'][i]:.6f}"])
log(f"  CSV → {bdg_path}")

# ---------------------------------------------------------------------------
# 4. Channel vertex
# ---------------------------------------------------------------------------

section("4. Channel vertex  (src/channels.py)")

from src.channels import channel_table, P_OPT_GPA

ct = channel_table(P_GRID_GPA)

log(f"  P_opt = {P_OPT_GPA:.1f} GPa (C_coh peak)")
log(f"  V_d_eff(P=0)  = {ct['V_d_eff'][0]:.4f} eV")
log(f"  V_d_eff(P=30) = {ct['V_d_eff'][-1]:.4f} eV")

check("V_d_eff(P) > lambda_hop(P) for all P",
      bool(np.all(np.array(ct["V_d_eff"]) > np.array(ct["lambda_hop"]))))
check("C_coh(P_opt) is maximum of C_coh",
      float(ct["C_coh"][np.argmin(np.abs(np.array(ct["P"]) - P_OPT_GPA))])
      == float(np.max(ct["C_coh"])))

ch_path = OUTPUT_DIR / "channels_table.csv"
with open(ch_path, "w", newline="") as f:
    w = csv.writer(f)
    keys = list(ct.keys())
    w.writerow(keys)
    for row in zip(*[ct[k] for k in keys]):
        w.writerow([f"{v:.6f}" for v in row])
log(f"  CSV → {ch_path}")

# ---------------------------------------------------------------------------
# 5. RPA mediator
# ---------------------------------------------------------------------------

section("5. RPA mediator  (src/mediator_rpa.py)")

from src.mediator_rpa import (
    rpa_pressure_scan, NK_CHI, T_RPA_EV, U_HUB, ETA_FS_EV,
    STONER_THRESHOLD, LABEL_RPA,
)

# Use coarser grid for speed (full P_GRID_GPA is slow at NK=32)
P_rpa = P_GRID_GPA[::4]
rpa = rpa_pressure_scan(P_rpa, NK_CHI, T_RPA_EV, U_HUB, ETA_FS_EV)

log(f"  Stoner S(P=0) = {rpa.Stoner[0]:.4f}  (must be < {STONER_THRESHOLD})")
log(f"  lambda_d(P=0) = {rpa.lambda_d[0]:.5f}  (must be > 0)")
log(f"  lambda_s(P=0) = {rpa.lambda_s[0]:.5f}  (must be < 0)")
log(f"  omega_sf(P=0) = {rpa.omega_sf[0]*1e3:.1f} meV")
log(f"  LABEL: {LABEL_RPA}")

check("Stoner S < threshold at all P", bool(np.all(rpa.Stoner < STONER_THRESHOLD)))
check("lambda_d > 0 at P=0", float(rpa.lambda_d[0]) > 0)
check("lambda_s < 0 at P=0", float(rpa.lambda_s[0]) < 0)
check("d-wave preferred (lambda_d > lambda_s) at P=0",
      float(rpa.lambda_d[0]) > float(rpa.lambda_s[0]))

rpa_path = OUTPUT_DIR / "rpa_table.csv"
with open(rpa_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P_GPa", "Stoner", "lambda_d", "lambda_s", "ratio_d_s",
                "omega_sf_eV", "lambda_med"])
    for i in range(len(rpa.P)):
        w.writerow([f"{rpa.P[i]:.1f}", f"{rpa.Stoner[i]:.5f}",
                    f"{rpa.lambda_d[i]:.6f}", f"{rpa.lambda_s[i]:.6f}",
                    f"{rpa.ratio_d_s[i]:.4f}", f"{rpa.omega_sf[i]:.6f}",
                    f"{rpa.lambda_med[i]:.6f}"])
log(f"  CSV → {rpa_path}")

# ---------------------------------------------------------------------------
# 6. Correlation proxies
# ---------------------------------------------------------------------------

section("6. Correlation proxies  (src/correlation.py)")

from src.correlation import (
    correlation_scan, overprediction_report,
    U_CORR_EV, NK_CORR,
    LABEL_A, LABEL_B, LABEL_C, LABEL_D,
)

corr = correlation_scan(P_GRID_GPA, NK_CORR, U_CORR_EV)

J_hub_arr = corr.J_Hub
J_em_arr  = corr.J_Em
ovp       = overprediction_report(corr.P, J_hub_arr, J_em_arr)

log(f"  Delta_HF(P=0) = {corr.Delta_HF[0]*1e3:.1f} meV  [LABEL A]")
log(f"  m_HF(P=0)     = {corr.m_HF[0]:.4f}       [LABEL B]")
log(f"  Z_BR(P=0)     = {corr.Z_BR[0]:.4f}       [LABEL C]")
log(f"  J_Hub(P=0)    = {J_hub_arr[0]*1e3:.1f} meV")
log(f"  J_Emery(P=0)  = {J_em_arr[0]*1e3:.1f} meV")
log(f"  Over-predict first P: {ovp['P_first_overpred']:.1f} GPa  [LABEL D]")

check("Delta_HF > 0 at P=0", float(corr.Delta_HF[0]) > 0)
check("Z_BR in [0,1] at all P",
      bool(np.all((corr.Z_BR >= 0) & (corr.Z_BR <= 1))))
check("J_Emery over-predicts J_Hub at high P", bool(ovp["overpred"].any()))

corr_path = OUTPUT_DIR / "correlation_table.csv"
with open(corr_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P_GPa", "Delta_HF_meV", "m_HF", "Z_BR", "U_c_eV",
                "J_Hub_meV", "J_Em_meV", "enh_Hub", "enh_Emery", "overpred"])
    for i in range(len(corr.P)):
        w.writerow([
            f"{corr.P[i]:.1f}", f"{corr.Delta_HF[i]*1e3:.3f}",
            f"{corr.m_HF[i]:.4f}", f"{corr.Z_BR[i]:.4f}",
            f"{corr.U_c[i]:.4f}", f"{J_hub_arr[i]*1e3:.3f}",
            f"{J_em_arr[i]*1e3:.3f}", f"{corr.enh_Hub[i]:.4f}",
            f"{corr.enh_Emery[i]:.4f}", str(int(corr.overpred[i])),
        ])
log(f"  CSV → {corr_path}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------

section("7. Figure generation  (src/figures.py)")

import matplotlib
matplotlib.use("Agg")

from src import figures as F
from src.lattice_bands import (
    build_kgrid, dispersion_square, t_of_P, tprime_of_P, mu_of_P,
    P_GRID_GPA as _P,
)

generated: list[Path] = []


# F01 — Band structure + DOS
def _band_and_dos() -> list[Path]:
    Nk_path = 200
    # Γ(0,0) → X(π,0) → M(π,π) → Γ(0,0)
    seg1 = np.linspace(0, np.pi, Nk_path // 3, endpoint=False)
    seg2 = np.linspace(0, np.pi, Nk_path // 3, endpoint=False)
    seg3 = np.linspace(0, np.pi, Nk_path - 2 * (Nk_path // 3), endpoint=True)
    kx = np.concatenate([seg1, np.full_like(seg2, np.pi), np.pi - seg3])
    ky = np.concatenate([np.zeros_like(seg1), seg2, np.pi - seg3])
    eps = dispersion_square(kx, ky, t_of_P(0.0), tprime_of_P(0.0), mu_of_P(0.0))
    kpath = np.linspace(0, 1, len(kx))

    # DOS via histogram
    kx_d, ky_d = build_kgrid(128, 128)
    eps_dos = dispersion_square(kx_d, ky_d, t_of_P(0.0), tprime_of_P(0.0), mu_of_P(0.0))
    dos_n, dos_e_edges = np.histogram(eps_dos.ravel(), bins=200, density=True)
    dos_e = 0.5 * (dos_e_edges[:-1] + dos_e_edges[1:])

    return F.figure_1_band_structure(
        kpath, eps,
        dos_e, dos_n,
        labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"],
    )


generated += _band_and_dos()
log(f"  F01 band structure → {[p.name for p in generated[-2:]]}")


# F02 — Fermi surface
def _fermi_surface() -> list[Path]:
    Nk = 200
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    pressures = [0.0, 10.0, 20.0, 30.0]
    eps_list = [
        dispersion_square(KX, KY, t_of_P(P), tprime_of_P(P), mu_of_P(P))
        for P in pressures
    ]
    return F.figure_2_fermi_surface(
        KX, KY, eps_list,
        P_labels=[f"P={P:.0f} GPa" for P in pressures],
    )


generated += _fermi_surface()
log(f"  F02 Fermi surface → {[p.name for p in generated[-2:]]}")


# F03 — BdG gap
generated += F.figure_3_bdg_gap(bdg)
log(f"  F03 BdG gap → {[p.name for p in generated[-2:]]}")


# F04 — BdG DOS
generated += F.figure_4_bdg_dos(bdg)
log(f"  F04 BdG DOS → {[p.name for p in generated[-2:]]}")


# F05 — Channels
generated += F.figure_5_channels(ct)
log(f"  F05 channels → {[p.name for p in generated[-2:]]}")


# F06 — Coherence
generated += F.figure_6_coherence(ct, P_OPT_GPA)
log(f"  F06 coherence → {[p.name for p in generated[-2:]]}")


# F07 — Two-scale
generated += F.figure_7_two_scale_cuprate(ts_table, HG1212_DATA)
log(f"  F07 two-scale → {[p.name for p in generated[-2:]]}")


# F08 — Null bootstrap
generated += F.figure_8_null_bootstrap(null_results)
log(f"  F08 null bootstrap → {[p.name for p in generated[-2:]]}")


# F09 — RPA χ map
generated += F.figure_9_rpa_chi(rpa)
log(f"  F09 RPA chi → {[p.name for p in generated[-2:]]}")


# F10 — RPA channels
generated += F.figure_10_rpa_channels(rpa)
log(f"  F10 RPA channels → {[p.name for p in generated[-2:]]}")


# F11 — Correlation HF
generated += F.figure_11_correlation_hf(corr)
log(f"  F11 correlation HF → {[p.name for p in generated[-2:]]}")


# F12 — Superexchange
generated += F.figure_12_superexchange(corr)
log(f"  F12 superexchange → {[p.name for p in generated[-2:]]}")


n_figs = len(generated)
log(f"\n  {n_figs} files written to {OUTPUT_DIR}/")
check(f"All 12 figures produced (24 files: PNG+PDF)", n_figs == 24)

# ---------------------------------------------------------------------------
# 8. Auto-audit report
# ---------------------------------------------------------------------------

section("8. Auto-audit report  (outputs/auto_auditoria.md)")

elapsed = time.time() - t0
all_pass = all(c for _, c in _acceptance)

# Collect package versions
pkgs = ["numpy", "scipy", "matplotlib", "pytest"]
ver_lines = [f"- `{p}` {_pkg_version(p)}" for p in pkgs]
ver_lines.append(f"- Python {sys.version.split()[0]}")
ver_lines.append(f"- Platform: {platform.platform()}")

# Run pytest count
try:
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q", "--no-header"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    test_summary = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "unknown"
except Exception as e:
    test_summary = f"pytest error: {e}"

audit = f"""# Auto-auditoria — Hg-1212/Hg-1223 cuprate pressure study

Gerado automaticamente por `run_all.py`
Data: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
Duração: {elapsed:.1f} s
Semente global: `SEED = {SEED}`

---

## 1. Dados utilizados

| Conjunto | Fonte | Status |
|---|---|---|
| `HG1212_DATA` | Chu et al. (1993); Nuñez-Regueiro et al. (1993) [APPROXIMATE — digitalizar] | APPROXIMATE |
| `HG1223_DATA` | — | **PLACEHOLDER** — `calibrate_model()` lança `ValueError` |

**Variáveis experimentais:**
- `P_GPa`: pressões de 0 a 30 GPa
- `Tc_onset_K`: temperatura de onset da resistência
- `Tc_zero_K`: temperatura de resistência zero

---

## 2. Placeholders explícitos

| Módulo | Quantidade | Descrição |
|---|---|---|
| `src/two_scale.py` | 1 | HG1223_DATA inteiro (status=PLACEHOLDER) |
| `src/lattice_bands.py` | vários | Harrison expoentes [EST/ASSUMED]; EOS de Birch-Murnaghan [LIT] |
| `src/pairing_bdg.py` | 1 | V_D_CALIB calibrado a Δ_d(P=0)≈30 meV [ASSUMED] |
| `src/correlation.py` | 5 | U_corr, α_Δ_pd, Harrison_N_pp, posições Wyckoff [ASSUMED] |
| `src/mediator_rpa.py` | 3 | U_HUB, T_RPA, η_FS [ASSUMED] |
| `src/qe_scaffold.py` | todos | Posições atômicas, constantes de rede, pseudopotenciais [PLACEHOLDER] |
| `qe_runs/` | 42 arquivos | Templates QE — não executados |

---

## 3. Parâmetros principais

### Tight-binding / EOS
| Parâmetro | Valor | Proveniência |
|---|---|---|
| t₀ (P=0) | 0.43 eV | [LIT] Andersen et al. PRB 1995 |
| t'/t₀ | −0.40 | [LIT] cuprate phenomenology |
| B (bulk modulus) | 100 GPa | [EST] cuprate típico |
| B' | 4.0 | [ASSUMED] |

### BdG / Pairing
| Parâmetro | Valor | Proveniência |
|---|---|---|
| V_D_CALIB | (calibrado) eV | alinha Δ_d(P=0)≈30 meV [ASSUMED] |
| σ_T bootstrap | {SIGMA_T} K | [ASSUMED] |
| NK_GAP | {NK_GAP} | [ASSUMED] |

### RPA
| Parâmetro | Valor | Proveniência |
|---|---|---|
| U_HUB | {U_HUB} eV | [ASSUMED] |
| T_RPA | {T_RPA_EV} eV | [ASSUMED] |
| η_FS | {ETA_FS_EV} eV | [ASSUMED] |

### Correlações (Hubbard-HF / Emery)
| Parâmetro | Valor | Proveniência |
|---|---|---|
| U_corr | {U_CORR_EV} eV | [ASSUMED] |
| t_pd₀ | 1.30 eV | [LIT] Hybertsen et al. PRB 1990 |
| Δ_pd₀ | 3.6 eV | [LIT] Emery 1987; ZSA 1985 |
| U_D | 8.0 eV | [LIT] |
| U_P | 4.0 eV | [EST] |

---

## 4. Testes executados

```
{test_summary}
```

Testes de sanidade verificam:
- Importabilidade de todos os módulos
- Chaves obrigatórias em `config.py`
- Stubs ainda não implementados levantam `NotImplementedError`
- Verificações físicas por módulo (ver `tests/test_sanity.py`)

---

## 5. Critérios de aceitação

| Critério | Resultado |
|---|---|
""" + "\n".join(
    f"| {label} | {'PASS' if cond else 'FAIL'} |"
    for label, cond in _acceptance
) + f"""

**Status global: {'APROVADO' if all_pass else 'REPROVADO'}**

---

## 6. Limitações

1. **Estrutura electrónica**: modelo tight-binding de banda única. Física de multi-banda de Hg-1212 (5 orbitais Cu-d + O-p) é ignorada.
2. **BdG Tc_MF ≠ Tc_onset**: a supressão de flutuações de fase é omitida. `BDG_DISCLAIMER` impresso em todos os contextos relevantes.
3. **Δ_d ≠ gap de transferência de carga**: `LABEL_A` afirma explicitamente que o gap Hubbard-HF de banda única não é o gap CT real do cuprato.
4. **RPA é aproximação de campo médio**: Stoner S < 1 por construção; spin-fluctuation vertex ignora vértice de Aslamazov-Larkin.
5. **Modelo de Emery super-prevê J**: detectado e reportado (`LABEL_D`; flag `overpred` no CSV).
6. **Dados HG1212 são APPROXIMATE**: digitalização manual; substituir por dados publicados tabulados.
7. **QE não executado**: scaffold gerado mas cálculos DFT dependem de pseudopotenciais e cluster.
8. **Modelos nulos são descritivos**: `RULE` afirma que qualidade de interpolação não implica mecanismo físico.

---

## 7. Taxonomia: dado / modelo / hipótese / rota DFT

### Dado (experimental)
- `Tc_onset(P)` e `Tc_zero(P)` do Hg-1212 [APPROXIMATE — digitalizar]
- Volume relativo V/V₀ via EOS de literatura

### Modelo (implementado)
- Two-scale: Tc_onset = C_coh × Tc_MF(BdG)
- BdG d-wave em rede quadrada tight-binding
- RPA Lindhard estática + interação singlete paramagnon
- Hubbard-HF, Brinkman-Rice Z, Emery Harrison scaling

### Hipótese (consistência, não prova)
- Mediador paramagnon identificado em nível RPA (`LABEL_RPA`)
- J_Emery cresce mais rápido que J_Hub sob pressão (`LABEL_D`)
- Z_BR ≈ 0 indica regime de Mott no composto pai a meia-banda

### Rota futura DFT
- Executar `qe_runs/P*kbar/submit_slurm.sh` após preparar pseudopotenciais
- Extrair t, t', μ via Wannier90 (`src/qe_scaffold.wannier_hoppings` — Phase 7+)
- Substituir parâmetros tight-binding por valores ab initio
- Calcular função de Green de DFT+DMFT para verificar Z_BR

---

## 8. Versões dos pacotes

{chr(10).join(ver_lines)}

---

## 9. Arquivos gerados

### CSVs
""" + "\n".join(f"- `outputs/{p.name}`" for p in OUTPUT_DIR.glob("*.csv")) + """

### Figuras (PNG 300 dpi + PDF vetorial)
""" + "\n".join(f"- `outputs/{p.name}`" for p in sorted(OUTPUT_DIR.glob("fig_*.png"))) + f"""

---

*Relatório gerado automaticamente. Não editar manualmente.*
*Para regenerar: `python run_all.py`*
"""

audit_path = OUTPUT_DIR / "auto_auditoria.md"
audit_path.write_text(audit, encoding="utf-8")
log(f"  Relatório → {audit_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

section("RESUMO FINAL")

log(f"  Tempo total: {elapsed:.1f} s")
log(f"  Semente: {SEED}")
log(f"  Figuras geradas: {n_figs} ({n_figs//2} PNG + {n_figs//2} PDF)")
log(f"  CSVs gerados: {len(list(OUTPUT_DIR.glob('*.csv')))}")
log()
log("  Critérios de aceitação:")
for label, cond in _acceptance:
    log(f"    [{'PASS' if cond else 'FAIL'}] {label}")

log()
log("  Avisos científicos obrigatórios:")
log(f"    [A] {LABEL_A}")
log(f"    [B] {LABEL_B}")
log(f"    [C] {LABEL_C}")
log(f"    [D] {LABEL_D}")
log(f"    [RPA] {LABEL_RPA}")
log(f"    [NULL] {RULE}")
log(f"    [BDG] {BDG_DISCLAIMER}")

log()
log(f"  Status global: {'APROVADO' if all_pass else 'REPROVADO'}")

sys.exit(0 if all_pass else 1)
