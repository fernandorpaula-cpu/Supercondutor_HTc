# Auto-auditoria — Hg-1212/Hg-1223 cuprate pressure study

Gerado automaticamente por `run_all.py`
Data: 2026-06-14 21:18:22 UTC
Duração: 18.7 s
Semente global: `SEED = 42`

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
| σ_T bootstrap | 1.5 K | [ASSUMED] |
| NK_GAP | 128 | [ASSUMED] |

### RPA
| Parâmetro | Valor | Proveniência |
|---|---|---|
| U_HUB | 0.3 eV | [ASSUMED] |
| T_RPA | 0.005 eV | [ASSUMED] |
| η_FS | 0.015 eV | [ASSUMED] |

### Correlações (Hubbard-HF / Emery)
| Parâmetro | Valor | Proveniência |
|---|---|---|
| U_corr | 1.5 eV | [ASSUMED] |
| t_pd₀ | 1.30 eV | [LIT] Hybertsen et al. PRB 1990 |
| Δ_pd₀ | 3.6 eV | [LIT] Emery 1987; ZSA 1985 |
| U_D | 8.0 eV | [LIT] |
| U_P | 4.0 eV | [EST] |

---

## 4. Testes executados

```
76 passed, 1 skipped in 1.47s
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
| Two-scale calibration RMSE_onset < 5 K | PASS |
| Hg1223 data remains PLACEHOLDER (ValueError guard) | PASS |
| Tc_zero best model = quadratic | PASS |
| Wtr saturating fit degenerate (rho > 1e6) | PASS |
| Delta_d(P=0) in [20, 45] meV | PASS |
| 2Delta/kBTc ratio in [3, 6] | PASS |
| V_d_eff(P) > lambda_hop(P) for all P | PASS |
| C_coh(P_opt) is maximum of C_coh | PASS |
| Stoner S < threshold at all P | PASS |
| lambda_d > 0 at P=0 | PASS |
| lambda_s < 0 at P=0 | PASS |
| d-wave preferred (lambda_d > lambda_s) at P=0 | PASS |
| Delta_HF > 0 at P=0 | PASS |
| Z_BR in [0,1] at all P | PASS |
| J_Emery over-predicts J_Hub at high P | PASS |
| All 12 figures produced (24 files: PNG+PDF) | PASS |

**Status global: APROVADO**

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

- `numpy` 2.4.6
- `scipy` 1.17.1
- `matplotlib` 3.11.0
- `pytest` 9.1.0
- Python 3.12.8
- Platform: Windows-10-10.0.19045-SP0

---

## 9. Arquivos gerados

### CSVs
- `outputs/bdg_table.csv`
- `outputs/channels_table.csv`
- `outputs/correlation_table.csv`
- `outputs/null_models_residuals.csv`
- `outputs/rpa_table.csv`
- `outputs/two_scale_table.csv`

### Figuras (PNG 300 dpi + PDF vetorial)
- `outputs/fig_band_structure.png`
- `outputs/fig_bdg_dos.png`
- `outputs/fig_bdg_gap.png`
- `outputs/fig_channels.png`
- `outputs/fig_coherence.png`
- `outputs/fig_correlation_hf.png`
- `outputs/fig_fermi_surface.png`
- `outputs/fig_null_bootstrap.png`
- `outputs/fig_rpa_channels.png`
- `outputs/fig_rpa_chi.png`
- `outputs/fig_superexchange.png`
- `outputs/fig_two_scale.png`

---

*Relatório gerado automaticamente. Não editar manualmente.*
*Para regenerar: `python run_all.py`*
