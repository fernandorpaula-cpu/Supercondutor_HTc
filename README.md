# Supercondutor_HTc

**Suíte de simulação para supercondutores cupratos de alta temperatura (Hg-1212 / Hg-1223) sob pressão**

> F. R. de Paula, UNESP

---

## Contexto científico

Esta suíte implementa um arcabouço teórico baseado em rede para estudar as
propriedades supercondutoras dos cupratos de mercúrio (HgBa₂CaCu₂O₆₊δ e
HgBa₂Ca₂Cu₃O₈₊δ) sob pressão hidrostática. O fluxo conecta o pareamento
local (escala BdG) à coerência de fase global por meio de um modelo de
duas escalas.

### Módulos físicos (`src/`)

| Módulo | Conteúdo |
|---|---|
| `lattice_bands.py` | Estrutura de bandas tight-binding; hopping dependente de pressão (regra de Harrison + EOS de Birch-Murnaghan) |
| `pairing_bdg.py` | Equação de gap BdG d-wave autoconsistente; Tc_MF(P), DOS; `Δ_d` = **proxy de pareamento local** (≠ Tc_zero) |
| `mediator_rpa.py` | Susceptibilidade RPA χ₀(q)/χ_RPA(q); vértice de flutuação de spin; critério de Stoner |
| `channels.py` | Decomposição do vértice d-wave (hopping + exchange); fator de coerência C_coh(P) |
| `two_scale.py` | Modelo de duas escalas: Tc_zero = C_coh × Tc_onset; calibrado a dados [APPROXIMATE] de Hg-1212 |
| `null_models.py` | Modelos nulos (linear, quadrático, saturante) para Tc_zero(P) e Wtr(P) com bootstrap |
| `correlation.py` | Proxy Hubbard-HF, peso de Brinkman-Rice, superexchange de Emery (três bandas) |
| `figures.py` | Biblioteca de 12 figuras (PNG 300 dpi + PDF vetorial) |
| `qe_scaffold.py` | Gerador de inputs Quantum ESPRESSO (vc-relax/scf/nscf/dos/projwfc) |

### Avisos científicos obrigatórios

- **`Δ_d` é um proxy de pareamento local — NUNCA igualado a Tc_zero.**
- **Tc_MF (campo médio BdG) ≠ Tc_onset** (flutuações de fase omitidas).
- **gap Hubbard-HF de banda única ≠ gap de transferência de carga real do cuprato.**
- **Mediador paramagnon (RPA): hipótese de consistência, não prova experimental.**
- **Qualidade de interpolação dos modelos nulos não implica mecanismo físico.**
- **Dados de Hg-1212 são [APPROXIMATE]; Hg-1223 é PLACEHOLDER** (`calibrate_model` lança `ValueError`).
- Todos os parâmetros são rotulados `[LIT]`, `[EST]` ou `[ASSUMED]`.

---

## Instalação

```bash
git clone https://github.com/fernandorpaula-cpu/Supercondutor_HTc.git
cd Supercondutor_HTc
pip install -r requirements.txt
```

Dependências: `numpy`, `scipy`, `matplotlib`, `pytest`.

---

## Uso

```bash
# Pipeline completo: roda todos os módulos, gera CSVs + 24 figuras + auditoria
python run_all.py

# Runners individuais
python run_bdg.py
python run_channels.py
python run_two_scale.py
python run_null_models.py
python run_rpa.py
python run_correlation.py
python run_qe_scaffold.py

# Testes (76 testes, incluindo os 10 critérios de aceitação CA-01..CA-10)
python -m pytest tests/test_sanity.py -q
```

---

## Saídas (`outputs/`)

- **CSVs**: `bdg_table.csv`, `channels_table.csv`, `two_scale_table.csv`,
  `null_models_residuals.csv`, `correlation_table.csv`, `rpa_table.csv`
- **Figuras**: 12 × (PNG 300 dpi + PDF vetorial) — geradas por `run_all.py`
- **`auto_auditoria.md`**: relatório com dados, placeholders, parâmetros,
  testes, critérios de aceitação, limitações, taxonomia dado/modelo/hipótese/rota-DFT,
  versões de pacotes e semente.

---

## Scaffold Quantum ESPRESSO (`qe_runs/`)

Árvore de inputs DFT por pressão (0–30 GPa). **QE não é executado localmente** —
posições atômicas, constantes de rede e pseudopotenciais são `[PLACEHOLDER]`.
Ver `qe_runs/README_SCAFFOLD.txt` para instruções de produção.

---

## Critérios de aceitação (CA-01 a CA-10)

Verificados automaticamente em `tests/test_sanity.py`:

1. `BDG_DISCLAIMER` presente (Tc_MF ≠ Tc_onset)
2. `INTERPRETATION_BLOCK` presente em two_scale
3. `RULE` presente em null_models
4. `LABEL_RPA` presente em mediator_rpa
5. `LABEL_A`..`LABEL_D` presentes em correlation
6. `Δ_d` rotulado "local pairing proxy"
7. `HG1212_DATA.status` contém "APPROXIMATE"
8. `calibrate_model(HG1223_DATA)` lança `ValueError`
9. 12 PNG + 12 PDF gerados em `outputs/`
10. `run_all.py` compila; `auto_auditoria.md` contém "APROVADO" com ≥ 16 PASS

---

## Licença

Código de pesquisa — contate o autor para termos de reuso.
