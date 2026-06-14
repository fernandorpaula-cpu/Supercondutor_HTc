# PROMPT PARA O CLAUDE CODE — Suíte de simulação blindada: pressão, correlação e mediadores em Hg-cupratos

> Cole o conteúdo a partir de "## OBJETIVO" como instrução inicial no Claude Code, num diretório vazio.
> É autocontido: carrega o arcabouço físico, os dados do artigo, a matriz de alegações, os controles
> estatísticos e os critérios de aceitação necessários para construir o programa do zero e blindá-lo
> contra revisão por pares.

---

## OBJETIVO

Construa uma suíte de simulação **modular, robusta, reprodutível e defensável em revisão por pares**
em Python, que estude a relação entre **pressão e correlação eletrônica** em supercondutores de
cuprato de mercúrio (Hg1212 bicamada; Hg1223 trilayer como restrição complementar), com extensão a
níquelatos. O programa deve produzir, com figuras, tabelas e testes automáticos, respostas sobre:

1. Como o **gap supercondutor** e a **densidade de estados (DOS)** variam com a pressão.
2. **Quem media a interação entre os elétrons** — identificar e quantificar o mediador do pareamento
   (flutuações de spin/paramagnons): intensidade, energia característica, estrutura em momento e
   dependência com a pressão. Tratar isso como **hipótese testada por consistência, não provada**.
3. A **decomposição em duas escalas**: pareamento local Δ_d(P) e coerência global C_coh(P) são
   distintos; T_c^zero(P) ≈ C_coh(P)·T_c^onset(P), com W_tr = T_c^onset − T_c^zero.

Trabalhe em **fases**; rode ao final de cada módulo, mostre a saída e corrija antes de prosseguir.
Não é DFT: é arcabouço de modelo efetivo calibrado a dados. Gere o scaffold ab initio (Quantum
ESPRESSO) como rota de validação para cluster/AWS, sem executá-lo localmente.

---

## ARCABOUÇO FÍSICO (cadeia de redução do artigo, Fig6)

Emery-type Cu–O model  →  projeção de baixa energia Zhang–Rice  →  modelo efetivo de bicamada.
A pressão hidrostática P é projetada em canais reduzidos retidos no arcabouço (Fig6C):

    P  →  t_perp(P) , J_perp(P)  →  V_d^eff(P) (pareamento)  e  C_coh(P) (coerência)
    Δ_d(P) permanece um *proxy de pareamento local*.

Renormalizações (todas editáveis no topo):
- **Banda** (Fig1): t(P)=t0(1−P/3B)^(−n); t′(P)=tp_rat·t(P).
- **Hibridização/transferência de carga Cu–O** (Fig6A): t_pd(P)↑, t_pp(P)↑, Δ_pd(P)↓ por Harrison.
- **Troca** J_perp(P) (Fig3B): proxy que cresce superlinearmente com P.
- **Coerência** C_coh(P): domo suave (Fig3/Fig4), **nunca forçada a zero**.
- **Competição** F_comp(P) (Fig3B): ~1 e cai levemente em alta P (degrada o canal local).

Equações a implementar:
- Dispersão 1 banda: ε(k) = −2t(cos kx + cos ky) + 4t′ cos kx cos ky − μ.
- Emery 3 bandas (Cu d, O px, O py) para caráter orbital e Δ_pd.
- Gap de onda-d: Δ(k)=Δ_d(cos kx − cos ky), via equação BdG autoconsistente.
- Superexchange: banda reduzida J=4t²/U ; Emery J ∝ t_pd⁴/Δ_pd²(1/U_d + 2/(2Δ_pd+U_p)).
- **Mediador (paramagnons)**: χ0(q,ω) Lindhard; χ_RPA=χ0/(1−Uχ0), pico em (π,π);
  interação singleto V(k,k′)=(3/2)U²χ_RPA(k−k′); autovalor de pareamento λ resolvido na
  superfície de Fermi (onda-d deve ser líder); energia do mediador ω_sf(P) e acoplamento λ(P).
- Duas escalas: C_coh(P)=T_c^zero/T_c^onset (fração física, domo suave); T_c^onset ∝ Δ_d
  (razão 2Δ_max/k_B T_c ≈ 5).

---

## DADOS DO ARTIGO — Hg1212 (EXATO, redigitalização secundária; embutir)

Fonte: redigitalização manual secundária da figura de transporte de Yamamoto et al.
Incertezas: σ_T = 1.5 K ; σ_P = 0.3 GPa. (Hg1212 = alvo bicamada primário.)

```
P (GPa):        0     2     4     8     12    16    19
Tc_onset (K):   126   128   133   145   148   158   160
Tc_mid   (K):   116   122   129   139   144   145   142     # aprox. (Fig2A) — verificar
Tc_zero  (K):   104   118   123   135   139   136   124
Wtr (K):         22    10    10    10     9    22    36     # = onset − zero
# Decomposição diagnóstica (Fig4), normalizada ao pico em P=12:
Ccoh_norm:      0.52  0.97  0.97  0.97  1.00  0.50  0.00
Delta_d_norm:   0.00  0.19  0.34  0.57  0.74  0.88  0.96
```

## DADOS DO ARTIGO — Hg1223 (PLACEHOLDER aproximado; SUBSTITUIR pela digitização do autor)

Hg1223 = trilayer, **restrição complementar**, NÃO evidência direta de bicamada (Fig7).
Comportamento documentado (Fig2C–D): onset e zero **sobem monotonicamente juntos**, transição
**estreita** (sem domo, sem colapso). Carregue o CSV digitizado do autor se existir; senão use
estes valores aproximados apenas como placeholder e marque-os como tais nas figuras:

```
P (GPa):        0     4     8     12    16    19    22
Tc_onset (K):   131   142   149   153   156   159   160    # APROX — substituir
Tc_zero  (K):   130   137   143   147   151   153   153    # APROX — substituir
Wtr (K):          1     6     6     6     5     6     7     # APROX — substituir
```

**Regra de dados (obrigatória):** nunca inventar números. Se um CSV do autor estiver presente em
`data/`, carregá-lo e usá-lo no lugar dos placeholders; caso contrário, rotular claramente como
"placeholder aproximado — não usar para alegações quantitativas".

---

## MATRIZ DE ALEGAÇÕES (manifesto do artigo — embutir e RESPEITAR)

Toda saída do programa deve mapear a uma alegação SUSTENTADA e nunca exceder uma NÃO sustentada.

| Figura | SUSTENTA | NÃO sustenta |
|---|---|---|
| Fig1 | pressão é multicanal; hopping-only é controle | lei de pressão absoluta; mecanismo de canal único |
| Fig2 | onset, mid, zero e Wtr são restrições redigitalizadas distintas | dado numérico original; inputs microscópicos exatos |
| Fig3 | canal exchange-enhanced gera janela de pareamento local além do controle hopping-only | previsão de Tc(P) absoluta |
| Fig4 | Δ_d e resistência zero são escalas separáveis | lei de Tc(P) absoluta; identificação Δ_d ≡ Tc_zero |
| Fig5 | descritores nulos e bootstrap restringem a interpretação | validação física de menor-resíduo única |
| Fig6 | física Cu–O motiva variáveis de pressão reduzidas e fator de coerência separado | cálculo Emery completo ou ab initio de pressão |
| Fig7 | Hg1223 requer modelagem resolvida OP/IP | validação de bicamada por Hg1223 |

---

## CONTROLES E MODELOS NULOS (blindagem estatística — Fig3, Fig5)

Implementar OBRIGATORIAMENTE, pois são o que sustenta o artigo perante referee:

1. **Controle hopping-only** (Fig3A): além do canal exchange-enhanced, computar o modelo só com
   banda (hopping) como CONTROLE. A diferença (exchange − hopping) deve ser positiva = canal de
   troca real (Fig3D). Reportar sempre o controle ao lado do modelo.
2. **Modelos nulos** (Fig5A–B): ajustar descritores nulos (linear, quadrático, saturante) a
   T_c^zero(P) e W_tr(P); reportar RMSE e MAE e os **resíduos** (observado − descritor) por pressão.
3. **Bootstrap** (Fig5C): reamostrar dentro de σ_T; reportar intervalos e um **índice de
   condicionamento** ρ por quantidade/composto; ρ>1 sinaliza fraca restrição (dado esparso).
4. **Regra de interpretação** (Fig5D): *qualidade de interpolação ≠ mecanismo físico*. Descritores
   nulos testam se a topologia de pressão é interpolável; o modelo efetivo testa **separação de
   canais**, não minimização de resíduo. Imprimir essa regra no relatório.
5. **Consistência microscópica honesta**: o superexchange (4t²/U; Emery t_pd⁴/Δ_pd²) **superprevê**
   o realce do pareamento sob pressão — cruzar com a calibração e **reportar a discrepância**,
   nunca escondê-la.

---

## ESCOPO E LIMITES (Fig7 — Hg1223 OP/IP)

Hg1212 = duas camadas CuO₂ equivalentes (canal efetivo de bicamada). Hg1223 = trilayer com planos
externo (OP) e interno (IP) **inequivalentes** → dispersões separadas → amplitudes de pareamento
diferentes → redistribuição de carga → travamento de fase interplanar. Portanto: tratar Hg1223
apenas como **restrição complementar**; qualquer modelagem de bicamada do Hg1223 deve ser marcada
como não-validada (requer modelo OP/IP resolvido). Não alegar validação de bicamada via Hg1223.

---

## ESTRUTURA DO REPOSITÓRIO

```
supercond-pressure/
├── config.py              # TODOS os parâmetros no topo; presets cuprate/nickelate
├── data/                  # CSVs do autor (se houver); senão placeholders rotulados
├── src/
│   ├── lattice_bands.py   # estrutura cristalina; dispersões TB e Emery; canais de pressão
│   ├── correlation.py     # Hubbard HF-AFM (gap correl., m, DOS) com t′; Brinkman-Rice Z(P)
│   ├── pairing_bdg.py     # gap BdG onda-d; Δ_d(P), Tc; DOS N(E,P)
│   ├── mediator_rpa.py    # χ0, χ_RPA, V(k,k′), autovalor λ_d/λ_s, ω_sf(P), λ(P)  [mediador]
│   ├── channels.py        # t_perp,J_perp,V_d^eff,C_coh,F_comp; CONTROLE hopping-only (Fig3)
│   ├── null_models.py     # descritores nulos + resíduos + bootstrap + índice de condic. (Fig5)
│   ├── two_scale.py       # Tc_onset, Tc_zero=Ccoh·Tonset; calibração + bootstrap + previsões
│   ├── qe_scaffold.py     # gerador de inputs Quantum ESPRESSO (vc-relax→scf→nscf→dos.x→projwfc)
│   └── figures.py         # plot unificado (cores de conceito, Type42, 300 dpi, PNG+PDF)
├── run_all.py             # orquestra tudo; CSV + figuras em outputs/; imprime tabela-resumo
├── outputs/
├── tests/test_sanity.py   # critérios de aceitação automáticos
└── README.md              # física, suposições, escopo, execução, matriz de alegações
```

---

## CONVENÇÕES (obrigatórias)

- Energias em eV; temperaturas em K; k_B = 8.617333e-5 eV/K.
- **Todos os parâmetros no topo** de `config.py`; sem números mágicos no meio do código.
- Modular, type hints, docstrings curtas, sem estado global, **semente fixa** (reprodutibilidade).
- Cada análise exporta **CSV + PNG (300 dpi) + PDF (vetorial, Type 42)** em `outputs/`.
- Legendas autocontidas; dependências mínimas (numpy, scipy, matplotlib).
- Cores de conceito consistentes: pareamento=verde, coerência=azul, zero=laranja, Cu-d=magenta,
  O-p=âmbar, mediador=vinho, controle=cinza tracejado.
- Registrar versões de pacotes e a semente no relatório (rastreabilidade).

---

## CRITÉRIOS DE ACEITAÇÃO (tests/test_sanity.py)

1. χ_RPA tem máximo em q=(π,π) (instabilidade AF).
2. Autovalor de pareamento d-wave é líder: λ_d > λ_s em P=0.
3. Δ_d(0) ∈ [25,40] meV e T_c^onset(0)=126±2 K após calibração; 2Δ_max/k_B T_c^onset ∈ [4,6].
4. RMSE(T_c^zero modelo vs Hg1212) < 8 K total e < 6 K no ramo P≥12 GPa.
5. Controle hopping-only é ~plano (não gera janela de pareamento); (exchange − hopping) > 0.
6. Modelos nulos reportados (RMSE/MAE + resíduos); bootstrap com índice de condicionamento.
7. DOS BdG em V com picos de coerência; gap de correlação > 0; Δ_pd(P) decrescente.
8. C_coh(P) domo suave com piso > 0 (sem colapso a zero).
9. Hg1223 tratado só como restrição complementar (sem alegação de validação de bicamada).
10. Todos os CSV/PNG/PDF esperados existem; relatório de auto-auditoria gerado.

---

## RESTRIÇÕES DE HONESTIDADE (não violar — alinhadas ao manifesto)

- Rotular sempre: **dado** vs **previsão de modelo** vs **(futuro) DFT**.
- NÃO alegar lei **absoluta** de Tc(P); o diagnóstico é de **forma/escala separável**.
- Gap de Hubbard de banda única é escala **Mott (~U)**; gap real do cuprato é **transferência de
  carga (~Δ_pd)**, menor — computar e rotular ambos.
- Identificação do mediador é **nível de modelo (RPA, paramagnons)**: hipótese testada por
  consistência; validação independente = rota ab initio (EPW p/ fônons; cRPA/TDDFT p/ spin).
- QE/VASP **não** rodam localmente: gerar scaffold; verificar inputs e pseudopotenciais antes do
  cluster/AWS.

---

## ANTECIPAÇÃO DE REFEREE (o programa deve pré-responder)

- "Você só ajustou uma curva." → mostrar **controle hopping-only** e **modelos nulos**; o modelo
  efetivo testa **separação de canais**, não minimização de resíduo (Fig5D).
- "É circular (poucos pontos)." → **bootstrap** + **índice de condicionamento** + declaração de
  dado esparso redigitalizado; intervalos como diagnósticos de incerteza, não constraints definitivos.
- "Cadê o mecanismo?" → módulo **mediador** (χ_RPA em (π,π), λ_d líder, ω_sf(P), λ(P)) como
  hipótese falseável, com a rota ab initio explicitada.
- "Tc(P) absoluto está errado." → o trabalho **não** alega Tc(P) absoluto; só **separação de escalas**.
- "Generaliza para Hg1223?" → não; Hg1223 é trilayer OP/IP, restrição complementar (Fig7).
- "Superexchange exagera o realce." → reportado explicitamente; calibração aos dados o atenua.

---

## FLUXO AB INITIO (documentar no README; gerar inputs por pressão)

Por pressão P (kbar=10·GPa): vc-relax (com `press`) → scf → nscf (malha densa) → dos.x + projwfc.x
(DOS total e PDOS Cu-3d/O-2p). Comparar N(E_F)(P), Δ_pd(P) e bandas DFT com o modelo. Opcional:
EPW para λ fonônica (mediador ab initio independente).

---

## EXECUÇÃO (Windows PowerShell, caminho explícito do Python)

```powershell
& "C:\Python312\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install numpy scipy matplotlib pytest
& ".\.venv\Scripts\python.exe" run_all.py
& ".\.venv\Scripts\python.exe" -m pytest tests/test_sanity.py -q
```

---

## ENTREGÁVEIS FINAIS

1. Repositório que roda de ponta a ponta com `run_all.py`.
2. `outputs/` com: Δ_d(P) e DOS N(E,P); gap de correlação, m(P), Δ_pd(P), Z(P); DOS Emery por
   orbital; **mediador** (mapa χ_RPA(q), λ_d/λ_s(P), ω_sf(P), λ(P)); duas escalas (curvas + bandas
   de bootstrap + previsão até 30 GPa); **controle hopping-only** e **modelos nulos/resíduos**;
   comparação Hg1212 vs Hg1223 vs níquelato; CSV+PNG+PDF de tudo.
3. `qe_scaffold/` com árvore de inputs QE por pressão + submit SLURM.
4. `tests/test_sanity.py` passando.
5. Seção LaTeX ("Origem microscópica dos canais e mediadores") pronta para Overleaf, referenciando
   os PDFs, com a **matriz de alegações** (sustenta / não sustenta) explícita.
6. **Auto-auditoria final** (estilo manifesto): listar cada critério de aceitação com o valor obtido;
   relatar suposições, limitações, checagens que falharam, versões e semente.
