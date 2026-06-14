"""
run_correlation.py — Correlation and exchange proxies vs pressure.

Usage:
    python run_correlation.py [--Nk N] [--no-plot]

Outputs (in outputs/):
    correlation_table.csv       — full pressure-dependent proxy table
    fig_hf_proxy                — Δ_HF(P) and m(P), HF half-filling proxy
    fig_brinkman_rice           — Z_BR(P) and U_c(P) vs pressure
    fig_emery_params            — t_pd, t_pp, Δ_pd vs pressure
    fig_superexchange           — J_Hub, J_Emery, enhancement ratios vs pressure

MANDATORY LABELS:
    A. gap Hubbard-HF (banda única) ≠ gap de transferência de carga real do cuprato
    B. m(P) é proxy de meia-banda (half-filling); não é magnetização do cuprato dopado
    C. Z_BR é aproximação de Gutzwiller; validade limitada perto do limite de Mott
    D. J_Emery pode super-prever o realce de J sob pressão
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, FIGURES
from src.correlation import (
    correlation_scan, CorrelationScanResult, overprediction_report,
    LABEL_A, LABEL_B, LABEL_C, LABEL_D,
    U_CORR_EV, T_PD_0_EV, DELTA_PD_0_EV, U_D_EV, U_P_EV, NK_CORR,
)
from src.lattice_bands import P_GRID_GPA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI   = FIGURES["dpi"]
FMTS  = FIGURES["format"]
STYLE = "seaborn-v0_8-whitegrid"


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths: list[Path] = []
    for fmt in FMTS:
        p = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _style() -> None:
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


# ── CSV ───────────────────────────────────────────────────────────────────────

def export_csv(res: CorrelationScanResult, path: Path) -> None:
    cols = [
        "P_GPa", "t_eV", "Delta_HF_eV", "m_HF",
        "Z_BR", "U_c_eV",
        "t_pd_eV", "t_pp_eV", "Delta_pd_eV",
        "J_Hub_meV", "J_Emery_meV",
        "enh_Hub", "enh_Emery", "overpred_flag",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(res.P)):
            w.writerow([
                f"{res.P[i]:.2f}",
                f"{res.t[i]:.5f}",
                f"{res.Delta_HF[i]*1e3:.3f}",   # meV
                f"{res.m_HF[i]:.5f}",
                f"{res.Z_BR[i]:.5f}",
                f"{res.U_c[i]:.4f}",
                f"{res.t_pd[i]:.5f}",
                f"{res.t_pp[i]:.5f}",
                f"{res.Delta_pd[i]:.5f}",
                f"{res.J_Hub[i]*1e3:.3f}",       # meV
                f"{res.J_Em[i]*1e3:.3f}",         # meV
                f"{res.enh_Hub[i]:.5f}",
                f"{res.enh_Emery[i]:.5f}",
                f"{int(res.overpred[i])}",
            ])
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_hf_proxy(res: CorrelationScanResult) -> list[Path]:
    """Hubbard-HF gap Δ_HF(P) and magnetization m(P) at half-filling."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P = res.P

    ax1.plot(P, res.Delta_HF * 1e3, color="navy", lw=2.2,
             label=r"$\Delta_{\rm HF}(P)$")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel(r"$\Delta_{\rm HF}$ [meV]")
    ax1.set_title(
        r"Hubbard-HF gap $\Delta_{\rm HF}(P)$ — half-filling proxy" + "\n"
        f"[LABEL A: {LABEL_A}]",
        fontsize=7,
    )
    ax1.legend()
    ax1.set_xlim(P[0], P[-1])

    color2 = "darkred"
    ax2.plot(P, res.m_HF, color=color2, lw=2.2,
             label=r"$m_{\rm HF}(P) = \Delta_{\rm HF}/U_{\rm corr}$")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--",
                label="$m = 0.5$ (full sublattice polarization)")
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel(r"$m_{\rm HF}$ [dimensionless]")
    ax2.set_title(
        r"HF staggered magnetization $m_{\rm HF}(P)$ — half-filling proxy" + "\n"
        f"[LABEL B: {LABEL_B}]",
        fontsize=7,
    )
    ax2.legend()
    ax2.set_xlim(P[0], P[-1])
    ax2.set_ylim(0, None)

    fig.tight_layout()
    return _save(fig, "fig_hf_proxy")


def fig_brinkman_rice(res: CorrelationScanResult) -> list[Path]:
    """Brinkman-Rice Z(P) and critical U_c(P)."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P = res.P

    ax1.plot(P, res.Z_BR, color="seagreen", lw=2.2,
             label=r"$Z_{\rm BR}(P)$")
    ax1.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6,
                label="non-interacting $Z=1$")
    ax1.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6,
                label="Mott limit $Z=0$")
    ax1.fill_between(P, res.Z_BR, 0, alpha=0.15, color="seagreen")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel(r"$Z_{\rm BR}$ [dimensionless]")
    ax1.set_title(
        r"Brinkman-Rice quasiparticle weight $Z_{\rm BR}(P)$" + "\n"
        f"[LABEL C: {LABEL_C}]",
        fontsize=7,
    )
    ax1.legend(fontsize=8)
    ax1.set_xlim(P[0], P[-1])
    ax1.set_ylim(0, 1.1)

    ax2.plot(P, res.U_c, color="purple", lw=2.0,
             label=r"$U_c(P) = 2|\langle T\rangle_0|$")
    ax2.axhline(res.U_corr, color="firebrick", lw=1.5, ls="--",
                label=f"$U_{{\\rm corr}} = {res.U_corr:.1f}$ eV  [ASSUMED]")
    ax2.fill_between(P, res.U_c, res.U_corr,
                     where=res.U_c > res.U_corr,
                     alpha=0.20, color="seagreen", label=r"$U_c > U_{\rm corr}$ (metallic)")
    ax2.fill_between(P, res.U_c, res.U_corr,
                     where=res.U_c < res.U_corr,
                     alpha=0.20, color="firebrick", label=r"$U_c < U_{\rm corr}$ (Mott proxy)")
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel(r"Energy [eV]")
    ax2.set_title(
        r"$U_c(P)$ vs $U_{\rm corr}$: Mott proximity" + "\n"
        r"$U_c$ increases with $P$ (bandwidth broadens) $\Rightarrow$ $Z_{\rm BR}$ recovers",
        fontsize=8,
    )
    ax2.legend(fontsize=8)
    ax2.set_xlim(P[0], P[-1])

    fig.tight_layout()
    return _save(fig, "fig_brinkman_rice")


def fig_emery_params(res: CorrelationScanResult) -> list[Path]:
    """Emery model parameters: t_pd, t_pp, Δ_pd vs pressure."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P = res.P

    ax1.plot(P, res.t_pd, color="firebrick", lw=2.0,
             label=r"$t_{pd}(P)$ [LIT Harrison]")
    ax1.plot(P, res.t_pp, color="steelblue", lw=1.8, ls="--",
             label=r"$t_{pp}(P)$ [EST Harrison]")
    ax1.plot(P, res.t,    color="gray", lw=1.5, ls=":",
             label=r"$t(P)$ in-plane single-band")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel(r"Hopping [eV]")
    ax1.set_title(
        r"Emery model hoppings $t_{pd}(P)$, $t_{pp}(P)$ vs pressure" + "\n"
        r"$t_{pd}$: Cu-O; $t_{pp}$: O-O  [Harrison scaling from $V/V_0(P)$]",
        fontsize=8,
    )
    ax1.legend()
    ax1.set_xlim(P[0], P[-1])

    ax2.plot(P, res.Delta_pd, color="darkorange", lw=2.0,
             label=r"$\Delta_{pd}(P)$ [ASSUMED $\alpha=0.3$]")
    ax2.axhline(DELTA_PD_0_EV, color="gray", lw=0.8, ls="--",
                label=f"$\\Delta_{{pd,0}} = {DELTA_PD_0_EV:.1f}$ eV  [LIT]")
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel(r"$\Delta_{pd}(P)$ [eV]")
    ax2.set_title(
        r"Charge-transfer energy $\Delta_{pd}(P) = \Delta_{pd,0}(V/V_0)^{0.3}$" + "\n"
        r"$\Delta_{pd}$ decreases slowly — mainly ionic, weakly pressure-dependent",
        fontsize=8,
    )
    ax2.legend()
    ax2.set_xlim(P[0], P[-1])

    fig.tight_layout()
    return _save(fig, "fig_emery_params")


def fig_superexchange(res: CorrelationScanResult) -> list[Path]:
    """J_Hub, J_Emery and enhancement ratios — over-prediction highlighted."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    P = res.P

    # Panel 1: J values in meV
    ax = axes[0]
    ax.plot(P, res.J_Hub * 1e3, color="steelblue", lw=2.0,
            label=r"$J_{\rm Hub} = 4t^2/U_{\rm corr}$")
    ax.plot(P, res.J_Em * 1e3, color="firebrick", lw=2.0, ls="--",
            label=r"$J_{\rm Emery}$  [Zhang-Rice]")
    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("$J(P)$ [meV]")
    ax.set_title(
        r"Superexchange $J_{\rm Hub}(P)$ and $J_{\rm Emery}(P)$",
        fontsize=9,
    )
    ax.legend()
    ax.set_xlim(P[0], P[-1])

    # Panel 2: Enhancement ratios
    ax = axes[1]
    ax.plot(P, res.enh_Hub,   color="steelblue", lw=2.0,
            label=r"$J_{\rm Hub}(P)/J_{\rm Hub}(0)$")
    ax.plot(P, res.enh_Emery, color="firebrick", lw=2.0, ls="--",
            label=r"$J_{\rm Emery}(P)/J_{\rm Emery}(0)$")
    # Shade over-prediction region
    ax.fill_between(P, res.enh_Hub, res.enh_Emery,
                    where=res.overpred,
                    alpha=0.25, color="firebrick",
                    label="Emery over-predicts vs Hub [LABEL D]")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("Enhancement ratio")
    ax.set_title(
        "Pressure enhancement of $J$\n"
        r"[LABEL D: $J_{\rm Emery}$ over-predicts vs $J_{\rm Hub}$ — see text]",
        fontsize=8,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(P[0], P[-1])

    # Panel 3: Z_BR and Δ_HF together as correlation proxies
    ax = axes[2]
    ax2r = ax.twinx()
    ax.plot(P, res.Z_BR, color="seagreen", lw=2.0, label=r"$Z_{\rm BR}(P)$")
    ax.set_ylabel(r"$Z_{\rm BR}$", color="seagreen")
    ax.tick_params(axis="y", labelcolor="seagreen")
    ax2r.plot(P, res.m_HF, color="navy", lw=1.8, ls="--",
              label=r"$m_{\rm HF}(P)$ (proxy)")
    ax2r.set_ylabel(r"$m_{\rm HF}$", color="navy")
    ax2r.tick_params(axis="y", labelcolor="navy")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    ax.set_xlabel("$P$ [GPa]")
    ax.set_title(
        r"Correlation proxies: $Z_{\rm BR}$ and $m_{\rm HF}$ vs $P$" + "\n"
        r"$Z_{\rm BR}$ increases with $P$ (pressure delocalises carriers)",
        fontsize=8,
    )
    ax.set_xlim(P[0], P[-1])

    fig.suptitle(
        f"[LABEL D: {LABEL_D}]",
        fontsize=7, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_superexchange")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(res: CorrelationScanResult) -> None:
    opr = overprediction_report(res.P, res.J_Hub, res.J_Em)
    sep = "─" * 88
    print(sep)
    print("  CORRELATION PROXY SUMMARY")
    print(sep)
    print(f"  U_corr = {res.U_corr:.1f} eV [ASSUMED]  |  Nk = {res.Nk}")
    print(f"  t_pd₀ = {T_PD_0_EV:.2f} eV [LIT]  |  "
          f"Δ_pd₀ = {DELTA_PD_0_EV:.1f} eV [LIT]  |  "
          f"U_D = {U_D_EV:.1f} eV [LIT]  |  U_P = {U_P_EV:.1f} eV [EST]")
    print(sep)
    print(f"  [LABEL A] {LABEL_A}")
    print(f"  [LABEL B] {LABEL_B}")
    print(f"  [LABEL C] {LABEL_C}")
    print(f"  [LABEL D] {LABEL_D}")
    print(sep)

    hdr  = (f"{'P[GPa]':>8} {'t[eV]':>7} {'D_HF[meV]':>10} {'m_HF':>7}"
            f" {'Z_BR':>7} {'U_c[eV]':>8}"
            f" {'J_Hub[meV]':>11} {'J_Em[meV]':>10} {'enh_H':>7} {'enh_E':>7} {'OvPr':>5}")
    print("  " + hdr)
    print("  " + "─" * len(hdr))
    P = res.P
    idx = np.round(np.linspace(0, len(P) - 1, 13)).astype(int)
    for i in idx:
        flag = "*" if res.overpred[i] else " "
        print(
            f"  {P[i]:8.1f} {res.t[i]:7.4f} {res.Delta_HF[i]*1e3:10.2f}"
            f" {res.m_HF[i]:7.4f}"
            f" {res.Z_BR[i]:7.4f} {res.U_c[i]:8.4f}"
            f" {res.J_Hub[i]*1e3:11.3f} {res.J_Em[i]*1e3:10.3f}"
            f" {res.enh_Hub[i]:7.4f} {res.enh_Emery[i]:7.4f} {flag:>5}"
        )

    n_over = np.sum(res.overpred)
    print(sep)
    print(f"\n  Over-prediction (Emery > Hub enhancement) [LABEL D]:")
    print(f"    First P with flag: {opr['P_first_overpred']:.1f} GPa")
    print(f"    Over-predicted at {n_over}/{len(P)} pressure points")
    print(f"    Max excess (r_Emery - r_Hub): {opr['max_excess']:.3f}")
    if n_over == len(P) - 1:
        print(f"    [WARN] J_Emery over-predicts at ALL pressures P > 0 GPa.")
        print(f"    [WARN] t_pd^4 grows faster than t^2 under compression.")
        print(f"    [WARN] If Δ_pd(P) is overestimated (α too large), J_Emery")
        print(f"    [WARN] enhancement is even larger — treat as upper bound.")

    print(f"\n  P=0 values:")
    print(f"    t       = {res.t[0]:.4f} eV     J_Hub   = {res.J_Hub[0]*1e3:.2f} meV")
    print(f"    Δ_HF    = {res.Delta_HF[0]*1e3:.2f} meV    m_HF    = {res.m_HF[0]:.4f}")
    print(f"    Z_BR    = {res.Z_BR[0]:.4f}       U_c     = {res.U_c[0]:.4f} eV")
    print(f"    t_pd    = {res.t_pd[0]:.4f} eV    J_Emery = {res.J_Em[0]*1e3:.2f} meV")
    print(f"    Δ_pd    = {res.Delta_pd[0]:.4f} eV")
    print(f"\n  P=30 GPa values:")
    print(f"    t       = {res.t[-1]:.4f} eV     J_Hub   = {res.J_Hub[-1]*1e3:.2f} meV")
    print(f"    Z_BR    = {res.Z_BR[-1]:.4f}       U_c     = {res.U_c[-1]:.4f} eV")
    print(f"    J_Emery = {res.J_Em[-1]*1e3:.2f} meV   enh_Emery = {res.enh_Emery[-1]:.3f}x")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nk", type=int, default=NK_CORR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\nCorrelation proxy scan  (Nk={args.Nk}², U_corr={U_CORR_EV} eV [ASSUMED])\n"
          f"  LABEL A: {LABEL_A}\n"
          f"  LABEL B: {LABEL_B}\n"
          f"  LABEL C: {LABEL_C}\n"
          f"  LABEL D: {LABEL_D}\n")

    print(f"  Running scan over {len(P_GRID_GPA)} pressure points…")
    res = correlation_scan(P_grid=P_GRID_GPA, Nk=args.Nk)

    export_csv(res, OUTPUT_DIR / "correlation_table.csv")

    if not args.no_plot:
        print("\n  Generating figures…")
        for p in fig_hf_proxy(res):
            print(f"  FIG  → {p}")
        for p in fig_brinkman_rice(res):
            print(f"  FIG  → {p}")
        for p in fig_emery_params(res):
            print(f"  FIG  → {p}")
        for p in fig_superexchange(res):
            print(f"  FIG  → {p}")

    print()
    print_summary(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
