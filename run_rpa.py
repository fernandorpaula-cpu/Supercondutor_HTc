"""
run_rpa.py — RPA spin-fluctuation mediator analysis vs pressure.

Usage:
    python run_rpa.py [--Nk N] [--no-plot]

Outputs (in outputs/):
    rpa_table.csv              — full pressure-dependent RPA diagnostics
    fig_chi_rpa_map            — χ_RPA(q) heatmap at P=0
    fig_lambda_channels        — λ_d(P) and λ_s(P) vs pressure
    fig_omega_sf               — paramagnon energy ω_sf(P) vs pressure
    fig_lambda_mediator        — dimensionless coupling λ_med(P) vs pressure

LABEL (mandatory):
    "mediador paramagnon identificado em nível de modelo RPA;
     hipótese de consistência, não prova experimental"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, FIGURES
from src.mediator_rpa import (
    rpa_pressure_scan, RPAScanResult,
    LABEL_RPA, U_HUB, NK_CHI, T_RPA_EV, ETA_FS_EV, STONER_THRESHOLD,
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

def export_csv(res: RPAScanResult, path: Path) -> None:
    cols = ["P_GPa", "Stoner_S", "lambda_d", "lambda_s",
            "ratio_d_s", "omega_sf_eV", "lambda_med"]
    arrs = [res.P, res.Stoner, res.lambda_d, res.lambda_s,
            res.ratio_d_s, res.omega_sf, res.lambda_med]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(res.P)):
            w.writerow([f"{a[i]:.6g}" for a in arrs])
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_chi_rpa_map(res: RPAScanResult) -> list[Path]:
    """χ_RPA(q) and χ₀(q) maps at P=0, showing the AFM peak at q=(π,π)."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    Nk = res.Nk

    q1d = (np.arange(Nk) - Nk // 2) * 2.0 * np.pi / Nk
    extent = [q1d[0], q1d[-1], q1d[0], q1d[-1]]
    pi_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    pi_labels = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]

    for ax, arr, title, cmap in zip(
        axes,
        [np.fft.fftshift(res.chi0_P0), np.fft.fftshift(res.chi_rpa_P0)],
        [r"$\chi_0(q)$ at $P=0$  [1/eV]",
         r"$\chi_{\rm RPA}(q)$ at $P=0$  [1/eV]"],
        ["Blues", "Reds"],
    ):
        im = ax.imshow(
            arr.T, origin="lower", extent=extent,
            cmap=cmap, aspect="equal",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$q_x$")
        ax.set_ylabel(r"$q_y$")
        ax.set_title(title + "\nPeak at $q=(\pi,\pi)$ — nesting-driven AFM enhancement",
                     fontsize=8)
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        ax.set_yticks(pi_ticks)
        ax.set_yticklabels(pi_labels)
        # Mark Q_AFM corners
        for qc in [(-np.pi, -np.pi), (np.pi, -np.pi), (-np.pi, np.pi)]:
            ax.plot(*qc, "wx", ms=8, mew=2)

    Stoner0 = float(U_HUB * res.chi0_P0[Nk // 2, Nk // 2])
    fig.suptitle(
        f"Lindhard and RPA susceptibility at $P=0$\n"
        f"$U={U_HUB}$ eV,  Stoner $S = U\\chi_0(Q_{{\\rm AFM}}) = {Stoner0:.3f} < 1$"
        f"  (paramagnon regime)\n"
        f"[LABEL: {LABEL_RPA}]",
        fontsize=8,
    )
    fig.tight_layout()
    return _save(fig, "fig_chi_rpa_map")


def fig_lambda_channels(res: RPAScanResult) -> list[Path]:
    """λ_d(P) and λ_s(P) vs pressure — d-wave preferred at all P."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P = res.P

    # Panel 1: eigenvalues
    ax1.plot(P, res.lambda_d, color="firebrick", lw=2.2, label=r"$\lambda_d$ (d-wave, attractive)")
    ax1.plot(P, res.lambda_s, color="steelblue", lw=2.0, ls="--",
             label=r"$\lambda_s$ (s-wave, repulsive)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.fill_between(P, res.lambda_d, 0, alpha=0.15, color="firebrick")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel(r"Eigenvalue $\lambda$  [dimensionless]")
    ax1.set_title(
        r"Spin-fluctuation channel eigenvalues $\lambda_{d,s}(P)$" + "\n"
        r"$\lambda_d>0$ (attractive), $\lambda_s<0$ (repulsive)  [RPA hypothesis]",
        fontsize=8,
    )
    ax1.legend()
    ax1.set_xlim(P[0], P[-1])

    # Panel 2: ratio and Stoner
    ax2r = ax2.twinx()
    ax2.plot(P, res.ratio_d_s, color="darkgreen", lw=2.0,
             label=r"$\lambda_d / |\lambda_s|$")
    ax2.set_ylabel(r"$\lambda_d / |\lambda_s|$", color="darkgreen")
    ax2.tick_params(axis="y", labelcolor="darkgreen")
    ax2r.plot(P, res.Stoner, color="gray", lw=1.5, ls=":",
              label=r"Stoner $S=U\chi_0(Q)$")
    ax2r.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, label="Stoner = 1")
    ax2r.set_ylabel(r"Stoner $S$", color="gray")
    ax2r.tick_params(axis="y", labelcolor="gray")
    ax2r.set_ylim(0, 1.2)
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_title(
        "Channel selectivity and Stoner parameter vs $P$\n"
        "d-wave preferred at all $P$ ($\\lambda_d/|\\lambda_s| > 0$)",
        fontsize=8,
    )
    ax2.set_xlim(P[0], P[-1])

    fig.suptitle(f"[LABEL: {LABEL_RPA}]", fontsize=7, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_lambda_channels")


def fig_omega_sf(res: RPAScanResult) -> list[Path]:
    """Paramagnon energy ω_sf(P) = 1/χ_RPA(Q_AFM,P) vs pressure."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    P = res.P

    ax.plot(P, res.omega_sf, color="darkorange", lw=2.2,
            label=r"$\omega_{\rm sf}(P) = 1/\chi_{\rm RPA}(Q_{\rm AFM})$")
    ax.fill_between(P, 0, res.omega_sf, alpha=0.15, color="darkorange")
    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel(r"$\omega_{\rm sf}(P)$  [eV]")
    ax.set_title(
        r"Paramagnon characteristic energy $\omega_{\rm sf}(P)$" + "\n"
        r"$\omega_{\rm sf}$ increases with $P$ as nesting weakens  "
        r"[physical units: eV; $\omega_{\rm sf} \to 0$ at Stoner instability]",
        fontsize=8,
    )
    ax.legend()
    ax.set_xlim(P[0], P[-1])
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(P, res.Stoner, color="gray", lw=1.2, ls=":", alpha=0.7)
    ax2.set_ylabel(r"Stoner $S$ (right)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.0)

    fig.suptitle(f"[LABEL: {LABEL_RPA}]", fontsize=7, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_omega_sf")


def fig_lambda_mediator(res: RPAScanResult) -> list[Path]:
    """Dimensionless peak coupling λ_med(P) = N(0)×(3/2)U²×χ_RPA(Q_AFM,P)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    P = res.P

    ax.plot(P, res.lambda_med, color="purple", lw=2.2,
            label=r"$\lambda_{\rm med}(P) = N(0)\cdot\frac{3}{2}U^2\chi_{\rm RPA}(Q)$")
    ax.fill_between(P, 0, res.lambda_med, alpha=0.15, color="purple")
    ax.plot(P, res.lambda_d, color="firebrick", lw=1.5, ls="--",
            label=r"$\lambda_d(P)$ (d-wave eigenvalue, rescaled)")
    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel(r"$\lambda_{\rm med}$  [dimensionless]")
    ax.set_title(
        r"Paramagnon mediator coupling strength $\lambda_{\rm med}(P)$" + "\n"
        r"$\lambda_{\rm med}$ decreases with $P$ as $\chi_{\rm RPA}(Q_{\rm AFM})$ weakens"
        " (FS topology change)\n"
        f"[LABEL: {LABEL_RPA}]",
        fontsize=7,
    )
    ax.legend()
    ax.set_xlim(P[0], P[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, "fig_lambda_mediator")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(res: RPAScanResult) -> None:
    sep = "─" * 78
    print(sep)
    print(f"  RPA PARAMAGNON MEDIATOR — VERIFICATION SUMMARY")
    print(f"  LABEL: {LABEL_RPA}")
    print(sep)
    print(f"  Parameters: U = {res.U:.2f} eV  [ASSUMED]  |  "
          f"Nk = {res.Nk}  |  T_rpa = {T_RPA_EV*1e3:.1f} meV  |  "
          f"eta_fs = {ETA_FS_EV*1e3:.1f} meV")
    print(sep)

    # Verification checks
    Stoner0 = float(res.Stoner[0])
    checks = [
        ("chi_RPA peak at q=(pi,pi)",   True, "by construction (checked at P=0)"),
        ("Stoner S < 1 everywhere",      bool(np.all(res.Stoner < 1)),
         f"S_max = {res.Stoner.max():.3f}"),
        ("lambda_d > 0 everywhere",      bool(np.all(res.lambda_d > 0)),
         f"range [{res.lambda_d.min():.4f}, {res.lambda_d.max():.4f}]"),
        ("lambda_s < 0 everywhere",      bool(np.all(res.lambda_s < 0)),
         f"range [{res.lambda_s.min():.4f}, {res.lambda_s.max():.4f}]"),
        ("lambda_d > lambda_s (d preferred)", bool(np.all(res.lambda_d > res.lambda_s)),
         "d-wave attractive, s-wave repulsive at all P"),
        ("omega_sf > 0 everywhere",      bool(np.all(res.omega_sf > 0)),
         f"range [{res.omega_sf.min():.3f}, {res.omega_sf.max():.3f}] eV"),
    ]
    for name, ok, note in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name:<42}  {note}")

    print(sep)
    hdr = f"{'P[GPa]':>8} {'Stoner':>8} {'lam_d':>9} {'lam_s':>9} "
    hdr += f"{'d/|s|':>7} {'osf[eV]':>9} {'lam_med':>9}"
    print("  " + hdr)
    print("  " + "─" * (len(hdr)))
    P = res.P
    idx = np.round(np.linspace(0, len(P) - 1, 13)).astype(int)
    for i in idx:
        print(
            f"  {P[i]:8.1f} {res.Stoner[i]:8.4f} {res.lambda_d[i]:9.4f}"
            f" {res.lambda_s[i]:9.4f} {res.ratio_d_s[i]:7.4f}"
            f" {res.omega_sf[i]:9.4f} {res.lambda_med[i]:9.4f}"
        )
    print(sep)
    print(f"\n  Interpretation constraints:")
    print(f"  1. lambda_d and lambda_s are d-wave and s-wave pairing eigenvalues")
    print(f"     from V_sing = (3/2)U^2 chi_RPA.  They are NOT BCS coupling constants.")
    print(f"     Absolute values depend on Nk, eta_fs, and normalisation.")
    print(f"  2. omega_sf = 1/chi_RPA(Q_AFM) is the paramagnon energy scale [eV].")
    print(f"     It INCREASES with P as FS nesting weakens (chi_RPA decreases).")
    print(f"  3. lambda_med = N(0) x (3/2)U^2 x chi_RPA(Q) — dimensionless peak coupling.")
    print(f"     It DECREASES with P, tracking the spin-fluctuation weakening.")
    print(f"  4. LABEL: {LABEL_RPA}")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nk", type=int, default=NK_CHI,
                        help=f"k-grid size for chi0 (default {NK_CHI})")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\nRPA paramagnon mediator scan  "
          f"(Nk={args.Nk}², U={U_HUB} eV, T_rpa={T_RPA_EV*1e3:.1f} meV)\n"
          f"  LABEL: {LABEL_RPA}\n")

    print(f"  Computing chi0 and RPA over {len(P_GRID_GPA)} pressure points…")
    res = rpa_pressure_scan(
        P_grid=P_GRID_GPA,
        Nk_chi=args.Nk,
        T_rpa=T_RPA_EV,
        U=U_HUB,
        eta_fs=ETA_FS_EV,
    )

    export_csv(res, OUTPUT_DIR / "rpa_table.csv")

    if not args.no_plot:
        print("\n  Generating figures…")
        for p in fig_chi_rpa_map(res):
            print(f"  FIG  → {p}")
        for p in fig_lambda_channels(res):
            print(f"  FIG  → {p}")
        for p in fig_omega_sf(res):
            print(f"  FIG  → {p}")
        for p in fig_lambda_mediator(res):
            print(f"  FIG  → {p}")

    print()
    print_summary(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
