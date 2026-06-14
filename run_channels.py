"""
run_channels.py — Compute, export, and plot all channel diagnostics.

Usage:
    python run_channels.py [--Nk N] [--no-plot]

Outputs (in outputs/):
    channels_table.csv    — full pressure-resolved table
    fig_hopping_params    — t(P), t'(P), t_perp(P), J_perp(P)
    fig_channel_vertex    — V_d_eff, lambda_hop, lambda_exch, exchange−hopping
    fig_coherence         — C_coh(P) dome and F_comp(P)
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

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, FIGURES
from src.channels import channel_table, P_OPT_GPA
from src.lattice_bands import P_GRID_GPA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE     = "seaborn-v0_8-whitegrid"
FONT      = "serif"
DPI       = FIGURES["dpi"]
FMTS      = FIGURES["format"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths: list[Path] = []
    for fmt in FMTS:
        p = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _apply_style() -> None:
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(tbl: dict[str, np.ndarray], path: Path) -> None:
    cols = [
        "P_GPa", "t_eV", "tprime_eV", "t_perp_eV", "J_perp_eV",
        "lambda_hop", "lambda_exch", "V_d_eff",
        "exch_minus_hop", "C_coh", "F_comp",
    ]
    keys = [
        "P", "t", "tprime", "t_perp", "J_perp",
        "lambda_hop", "lambda_exch", "V_d_eff",
        "exch_minus_hop", "C_coh", "F_comp",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for i in range(len(tbl["P"])):
            writer.writerow([f"{tbl[k][i]:.6g}" for k in keys])
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_hopping_params(tbl: dict) -> list[Path]:
    """Four-panel: t, |t'/t|, t_perp, J_perp vs pressure."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    P = tbl["P"]

    axes[0, 0].plot(P, tbl["t"],        color="steelblue",  lw=1.8)
    axes[0, 0].set_ylabel("$t(P)$ [eV]")
    axes[0, 0].set_title("Nearest-neighbour hopping")

    ratio = np.abs(tbl["tprime"] / tbl["t"])
    axes[0, 1].plot(P, ratio,           color="darkorange",  lw=1.8)
    axes[0, 1].set_ylabel("$|t'(P)/t(P)|$")
    axes[0, 1].set_title("NNN / NN hopping ratio")

    axes[1, 0].plot(P, tbl["t_perp"]*1e3, color="seagreen", lw=1.8)
    axes[1, 0].set_ylabel("$t_\\perp(P)$ [meV]")
    axes[1, 0].set_title("Interlayer hopping")

    axes[1, 1].plot(P, tbl["J_perp"]*1e3, color="firebrick", lw=1.8)
    axes[1, 1].set_ylabel("$J_\\perp(P)$ [meV]")
    axes[1, 1].set_title("Interlayer superexchange")

    for ax in axes.flat:
        ax.set_xlabel("$P$ [GPa]")
        ax.set_xlim(P[0], P[-1])

    fig.suptitle(
        "Pressure-dependent hopping parameters — Hg1212 model\n"
        "[Harrison scaling + 3rd-order BM EOS; parameters: see lattice_bands.py]",
        fontsize=9, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_hopping_params")


def fig_channel_vertex(tbl: dict) -> list[Path]:
    """Three-panel: V_d_eff, decomposition, exchange−hopping."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    P = tbl["P"]

    # Panel 1: full vertex
    axes[0].plot(P, tbl["V_d_eff"],    color="navy",       lw=2.0, label="$V_d^{\\rm eff}$")
    axes[0].plot(P, tbl["lambda_hop"], color="steelblue",  lw=1.5, ls="--", label="hopping only")
    axes[0].plot(P, tbl["lambda_exch"],color="firebrick",  lw=1.5, ls=":",  label="exchange")
    axes[0].set_ylabel("$\\lambda_d$ (dimensionless)")
    axes[0].set_title("$d$-wave pairing vertex")
    axes[0].legend()

    # Panel 2: stacked area for decomposition
    axes[1].fill_between(P, 0, tbl["lambda_hop"],
                         alpha=0.55, color="steelblue", label="hopping")
    axes[1].fill_between(P, tbl["lambda_hop"], tbl["V_d_eff"],
                         alpha=0.55, color="firebrick", label="exchange")
    axes[1].set_ylabel("$\\lambda_d$ (dimensionless)")
    axes[1].set_title("Channel decomposition")
    axes[1].legend()

    # Panel 3: exchange − hopping
    diff = tbl["exch_minus_hop"]
    color_pos = "firebrick"
    color_neg = "steelblue"
    axes[2].bar(P, diff,
                width=(P[1]-P[0])*0.9,
                color=np.where(diff >= 0, color_pos, color_neg),
                alpha=0.7)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("$\\lambda_{\\rm exch} - \\lambda_{\\rm hop}$")
    axes[2].set_title("Exchange minus hopping")

    for ax in axes:
        ax.set_xlabel("$P$ [GPa]")
        ax.set_xlim(P[0], P[-1])

    fig.suptitle(
        "Effective $d$-wave channel vertex vs pressure\n"
        "[ASSUMED coupling: α_exch = 1; see channels.py]",
        fontsize=9, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_channel_vertex")


def fig_coherence(tbl: dict) -> list[Path]:
    """Two-panel: C_coh dome and F_comp(P)."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    P = tbl["P"]

    ax1.plot(P, tbl["C_coh"], color="purple", lw=2.0)
    ax1.axhline(0.05, color="gray", lw=0.8, ls="--", label="floor $C_{\\rm floor}=0.05$")
    ax1.axvline(P_OPT_GPA, color="purple", lw=0.8, ls=":",
                label=f"$P_{{\\rm opt}}={P_OPT_GPA}$ GPa")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel("$C_{\\rm coh}(P)$")
    ax1.set_title("Coherence dome")
    ax1.legend(fontsize=8)

    ax2.plot(P, tbl["F_comp"], color="seagreen", lw=2.0)
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel("$F_{\\rm comp}(P) = V(P)/V_0$")
    ax2.set_title("Compressibility factor (BM EOS)")
    ax2.set_ylim(0.65, 1.02)

    fig.suptitle(
        "Coherence dome and compressibility — [ASSUMED] dome shape;\n"
        "calibrate $P_{\\rm opt}$, $\\sigma_P$ to experimental $T_c(P)$",
        fontsize=9, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_coherence")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(tbl: dict, n_rows: int = 13) -> None:
    col_trat = "t'/t"
    hdr = (
        f"{'P[GPa]':>8} {'t[eV]':>8} {col_trat:>7} "
        f"{'t_perp/meV':>10} {'J_perp/meV':>10} "
        f"{'l_hop':>8} {'l_exch':>8} {'V_d_eff':>8} "
        f"{'dl':>8} {'C_coh':>7} {'F_comp':>7}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    indices = np.round(np.linspace(0, len(tbl["P"]) - 1, n_rows)).astype(int)
    for i in indices:
        P    = tbl["P"][i]
        t    = tbl["t"][i]
        rr   = tbl["tprime"][i] / tbl["t"][i]
        tpx  = tbl["t_perp"][i] * 1e3
        Jp   = tbl["J_perp"][i] * 1e3
        lh   = tbl["lambda_hop"][i]
        le   = tbl["lambda_exch"][i]
        vd   = tbl["V_d_eff"][i]
        diff = tbl["exch_minus_hop"][i]
        cc   = tbl["C_coh"][i]
        fc   = tbl["F_comp"][i]
        print(
            f"{P:8.1f} {t:8.4f} {rr:7.4f} "
            f"{tpx:9.3f} {Jp:8.4f} "
            f"{lh:8.4f} {le:8.4f} {vd:8.4f} "
            f"{diff:+8.4f} {cc:7.4f} {fc:7.4f}"
        )
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nk",      type=int,  default=128, help="k-grid size per axis")
    parser.add_argument("--no-plot", action="store_true",    help="skip figure generation")
    args = parser.parse_args()

    print(f"\nComputing channel table  (Nk={args.Nk}×{args.Nk}, {len(P_GRID_GPA)} pressures)…")
    tbl = channel_table(P_GRID_GPA, Nk=args.Nk)
    print("  done.")

    # CSV
    csv_path = OUTPUT_DIR / "channels_table.csv"
    export_csv(tbl, csv_path)

    # Figures
    if not args.no_plot:
        print("  Generating figures…")
        for p in fig_hopping_params(tbl):
            print(f"  FIG  → {p}")
        for p in fig_channel_vertex(tbl):
            print(f"  FIG  → {p}")
        for p in fig_coherence(tbl):
            print(f"  FIG  → {p}")

    # Summary table
    print()
    print_summary(tbl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
