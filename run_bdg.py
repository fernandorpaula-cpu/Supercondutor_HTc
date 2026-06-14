"""
run_bdg.py — Compute BdG d-wave gap, Tc_MF, and DOS vs pressure.

Usage:
    python run_bdg.py [--Nk-gap N] [--Nk-dos N] [--no-plot]

Outputs (in outputs/):
    bdg_table.csv          — Δ_d(P), Tc_MF(P), ratio, V_d(P) for all pressures
    fig_gap_vs_P           — Δ_d(P) in meV
    fig_Tc_vs_P            — Tc_MF(P) and 2Δ/kBTc ratio
    fig_dos_bdg            — BdG DOS N(E) at selected pressures
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
from src.pairing_bdg import (
    K_B, V_D_CALIB, NK_GAP, NK_DOS, bdg_pressure_scan,
)
from src.lattice_bands import P_GRID_GPA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STYLE = "seaborn-v0_8-whitegrid"
DPI   = FIGURES["dpi"]
FMTS  = FIGURES["format"]

DOS_PRESSURES = [0.0, 10.0, 20.0, 30.0]
DOS_COLORS    = ["steelblue", "darkorange", "seagreen", "firebrick"]


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

def export_csv(res: dict, path: Path) -> None:
    cols = [
        "P_GPa", "V_d_eV", "V_d_eff_ratio",
        "Delta_d_meV", "Tc_MF_K", "ratio_2DkT",
    ]
    keys = ["P", "V_d_P", "V_d_eff_ratio", "Delta_d_meV", "Tc_MF_K", "ratio_2DkT"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(res["P"])):
            w.writerow([f"{res[k][i]:.6g}" for k in keys])
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_gap_vs_P(res: dict) -> list[Path]:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    P = res["P"]
    D = res["Delta_d_meV"]

    ax.plot(P, D, color="navy", lw=2.2, label="$\\Delta_d(P)$")
    ax.fill_between(P, 25, 40, alpha=0.12, color="navy",
                    label="target band [25–40 meV]")
    ax.axhline(D[0], color="gray", lw=0.8, ls="--",
               label=f"$\\Delta_d(0)={D[0]:.1f}$ meV")

    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("$\\Delta_d(P)$ [meV]  —  local pairing proxy")
    ax.set_title(
        "BdG d-wave pairing amplitude vs pressure\n"
        "[LABEL: $\\Delta_d$ is a LOCAL PAIRING PROXY, not $T_{c,\\rm zero}$]",
        fontsize=9,
    )
    ax.legend()
    ax.set_xlim(P[0], P[-1])
    fig.tight_layout()
    return _save(fig, "fig_gap_vs_P")


def fig_Tc_vs_P(res: dict) -> list[Path]:
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    P = res["P"]

    # Panel 1: Tc_MF
    ax1.plot(P, res["Tc_MF_K"], color="firebrick", lw=2.0,
             label="$T_c^{\\rm MF}$ (pair-formation onset)")
    ax1.axhline(126.0, color="gray", lw=1.0, ls="--",
                label="Hg1212 $T_{c,\\rm onset}$(P=0) = 126 K  [expt.]")
    ax1.set_xlabel("$P$ [GPa]")
    ax1.set_ylabel("$T_c^{\\rm MF}(P)$ [K]")
    ax1.set_title(
        "Mean-field onset temperature\n"
        "[$T_c^{\\rm MF}$: pair-formation scale; includes no phase-fluctuation suppression]",
        fontsize=8,
    )
    ax1.legend(fontsize=8)
    ax1.set_xlim(P[0], P[-1])

    # Panel 2: ratio 2Δ/kBTc
    ratio = res["ratio_2DkT"]
    ax2.plot(P, ratio, color="purple", lw=2.0)
    ax2.axhline(4.28, color="gray", lw=0.8, ls="--",
                label="weak-coupling d-wave BCS ≈ 4.28")
    ax2.axhline(ratio[0], color="purple", lw=0.8, ls=":",
                label=f"P=0 value: {ratio[0]:.2f}")
    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel("$2\\Delta_d / k_B T_c^{\\rm MF}$")
    ax2.set_title(
        "Pairing ratio vs pressure\n"
        "[Enhanced above weak-coupling limit: near-VH FS topology with $|t\'/t|=0.40$]",
        fontsize=8,
    )
    ax2.legend(fontsize=8)
    ax2.set_xlim(P[0], P[-1])

    fig.tight_layout()
    return _save(fig, "fig_Tc_vs_P")


def fig_dos_bdg(res: dict) -> list[Path]:
    _style()
    dos_dict = res["dos_P"]
    if not dos_dict:
        print("  No DOS data — skipping fig_dos_bdg.")
        return []

    fig, ax = plt.subplots(figsize=(7, 5))

    P_keys = sorted(dos_dict.keys())
    for P_val, color in zip(P_keys, DOS_COLORS):
        E_grid, N_E = dos_dict[P_val]
        D_meV = res["Delta_d_meV"][np.argmin(np.abs(res["P"] - P_val))]
        Tc_K  = res["Tc_MF_K"][np.argmin(np.abs(res["P"] - P_val))]
        label = f"$P={P_val:.0f}$ GPa  ($\\Delta_d={D_meV:.1f}$ meV, $T_c^{{\\rm MF}}={Tc_K:.0f}$ K)"
        # Normalise to N(0_normal) — pick N at |E| >> Delta_d as reference
        N_ref = N_E[np.argmin(np.abs(E_grid - 4 * D_meV / 1e3))]
        ax.plot(E_grid * 1e3, N_E / max(N_ref, 1e-10),
                color=color, lw=1.5, label=label)

    # Mark coherence peak positions at P=0
    D0 = res["Delta_d_meV"][0]
    ax.axvline( D0, color="k", lw=0.7, ls="--", alpha=0.5, label=f"$\\pm\\Delta_d(0)={D0:.1f}$ meV")
    ax.axvline(-D0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.3)

    ax.set_xlabel("$E$ [meV]")
    ax.set_ylabel("$N(E) / N_{\\rm ref}$")
    ax.set_title(
        "BdG $d$-wave DOS at selected pressures\n"
        "V-shape (nodal QP) + coherence peaks at $E = \\pm\\Delta_d$\n"
        "[$\\Delta_d$: local pairing proxy — NOT $T_{c,\\rm zero}$;  "
        "broadening $\\eta = 3$ meV]",
        fontsize=8,
    )
    ax.legend(fontsize=7)
    ax.set_xlim(-4 * D0, 4 * D0)
    fig.tight_layout()
    return _save(fig, "fig_dos_bdg")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(res: dict, n_rows: int = 13) -> None:
    P = res["P"]
    indices = np.round(np.linspace(0, len(P) - 1, n_rows)).astype(int)

    hdr_fmt = "{:>8} {:>9} {:>12} {:>10} {:>11} {:>9}"
    row_fmt = "{:8.1f} {:9.4f} {:12.3f} {:10.2f} {:11.2f} {:9.4f}"
    header = hdr_fmt.format(
        "P[GPa]", "V_d[eV]", "V_d_eff_ratio",
        "Δd[meV]", "Tc_MF[K]", "2Δ/kBTc",
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i in indices:
        print(row_fmt.format(
            res["P"][i], res["V_d_P"][i], res["V_d_eff_ratio"][i],
            res["Delta_d_meV"][i], res["Tc_MF_K"][i], res["ratio_2DkT"][i],
        ))
    print(sep)

    D0   = res["Delta_d_meV"][0]
    Tc0  = res["Tc_MF_K"][0]
    r0   = res["ratio_2DkT"][0]
    Dmax = res["Delta_d_meV"].max()
    Pmax = res["P"][res["Delta_d_meV"].argmax()]
    print(f"\n  Calibration:  V_d = {V_D_CALIB:.4f} eV → Tc_MF(0) = {Tc0:.1f} K [target: 126 K]")
    print(f"  Δ_d(P=0)    = {D0:.2f} meV  [target: 25–40 meV]  ← LOCAL PAIRING PROXY")
    print(f"  2Δ/kBTc(0)  = {r0:.3f}        [weak-coupling d-wave BCS: 4.28]")
    print(f"  Δ_d peak    = {Dmax:.2f} meV at P = {Pmax:.1f} GPa")
    print(f"\n  NOTE: Δ_d is NOT Tc_zero.  Tc_MF is the BdG pair-formation scale (T*).")
    print(f"        Tc_zero requires phase-coherence physics beyond mean-field BdG.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nk-gap", type=int, default=NK_GAP)
    parser.add_argument("--Nk-dos", type=int, default=NK_DOS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"\nBdG gap scan  (Nk_gap={args.Nk_gap}², Nk_dos={args.Nk_dos}²,  "
          f"V_d_calib={V_D_CALIB:.4f} eV)")
    print(f"  Target: Tc_MF(0) = 126 K,  Δ_d(0) ∈ [25, 40] meV\n")

    res = bdg_pressure_scan(
        P_grid=P_GRID_GPA,
        Nk_gap=args.Nk_gap,
        Nk_dos=args.Nk_dos,
        P_dos_select=DOS_PRESSURES,
        V_d_calib=V_D_CALIB,
    )

    # CSV
    export_csv(res, OUTPUT_DIR / "bdg_table.csv")

    # Figures
    if not args.no_plot:
        print("\n  Generating figures…")
        for p in fig_gap_vs_P(res):
            print(f"  FIG  → {p}")
        for p in fig_Tc_vs_P(res):
            print(f"  FIG  → {p}")
        for p in fig_dos_bdg(res):
            print(f"  FIG  → {p}")

    # Summary
    print()
    print_summary(res)

    return 0


if __name__ == "__main__":
    sys.exit(main())
