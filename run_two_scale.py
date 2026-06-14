"""
run_two_scale.py — Two-scale calibration and diagnostics for Hg1212.

Usage:
    python run_two_scale.py [--no-plot] [--P-threshold GPa]

Outputs (in outputs/):
    two_scale_table.csv   — Tc_onset, C_coh, Tc_zero, Wtr vs P (model + data)
    fig_two_scale         — main comparison: data vs model
    fig_Wtr               — transition width Wtr(P)
    fig_Ccoh              — coherence factor C_coh(P)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, FIGURES
from src.two_scale import (
    HG1212_DATA, HG1223_DATA,
    calibrate_model, two_scale_table,
    wtr_data, kappa_data,
    Tc_onset_model, C_coh_model, Tc_zero_model_fn,
)
from src.lattice_bands import P_GRID_GPA
from src.pairing_bdg import bdg_pressure_scan, V_D_CALIB

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STYLE = "seaborn-v0_8-whitegrid"
DPI   = FIGURES["dpi"]
FMTS  = FIGURES["format"]
FINE_P = np.linspace(0, 30, 301)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for fmt in FMTS:
        p = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _style():
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 8


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(tbl: dict, data: dict, cal, path: Path) -> None:
    P_data    = data["P_GPa"]
    Tc_on_d   = data["Tc_onset_K"]
    Tc_z_d    = data["Tc_zero_K"]
    Wtr_d     = wtr_data(data)
    kappa_d   = kappa_data(data)

    # Model at data pressures for residuals
    Tc_on_m = Tc_onset_model(P_data, cal.coeffs_onset)
    Tc_z_m  = Tc_zero_model_fn(P_data, cal.coeffs_onset, cal.coeffs_coh)
    C_coh_m = C_coh_model(P_data, cal.coeffs_coh)
    Wtr_m   = Tc_on_m - Tc_z_m

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "P_GPa",
            "Tc_onset_exp_K", "Tc_zero_exp_K", "Wtr_exp_K", "kappa_exp",
            "Tc_onset_model_K", "Tc_zero_model_K", "Wtr_model_K", "C_coh_model",
            "res_Tc_zero_K",
        ])
        for i in range(len(P_data)):
            w.writerow([
                f"{P_data[i]:.1f}",
                f"{Tc_on_d[i]:.2f}", f"{Tc_z_d[i]:.2f}",
                f"{Wtr_d[i]:.2f}", f"{kappa_d[i]:.5f}",
                f"{Tc_on_m[i]:.3f}", f"{Tc_z_m[i]:.3f}",
                f"{Wtr_m[i]:.3f}", f"{C_coh_m[i]:.6f}",
                f"{Tc_z_m[i]-Tc_z_d[i]:.3f}",
            ])
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_two_scale(tbl: dict, data: dict, cal, P_threshold: float) -> list[Path]:
    """Main figure: experimental data vs two-scale model."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    P_fine = tbl["P"]
    P_d    = data["P_GPa"]

    # ── Panel 1: Tc_onset and Tc_zero ─────────────────────────────────────
    ax = axes[0]
    # Shaded band = transition region
    ax.fill_between(
        P_fine, tbl["Tc_zero_model"], tbl["Tc_onset_model"],
        alpha=0.18, color="steelblue", label="$W_{\\rm tr}$ band (model)",
    )
    # Model curves
    ax.plot(P_fine, tbl["Tc_onset_model"], color="firebrick", lw=2.0,
            label="$T_{c,{\\rm onset}}^{\\rm model}$")
    ax.plot(P_fine, tbl["Tc_zero_model"],  color="steelblue", lw=2.0,
            label="$T_{c,{\\rm zero}}^{\\rm model}$")

    # Data points
    ax.scatter(P_d, data["Tc_onset_K"], marker="^", s=55, color="firebrick",
               zorder=5, label="$T_{c,{\\rm onset}}$ [data, ≈LIT]")
    ax.scatter(P_d, data["Tc_zero_K"],  marker="o", s=55, color="steelblue",
               zorder=5, label="$T_{c,{\\rm zero}}$ [data, ≈LIT]")

    # BdG pairing proxy
    if "Delta_d_meV" in tbl and "Tc_MF_K" in tbl:
        mask_valid = tbl["Tc_MF_K"] > 5
        ax.plot(P_fine[mask_valid], tbl["Tc_MF_K"][mask_valid],
                color="gray", lw=1.2, ls="--", alpha=0.7,
                label="$T_c^{\\rm MF}$ (BdG — pair formation proxy)")

    # P_threshold marker
    ax.axvline(P_threshold, color="purple", lw=0.8, ls=":",
               label=f"$P_{{\\rm thresh}}={P_threshold:.0f}$ GPa")

    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(
        "Two-scale model: $T_{c,{\\rm onset}}$ vs $T_{c,{\\rm zero}}$\n"
        "[DATA: APPROXIMATE ≈LIT; see HG1212_DATA status]",
        fontsize=8,
    )
    ax.legend(loc="lower left", fontsize=7)
    ax.set_xlim(0, 30)

    # ── Panel 2: Residuals ────────────────────────────────────────────────
    ax2 = axes[1]
    Tc_z_model_at_data = Tc_zero_model_fn(P_d, cal.coeffs_onset, cal.coeffs_coh)
    residuals = Tc_z_model_at_data - data["Tc_zero_K"]
    colors = ["firebrick" if r > 0 else "steelblue" for r in residuals]
    ax2.bar(P_d, residuals, width=2.0, color=colors, alpha=0.75)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(P_threshold, color="purple", lw=0.8, ls=":")

    # RMSE annotations
    mask_high = P_d >= P_threshold
    rmse_all  = cal.rmse_all
    rmse_high = cal.rmse_high_P
    ax2.text(0.05, 0.94, f"RMSE (all P) = {rmse_all:.2f} K",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax2.text(0.05, 0.83, f"RMSE (P≥{P_threshold:.0f} GPa) = {rmse_high:.2f} K",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax2.set_xlabel("$P$ [GPa]")
    ax2.set_ylabel("$T_{c,{\\rm zero}}^{\\rm model} - T_{c,{\\rm zero}}^{\\rm data}$ [K]")
    ax2.set_title("Residuals: model − data  ($T_{c,{\\rm zero}}$)", fontsize=9)
    ax2.set_xlim(0, 30)

    fig.suptitle(
        "Hg1212 two-scale diagnostic  —  "
        "$T_{c,{\\rm zero}} = C_{\\rm coh}(P) \\times T_{c,{\\rm onset}}(P)$\n"
        r"\textbf{APPROXIMATE DATA: not for quantitative conclusions}",
        fontsize=9, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_two_scale")


def fig_Wtr(tbl: dict, data: dict) -> list[Path]:
    """Transition width Wtr(P): model vs data."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    P_fine = tbl["P"]
    P_d    = data["P_GPa"]
    Wtr_d  = wtr_data(data)

    ax.plot(P_fine, tbl["Wtr_model"], color="purple", lw=2.0,
            label="$W_{\\rm tr}^{\\rm model}(P)$")
    ax.scatter(P_d, Wtr_d, marker="D", s=50, color="purple", zorder=5,
               label="$W_{\\rm tr}^{\\rm data}$ [≈LIT]")

    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("$W_{\\rm tr} = T_{c,{\\rm onset}} - T_{c,{\\rm zero}}$ [K]")
    ax.set_title(
        "Transition width $W_{\\rm tr}(P)$\n"
        "Wtr measures decoherence: NOT a sample-quality artefact alone",
        fontsize=9,
    )
    ax.legend()
    ax.set_xlim(0, 30)
    fig.tight_layout()
    return _save(fig, "fig_Wtr")


def fig_Ccoh(tbl: dict, data: dict, cal) -> list[Path]:
    """Coherence factor C_coh(P): model calibration and data points."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    P_fine = tbl["P"]
    P_d    = data["P_GPa"]
    kd     = kappa_data(data)

    ax.plot(P_fine, tbl["C_coh"], color="seagreen", lw=2.0,
            label=f"$C_{{\\rm coh}}(P)$ — poly fit (deg {cal.deg_coh})")
    ax.scatter(P_d, kd, marker="s", s=50, color="seagreen", zorder=5,
               label="$\\kappa_{\\rm data} = T_{{c,{\\rm zero}}} / T_{{c,{\\rm onset}}}$ [≈LIT]")
    ax.axhline(1.0, color="gray", lw=0.7, ls="--", label="$C_{\\rm coh}=1$ (BCS limit)")
    ax.axhline(kd[0], color="k", lw=0.5, ls=":", alpha=0.5,
               label=f"$\\kappa$(P=0) = {kd[0]:.4f}")

    ax.set_xlabel("$P$ [GPa]")
    ax.set_ylabel("$C_{\\rm coh}(P) = T_{c,{\\rm zero}} / T_{c,{\\rm onset}}$")
    ax.set_title(
        "Global coherence factor $C_{\\rm coh}(P)$\n"
        r"$C_{\rm coh}\to 1$: BCS limit;  $C_{\rm coh}\ll 1$: incoherent pairs",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.85, 1.02)
    fig.tight_layout()
    return _save(fig, "fig_Ccoh")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(tbl: dict, data: dict, cal, P_threshold: float) -> None:
    P_d   = data["P_GPa"]
    Ton_d = data["Tc_onset_K"]
    Tz_d  = data["Tc_zero_K"]
    Wtr_d = wtr_data(data)
    kd    = kappa_data(data)

    Ton_m = Tc_onset_model(P_d, cal.coeffs_onset)
    Tz_m  = Tc_zero_model_fn(P_d, cal.coeffs_onset, cal.coeffs_coh)
    Wtr_m = Ton_m - Tz_m
    Cc_m  = C_coh_model(P_d, cal.coeffs_coh)
    res   = Tz_m - Tz_d

    col = "{:>7} {:>9} {:>9} {:>6} {:>7} | {:>9} {:>9} {:>7} {:>8} {:>7}"
    row = "{:7.1f} {:9.2f} {:9.2f} {:6.2f} {:7.5f} | {:9.3f} {:9.3f} {:7.3f} {:8.5f} {:+7.3f}"
    header = col.format(
        "P[GPa]",
        "Ton_exp", "Tz_exp", "Wtr_d", "κ_data",
        "Ton_mdl", "Tz_mdl", "Wtr_m", "C_coh_m", "res[K]",
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i in range(len(P_d)):
        mark = " ◄" if P_d[i] >= P_threshold else ""
        print(row.format(
            P_d[i],
            Ton_d[i], Tz_d[i], Wtr_d[i], kd[i],
            Ton_m[i], Tz_m[i], Wtr_m[i], Cc_m[i], res[i],
        ) + mark)
    print(sep)

    print(f"\n  Calibration target: Hg1212 — {data['label']}")
    print(f"  Data status: {data['status']}")
    print(f"\n  RMSE Tc_onset  (model vs data):  {cal.rmse_onset:.3f} K")
    print(f"  RMSE Tc_zero   (all P):           {cal.rmse_all:.3f} K")
    print(f"  RMSE Tc_zero   (P ≥ {P_threshold:.0f} GPa):    {cal.rmse_high_P:.3f} K  ◄")
    print()
    print("  Interpretation (mandatory):")
    print("  · Tc_onset: tracks local pairing — phenomenological fit, NOT BdG Tc_MF")
    print("  · Tc_zero:  requires global coherence; = C_coh × Tc_onset")
    print("  · Wtr:      broadening from phase fluctuations / decoherence")
    print("  · C_coh:    calibrated to κ_data; DISTINCT from channels.py C_coh dome")
    print("  · This model is a DIAGNOSTIC DECOMPOSITION, not a theory of Tc(P)")
    print()
    # BdG comparison
    if "Tc_MF_K" in tbl:
        print("  BdG note: Tc_MF(P) from BdG drops with pressure (FS topology artefact)")
        print("  BdG Tc_MF does NOT reproduce Tc_onset(P). Δ_d(P) is a local pairing proxy.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--P-threshold", type=float, default=12.0,
                        help="P cutoff for high-P RMSE [GPa]")
    parser.add_argument("--Nk-bdg",     type=int,   default=64,
                        help="k-grid for BdG pairing proxy (coarse OK)")
    args = parser.parse_args()

    print("\n── Two-scale model: Hg1212 calibration ──────────────────────────")
    print(f"  Data status: {HG1212_DATA['status']}\n")

    # 1. Calibrate model
    cal = calibrate_model(HG1212_DATA, P_threshold=args.P_threshold)
    print(f"  Tc_onset poly (deg {cal.deg_onset}): {cal.coeffs_onset}")
    print(f"  C_coh    poly (deg {cal.deg_coh}):   {cal.coeffs_coh}")
    print(f"  RMSE Tc_zero (all P):       {cal.rmse_all:.3f} K")
    print(f"  RMSE Tc_zero (P≥{args.P_threshold:.0f} GPa):  {cal.rmse_high_P:.3f} K")

    # 2. BdG pairing proxy on fine grid (optional context — Tc_MF collapses
    #    in the intermediate pressure range due to FS topology artefact)
    print(f"\n  Loading BdG pairing proxy (Nk={args.Nk_bdg}²)…")
    print("  (Tc_MF bracket warnings below are expected — BdG Tc_MF drops to near-zero")
    print("   at intermediate P due to FS topology change; see docstring for details)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Tc_MF not bracketed")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="invalid value encountered")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Tc_MF not bracketed")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="invalid value encountered")
        bdg = bdg_pressure_scan(
            P_grid=FINE_P, Nk_gap=args.Nk_bdg, Nk_dos=args.Nk_bdg,
            P_dos_select=[], V_d_calib=V_D_CALIB,
        )

    # 3. Full table on fine grid
    tbl = two_scale_table(
        FINE_P, cal,
        Delta_d_meV=bdg["Delta_d_meV"],
        Tc_MF_K=bdg["Tc_MF_K"],
    )

    # 4. CSV at data pressures + fine grid
    export_csv(tbl, HG1212_DATA, cal, OUTPUT_DIR / "two_scale_table.csv")

    # 5. Figures
    if not args.no_plot:
        print("\n  Generating figures…")
        for p in fig_two_scale(tbl, HG1212_DATA, cal, args.P_threshold):
            print(f"  FIG  → {p}")
        for p in fig_Wtr(tbl, HG1212_DATA):
            print(f"  FIG  → {p}")
        for p in fig_Ccoh(tbl, HG1212_DATA, cal):
            print(f"  FIG  → {p}")

    # 6. Summary
    print()
    print_summary(tbl, HG1212_DATA, cal, args.P_threshold)

    # 7. Warn about Hg1223
    print(f"\n  Hg1223 status: {HG1223_DATA['status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
