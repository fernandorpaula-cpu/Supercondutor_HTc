"""
run_null_models.py — Fit null models to Tc_zero(P) and Wtr(P).

Usage:
    python run_null_models.py [--no-plot]

Outputs (in outputs/):
    null_models_residuals.csv   — residuals per pressure for each model/observable
    fig_null_descriptors        — data + all three fits
    fig_null_residuals          — residual bar charts
    fig_null_bootstrap          — bootstrap 95 % CI bands

RULE: qualidade de interpolação não implica mecanismo físico
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
from config import OUTPUT_DIR, FIGURES, NULL_MODELS
from src.null_models import (
    run_null_analysis, NullModelAnalysis, RULE, DEGENERACY_THRESHOLD,
    model_linear, model_quadratic, model_saturating,
)
from src.two_scale import HG1212_DATA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI   = FIGURES["dpi"]
FMTS  = FIGURES["format"]
STYLE = "seaborn-v0_8-whitegrid"

MODEL_COLORS = {
    "linear":     "steelblue",
    "quadratic":  "darkorange",
    "saturating": "seagreen",
}
MODEL_LS = {
    "linear":     "-",
    "quadratic":  "--",
    "saturating": "-.",
}


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
    plt.rcParams["legend.fontsize"] = 8


# ── CSV export ────────────────────────────────────────────────────────────────

def export_residuals_csv(results: dict, path: Path) -> None:
    P = HG1212_DATA["P_GPa"]
    rows = []
    header = ["P_GPa"]
    for obs_name, ana in results.items():
        for mname, fr in ana.fits.items():
            col = f"{obs_name}_{mname}_resid_K"
            header.append(col)
            for i, r in enumerate(fr.residuals):
                if len(rows) <= i:
                    rows.append({"P_GPa": f"{P[i]:.1f}"})
                rows[i][col] = f"{r:.4f}"

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  → {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_descriptors(results: dict) -> list[Path]:
    """Data + all three fits for each observable."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    obs_labels = {
        "Tc_zero": "$T_{c,\\rm zero}(P)$ [K]",
        "Wtr":     "$W_{\\rm tr}(P)$ [K]",
    }
    titles = {
        "Tc_zero": "Null models: $T_{c,\\rm zero}$ vs pressure",
        "Wtr":     "Null models: transition width $W_{\\rm tr}$ vs pressure",
    }

    P_fine = np.linspace(0, 30, 300)
    model_fns = {
        "linear": model_linear,
        "quadratic": model_quadratic,
        "saturating": model_saturating,
    }

    for ax, (obs_name, ana) in zip(axes, results.items()):
        ax.scatter(ana.P, ana.y_obs, color="black", zorder=5,
                   s=40, label="Hg1212 data [APPROX]")
        for mname, fr in ana.fits.items():
            deg_tag = " [DEGEN]" if fr.degenerate else ""
            lbl = (f"{mname}  RMSE={fr.rmse:.2f} K  ρ={fr.rho:.0f}"
                   f"{deg_tag}")
            y_fine = model_fns[mname](P_fine, *fr.popt)
            ax.plot(P_fine, y_fine,
                    color=MODEL_COLORS[mname],
                    ls=MODEL_LS[mname],
                    lw=1.8, label=lbl)
        ax.set_xlabel("$P$ [GPa]")
        ax.set_ylabel(obs_labels[obs_name])
        ax.set_title(titles[obs_name], fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 30)

    fig.suptitle(
        f"RULE: {RULE}",
        fontsize=8, style="italic", y=1.01,
    )
    fig.tight_layout()
    return _save(fig, "fig_null_descriptors")


def fig_residuals(results: dict) -> list[Path]:
    """Grouped bar chart of residuals per pressure for each model."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_names = ["linear", "quadratic", "saturating"]

    for ax, (obs_name, ana) in zip(axes, results.items()):
        P = ana.P
        n_models = len(model_names)
        width = 0.25
        offsets = np.linspace(-(n_models - 1) * width / 2,
                              (n_models - 1) * width / 2,
                              n_models)
        for offset, mname in zip(offsets, model_names):
            fr = ana.fits[mname]
            deg_tag = " [DEGEN]" if fr.degenerate else ""
            ax.bar(P + offset, fr.residuals, width=width,
                   color=MODEL_COLORS[mname], alpha=0.75,
                   label=f"{mname}{deg_tag}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("$P$ [GPa]")
        ax.set_ylabel(f"Residual [K]  (obs − model)")
        ax.set_title(f"Residuals: {obs_name}", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(f"RULE: {RULE}", fontsize=8, style="italic", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig_null_residuals")


def fig_bootstrap(results: dict) -> list[Path]:
    """Bootstrap 95 % CI bands for each model."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    model_names = ["linear", "quadratic", "saturating"]
    obs_list = list(results.items())

    for row, (obs_name, ana) in enumerate(obs_list):
        for col, mname in enumerate(model_names):
            ax = axes[row][col]
            br = ana.boots[mname]
            fr = ana.fits[mname]

            ax.scatter(ana.P, ana.y_obs, color="black", s=30,
                       zorder=5, label="data [APPROX]")
            ax.plot(br.P_fine, model_quadratic(br.P_fine, *fr.popt)
                    if mname == "quadratic"
                    else model_linear(br.P_fine, *fr.popt)
                    if mname == "linear"
                    else model_saturating(br.P_fine, *fr.popt),
                    color=MODEL_COLORS[mname], lw=2.0, label="best fit")
            ax.fill_between(br.P_fine, br.ci_lo, br.ci_hi,
                            alpha=0.25, color=MODEL_COLORS[mname],
                            label=f"95% CI  (n={br.n_boot})")

            deg_tag = "  [DEGENERATE]" if fr.degenerate else ""
            ax.set_title(
                f"{obs_name} — {mname}{deg_tag}\n"
                f"RMSE={fr.rmse:.2f} K  ρ={fr.rho:.0f}",
                fontsize=8,
            )
            ax.set_xlabel("$P$ [GPa]")
            ax.set_ylabel("[K]")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 30)

    fig.suptitle(
        f"Bootstrap 95 % CI  (σ_T = 1.5 K)\nRULE: {RULE}",
        fontsize=9, style="italic",
    )
    fig.tight_layout()
    return _save(fig, "fig_null_bootstrap")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    sep = "─" * 75
    print(sep)
    print(f"  NULL MODEL SUMMARY")
    print(f"  {RULE}")
    print(sep)

    hdr = f"{'Observable':<12} {'Model':<12} {'RMSE[K]':>8} {'MAE[K]':>8} "
    hdr += f"{'rho':>10} {'best?':>6} {'degen?':>8}"
    print(hdr)
    print(sep)

    for obs_name, ana in results.items():
        for mname, fr in ana.fits.items():
            is_best = "✓" if mname == ana.best_model else ""
            degen = "[DEGEN]" if fr.degenerate else "no"
            print(
                f"  {obs_name:<10} {mname:<12} {fr.rmse:8.3f} {fr.mae:8.3f}"
                f" {fr.rho:10.1f} {is_best:>6} {degen:>8}"
            )
        print()

    print(sep)
    print("  Residuals (obs − model) per pressure [K]:")
    print()
    P = HG1212_DATA["P_GPa"]
    for obs_name, ana in results.items():
        print(f"  {obs_name}:")
        hdr2 = f"  {'P[GPa]':>8}"
        for mname in ("linear", "quadratic", "saturating"):
            hdr2 += f" {mname:>12}"
        print(hdr2)
        for i, p in enumerate(P):
            row = f"  {p:8.1f}"
            for mname in ("linear", "quadratic", "saturating"):
                row += f" {ana.fits[mname].residuals[i]:12.4f}"
            print(row)
        print()

    print(sep)
    print(f"\n  RULE: {RULE}\n")
    print("  Interpretation constraints:")
    print("  1. Best null RMSE does NOT select a physical mechanism.")
    print("  2. Degenerate saturating fit on Wtr ≈ linear — no saturation physics inferred.")
    print("  3. All models here are interpolation benchmarks, not physical predictions.")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n_boot = NULL_MODELS["n_random"]
    seed   = NULL_MODELS["seed"]

    print(f"\nNull model analysis  (n_boot={n_boot}, seed={seed}, σ_T={1.5} K)")
    print(f"  RULE: {RULE}\n")

    results = run_null_analysis(
        data=HG1212_DATA,
        n_boot=n_boot,
        seed=seed,
    )

    # CSV
    export_residuals_csv(results, OUTPUT_DIR / "null_models_residuals.csv")

    # Figures
    if not args.no_plot:
        print("\n  Generating figures…")
        for p in fig_descriptors(results):
            print(f"  FIG  → {p}")
        for p in fig_residuals(results):
            print(f"  FIG  → {p}")
        for p in fig_bootstrap(results):
            print(f"  FIG  → {p}")

    print()
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
