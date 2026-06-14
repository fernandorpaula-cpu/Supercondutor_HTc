"""
figures.py — Centralised figure library for the cuprate pressure study.

Every public function:
  - Receives pre-computed result objects / arrays.
  - Saves PNG (300 dpi) + PDF (vector) via _save().
  - Returns list[Path] of written files.

Figures produced:
  F01  band_structure      — ε(k) along Γ-X-M-Γ + DOS side panel
  F02  fermi_surface       — FS contours at P = 0, 10, 20, 30 GPa
  F03  bdg_gap             — Δ_d(P) and Tc_MF(P) vs pressure
  F04  bdg_dos             — BdG DOS N(E) at selected pressures
  F05  channels            — V_d_eff, λ_hop, λ_exch vs pressure
  F06  coherence           — C_coh(P) dome + F_comp(P)
  F07  two_scale           — Tc_onset, Tc_zero data+model vs P
  F08  null_bootstrap      — Null-model bootstrap CI for Tc_zero and Wtr
  F09  rpa_chi             — χ_RPA(q) map + Stoner criterion
  F10  rpa_channels        — λ_d and λ_s vs P
  F11  correlation_hf      — Δ_HF(P), m(P), Z_BR(P)
  F12  superexchange       — J_Hub(P) and J_Emery(P) with over-prediction flag
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

from config import OUTPUT_DIR

DPI: int = 300
FMTS: tuple[str, ...] = ("png", "pdf")
STYLE: str = "seaborn-v0_8-whitegrid"

_LABEL_BDG = "Δ_d = local pairing proxy (BdG); NOT Tc_zero"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _style() -> None:
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": DPI,
    })


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in FMTS:
        p = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# F01  Band structure along Γ-X-M-Γ + DOS
# ---------------------------------------------------------------------------

def figure_1_band_structure(
    kpath: NDArray,
    energies: NDArray,
    dos_e: NDArray,
    dos_n: NDArray,
    labels: list[str],
) -> list[Path]:
    """Band structure ε(k) along high-symmetry path with DOS side panel."""
    _style()
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_b = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1], sharey=ax_b)

    ax_b.plot(kpath, energies, color="steelblue", lw=0.8)
    ax_b.axhline(0, color="k", lw=0.8, ls="--", label="$E_F$")
    if labels:
        n_seg = len(labels) - 1
        ticks = np.linspace(kpath[0], kpath[-1], len(labels))
        ax_b.set_xticks(ticks)
        ax_b.set_xticklabels(labels)
    ax_b.set_ylabel(r"$\varepsilon(\mathbf{k})$ [eV]")
    ax_b.set_xlabel("k-path")
    ax_b.set_title("Band structure (tight-binding)")
    ax_b.legend(fontsize=8)

    ax_d.plot(dos_n, dos_e, color="darkorange", lw=1.2)
    ax_d.axhline(0, color="k", lw=0.8, ls="--")
    ax_d.set_xlabel("DOS [a.u.]")
    ax_d.tick_params(labelleft=False)
    ax_d.set_title("DOS")

    fig.suptitle("Hg-1212 — square-lattice tight-binding [APPROXIMATE]", fontsize=10)
    return _save(fig, "fig_band_structure")


# ---------------------------------------------------------------------------
# F02  Fermi surface at multiple pressures
# ---------------------------------------------------------------------------

def figure_2_fermi_surface(
    kx: NDArray,
    ky: NDArray,
    eps_k_list: list[NDArray],
    P_labels: list[str],
    colors: list[str] | None = None,
) -> list[Path]:
    """Fermi surface contours for several pressures on one panel."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 5))
    if colors is None:
        colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    for eps_k, label, c in zip(eps_k_list, P_labels, colors):
        ax.contour(kx, ky, eps_k, levels=[0.0], colors=[c], linewidths=1.4)
        ax.plot([], [], color=c, lw=1.4, label=label)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$k_x$ [rad]")
    ax.set_ylabel(r"$k_y$ [rad]")
    ax.set_aspect("equal")
    ax.legend(title="Pressure", loc="upper right")
    ax.set_title("Fermi surface evolution under pressure\n[tight-binding, APPROXIMATE]")
    # Mark high-symmetry points
    for pt, lbl in [((0, 0), r"$\Gamma$"), ((np.pi, 0), "X"), ((np.pi, np.pi), "M")]:
        ax.plot(*pt, "k.", ms=4)
        ax.annotate(lbl, xy=pt, xytext=(pt[0]+0.1, pt[1]+0.1), fontsize=9)
    return _save(fig, "fig_fermi_surface")


# ---------------------------------------------------------------------------
# F03  BdG gap and Tc_MF vs pressure
# ---------------------------------------------------------------------------

def figure_3_bdg_gap(bdg: dict) -> list[Path]:
    """Δ_d(P) and Tc_MF(P) vs pressure, two panels."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    P = bdg["P"]
    ax1.plot(P, bdg["Delta_d_meV"], "o-", color="steelblue", ms=3, lw=1.4,
             label=r"$\Delta_d(P)$")
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel(r"$\Delta_d$ [meV]")
    ax1.set_title(r"d-wave gap $\Delta_d(P)$")
    ax1.text(0.05, 0.05, _LABEL_BDG, transform=ax1.transAxes,
             fontsize=7, color="gray", va="bottom")
    ax1.legend()

    ax2.plot(P, bdg["Tc_MF_K"], "s-", color="firebrick", ms=3, lw=1.4,
             label=r"$T_c^{\rm MF}(P)$")
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel(r"$T_c^{\rm MF}$ [K]")
    ax2.set_title(r"Mean-field $T_c$ (BdG)")
    ax2.text(0.05, 0.05, "Tc_MF != Tc_onset\n(phase fluctuations neglected)",
             transform=ax2.transAxes, fontsize=7, color="gray", va="bottom")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "fig_bdg_gap")


# ---------------------------------------------------------------------------
# F04  BdG DOS at selected pressures
# ---------------------------------------------------------------------------

def figure_4_bdg_dos(bdg: dict) -> list[Path]:
    """BdG DOS N(E) at selected pressures."""
    _style()
    dos_dict: dict = bdg.get("dos_P", {})
    if not dos_dict:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "DOS not computed\n(increase NK_DOS)", ha="center",
                va="center", transform=ax.transAxes)
        return _save(fig, "fig_bdg_dos")

    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (P_val, (E_arr, N_arr)), c in zip(dos_dict.items(), colors):
        ax.plot(E_arr * 1e3, N_arr, lw=1.2, color=c, label=f"P={P_val:.0f} GPa")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Energy [meV]")
    ax.set_ylabel("BdG DOS [a.u.]")
    ax.set_title("BdG density of states — d-wave gap\n" + _LABEL_BDG)
    ax.set_xlim(-60, 60)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig_bdg_dos")


# ---------------------------------------------------------------------------
# F05  Channel vertex vs pressure
# ---------------------------------------------------------------------------

def figure_5_channels(ct: dict) -> list[Path]:
    """V_d_eff, λ_hop, λ_exch and exchange-hopping split vs pressure."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    P = ct["P"]
    ax1.plot(P, ct["V_d_eff"], "o-", ms=3, lw=1.4, label=r"$V_d^{\rm eff}$")
    ax1.plot(P, ct["lambda_hop"], "s--", ms=3, lw=1.2, label=r"$\lambda_{\rm hop}$")
    ax1.plot(P, ct["lambda_exch"], "^--", ms=3, lw=1.2, label=r"$\lambda_{\rm exch}$")
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel("Coupling [eV]")
    ax1.set_title("Pairing channel vertex")
    ax1.legend()

    ax2.plot(P, ct["exch_minus_hop"], "D-", color="purple", ms=3, lw=1.4,
             label=r"$\lambda_{\rm exch} - \lambda_{\rm hop}$")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel(r"$\lambda_{\rm exch} - \lambda_{\rm hop}$ [eV]")
    ax2.set_title("Exchange–hopping competition")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "fig_channels")


# ---------------------------------------------------------------------------
# F06  Coherence dome C_coh(P) + F_comp(P)
# ---------------------------------------------------------------------------

def figure_6_coherence(ct: dict, P_opt: float) -> list[Path]:
    """C_coh(P) dome and F_comp(P)."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    P = ct["P"]
    ax1.plot(P, ct["C_coh"], "o-", ms=3, lw=1.4, color="steelblue",
             label=r"$C_{\rm coh}(P)$")
    ax1.axvline(P_opt, color="firebrick", ls="--", lw=1,
                label=f"$P_{{opt}}={P_opt:.0f}$ GPa")
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel(r"$C_{\rm coh}$")
    ax1.set_title("Coherence dome")
    ax1.legend()

    ax2.plot(P, ct["F_comp"], "s-", ms=3, lw=1.4, color="darkorange",
             label=r"$F_{\rm comp}(P)$")
    ax2.axhline(1.0, color="k", lw=0.8, ls="--", label="P=0 baseline")
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel(r"$F_{\rm comp}$")
    ax2.set_title("Compressibility factor")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, "fig_coherence")


# ---------------------------------------------------------------------------
# F07  Two-scale: Tc_onset and Tc_zero data vs model
# ---------------------------------------------------------------------------

def figure_7_two_scale_cuprate(
    ts_table: dict,
    data: Any,
) -> list[Path]:
    """Tc_onset and Tc_zero data+model vs pressure."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    P_mod = ts_table["P"]
    ax1.plot(P_mod, ts_table["Tc_onset_model"], "-", lw=1.4, color="steelblue",
             label="model")
    # data may be a dict (HG1212_DATA) or a dataclass
    _get = (lambda d, k: d[k]) if isinstance(data, dict) else (lambda d, k: getattr(d, k, None))
    _key_p = "P_GPa" if isinstance(data, dict) else "P"
    _key_to = "Tc_onset_K" if isinstance(data, dict) else "Tc_onset"
    _key_tz = "Tc_zero_K" if isinstance(data, dict) else "Tc_zero"
    try:
        ax1.plot(_get(data, _key_p), _get(data, _key_to), "o",
                 color="steelblue", ms=5, label="data [APPROX]")
    except (KeyError, TypeError):
        pass
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel(r"$T_c^{\rm onset}$ [K]")
    ax1.set_title(r"$T_c^{\rm onset}(P)$")
    ax1.legend()

    ax2.plot(P_mod, ts_table["Tc_zero_model"], "-", lw=1.4, color="firebrick",
             label="model")
    try:
        ax2.plot(_get(data, _key_p), _get(data, _key_tz), "s",
                 color="firebrick", ms=5, label="data [APPROX]")
    except (KeyError, TypeError):
        pass
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel(r"$T_c^{\rm zero}$ [K]")
    ax2.set_title(r"$T_c^{\rm zero}(P)$ — two-scale model")
    ax2.legend()

    fig.suptitle("Hg-1212 two-scale calibration [HG1212_DATA = APPROXIMATE]",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig_two_scale")


# ---------------------------------------------------------------------------
# F08  Null-model bootstrap CI
# ---------------------------------------------------------------------------

def figure_8_null_bootstrap(null_results: dict) -> list[Path]:
    """Bootstrap CI bands for Tc_zero and Wtr null models."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    observables = ["Tc_zero", "Wtr"]
    ylabels = [r"$T_c^{\rm zero}$ [K]", r"$W_{\rm tr}$ [K]"]

    for ax, obs, ylabel in zip(axes, observables, ylabels):
        if obs not in null_results:
            continue
        nr = null_results[obs]
        colors = {"linear": "steelblue", "quadratic": "firebrick", "saturating": "seagreen"}
        P_d = nr.P
        y_d = nr.y_obs
        ax.scatter(P_d, y_d, color="k", zorder=5, s=25, label="data [APPROX]")
        for mname, boot in nr.boots.items():
            c = colors.get(mname, "gray")
            P_f = boot.P_fine
            fit = nr.fits[mname]
            ax.plot(P_f, _eval_model(mname, P_f, fit.popt),
                    "-", color=c, lw=1.2, label=mname)
            ax.fill_between(P_f, boot.ci_lo, boot.ci_hi,
                            color=c, alpha=0.15)
        ax.set_xlabel("Pressure [GPa]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Null models — {obs}\n(shading = 95% bootstrap CI)")
        ax.legend(fontsize=8)

    fig.suptitle("Null model fit quality does NOT imply physical mechanism",
                 fontsize=9, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_null_bootstrap")


def _eval_model(name: str, P: NDArray, popt: NDArray) -> NDArray:
    if name == "linear":
        return popt[0] + popt[1] * P
    elif name == "quadratic":
        return popt[0] + popt[1] * P + popt[2] * P**2
    elif name == "saturating":
        return popt[0] + popt[1] * (1 - np.exp(-P / popt[2]))
    return np.full_like(P, np.nan)


# ---------------------------------------------------------------------------
# F09  RPA χ map
# ---------------------------------------------------------------------------

def figure_9_rpa_chi(rpa: Any) -> list[Path]:
    """χ₀ and χ_RPA(q) heatmaps at P=0."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    import numpy.fft as fft
    chi0 = rpa.chi0_P0
    chi_r = rpa.chi_rpa_P0
    Nk = rpa.Nk

    ext = [-np.pi, np.pi, -np.pi, np.pi]
    kw = dict(origin="lower", extent=ext, aspect="equal", cmap="inferno")

    im1 = ax1.imshow(fft.fftshift(chi0).T, **kw)
    plt.colorbar(im1, ax=ax1, label=r"$\chi_0$ [1/eV]")
    ax1.set_title(r"Static Lindhard $\chi_0(\mathbf{q})$  [P=0]")
    ax1.set_xlabel(r"$q_x$"); ax1.set_ylabel(r"$q_y$")
    ax1.plot(np.pi, np.pi, "w+", ms=10, label=r"$Q_{\rm AFM}=(\pi,\pi)$")
    ax1.legend(fontsize=8)

    im2 = ax2.imshow(fft.fftshift(chi_r).T, **kw)
    plt.colorbar(im2, ax=ax2, label=r"$\chi_{\rm RPA}$ [1/eV]")
    ax2.set_title(r"RPA susceptibility $\chi_{\rm RPA}(\mathbf{q})$  [P=0]")
    ax2.set_xlabel(r"$q_x$"); ax2.set_ylabel(r"$q_y$")

    fig.suptitle(rpa.label, fontsize=8, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_rpa_chi")


# ---------------------------------------------------------------------------
# F10  RPA channel eigenvalues vs P
# ---------------------------------------------------------------------------

def figure_10_rpa_channels(rpa: Any) -> list[Path]:
    """λ_d and λ_s vs pressure."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    P = rpa.P
    ax1.plot(P, rpa.lambda_d, "o-", ms=3, lw=1.4, color="steelblue",
             label=r"$\lambda_d$  (d-wave)")
    ax1.plot(P, rpa.lambda_s, "s--", ms=3, lw=1.2, color="firebrick",
             label=r"$\lambda_s$  (s-wave)")
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel("Pairing eigenvalue")
    ax1.set_title("RPA channel eigenvalues")
    ax1.legend()

    ax2.plot(P, rpa.omega_sf * 1e3, "D-", ms=3, lw=1.4, color="purple",
             label=r"$\omega_{\rm sf}$ (meV)")
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel(r"$\omega_{\rm sf}$ [meV]")
    ax2.set_title("Paramagnon energy scale")
    ax2.legend()

    fig.suptitle(rpa.label, fontsize=8, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_rpa_channels")


# ---------------------------------------------------------------------------
# F11  Correlation: Δ_HF, m_HF, Z_BR vs P
# ---------------------------------------------------------------------------

def figure_11_correlation_hf(corr: Any) -> list[Path]:
    """Δ_HF(P), m(P) and Z_BR(P) correlation proxies."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    P = corr.P
    axes[0].plot(P, corr.Delta_HF * 1e3, "o-", ms=3, lw=1.4, color="steelblue")
    axes[0].set_xlabel("Pressure [GPa]")
    axes[0].set_ylabel(r"$\Delta_{\rm HF}$ [meV]")
    axes[0].set_title("Hubbard-HF gap [LABEL A]")
    axes[0].text(0.05, 0.05,
                 "banda única != gap\nde transferência de carga",
                 transform=axes[0].transAxes, fontsize=7, color="gray")

    axes[1].plot(P, corr.m_HF, "s-", ms=3, lw=1.4, color="darkorange")
    axes[1].set_xlabel("Pressure [GPa]")
    axes[1].set_ylabel(r"$m_{\rm HF}$ (half-filling proxy)")
    axes[1].set_title("HF proxy magnetisation [LABEL B]")

    axes[2].plot(P, corr.Z_BR, "^-", ms=3, lw=1.4, color="seagreen")
    axes[2].axhline(0, color="k", lw=0.6, ls=":")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Pressure [GPa]")
    axes[2].set_ylabel(r"$Z_{\rm BR}$")
    axes[2].set_title("Brinkman-Rice Z [LABEL C]")
    axes[2].text(0.05, 0.95, f"$U_{{corr}}={corr.U_corr:.1f}$ eV [ASSUMED]",
                 transform=axes[2].transAxes, fontsize=7, color="gray", va="top")

    fig.tight_layout()
    return _save(fig, "fig_correlation_hf")


# ---------------------------------------------------------------------------
# F12  Superexchange J_Hub and J_Emery vs P
# ---------------------------------------------------------------------------

def figure_12_superexchange(corr: Any) -> list[Path]:
    """J_Hub(P) and J_Emery(P) enhancement, with over-prediction flag."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    P = corr.P
    ax1.plot(P, corr.J_Hub * 1e3, "o-", ms=3, lw=1.4, color="steelblue",
             label=r"$J_{\rm Hub}=4t^2/U$")
    ax1.plot(P, corr.J_Em * 1e3, "s--", ms=3, lw=1.2, color="firebrick",
             label=r"$J_{\rm Emery}$")
    ax1.set_xlabel("Pressure [GPa]")
    ax1.set_ylabel("Superexchange [meV]")
    ax1.set_title("Superexchange vs pressure")
    ax1.legend()

    # Enhancement ratio
    ax2.plot(P, corr.enh_Hub, "o-", ms=3, lw=1.4, color="steelblue",
             label=r"$r_{\rm Hub}$")
    ax2.plot(P, corr.enh_Emery, "s--", ms=3, lw=1.2, color="firebrick",
             label=r"$r_{\rm Emery}$")
    # Shade over-predicted region
    overpred = np.asarray(corr.overpred, dtype=bool)
    if overpred.any():
        ax2.fill_between(P, corr.enh_Hub, corr.enh_Emery,
                         where=overpred, color="firebrick", alpha=0.15,
                         label="Emery over-predicts [LABEL D]")
    ax2.axhline(1.0, color="k", lw=0.6, ls="--")
    ax2.set_xlabel("Pressure [GPa]")
    ax2.set_ylabel("Enhancement ratio (P=0 baseline)")
    ax2.set_title("J enhancement [LABEL D]")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "fig_superexchange")


# ---------------------------------------------------------------------------
# Legacy stubs (Phase 3–8 originals) — kept for API compatibility
# ---------------------------------------------------------------------------

def figure_3_bdg_spectrum(
    k_flat: NDArray,
    E_k: NDArray,
    delta_k: NDArray,
    symmetry: str,
) -> list[Path]:
    raise NotImplementedError


def figure_4_convergence(residuals: list[float]) -> list[Path]:
    raise NotImplementedError


def figure_5_correlator(
    r: NDArray,
    C_physical: NDArray,
    C_null: dict[str, NDArray],
) -> list[Path]:
    raise NotImplementedError


def figure_6_channels(
    mu_grid: NDArray,
    lambda_vs_mu: dict[str, NDArray],
) -> list[Path]:
    raise NotImplementedError


def figure_7_two_scale(
    mu_grid: NDArray,
    lambda_grid: NDArray,
    R_map: NDArray,
    sector_map: NDArray,
) -> list[Path]:
    raise NotImplementedError


def figure_8_robustness(proxy_values: dict[str, float]) -> list[Path]:
    raise NotImplementedError
