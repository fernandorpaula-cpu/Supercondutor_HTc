r"""
Reporting: render the PT0 validation results to

    * a human-readable error table (markdown + plain text),
    * the GO/NO-GO log file required by the prompt,
    * profile figures (density, mass, metric potentials).

No physics here; pure presentation.  All file paths are under output/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import m_to_pc
from .validate import CaseResult, aggregate_decision

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


def _fmt(x) -> str:
    if x is None:
        return "   ---   "
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-3:
            return f"{x:.4e}"
        return f"{x:.4f}"
    return str(x)


def error_table_markdown(results: list[CaseResult]) -> str:
    lines = ["# PT0 error table vs Crespi targets", ""]
    for r in results:
        lines.append(f"## Case `{r.case_id}`  (mc^2 = {r.mc2_keV:g} keV)")
        lines.append("")
        lines.append(f"- converged: **{r.converged}**")
        cp = r.central_params
        lines.append(f"- central params: theta0={cp['theta0']:.4f}, "
                     f"beta0={cp['beta0']:.3e}, W0={cp['W0']:.3f}")
        lines.append("")
        lines.append("| observable | unit | model | target | rel.err | verdict |")
        lines.append("|---|---|---|---|---|---|")
        for c in r.comparisons:
            lines.append(
                f"| {c.name} | {c.unit} | {_fmt(c.model)} | {_fmt(c.target)} "
                f"| {_fmt(c.rel_error)} | {c.verdict} |")
        lines.append("")
        if r.notes:
            lines.append("Notes:")
            for n in r.notes:
                lines.append(f"  - {n}")
            lines.append("")
        lines.append(f"**Case verdict: {r.verdict}**")
        lines.append("")
    lines.append(f"## OVERALL DECISION: {aggregate_decision(results)}")
    lines.append("")
    return "\n".join(lines)


def write_go_no_go(results: list[CaseResult], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = aggregate_decision(results)

    recommend = {
        "EXCELLENT": "ADVANCE to quantitative paper (reproduction excellent).",
        "GO": "ADVANCE to quantitative paper (reproduction sufficient).",
        "BORDERLINE": "DO NOT advance yet: audit numerics/targets first.",
        "NO-GO": "DO NOT advance to a quantitative paper: reproduction failed.",
        "NO-TARGET": ("CANNOT DECIDE: no numeric Crespi targets present. "
                      "Fill data/crespi_table4_targets.yaml with real Table 4 "
                      "values, then re-run."),
    }[decision]

    lines = [
        "================ PT0 GO / NO-GO ================",
        f"overall decision : {decision}",
        f"recommendation   : {recommend}",
        "",
        "error bands: err<=0.01 EXCELLENT | err<=0.03 GO | "
        "err<=0.10 BORDERLINE | err>0.10 NO-GO",
        "err = |model - target| / |target|",
        "",
    ]
    for r in results:
        lines.append(f"--- case {r.case_id} (mc^2={r.mc2_keV:g} keV) "
                     f"verdict={r.verdict} converged={r.converged} ---")
        for c in r.comparisons:
            lines.append(
                f"    {c.name:<34s} model={_fmt(c.model):>12s} "
                f"target={_fmt(c.target):>12s} err={_fmt(c.rel_error):>10s} "
                f"-> {c.verdict}")
        for n in r.notes:
            lines.append(f"    note: {n}")
        lines.append("")
    lines.append("================================================")
    text = "\n".join(lines)
    path.write_text(text)
    return text


def plot_profiles(results: list[CaseResult], outdir: str | Path) -> list[str]:
    if not _HAVE_MPL:
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # density + enclosed mass
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in results:
        if r.profile is None:
            continue
        p = r.profile
        r_pc = np.array([m_to_pc(x) for x in p.r_m])
        good = p.rho_kg_m3 > 0
        axes[0].loglog(r_pc[good], p.rho_kg_m3[good],
                       label=f"{r.mc2_keV:g} keV")
        axes[1].loglog(r_pc, p.mass_kg / 1.98847e30,
                       label=f"{r.mc2_keV:g} keV")
    axes[0].set_xlabel("r [pc]"); axes[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    axes[0].set_title("RAR density profile"); axes[0].legend()
    axes[1].set_xlabel("r [pc]"); axes[1].set_ylabel(r"$M(<r)$ [M$_\odot$]")
    axes[1].set_title("Enclosed mass"); axes[1].legend()
    fig.tight_layout()
    f1 = outdir / "pt0_profiles_density_mass.png"
    fig.savefig(f1, dpi=160); plt.close(fig); written.append(str(f1))

    # metric potentials
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in results:
        if r.profile is None:
            continue
        p = r.profile
        r_pc = np.array([m_to_pc(x) for x in p.r_m])
        ax.semilogx(r_pc, p.nu_metric, label=fr"$\nu$ {r.mc2_keV:g} keV")
        ax.semilogx(r_pc, p.lambda_metric, ls="--",
                    label=fr"$\lambda$ {r.mc2_keV:g} keV")
    ax.set_xlabel("r [pc]"); ax.set_ylabel("metric potential")
    ax.set_title("Metric potentials"); ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = outdir / "pt0_metric_potentials.png"
    fig.savefig(f2, dpi=160); plt.close(fig); written.append(str(f2))
    return written
