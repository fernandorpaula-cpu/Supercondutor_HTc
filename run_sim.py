"""
run_sim.py — Master runner for the lattice / BdG / RPA simulation suite.

Usage:
    python run_sim.py [--phase N]

Without --phase, runs all completed phases in order and skips stubs.

Exit codes:
    0 — all completed phases ran without error.
    1 — at least one phase failed.

Phase map:
    1  config + scaffold sanity (always runs)
    2  lattice_bands  — band structure and DOS
    3  pairing_bdg    — BdG self-consistency
    4  mediator_rpa   — RPA vertex and Eliashberg
    5  channels + null_models — symmetry decomposition and null benchmarks
    6  two_scale      — crossover diagnostics and sector map
    7  qe_scaffold    — QE I/O (requires external QE installation)
    8  figures        — generate all figures
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from config import OUTPUT_DIR, DATA_DIR


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def phase_1_sanity() -> None:
    """Import all modules and confirm scaffold is in place."""
    import importlib
    modules = [
        "src.lattice_bands", "src.correlation", "src.pairing_bdg",
        "src.mediator_rpa", "src.channels", "src.null_models",
        "src.two_scale", "src.qe_scaffold", "src.figures",
    ]
    for name in modules:
        importlib.import_module(name)
    print("  [Phase 1] All modules imported — scaffold OK.")


def phase_2_bands() -> None:
    """Band structure and density of states — Phase 2 stub."""
    # Phase 2: call lattice_bands.dispersion_square(), density_of_states(),
    #          save results to data/bands.npz, generate figure_1.
    raise NotImplementedError("Phase 2 not yet implemented.")


def phase_3_bdg() -> None:
    """BdG self-consistency — Phase 3 stub."""
    raise NotImplementedError("Phase 3 not yet implemented.")


def phase_4_rpa() -> None:
    """RPA mediator and Eliashberg function — Phase 4 stub."""
    raise NotImplementedError("Phase 4 not yet implemented.")


def phase_5_channels() -> None:
    """Symmetry channel decomposition and null models — Phase 5 stub."""
    raise NotImplementedError("Phase 5 not yet implemented.")


def phase_6_two_scale() -> None:
    """Two-scale crossover diagnostics — Phase 6 stub."""
    raise NotImplementedError("Phase 6 not yet implemented.")


def phase_7_qe() -> None:
    """QE I/O scaffold — Phase 7 stub (requires external QE)."""
    raise NotImplementedError("Phase 7 not yet implemented.")


def phase_8_figures() -> None:
    """Generate all figures — Phase 8 stub."""
    raise NotImplementedError("Phase 8 not yet implemented.")


PHASES: dict[int, tuple[str, callable]] = {
    1: ("Scaffold sanity",              phase_1_sanity),
    2: ("Lattice bands + DOS",          phase_2_bands),
    3: ("BdG self-consistency",         phase_3_bdg),
    4: ("RPA mediator / Eliashberg",    phase_4_rpa),
    5: ("Channels + null models",       phase_5_channels),
    6: ("Two-scale crossover",          phase_6_two_scale),
    7: ("QE scaffold",                  phase_7_qe),
    8: ("Figures",                      phase_8_figures),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run a single phase (1–8). Omit to run all available phases.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    phases_to_run = [args.phase] if args.phase else list(PHASES.keys())
    failed = False

    for n in phases_to_run:
        label, fn = PHASES[n]
        print(f"\n=== Phase {n}: {label} ===")
        try:
            fn()
            print(f"  [Phase {n}] OK")
        except NotImplementedError:
            print(f"  [Phase {n}] SKIPPED — not yet implemented.")
        except Exception:
            print(f"  [Phase {n}] FAILED:")
            traceback.print_exc()
            failed = True

    print("\n--- run_sim complete ---")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
