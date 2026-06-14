"""
run_qe_scaffold.py — Generate QE scaffold directory tree for Hg-1212 pressure study.

Produces:
  qe_runs/
    README_SCAFFOLD.txt
    P0kbar/   P50kbar/   P100kbar/ ... P300kbar/
      vc-relax.in
      scf.in
      nscf.in
      dos.in
      projwfc.in
      submit_slurm.sh
      out/

Does NOT run Quantum ESPRESSO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.qe_scaffold import generate_qe_scaffold, LABEL_QE

# ---------------------------------------------------------------------------
# Pressure grid
# ---------------------------------------------------------------------------
# Match the experimental range for Hg-1212 (0–30 GPa, steps of 5 GPa)
# plus a few extra points matching the two-scale model grid
P_GRID_GPA: np.ndarray = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)

QE_ROOT = ROOT / "qe_runs"


def print_tree(root: Path, manifest: dict) -> None:
    print(f"\n  {root}/")
    print(f"  └── README_SCAFFOLD.txt")
    for P_GPa in sorted(manifest):
        P_kbar = P_GPa * 10.0
        folder = f"P{P_kbar:.0f}kbar"
        files = list(manifest[P_GPa].keys()) + ["out/"]
        print(f"  ├── {folder}/")
        for i, fname in enumerate(files):
            connector = "└──" if i == len(files) - 1 else "├──"
            print(f"  │   {connector} {fname}")
    print()


def main() -> None:
    print("=" * 72)
    print("  QE SCAFFOLD GENERATOR — Hg-1212 under pressure")
    print("=" * 72)
    print(f"  [LABEL] {LABEL_QE}")
    print()
    print(f"  Pressure points: {list(P_GRID_GPA)} GPa")
    print(f"  Output root    : {QE_ROOT}")
    print()

    manifest = generate_qe_scaffold(
        P_grid_GPa=P_GRID_GPA,
        root=QE_ROOT,
        compound="Hg1212",
    )

    print("  Files generated:")
    print_tree(QE_ROOT, manifest)

    # Count totals
    n_folders = len(manifest)
    n_files = sum(len(v) for v in manifest.values())
    print(f"  {n_folders} pressure folders × 6 input files = {n_files} files total")
    print(f"  + README_SCAFFOLD.txt + {n_folders} empty out/ directories")
    print()

    # Show a sample snippet from vc-relax.in at P=0
    sample_path = QE_ROOT / "P0kbar" / "vc-relax.in"
    print("─" * 72)
    print("  SAMPLE: P0kbar/vc-relax.in  (first 30 lines)")
    print("─" * 72)
    lines = sample_path.read_text().splitlines()
    for i, line in enumerate(lines[:30], 1):
        print(f"  {i:3d}  {line}")
    print("  ...")
    print()

    # Show README
    readme_path = QE_ROOT / "README_SCAFFOLD.txt"
    print("─" * 72)
    print("  README_SCAFFOLD.txt")
    print("─" * 72)
    for line in readme_path.read_text().splitlines():
        print(f"  {line}")
    print()

    print("=" * 72)
    print("  IMPORTANT: QE was NOT executed.")
    print("  These are template files only.")
    print("  All lattice constants and atomic positions are [PLACEHOLDER].")
    print("  Pseudopotentials must be downloaded separately.")
    print("=" * 72)


if __name__ == "__main__":
    main()
