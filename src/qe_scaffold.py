"""Generate Quantum ESPRESSO scaffold files for pressure-dependent runs.

This module does not run Quantum ESPRESSO. It only creates placeholder input
files for future vc-relax, scf, nscf, dos.x and projwfc.x calculations.

Scientific status:
- Scaffold only.
- Not a production DFT calculation.
- Structures and pseudopotentials must be verified before cluster use.
"""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable


ANGSTROM_TO_BOHR = 1.889726124565062
KBAR_PER_GPA = 10.0


# Placeholder Hg1212-like motif.
# This is NOT a validated crystallographic model.
ATOM_POSITIONS_CRYSTAL: list[tuple[str, float, float, float]] = [
    ("Hg", 0.000000, 0.000000, 0.000000),
    ("Ba", 0.500000, 0.500000, 0.180000),
    ("Ba", 0.500000, 0.500000, 0.820000),
    ("Ca", 0.000000, 0.000000, 0.500000),
    ("Cu", 0.000000, 0.000000, 0.360000),
    ("Cu", 0.000000, 0.000000, 0.640000),
    ("O", 0.500000, 0.000000, 0.360000),
    ("O", 0.000000, 0.500000, 0.360000),
    ("O", 0.500000, 0.000000, 0.640000),
    ("O", 0.000000, 0.500000, 0.640000),
    ("O", 0.000000, 0.000000, 0.250000),
    ("O", 0.000000, 0.000000, 0.750000),
]


def _ensure_folder(folder: Path) -> Path:
    """Create folder if needed and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_text_utf8(path: Path, text: str) -> Path:
    """Write UTF-8 text with LF line endings."""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _ang_to_bohr(value_angstrom: float) -> float:
    """Convert angstrom to bohr."""
    return value_angstrom * ANGSTROM_TO_BOHR


def _lattice_at_P(P_GPa: float) -> tuple[float, float]:
    """Return placeholder pressure-dependent lattice constants a and c.

    These are scaffold values only. Replace with CIF-derived or relaxed
    structural parameters before production DFT use.
    """
    a0 = 3.86
    c0 = 12.70

    a = a0 * (1.0 - 0.0015 * P_GPa)
    c = c0 * (1.0 - 0.0030 * P_GPa)

    return a, c


def _pressure_folder_name(P_GPa: float) -> str:
    """Return deterministic pressure folder name."""
    P_kbar = P_GPa * KBAR_PER_GPA
    return f"P{P_kbar:.0f}kbar"


def _atomic_positions_block() -> str:
    """Return QE ATOMIC_POSITIONS crystal block."""
    lines = ["ATOMIC_POSITIONS crystal"]
    for sym, x, y, z in ATOM_POSITIONS_CRYSTAL:
        lines.append(f"  {sym:<2s}  {x:.8f}  {y:.8f}  {z:.8f}")
    return "\n".join(lines)


def write_vc_relax(folder: Path, P_GPa: float, prefix: str) -> Path:
    """Write vc-relax.in for variable-cell relaxation at pressure P_GPa."""
    P_kbar = P_GPa * KBAR_PER_GPA
    a_ang, c_ang = _lattice_at_P(P_GPa)
    a_bohr = _ang_to_bohr(a_ang)
    c_bohr = _ang_to_bohr(c_ang)

    path = folder / "vc-relax.in"

    text = textwrap.dedent(
        f"""\
        ! vc-relax.in - Hg1212 scaffold, P = {P_GPa:.1f} GPa ({P_kbar:.0f} kbar)
        ! [PLACEHOLDER] Atomic positions and lattice constants must be verified from CIF.
        ! [PLACEHOLDER] Pseudopotentials must be downloaded and tested before running.
        ! [PLACEHOLDER] This input is a scaffold, not a validated DFT setup.

        &CONTROL
          calculation   = 'vc-relax'
          restart_mode  = 'from_scratch'
          prefix        = '{prefix}'
          outdir        = './out'
          pseudo_dir    = './pseudo'
          verbosity     = 'high'
        /

        &SYSTEM
          ibrav       = 0
          nat         = {len(ATOM_POSITIONS_CRYSTAL)}
          ntyp        = 5
          ecutwfc     = 60
          ecutrho     = 480
          occupations = 'smearing'
          smearing    = 'mv'
          degauss     = 0.01
        /

        &ELECTRONS
          conv_thr    = 1.0d-8
          mixing_beta = 0.3
        /

        &IONS
          ion_dynamics = 'bfgs'
        /

        &CELL
          cell_dynamics  = 'bfgs'
          press          = {P_kbar:.1f}
          press_conv_thr = 0.5
        /

        CELL_PARAMETERS bohr
          {a_bohr:.8f}  0.00000000  0.00000000
          0.00000000  {a_bohr:.8f}  0.00000000
          0.00000000  0.00000000  {c_bohr:.8f}

        ATOMIC_SPECIES
          Hg  200.59  Hg.UPF
          Ba  137.33  Ba.UPF
          Ca   40.08  Ca.UPF
          Cu   63.55  Cu.UPF
          O    16.00  O.UPF

        {_atomic_positions_block()}

        K_POINTS automatic
          8 8 4  0 0 0
        """
    )

    return _write_text_utf8(path, text)


def write_scf(folder: Path, P_GPa: float, prefix: str) -> Path:
    """Write scf.in for a fixed-cell SCF calculation."""
    a_ang, c_ang = _lattice_at_P(P_GPa)
    a_bohr = _ang_to_bohr(a_ang)
    c_bohr = _ang_to_bohr(c_ang)

    path = folder / "scf.in"

    text = textwrap.dedent(
        f"""\
        ! scf.in - Hg1212 scaffold, P = {P_GPa:.1f} GPa
        ! [PLACEHOLDER] Use a validated relaxed structure before production calculations.
        ! [PLACEHOLDER] Pseudopotentials must be tested before running.

        &CONTROL
          calculation  = 'scf'
          restart_mode = 'from_scratch'
          prefix       = '{prefix}'
          outdir       = './out'
          pseudo_dir   = './pseudo'
        /

        &SYSTEM
          ibrav       = 0
          nat         = {len(ATOM_POSITIONS_CRYSTAL)}
          ntyp        = 5
          ecutwfc     = 60
          ecutrho     = 480
          occupations = 'smearing'
          smearing    = 'mv'
          degauss     = 0.01
        /

        &ELECTRONS
          conv_thr    = 1.0d-8
          mixing_beta = 0.3
        /

        CELL_PARAMETERS bohr
          {a_bohr:.8f}  0.00000000  0.00000000
          0.00000000  {a_bohr:.8f}  0.00000000
          0.00000000  0.00000000  {c_bohr:.8f}

        ATOMIC_SPECIES
          Hg  200.59  Hg.UPF
          Ba  137.33  Ba.UPF
          Ca   40.08  Ca.UPF
          Cu   63.55  Cu.UPF
          O    16.00  O.UPF

        {_atomic_positions_block()}

        K_POINTS automatic
          12 12 6  0 0 0
        """
    )

    return _write_text_utf8(path, text)


def write_nscf(folder: Path, P_GPa: float, prefix: str) -> Path:
    """Write nscf.in for dense-grid DOS calculation."""
    a_ang, c_ang = _lattice_at_P(P_GPa)
    a_bohr = _ang_to_bohr(a_ang)
    c_bohr = _ang_to_bohr(c_ang)

    path = folder / "nscf.in"

    text = textwrap.dedent(
        f"""\
        ! nscf.in - Hg1212 scaffold, P = {P_GPa:.1f} GPa
        ! [PLACEHOLDER] Dense k-mesh must be converged before production use.
        ! [PLACEHOLDER] Use only after validated SCF convergence.

        &CONTROL
          calculation = 'nscf'
          prefix      = '{prefix}'
          outdir      = './out'
          pseudo_dir  = './pseudo'
        /

        &SYSTEM
          ibrav       = 0
          nat         = {len(ATOM_POSITIONS_CRYSTAL)}
          ntyp        = 5
          ecutwfc     = 60
          ecutrho     = 480
          occupations = 'tetrahedra'
        /

        &ELECTRONS
          conv_thr    = 1.0d-8
          mixing_beta = 0.3
        /

        CELL_PARAMETERS bohr
          {a_bohr:.8f}  0.00000000  0.00000000
          0.00000000  {a_bohr:.8f}  0.00000000
          0.00000000  0.00000000  {c_bohr:.8f}

        ATOMIC_SPECIES
          Hg  200.59  Hg.UPF
          Ba  137.33  Ba.UPF
          Ca   40.08  Ca.UPF
          Cu   63.55  Cu.UPF
          O    16.00  O.UPF

        {_atomic_positions_block()}

        K_POINTS automatic
          24 24 12  0 0 0
        """
    )

    return _write_text_utf8(path, text)


def write_dos(folder: Path, prefix: str) -> Path:
    """Write dos.in for Quantum ESPRESSO dos.x."""
    path = folder / "dos.in"

    text = textwrap.dedent(
        f"""\
        ! dos.in - Quantum ESPRESSO DOS scaffold
        ! [PLACEHOLDER] DOS energy window must be checked against the converged band structure.
        ! [PLACEHOLDER] This file is not a production DFT input.

        &DOS
          prefix = '{prefix}'
          outdir = './out'
          fildos = '{prefix}.dos'
          Emin   = -8.0
          Emax   =  8.0
          DeltaE =  0.01
        /
        """
    )

    return _write_text_utf8(path, text)


def write_projwfc(folder: Path, prefix: str) -> Path:
    """Write projwfc.in for orbital-projected DOS."""
    path = folder / "projwfc.in"

    text = textwrap.dedent(
        f"""\
        ! projwfc.in - Quantum ESPRESSO projected DOS scaffold
        ! [PLACEHOLDER] Orbital projections must be validated against chosen pseudopotentials.
        ! [PLACEHOLDER] This file is not a production DFT input.

        &PROJWFC
          prefix  = '{prefix}'
          outdir  = './out'
          filpdos = '{prefix}.pdos'
        /
        """
    )

    return _write_text_utf8(path, text)


def write_submit_slurm(folder: Path, P_GPa: float, prefix: str) -> Path:
    """Write submit_slurm.sh and set executable bit when supported."""
    P_kbar = P_GPa * KBAR_PER_GPA
    path = folder / "submit_slurm.sh"

    text = textwrap.dedent(
        f"""\
        #!/bin/bash
        # SLURM submit script for Hg1212, P = {P_GPa:.1f} GPa ({P_kbar:.0f} kbar)
        # [PLACEHOLDER] Edit SBATCH directives for your cluster.
        # [PLACEHOLDER] Verify pseudopotentials before running.
        # [PLACEHOLDER] Inspect vc-relax output before scf/nscf/DOS.

        #SBATCH --job-name={prefix}
        #SBATCH --output={prefix}.out
        #SBATCH --error={prefix}.err
        #SBATCH --time=24:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=16

        set -euo pipefail

        module purge
        module load quantum-espresso

        mkdir -p out pseudo

        echo "=== Running vc-relax ==="
        mpirun pw.x < vc-relax.in > vc-relax.out

        echo "=== Running scf ==="
        mpirun pw.x < scf.in > scf.out

        echo "=== Running nscf ==="
        mpirun pw.x < nscf.in > nscf.out

        echo "=== Running total DOS ==="
        mpirun dos.x < dos.in > dos.out

        echo "=== Running projected DOS ==="
        mpirun projwfc.x < projwfc.in > projwfc.out

        echo "=== Done: {prefix} P={P_GPa:.1f} GPa ==="
        """
    )

    _write_text_utf8(path, text)

    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass

    return path


def write_readme_scaffold(root: str | Path = "qe_scaffold") -> Path:
    """Write README_SCAFFOLD.txt at the scaffold root."""
    root_path = _ensure_folder(Path(root))
    path = root_path / "README_SCAFFOLD.txt"

    text = textwrap.dedent(
        """\
        Quantum ESPRESSO scaffold for pressure-dependent Hg-cuprate calculations

        Status:
        [PLACEHOLDER] This directory contains input scaffolds only.
        [PLACEHOLDER] These files are not production DFT calculations.
        [PLACEHOLDER] Atomic positions must be replaced by validated CIF-derived structures.
        [PLACEHOLDER] Pseudopotentials are not included.
        [PLACEHOLDER] K-point meshes and energy cutoffs must be converged.
        [PLACEHOLDER] SLURM directives must be edited for the target cluster.

        Intended workflow:
        vc-relax -> scf -> nscf -> dos.x -> projwfc.x

        Pressure convention:
        Quantum ESPRESSO uses kbar for press.
        The model interface uses GPa.
        Conversion: 1 GPa = 10 kbar.

        Scientific interpretation:
        This scaffold is a future validation route.
        It does not prove the effective-model mechanism.
        It should be used to compare structural trends, DOS, PDOS and charge-transfer proxies.
        """
    )

    return _write_text_utf8(path, text)


def write_qe_readme(root: str | Path = "qe_scaffold") -> Path:
    """Compatibility alias for older calls."""
    return write_readme_scaffold(root)


def generate_qe_scaffold(
    pressures_GPa: Iterable[float],
    root: str | Path = "qe_scaffold",
) -> Dict[float, Dict[str, Path]]:
    """Generate QE input scaffold for each pressure.

    Returns
    -------
    dict
        Mapping pressure_GPa -> mapping filename -> Path.
    """
    root_path = _ensure_folder(Path(root))
    manifest: Dict[float, Dict[str, Path]] = {}

    write_readme_scaffold(root_path)

    for P_raw in pressures_GPa:
        P_GPa = float(P_raw)
        P_kbar = P_GPa * KBAR_PER_GPA
        folder = _ensure_folder(root_path / _pressure_folder_name(P_GPa))
        prefix = f"Hg1212_P{P_kbar:.0f}kbar"

        files: Dict[str, Path] = {}
        files["vc-relax.in"] = write_vc_relax(folder, P_GPa, prefix)
        files["scf.in"] = write_scf(folder, P_GPa, prefix)
        files["nscf.in"] = write_nscf(folder, P_GPa, prefix)
        files["dos.in"] = write_dos(folder, prefix)
        files["projwfc.in"] = write_projwfc(folder, prefix)
        files["submit_slurm.sh"] = write_submit_slurm(folder, P_GPa, prefix)

        manifest[P_GPa] = files

    return manifest


# Required future DFT interfaces.
# These are intentionally not implemented in this effective-model package.
# Tests expect them to exist and to raise NotImplementedError.


def parse_pw_bands(*args: Any, **kwargs: Any) -> Any:
    """Future parser for Quantum ESPRESSO band outputs."""
    raise NotImplementedError(
        "parse_pw_bands is a future DFT parser, not implemented in this model package."
    )


def write_ph_input(*args: Any, **kwargs: Any) -> Any:
    """Future writer for Quantum ESPRESSO ph.x inputs."""
    raise NotImplementedError(
        "write_ph_input is a future phonon scaffold, not implemented in this model package."
    )


def parse_dynamical_matrix(*args: Any, **kwargs: Any) -> Any:
    """Future parser for dynamical matrices."""
    raise NotImplementedError(
        "parse_dynamical_matrix is a future DFT parser, not implemented in this model package."
    )


def wannier_hoppings(*args: Any, **kwargs: Any) -> Any:
    """Future extractor for Wannier hopping parameters."""
    raise NotImplementedError(
        "wannier_hoppings is a future Wannier interface, not implemented in this model package."
    )


def qe_to_lattice_params(*args: Any, **kwargs: Any) -> Any:
    """Future converter from QE outputs to effective-model lattice parameters."""
    raise NotImplementedError(
        "qe_to_lattice_params is a future DFT-to-model bridge, not implemented."
    )


if __name__ == "__main__":
    default_pressures = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 19.0]
    out = generate_qe_scaffold(default_pressures)

    print("Generated Quantum ESPRESSO scaffold:")
    for pressure, files in out.items():
        print(f"P = {pressure:.1f} GPa")
        for name, path in files.items():
            print(f"  {name}: {path}")