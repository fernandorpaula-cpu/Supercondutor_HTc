#!/bin/bash
# SLURM submit script — Hg-1212, P = 20.0 GPa (200 kbar)
# [PLACEHOLDER] edit #SBATCH directives for your cluster
#
#SBATCH --job-name=hg1212_P200kbar
#SBATCH --partition=cpu      # [PLACEHOLDER]
#SBATCH --account=your_account          # [PLACEHOLDER]
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load QuantumESPRESSO/7.2-foss-2022b   # [PLACEHOLDER] verify module name

MPI="srun -n $SLURM_NTASKS"

echo "=== Step 1: vc-relax ==="
$MPI pw.x < vc-relax.in > vc-relax.out

echo "=== Step 2: SCF ==="
$MPI pw.x < scf.in > scf.out

echo "=== Step 3: NSCF (dense k-mesh) ==="
$MPI pw.x < nscf.in > nscf.out

echo "=== Step 4: DOS ==="
$MPI dos.x < dos.in > dos.out

echo "=== Step 5: Projected DOS ==="
$MPI projwfc.x < projwfc.in > projwfc.out

echo "=== Done: Hg1212_P200kbar P=20.0 GPa ==="
