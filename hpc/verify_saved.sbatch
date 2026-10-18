#!/bin/bash
#SBATCH --job-name=dl-saved-audit
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/hpc-audit-%j.out
#SBATCH --error=logs/hpc-audit-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch from the repository root}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
dl_python="${DL_PYTHON:-$HOME/.conda/envs/dl-project/bin/python}"
"$dl_python" scripts/hpc_smoke.py --device cpu --threads "$SLURM_CPUS_PER_TASK" \
    --output "outputs/hpc_readiness/cpu-smoke-${SLURM_JOB_ID}.json"
"$dl_python" scripts/verify_artifacts.py
"$dl_python" scripts/verify_stage2_artifacts.py
"$dl_python" scripts/verify_stage3_artifacts.py
