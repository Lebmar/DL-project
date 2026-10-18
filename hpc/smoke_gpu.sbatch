#!/bin/bash
#SBATCH --job-name=dl-gpu-smoke
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/hpc-gpu-%j.out
#SBATCH --error=logs/hpc-gpu-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch from the repository root}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
dl_python="${DL_PYTHON:-$HOME/.conda/envs/dl-project/bin/python}"
"$dl_python" scripts/hpc_smoke.py --device cuda --threads "$SLURM_CPUS_PER_TASK" \
    --output "outputs/hpc_readiness/gpu-smoke-${SLURM_JOB_ID}.json"
