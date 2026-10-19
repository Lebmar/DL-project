#!/bin/bash
#SBATCH --job-name=dl-finite-baseline
#SBATCH --partition=i64m512u
#SBATCH --qos=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/finite-baseline-%j.out
#SBATCH --error=logs/finite-baseline-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Requires Slurm}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$HOME/.conda/envs/dl-project/bin/python" scripts/benchmark_population_matched_baseline.py
