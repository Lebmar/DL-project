#!/bin/bash
#SBATCH --job-name=dl-long-validation
#SBATCH --partition=i64m512u
#SBATCH --qos=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/long-validation-%j.out
#SBATCH --error=logs/long-validation-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch from the committed repository root}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
dl_python="${DL_PYTHON:-$HOME/.conda/envs/dl-project/bin/python}"
dl_status="logs/long-validation-${SLURM_JOB_ID}.status"
trap 'dl_exit=$?; if [ "$dl_exit" -eq 0 ]; then printf "COMPLETE\n" > "$dl_status"; else printf "FAILED exit=%s\n" "$dl_exit" > "$dl_status"; fi' EXIT
printf 'SELF_CHECKS\n' > "$dl_status"
"$dl_python" scripts/evaluate_long_window_validation.py --self-check
"$dl_python" scripts/analyze_long_window_validation.py --self-check
"$dl_python" scripts/verify_long_window_validation.py --self-check
printf 'EVALUATION\n' > "$dl_status"
"$dl_python" scripts/evaluate_long_window_validation.py
printf 'ANALYSIS\n' > "$dl_status"
"$dl_python" scripts/analyze_long_window_validation.py
printf 'INDEPENDENT_VERIFICATION\n' > "$dl_status"
"$dl_python" scripts/verify_long_window_validation.py
