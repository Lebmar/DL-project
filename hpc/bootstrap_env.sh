#!/bin/bash
# Run on the HPC II login host; installs only in this user's dedicated prefix.
set -euo pipefail
dl_repo="$(cd "$(dirname "$0")/.." && pwd)"
dl_conda=/hpc2ssd/softwares/anaconda3/bin/conda
dl_env="${DL_ENV_PREFIX:-$HOME/.conda/envs/dl-project}"
if [ "$(uname -s)" != Linux ] || [ ! -x "$dl_conda" ]; then
    echo 'This bootstrap targets the verified HKUST(GZ) HPC II software layout.' >&2
    exit 1
fi
if [ ! -e "$dl_env" ]; then
    "$dl_conda" create --prefix "$dl_env" --override-channels -c conda-forge \
        python=3.12.14 pip -y
fi
dl_python="$dl_env/bin/python"
"$dl_python" -c 'import sys; assert sys.version_info[:3] == (3,12,14), sys.version'
export PYTHONNOUSERSITE=1
# CUDA 12.6 wheel: verified for the debug A40 driver, not the PyPI CUDA 13 default.
"$dl_python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.14.1+cu126'
"$dl_python" -m pip install --no-cache-dir --index-url https://pypi.org/simple \
    -r "$dl_repo/requirements-lock.txt"
"$dl_python" -m pip check
mkdir -p "$dl_repo/logs"
"$dl_python" -m pip list --format=freeze > "$dl_repo/logs/requirements-hpc-installed.txt"
