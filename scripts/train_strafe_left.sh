#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional: activate env
# source /home/dils/miniconda3/etc/profile.d/conda.sh
# conda activate base

python -m trainers.strafe_left "$@"
