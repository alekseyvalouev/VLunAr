#!/usr/bin/env bash
set -e

# Go to repo root (parent of scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional: set your virtualenv here
# source venv/bin/activate

python -m trainers.land "$@"
