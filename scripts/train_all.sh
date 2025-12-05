#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

./scripts/train_land.sh "$@"
./scripts/train_strafe_left.sh "$@"
./scripts/train_strafe_right.sh "$@"
./scripts/train_takeoff.sh "$@"
