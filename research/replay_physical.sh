#!/usr/bin/env bash
# Native-only replay and real-video regression checks. No benchmark by default.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target fringe_count
"${PHYSIC_PYTHON:-.venv/bin/python}" research/check_real.py --binary "$TASK_ROOT/build-native/fringe_count" --csv experiments/runs/self-recorded/frames.csv --reference experiments/results/physical-count/reference-v4.json --output-dir experiments/runs/self-recorded/evaluation --prefixes
