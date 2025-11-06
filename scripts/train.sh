#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd)" \
python -m tactile_module.train_force_dummy \
  --config "${PROJECT_ROOT}/configs/default.yaml" \
  "$@"
