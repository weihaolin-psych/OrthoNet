#!/usr/bin/env bash
# Process model outputs using project-relative defaults. Run from any directory.
set -euo pipefail

# ---- Configuration ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SEEDS=(7 42 88 123 200 314 456 555 789 999)
SCRIPT="$SCRIPT_DIR/compute_logit_measures.py"

# ---- Run only after the required weights/evaluations/NPZs have been prepared --
cd "$PROJECT_ROOT"
"$PYTHON" -u "$SCRIPT"
