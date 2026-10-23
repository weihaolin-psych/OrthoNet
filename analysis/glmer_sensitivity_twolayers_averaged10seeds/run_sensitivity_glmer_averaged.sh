#!/usr/bin/env bash
# Run the saved ten-seed sensitivity analysis; no automatic package installation.
set -euo pipefail

# ---- Configuration ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RSCRIPT="${RSCRIPT:-Rscript}"

# ---- Input paths are resolved from the project root -------------------------
cd "$PROJECT_ROOT"
"$RSCRIPT" --vanilla "$SCRIPT_DIR/02_sensitivity_glmer_averaged.R"
