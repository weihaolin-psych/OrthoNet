#!/usr/bin/env bash
# Run the ten-seed model workflow. Default executes sequentially; no jobs are
# submitted unless SUBMIT_TO_SLURM=1. Run: bash models/twolayers_10-random-seeds/eval_twolayers_all.sh [hidden_dim ...].
set -euo pipefail

# ---- Configuration ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SUBMIT_TO_SLURM="${SUBMIT_TO_SLURM:-0}"
DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-}"
MEMORY="${MEMORY:-32G}"
WALL_TIME="${WALL_TIME:-04:00:00}"
CPUS="${CPUS:-4}"
GPU_COUNT="${GPU_COUNT:-1}"
SEEDS=(7 42 88 123 200 314 456 555 789 999)
HIDDEN_DIMS=(4 8 10 12 14 16 20 22 24 26 28 32 64 128 256)
EPOCHS=25
MODEL_TYPES=(twolayers)
SCRIPT="$PROJECT_ROOT/models/twolayers_10-random-seeds/evaluate_twolayers.py"

# ---- Validate the whole request before running commands ---------------------
if (( $# > 0 )); then HIDDEN_DIMS=("$@"); fi
[[ -f "$SCRIPT" ]] || { echo "Missing script: $SCRIPT" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "Python is not available: $PYTHON" >&2; exit 1; }
for DIM in "${HIDDEN_DIMS[@]}"; do
    [[ "$DIM" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid dimension: $DIM" >&2; exit 1; }
done
cd "$PROJECT_ROOT"

# ---- Execute locally or submit through an already configured scheduler -------
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DIM in "${HIDDEN_DIMS[@]}"; do
            COMMAND=("$PYTHON" -u "$SCRIPT" --hidden_dim "$DIM" --seed "$SEED")
            echo "evaluation: model=$MODEL_TYPE dimension=$DIM seed=$SEED"
            if [[ "$DRY_RUN" == "1" ]]; then
                printf "%q " "${COMMAND[@]}"
                printf "\n"
                continue
            fi
            if [[ "$SUBMIT_TO_SLURM" == "1" ]]; then
                OPTIONS=(--nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEMORY" --time="$WALL_TIME")
                if [[ -n "$PARTITION" ]]; then OPTIONS+=(--partition="$PARTITION"); fi
                if [[ "$GPU_COUNT" != "0" ]]; then OPTIONS+=(--gres="gpu:$GPU_COUNT"); fi
                printf -v WRAPPED '%q ' "${COMMAND[@]}"
                sbatch "${OPTIONS[@]}" --chdir="$PROJECT_ROOT" --wrap="$WRAPPED"
            else
                "${COMMAND[@]}"
            fi
        done
    done
done
