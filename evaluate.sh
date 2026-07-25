#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

METHOD="${METHOD:-topowheat_global}"
SPLIT="${SPLIT:-val}"
EVAL_WHO="${EVAL_WHO:-TEACHER}"
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$SPLIT" in
    val)
        DATASET="gwfss_sem_seg_val"
        ;;
    test)
        DATASET="gwfss_sem_seg_test"
        ;;
    *)
        echo "SPLIT must be 'val' or 'test', got: $SPLIT" >&2
        exit 2
        ;;
esac

case "$EVAL_WHO" in
    TEACHER|STUDENT)
        ;;
    *)
        echo "EVAL_WHO must be 'TEACHER' or 'STUDENT', got: $EVAL_WHO" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    project_global)
        CONFIG_FILE="configs/gwfss/experiments/competition_stage2_stem4500.yaml"
        DEFAULT_CHECKPOINT="outputs/competition_stage2_stem4500_seed2025/model_best.pth"
        USE_DENSE_TTA="False"
        ;;
    project_zoom)
        CONFIG_FILE="configs/gwfss/experiments/competition_stage2_stem4500.yaml"
        DEFAULT_CHECKPOINT="outputs/competition_stage2_stem4500_seed2025/model_best.pth"
        USE_DENSE_TTA="True"
        ;;
    topowheat_global)
        CONFIG_FILE="configs/gwfss/experiments/competition_topowheat_bazr_train.yaml"
        DEFAULT_CHECKPOINT="outputs/competition_topowheat_seed2025/model_best.pth"
        USE_DENSE_TTA="False"
        ;;
    topowheat_bazr)
        CONFIG_FILE="configs/gwfss/experiments/competition_topowheat_bazr.yaml"
        DEFAULT_CHECKPOINT="outputs/competition_topowheat_seed2025/model_best.pth"
        USE_DENSE_TTA="False"
        ;;
    *)
        echo "Unknown METHOD: $METHOD" >&2
        echo "Use project_global, project_zoom, topowheat_global, or topowheat_bazr." >&2
        exit 2
        ;;
esac

CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_${METHOD}_${SPLIT}}"

command=(
    python3 -W ignore train_net.py
    --eval-only
    --config-file "$CONFIG_FILE"
    --num-gpus "$NUM_GPUS"
    --num-machines 1
    MODEL.WEIGHTS "$CHECKPOINT"
    DATASETS.TEST "('$DATASET',)"
    SSL.EVAL_WHO "$EVAL_WHO"
    TEST.AUG.ENABLED "$USE_DENSE_TTA"
    OUTPUT_DIR "$OUTPUT_DIR"
)

cd "$ROOT_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR/GWFSS/gwfss_competition_${SPLIT}/class_id" ]]; then
    echo "Ground-truth directory not found for split: $SPLIT" >&2
    exit 1
fi

exec "${command[@]}"
