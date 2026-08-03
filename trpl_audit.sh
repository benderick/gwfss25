#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_topowheat_trpl.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/competition_baseline_seed2025/model_best.pth}"
CHECKPOINT_BRANCH="${CHECKPOINT_BRANCH:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/trpl_audit_stage1}"
DATASET="${DATASET:-gwfss_sem_seg_val}"
DEVICE="${DEVICE:-cuda}"
THRESHOLDS="${THRESHOLDS:-0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95}"
MAX_IMAGES="${MAX_IMAGES:-0}"
VISUALIZE_WORST="${VISUALIZE_WORST:-20}"
DRY_RUN="${DRY_RUN:-0}"

command=(
    python3 -W ignore tools/audit_trpl.py
    --config-file "$CONFIG_FILE"
    --checkpoint "$CHECKPOINT"
    --checkpoint-branch "$CHECKPOINT_BRANCH"
    --dataset "$DATASET"
    --device "$DEVICE"
    --thresholds "$THRESHOLDS"
    --max-images "$MAX_IMAGES"
    --visualize-worst "$VISUALIZE_WORST"
    --output "$OUTPUT_DIR"
)

if [[ -n "${RELIABILITY_THRESHOLD:-}" ]]; then
    command+=(--reliability-threshold "$RELIABILITY_THRESHOLD")
fi
if [[ -n "${VIEW_SCALE:-}" ]]; then
    command+=(--view-scale "$VIEW_SCALE")
fi
if [[ -n "${MIN_SIZE:-}" ]]; then
    command+=(--min-size "$MIN_SIZE")
fi
if [[ -n "${MAX_SIZE:-}" ]]; then
    command+=(--max-size "$MAX_SIZE")
fi
command+=("$@")

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
DATASET_ROOT="${DETECTRON2_DATASETS%/}"
if [[ ! -d "$DATASET_ROOT/GWFSS/gwfss_competition_val/class_id" ]]; then
    echo "Validation ground truth not found under $DATASET_ROOT/GWFSS" >&2
    exit 1
fi

exec "${command[@]}"
