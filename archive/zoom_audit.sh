#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_baseline.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/competition_baseline/model_best.pth}"
CHECKPOINT_BRANCH="${CHECKPOINT_BRANCH:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/zoom_audit_stage1}"
DATASET="${DATASET:-gwfss_sem_seg_val}"
DEVICE="${DEVICE:-cuda}"
MIN_SIZE="${MIN_SIZE:-512}"
MAX_SIZE="${MAX_SIZE:-2048}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
WINDOW_STRIDE="${WINDOW_STRIDE:-128}"
ZOOM_SIZE="${ZOOM_SIZE:-512}"
DENSE_SHORT_EDGE="${DENSE_SHORT_EDGE:-768}"
BUDGETS="${BUDGETS:-1,2,4,8}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.3}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-2000}"
MAX_IMAGES="${MAX_IMAGES:-0}"
VISUALIZE_BEST="${VISUALIZE_BEST:-10}"
EXPECTED_BASELINE_MIOU="${EXPECTED_BASELINE_MIOU:-0.7310942}"
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-0.001}"
MINIMUM_DENSE_SCALE_GAIN="${MINIMUM_DENSE_SCALE_GAIN:-0.005}"
MINIMUM_ORACLE_GAIN="${MINIMUM_ORACLE_GAIN:-0.005}"
MINIMUM_RECOVERY="${MINIMUM_RECOVERY:-0.5}"
MINIMUM_SELECTOR_GAIN="${MINIMUM_SELECTOR_GAIN:-0.001}"
DRY_RUN="${DRY_RUN:-0}"

command=(
    python3 -W ignore tools/audit_zoom.py
    --config-file "$CONFIG_FILE"
    --checkpoint "$CHECKPOINT"
    --checkpoint-branch "$CHECKPOINT_BRANCH"
    --dataset "$DATASET"
    --device "$DEVICE"
    --min-size "$MIN_SIZE"
    --max-size "$MAX_SIZE"
    --window-size "$WINDOW_SIZE"
    --window-stride "$WINDOW_STRIDE"
    --zoom-size "$ZOOM_SIZE"
    --dense-short-edge "$DENSE_SHORT_EDGE"
    --budgets "$BUDGETS"
    --nms-threshold "$NMS_THRESHOLD"
    --random-repeats "$RANDOM_REPEATS"
    --bootstrap-repeats "$BOOTSTRAP_REPEATS"
    --max-images "$MAX_IMAGES"
    --visualize-best "$VISUALIZE_BEST"
    --expected-baseline-miou "$EXPECTED_BASELINE_MIOU"
    --baseline-tolerance "$BASELINE_TOLERANCE"
    --minimum-dense-scale-gain "$MINIMUM_DENSE_SCALE_GAIN"
    --minimum-oracle-gain "$MINIMUM_ORACLE_GAIN"
    --minimum-recovery "$MINIMUM_RECOVERY"
    --minimum-selector-gain "$MINIMUM_SELECTOR_GAIN"
    --output "$OUTPUT_DIR"
)
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
