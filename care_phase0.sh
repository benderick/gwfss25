#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_baseline.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/competition_baseline/model_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/care_phase0}"
ANCHOR_DATASET="${ANCHOR_DATASET:-gwfss_sem_seg_train}"
VALIDATION_DATASET="${VALIDATION_DATASET:-gwfss_sem_seg_val}"
DONOR_DATASET="${DONOR_DATASET:-gwfss_unlabel_random4500_seed2025}"
CHECKPOINT_BRANCH="${CHECKPOINT_BRANCH:-auto}"
DEVICE="${DEVICE:-cuda}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"
MAX_VALIDATION="${MAX_VALIDATION:-0}"
MAX_DONORS="${MAX_DONORS:-0}"
VISUALIZE="${VISUALIZE:-8}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
EXPECTED_VALIDATION_MIOU="${EXPECTED_VALIDATION_MIOU:-73.10942}"
PARITY_TOLERANCE="${PARITY_TOLERANCE:-0.001}"
RECOMPUTE_CACHE="${RECOMPUTE_CACHE:-0}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing config: $CONFIG_FILE" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi

ARGS=(
    python3 -W ignore tools/audit_care_phase0.py
    --config-file "$CONFIG_FILE"
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT_DIR"
    --anchor-dataset "$ANCHOR_DATASET"
    --validation-dataset "$VALIDATION_DATASET"
    --donor-dataset "$DONOR_DATASET"
    --checkpoint-branch "$CHECKPOINT_BRANCH"
    --device "$DEVICE"
    --max-anchors "$MAX_ANCHORS"
    --max-validation "$MAX_VALIDATION"
    --max-donors "$MAX_DONORS"
    --visualize "$VISUALIZE"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --expected-validation-miou "$EXPECTED_VALIDATION_MIOU"
    --parity-tolerance "$PARITY_TOLERANCE"
)

if [[ "$RECOMPUTE_CACHE" == "1" ]]; then
    ARGS+=(--recompute-cache)
fi
if [[ -n "${MIN_SIZE:-}" ]]; then
    ARGS+=(--min-size "$MIN_SIZE")
fi
if [[ -n "${MAX_SIZE:-}" ]]; then
    ARGS+=(--max-size "$MAX_SIZE")
fi

"${ARGS[@]}" "$@"
