#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_baseline.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/competition_baseline/model_best.pth}"
PHASE0_DIR="${PHASE0_DIR:-outputs/care_phase0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/care_phase1_bank}"
DEVICE="${DEVICE:-cuda}"
FEATURE_NAME="${FEATURE_NAME:-res2}"

cd "$ROOT_DIR"
for path in \
    "$CHECKPOINT" \
    "$PHASE0_DIR/summary.json" \
    "$PHASE0_DIR/anchor_support.csv" \
    "$PHASE0_DIR/donor_matches.csv"; do
    if [[ ! -f "$path" ]]; then
        echo "Required CARE artifact not found: $path" >&2
        exit 1
    fi
done

python3 -W ignore tools/prepare_care_phase1.py \
    --config-file "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-branch auto \
    --phase0-dir "$PHASE0_DIR" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --feature-name "$FEATURE_NAME" \
    "$@"
