#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_stage2_stem4500.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/competition_stage2_stem4500_seed2025}"
TEACHER_CKPT="${TEACHER_CKPT:-outputs/competition_sapa_seed2025/model_best.pth}"

cd "$ROOT_DIR"
if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "Teacher checkpoint not found: $TEACHER_CKPT" >&2
    exit 1
fi

python3 -W ignore train_net.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus "$NUM_GPUS" \
    --num-machines 1 \
    SSL.TEACHER_CKPT "$TEACHER_CKPT" \
    OUTPUT_DIR "$OUTPUT_DIR"
