#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_sapa.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/competition_sapa_seed2025}"

cd "$ROOT_DIR"
python3 -W ignore train_net.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus "$NUM_GPUS" \
    --num-machines 1 \
    OUTPUT_DIR "$OUTPUT_DIR"
