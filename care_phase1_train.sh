#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-care}"
NUM_GPUS="${NUM_GPUS:-1}"
CHECKPOINT="${CHECKPOINT:-outputs/competition_baseline/model_best.pth}"
BANK_DIR="${BANK_DIR:-outputs/care_phase1_bank}"
RESUME="${RESUME:-0}"

cd "$ROOT_DIR"

case "$MODE" in
    c0|control)
        CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_care_phase1_c0.yaml}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/care_phase1_c0}"
        ;;
    care|c1)
        CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_care_phase1.yaml}"
        OUTPUT_DIR="${OUTPUT_DIR:-outputs/care_phase1_c1}"
        for path in "$BANK_DIR/manifest.json" "$BANK_DIR/feature_bank.npz"; do
            if [[ ! -f "$path" ]]; then
                echo "CARE bank artifact not found: $path" >&2
                exit 1
            fi
        done
        ;;
    *)
        echo "Unknown MODE '$MODE'; use MODE=c0 or MODE=care" >&2
        exit 2
        ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Stage-I checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

arguments=(
    --config-file "$CONFIG_FILE"
    --num-gpus "$NUM_GPUS"
    --num-machines 1
)
if [[ "$RESUME" == "1" ]]; then
    arguments+=(--resume)
fi

overrides=(
    MODEL.WEIGHTS "$CHECKPOINT"
    OUTPUT_DIR "$OUTPUT_DIR"
)
if [[ "$MODE" == "care" || "$MODE" == "c1" ]]; then
    overrides+=(MODEL.CARE.BANK_DIR "$BANK_DIR")
fi

python3 -W ignore train_net.py "${arguments[@]}" "${overrides[@]}" "$@"
