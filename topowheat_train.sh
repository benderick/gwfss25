#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_FILE="${CONFIG_FILE:-configs/gwfss/experiments/competition_topowheat_bazr_train.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/competition_topowheat_seed2025}"
export TEACHER_CKPT="${TEACHER_CKPT:-outputs/competition_sapa_seed2025/model_best.pth}"

exec bash "$ROOT_DIR/stage2_train.sh"
