#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-global}"

case "$MODE" in
    global)
        export METHOD="topowheat_global"
        ;;
    bazr)
        export METHOD="topowheat_bazr"
        ;;
    *)
        echo "MODE must be 'global' or 'bazr', got: $MODE" >&2
        exit 2
        ;;
esac

exec bash "$ROOT_DIR/evaluate.sh"
