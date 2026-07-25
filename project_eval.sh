#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-global}"

case "$MODE" in
    global)
        export METHOD="project_global"
        ;;
    zoom)
        export METHOD="project_zoom"
        ;;
    *)
        echo "MODE must be 'global' or 'zoom', got: $MODE" >&2
        exit 2
        ;;
esac

exec bash "$ROOT_DIR/evaluate.sh"
