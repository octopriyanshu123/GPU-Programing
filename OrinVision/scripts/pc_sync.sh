#!/usr/bin/env bash
# =====================================================================
# pc_sync.sh - Run this on your PC (Linux / macOS / Windows Git Bash/WSL)
# =====================================================================
# Moves files between your PC and the Jetson over SSH.
#
#   ./scripts/pc_sync.sh push      # copy project code  PC -> Jetson
#   ./scripts/pc_sync.sh pull      # copy results       Jetson -> PC
#   ./scripts/pc_sync.sh run ARGS  # run orin_vision on the Jetson remotely
#
# Configure once (or edit the defaults below):
#   export JETSON=octo@192.168.1.50
#   export JETSON_DIR=jetson_gpu_programing/orin_vision
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

JETSON="${JETSON:-octo@jetson.local}"
JETSON_DIR="${JETSON_DIR:-jetson_gpu_programing/orin_vision}"

usage() { sed -n '2,15p' "$0"; exit 1; }
[[ $# -ge 1 ]] || usage

case "$1" in
    push)
        echo "==> Pushing code to $JETSON:$JETSON_DIR"
        ssh "$JETSON" "mkdir -p '$JETSON_DIR'"
        if command -v rsync >/dev/null; then
            rsync -avz --exclude build/ --exclude build-debug/ --exclude results/ \
                --exclude .git/ ./ "$JETSON:$JETSON_DIR/"
        else
            scp -r CMakeLists.txt README.md CLAUDE.md include src tests scripts \
                "$JETSON:$JETSON_DIR/"
        fi
        ;;
    pull)
        echo "==> Pulling results from $JETSON:$JETSON_DIR/results"
        mkdir -p results
        if command -v rsync >/dev/null; then
            rsync -avz "$JETSON:$JETSON_DIR/results/" ./results/
        else
            scp -r "$JETSON:$JETSON_DIR/results/." ./results/
        fi
        echo "Done. Open the .png files in ./results/"
        ;;
    run)
        shift
        # -t gives a real terminal so output appears live
        ssh -t "$JETSON" "cd '$JETSON_DIR' && ./build/orin_vision $*"
        ;;
    *)
        usage
        ;;
esac
