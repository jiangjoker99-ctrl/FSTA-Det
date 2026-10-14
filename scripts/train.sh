#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG GPU_ID WORK_DIR [TRAIN_OPTIONS...]" >&2
  exit 2
fi

CONFIG="$1"
GPU_ID="$2"
WORK_DIR="$3"
shift 3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec "$PYTHON_BIN" tools/train.py "$CONFIG" \
  --gpu-id 0 --work-dir "$WORK_DIR" --seed 42 "$@"
