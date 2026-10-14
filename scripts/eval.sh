#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT GPU_ID [TEST_OPTIONS...]" >&2
  exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
GPU_ID="$3"
shift 3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec "$PYTHON_BIN" tools/test.py "$CONFIG" "$CHECKPOINT" --eval bbox "$@"
