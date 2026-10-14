#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 GPU_ID RUN_ROOT CONFIG [CONFIG...]" >&2
  exit 2
fi

GPU_ID="$1"
RUN_ROOT="$2"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
cd "$ROOT"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a single physical GPU index." >&2
  exit 2
fi
for config in "$@"; do
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
done
command -v flock >/dev/null || { echo "flock is required." >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required." >&2; exit 2; }
nvidia-smi -i "$GPU_ID" --query-gpu=index --format=csv,noheader >/dev/null

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
STATUS="$RUN_ROOT/queue_status.tsv"
if [[ ! -f "$STATUS" ]]; then
  printf 'timestamp\texperiment\tstatus\texit_code\n' > "$STATUS"
fi
record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" "$3" >> "$STATUS"
}

exec 9>"/tmp/fsta_det_gpu_${GPU_ID}.lock"
record QUEUE waiting_for_lock -
flock 9

for config in "$@"; do
  name="$(basename "$config" .py)"
  work_dir="$RUN_ROOT/$name"
  if [[ -f "$work_dir/COMPLETED" ]]; then
    record "$name" skipped_completed 0
    continue
  fi
  if [[ -d "$work_dir" && -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing a non-empty run directory: $work_dir" >&2
    record "$name" existing_directory 2
    exit 2
  fi

  record "$name" waiting_for_gpu -
  while true; do
    if ! pids="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
      record "$name" gpu_query_failed 3
      exit 3
    fi
    [[ -z "${pids//[[:space:]]/}" ]] && break
    echo "GPU $GPU_ID is busy; waiting before $name."
    sleep "$WAIT_SECONDS"
  done

  mkdir -p "$work_dir"
  record "$name" training -
  set +e
  bash scripts/train.sh "$config" "$GPU_ID" "$work_dir" \
    2>&1 | tee "$work_dir/train.console.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    record "$name" failed "$rc"
    exit "$rc"
  fi
  touch "$work_dir/COMPLETED"
  record "$name" completed 0
done

record QUEUE completed 0
