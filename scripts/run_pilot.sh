#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${GRACE_PYTHON:?Set GRACE_PYTHON}
GRACE=${GRACE_ROOT:?Set GRACE_ROOT}
VIDEOS=${GRACE_VIDEOS:?Set GRACE_VIDEOS}
for Q in 1 2; do
  if [ ! -f "cache/q${Q}/manifest.json" ]; then
    "$PYTHON" -m grace_rtp.experiment prepare --grace "$GRACE" \
      --checkpoint "$GRACE/models/grace/4096_freeze.model" \
      --videos-dir "$VIDEOS" --videos 12 --frames 16 --q "$Q" --scale 1 \
      --cache "cache/q${Q}" > "results/prepare_q${Q}.log" 2>&1
  fi
  "$PYTHON" -m grace_rtp.experiment run --grace "$GRACE" \
    --checkpoint "$GRACE/models/grace/4096_freeze.model" --cache "cache/q${Q}" \
    --seeds 3 --trace --output "results/pilot_q${Q}" > "results/run_q${Q}.log" 2>&1
done
for LOSS in 0.2 0.5; do
  "$PYTHON" -m grace_rtp.experiment run --grace "$GRACE" \
    --checkpoint "$GRACE/models/grace/4096_freeze.model" --cache cache/q1 \
    --seeds 3 --fixed-loss "$LOSS" --profiles configs/controlled.yaml \
    --output "results/controlled_${LOSS}" > "results/controlled_${LOSS}.log" 2>&1
done
