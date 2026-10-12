#!/usr/bin/env bash
# Run as root. The namespace is unique; existing host qdiscs are never touched.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${GRACE_PYTHON:?Set GRACE_PYTHON to the Python interpreter}
TESTBED=${TESTBED_ROOT:?Set TESTBED_ROOT to independently cloned 6G-Testbed}
GRACE=${GRACE_ROOT:?Set GRACE_ROOT to independently obtained Grace}
NS="grace-rtp-$$"
PREFIX=${GRACE_RUN_PREFIX:-rtc_netem}
cleanup() {
  if [ -n "${CAPTURE_PID:-}" ]; then kill -INT "$CAPTURE_PID" 2>/dev/null || true; wait "$CAPTURE_PID" 2>/dev/null || true; fi
  ip netns del "$NS" 2>/dev/null || true
}
trap cleanup EXIT
ip netns add "$NS"
ip -n "$NS" link set lo up
cd "$ROOT"
for PROFILE in no_emulation lossy 5qi_7 congested; do
  ip netns exec "$NS" tcpdump -i lo -U -w "$ROOT/results/${PREFIX}_${PROFILE}.pcap" udp > /dev/null 2>&1 &
  CAPTURE_PID=$!
  ip netns exec "$NS" "$PYTHON" -m grace_rtp.testbed --cache cache/q1 \
    --grace "$GRACE" --checkpoint "$GRACE/models/grace/4096_freeze.model" \
    --testbed "$TESTBED" --profile "$PROFILE" --isolated-loopback \
    --output "results/${PREFIX}_${PROFILE}.json"
  kill -INT "$CAPTURE_PID"
  wait "$CAPTURE_PID" || true
  CAPTURE_PID=
done
ip netns exec "$NS" tc -j qdisc show dev lo > results/netem_cleanup.json
