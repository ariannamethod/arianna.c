#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_gate_consumer_smoke.sh - produce and consume weighted Resonance graft gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_GATE_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-gate-consumer.XXXXXX")}"
GRAFT_GATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_gate.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_gate_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_gate_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-gate-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 220 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_GATE_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_GATE_REPORT="$GRAFT_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_gate_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft gate producer failed"
fi

[[ -s "$GRAFT_GATE_REPORT" ]] || die "weighted admission resonance graft gate report not written: $GRAFT_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_gate_assert.sh" "$GRAFT_GATE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft gate assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-gate-consumer-smoke] pass: resonance_graft_gate_report=$GRAFT_GATE_REPORT"
