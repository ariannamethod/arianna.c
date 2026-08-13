#!/usr/bin/env bash
# admission_live_route_weighted_admission_final_gate_consumer_smoke.sh - produce and consume weighted admission final gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-final-gate-consumer.XXXXXX")}"
FINAL_GATE_REPORT="$WORKDIR/live_route_weighted_admission_final_gate.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_final_gate_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_final_gate_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-final-gate-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 160 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_REPORT="$FINAL_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_final_gate_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission final gate producer failed"
fi

[[ -s "$FINAL_GATE_REPORT" ]] || die "weighted admission final gate report not written: $FINAL_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_final_gate_assert.sh" "$FINAL_GATE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission final gate assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-final-gate-consumer-smoke] pass: final_gate_report=$FINAL_GATE_REPORT"
