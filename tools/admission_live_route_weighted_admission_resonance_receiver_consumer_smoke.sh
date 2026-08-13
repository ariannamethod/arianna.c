#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_receiver_consumer_smoke.sh - produce and consume weighted Resonance receiver.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-receiver-consumer.XXXXXX")}"
RECEIVER_REPORT="$WORKDIR/live_route_weighted_admission_resonance_receiver.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_receiver_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_receiver_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-receiver-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 180 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_REPORT="$RECEIVER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_receiver_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance receiver producer failed"
fi

[[ -s "$RECEIVER_REPORT" ]] || die "weighted admission resonance receiver report not written: $RECEIVER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_receiver_assert.sh" "$RECEIVER_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance receiver assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-receiver-consumer-smoke] pass: resonance_receiver_report=$RECEIVER_REPORT"
