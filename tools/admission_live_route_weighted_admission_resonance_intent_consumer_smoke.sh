#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_intent_consumer_smoke.sh - produce and consume weighted Resonance intent.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-intent-consumer.XXXXXX")}"
RESONANCE_INTENT_REPORT="$WORKDIR/live_route_weighted_admission_resonance_intent.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_intent_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_intent_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-intent-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 160 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_REPORT="$RESONANCE_INTENT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_intent_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance intent producer failed"
fi

[[ -s "$RESONANCE_INTENT_REPORT" ]] || die "weighted admission resonance intent report not written: $RESONANCE_INTENT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_intent_assert.sh" "$RESONANCE_INTENT_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance intent assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-intent-consumer-smoke] pass: resonance_intent_report=$RESONANCE_INTENT_REPORT"
