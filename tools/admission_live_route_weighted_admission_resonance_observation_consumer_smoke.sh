#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_observation_consumer_smoke.sh - produce and consume weighted Resonance observation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-observation-consumer.XXXXXX")}"
OBSERVATION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_observation.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_observation_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_observation_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-observation-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 180 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_REPORT="$OBSERVATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_observation_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance observation producer failed"
fi

[[ -s "$OBSERVATION_REPORT" ]] || die "weighted admission resonance observation report not written: $OBSERVATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_observation_assert.sh" "$OBSERVATION_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance observation assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-observation-consumer-smoke] pass: resonance_observation_report=$OBSERVATION_REPORT"
