#!/usr/bin/env bash
# admission_live_route_weighted_readiness_consumer_smoke.sh - produce and consume the weighted pre-live receipt.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-readiness-consumer.XXXXXX")}"
READINESS_REPORT="$WORKDIR/live_route_weighted_readiness.json"
RUN_LOG="$WORKDIR/weighted_readiness.log"
ASSERT_LOG="$WORKDIR/weighted_readiness_assert.log"

die() {
    echo "[admission-live-route-weighted-readiness-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 500 "$RUN_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 120 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_REPORT="$READINESS_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_readiness_smoke.sh" >"$RUN_LOG" 2>&1; then
    die "weighted readiness producer failed"
fi

[[ -s "$READINESS_REPORT" ]] || die "weighted readiness report not written: $READINESS_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_readiness_assert.sh" "$READINESS_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted readiness consumer rejected producer report"
fi

grep -q '\[admission-live-route-weighted-readiness-smoke\] pass:' "$RUN_LOG" || die "producer pass line missing"

echo "[admission-live-route-weighted-readiness-consumer-smoke] pass: readiness_report=$READINESS_REPORT"
