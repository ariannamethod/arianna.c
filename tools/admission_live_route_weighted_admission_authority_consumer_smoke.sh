#!/usr/bin/env bash
# admission_live_route_weighted_admission_authority_consumer_smoke.sh - produce and consume weighted admission authority.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-authority-consumer.XXXXXX")}"
AUTHORITY_REPORT="$WORKDIR/live_route_weighted_admission_authority.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_authority_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_authority_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-authority-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 160 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_REPORT="$AUTHORITY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_authority_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission authority producer failed"
fi

[[ -s "$AUTHORITY_REPORT" ]] || die "weighted admission authority report not written: $AUTHORITY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_authority_assert.sh" "$AUTHORITY_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission authority assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-authority-consumer-smoke] pass: authority_report=$AUTHORITY_REPORT"
