#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight_consumer_smoke.sh - produce and consume weighted Resonance graft admission writer preflight.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-writer-preflight-consumer.XXXXXX")}"
GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_writer_preflight.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_writer_preflight_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_writer_preflight_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 220 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT="$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft admission writer preflight producer failed"
fi

[[ -s "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft admission writer preflight report not written: $GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight_assert.sh" "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission writer preflight assert rejected producer report"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-consumer-smoke] pass: resonance_graft_admission_writer_preflight_report=$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT"
