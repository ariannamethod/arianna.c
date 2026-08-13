#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_boundary_consumer_smoke.sh - produce and consume weighted Resonance graft boundary.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-boundary-consumer.XXXXXX")}"
GRAFT_BOUNDARY_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_boundary.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_boundary_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_boundary_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-boundary-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 180 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_REPORT="$GRAFT_BOUNDARY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_boundary_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft boundary producer failed"
fi

[[ -s "$GRAFT_BOUNDARY_REPORT" ]] || die "weighted admission resonance graft boundary report not written: $GRAFT_BOUNDARY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_boundary_assert.sh" "$GRAFT_BOUNDARY_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft boundary assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-boundary-consumer-smoke] pass: resonance_graft_boundary_report=$GRAFT_BOUNDARY_REPORT"
