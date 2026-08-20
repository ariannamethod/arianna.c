#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_ledger_implementation_consumer_smoke.sh - produce and consume weighted Resonance graft admission ledger implementation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-consumer.XXXXXX")}"
GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_implementation.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_implementation_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_implementation_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 220 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT="$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_implementation_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger implementation producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" ]] || die "weighted admission resonance graft admission ledger implementation report not written: $GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_implementation_assert.sh" "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger implementation assert rejected producer report"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-consumer-smoke] pass: resonance_graft_admission_ledger_implementation_report=$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"
