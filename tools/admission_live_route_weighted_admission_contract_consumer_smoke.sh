#!/usr/bin/env bash
# admission_live_route_weighted_admission_contract_consumer_smoke.sh - produce and consume the weighted admission contract.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-contract-consumer.XXXXXX")}"
CONTRACT_REPORT="$WORKDIR/live_route_weighted_admission_contract.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_contract_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_contract_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-contract-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 160 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_REPORT="$CONTRACT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_contract_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission contract producer failed"
fi

[[ -s "$CONTRACT_REPORT" ]] || die "weighted admission contract report not written: $CONTRACT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_contract_assert.sh" "$CONTRACT_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission contract assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-contract-consumer-smoke] pass: contract_report=$CONTRACT_REPORT"
