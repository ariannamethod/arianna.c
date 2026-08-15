#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_candidate_consumer_smoke.sh - produce and consume weighted Resonance graft candidate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-candidate-consumer.XXXXXX")}"
GRAFT_CANDIDATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_candidate.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-candidate-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 220 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_REPORT="$GRAFT_CANDIDATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft candidate producer failed"
fi

[[ -s "$GRAFT_CANDIDATE_REPORT" ]] || die "weighted admission resonance graft candidate report not written: $GRAFT_CANDIDATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_assert.sh" "$GRAFT_CANDIDATE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft candidate assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-candidate-consumer-smoke] pass: resonance_graft_candidate_report=$GRAFT_CANDIDATE_REPORT"
