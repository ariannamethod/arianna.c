#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_consumer_smoke.sh - produce and consume weighted Resonance graft admission final-gate observation boundary.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-consumer.XXXXXX")}"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 260 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT="$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary producer failed"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_assert.sh" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary assert rejected producer report"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-consumer-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT"
