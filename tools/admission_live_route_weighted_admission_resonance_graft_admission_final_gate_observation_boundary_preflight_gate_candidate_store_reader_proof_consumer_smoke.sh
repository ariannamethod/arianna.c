#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_consumer_smoke.sh - produce and consume blocked weighted Resonance admission final-gate observation-boundary preflight-gate candidate store reader proof.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-consumer.XXXXXX")}"
PROOF_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.json"
PRODUCER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_producer.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 260 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT="$PROOF_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof producer failed"
fi

[[ -s "$PROOF_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof report not written: $PROOF_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_assert.sh" "$PROOF_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof assert rejected producer report"
fi

if [[ -s "$WORKDIR/unexpected_state_mutation" ]]; then
    die "unexpected mutation sentinel exists"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-consumer-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=$PROOF_REPORT"
