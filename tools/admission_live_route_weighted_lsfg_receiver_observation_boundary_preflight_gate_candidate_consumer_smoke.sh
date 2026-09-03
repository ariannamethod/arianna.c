#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_consumer_smoke.sh - produce and consume weighted Resonance graft admission final-gate receiver observation boundary preflight gate candidate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_CONSUMER_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsfgrobpgc-c.XXXXXX")}"
CANDIDATE_REPORT="$WORKDIR/c.json"
PRODUCER_LOG="$WORKDIR/prod.log"
ASSERT_LOG="$WORKDIR/assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-consumer-smoke] FAIL: $*" >&2
    if [[ -f "$PRODUCER_LOG" ]]; then
        tail -n 500 "$PRODUCER_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 260 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT="$CANDIDATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_smoke.sh" >"$PRODUCER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate receiver observation boundary preflight gate candidate producer failed"
fi

[[ -s "$CANDIDATE_REPORT" ]] || die "weighted admission resonance graft admission final gate receiver observation boundary preflight gate candidate report not written: $CANDIDATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_assert.sh" "$CANDIDATE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate receiver observation boundary preflight gate candidate assert rejected producer report"
fi

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-consumer-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_report=$CANDIDATE_REPORT"
