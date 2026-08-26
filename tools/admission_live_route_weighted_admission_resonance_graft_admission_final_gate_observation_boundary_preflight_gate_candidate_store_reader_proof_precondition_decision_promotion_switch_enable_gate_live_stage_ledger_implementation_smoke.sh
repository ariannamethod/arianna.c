#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_smoke.sh - block weighted Resonance graft admission ledger implementation behind blocked admission ledger.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_IMPLEMENTATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsli.XXXXXX")}"
LEDGER_WORKDIR="$WORKDIR/ledger"
GRAFT_ADMISSION_LEDGER_REPORT="$WORKDIR/ledger.json"
GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_IMPLEMENTATION_REPORT:-$WORKDIR/impl.json}"
ADMISSION_LEDGER_LOG="$WORKDIR/ledger.log"
LEDGER_IMPLEMENTATION_LOG="$WORKDIR/impl.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-smoke] FAIL: $*" >&2
    if [[ -f "$ADMISSION_LEDGER_LOG" ]]; then
        tail -n 500 "$ADMISSION_LEDGER_LOG" >&2 || true
    fi
    if [[ -f "$LEDGER_IMPLEMENTATION_LOG" ]]; then
        tail -n 240 "$LEDGER_IMPLEMENTATION_LOG" >&2 || true
    fi
    exit 1
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        die "$label missing in $file"
    fi
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_WORKDIR="$LEDGER_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_REPORT="$GRAFT_ADMISSION_LEDGER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_smoke.sh" >"$ADMISSION_LEDGER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger report not written: $GRAFT_ADMISSION_LEDGER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation.sh" "$GRAFT_ADMISSION_LEDGER_REPORT" "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" >"$LEDGER_IMPLEMENTATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation rejected ledger report"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation report not written: $GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation.v1"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation target kind"
require_grep '"target_mode": "closed_ledger_implementation_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation action"
require_grep '"writer_action": "reject_blocked_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger append flag"
require_grep '"ledger_implementation_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation state"
require_grep '"ledger_implementation_action": "reject_blocked_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation action field"
require_grep '"ledger_implementation_target": "admission_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation target"
require_grep '"ledger_implementation_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation target kind field"
require_grep '"ledger_implementation_target_mode": "closed_append_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation target mode field"
require_grep '"ledger_implementation_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation entrypoint"
require_grep '"ledger_implementation_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation receipt shape"
require_grep '"ledger_implementation_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation write scope"
require_grep '"ledger_implementation_append_only": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation append flag"
require_grep '"ledger_implementation_dry_run_only": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation dry-run flag"
require_grep '"ledger_implementation_receipt_persisted": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation receipt persisted flag"
require_grep '"ledger_implementation_ready": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_ready": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "weighted ledger implementation ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_consumed": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_required": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_implementation_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-id-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-causal-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation causal id"
require_grep '"ledger_implementation_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation hash"
require_grep '"ledger_implementation_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-read-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger.v1"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger status"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-id-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_ready": true' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger ready"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-causal-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger hash"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-read-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger read-back"
require_grep '"source_admission_ledger_receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_receipt"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger receipt shape"
require_grep '"source_admission_ledger_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger kind"
require_grep '"source_admission_ledger_mode": "closed_writer_contract_ledger_guard"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger mode"
require_grep '"source_admission_ledger_stage": "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_pre_ledger_append"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger stage"
require_grep '"source_admission_ledger_ledger_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger state"
require_grep '"source_admission_ledger_ledger_action": "reject_blocked_writer_contract"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger action"
require_grep '"source_admission_ledger_ledger_contract": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger contract"
require_grep '"source_admission_ledger_ledger_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger entrypoint"
require_grep '"source_admission_ledger_ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger receipt shape"
require_grep '"source_admission_ledger_ledger_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger write scope"
require_grep '"source_admission_ledger_ledger_ready": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger ready flag"
require_grep '"source_admission_ledger_ledger_append_allowed": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger append flag"
require_grep '"source_writer_contract_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract.v1"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source writer contract schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-id-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source writer contract id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source writer contract hash"
require_grep '"source_admission_ledger_reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger blocked by blocked writer contract; ledger receipt append remains closed"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "source ledger reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "body mutation guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger implementation blocked by blocked admission ledger; implementation append contract remains closed"' "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "ledger implementation reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation\] pass:' "$LEDGER_IMPLEMENTATION_LOG" "ledger implementation pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-smoke] pass: resonance_graft_admission_ledger_report=$GRAFT_ADMISSION_LEDGER_REPORT resonance_graft_admission_ledger_implementation_report=$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"
