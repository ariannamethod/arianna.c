#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_smoke.sh - block weighted Resonance graft admission permit behind blocked readiness.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_PERMIT_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsp.XXXXXX")}"
READINESS_WORKDIR="$WORKDIR/r"
GRAFT_ADMISSION_READINESS_REPORT="$WORKDIR/readiness.json"
GRAFT_ADMISSION_PERMIT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_PERMIT_REPORT:-$WORKDIR/permit.json}"
READINESS_LOG="$WORKDIR/readiness.log"
PERMIT_LOG="$WORKDIR/permit.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-smoke] FAIL: $*" >&2
    if [[ -f "$READINESS_LOG" ]]; then
        tail -n 500 "$READINESS_LOG" >&2 || true
    fi
    if [[ -f "$PERMIT_LOG" ]]; then
        tail -n 240 "$PERMIT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_READINESS_WORKDIR="$READINESS_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_READINESS_REPORT="$GRAFT_ADMISSION_READINESS_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_smoke.sh" >"$READINESS_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage readiness producer failed"
fi

[[ -s "$GRAFT_ADMISSION_READINESS_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage readiness report not written: $GRAFT_ADMISSION_READINESS_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit.sh" "$GRAFT_ADMISSION_READINESS_REPORT" "$GRAFT_ADMISSION_PERMIT_REPORT" >"$PERMIT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit rejected readiness report"
fi

[[ -s "$GRAFT_ADMISSION_PERMIT_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit report not written: $GRAFT_ADMISSION_PERMIT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit.v1"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_blocked_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit target kind"
require_grep '"target_mode": "closed_permit_guard_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_blocked_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit action"
require_grep '"writer_action": "reject_blocked_admission_readiness"' "$GRAFT_ADMISSION_PERMIT_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_readiness"' "$GRAFT_ADMISSION_PERMIT_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_readiness"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "ledger append flag"
require_grep '"admission_permit_state": "blocked"' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit state"
require_grep '"admission_permit_action": "reject_blocked_admission_readiness"' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit action"
require_grep '"admission_permit_target": "live_admission"' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit target"
require_grep '"admission_permit_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness"' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit target kind"
require_grep '"admission_permit_target_mode": "closed_permit_guard_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit target mode"
require_grep '"admission_permit_dry_run_only": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit dry-run flag"
require_grep '"admission_permit_readiness_verified": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit readiness flag"
require_grep '"admission_permit_ledger_verified": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit ledger flag"
require_grep '"admission_permit_writer_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit writer flag"
require_grep '"admission_permit_rollback_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit rollback flag"
require_grep '"admission_permit_ledger_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit ledger ready flag"
require_grep '"admission_permit_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "admission permit ready flag"
require_grep '"manual_permit_requested": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "manual permit flag"
require_grep '"permit_key_matched": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit key flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "weighted permit ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_consumed": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "readiness consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_required": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "readiness required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-id-' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-causal-' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit causal id"
require_grep '"admission_permit_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit hash"
require_grep '"admission_permit_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-read-' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness.v1"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_blocked_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness status"
require_grep '"source_ledger_verification_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification.v1"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source ledger verification schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-id-' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_ready": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness ready"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-causal-' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness hash"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-read-' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness read-back"
require_grep '"source_admission_readiness_report_receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_receipt"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness report receipt shape"
require_grep '"source_admission_readiness_state": "blocked"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness state"
require_grep '"source_admission_readiness_action": "reject_blocked_ledger_verification"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness action"
require_grep '"source_admission_readiness_target": "live_admission"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness target"
require_grep '"source_admission_readiness_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_verification"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness target kind"
require_grep '"source_admission_readiness_target_mode": "closed_readiness_guard_dry_run"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness target mode"
require_grep '"source_admission_readiness_dry_run_only": true' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness dry-run flag"
require_grep '"source_admission_readiness_ledger_verified": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness ledger flag"
require_grep '"source_admission_readiness_writer_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness writer flag"
require_grep '"source_admission_readiness_rollback_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness rollback flag"
require_grep '"source_admission_readiness_ledger_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness ledger ready flag"
require_grep '"source_admission_readiness_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness ready flag"
require_grep '"source_admission_readiness_reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage readiness blocked by blocked ledger verification; live admission readiness remains closed"' "$GRAFT_ADMISSION_PERMIT_REPORT" "source readiness reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_PERMIT_REPORT" "body mutation guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_PERMIT_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit blocked by blocked readiness; manual permit remains closed"' "$GRAFT_ADMISSION_PERMIT_REPORT" "permit reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit\] pass:' "$PERMIT_LOG" "permit pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness_report=$GRAFT_ADMISSION_READINESS_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_report=$GRAFT_ADMISSION_PERMIT_REPORT"
