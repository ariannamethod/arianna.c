#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_smoke.sh - block weighted Resonance graft admission authority behind blocked permit.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_AUTHORITY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-authority.XXXXXX")}"
PERMIT_WORKDIR="$WORKDIR/permit"
GRAFT_ADMISSION_PERMIT_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit.json"
GRAFT_ADMISSION_AUTHORITY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_AUTHORITY_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority.json}"
PERMIT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit.log"
AUTHORITY_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-smoke] FAIL: $*" >&2
    if [[ -f "$PERMIT_LOG" ]]; then
        tail -n 500 "$PERMIT_LOG" >&2 || true
    fi
    if [[ -f "$AUTHORITY_LOG" ]]; then
        tail -n 240 "$AUTHORITY_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_PERMIT_WORKDIR="$PERMIT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_PERMIT_REPORT="$GRAFT_ADMISSION_PERMIT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_smoke.sh" >"$PERMIT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit producer failed"
fi

[[ -s "$GRAFT_ADMISSION_PERMIT_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit report not written: $GRAFT_ADMISSION_PERMIT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority.sh" "$GRAFT_ADMISSION_PERMIT_REPORT" "$GRAFT_ADMISSION_AUTHORITY_REPORT" >"$AUTHORITY_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage authority rejected permit report"
fi

[[ -s "$GRAFT_ADMISSION_AUTHORITY_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage authority report not written: $GRAFT_ADMISSION_AUTHORITY_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority.v1"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_blocked_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority target kind"
require_grep '"target_mode": "closed_authority_guard_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_blocked_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority action"
require_grep '"writer_action": "reject_blocked_admission_permit"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_permit"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_permit"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "ledger append flag"
require_grep '"admission_authority_state": "blocked"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority state"
require_grep '"admission_authority_action": "reject_blocked_admission_permit"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority action"
require_grep '"admission_authority_target": "live_admission_authority"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority target"
require_grep '"admission_authority_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority target kind"
require_grep '"admission_authority_target_mode": "closed_authority_guard_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority target mode"
require_grep '"admission_authority_dry_run_only": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority dry-run flag"
require_grep '"admission_authority_permit_verified": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority permit flag"
require_grep '"admission_authority_ledger_verified": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority ledger flag"
require_grep '"admission_authority_writer_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority writer flag"
require_grep '"admission_authority_rollback_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority rollback flag"
require_grep '"admission_authority_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority ready flag"
require_grep '"admission_authority_granted": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "admission authority grant flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_ready": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "weighted authority ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_consumed": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "permit consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_required": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "permit required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-id-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-causal-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority causal id"
require_grep '"admission_authority_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority hash"
require_grep '"admission_authority_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-read-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit.v1"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_blocked_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit status"
require_grep '"source_admission_readiness_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness.v1"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source readiness schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-id-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_ready": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit ready"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-causal-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit hash"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-read-' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit read-back"
require_grep '"source_admission_permit_report_receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_receipt"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit report receipt shape"
require_grep '"source_admission_permit_state": "blocked"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit state"
require_grep '"source_admission_permit_action": "reject_blocked_admission_readiness"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit action"
require_grep '"source_admission_permit_target": "live_admission"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit target"
require_grep '"source_admission_permit_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_readiness"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit target kind"
require_grep '"source_admission_permit_target_mode": "closed_permit_guard_dry_run"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit target mode"
require_grep '"source_admission_permit_dry_run_only": true' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit dry-run flag"
require_grep '"source_admission_permit_readiness_verified": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit readiness flag"
require_grep '"source_admission_permit_ledger_verified": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit ledger flag"
require_grep '"source_admission_permit_writer_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit writer flag"
require_grep '"source_admission_permit_rollback_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit rollback flag"
require_grep '"source_admission_permit_ledger_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit ledger ready flag"
require_grep '"source_admission_permit_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit ready flag"
require_grep '"source_manual_permit_requested": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source manual permit flag"
require_grep '"source_permit_key_matched": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit key flag"
require_grep '"source_admission_permit_reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage permit blocked by blocked readiness; manual permit remains closed"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "source permit reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "base authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage authority blocked by blocked permit; live authority remains closed"' "$GRAFT_ADMISSION_AUTHORITY_REPORT" "authority reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority\] pass:' "$AUTHORITY_LOG" "authority pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_permit_report=$GRAFT_ADMISSION_PERMIT_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_authority_report=$GRAFT_ADMISSION_AUTHORITY_REPORT"
