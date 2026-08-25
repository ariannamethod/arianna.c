#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_smoke.sh - produce and validate blocked writer inventory from blocked live-stage writer preflight.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_INVENTORY_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-wi.XXXXXX")}"
WRITER_PREFLIGHT_WORKDIR="$WORKDIR/wpdir"
WRITER_PREFLIGHT_REPORT="$WORKDIR/wp.json"
WRITER_INVENTORY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_INVENTORY_REPORT:-$WORKDIR/wi.json}"
WRITER_PREFLIGHT_LOG="$WORKDIR/wp.log"
WRITER_INVENTORY_LOG="$WORKDIR/wi.log"
ASSERT_LOG="$WORKDIR/assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-smoke] FAIL: $*" >&2
    if [[ -f "$WRITER_PREFLIGHT_LOG" ]]; then
        tail -n 500 "$WRITER_PREFLIGHT_LOG" >&2 || true
    fi
    if [[ -f "$WRITER_INVENTORY_LOG" ]]; then
        tail -n 260 "$WRITER_INVENTORY_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 260 "$ASSERT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_PREFLIGHT_WORKDIR="$WRITER_PREFLIGHT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_PREFLIGHT_REPORT="$WRITER_PREFLIGHT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_smoke.sh" >"$WRITER_PREFLIGHT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight producer failed"
fi

[[ -s "$WRITER_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight report not written: $WRITER_PREFLIGHT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory.sh" "$WRITER_PREFLIGHT_REPORT" "$WRITER_INVENTORY_REPORT" >"$WRITER_INVENTORY_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory rejected writer preflight report"
fi

[[ -s "$WRITER_INVENTORY_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory report not written: $WRITER_INVENTORY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_assert.sh" "$WRITER_INVENTORY_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory.v1"' "$WRITER_INVENTORY_REPORT" "writer inventory schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_blocked_dry_run"' "$WRITER_INVENTORY_REPORT" "writer inventory status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory"' "$WRITER_INVENTORY_REPORT" "writer inventory target kind"
require_grep '"target_mode": "closed_writer_inventory_guard_dry_run"' "$WRITER_INVENTORY_REPORT" "writer inventory target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run"' "$WRITER_INVENTORY_REPORT" "writer inventory action"
require_grep '"writer_state": "blocked"' "$WRITER_INVENTORY_REPORT" "writer state"
require_grep '"writer_action": "reject_blocked_writer_preflight"' "$WRITER_INVENTORY_REPORT" "writer action"
require_grep '"rollback_state": "blocked"' "$WRITER_INVENTORY_REPORT" "rollback state"
require_grep '"rollback_action": "reject_blocked_writer_preflight"' "$WRITER_INVENTORY_REPORT" "rollback action"
require_grep '"inventory_state": "blocked"' "$WRITER_INVENTORY_REPORT" "inventory state"
require_grep '"inventory_action": "reject_blocked_writer_preflight"' "$WRITER_INVENTORY_REPORT" "inventory action"
require_grep '"writer_contract": "none"' "$WRITER_INVENTORY_REPORT" "writer contract"
require_grep '"rollback_contract": "none"' "$WRITER_INVENTORY_REPORT" "rollback contract"
require_grep '"admission_ledger_contract": "none"' "$WRITER_INVENTORY_REPORT" "admission ledger contract"
require_grep '"writer_contract_present": false' "$WRITER_INVENTORY_REPORT" "writer contract guard"
require_grep '"rollback_contract_present": false' "$WRITER_INVENTORY_REPORT" "rollback contract guard"
require_grep '"ledger_contract_present": false' "$WRITER_INVENTORY_REPORT" "ledger contract guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_ready": true' "$WRITER_INVENTORY_REPORT" "writer inventory ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_consumed": true' "$WRITER_INVENTORY_REPORT" "writer preflight consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_required": true' "$WRITER_INVENTORY_REPORT" "writer preflight required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory": true' "$WRITER_INVENTORY_REPORT" "next-step block"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_receipt"' "$WRITER_INVENTORY_REPORT" "receipt shape"
require_grep '"writer_inventory_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory"' "$WRITER_INVENTORY_REPORT" "writer inventory kind"
require_grep '"writer_inventory_mode": "closed_writer_preflight_inventory_guard"' "$WRITER_INVENTORY_REPORT" "writer inventory mode"
require_grep '"writer_inventory_stage": "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_pre_writer_contract_inventory"' "$WRITER_INVENTORY_REPORT" "writer inventory stage"
require_grep '"writer_preflight_verified": true' "$WRITER_INVENTORY_REPORT" "writer preflight verification"
require_grep '"writer_preflight_hash_verified": true' "$WRITER_INVENTORY_REPORT" "writer preflight hash verification"
require_grep '"writer_preflight_read_back_verified": true' "$WRITER_INVENTORY_REPORT" "writer preflight read-back verification"
require_grep '"live_stage_verified": true' "$WRITER_INVENTORY_REPORT" "live stage verification"
require_grep '"enable_gate_verified": true' "$WRITER_INVENTORY_REPORT" "enable gate verification"
require_grep '"switch_verified": true' "$WRITER_INVENTORY_REPORT" "switch verification"
require_grep '"promotion_verified": true' "$WRITER_INVENTORY_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$WRITER_INVENTORY_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$WRITER_INVENTORY_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$WRITER_INVENTORY_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$WRITER_INVENTORY_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$WRITER_INVENTORY_REPORT" "candidate verification"
require_grep '"admission_required": true' "$WRITER_INVENTORY_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$WRITER_INVENTORY_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$WRITER_INVENTORY_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$WRITER_INVENTORY_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$WRITER_INVENTORY_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$WRITER_INVENTORY_REPORT" "body mutation guard"
require_grep '"requires_writer": true' "$WRITER_INVENTORY_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$WRITER_INVENTORY_REPORT" "writer guard"
require_grep '"rollback_required": true' "$WRITER_INVENTORY_REPORT" "rollback required"
require_grep '"requires_rollback": true' "$WRITER_INVENTORY_REPORT" "rollback requirement"
require_grep '"rollback_ready": false' "$WRITER_INVENTORY_REPORT" "rollback guard"
require_grep '"read_only": true' "$WRITER_INVENTORY_REPORT" "read-only flag"
require_grep '"replay_only": true' "$WRITER_INVENTORY_REPORT" "replay-only flag"
require_grep '"contracts_ready": false' "$WRITER_INVENTORY_REPORT" "contracts guard"
require_grep '"write_allowed": false' "$WRITER_INVENTORY_REPORT" "write guard"
require_grep '"admission_allowed": false' "$WRITER_INVENTORY_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "live guard"
require_grep '"mutates_state": false' "$WRITER_INVENTORY_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$WRITER_INVENTORY_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v1"' "$WRITER_INVENTORY_REPORT" "source writer preflight schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run"' "$WRITER_INVENTORY_REPORT" "source writer preflight status"
require_grep '"source_writer_preflight_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight"' "$WRITER_INVENTORY_REPORT" "source writer preflight kind"
require_grep '"source_writer_preflight_live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "source writer preflight live guard"
require_grep '"source_live_stage_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage"' "$WRITER_INVENTORY_REPORT" "source live stage kind"
require_grep '"source_live_stage_live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "source live stage live guard"
require_grep '"source_enable_state": "disabled"' "$WRITER_INVENTORY_REPORT" "source enable state"
require_grep '"source_enable_action": "require_operator_key"' "$WRITER_INVENTORY_REPORT" "source enable action"
require_grep '"source_enable_gate_live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "source enable gate live guard"
require_grep '"source_switch_state": "disabled"' "$WRITER_INVENTORY_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$WRITER_INVENTORY_REPORT" "source switch action"
require_grep '"source_switch_live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "source switch live guard"
require_grep '"source_promotion": "pending_live_admission"' "$WRITER_INVENTORY_REPORT" "source promotion"
require_grep '"source_promotion_live_admission_enabled": false' "$WRITER_INVENTORY_REPORT" "source promotion live guard"
require_grep '"passed": true' "$WRITER_INVENTORY_REPORT" "writer inventory pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory\] pass:' "$WRITER_INVENTORY_LOG" "writer inventory pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_report=$WRITER_PREFLIGHT_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_report=$WRITER_INVENTORY_REPORT"
