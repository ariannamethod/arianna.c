#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_smoke.sh - produce and validate blocked live stage from disabled switch enable gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-ls.XXXXXX")}"
ENABLE_GATE_WORKDIR="$WORKDIR/eg"
ENABLE_GATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.json"
LIVE_STAGE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage.json}"
ENABLE_GATE_LOG="$WORKDIR/enable_gate.log"
LIVE_STAGE_LOG="$WORKDIR/live_stage.log"
ASSERT_LOG="$WORKDIR/assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-smoke] FAIL: $*" >&2
    if [[ -f "$ENABLE_GATE_LOG" ]]; then
        tail -n 500 "$ENABLE_GATE_LOG" >&2 || true
    fi
    if [[ -f "$LIVE_STAGE_LOG" ]]; then
        tail -n 260 "$LIVE_STAGE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_WORKDIR="$ENABLE_GATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_REPORT="$ENABLE_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_smoke.sh" >"$ENABLE_GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate producer failed"
fi

[[ -s "$ENABLE_GATE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate report not written: $ENABLE_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage.sh" "$ENABLE_GATE_REPORT" "$LIVE_STAGE_REPORT" >"$LIVE_STAGE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage rejected enable gate report"
fi

[[ -s "$LIVE_STAGE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage report not written: $LIVE_STAGE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_assert.sh" "$LIVE_STAGE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage.v1"' "$LIVE_STAGE_REPORT" "live stage schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run"' "$LIVE_STAGE_REPORT" "live stage status"
require_grep '"target": "live_route_admission_next_step"' "$LIVE_STAGE_REPORT" "live stage target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage"' "$LIVE_STAGE_REPORT" "live stage target kind"
require_grep '"target_mode": "closed_live_stage_guard_dry_run"' "$LIVE_STAGE_REPORT" "live stage target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_disabled_dry_run"' "$LIVE_STAGE_REPORT" "live stage action"
require_grep '"stage_state": "blocked"' "$LIVE_STAGE_REPORT" "stage state"
require_grep '"stage_action": "reject_disabled_enable_gate"' "$LIVE_STAGE_REPORT" "stage action"
require_grep '"enable_state": "disabled"' "$LIVE_STAGE_REPORT" "enable state"
require_grep '"enable_action": "require_operator_key"' "$LIVE_STAGE_REPORT" "enable action"
require_grep '"switch_state": "disabled"' "$LIVE_STAGE_REPORT" "switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$LIVE_STAGE_REPORT" "switch action"
require_grep '"promotion": "pending_live_admission"' "$LIVE_STAGE_REPORT" "promotion"
require_grep '"ledger_state": "blocked"' "$LIVE_STAGE_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$LIVE_STAGE_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ready": true' "$LIVE_STAGE_REPORT" "live stage ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_consumed": true' "$LIVE_STAGE_REPORT" "enable gate consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_required": true' "$LIVE_STAGE_REPORT" "enable gate required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage": true' "$LIVE_STAGE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-id-' "$LIVE_STAGE_REPORT" "live stage id"
require_grep '"live_stage_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-' "$LIVE_STAGE_REPORT" "live stage hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-read-' "$LIVE_STAGE_REPORT" "read-back hash"
require_grep '"enable_gate_verified": true' "$LIVE_STAGE_REPORT" "enable gate verification"
require_grep '"enable_gate_hash_verified": true' "$LIVE_STAGE_REPORT" "enable gate hash verification"
require_grep '"enable_gate_read_back_verified": true' "$LIVE_STAGE_REPORT" "enable gate read-back verification"
require_grep '"switch_verified": true' "$LIVE_STAGE_REPORT" "switch verification"
require_grep '"promotion_verified": true' "$LIVE_STAGE_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$LIVE_STAGE_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$LIVE_STAGE_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$LIVE_STAGE_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$LIVE_STAGE_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$LIVE_STAGE_REPORT" "candidate verification"
require_grep '"admission_required": true' "$LIVE_STAGE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$LIVE_STAGE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$LIVE_STAGE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$LIVE_STAGE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$LIVE_STAGE_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$LIVE_STAGE_REPORT" "body mutation guard"
require_grep '"requires_writer": true' "$LIVE_STAGE_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$LIVE_STAGE_REPORT" "writer guard"
require_grep '"rollback_required": true' "$LIVE_STAGE_REPORT" "rollback required"
require_grep '"requires_rollback": true' "$LIVE_STAGE_REPORT" "rollback requirement"
require_grep '"rollback_ready": false' "$LIVE_STAGE_REPORT" "rollback guard"
require_grep '"read_only": true' "$LIVE_STAGE_REPORT" "read-only flag"
require_grep '"replay_only": true' "$LIVE_STAGE_REPORT" "replay-only flag"
require_grep '"write_allowed": false' "$LIVE_STAGE_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$LIVE_STAGE_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$LIVE_STAGE_REPORT" "live guard"
require_grep '"mutates_state": false' "$LIVE_STAGE_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$LIVE_STAGE_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.v1"' "$LIVE_STAGE_REPORT" "source enable gate schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-' "$LIVE_STAGE_REPORT" "source enable gate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-' "$LIVE_STAGE_REPORT" "source enable gate hash"
require_grep '"source_enable_state": "disabled"' "$LIVE_STAGE_REPORT" "source enable state"
require_grep '"source_enable_action": "require_operator_key"' "$LIVE_STAGE_REPORT" "source enable action"
require_grep '"source_enable_gate_ledger_append_allowed": false' "$LIVE_STAGE_REPORT" "source enable gate ledger guard"
require_grep '"source_enable_gate_write_allowed": false' "$LIVE_STAGE_REPORT" "source enable gate writer guard"
require_grep '"source_enable_gate_live_admission_enabled": false' "$LIVE_STAGE_REPORT" "source enable gate live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-' "$LIVE_STAGE_REPORT" "source switch id"
require_grep '"source_switch_state": "disabled"' "$LIVE_STAGE_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$LIVE_STAGE_REPORT" "source switch action"
require_grep '"source_switch_ledger_append_allowed": false' "$LIVE_STAGE_REPORT" "source switch ledger guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-' "$LIVE_STAGE_REPORT" "source promotion id"
require_grep '"source_promotion": "pending_live_admission"' "$LIVE_STAGE_REPORT" "source promotion"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$LIVE_STAGE_REPORT" "source decision id"
require_grep '"source_decision": "shadow_ready"' "$LIVE_STAGE_REPORT" "source decision"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$LIVE_STAGE_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$LIVE_STAGE_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$LIVE_STAGE_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$LIVE_STAGE_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$LIVE_STAGE_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$LIVE_STAGE_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$LIVE_STAGE_REPORT" "source gate closed"
require_grep '"passed": true' "$LIVE_STAGE_REPORT" "live stage pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage\] pass:' "$LIVE_STAGE_LOG" "live stage pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_report=$ENABLE_GATE_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_report=$LIVE_STAGE_REPORT"
