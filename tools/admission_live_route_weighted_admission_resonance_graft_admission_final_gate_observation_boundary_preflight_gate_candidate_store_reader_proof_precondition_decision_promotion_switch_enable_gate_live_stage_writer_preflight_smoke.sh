#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_smoke.sh - produce and validate blocked writer preflight from blocked live stage.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_PREFLIGHT_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-wp.XXXXXX")}"
LIVE_STAGE_WORKDIR="$WORKDIR/ls"
LIVE_STAGE_REPORT="$WORKDIR/ls.json"
WRITER_PREFLIGHT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_PREFLIGHT_REPORT:-$WORKDIR/wp.json}"
LIVE_STAGE_LOG="$WORKDIR/live_stage.log"
WRITER_PREFLIGHT_LOG="$WORKDIR/writer_preflight.log"
ASSERT_LOG="$WORKDIR/assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-smoke] FAIL: $*" >&2
    if [[ -f "$LIVE_STAGE_LOG" ]]; then
        tail -n 500 "$LIVE_STAGE_LOG" >&2 || true
    fi
    if [[ -f "$WRITER_PREFLIGHT_LOG" ]]; then
        tail -n 260 "$WRITER_PREFLIGHT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WORKDIR="$LIVE_STAGE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_REPORT="$LIVE_STAGE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_smoke.sh" >"$LIVE_STAGE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage producer failed"
fi

[[ -s "$LIVE_STAGE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage report not written: $LIVE_STAGE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.sh" "$LIVE_STAGE_REPORT" "$WRITER_PREFLIGHT_REPORT" >"$WRITER_PREFLIGHT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight rejected live stage report"
fi

[[ -s "$WRITER_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight report not written: $WRITER_PREFLIGHT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_assert.sh" "$WRITER_PREFLIGHT_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v1"' "$WRITER_PREFLIGHT_REPORT" "writer preflight schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run"' "$WRITER_PREFLIGHT_REPORT" "writer preflight status"
require_grep '"target": "live_route_admission_next_step"' "$WRITER_PREFLIGHT_REPORT" "writer preflight target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight"' "$WRITER_PREFLIGHT_REPORT" "writer preflight target kind"
require_grep '"target_mode": "closed_writer_preflight_guard_dry_run"' "$WRITER_PREFLIGHT_REPORT" "writer preflight target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run"' "$WRITER_PREFLIGHT_REPORT" "writer preflight action"
require_grep '"writer_state": "blocked"' "$WRITER_PREFLIGHT_REPORT" "writer state"
require_grep '"writer_action": "reject_blocked_live_stage"' "$WRITER_PREFLIGHT_REPORT" "writer action"
require_grep '"rollback_state": "blocked"' "$WRITER_PREFLIGHT_REPORT" "rollback state"
require_grep '"rollback_action": "reject_blocked_live_stage"' "$WRITER_PREFLIGHT_REPORT" "rollback action"
require_grep '"stage_state": "blocked"' "$WRITER_PREFLIGHT_REPORT" "stage state"
require_grep '"stage_action": "reject_disabled_enable_gate"' "$WRITER_PREFLIGHT_REPORT" "stage action"
require_grep '"enable_state": "disabled"' "$WRITER_PREFLIGHT_REPORT" "enable state"
require_grep '"enable_action": "require_operator_key"' "$WRITER_PREFLIGHT_REPORT" "enable action"
require_grep '"switch_state": "disabled"' "$WRITER_PREFLIGHT_REPORT" "switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$WRITER_PREFLIGHT_REPORT" "switch action"
require_grep '"promotion": "pending_live_admission"' "$WRITER_PREFLIGHT_REPORT" "promotion"
require_grep '"ledger_state": "blocked"' "$WRITER_PREFLIGHT_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$WRITER_PREFLIGHT_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ready": true' "$WRITER_PREFLIGHT_REPORT" "writer preflight ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_consumed": true' "$WRITER_PREFLIGHT_REPORT" "live stage consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_required": true' "$WRITER_PREFLIGHT_REPORT" "live stage required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight": true' "$WRITER_PREFLIGHT_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-id-' "$WRITER_PREFLIGHT_REPORT" "writer preflight id"
require_grep '"writer_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-' "$WRITER_PREFLIGHT_REPORT" "writer preflight hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-read-' "$WRITER_PREFLIGHT_REPORT" "read-back hash"
require_grep '"live_stage_verified": true' "$WRITER_PREFLIGHT_REPORT" "live stage verification"
require_grep '"live_stage_hash_verified": true' "$WRITER_PREFLIGHT_REPORT" "live stage hash verification"
require_grep '"live_stage_read_back_verified": true' "$WRITER_PREFLIGHT_REPORT" "live stage read-back verification"
require_grep '"enable_gate_verified": true' "$WRITER_PREFLIGHT_REPORT" "enable gate verification"
require_grep '"switch_verified": true' "$WRITER_PREFLIGHT_REPORT" "switch verification"
require_grep '"promotion_verified": true' "$WRITER_PREFLIGHT_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$WRITER_PREFLIGHT_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$WRITER_PREFLIGHT_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$WRITER_PREFLIGHT_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$WRITER_PREFLIGHT_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$WRITER_PREFLIGHT_REPORT" "candidate verification"
require_grep '"admission_required": true' "$WRITER_PREFLIGHT_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$WRITER_PREFLIGHT_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$WRITER_PREFLIGHT_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$WRITER_PREFLIGHT_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$WRITER_PREFLIGHT_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$WRITER_PREFLIGHT_REPORT" "body mutation guard"
require_grep '"requires_writer": true' "$WRITER_PREFLIGHT_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$WRITER_PREFLIGHT_REPORT" "writer guard"
require_grep '"rollback_required": true' "$WRITER_PREFLIGHT_REPORT" "rollback required"
require_grep '"requires_rollback": true' "$WRITER_PREFLIGHT_REPORT" "rollback requirement"
require_grep '"rollback_ready": false' "$WRITER_PREFLIGHT_REPORT" "rollback guard"
require_grep '"read_only": true' "$WRITER_PREFLIGHT_REPORT" "read-only flag"
require_grep '"replay_only": true' "$WRITER_PREFLIGHT_REPORT" "replay-only flag"
require_grep '"write_allowed": false' "$WRITER_PREFLIGHT_REPORT" "write guard"
require_grep '"admission_allowed": false' "$WRITER_PREFLIGHT_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$WRITER_PREFLIGHT_REPORT" "live guard"
require_grep '"mutates_state": false' "$WRITER_PREFLIGHT_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$WRITER_PREFLIGHT_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage.v1"' "$WRITER_PREFLIGHT_REPORT" "source live stage schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-id-' "$WRITER_PREFLIGHT_REPORT" "source live stage id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-' "$WRITER_PREFLIGHT_REPORT" "source live stage hash"
require_grep '"source_stage_state": "blocked"' "$WRITER_PREFLIGHT_REPORT" "source stage state"
require_grep '"source_stage_action": "reject_disabled_enable_gate"' "$WRITER_PREFLIGHT_REPORT" "source stage action"
require_grep '"source_live_stage_ledger_append_allowed": false' "$WRITER_PREFLIGHT_REPORT" "source live stage ledger guard"
require_grep '"source_live_stage_writer_ready": false' "$WRITER_PREFLIGHT_REPORT" "source live stage writer guard"
require_grep '"source_live_stage_rollback_ready": false' "$WRITER_PREFLIGHT_REPORT" "source live stage rollback guard"
require_grep '"source_live_stage_live_admission_enabled": false' "$WRITER_PREFLIGHT_REPORT" "source live stage live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-' "$WRITER_PREFLIGHT_REPORT" "source enable gate id"
require_grep '"source_enable_state": "disabled"' "$WRITER_PREFLIGHT_REPORT" "source enable state"
require_grep '"source_enable_action": "require_operator_key"' "$WRITER_PREFLIGHT_REPORT" "source enable action"
require_grep '"source_enable_gate_live_admission_enabled": false' "$WRITER_PREFLIGHT_REPORT" "source enable gate live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-' "$WRITER_PREFLIGHT_REPORT" "source switch id"
require_grep '"source_switch_state": "disabled"' "$WRITER_PREFLIGHT_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$WRITER_PREFLIGHT_REPORT" "source switch action"
require_grep '"source_switch_ledger_append_allowed": false' "$WRITER_PREFLIGHT_REPORT" "source switch ledger guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-' "$WRITER_PREFLIGHT_REPORT" "source promotion id"
require_grep '"source_promotion": "pending_live_admission"' "$WRITER_PREFLIGHT_REPORT" "source promotion"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$WRITER_PREFLIGHT_REPORT" "source decision id"
require_grep '"source_decision": "shadow_ready"' "$WRITER_PREFLIGHT_REPORT" "source decision"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$WRITER_PREFLIGHT_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$WRITER_PREFLIGHT_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$WRITER_PREFLIGHT_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$WRITER_PREFLIGHT_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$WRITER_PREFLIGHT_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$WRITER_PREFLIGHT_REPORT" "source gate id"
require_grep '"passed": true' "$WRITER_PREFLIGHT_REPORT" "writer preflight pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight\] pass:' "$WRITER_PREFLIGHT_LOG" "writer preflight pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_report=$LIVE_STAGE_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_report=$WRITER_PREFLIGHT_REPORT"
