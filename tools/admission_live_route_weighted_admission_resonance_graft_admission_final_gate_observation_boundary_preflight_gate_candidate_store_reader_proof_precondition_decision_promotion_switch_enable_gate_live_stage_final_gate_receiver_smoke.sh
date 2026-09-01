#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_smoke.sh - preview final-gate receiver from compact weighted graft admission final-gate intent.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsfgr.XXXXXX")}"
INTENT_WORKDIR="$WORKDIR/i"
GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT="$WORKDIR/i.json"
GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_REPORT:-$WORKDIR/r.json}"
INTENT_LOG="$WORKDIR/i.log"
RECEIVER_LOG="$WORKDIR/r.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-smoke] FAIL: $*" >&2
    if [[ -f "$INTENT_LOG" ]]; then
        tail -n 500 "$INTENT_LOG" >&2 || true
    fi
    if [[ -f "$RECEIVER_LOG" ]]; then
        tail -n 260 "$RECEIVER_LOG" >&2 || true
    fi
    exit 1
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -Fq "$pattern" "$file"; then
        die "$label missing in $file"
    fi
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_INTENT_WORKDIR="$INTENT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_INTENT_REPORT="$GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_smoke.sh" >"$INTENT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent producer failed"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent report not written: $GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver.sh" "$GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT" "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" >"$RECEIVER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver rejected intent report"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver report not written: $GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver.v1"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_previewed_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target kind"
require_grep '"target_mode": "bounded_receiver_preview_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target mode"
require_grep '"action": "preview_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver action"
require_grep '"writer_action": "reject_blocked_admission_final_gate_receiver"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_final_gate_receiver"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_final_gate_receiver"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "ledger write scope"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receipt shape"
require_grep '"admission_final_gate_receiver_state": "previewed"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver state"
require_grep '"admission_final_gate_receiver_action": "preview_blocked_final_gate_receiver"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver action field"
require_grep '"admission_final_gate_receiver_target": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target field"
require_grep '"admission_final_gate_receiver_target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target kind field"
require_grep '"admission_final_gate_receiver_target_mode": "bounded_receiver_preview_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver target mode field"
require_grep '"admission_final_gate_receiver_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver dry-run flag"
require_grep '"admission_final_gate_receiver_intent_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver intent verified flag"
require_grep '"admission_final_gate_receiver_final_gate_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver final gate verified flag"
require_grep '"admission_final_gate_receiver_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver ready flag"
require_grep '"final_gate_receiver": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver"
require_grep '"final_gate_receiver_kind": "internal_world"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver kind"
require_grep '"final_gate_receiver_influence_kind": "bounded_direction"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "influence kind"
require_grep '"final_gate_receiver_max_influence": 0.05' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "influence cap"
require_grep '"final_gate_receiver_ttl_turns": 1' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver ttl"
require_grep '"final_gate_receiver_state_hash_mode": "blocked_intent_receiver_preview"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "state hash mode"
require_grep '"final_gate_receiver_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "raw observed guard"
require_grep '"final_gate_receiver_raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "raw forwarded guard"
require_grep '"final_gate_receiver_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "raw allowed guard"
require_grep '"final_gate_receiver_janus_surface_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "Janus guard"
require_grep '"final_gate_receiver_cooc_learning_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "cooc guard"
require_grep '"final_gate_receiver_delta_harvest_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "delta guard"
require_grep '"final_gate_receiver_body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "body mutation guard"
require_grep '"final_gate_receiver_pre_state_hash_required": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "pre hash required"
require_grep '"final_gate_receiver_post_state_hash_required": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "post hash required"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "weighted receiver ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_consumed": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "intent consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_required": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "intent required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-id-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-causal-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver causal id"
require_grep '"admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-pre-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver pre hash"
require_grep '"admission_final_gate_receiver_post_state_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-post-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver post hash"
require_grep '"admission_final_gate_receiver_state_delta_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-delta-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver delta hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent.v1"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent status"
require_grep '"source_target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent target"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-id-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source weighted intent ready"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-causal-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent causal"
require_grep '"source_admission_final_gate_intent_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent hash"
require_grep '"source_admission_final_gate_intent_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-read-' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent read-back"
require_grep '"source_admission_final_gate_intent_receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent receipt shape"
require_grep '"source_admission_final_gate_intent_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent state"
require_grep '"source_admission_final_gate_intent_action": "draft_blocked_final_gate_intent"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent action"
require_grep '"source_admission_final_gate_intent_target": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent target field"
require_grep '"source_admission_final_gate_intent_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source intent ready flag"
require_grep '"source_final_gate_intent_receiver": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source receiver"
require_grep '"source_final_gate_intent_receiver_kind": "internal_world"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source receiver kind"
require_grep '"source_final_gate_intent_influence_kind": "bounded_direction"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source influence kind"
require_grep '"source_final_gate_intent_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source raw guard"
require_grep '"source_final_gate_intent_janus_surface_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source Janus guard"
require_grep '"source_final_gate_intent_cooc_learning_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source cooc guard"
require_grep '"source_final_gate_intent_delta_harvest_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source delta guard"
require_grep '"source_final_gate_intent_pre_state_hash_required": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source pre hash required"
require_grep '"source_final_gate_intent_post_state_hash_required": true' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source post hash required"
require_grep '"source_admission_final_gate_intent_reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate intent drafted from blocked final gate; live admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "source reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "base authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver previewed from blocked final gate intent; live admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "receiver reason"
require_grep '[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver] pass:' "$RECEIVER_LOG" "receiver pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_intent_report=$GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_report=$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT"
