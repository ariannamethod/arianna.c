#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_smoke.sh - declare final-gate observation boundary from compact weighted graft admission final-gate observation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsfgrob.XXXXXX")}"
OBSERVATION_WORKDIR="$WORKDIR/o"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT="$WORKDIR/o.json"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_REPORT:-$WORKDIR/b.json}"
OBSERVATION_LOG="$WORKDIR/o.log"
BOUNDARY_LOG="$WORKDIR/b.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-smoke] FAIL: $*" >&2
    if [[ -f "$OBSERVATION_LOG" ]]; then
        tail -n 500 "$OBSERVATION_LOG" >&2 || true
    fi
    if [[ -f "$BOUNDARY_LOG" ]]; then
        tail -n 260 "$BOUNDARY_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_WORKDIR="$OBSERVATION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_REPORT="$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_smoke.sh" >"$OBSERVATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation producer failed"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_lsfg_receiver_observation_boundary.sh" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" >"$BOUNDARY_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary rejected observation report"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_declared_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary target kind"
require_grep '"target_mode": "receipt_only_closed_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary target mode"
require_grep '"action": "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary action"
require_grep '"writer_action": "reject_blocked_admission_final_gate_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_final_gate_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_final_gate_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "ledger write scope"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "receipt shape"
require_grep '"admission_final_gate_observation_boundary_state": "declared"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary state"
require_grep '"admission_final_gate_observation_boundary_action": "declare_blocked_final_gate_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary action field"
require_grep '"admission_final_gate_observation_boundary_target": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary target field"
require_grep '"admission_final_gate_observation_boundary_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary dry-run"
require_grep '"admission_final_gate_observation_boundary_observation_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "observation verified"
require_grep '"admission_final_gate_observation_boundary_read_back_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "read-back verified"
require_grep '"admission_final_gate_observation_boundary_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary ready guard"
require_grep '"final_gate_observation_boundary_kind": "blocked_final_gate_observation_boundary"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary kind"
require_grep '"final_gate_observation_boundary_mode": "no_mutation_closed_boundary_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary mode"
require_grep '"final_gate_observation_boundary_stage": "post_observation_pre_live_admission"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary stage"
require_grep '"final_gate_observation_boundary_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "raw observed guard"
require_grep '"final_gate_observation_boundary_raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "raw forwarded guard"
require_grep '"final_gate_observation_boundary_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "raw allowed guard"
require_grep '"final_gate_observation_boundary_janus_surface_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "Janus guard"
require_grep '"final_gate_observation_boundary_cooc_learning_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "cooc guard"
require_grep '"final_gate_observation_boundary_delta_harvest_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "delta guard"
require_grep '"final_gate_observation_boundary_body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "body mutation guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "weighted boundary ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_consumed": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "observation consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_required": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "observation required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-causal-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary causal"
require_grep '"admission_final_gate_observation_boundary_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary hash"
require_grep '"admission_final_gate_observation_boundary_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-read-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_recorded_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation status"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation ready"
require_grep '"source_admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-append-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source append hash"
require_grep '"source_admission_final_gate_observation_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-read-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source read-back hash"
require_grep '"source_admission_final_gate_observation_state": "recorded"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation state"
require_grep '"source_admission_final_gate_observation_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observation ready guard"
require_grep '"source_final_gate_observation_observer": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source observer"
require_grep '"source_final_gate_observation_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "source raw observed guard"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary declared from recorded observation; live admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT" "boundary reason"
require_grep '[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary] pass:' "$BOUNDARY_LOG" "boundary pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT"
