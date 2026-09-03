#!/usr/bin/env bash
# admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_store_smoke.sh - store blocked weighted Resonance admission final-gate observation-boundary preflight-gate candidate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-lsfgrobpgcs.XXXXXX")}"
CANDIDATE_WORKDIR="$WORKDIR/c"
CANDIDATE_REPORT="$WORKDIR/c.json"
CANDIDATE_STORE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT:-$WORKDIR/s.json}"
CANDIDATE_LOG="$WORKDIR/c.log"
STORE_LOG="$WORKDIR/s.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-smoke] FAIL: $*" >&2
    if [[ -f "$CANDIDATE_LOG" ]]; then
        tail -n 500 "$CANDIDATE_LOG" >&2 || true
    fi
    if [[ -f "$STORE_LOG" ]]; then
        tail -n 260 "$STORE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_WORKDIR="$CANDIDATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_FINAL_GATE_RECEIVER_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT="$CANDIDATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_smoke.sh" >"$CANDIDATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate producer failed"
fi

[[ -s "$CANDIDATE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate report not written: $CANDIDATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_store.sh" "$CANDIDATE_REPORT" "$CANDIDATE_STORE_REPORT" >"$STORE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store rejected candidate report"
fi

[[ -s "$CANDIDATE_STORE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store report not written: $CANDIDATE_STORE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store.v1"' "$CANDIDATE_STORE_REPORT" "candidate-store schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run"' "$CANDIDATE_STORE_REPORT" "candidate-store status"
require_grep '"target": "live_route_admission_next_step"' "$CANDIDATE_STORE_REPORT" "candidate-store target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store"' "$CANDIDATE_STORE_REPORT" "candidate-store target kind"
require_grep '"target_mode": "append_only_read_back_store_dry_run"' "$CANDIDATE_STORE_REPORT" "candidate-store target mode"
require_grep '"action": "store_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run"' "$CANDIDATE_STORE_REPORT" "candidate-store action"
require_grep '"ledger_state": "blocked"' "$CANDIDATE_STORE_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$CANDIDATE_STORE_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready": true' "$CANDIDATE_STORE_REPORT" "store ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_consumed": true' "$CANDIDATE_STORE_REPORT" "candidate consumed flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_required": true' "$CANDIDATE_STORE_REPORT" "candidate required flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store": true' "$CANDIDATE_STORE_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-id-' "$CANDIDATE_STORE_REPORT" "store id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt"' "$CANDIDATE_STORE_REPORT" "receipt shape"
require_grep '"store_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store"' "$CANDIDATE_STORE_REPORT" "store kind"
require_grep '"store_mode": "append_only_read_back_store"' "$CANDIDATE_STORE_REPORT" "store mode"
require_grep '"store_stage": "post_preflight_gate_candidate_pre_live_admission_store"' "$CANDIDATE_STORE_REPORT" "store stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-causal-' "$CANDIDATE_STORE_REPORT" "causal id"
require_grep '"store_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-' "$CANDIDATE_STORE_REPORT" "store hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-read-' "$CANDIDATE_STORE_REPORT" "read-back hash"
require_grep '"candidate_verified": true' "$CANDIDATE_STORE_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$CANDIDATE_STORE_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$CANDIDATE_STORE_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$CANDIDATE_STORE_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$CANDIDATE_STORE_REPORT" "observation verification"
require_grep '"final_gate_verified": true' "$CANDIDATE_STORE_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$CANDIDATE_STORE_REPORT" "seal verification"
require_grep '"permit_verified": true' "$CANDIDATE_STORE_REPORT" "permit verification"
require_grep '"authority_verified": true' "$CANDIDATE_STORE_REPORT" "authority verification"
require_grep '"admission_required": true' "$CANDIDATE_STORE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$CANDIDATE_STORE_REPORT" "shadow flag"
require_grep '"dry_run_only": true' "$CANDIDATE_STORE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$CANDIDATE_STORE_REPORT" "live-ready flag"
require_grep '"rollback_required": true' "$CANDIDATE_STORE_REPORT" "rollback requirement"
require_grep '"append_only": true' "$CANDIDATE_STORE_REPORT" "append-only flag"
require_grep '"read_back": true' "$CANDIDATE_STORE_REPORT" "read-back flag"
require_grep '"receipt_persisted": true' "$CANDIDATE_STORE_REPORT" "receipt persisted flag"
require_grep '"receipt_verified": true' "$CANDIDATE_STORE_REPORT" "receipt verified flag"
require_grep '"raw_dream_text_allowed": false' "$CANDIDATE_STORE_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$CANDIDATE_STORE_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$CANDIDATE_STORE_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$CANDIDATE_STORE_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$CANDIDATE_STORE_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$CANDIDATE_STORE_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$CANDIDATE_STORE_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$CANDIDATE_STORE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$CANDIDATE_STORE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$CANDIDATE_STORE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$CANDIDATE_STORE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$CANDIDATE_STORE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$CANDIDATE_STORE_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$CANDIDATE_STORE_REPORT" "body target"
require_grep '"passed": true' "$CANDIDATE_STORE_REPORT" "candidate-store pass flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate.v1"' "$CANDIDATE_STORE_REPORT" "source candidate schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_blocked_dry_run"' "$CANDIDATE_STORE_REPORT" "source candidate status"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-id-' "$CANDIDATE_STORE_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready": true' "$CANDIDATE_STORE_REPORT" "source weighted candidate ready"
require_grep '"source_candidate_state": "blocked"' "$CANDIDATE_STORE_REPORT" "source candidate state"
require_grep '"source_candidate_action": "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate"' "$CANDIDATE_STORE_REPORT" "source candidate action"
require_grep '"source_candidate_kind": "blocked_final_gate_observation_boundary_preflight_gate_candidate"' "$CANDIDATE_STORE_REPORT" "source candidate kind"
require_grep '"source_candidate_opened": false' "$CANDIDATE_STORE_REPORT" "source candidate closed flag"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-id-' "$CANDIDATE_STORE_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready": true' "$CANDIDATE_STORE_REPORT" "source gate ready"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$CANDIDATE_STORE_REPORT" "source gate closed"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-id-' "$CANDIDATE_STORE_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-id-' "$CANDIDATE_STORE_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-id-' "$CANDIDATE_STORE_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-id-' "$CANDIDATE_STORE_REPORT" "source receiver id"
require_grep '"source_writer_inventory_verified": true' "$CANDIDATE_STORE_REPORT" "source writer inventory"
require_grep '"source_writer_preflight_verified": true' "$CANDIDATE_STORE_REPORT" "source writer preflight"
require_grep '"source_ledger_append_allowed": false' "$CANDIDATE_STORE_REPORT" "source ledger append guard"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store\] pass:' "$STORE_LOG" "candidate-store pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_report=$CANDIDATE_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_report=$CANDIDATE_STORE_REPORT"
