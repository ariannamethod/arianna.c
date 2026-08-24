#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_smoke.sh - produce and validate disabled switch enable gate from closed decision promotion switch.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/a2a-w-eg.XXXXXX")}"
SWITCH_WORKDIR="$WORKDIR/s"
SWITCH_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.json"
ENABLE_GATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.json}"
SWITCH_LOG="$WORKDIR/switch.log"
ENABLE_GATE_LOG="$WORKDIR/enable_gate.log"
ASSERT_LOG="$WORKDIR/assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-smoke] FAIL: $*" >&2
    if [[ -f "$SWITCH_LOG" ]]; then
        tail -n 500 "$SWITCH_LOG" >&2 || true
    fi
    if [[ -f "$ENABLE_GATE_LOG" ]]; then
        tail -n 260 "$ENABLE_GATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_WORKDIR="$SWITCH_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_REPORT="$SWITCH_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_smoke.sh" >"$SWITCH_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch producer failed"
fi

[[ -s "$SWITCH_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch report not written: $SWITCH_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.sh" "$SWITCH_REPORT" "$ENABLE_GATE_REPORT" >"$ENABLE_GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate rejected switch report"
fi

[[ -s "$ENABLE_GATE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate report not written: $ENABLE_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_assert.sh" "$ENABLE_GATE_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.v1"' "$ENABLE_GATE_REPORT" "enable gate schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_disabled_dry_run"' "$ENABLE_GATE_REPORT" "enable gate status"
require_grep '"target": "live_route_admission_next_step"' "$ENABLE_GATE_REPORT" "enable gate target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"' "$ENABLE_GATE_REPORT" "enable gate target kind"
require_grep '"target_mode": "closed_enable_gate_dry_run"' "$ENABLE_GATE_REPORT" "enable gate target mode"
require_grep '"action": "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run"' "$ENABLE_GATE_REPORT" "enable gate action"
require_grep '"enable_state": "disabled"' "$ENABLE_GATE_REPORT" "enable state"
require_grep '"enable_action": "require_operator_key"' "$ENABLE_GATE_REPORT" "enable action"
require_grep '"switch_state": "disabled"' "$ENABLE_GATE_REPORT" "switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$ENABLE_GATE_REPORT" "switch action"
require_grep '"promotion": "pending_live_admission"' "$ENABLE_GATE_REPORT" "promotion"
require_grep '"ledger_state": "blocked"' "$ENABLE_GATE_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$ENABLE_GATE_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready": true' "$ENABLE_GATE_REPORT" "enable gate ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_consumed": true' "$ENABLE_GATE_REPORT" "switch consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_required": true' "$ENABLE_GATE_REPORT" "switch required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate": true' "$ENABLE_GATE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-' "$ENABLE_GATE_REPORT" "enable gate id"
require_grep '"enable_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-' "$ENABLE_GATE_REPORT" "enable gate hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-read-' "$ENABLE_GATE_REPORT" "read-back hash"
require_grep '"switch_verified": true' "$ENABLE_GATE_REPORT" "switch verification"
require_grep '"switch_hash_verified": true' "$ENABLE_GATE_REPORT" "switch hash verification"
require_grep '"switch_read_back_verified": true' "$ENABLE_GATE_REPORT" "switch read-back verification"
require_grep '"promotion_verified": true' "$ENABLE_GATE_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$ENABLE_GATE_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$ENABLE_GATE_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$ENABLE_GATE_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$ENABLE_GATE_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$ENABLE_GATE_REPORT" "candidate verification"
require_grep '"admission_required": true' "$ENABLE_GATE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$ENABLE_GATE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$ENABLE_GATE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$ENABLE_GATE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$ENABLE_GATE_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$ENABLE_GATE_REPORT" "raw text guard"
require_grep '"janus_surface_allowed": false' "$ENABLE_GATE_REPORT" "janus guard"
require_grep '"cooc_learning_allowed": false' "$ENABLE_GATE_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$ENABLE_GATE_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$ENABLE_GATE_REPORT" "body mutation guard"
require_grep '"read_only": true' "$ENABLE_GATE_REPORT" "read-only flag"
require_grep '"replay_only": true' "$ENABLE_GATE_REPORT" "replay-only flag"
require_grep '"authority_granted": false' "$ENABLE_GATE_REPORT" "authority guard"
require_grep '"contracts_ready": false' "$ENABLE_GATE_REPORT" "contracts guard"
require_grep '"write_allowed": false' "$ENABLE_GATE_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$ENABLE_GATE_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$ENABLE_GATE_REPORT" "live guard"
require_grep '"mutates_state": false' "$ENABLE_GATE_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$ENABLE_GATE_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.v1"' "$ENABLE_GATE_REPORT" "source switch schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-' "$ENABLE_GATE_REPORT" "source switch id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-' "$ENABLE_GATE_REPORT" "source switch hash"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-read-' "$ENABLE_GATE_REPORT" "source switch read-back hash"
require_grep '"source_switch_state": "disabled"' "$ENABLE_GATE_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$ENABLE_GATE_REPORT" "source switch action"
require_grep '"source_switch_stage": "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_pre_live_admission_switch"' "$ENABLE_GATE_REPORT" "source switch stage"
require_grep '"source_switch_ledger_append_allowed": false' "$ENABLE_GATE_REPORT" "source switch ledger guard"
require_grep '"source_switch_graft_allowed": false' "$ENABLE_GATE_REPORT" "source switch graft guard"
require_grep '"source_switch_write_allowed": false' "$ENABLE_GATE_REPORT" "source switch writer guard"
require_grep '"source_switch_live_admission_enabled": false' "$ENABLE_GATE_REPORT" "source switch live guard"
require_grep '"source_switch_body_mutation_allowed": false' "$ENABLE_GATE_REPORT" "source switch body guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-' "$ENABLE_GATE_REPORT" "source promotion id"
require_grep '"source_promotion": "pending_live_admission"' "$ENABLE_GATE_REPORT" "source promotion"
require_grep '"source_promotion_ledger_append_allowed": false' "$ENABLE_GATE_REPORT" "source promotion ledger guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$ENABLE_GATE_REPORT" "source decision id"
require_grep '"source_decision": "shadow_ready"' "$ENABLE_GATE_REPORT" "source decision"
require_grep '"source_decision_ledger_append_allowed": false' "$ENABLE_GATE_REPORT" "source decision ledger guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$ENABLE_GATE_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$ENABLE_GATE_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$ENABLE_GATE_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$ENABLE_GATE_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$ENABLE_GATE_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$ENABLE_GATE_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$ENABLE_GATE_REPORT" "source gate closed"
require_grep '"passed": true' "$ENABLE_GATE_REPORT" "enable gate pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate\] pass:' "$ENABLE_GATE_LOG" "enable gate pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_report=$SWITCH_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_report=$ENABLE_GATE_REPORT"
