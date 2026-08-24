#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_smoke.sh - produce and validate disabled switch from closed decision promotion.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch.XXXXXX")}"
PROMOTION_WORKDIR="$WORKDIR/promotion"
PROMOTION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.json"
SWITCH_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.json}"
PROMOTION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.log"
SWITCH_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-smoke] FAIL: $*" >&2
    if [[ -f "$PROMOTION_LOG" ]]; then
        tail -n 500 "$PROMOTION_LOG" >&2 || true
    fi
    if [[ -f "$SWITCH_LOG" ]]; then
        tail -n 260 "$SWITCH_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_WORKDIR="$PROMOTION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_REPORT="$PROMOTION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_smoke.sh" >"$PROMOTION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion producer failed"
fi

[[ -s "$PROMOTION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion report not written: $PROMOTION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.sh" "$PROMOTION_REPORT" "$SWITCH_REPORT" >"$SWITCH_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch rejected promotion report"
fi

[[ -s "$SWITCH_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch report not written: $SWITCH_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_assert.sh" "$SWITCH_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch.v1"' "$SWITCH_REPORT" "switch schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run"' "$SWITCH_REPORT" "switch status"
require_grep '"target": "live_route_admission_next_step"' "$SWITCH_REPORT" "switch target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch"' "$SWITCH_REPORT" "switch target kind"
require_grep '"target_mode": "closed_switch_guard_dry_run"' "$SWITCH_REPORT" "switch target mode"
require_grep '"action": "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_dry_run"' "$SWITCH_REPORT" "switch action"
require_grep '"switch_state": "disabled"' "$SWITCH_REPORT" "switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$SWITCH_REPORT" "switch hold action"
require_grep '"promotion": "pending_live_admission"' "$SWITCH_REPORT" "promotion verdict"
require_grep '"ledger_state": "blocked"' "$SWITCH_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$SWITCH_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready": true' "$SWITCH_REPORT" "switch ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_consumed": true' "$SWITCH_REPORT" "promotion consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_required": true' "$SWITCH_REPORT" "promotion required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch": true' "$SWITCH_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-' "$SWITCH_REPORT" "switch id"
require_grep '"switch_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-' "$SWITCH_REPORT" "switch hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-read-' "$SWITCH_REPORT" "read-back hash"
require_grep '"promotion_verified": true' "$SWITCH_REPORT" "promotion verification"
require_grep '"promotion_hash_verified": true' "$SWITCH_REPORT" "promotion hash verification"
require_grep '"promotion_read_back_verified": true' "$SWITCH_REPORT" "promotion read-back verification"
require_grep '"decision_verified": true' "$SWITCH_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$SWITCH_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$SWITCH_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$SWITCH_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$SWITCH_REPORT" "candidate verification"
require_grep '"admission_required": true' "$SWITCH_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$SWITCH_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$SWITCH_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$SWITCH_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$SWITCH_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$SWITCH_REPORT" "raw text guard"
require_grep '"janus_surface_allowed": false' "$SWITCH_REPORT" "janus guard"
require_grep '"cooc_learning_allowed": false' "$SWITCH_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$SWITCH_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$SWITCH_REPORT" "body mutation guard"
require_grep '"read_only": true' "$SWITCH_REPORT" "read-only flag"
require_grep '"replay_only": true' "$SWITCH_REPORT" "replay-only flag"
require_grep '"authority_granted": false' "$SWITCH_REPORT" "authority guard"
require_grep '"contracts_ready": false' "$SWITCH_REPORT" "contracts guard"
require_grep '"write_allowed": false' "$SWITCH_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$SWITCH_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$SWITCH_REPORT" "live guard"
require_grep '"mutates_state": false' "$SWITCH_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$SWITCH_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.v1"' "$SWITCH_REPORT" "source promotion schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-' "$SWITCH_REPORT" "source promotion id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-' "$SWITCH_REPORT" "source promotion hash"
require_grep '"source_promotion": "pending_live_admission"' "$SWITCH_REPORT" "source promotion verdict"
require_grep '"source_promotion_ledger_append_allowed": false' "$SWITCH_REPORT" "source promotion ledger guard"
require_grep '"source_promotion_graft_allowed": false' "$SWITCH_REPORT" "source promotion graft guard"
require_grep '"source_promotion_write_allowed": false' "$SWITCH_REPORT" "source promotion writer guard"
require_grep '"source_promotion_live_admission_enabled": false' "$SWITCH_REPORT" "source promotion live guard"
require_grep '"source_promotion_body_mutation_allowed": false' "$SWITCH_REPORT" "source promotion body guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$SWITCH_REPORT" "source decision id"
require_grep '"source_decision_ledger_append_allowed": false' "$SWITCH_REPORT" "source decision ledger guard"
require_grep '"source_decision_graft_allowed": false' "$SWITCH_REPORT" "source decision graft guard"
require_grep '"source_decision_write_allowed": false' "$SWITCH_REPORT" "source decision writer guard"
require_grep '"source_decision_live_admission_enabled": false' "$SWITCH_REPORT" "source decision live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$SWITCH_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$SWITCH_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$SWITCH_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$SWITCH_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$SWITCH_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$SWITCH_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$SWITCH_REPORT" "source gate closed"
require_grep '"passed": true' "$SWITCH_REPORT" "switch pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch\] pass:' "$SWITCH_LOG" "switch pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_report=$PROMOTION_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_report=$SWITCH_REPORT"
