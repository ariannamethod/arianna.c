#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_smoke.sh - produce and validate closed promotion from reader-proof precondition decision.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion.XXXXXX")}"
DECISION_WORKDIR="$WORKDIR/decision"
DECISION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.json"
PROMOTION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.json}"
DECISION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.log"
PROMOTION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-smoke] FAIL: $*" >&2
    if [[ -f "$DECISION_LOG" ]]; then
        tail -n 500 "$DECISION_LOG" >&2 || true
    fi
    if [[ -f "$PROMOTION_LOG" ]]; then
        tail -n 260 "$PROMOTION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_WORKDIR="$DECISION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_REPORT="$DECISION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_smoke.sh" >"$DECISION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision producer failed"
fi

[[ -s "$DECISION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision report not written: $DECISION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.sh" "$DECISION_REPORT" "$PROMOTION_REPORT" >"$PROMOTION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion rejected decision report"
fi

[[ -s "$PROMOTION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion report not written: $PROMOTION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_assert.sh" "$PROMOTION_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion.v1"' "$PROMOTION_REPORT" "promotion schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready_dry_run"' "$PROMOTION_REPORT" "promotion status"
require_grep '"target": "live_route_admission_next_step"' "$PROMOTION_REPORT" "promotion target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion"' "$PROMOTION_REPORT" "promotion target kind"
require_grep '"target_mode": "closed_promotion_receipt_dry_run"' "$PROMOTION_REPORT" "promotion target mode"
require_grep '"action": "promote_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_dry_run"' "$PROMOTION_REPORT" "promotion action"
require_grep '"promotion": "pending_live_admission"' "$PROMOTION_REPORT" "promotion verdict"
require_grep '"ledger_state": "blocked"' "$PROMOTION_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$PROMOTION_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready": true' "$PROMOTION_REPORT" "promotion ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_consumed": true' "$PROMOTION_REPORT" "decision consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_required": true' "$PROMOTION_REPORT" "decision required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion": true' "$PROMOTION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-' "$PROMOTION_REPORT" "promotion id"
require_grep '"promotion_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-' "$PROMOTION_REPORT" "promotion hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-read-' "$PROMOTION_REPORT" "read-back hash"
require_grep '"decision_verified": true' "$PROMOTION_REPORT" "decision verification"
require_grep '"decision_hash_verified": true' "$PROMOTION_REPORT" "decision hash verification"
require_grep '"decision_read_back_verified": true' "$PROMOTION_REPORT" "decision read-back verification"
require_grep '"proof_precondition_verified": true' "$PROMOTION_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$PROMOTION_REPORT" "proof verification"
require_grep '"reader_hash_verified": true' "$PROMOTION_REPORT" "reader hash verification"
require_grep '"store_hash_verified": true' "$PROMOTION_REPORT" "store hash verification"
require_grep '"admission_required": true' "$PROMOTION_REPORT" "admission required"
require_grep '"shadow_only": true' "$PROMOTION_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$PROMOTION_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$PROMOTION_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$PROMOTION_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$PROMOTION_REPORT" "raw text guard"
require_grep '"janus_surface_allowed": false' "$PROMOTION_REPORT" "janus guard"
require_grep '"cooc_learning_allowed": false' "$PROMOTION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$PROMOTION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$PROMOTION_REPORT" "body mutation guard"
require_grep '"read_only": true' "$PROMOTION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$PROMOTION_REPORT" "replay-only flag"
require_grep '"authority_granted": false' "$PROMOTION_REPORT" "authority guard"
require_grep '"contracts_ready": false' "$PROMOTION_REPORT" "contracts guard"
require_grep '"write_allowed": false' "$PROMOTION_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$PROMOTION_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$PROMOTION_REPORT" "live guard"
require_grep '"mutates_state": false' "$PROMOTION_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$PROMOTION_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v1"' "$PROMOTION_REPORT" "source decision schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$PROMOTION_REPORT" "source decision id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-' "$PROMOTION_REPORT" "source decision hash"
require_grep '"source_decision_ledger_append_allowed": false' "$PROMOTION_REPORT" "source decision ledger guard"
require_grep '"source_decision_graft_allowed": false' "$PROMOTION_REPORT" "source decision graft guard"
require_grep '"source_decision_write_allowed": false' "$PROMOTION_REPORT" "source decision writer guard"
require_grep '"source_decision_live_admission_enabled": false' "$PROMOTION_REPORT" "source decision live guard"
require_grep '"source_decision_body_mutation_allowed": false' "$PROMOTION_REPORT" "source decision body guard"
require_grep '"source_decision_body_target": "none"' "$PROMOTION_REPORT" "source decision body target"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$PROMOTION_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$PROMOTION_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$PROMOTION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$PROMOTION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$PROMOTION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$PROMOTION_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$PROMOTION_REPORT" "source gate closed"
require_grep '"passed": true' "$PROMOTION_REPORT" "promotion pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion\] pass:' "$PROMOTION_LOG" "promotion pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_report=$DECISION_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_report=$PROMOTION_REPORT"
