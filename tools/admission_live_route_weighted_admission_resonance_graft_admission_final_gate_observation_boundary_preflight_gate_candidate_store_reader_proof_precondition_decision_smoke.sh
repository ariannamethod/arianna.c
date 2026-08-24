#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_smoke.sh - produce and validate closed shadow decision from reader-proof precondition.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision.XXXXXX")}"
PRECONDITION_WORKDIR="$WORKDIR/precondition"
PRECONDITION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.json"
DECISION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.json}"
PRECONDITION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.log"
DECISION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-smoke] FAIL: $*" >&2
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 500 "$PRECONDITION_LOG" >&2 || true
    fi
    if [[ -f "$DECISION_LOG" ]]; then
        tail -n 260 "$DECISION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_WORKDIR="$PRECONDITION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT="$PRECONDITION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_smoke.sh" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition producer failed"
fi

[[ -s "$PRECONDITION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition report not written: $PRECONDITION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.sh" "$PRECONDITION_REPORT" "$DECISION_REPORT" >"$DECISION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision rejected precondition report"
fi

[[ -s "$DECISION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision report not written: $DECISION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_assert.sh" "$DECISION_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v1"' "$DECISION_REPORT" "decision schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run"' "$DECISION_REPORT" "decision status"
require_grep '"target": "live_route_admission_next_step"' "$DECISION_REPORT" "decision target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision"' "$DECISION_REPORT" "decision target kind"
require_grep '"target_mode": "closed_decision_receipt_dry_run"' "$DECISION_REPORT" "decision target mode"
require_grep '"action": "decide_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_dry_run"' "$DECISION_REPORT" "decision action"
require_grep '"decision": "shadow_ready"' "$DECISION_REPORT" "decision verdict"
require_grep '"ledger_state": "blocked"' "$DECISION_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$DECISION_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready": true' "$DECISION_REPORT" "decision ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_consumed": true' "$DECISION_REPORT" "precondition consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_required": true' "$DECISION_REPORT" "precondition required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision": true' "$DECISION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-' "$DECISION_REPORT" "decision id"
require_grep '"decision_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-' "$DECISION_REPORT" "decision hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-read-' "$DECISION_REPORT" "read-back hash"
require_grep '"proof_precondition_verified": true' "$DECISION_REPORT" "precondition verification"
require_grep '"precondition_hash_verified": true' "$DECISION_REPORT" "precondition hash verification"
require_grep '"precondition_read_back_verified": true' "$DECISION_REPORT" "precondition read-back verification"
require_grep '"proof_verified": true' "$DECISION_REPORT" "proof verification"
require_grep '"reader_hash_verified": true' "$DECISION_REPORT" "reader hash verification"
require_grep '"reader_replay_verified": true' "$DECISION_REPORT" "reader replay verification"
require_grep '"store_hash_verified": true' "$DECISION_REPORT" "store hash verification"
require_grep '"admission_required": true' "$DECISION_REPORT" "admission required"
require_grep '"shadow_only": true' "$DECISION_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$DECISION_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$DECISION_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$DECISION_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$DECISION_REPORT" "raw text guard"
require_grep '"janus_surface_allowed": false' "$DECISION_REPORT" "janus guard"
require_grep '"cooc_learning_allowed": false' "$DECISION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$DECISION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$DECISION_REPORT" "body mutation guard"
require_grep '"read_only": true' "$DECISION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$DECISION_REPORT" "replay-only flag"
require_grep '"authority_granted": false' "$DECISION_REPORT" "authority guard"
require_grep '"contracts_ready": false' "$DECISION_REPORT" "contracts guard"
require_grep '"write_allowed": false' "$DECISION_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$DECISION_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$DECISION_REPORT" "live guard"
require_grep '"mutates_state": false' "$DECISION_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$DECISION_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v1"' "$DECISION_REPORT" "source precondition schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$DECISION_REPORT" "source precondition id"
require_grep '"source_precondition_ledger_append_allowed": false' "$DECISION_REPORT" "source precondition ledger guard"
require_grep '"source_precondition_graft_allowed": false' "$DECISION_REPORT" "source precondition graft guard"
require_grep '"source_precondition_live_admission_enabled": false' "$DECISION_REPORT" "source precondition live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$DECISION_REPORT" "source proof id"
require_grep '"source_proof_ledger_append_allowed": false' "$DECISION_REPORT" "source proof ledger guard"
require_grep '"source_proof_graft_allowed": false' "$DECISION_REPORT" "source proof graft guard"
require_grep '"source_proof_live_admission_enabled": false' "$DECISION_REPORT" "source proof live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$DECISION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$DECISION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$DECISION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$DECISION_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$DECISION_REPORT" "source gate closed"
require_grep '"passed": true' "$DECISION_REPORT" "decision pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision\] pass:' "$DECISION_LOG" "decision pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_report=$PRECONDITION_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_report=$DECISION_REPORT"
