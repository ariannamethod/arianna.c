#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_smoke.sh - consume blocked weighted Resonance admission final-gate observation-boundary preflight-gate candidate store reader proof as a closed precondition.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition.XXXXXX")}"
PROOF_WORKDIR="$WORKDIR/proof"
PROOF_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.json"
PRECONDITION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.json}"
PROOF_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.log"
PRECONDITION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.log"
ASSERT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_assert.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-smoke] FAIL: $*" >&2
    if [[ -f "$PROOF_LOG" ]]; then
        tail -n 500 "$PROOF_LOG" >&2 || true
    fi
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 260 "$PRECONDITION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_WORKDIR="$PROOF_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT="$PROOF_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_smoke.sh" >"$PROOF_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof producer failed"
fi

[[ -s "$PROOF_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof report not written: $PROOF_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.sh" "$PROOF_REPORT" "$PRECONDITION_REPORT" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition rejected proof report"
fi

[[ -s "$PRECONDITION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition report not written: $PRECONDITION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_assert.sh" "$PRECONDITION_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition assert rejected producer report"
fi

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition.v1"' "$PRECONDITION_REPORT" "precondition schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run"' "$PRECONDITION_REPORT" "precondition status"
require_grep '"target": "live_route_admission_next_step"' "$PRECONDITION_REPORT" "precondition target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition"' "$PRECONDITION_REPORT" "precondition target kind"
require_grep '"target_mode": "closed_receipt_precondition_dry_run"' "$PRECONDITION_REPORT" "precondition target mode"
require_grep '"action": "consume_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_before_live_route_admission"' "$PRECONDITION_REPORT" "precondition action"
require_grep '"ledger_state": "blocked"' "$PRECONDITION_REPORT" "ledger state"
require_grep '"ledger_append_allowed": false' "$PRECONDITION_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready": true' "$PRECONDITION_REPORT" "precondition ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_consumed": true' "$PRECONDITION_REPORT" "proof consumed flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition": true' "$PRECONDITION_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-' "$PRECONDITION_REPORT" "precondition id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-causal-' "$PRECONDITION_REPORT" "causal id"
require_grep '"precondition_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-' "$PRECONDITION_REPORT" "precondition hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-read-' "$PRECONDITION_REPORT" "read-back hash"
require_grep '"proof_verified": true' "$PRECONDITION_REPORT" "proof verified"
require_grep '"proof_hash_verified": true' "$PRECONDITION_REPORT" "proof hash verified"
require_grep '"proof_read_back_verified": true' "$PRECONDITION_REPORT" "proof read-back verified"
require_grep '"store_reader_verified": true' "$PRECONDITION_REPORT" "reader verification"
require_grep '"store_verified": true' "$PRECONDITION_REPORT" "store verification"
require_grep '"candidate_verified": true' "$PRECONDITION_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$PRECONDITION_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$PRECONDITION_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$PRECONDITION_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$PRECONDITION_REPORT" "observation verification"
require_grep '"reader_hash_verified": true' "$PRECONDITION_REPORT" "reader hash verification"
require_grep '"reader_replay_verified": true' "$PRECONDITION_REPORT" "reader replay verification"
require_grep '"reader_read_back_verified": true' "$PRECONDITION_REPORT" "reader read-back verification"
require_grep '"store_hash_verified": true' "$PRECONDITION_REPORT" "store hash verification"
require_grep '"store_read_back_verified": true' "$PRECONDITION_REPORT" "store read-back verification"
require_grep '"read_only": true' "$PRECONDITION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$PRECONDITION_REPORT" "replay-only flag"
require_grep '"graft_allowed": false' "$PRECONDITION_REPORT" "graft guard"
require_grep '"raw_dream_text_allowed": false' "$PRECONDITION_REPORT" "raw dream text guard"
require_grep '"janus_surface_allowed": false' "$PRECONDITION_REPORT" "janus guard"
require_grep '"cooc_learning_allowed": false' "$PRECONDITION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$PRECONDITION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$PRECONDITION_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$PRECONDITION_REPORT" "authority guard"
require_grep '"contracts_ready": false' "$PRECONDITION_REPORT" "contract guard"
require_grep '"write_allowed": false' "$PRECONDITION_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$PRECONDITION_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$PRECONDITION_REPORT" "live guard"
require_grep '"mutates_state": false' "$PRECONDITION_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$PRECONDITION_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"' "$PRECONDITION_REPORT" "source proof schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$PRECONDITION_REPORT" "source proof id"
require_grep '"source_proof_ledger_append_allowed": false' "$PRECONDITION_REPORT" "source proof ledger append guard"
require_grep '"source_proof_graft_allowed": false' "$PRECONDITION_REPORT" "source proof graft guard"
require_grep '"source_proof_raw_dream_text_allowed": false' "$PRECONDITION_REPORT" "source proof raw text guard"
require_grep '"source_proof_body_mutation_allowed": false' "$PRECONDITION_REPORT" "source proof body mutation guard"
require_grep '"source_proof_write_allowed": false' "$PRECONDITION_REPORT" "source proof writer guard"
require_grep '"source_proof_admission_allowed": false' "$PRECONDITION_REPORT" "source proof admission guard"
require_grep '"source_proof_live_admission_enabled": false' "$PRECONDITION_REPORT" "source proof live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$PRECONDITION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$PRECONDITION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$PRECONDITION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$PRECONDITION_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$PRECONDITION_REPORT" "source gate closed"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition\] pass:' "$PRECONDITION_LOG" "precondition pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=$PROOF_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_report=$PRECONDITION_REPORT"
