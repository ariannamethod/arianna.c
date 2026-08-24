#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_smoke.sh - prove blocked weighted Resonance admission final-gate observation-boundary preflight-gate candidate store reader.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof.XXXXXX")}"
READER_WORKDIR="$WORKDIR/reader"
READER_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.json"
PROOF_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.json}"
READER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.log"
PROOF_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-smoke] FAIL: $*" >&2
    if [[ -f "$READER_LOG" ]]; then
        tail -n 500 "$READER_LOG" >&2 || true
    fi
    if [[ -f "$PROOF_LOG" ]]; then
        tail -n 260 "$PROOF_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_WORKDIR="$READER_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT="$READER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_smoke.sh" >"$READER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader producer failed"
fi

[[ -s "$READER_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader report not written: $READER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.sh" "$READER_REPORT" "$PROOF_REPORT" >"$PROOF_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof rejected reader report"
fi

[[ -s "$PROOF_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof report not written: $PROOF_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"' "$PROOF_REPORT" "proof schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run"' "$PROOF_REPORT" "proof status"
require_grep '"target": "live_route_admission_next_step"' "$PROOF_REPORT" "proof target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof"' "$PROOF_REPORT" "proof target kind"
require_grep '"target_mode": "receipt_only_closed_reader_proof_dry_run"' "$PROOF_REPORT" "proof target mode"
require_grep '"action": "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run"' "$PROOF_REPORT" "proof action"
require_grep '"ledger_append_allowed": false' "$PROOF_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready": true' "$PROOF_REPORT" "proof ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_consumed": true' "$PROOF_REPORT" "reader consumed flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof": true' "$PROOF_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-' "$PROOF_REPORT" "proof id"
require_grep '"proof_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-' "$PROOF_REPORT" "proof hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-read-' "$PROOF_REPORT" "read-back hash"
require_grep '"store_reader_verified": true' "$PROOF_REPORT" "reader verification"
require_grep '"store_verified": true' "$PROOF_REPORT" "store verification"
require_grep '"candidate_verified": true' "$PROOF_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$PROOF_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$PROOF_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$PROOF_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$PROOF_REPORT" "observation verification"
require_grep '"reader_hash_verified": true' "$PROOF_REPORT" "reader hash verification"
require_grep '"reader_replay_verified": true' "$PROOF_REPORT" "reader replay verification"
require_grep '"reader_read_back_verified": true' "$PROOF_REPORT" "reader read-back verification"
require_grep '"read_only": true' "$PROOF_REPORT" "read-only flag"
require_grep '"replay_only": true' "$PROOF_REPORT" "replay-only flag"
require_grep '"graft_allowed": false' "$PROOF_REPORT" "graft guard"
require_grep '"raw_dream_text_allowed": false' "$PROOF_REPORT" "raw dream text guard"
require_grep '"body_mutation_allowed": false' "$PROOF_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$PROOF_REPORT" "authority guard"
require_grep '"write_allowed": false' "$PROOF_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$PROOF_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$PROOF_REPORT" "live guard"
require_grep '"mutates_state": false' "$PROOF_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$PROOF_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"' "$PROOF_REPORT" "source reader schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$PROOF_REPORT" "source reader id"
require_grep '"source_reader_ledger_append_allowed": false' "$PROOF_REPORT" "source reader ledger append guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$PROOF_REPORT" "source store id"
require_grep '"source_store_ledger_append_allowed": false' "$PROOF_REPORT" "source store ledger append guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$PROOF_REPORT" "source candidate id"
require_grep '"source_candidate_opened": false' "$PROOF_REPORT" "source candidate closed flag"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$PROOF_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$PROOF_REPORT" "source gate closed"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof\] pass:' "$PROOF_LOG" "proof pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_report=$READER_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=$PROOF_REPORT"
