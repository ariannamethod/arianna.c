#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_decision_smoke.sh - decide weighted Resonance graft admission from proof precondition.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_DECISION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-decision.XXXXXX")}"
PRECONDITION_WORKDIR="$WORKDIR/precondition"
GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_proof_precondition.json"
GRAFT_ADMISSION_DECISION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_DECISION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_decision.json}"
PRECONDITION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_proof_precondition.log"
DECISION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_decision.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-decision-smoke] FAIL: $*" >&2
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 500 "$PRECONDITION_LOG" >&2 || true
    fi
    if [[ -f "$DECISION_LOG" ]]; then
        tail -n 220 "$DECISION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_WORKDIR="$PRECONDITION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT="$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_proof_precondition_smoke.sh" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted admission resonance graft admission proof precondition producer failed"
fi

[[ -s "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" ]] || die "weighted admission resonance graft admission proof precondition report not written: $GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_decision.sh" "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "$GRAFT_ADMISSION_DECISION_REPORT" >"$DECISION_LOG" 2>&1; then
    die "weighted admission resonance graft admission decision rejected proof precondition report"
fi

[[ -s "$GRAFT_ADMISSION_DECISION_REPORT" ]] || die "weighted admission resonance graft admission decision report not written: $GRAFT_ADMISSION_DECISION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision schema"
require_grep '"status": "shadow_graft_admission_decision_ready_dry_run"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_decision"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision target kind"
require_grep '"target_mode": "closed_decision_receipt_dry_run"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision target mode"
require_grep '"action": "decide_weighted_resonance_shadow_graft_admission_dry_run"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision action"
require_grep '"decision": "shadow_ready"' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision verdict"
require_grep '"weighted_admission_resonance_graft_admission_decision_ready": true' "$GRAFT_ADMISSION_DECISION_REPORT" "admission decision ready"
require_grep '"weighted_admission_resonance_graft_admission_proof_precondition_consumed": true' "$GRAFT_ADMISSION_DECISION_REPORT" "precondition consumed"
require_grep '"weighted_admission_resonance_graft_admission_proof_precondition_required": true' "$GRAFT_ADMISSION_DECISION_REPORT" "precondition required"
require_grep '"next_step_blocked_without_resonance_graft_admission_decision": true' "$GRAFT_ADMISSION_DECISION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_decision_id": "weighted-resonance-graft-admission-decision-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "decision id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_decision_receipt"' "$GRAFT_ADMISSION_DECISION_REPORT" "receipt shape"
require_grep '"decision_kind": "shadow_graft_admission_decision"' "$GRAFT_ADMISSION_DECISION_REPORT" "decision kind"
require_grep '"decision_mode": "closed_precondition_decision"' "$GRAFT_ADMISSION_DECISION_REPORT" "decision mode"
require_grep '"decision_stage": "pre_live_graft_admission_decision"' "$GRAFT_ADMISSION_DECISION_REPORT" "decision stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-decision-causal-' "$GRAFT_ADMISSION_DECISION_REPORT" "causal id"
require_grep '"decision_hash": "weighted-resonance-graft-admission-decision-' "$GRAFT_ADMISSION_DECISION_REPORT" "decision hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-decision-read-' "$GRAFT_ADMISSION_DECISION_REPORT" "read-back hash"
require_grep '"proof_precondition_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "precondition verification"
require_grep '"precondition_hash_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "precondition hash verification"
require_grep '"precondition_read_back_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "precondition read-back verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "proof verification"
require_grep '"proof_hash_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "proof hash verification"
require_grep '"proof_read_back_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "proof read-back verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "store-reader verification"
require_grep '"store_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "store verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_DECISION_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_DECISION_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_DECISION_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_DECISION_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_DECISION_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "raw dream text allow guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_DECISION_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_DECISION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_DECISION_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition schema"
require_grep '"source_status": "shadow_graft_admission_proof_precondition_satisfied_dry_run"' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition status"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_id": "weighted-resonance-graft-admission-proof-precondition-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_ready": true' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition ready"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_hash": "weighted-resonance-graft-admission-proof-precondition-' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition hash"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_read_back_hash": "weighted-resonance-graft-admission-proof-precondition-read-' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition read-back"
require_grep '"source_precondition_kind": "shadow_graft_admission_proof_precondition"' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition kind"
require_grep '"source_precondition_graft_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "source precondition graft guard"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_ready": true' "$GRAFT_ADMISSION_DECISION_REPORT" "source proof ready"
require_grep '"source_proof_kind": "shadow_graft_admission_proof"' "$GRAFT_ADMISSION_DECISION_REPORT" "source proof kind"
require_grep '"source_proof_graft_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "source proof graft guard"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_DECISION_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_DECISION_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_DECISION_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_DECISION_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_DECISION_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_DECISION_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_DECISION_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_DECISION_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_DECISION_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_DECISION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_DECISION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_DECISION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_DECISION_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_DECISION_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_DECISION_REPORT" "decision pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-decision\] pass:' "$DECISION_LOG" "decision pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-decision-smoke] pass: resonance_graft_admission_proof_precondition_report=$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT resonance_graft_admission_decision_report=$GRAFT_ADMISSION_DECISION_REPORT"
