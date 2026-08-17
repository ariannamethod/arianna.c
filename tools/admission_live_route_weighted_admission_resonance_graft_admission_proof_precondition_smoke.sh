#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_proof_precondition_smoke.sh - consume weighted Resonance graft admission proof as precondition.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-proof-precondition.XXXXXX")}"
PROOF_WORKDIR="$WORKDIR/proof"
GRAFT_ADMISSION_PROOF_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_proof.json"
GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_proof_precondition.json}"
PROOF_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_proof.log"
PRECONDITION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_proof_precondition.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-smoke] FAIL: $*" >&2
    if [[ -f "$PROOF_LOG" ]]; then
        tail -n 500 "$PROOF_LOG" >&2 || true
    fi
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 220 "$PRECONDITION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_WORKDIR="$PROOF_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_REPORT="$GRAFT_ADMISSION_PROOF_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_proof_smoke.sh" >"$PROOF_LOG" 2>&1; then
    die "weighted admission resonance graft admission proof producer failed"
fi

[[ -s "$GRAFT_ADMISSION_PROOF_REPORT" ]] || die "weighted admission resonance graft admission proof report not written: $GRAFT_ADMISSION_PROOF_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_proof_precondition.sh" "$GRAFT_ADMISSION_PROOF_REPORT" "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted admission resonance graft admission proof precondition rejected proof report"
fi

[[ -s "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" ]] || die "weighted admission resonance graft admission proof precondition report not written: $GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition schema"
require_grep '"status": "shadow_graft_admission_proof_precondition_satisfied_dry_run"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_proof_precondition"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition target kind"
require_grep '"target_mode": "closed_receipt_precondition_dry_run"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition target mode"
require_grep '"action": "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition action"
require_grep '"weighted_admission_resonance_graft_admission_proof_precondition_ready": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition ready flag"
require_grep '"weighted_admission_resonance_graft_admission_proof_consumed": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "proof consumed flag"
require_grep '"weighted_admission_resonance_graft_admission_proof_required": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "proof required flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_proof_precondition_id": "weighted-resonance-graft-admission-proof-precondition-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_proof_precondition_receipt"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "receipt shape"
require_grep '"precondition_kind": "shadow_graft_admission_proof_precondition"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition kind"
require_grep '"precondition_mode": "closed_receipt_consumption"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition mode"
require_grep '"precondition_stage": "pre_live_graft_admission_proof_precondition"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-proof-precondition-causal-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "causal id"
require_grep '"precondition_hash": "weighted-resonance-graft-admission-proof-precondition-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-proof-precondition-read-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "read-back hash"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "proof verification"
require_grep '"proof_hash_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "proof hash verification"
require_grep '"proof_read_back_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "proof read-back verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "store-reader verification"
require_grep '"store_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "store verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof schema"
require_grep '"source_status": "shadow_graft_admission_proof_ready_dry_run"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof status"
require_grep '"source_target": "resonance"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof target"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_ready": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof ready"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_causal_id": "weighted-resonance-graft-admission-proof-causal-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof causal"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_hash": "weighted-resonance-graft-admission-proof-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof hash"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_read_back_hash": "weighted-resonance-graft-admission-proof-read-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof read-back"
require_grep '"source_proof_action": "prove_weighted_resonance_shadow_graft_admission_dry_run"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof action"
require_grep '"source_proof_receipt_shape": "weighted_resonance_shadow_graft_admission_proof_receipt"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof receipt"
require_grep '"source_proof_kind": "shadow_graft_admission_proof"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof kind"
require_grep '"source_proof_mode": "closed_read_back_admission_proof"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof mode"
require_grep '"source_proof_stage": "pre_live_graft_admission_proof"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof stage"
require_grep '"source_proof_graft_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof graft guard"
require_grep '"source_proof_live_admission_enabled": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source proof live guard"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT" "precondition pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition\] pass:' "$PRECONDITION_LOG" "precondition pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-smoke] pass: resonance_graft_admission_proof_report=$GRAFT_ADMISSION_PROOF_REPORT resonance_graft_admission_proof_precondition_report=$GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT"
