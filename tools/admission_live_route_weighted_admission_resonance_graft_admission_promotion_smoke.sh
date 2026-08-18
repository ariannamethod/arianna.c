#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_promotion_smoke.sh - promote weighted Resonance graft admission from decision.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROMOTION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-promotion.XXXXXX")}"
DECISION_WORKDIR="$WORKDIR/decision"
GRAFT_ADMISSION_DECISION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_decision.json"
GRAFT_ADMISSION_PROMOTION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_promotion.json}"
DECISION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_decision.log"
PROMOTION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_promotion.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-promotion-smoke] FAIL: $*" >&2
    if [[ -f "$DECISION_LOG" ]]; then
        tail -n 500 "$DECISION_LOG" >&2 || true
    fi
    if [[ -f "$PROMOTION_LOG" ]]; then
        tail -n 220 "$PROMOTION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_DECISION_WORKDIR="$DECISION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_DECISION_REPORT="$GRAFT_ADMISSION_DECISION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_decision_smoke.sh" >"$DECISION_LOG" 2>&1; then
    die "weighted admission resonance graft admission decision producer failed"
fi

[[ -s "$GRAFT_ADMISSION_DECISION_REPORT" ]] || die "weighted admission resonance graft admission decision report not written: $GRAFT_ADMISSION_DECISION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_promotion.sh" "$GRAFT_ADMISSION_DECISION_REPORT" "$GRAFT_ADMISSION_PROMOTION_REPORT" >"$PROMOTION_LOG" 2>&1; then
    die "weighted admission resonance graft admission promotion rejected decision report"
fi

[[ -s "$GRAFT_ADMISSION_PROMOTION_REPORT" ]] || die "weighted admission resonance graft admission promotion report not written: $GRAFT_ADMISSION_PROMOTION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion schema"
require_grep '"status": "shadow_graft_admission_promotion_ready_dry_run"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_promotion"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion target kind"
require_grep '"target_mode": "closed_promotion_receipt_dry_run"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion target mode"
require_grep '"action": "promote_weighted_resonance_shadow_graft_admission_dry_run"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion action"
require_grep '"promotion": "pending_live_admission"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion verdict"
require_grep '"weighted_admission_resonance_graft_admission_promotion_ready": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission promotion ready"
require_grep '"weighted_admission_resonance_graft_admission_decision_consumed": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "decision consumed"
require_grep '"weighted_admission_resonance_graft_admission_decision_required": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "decision required"
require_grep '"next_step_blocked_without_resonance_graft_admission_promotion": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_promotion_id": "weighted-resonance-graft-admission-promotion-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_promotion_receipt"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "receipt shape"
require_grep '"promotion_kind": "shadow_graft_admission_promotion"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion kind"
require_grep '"promotion_mode": "closed_decision_promotion"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion mode"
require_grep '"promotion_stage": "pre_live_graft_admission_promotion"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-promotion-causal-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "causal id"
require_grep '"promotion_hash": "weighted-resonance-graft-admission-promotion-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-promotion-read-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "read-back hash"
require_grep '"decision_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "decision verification"
require_grep '"decision_hash_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "decision hash verification"
require_grep '"decision_read_back_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "decision read-back verification"
require_grep '"proof_precondition_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "candidate verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "raw dream text allow guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision schema"
require_grep '"source_status": "shadow_graft_admission_decision_ready_dry_run"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision status"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_id": "weighted-resonance-graft-admission-decision-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision id"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_ready": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision ready"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_hash": "weighted-resonance-graft-admission-decision-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision hash"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_read_back_hash": "weighted-resonance-graft-admission-decision-read-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision read-back"
require_grep '"source_decision": "shadow_ready"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision verdict"
require_grep '"source_decision_kind": "shadow_graft_admission_decision"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision kind"
require_grep '"source_decision_graft_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision graft guard"
require_grep '"source_decision_write_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision writer guard"
require_grep '"source_decision_admission_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision admission guard"
require_grep '"source_decision_live_admission_enabled": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision live guard"
require_grep '"source_decision_mutates_state": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source decision non-mutation flag"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_id": "weighted-resonance-graft-admission-proof-precondition-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_PROMOTION_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_PROMOTION_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_PROMOTION_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_PROMOTION_REPORT" "promotion pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-promotion\] pass:' "$PROMOTION_LOG" "promotion pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-promotion-smoke] pass: resonance_graft_admission_decision_report=$GRAFT_ADMISSION_DECISION_REPORT resonance_graft_admission_promotion_report=$GRAFT_ADMISSION_PROMOTION_REPORT"
