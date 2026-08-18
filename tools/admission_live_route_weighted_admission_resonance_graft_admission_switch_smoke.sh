#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_switch_smoke.sh - hold weighted Resonance graft admission promotion at disabled switch.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SWITCH_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-switch.XXXXXX")}"
PROMOTION_WORKDIR="$WORKDIR/promotion"
GRAFT_ADMISSION_PROMOTION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_promotion.json"
GRAFT_ADMISSION_SWITCH_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_switch.json}"
PROMOTION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_promotion.log"
SWITCH_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_switch.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-switch-smoke] FAIL: $*" >&2
    if [[ -f "$PROMOTION_LOG" ]]; then
        tail -n 500 "$PROMOTION_LOG" >&2 || true
    fi
    if [[ -f "$SWITCH_LOG" ]]; then
        tail -n 220 "$SWITCH_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROMOTION_WORKDIR="$PROMOTION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT="$GRAFT_ADMISSION_PROMOTION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_promotion_smoke.sh" >"$PROMOTION_LOG" 2>&1; then
    die "weighted admission resonance graft admission promotion producer failed"
fi

[[ -s "$GRAFT_ADMISSION_PROMOTION_REPORT" ]] || die "weighted admission resonance graft admission promotion report not written: $GRAFT_ADMISSION_PROMOTION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_switch.sh" "$GRAFT_ADMISSION_PROMOTION_REPORT" "$GRAFT_ADMISSION_SWITCH_REPORT" >"$SWITCH_LOG" 2>&1; then
    die "weighted admission resonance graft admission switch rejected promotion report"
fi

[[ -s "$GRAFT_ADMISSION_SWITCH_REPORT" ]] || die "weighted admission resonance graft admission switch report not written: $GRAFT_ADMISSION_SWITCH_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch schema"
require_grep '"status": "shadow_graft_admission_switch_disabled_dry_run"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_switch"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch target kind"
require_grep '"target_mode": "closed_switch_guard_dry_run"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch target mode"
require_grep '"action": "hold_weighted_resonance_shadow_graft_admission_promotion_dry_run"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch action"
require_grep '"switch_state": "disabled"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch hold action"
require_grep '"promotion": "pending_live_admission"' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch promotion"
require_grep '"weighted_admission_resonance_graft_admission_switch_ready": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission switch ready"
require_grep '"weighted_admission_resonance_graft_admission_promotion_consumed": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "promotion consumed"
require_grep '"weighted_admission_resonance_graft_admission_promotion_required": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "promotion required"
require_grep '"next_step_blocked_without_resonance_graft_admission_switch": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_switch_id": "weighted-resonance-graft-admission-switch-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_switch_receipt"' "$GRAFT_ADMISSION_SWITCH_REPORT" "receipt shape"
require_grep '"switch_kind": "shadow_graft_admission_switch"' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch kind"
require_grep '"switch_mode": "closed_promotion_switch_guard"' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch mode"
require_grep '"switch_stage": "pre_live_graft_admission_switch"' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-switch-causal-' "$GRAFT_ADMISSION_SWITCH_REPORT" "causal id"
require_grep '"switch_hash": "weighted-resonance-graft-admission-switch-' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-switch-read-' "$GRAFT_ADMISSION_SWITCH_REPORT" "read-back hash"
require_grep '"promotion_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "promotion verification"
require_grep '"promotion_hash_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "promotion hash verification"
require_grep '"promotion_read_back_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "promotion read-back verification"
require_grep '"decision_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "decision verification"
require_grep '"proof_precondition_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "precondition verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "candidate verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion schema"
require_grep '"source_status": "shadow_graft_admission_promotion_ready_dry_run"' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion status"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_id": "weighted-resonance-graft-admission-promotion-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion id"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_ready": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion ready"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_hash": "weighted-resonance-graft-admission-promotion-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion hash"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash": "weighted-resonance-graft-admission-promotion-read-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion read-back"
require_grep '"source_promotion": "pending_live_admission"' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion verdict"
require_grep '"source_promotion_kind": "shadow_graft_admission_promotion"' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion kind"
require_grep '"source_promotion_graft_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion graft guard"
require_grep '"source_promotion_write_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion writer guard"
require_grep '"source_promotion_admission_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion admission guard"
require_grep '"source_promotion_live_admission_enabled": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion live guard"
require_grep '"source_promotion_mutates_state": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "source promotion non-mutation flag"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_id": "weighted-resonance-graft-admission-decision-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source decision id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_precondition_id": "weighted-resonance-graft-admission-proof-precondition-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source precondition id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_SWITCH_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_SWITCH_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_SWITCH_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_SWITCH_REPORT" "switch pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-switch\] pass:' "$SWITCH_LOG" "switch pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-switch-smoke] pass: resonance_graft_admission_promotion_report=$GRAFT_ADMISSION_PROMOTION_REPORT resonance_graft_admission_switch_report=$GRAFT_ADMISSION_SWITCH_REPORT"
