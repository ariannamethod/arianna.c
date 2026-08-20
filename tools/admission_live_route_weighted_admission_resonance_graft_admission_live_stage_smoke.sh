#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_live_stage_smoke.sh - block weighted Resonance graft admission live stage behind disabled enable gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-live-stage.XXXXXX")}"
ENABLE_GATE_WORKDIR="$WORKDIR/enable_gate"
GRAFT_ADMISSION_ENABLE_GATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_enable_gate.json"
GRAFT_ADMISSION_LIVE_STAGE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_live_stage.json}"
ENABLE_GATE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_enable_gate.log"
LIVE_STAGE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_live_stage.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-live-stage-smoke] FAIL: $*" >&2
    if [[ -f "$ENABLE_GATE_LOG" ]]; then
        tail -n 500 "$ENABLE_GATE_LOG" >&2 || true
    fi
    if [[ -f "$LIVE_STAGE_LOG" ]]; then
        tail -n 220 "$LIVE_STAGE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_WORKDIR="$ENABLE_GATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT="$GRAFT_ADMISSION_ENABLE_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_enable_gate_smoke.sh" >"$ENABLE_GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission enable gate producer failed"
fi

[[ -s "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" ]] || die "weighted admission resonance graft admission enable gate report not written: $GRAFT_ADMISSION_ENABLE_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_live_stage.sh" "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" >"$LIVE_STAGE_LOG" 2>&1; then
    die "weighted admission resonance graft admission live stage rejected enable gate report"
fi

[[ -s "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" ]] || die "weighted admission resonance graft admission live stage report not written: $GRAFT_ADMISSION_LIVE_STAGE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage schema"
require_grep '"status": "shadow_graft_admission_live_stage_blocked_dry_run"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_live_stage"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage target kind"
require_grep '"target_mode": "closed_live_stage_guard_dry_run"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_enable_gate_disabled_dry_run"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage action"
require_grep '"stage_state": "blocked"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage state"
require_grep '"stage_action": "reject_disabled_enable_gate"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage action"
require_grep '"enable_state": "disabled"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate state"
require_grep '"enable_action": "require_operator_key"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate action"
require_grep '"switch_state": "disabled"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source switch action"
require_grep '"promotion": "pending_live_admission"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission promotion"
require_grep '"weighted_admission_resonance_graft_admission_live_stage_ready": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission live stage ready"
require_grep '"weighted_admission_resonance_graft_admission_enable_gate_consumed": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate consumed"
require_grep '"weighted_admission_resonance_graft_admission_enable_gate_required": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate required"
require_grep '"next_step_blocked_without_resonance_graft_admission_live_stage": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_live_stage_id": "weighted-resonance-graft-admission-live-stage-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_live_stage_receipt"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "receipt shape"
require_grep '"live_stage_kind": "shadow_graft_admission_live_stage"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage kind"
require_grep '"live_stage_mode": "closed_enable_gate_live_stage_guard"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage mode"
require_grep '"live_stage_stage": "pre_writer_graft_admission_live_stage"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-live-stage-causal-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "causal id"
require_grep '"live_stage_hash": "weighted-resonance-graft-admission-live-stage-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-live-stage-read-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "read-back hash"
require_grep '"enable_gate_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate verification"
require_grep '"enable_gate_hash_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate hash verification"
require_grep '"enable_gate_read_back_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "enable gate read-back verification"
require_grep '"switch_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "switch verification"
require_grep '"switch_hash_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "switch hash verification"
require_grep '"switch_read_back_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "switch read-back verification"
require_grep '"promotion_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "decision verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "candidate verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "body mutation guard"
require_grep '"requires_writer": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "writer closed flag"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "rollback requirement"
require_grep '"requires_rollback": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "rollback proof requirement"
require_grep '"rollback_ready": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "rollback closed flag"
require_grep '"read_only": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate schema"
require_grep '"source_status": "shadow_graft_admission_enable_gate_disabled_dry_run"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate status"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_id": "weighted-resonance-graft-admission-enable-gate-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate id"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_ready": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate ready"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate hash"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash": "weighted-resonance-graft-admission-enable-gate-read-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate read-back"
require_grep '"source_enable_state": "disabled"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate state"
require_grep '"source_enable_action": "require_operator_key"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate action"
require_grep '"source_enable_gate_kind": "shadow_graft_admission_enable_gate"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate kind"
require_grep '"source_enable_gate_mode": "closed_switch_enable_guard"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate mode"
require_grep '"source_enable_gate_stage": "pre_live_graft_admission_enable_gate"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate stage"
require_grep '"source_enable_gate_graft_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate graft guard"
require_grep '"source_enable_gate_write_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate writer guard"
require_grep '"source_enable_gate_admission_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate admission guard"
require_grep '"source_enable_gate_live_admission_enabled": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate live guard"
require_grep '"source_enable_gate_mutates_state": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate mutation guard"
require_grep '"source_enable_gate_body_target": "none"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate body target"
require_grep '"source_enable_gate_reason": "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source enable gate reason"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_id": "weighted-resonance-graft-admission-switch-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source switch id"
require_grep '"source_switch_state": "disabled"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source switch action"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_id": "weighted-resonance-graft-admission-promotion-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source promotion id"
require_grep '"source_promotion": "pending_live_admission"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source promotion verdict"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_id": "weighted-resonance-graft-admission-decision-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source decision id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "live stage pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-live-stage\] pass:' "$LIVE_STAGE_LOG" "live stage pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-live-stage-smoke] pass: resonance_graft_admission_enable_gate_report=$GRAFT_ADMISSION_ENABLE_GATE_REPORT resonance_graft_admission_live_stage_report=$GRAFT_ADMISSION_LIVE_STAGE_REPORT"
