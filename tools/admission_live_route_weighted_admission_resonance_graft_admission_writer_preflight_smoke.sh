#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight_smoke.sh - block weighted Resonance graft admission writer preflight behind blocked live stage.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-writer-preflight.XXXXXX")}"
LIVE_STAGE_WORKDIR="$WORKDIR/live_stage"
GRAFT_ADMISSION_LIVE_STAGE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_live_stage.json"
GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_writer_preflight.json}"
LIVE_STAGE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_live_stage.log"
WRITER_PREFLIGHT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_writer_preflight.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-smoke] FAIL: $*" >&2
    if [[ -f "$LIVE_STAGE_LOG" ]]; then
        tail -n 500 "$LIVE_STAGE_LOG" >&2 || true
    fi
    if [[ -f "$WRITER_PREFLIGHT_LOG" ]]; then
        tail -n 220 "$WRITER_PREFLIGHT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_WORKDIR="$LIVE_STAGE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT="$GRAFT_ADMISSION_LIVE_STAGE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_live_stage_smoke.sh" >"$LIVE_STAGE_LOG" 2>&1; then
    die "weighted admission resonance graft admission live stage producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" ]] || die "weighted admission resonance graft admission live stage report not written: $GRAFT_ADMISSION_LIVE_STAGE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight.sh" "$GRAFT_ADMISSION_LIVE_STAGE_REPORT" "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" >"$WRITER_PREFLIGHT_LOG" 2>&1; then
    die "weighted admission resonance graft admission writer preflight rejected live stage report"
fi

[[ -s "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft admission writer preflight report not written: $GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight schema"
require_grep '"status": "shadow_graft_admission_writer_preflight_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_writer_preflight"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight target kind"
require_grep '"target_mode": "closed_writer_preflight_guard_dry_run"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_live_stage_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight action"
require_grep '"writer_state": "blocked"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer state"
require_grep '"writer_action": "reject_blocked_live_stage"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer action"
require_grep '"rollback_state": "blocked"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "rollback state"
require_grep '"rollback_action": "reject_blocked_live_stage"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "rollback action"
require_grep '"stage_state": "blocked"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source stage state"
require_grep '"stage_action": "reject_disabled_enable_gate"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source stage action"
require_grep '"enable_state": "disabled"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "enable gate state"
require_grep '"enable_action": "require_operator_key"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "enable gate action"
require_grep '"switch_state": "disabled"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source switch action"
require_grep '"promotion": "pending_live_admission"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission promotion"
require_grep '"weighted_admission_resonance_graft_admission_writer_preflight_ready": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission writer preflight ready"
require_grep '"weighted_admission_resonance_graft_admission_live_stage_consumed": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live stage consumed"
require_grep '"weighted_admission_resonance_graft_admission_live_stage_required": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live stage required"
require_grep '"next_step_blocked_without_resonance_graft_admission_writer_preflight": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_writer_preflight_id": "weighted-resonance-graft-admission-writer-preflight-id-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_writer_preflight_receipt"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "receipt shape"
require_grep '"writer_preflight_kind": "shadow_graft_admission_writer_preflight"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight kind"
require_grep '"writer_preflight_mode": "closed_live_stage_writer_preflight_guard"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight mode"
require_grep '"writer_preflight_stage": "pre_writer_inventory_graft_admission_writer_preflight"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-writer-preflight-causal-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "causal id"
require_grep '"writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-writer-preflight-read-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "read-back hash"
require_grep '"live_stage_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live stage verification"
require_grep '"live_stage_hash_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live stage hash verification"
require_grep '"live_stage_read_back_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live stage read-back verification"
require_grep '"enable_gate_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "enable gate verification"
require_grep '"switch_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "switch verification"
require_grep '"promotion_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "decision verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "proof verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "body mutation guard"
require_grep '"requires_writer": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer closed flag"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "rollback requirement"
require_grep '"requires_rollback": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "rollback proof requirement"
require_grep '"rollback_ready": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "rollback closed flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage schema"
require_grep '"source_status": "shadow_graft_admission_live_stage_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage status"
require_grep '"source_weighted_admission_resonance_graft_admission_live_stage_id": "weighted-resonance-graft-admission-live-stage-id-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage id"
require_grep '"source_weighted_admission_resonance_graft_admission_live_stage_hash": "weighted-resonance-graft-admission-live-stage-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage hash"
require_grep '"source_weighted_admission_resonance_graft_admission_live_stage_read_back_hash": "weighted-resonance-graft-admission-live-stage-read-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage read-back"
require_grep '"source_stage_state": "blocked"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source stage state"
require_grep '"source_stage_action": "reject_disabled_enable_gate"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source stage action"
require_grep '"source_live_stage_kind": "shadow_graft_admission_live_stage"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage kind"
require_grep '"source_live_stage_mode": "closed_enable_gate_live_stage_guard"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage mode"
require_grep '"source_live_stage_stage": "pre_writer_graft_admission_live_stage"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage stage"
require_grep '"source_live_stage_writer_ready": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage writer guard"
require_grep '"source_live_stage_rollback_ready": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage rollback guard"
require_grep '"source_live_stage_write_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage write guard"
require_grep '"source_live_stage_admission_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage admission guard"
require_grep '"source_live_stage_live_admission_enabled": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage live guard"
require_grep '"source_live_stage_body_target": "none"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage body target"
require_grep '"source_live_stage_reason": "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source live stage reason"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_id": "weighted-resonance-graft-admission-enable-gate-id-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source enable gate id"
require_grep '"source_enable_gate_reason": "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source enable gate reason"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_id": "weighted-resonance-graft-admission-switch-id-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source switch id"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_id": "weighted-resonance-graft-admission-promotion-id-' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "source promotion id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent"' "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "writer preflight reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight\] pass:' "$WRITER_PREFLIGHT_LOG" "writer preflight pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-smoke] pass: resonance_graft_admission_live_stage_report=$GRAFT_ADMISSION_LIVE_STAGE_REPORT resonance_graft_admission_writer_preflight_report=$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT"
