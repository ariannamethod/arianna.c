#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_enable_gate_smoke.sh - keep weighted Resonance graft admission behind disabled enable gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-enable-gate.XXXXXX")}"
SWITCH_WORKDIR="$WORKDIR/switch"
GRAFT_ADMISSION_SWITCH_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_switch.json"
GRAFT_ADMISSION_ENABLE_GATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_enable_gate.json}"
SWITCH_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_switch.log"
ENABLE_GATE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_enable_gate.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-smoke] FAIL: $*" >&2
    if [[ -f "$SWITCH_LOG" ]]; then
        tail -n 500 "$SWITCH_LOG" >&2 || true
    fi
    if [[ -f "$ENABLE_GATE_LOG" ]]; then
        tail -n 220 "$ENABLE_GATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SWITCH_WORKDIR="$SWITCH_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT="$GRAFT_ADMISSION_SWITCH_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_switch_smoke.sh" >"$SWITCH_LOG" 2>&1; then
    die "weighted admission resonance graft admission switch producer failed"
fi

[[ -s "$GRAFT_ADMISSION_SWITCH_REPORT" ]] || die "weighted admission resonance graft admission switch report not written: $GRAFT_ADMISSION_SWITCH_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_enable_gate.sh" "$GRAFT_ADMISSION_SWITCH_REPORT" "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" >"$ENABLE_GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission enable gate rejected switch report"
fi

[[ -s "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" ]] || die "weighted admission resonance graft admission enable gate report not written: $GRAFT_ADMISSION_ENABLE_GATE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate schema"
require_grep '"status": "shadow_graft_admission_enable_gate_disabled_dry_run"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_enable_gate"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate target kind"
require_grep '"target_mode": "closed_enable_gate_dry_run"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate target mode"
require_grep '"action": "hold_weighted_resonance_shadow_graft_admission_switch_disabled_dry_run"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate action"
require_grep '"enable_state": "disabled"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate state"
require_grep '"enable_action": "require_operator_key"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate action"
require_grep '"switch_state": "disabled"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch state"
require_grep '"switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch action"
require_grep '"promotion": "pending_live_admission"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission promotion"
require_grep '"weighted_admission_resonance_graft_admission_enable_gate_ready": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission enable gate ready"
require_grep '"weighted_admission_resonance_graft_admission_switch_consumed": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "switch consumed"
require_grep '"weighted_admission_resonance_graft_admission_switch_required": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "switch required"
require_grep '"next_step_blocked_without_resonance_graft_admission_enable_gate": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_enable_gate_id": "weighted-resonance-graft-admission-enable-gate-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_enable_gate_receipt"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "receipt shape"
require_grep '"enable_gate_kind": "shadow_graft_admission_enable_gate"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate kind"
require_grep '"enable_gate_mode": "closed_switch_enable_guard"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate mode"
require_grep '"enable_gate_stage": "pre_live_graft_admission_enable_gate"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-enable-gate-causal-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "causal id"
require_grep '"enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-enable-gate-read-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "read-back hash"
require_grep '"switch_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "switch verification"
require_grep '"switch_hash_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "switch hash verification"
require_grep '"switch_read_back_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "switch read-back verification"
require_grep '"promotion_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "promotion verification"
require_grep '"decision_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "decision verification"
require_grep '"proof_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "proof verification"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "store-reader verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "candidate verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "live-ready flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch schema"
require_grep '"source_status": "shadow_graft_admission_switch_disabled_dry_run"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch status"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_id": "weighted-resonance-graft-admission-switch-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch id"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_ready": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch ready"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_hash": "weighted-resonance-graft-admission-switch-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch hash"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_read_back_hash": "weighted-resonance-graft-admission-switch-read-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch read-back"
require_grep '"source_switch_state": "disabled"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch state"
require_grep '"source_switch_action": "hold_pending_live_admission"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch action"
require_grep '"source_switch_kind": "shadow_graft_admission_switch"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch kind"
require_grep '"source_switch_graft_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch graft guard"
require_grep '"source_switch_write_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch writer guard"
require_grep '"source_switch_admission_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch admission guard"
require_grep '"source_switch_live_admission_enabled": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch live guard"
require_grep '"source_switch_mutates_state": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source switch mutation guard"
require_grep '"source_weighted_admission_resonance_graft_admission_promotion_id": "weighted-resonance-graft-admission-promotion-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source promotion id"
require_grep '"source_promotion": "pending_live_admission"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source promotion verdict"
require_grep '"source_weighted_admission_resonance_graft_admission_decision_id": "weighted-resonance-graft-admission-decision-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source decision id"
require_grep '"source_weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source proof id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_ENABLE_GATE_REPORT" "enable gate pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-enable-gate\] pass:' "$ENABLE_GATE_LOG" "enable gate pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-smoke] pass: resonance_graft_admission_switch_report=$GRAFT_ADMISSION_SWITCH_REPORT resonance_graft_admission_enable_gate_report=$GRAFT_ADMISSION_ENABLE_GATE_REPORT"
