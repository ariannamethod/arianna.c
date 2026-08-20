#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_writer_inventory_smoke.sh - block weighted Resonance graft admission writer inventory behind blocked writer preflight.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-writer-inventory.XXXXXX")}"
WRITER_PREFLIGHT_WORKDIR="$WORKDIR/writer_preflight"
GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_writer_preflight.json"
GRAFT_ADMISSION_WRITER_INVENTORY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_writer_inventory.json}"
WRITER_PREFLIGHT_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_writer_preflight.log"
WRITER_INVENTORY_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_writer_inventory.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-smoke] FAIL: $*" >&2
    if [[ -f "$WRITER_PREFLIGHT_LOG" ]]; then
        tail -n 500 "$WRITER_PREFLIGHT_LOG" >&2 || true
    fi
    if [[ -f "$WRITER_INVENTORY_LOG" ]]; then
        tail -n 240 "$WRITER_INVENTORY_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_WORKDIR="$WRITER_PREFLIGHT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT="$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_writer_preflight_smoke.sh" >"$WRITER_PREFLIGHT_LOG" 2>&1; then
    die "weighted admission resonance graft admission writer preflight producer failed"
fi

[[ -s "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft admission writer preflight report not written: $GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_writer_inventory.sh" "$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT" "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" >"$WRITER_INVENTORY_LOG" 2>&1; then
    die "weighted admission resonance graft admission writer inventory rejected writer preflight report"
fi

[[ -s "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" ]] || die "weighted admission resonance graft admission writer inventory report not written: $GRAFT_ADMISSION_WRITER_INVENTORY_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory schema"
require_grep '"status": "shadow_graft_admission_writer_inventory_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_writer_inventory"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory target kind"
require_grep '"target_mode": "closed_writer_inventory_guard_dry_run"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_writer_preflight_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory action"
require_grep '"writer_action": "reject_blocked_writer_preflight"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_writer_preflight"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "rollback action"
require_grep '"inventory_state": "blocked"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "inventory state"
require_grep '"inventory_action": "reject_blocked_writer_preflight"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "inventory action"
require_grep '"writer_contract": "none"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer contract"
require_grep '"rollback_contract": "none"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "rollback contract"
require_grep '"admission_ledger_contract": "none"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "ledger contract"
require_grep '"writer_contract_present": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer contract present flag"
require_grep '"rollback_contract_present": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "rollback contract present flag"
require_grep '"ledger_contract_present": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "ledger contract present flag"
require_grep '"weighted_admission_resonance_graft_admission_writer_inventory_ready": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "admission writer inventory ready"
require_grep '"weighted_admission_resonance_graft_admission_writer_preflight_consumed": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer preflight consumed"
require_grep '"weighted_admission_resonance_graft_admission_writer_preflight_required": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer preflight required"
require_grep '"next_step_blocked_without_resonance_graft_admission_writer_inventory": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_writer_inventory_id": "weighted-resonance-graft-admission-writer-inventory-id-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory id"
require_grep '"writer_inventory_kind": "shadow_graft_admission_writer_inventory"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory kind"
require_grep '"writer_inventory_mode": "closed_writer_preflight_inventory_guard"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory mode"
require_grep '"writer_inventory_stage": "pre_writer_contract_graft_admission_writer_inventory"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory stage"
require_grep '"writer_inventory_hash": "weighted-resonance-graft-admission-writer-inventory-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory hash"
require_grep '"writer_preflight_verified": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer preflight verification"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight schema"
require_grep '"source_status": "shadow_graft_admission_writer_preflight_blocked_dry_run"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight status"
require_grep '"source_weighted_admission_resonance_graft_admission_writer_preflight_id": "weighted-resonance-graft-admission-writer-preflight-id-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight id"
require_grep '"source_weighted_admission_resonance_graft_admission_writer_preflight_hash": "weighted-resonance-graft-admission-writer-preflight-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight hash"
require_grep '"source_writer_preflight_kind": "shadow_graft_admission_writer_preflight"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight kind"
require_grep '"source_writer_preflight_writer_action": "reject_blocked_live_stage"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight writer action"
require_grep '"source_writer_preflight_rollback_action": "reject_blocked_live_stage"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight rollback action"
require_grep '"source_writer_preflight_write_allowed": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight write guard"
require_grep '"source_writer_preflight_live_admission_enabled": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source writer preflight live guard"
require_grep '"source_weighted_admission_resonance_graft_admission_live_stage_id": "weighted-resonance-graft-admission-live-stage-id-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source live stage id"
require_grep '"source_weighted_admission_resonance_graft_admission_enable_gate_id": "weighted-resonance-graft-admission-enable-gate-id-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source enable gate id"
require_grep '"source_weighted_admission_resonance_graft_admission_switch_id": "weighted-resonance-graft-admission-switch-id-' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "source switch id"
require_grep '"requires_writer": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer requirement"
require_grep '"writer_ready": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer closed flag"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "rollback requirement"
require_grep '"rollback_ready": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "rollback closed flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent"' "$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT" "writer inventory reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory\] pass:' "$WRITER_INVENTORY_LOG" "writer inventory pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-smoke] pass: resonance_graft_admission_writer_preflight_report=$GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT resonance_graft_admission_writer_inventory_report=$GRAFT_ADMISSION_WRITER_INVENTORY_REPORT"
