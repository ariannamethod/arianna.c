#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_smoke.sh - block weighted Resonance graft admission final gate behind sealed provenance.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate.XXXXXX")}"
SEAL_WORKDIR="$WORKDIR/seal"
GRAFT_ADMISSION_SEAL_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_seal.json"
GRAFT_ADMISSION_FINAL_GATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate.json}"
SEAL_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_seal.log"
FINAL_GATE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-smoke] FAIL: $*" >&2
    if [[ -f "$SEAL_LOG" ]]; then
        tail -n 500 "$SEAL_LOG" >&2 || true
    fi
    if [[ -f "$FINAL_GATE_LOG" ]]; then
        tail -n 260 "$FINAL_GATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SEAL_WORKDIR="$SEAL_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SEAL_REPORT="$GRAFT_ADMISSION_SEAL_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_seal_smoke.sh" >"$SEAL_LOG" 2>&1; then
    die "weighted admission resonance graft admission seal producer failed"
fi

[[ -s "$GRAFT_ADMISSION_SEAL_REPORT" ]] || die "weighted admission resonance graft admission seal report not written: $GRAFT_ADMISSION_SEAL_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate.sh" "$GRAFT_ADMISSION_SEAL_REPORT" "$GRAFT_ADMISSION_FINAL_GATE_REPORT" >"$FINAL_GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate rejected seal report"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_REPORT" ]] || die "weighted admission resonance graft admission final gate report not written: $GRAFT_ADMISSION_FINAL_GATE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate schema"
require_grep '"status": "shadow_graft_admission_final_gate_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate target kind"
require_grep '"target_mode": "closed_final_gate_guard_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final-gate action"
require_grep '"writer_action": "reject_blocked_admission_seal"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_seal"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_seal"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "ledger append flag"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "receipt shape"
require_grep '"admission_final_gate_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate state"
require_grep '"admission_final_gate_action": "reject_blocked_admission_seal"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate action"
require_grep '"admission_final_gate_target": "live_admission_final_gate"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate target"
require_grep '"admission_final_gate_target_kind": "weighted_internal_world_shadow_graft_admission_seal"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate target kind"
require_grep '"admission_final_gate_target_mode": "closed_final_gate_guard_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate target mode"
require_grep '"admission_final_gate_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate dry-run flag"
require_grep '"admission_final_gate_seal_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate seal flag"
require_grep '"admission_final_gate_authority_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate authority flag"
require_grep '"admission_final_gate_permit_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate permit flag"
require_grep '"admission_final_gate_ledger_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate ledger flag"
require_grep '"admission_final_gate_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "admission final gate ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "weighted final gate ready"
require_grep '"weighted_admission_resonance_graft_admission_seal_consumed": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "seal consumed"
require_grep '"weighted_admission_resonance_graft_admission_seal_required": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "seal required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_id": "weighted-resonance-graft-admission-final-gate-id-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final gate id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-causal-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final gate causal id"
require_grep '"admission_final_gate_hash": "weighted-resonance-graft-admission-final-gate-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final gate hash"
require_grep '"admission_final_gate_read_back_hash": "weighted-resonance-graft-admission-final-gate-read-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final gate read-back"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal schema"
require_grep '"source_status": "shadow_graft_admission_seal_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal status"
require_grep '"source_admission_authority_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source authority schema"
require_grep '"source_weighted_admission_resonance_graft_admission_seal_id": "weighted-resonance-graft-admission-seal-id-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal id"
require_grep '"source_weighted_admission_resonance_graft_admission_seal_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal ready"
require_grep '"source_weighted_admission_resonance_graft_admission_seal_causal_id": "weighted-resonance-graft-admission-seal-causal-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_seal_hash": "weighted-resonance-graft-admission-seal-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal hash"
require_grep '"source_weighted_admission_resonance_graft_admission_seal_read_back_hash": "weighted-resonance-graft-admission-seal-read-' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal read-back"
require_grep '"source_admission_seal_report_receipt_shape": "weighted_resonance_shadow_graft_admission_seal_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal receipt shape"
require_grep '"source_admission_seal_state": "sealed"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal state"
require_grep '"source_admission_seal_action": "seal_blocked_admission_authority"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal action"
require_grep '"source_admission_seal_target": "live_admission_authority"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal target"
require_grep '"source_admission_seal_target_kind": "weighted_internal_world_shadow_graft_admission_authority"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal target kind"
require_grep '"source_admission_seal_target_mode": "closed_seal_guard_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal target mode"
require_grep '"source_admission_seal_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal dry-run flag"
require_grep '"source_admission_seal_authority_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal authority flag"
require_grep '"source_admission_seal_permit_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal permit flag"
require_grep '"source_admission_seal_ledger_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal ledger flag"
require_grep '"source_admission_seal_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal ready flag"
require_grep '"source_admission_seal_immutable_receipt": true' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal immutable flag"
require_grep '"source_admission_seal_reason": "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "source seal reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "base authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate blocked by blocked seal; final admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_REPORT" "final gate reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate\] pass:' "$FINAL_GATE_LOG" "final-gate pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-smoke] pass: resonance_graft_admission_seal_report=$GRAFT_ADMISSION_SEAL_REPORT resonance_graft_admission_final_gate_report=$GRAFT_ADMISSION_FINAL_GATE_REPORT"
