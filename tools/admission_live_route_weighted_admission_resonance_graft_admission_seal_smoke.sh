#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_seal_smoke.sh - seal weighted Resonance graft admission authority as closed provenance.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SEAL_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-seal.XXXXXX")}"
AUTHORITY_WORKDIR="$WORKDIR/authority"
GRAFT_ADMISSION_AUTHORITY_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_authority.json"
GRAFT_ADMISSION_SEAL_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_SEAL_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_seal.json}"
AUTHORITY_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_authority.log"
SEAL_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_seal.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-seal-smoke] FAIL: $*" >&2
    if [[ -f "$AUTHORITY_LOG" ]]; then
        tail -n 500 "$AUTHORITY_LOG" >&2 || true
    fi
    if [[ -f "$SEAL_LOG" ]]; then
        tail -n 240 "$SEAL_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_AUTHORITY_WORKDIR="$AUTHORITY_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT="$GRAFT_ADMISSION_AUTHORITY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_authority_smoke.sh" >"$AUTHORITY_LOG" 2>&1; then
    die "weighted admission resonance graft admission authority producer failed"
fi

[[ -s "$GRAFT_ADMISSION_AUTHORITY_REPORT" ]] || die "weighted admission resonance graft admission authority report not written: $GRAFT_ADMISSION_AUTHORITY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_seal.sh" "$GRAFT_ADMISSION_AUTHORITY_REPORT" "$GRAFT_ADMISSION_SEAL_REPORT" >"$SEAL_LOG" 2>&1; then
    die "weighted admission resonance graft admission seal rejected authority report"
fi

[[ -s "$GRAFT_ADMISSION_SEAL_REPORT" ]] || die "weighted admission resonance graft admission seal report not written: $GRAFT_ADMISSION_SEAL_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal schema"
require_grep '"status": "shadow_graft_admission_seal_blocked_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_seal"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal target kind"
require_grep '"target_mode": "closed_seal_guard_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal target mode"
require_grep '"action": "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal action"
require_grep '"writer_action": "reject_blocked_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_SEAL_REPORT" "ledger append flag"
require_grep '"admission_seal_state": "sealed"' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal state"
require_grep '"admission_seal_action": "seal_blocked_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal action"
require_grep '"admission_seal_target": "live_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal target"
require_grep '"admission_seal_target_kind": "weighted_internal_world_shadow_graft_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal target kind"
require_grep '"admission_seal_target_mode": "closed_seal_guard_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal target mode"
require_grep '"admission_seal_dry_run_only": true' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal dry-run flag"
require_grep '"admission_seal_authority_verified": false' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal authority flag"
require_grep '"admission_seal_permit_verified": false' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal permit flag"
require_grep '"admission_seal_ledger_verified": false' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal ledger flag"
require_grep '"admission_seal_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal ready flag"
require_grep '"admission_seal_immutable_receipt": true' "$GRAFT_ADMISSION_SEAL_REPORT" "admission seal immutable flag"
require_grep '"weighted_admission_resonance_graft_admission_seal_ready": true' "$GRAFT_ADMISSION_SEAL_REPORT" "weighted seal ready"
require_grep '"weighted_admission_resonance_graft_admission_authority_consumed": true' "$GRAFT_ADMISSION_SEAL_REPORT" "authority consumed"
require_grep '"weighted_admission_resonance_graft_admission_authority_required": true' "$GRAFT_ADMISSION_SEAL_REPORT" "authority required"
require_grep '"next_step_blocked_without_resonance_graft_admission_seal": true' "$GRAFT_ADMISSION_SEAL_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_seal_id": "weighted-resonance-graft-admission-seal-id-' "$GRAFT_ADMISSION_SEAL_REPORT" "seal id"
require_grep '"causal_id": "weighted-resonance-graft-admission-seal-causal-' "$GRAFT_ADMISSION_SEAL_REPORT" "seal causal id"
require_grep '"admission_seal_hash": "weighted-resonance-graft-admission-seal-' "$GRAFT_ADMISSION_SEAL_REPORT" "seal hash"
require_grep '"admission_seal_read_back_hash": "weighted-resonance-graft-admission-seal-read-' "$GRAFT_ADMISSION_SEAL_REPORT" "seal read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority schema"
require_grep '"source_status": "shadow_graft_admission_authority_blocked_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority status"
require_grep '"source_admission_permit_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1"' "$GRAFT_ADMISSION_SEAL_REPORT" "source permit schema"
require_grep '"source_weighted_admission_resonance_graft_admission_authority_id": "weighted-resonance-graft-admission-authority-id-' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority id"
require_grep '"source_weighted_admission_resonance_graft_admission_authority_ready": true' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority ready"
require_grep '"source_weighted_admission_resonance_graft_admission_authority_causal_id": "weighted-resonance-graft-admission-authority-causal-' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_authority_hash": "weighted-resonance-graft-admission-authority-' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority hash"
require_grep '"source_weighted_admission_resonance_graft_admission_authority_read_back_hash": "weighted-resonance-graft-admission-authority-read-' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority read-back"
require_grep '"source_admission_authority_report_receipt_shape": "weighted_resonance_shadow_graft_admission_authority_receipt"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority receipt shape"
require_grep '"source_admission_authority_state": "blocked"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority state"
require_grep '"source_admission_authority_action": "reject_blocked_admission_permit"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority action"
require_grep '"source_admission_authority_target": "live_admission_authority"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority target"
require_grep '"source_admission_authority_target_kind": "weighted_internal_world_shadow_graft_admission_permit"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority target kind"
require_grep '"source_admission_authority_target_mode": "closed_authority_guard_dry_run"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority target mode"
require_grep '"source_admission_authority_dry_run_only": true' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority dry-run flag"
require_grep '"source_admission_authority_permit_verified": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority permit flag"
require_grep '"source_admission_authority_ledger_verified": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority ledger flag"
require_grep '"source_admission_authority_writer_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority writer flag"
require_grep '"source_admission_authority_rollback_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority rollback flag"
require_grep '"source_admission_authority_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority ready flag"
require_grep '"source_admission_authority_granted": false' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority granted flag"
require_grep '"source_admission_authority_reason": "weighted resonance shadow graft admission authority blocked by blocked permit; live authority remains closed"' "$GRAFT_ADMISSION_SEAL_REPORT" "source authority reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_SEAL_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_SEAL_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_SEAL_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_SEAL_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_SEAL_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_SEAL_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_SEAL_REPORT" "base authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_SEAL_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed"' "$GRAFT_ADMISSION_SEAL_REPORT" "seal reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-seal\] pass:' "$SEAL_LOG" "seal pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-seal-smoke] pass: resonance_graft_admission_authority_report=$GRAFT_ADMISSION_AUTHORITY_REPORT resonance_graft_admission_seal_report=$GRAFT_ADMISSION_SEAL_REPORT"
