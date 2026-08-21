#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_readiness_smoke.sh - block weighted Resonance graft admission readiness behind blocked ledger verification.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_READINESS_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-readiness.XXXXXX")}"
LEDGER_VERIFICATION_WORKDIR="$WORKDIR/ledger_verification"
GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_verification.json"
GRAFT_ADMISSION_READINESS_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_READINESS_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_readiness.json}"
LEDGER_VERIFICATION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_verification.log"
READINESS_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_readiness.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-readiness-smoke] FAIL: $*" >&2
    if [[ -f "$LEDGER_VERIFICATION_LOG" ]]; then
        tail -n 500 "$LEDGER_VERIFICATION_LOG" >&2 || true
    fi
    if [[ -f "$READINESS_LOG" ]]; then
        tail -n 240 "$READINESS_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_WORKDIR="$LEDGER_VERIFICATION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT="$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_verification_smoke.sh" >"$LEDGER_VERIFICATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger verification producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" ]] || die "weighted admission resonance graft admission ledger verification report not written: $GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_readiness.sh" "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "$GRAFT_ADMISSION_READINESS_REPORT" >"$READINESS_LOG" 2>&1; then
    die "weighted admission resonance graft admission readiness rejected ledger verification report"
fi

[[ -s "$GRAFT_ADMISSION_READINESS_REPORT" ]] || die "weighted admission resonance graft admission readiness report not written: $GRAFT_ADMISSION_READINESS_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness schema"
require_grep '"status": "shadow_graft_admission_readiness_blocked_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_readiness"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness target kind"
require_grep '"target_mode": "closed_readiness_guard_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness action"
require_grep '"writer_action": "reject_blocked_ledger_verification"' "$GRAFT_ADMISSION_READINESS_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_ledger_verification"' "$GRAFT_ADMISSION_READINESS_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_ledger_verification"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger append flag"
require_grep '"admission_readiness_state": "blocked"' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness state"
require_grep '"admission_readiness_action": "reject_blocked_ledger_verification"' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness action"
require_grep '"admission_readiness_target": "live_admission"' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness target"
require_grep '"admission_readiness_target_kind": "weighted_internal_world_shadow_graft_admission_ledger_verification"' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness target kind"
require_grep '"admission_readiness_target_mode": "closed_readiness_guard_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness target mode"
require_grep '"admission_readiness_dry_run_only": true' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness dry-run flag"
require_grep '"admission_readiness_ledger_verified": false' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness ledger verified flag"
require_grep '"admission_readiness_writer_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness writer flag"
require_grep '"admission_readiness_rollback_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness rollback flag"
require_grep '"admission_readiness_ledger_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness ledger flag"
require_grep '"admission_readiness_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "admission readiness ready flag"
require_grep '"weighted_admission_resonance_graft_admission_readiness_ready": true' "$GRAFT_ADMISSION_READINESS_REPORT" "weighted readiness ready"
require_grep '"weighted_admission_resonance_graft_admission_ledger_verification_consumed": true' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger verification consumed"
require_grep '"weighted_admission_resonance_graft_admission_ledger_verification_required": true' "$GRAFT_ADMISSION_READINESS_REPORT" "ledger verification required"
require_grep '"next_step_blocked_without_resonance_graft_admission_readiness": true' "$GRAFT_ADMISSION_READINESS_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_readiness_id": "weighted-resonance-graft-admission-readiness-id-' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness id"
require_grep '"causal_id": "weighted-resonance-graft-admission-readiness-causal-' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness causal id"
require_grep '"admission_readiness_hash": "weighted-resonance-graft-admission-readiness-' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness hash"
require_grep '"admission_readiness_read_back_hash": "weighted-resonance-graft-admission-readiness-read-' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification schema"
require_grep '"source_status": "shadow_graft_admission_ledger_verification_blocked_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification status"
require_grep '"source_ledger_persistence_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger persistence schema"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_verification_id": "weighted-resonance-graft-admission-ledger-verification-id-' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_verification_ready": true' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification ready"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_verification_causal_id": "weighted-resonance-graft-admission-ledger-verification-causal-' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification hash"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_verification_read_back_hash": "weighted-resonance-graft-admission-ledger-verification-read-' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification read-back"
require_grep '"source_ledger_verification_report_receipt_shape": "weighted_resonance_shadow_graft_admission_ledger_verification_receipt"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification report receipt shape"
require_grep '"source_ledger_verification_state": "blocked"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification state"
require_grep '"source_ledger_verification_action": "reject_blocked_ledger_persistence"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification action"
require_grep '"source_ledger_verification_target": "admission_ledger_receipt"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification target"
require_grep '"source_ledger_verification_target_kind": "weighted_internal_world_shadow_graft_admission_ledger_persistence"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification target kind"
require_grep '"source_ledger_verification_target_mode": "closed_read_back_guard_dry_run"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification target mode"
require_grep '"source_ledger_verification_receipt_shape": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification receipt shape"
require_grep '"source_ledger_verification_append_only": false' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification append flag"
require_grep '"source_ledger_verification_dry_run_only": true' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification dry-run flag"
require_grep '"source_ledger_verification_receipt_read_back": false' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification read-back flag"
require_grep '"source_ledger_verification_receipt_verified": false' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification verified flag"
require_grep '"source_ledger_verification_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification ready flag"
require_grep '"source_ledger_verification_reason": "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed"' "$GRAFT_ADMISSION_READINESS_REPORT" "source ledger verification reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_READINESS_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_READINESS_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_READINESS_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_READINESS_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_READINESS_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_READINESS_REPORT" "body mutation guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_READINESS_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission readiness blocked by blocked ledger verification; live admission readiness remains closed"' "$GRAFT_ADMISSION_READINESS_REPORT" "readiness reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-readiness\] pass:' "$READINESS_LOG" "readiness pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-readiness-smoke] pass: resonance_graft_admission_ledger_verification_report=$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT resonance_graft_admission_readiness_report=$GRAFT_ADMISSION_READINESS_REPORT"
