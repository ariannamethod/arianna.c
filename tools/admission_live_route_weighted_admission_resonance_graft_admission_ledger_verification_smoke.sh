#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_ledger_verification_smoke.sh - block weighted Resonance graft admission ledger verification behind blocked ledger persistence.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-ledger-verification.XXXXXX")}"
PERSISTENCE_WORKDIR="$WORKDIR/ledger_persistence"
GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_persistence.json"
GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_verification.json}"
LEDGER_PERSISTENCE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_persistence.log"
LEDGER_VERIFICATION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_verification.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-smoke] FAIL: $*" >&2
    if [[ -f "$LEDGER_PERSISTENCE_LOG" ]]; then
        tail -n 500 "$LEDGER_PERSISTENCE_LOG" >&2 || true
    fi
    if [[ -f "$LEDGER_VERIFICATION_LOG" ]]; then
        tail -n 240 "$LEDGER_VERIFICATION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_WORKDIR="$PERSISTENCE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT="$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_persistence_smoke.sh" >"$LEDGER_PERSISTENCE_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger persistence producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" ]] || die "weighted admission resonance graft admission ledger persistence report not written: $GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_verification.sh" "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" >"$LEDGER_VERIFICATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger verification rejected ledger persistence report"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" ]] || die "weighted admission resonance graft admission ledger verification report not written: $GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification schema"
require_grep '"status": "shadow_graft_admission_ledger_verification_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_ledger_verification"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification target kind"
require_grep '"target_mode": "closed_ledger_verification_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification action"
require_grep '"writer_action": "reject_blocked_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger append flag"
require_grep '"ledger_verification_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification state"
require_grep '"ledger_verification_action": "reject_blocked_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification action field"
require_grep '"ledger_verification_target": "admission_ledger_receipt"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification target"
require_grep '"ledger_verification_target_kind": "weighted_internal_world_shadow_graft_admission_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification target kind field"
require_grep '"ledger_verification_target_mode": "closed_read_back_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification target mode field"
require_grep '"ledger_verification_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification receipt shape"
require_grep '"ledger_verification_append_only": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification append flag"
require_grep '"ledger_verification_dry_run_only": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification dry-run flag"
require_grep '"ledger_verification_receipt_read_back": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification read-back flag"
require_grep '"ledger_verification_receipt_verified": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification verified flag"
require_grep '"ledger_verification_ready": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification ready flag"
require_grep '"weighted_admission_resonance_graft_admission_ledger_verification_ready": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "weighted ledger verification ready"
require_grep '"weighted_admission_resonance_graft_admission_ledger_persistence_consumed": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger persistence consumed"
require_grep '"weighted_admission_resonance_graft_admission_ledger_persistence_required": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger persistence required"
require_grep '"next_step_blocked_without_resonance_graft_admission_ledger_verification": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_ledger_verification_id": "weighted-resonance-graft-admission-ledger-verification-id-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification id"
require_grep '"causal_id": "weighted-resonance-graft-admission-ledger-verification-causal-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification causal id"
require_grep '"ledger_verification_hash": "weighted-resonance-graft-admission-ledger-verification-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification hash"
require_grep '"ledger_verification_read_back_hash": "weighted-resonance-graft-admission-ledger-verification-read-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence schema"
require_grep '"source_status": "shadow_graft_admission_ledger_persistence_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence status"
require_grep '"source_ledger_implementation_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger implementation schema"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_persistence_id": "weighted-resonance-graft-admission-ledger-persistence-id-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_persistence_ready": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence ready"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_persistence_causal_id": "weighted-resonance-graft-admission-ledger-persistence-causal-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence hash"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_persistence_read_back_hash": "weighted-resonance-graft-admission-ledger-persistence-read-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence read-back"
require_grep '"source_ledger_persistence_report_receipt_shape": "weighted_resonance_shadow_graft_admission_ledger_persistence_receipt"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence report receipt shape"
require_grep '"source_ledger_persistence_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence state"
require_grep '"source_ledger_persistence_action": "reject_blocked_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence action"
require_grep '"source_ledger_persistence_target": "admission_ledger_receipt"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence target"
require_grep '"source_ledger_persistence_target_kind": "weighted_internal_world_shadow_graft_admission_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence target kind"
require_grep '"source_ledger_persistence_target_mode": "closed_persistence_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence target mode"
require_grep '"source_ledger_persistence_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence receipt shape"
require_grep '"source_ledger_persistence_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence write scope"
require_grep '"source_ledger_persistence_append_only": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence append flag"
require_grep '"source_ledger_persistence_dry_run_only": true' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence dry-run flag"
require_grep '"source_ledger_persistence_receipt_persisted": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence receipt persisted flag"
require_grep '"source_ledger_persistence_ready": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence ready flag"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_id": "weighted-resonance-graft-admission-ledger-implementation-id-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger implementation id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_id": "weighted-resonance-graft-admission-ledger-id-' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger id"
require_grep '"source_ledger_persistence_reason": "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "source ledger persistence reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "body mutation guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed"' "$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT" "ledger verification reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification\] pass:' "$LEDGER_VERIFICATION_LOG" "ledger verification pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-smoke] pass: resonance_graft_admission_ledger_persistence_report=$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT resonance_graft_admission_ledger_verification_report=$GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT"
