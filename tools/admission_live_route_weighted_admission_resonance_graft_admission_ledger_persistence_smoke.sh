#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_ledger_persistence_smoke.sh - block weighted Resonance graft admission ledger persistence behind blocked ledger implementation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-ledger-persistence.XXXXXX")}"
IMPLEMENTATION_WORKDIR="$WORKDIR/ledger_implementation"
GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_implementation.json"
GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_ledger_persistence.json}"
LEDGER_IMPLEMENTATION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_implementation.log"
LEDGER_PERSISTENCE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_ledger_persistence.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-smoke] FAIL: $*" >&2
    if [[ -f "$LEDGER_IMPLEMENTATION_LOG" ]]; then
        tail -n 500 "$LEDGER_IMPLEMENTATION_LOG" >&2 || true
    fi
    if [[ -f "$LEDGER_PERSISTENCE_LOG" ]]; then
        tail -n 240 "$LEDGER_PERSISTENCE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_WORKDIR="$IMPLEMENTATION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT="$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_implementation_smoke.sh" >"$LEDGER_IMPLEMENTATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger implementation producer failed"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" ]] || die "weighted admission resonance graft admission ledger implementation report not written: $GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_ledger_persistence.sh" "$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT" "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" >"$LEDGER_PERSISTENCE_LOG" 2>&1; then
    die "weighted admission resonance graft admission ledger persistence rejected ledger implementation report"
fi

[[ -s "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" ]] || die "weighted admission resonance graft admission ledger persistence report not written: $GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence schema"
require_grep '"status": "shadow_graft_admission_ledger_persistence_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence status"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_ledger_persistence"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence target kind"
require_grep '"target_mode": "closed_ledger_persistence_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence target mode"
require_grep '"action": "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence action"
require_grep '"writer_action": "reject_blocked_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger write scope"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger ready flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger append flag"
require_grep '"ledger_persistence_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence state"
require_grep '"ledger_persistence_action": "reject_blocked_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence action field"
require_grep '"ledger_persistence_target": "admission_ledger_receipt"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence target"
require_grep '"ledger_persistence_target_kind": "weighted_internal_world_shadow_graft_admission_ledger_implementation"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence target kind field"
require_grep '"ledger_persistence_target_mode": "closed_persistence_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence target mode field"
require_grep '"ledger_persistence_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence receipt shape"
require_grep '"ledger_persistence_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence write scope"
require_grep '"ledger_persistence_append_only": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence append flag"
require_grep '"ledger_persistence_dry_run_only": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence dry-run flag"
require_grep '"ledger_persistence_receipt_persisted": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence receipt persisted flag"
require_grep '"ledger_persistence_ready": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence ready flag"
require_grep '"weighted_admission_resonance_graft_admission_ledger_persistence_ready": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "weighted ledger persistence ready"
require_grep '"weighted_admission_resonance_graft_admission_ledger_implementation_consumed": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger implementation consumed"
require_grep '"weighted_admission_resonance_graft_admission_ledger_implementation_required": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger implementation required"
require_grep '"next_step_blocked_without_resonance_graft_admission_ledger_persistence": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_ledger_persistence_id": "weighted-resonance-graft-admission-ledger-persistence-id-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence id"
require_grep '"causal_id": "weighted-resonance-graft-admission-ledger-persistence-causal-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence causal id"
require_grep '"ledger_persistence_hash": "weighted-resonance-graft-admission-ledger-persistence-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence hash"
require_grep '"ledger_persistence_read_back_hash": "weighted-resonance-graft-admission-ledger-persistence-read-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation schema"
require_grep '"source_status": "shadow_graft_admission_ledger_implementation_blocked_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation status"
require_grep '"source_admission_ledger_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source admission ledger schema"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_id": "weighted-resonance-graft-admission-ledger-implementation-id-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_ready": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation ready"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_causal_id": "weighted-resonance-graft-admission-ledger-implementation-causal-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation causal id"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_hash": "weighted-resonance-graft-admission-ledger-implementation-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation hash"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_implementation_read_back_hash": "weighted-resonance-graft-admission-ledger-implementation-read-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation read-back"
require_grep '"source_ledger_implementation_report_receipt_shape": "weighted_resonance_shadow_graft_admission_ledger_implementation_receipt"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation report receipt shape"
require_grep '"source_ledger_implementation_state": "blocked"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation state"
require_grep '"source_ledger_implementation_action": "reject_blocked_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation action"
require_grep '"source_ledger_implementation_target": "admission_ledger"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation target"
require_grep '"source_ledger_implementation_target_kind": "weighted_internal_world_shadow_graft_admission_ledger"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation target kind"
require_grep '"source_ledger_implementation_target_mode": "closed_append_guard_dry_run"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation target mode"
require_grep '"source_ledger_implementation_entrypoint": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation entrypoint"
require_grep '"source_ledger_implementation_receipt_shape": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation receipt shape"
require_grep '"source_ledger_implementation_write_scope": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation write scope"
require_grep '"source_ledger_implementation_append_only": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation append flag"
require_grep '"source_ledger_implementation_dry_run_only": true' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation dry-run flag"
require_grep '"source_ledger_implementation_receipt_persisted": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation receipt persisted flag"
require_grep '"source_ledger_implementation_ready": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation ready flag"
require_grep '"source_weighted_admission_resonance_graft_admission_ledger_id": "weighted-resonance-graft-admission-ledger-id-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger id"
require_grep '"source_weighted_admission_resonance_graft_admission_writer_contract_id": "weighted-resonance-graft-admission-writer-contract-id-' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source writer contract id"
require_grep '"source_ledger_implementation_reason": "weighted resonance shadow graft admission ledger implementation blocked by blocked admission ledger; implementation append contract remains closed"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "source ledger implementation reason"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "body mutation guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed"' "$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT" "ledger persistence reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence\] pass:' "$LEDGER_PERSISTENCE_LOG" "ledger persistence pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-smoke] pass: resonance_graft_admission_ledger_implementation_report=$GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT resonance_graft_admission_ledger_persistence_report=$GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT"
