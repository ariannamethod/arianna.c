#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_ledger_implementation_smoke.sh - real nano direct -> ledger implementation dry-run.
#
# Runs the nano-direct chat shadow chain through writer receipt and rollback
# implementation, then proves the append-only ledger implementation without
# enabling contracts, live admission, or body mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_LEDGER_IMPLEMENTATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-ledger-implementation.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
LIVE_STAGE_LOG="$WORKDIR/live_route_candidate_admission_live_stage_nano_direct.jsonl"
WRITER_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_writer_preflight_nano_direct.jsonl"
WRITER_INVENTORY_LOG="$WORKDIR/live_route_candidate_admission_writer_inventory_nano_direct.jsonl"
WRITER_CONTRACT_LOG="$WORKDIR/live_route_candidate_admission_writer_contract_nano_direct.jsonl"
LEDGER_LOG="$WORKDIR/live_route_candidate_admission_ledger_nano_direct.jsonl"
WRITER_IMPL_LOG="$WORKDIR/live_route_candidate_admission_writer_implementation_nano_direct.jsonl"
WRITER_RECEIPT_LOG="$WORKDIR/live_route_candidate_admission_writer_receipt_nano_direct.jsonl"
ROLLBACK_IMPL_LOG="$WORKDIR/live_route_candidate_admission_rollback_implementation_nano_direct.jsonl"
LEDGER_IMPL_LOG="$WORKDIR/live_route_candidate_admission_ledger_implementation_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 800 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_CHAT_SHADOW_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG="$DECISION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG="$PROMOTION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG="$SWITCH_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG="$ENABLE_GATE_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY=ARIANNA_LIVE_ADMISSION_ENABLE_DRY_RUN_ONLY \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG="$LIVE_STAGE_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_PREFLIGHT_LOG="$WRITER_PREFLIGHT_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_INVENTORY_LOG="$WRITER_INVENTORY_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_CONTRACT_LOG="$WRITER_CONTRACT_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_LOG="$LEDGER_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_IMPLEMENTATION_LOG="$WRITER_IMPL_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WRITER_RECEIPT_LOG="$WRITER_RECEIPT_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ROLLBACK_IMPLEMENTATION_LOG="$ROLLBACK_IMPL_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_IMPLEMENTATION_LOG="$LEDGER_IMPL_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with ledger implementation failed"
fi

[[ -s "$ROLLBACK_IMPL_LOG" ]] || die "candidate admission rollback implementation JSONL log not written"
[[ -s "$LEDGER_IMPL_LOG" ]] || die "candidate admission ledger implementation JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_rollback_implementation.v1"' "$ROLLBACK_IMPL_LOG" || die "rollback implementation schema missing"
grep -q '"rollback_implementation_id":"rollback-implementation-' "$ROLLBACK_IMPL_LOG" || die "rollback implementation id missing"
grep -q '"rollback_implementation_ready":true' "$ROLLBACK_IMPL_LOG" || die "rollback implementation readiness missing"
grep -q '"passed":true' "$ROLLBACK_IMPL_LOG" || die "rollback implementation did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger_implementation.v1"' "$LEDGER_IMPL_LOG" || die "ledger implementation schema missing"
grep -q '"ledger_implementation_state":"ledger_contract_drafted_dry_run"' "$LEDGER_IMPL_LOG" || die "ledger implementation state missing"
grep -q '"ledger_implementation_action":"append_admission_ledger_receipt_dry_run"' "$LEDGER_IMPL_LOG" || die "ledger implementation action missing"
grep -q '"ledger_entrypoint_resolved":"append_admission_ledger_receipt_dry_run"' "$LEDGER_IMPL_LOG" || die "ledger entrypoint missing"
grep -q '"ledger_implementation_target":"admission_ledger"' "$LEDGER_IMPL_LOG" || die "ledger implementation target missing"
grep -q '"ledger_implementation_target_kind":"dream_candidate_admission"' "$LEDGER_IMPL_LOG" || die "ledger implementation target kind missing"
grep -q '"ledger_implementation_target_mode":"append_only_dry_run"' "$LEDGER_IMPL_LOG" || die "ledger implementation target mode missing"
grep -q '"ledger_implementation_append_only":true' "$LEDGER_IMPL_LOG" || die "ledger implementation append-only flag missing"
grep -q '"ledger_implementation_dry_run_only":true' "$LEDGER_IMPL_LOG" || die "ledger implementation must remain dry-run only"
grep -q '"ledger_implementation_receipt_persisted":false' "$LEDGER_IMPL_LOG" || die "ledger implementation must not persist a receipt in this layer"
grep -q '"source_rollback_implementation_schema":"arianna.live_route_turn_candidate_admission_rollback_implementation.v1"' "$LEDGER_IMPL_LOG" || die "source rollback implementation schema missing"
grep -q '"source_rollback_implementation_passed":true' "$LEDGER_IMPL_LOG" || die "source rollback implementation did not pass"
grep -q '"source_rollback_implementation_id":"rollback-implementation-' "$LEDGER_IMPL_LOG" || die "source rollback implementation id missing"
grep -q '"source_rollback_implementation_ready":true' "$LEDGER_IMPL_LOG" || die "source rollback implementation readiness missing"
grep -q '"source_writer_receipt_id_for_ledger":"writer-receipt-' "$LEDGER_IMPL_LOG" || die "source writer receipt id missing"
grep -q '"writer_implementation_ready":true' "$LEDGER_IMPL_LOG" || die "writer implementation readiness missing"
grep -q '"rollback_implementation_ready":true' "$LEDGER_IMPL_LOG" || die "rollback implementation readiness missing"
grep -q '"ledger_implementation_ready":true' "$LEDGER_IMPL_LOG" || die "ledger implementation readiness missing"
grep -q '"contracts_ready":false' "$LEDGER_IMPL_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$LEDGER_IMPL_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$LEDGER_IMPL_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$LEDGER_IMPL_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$LEDGER_IMPL_LOG" || die "ledger implementation must not mutate organism state"
grep -q '"ledger_implementation_id":"ledger-implementation-' "$LEDGER_IMPL_LOG" || die "ledger implementation id missing"
grep -q '"passed":true' "$LEDGER_IMPL_LOG" || die "ledger implementation did not pass dry-run"

grep -q 'live-route candidate admission ledger implementation dry-run: class=dream route=direct source=direct rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "ledger implementation chat line missing"
grep -q 'ledger=ledger_contract_drafted_dry_run ledger_action=append_admission_ledger_receipt_dry_run ledger_entrypoint=append_admission_ledger_receipt_dry_run' "$RUN_LOG" || die "ledger implementation action line missing"
grep -q 'ledger_target=admission_ledger ledger_target_kind=dream_candidate_admission ledger_target_mode=append_only_dry_run' "$RUN_LOG" || die "ledger target line missing"
grep -q 'append_only=true dry_run_only=true receipt_persisted=false writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "ledger readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false ledger_implementation_id=ledger-implementation-' "$RUN_LOG" || die "ledger verdict line missing"
grep -q 'passed=true reason=ledger implementation drafted for append-only admission receipts; contracts remain disabled' "$RUN_LOG" || die "ledger reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke] pass: rollback_implementation=$ROLLBACK_IMPL_LOG ledger_implementation=$LEDGER_IMPL_LOG"
