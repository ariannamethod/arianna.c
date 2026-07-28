#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_rollback_implementation_smoke.sh - real nano direct -> rollback implementation dry-run.
#
# Runs the nano-direct chat shadow chain through the writer receipt and proves
# an exact rollback implementation for that shadow receipt without removing it
# or enabling body mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_ROLLBACK_IMPLEMENTATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-rollback-implementation.XXXXXX")}"
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
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 700 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with rollback implementation failed"
fi

[[ -s "$WRITER_RECEIPT_LOG" ]] || die "candidate admission writer receipt JSONL log not written"
[[ -s "$ROLLBACK_IMPL_LOG" ]] || die "candidate admission rollback implementation JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_receipt.v1"' "$WRITER_RECEIPT_LOG" || die "writer receipt schema missing"
grep -q '"writer_receipt_id":"writer-receipt-' "$WRITER_RECEIPT_LOG" || die "writer receipt id missing"
grep -q '"writer_receipt_persisted":true' "$WRITER_RECEIPT_LOG" || die "writer receipt was not persisted"
grep -q '"passed":true' "$WRITER_RECEIPT_LOG" || die "writer receipt did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_rollback_implementation.v1"' "$ROLLBACK_IMPL_LOG" || die "rollback implementation schema missing"
grep -q '"rollback_implementation_state":"rollback_contract_drafted_dry_run"' "$ROLLBACK_IMPL_LOG" || die "rollback implementation state missing"
grep -q '"rollback_implementation_action":"remove_exact_shadow_candidate_receipt_dry_run"' "$ROLLBACK_IMPL_LOG" || die "rollback implementation action missing"
grep -q '"rollback_entrypoint_resolved":"remove_exact_shadow_candidate_receipt_dry_run"' "$ROLLBACK_IMPL_LOG" || die "rollback entrypoint missing"
grep -q '"rollback_target":"shadow_receipt_log"' "$ROLLBACK_IMPL_LOG" || die "rollback target missing"
grep -q '"rollback_target_kind":"dream_candidate_admission"' "$ROLLBACK_IMPL_LOG" || die "rollback target kind missing"
grep -q '"rollback_target_id":"writer-receipt-' "$ROLLBACK_IMPL_LOG" || die "rollback target id missing"
grep -q '"rollback_mode":"exact_receipt_id_dry_run"' "$ROLLBACK_IMPL_LOG" || die "rollback mode missing"
grep -q '"exact_receipt_match_required":true' "$ROLLBACK_IMPL_LOG" || die "exact receipt match not required"
grep -q '"rollback_dry_run_only":true' "$ROLLBACK_IMPL_LOG" || die "rollback must remain dry-run only"
grep -q '"rollback_receipt_removed":false' "$ROLLBACK_IMPL_LOG" || die "rollback must not remove the receipt in this layer"
grep -q '"source_writer_receipt_schema":"arianna.live_route_turn_candidate_admission_writer_receipt.v1"' "$ROLLBACK_IMPL_LOG" || die "source writer receipt schema missing"
grep -q '"source_writer_receipt_passed":true' "$ROLLBACK_IMPL_LOG" || die "source writer receipt did not pass"
grep -q '"source_writer_receipt_id":"writer-receipt-' "$ROLLBACK_IMPL_LOG" || die "source writer receipt id missing"
grep -q '"source_writer_receipt_persisted":true' "$ROLLBACK_IMPL_LOG" || die "source writer receipt persisted flag missing"
grep -q '"source_writer_receipt_shadow_writable":true' "$ROLLBACK_IMPL_LOG" || die "source writer receipt shadow writable flag missing"
grep -q '"rollback_ready":true' "$ROLLBACK_IMPL_LOG" || die "rollback readiness missing"
grep -q '"rollback_implementation_ready":true' "$ROLLBACK_IMPL_LOG" || die "rollback implementation readiness missing"
grep -q '"writer_implementation_ready":true' "$ROLLBACK_IMPL_LOG" || die "writer implementation readiness missing"
grep -q '"ledger_implementation_ready":false' "$ROLLBACK_IMPL_LOG" || die "ledger implementation must remain absent"
grep -q '"contracts_ready":false' "$ROLLBACK_IMPL_LOG" || die "contracts must not be ready"
grep -q '"write_allowed":false' "$ROLLBACK_IMPL_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$ROLLBACK_IMPL_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$ROLLBACK_IMPL_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$ROLLBACK_IMPL_LOG" || die "rollback implementation must not mutate organism state"
grep -q '"rollback_implementation_id":"rollback-implementation-' "$ROLLBACK_IMPL_LOG" || die "rollback implementation id missing"
grep -q '"passed":true' "$ROLLBACK_IMPL_LOG" || die "rollback implementation did not pass dry-run"

grep -q 'live-route candidate admission rollback implementation dry-run: class=dream route=direct source=direct writer_receipt=writer-receipt-' "$RUN_LOG" || die "rollback implementation chat line missing"
grep -q 'rollback=rollback_contract_drafted_dry_run rollback_action=remove_exact_shadow_candidate_receipt_dry_run rollback_entrypoint=remove_exact_shadow_candidate_receipt_dry_run' "$RUN_LOG" || die "rollback implementation action line missing"
grep -q 'rollback_target=shadow_receipt_log rollback_target_kind=dream_candidate_admission rollback_target_id=writer-receipt-' "$RUN_LOG" || die "rollback target line missing"
grep -q 'rollback_mode=exact_receipt_id_dry_run exact_match=true dry_run_only=true receipt_removed=false' "$RUN_LOG" || die "rollback mode line missing"
grep -q 'exact_match=true dry_run_only=true receipt_removed=false writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=false' "$RUN_LOG" || die "rollback readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false rollback_implementation_id=rollback-implementation-' "$RUN_LOG" || die "rollback verdict line missing"
grep -q 'passed=true reason=rollback implementation drafted for exact writer receipt; body write remains disabled' "$RUN_LOG" || die "rollback reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke] pass: writer_receipt=$WRITER_RECEIPT_LOG rollback_implementation=$ROLLBACK_IMPL_LOG"
