#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_writer_receipt_smoke.sh - real nano direct -> writer receipt dry-run.
#
# Runs the nano-direct chat shadow chain through the writer implementation
# contract and appends an exact shadow writer receipt without enabling body
# mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_WRITER_RECEIPT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-writer-receipt.XXXXXX")}"
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
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 640 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with writer receipt failed"
fi

[[ -s "$WRITER_IMPL_LOG" ]] || die "candidate admission writer implementation JSONL log not written"
[[ -s "$WRITER_RECEIPT_LOG" ]] || die "candidate admission writer receipt JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_implementation.v1"' "$WRITER_IMPL_LOG" || die "writer implementation schema missing"
grep -q '"writer_implementation_id":"writer-implementation-' "$WRITER_IMPL_LOG" || die "writer implementation id missing"
grep -q '"passed":true' "$WRITER_IMPL_LOG" || die "writer implementation did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_receipt.v1"' "$WRITER_RECEIPT_LOG" || die "writer receipt schema missing"
grep -q '"writer_receipt_state":"shadow_receipt_appended_dry_run"' "$WRITER_RECEIPT_LOG" || die "writer receipt state missing"
grep -q '"writer_receipt_action":"append_shadow_candidate_receipt_dry_run"' "$WRITER_RECEIPT_LOG" || die "writer receipt action missing"
grep -q '"writer_receipt_kind":"dream_candidate_admission"' "$WRITER_RECEIPT_LOG" || die "writer receipt kind missing"
grep -q '"writer_receipt_target":"shadow_receipt_log"' "$WRITER_RECEIPT_LOG" || die "writer receipt target missing"
grep -q '"writer_receipt_mode":"append_only_dry_run"' "$WRITER_RECEIPT_LOG" || die "writer receipt mode missing"
grep -q '"writer_receipt_shape":"candidate_contract_provenance"' "$WRITER_RECEIPT_LOG" || die "writer receipt shape missing"
grep -q '"writer_receipt_persisted":true' "$WRITER_RECEIPT_LOG" || die "writer receipt was not persisted to shadow log"
grep -q '"shadow_write_allowed":true' "$WRITER_RECEIPT_LOG" || die "shadow write was not allowed"
grep -q '"writer_ready":true' "$WRITER_RECEIPT_LOG" || die "writer readiness missing"
grep -q '"writer_implementation_ready":true' "$WRITER_RECEIPT_LOG" || die "writer implementation readiness missing"
grep -q '"rollback_implementation_ready":false' "$WRITER_RECEIPT_LOG" || die "rollback implementation must remain absent"
grep -q '"ledger_implementation_ready":false' "$WRITER_RECEIPT_LOG" || die "ledger implementation must remain absent"
grep -q '"contracts_ready":false' "$WRITER_RECEIPT_LOG" || die "contracts must not be ready"
grep -q '"write_allowed":false' "$WRITER_RECEIPT_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$WRITER_RECEIPT_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$WRITER_RECEIPT_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$WRITER_RECEIPT_LOG" || die "writer receipt must not mutate organism state"
grep -q '"source_writer_implementation_passed":true' "$WRITER_RECEIPT_LOG" || die "writer receipt did not consume a passed implementation"
grep -q '"source_writer_implementation_id":"writer-implementation-' "$WRITER_RECEIPT_LOG" || die "source writer implementation id missing"
grep -q '"writer_receipt_id":"writer-receipt-' "$WRITER_RECEIPT_LOG" || die "writer receipt id missing"
grep -q '"passed":true' "$WRITER_RECEIPT_LOG" || die "writer receipt did not pass dry-run"

grep -q 'live-route candidate admission writer receipt dry-run: class=dream route=direct source=direct writer_implementation=writer-implementation-' "$RUN_LOG" || die "writer receipt chat line missing"
grep -q 'writer_receipt=shadow_receipt_appended_dry_run receipt_action=append_shadow_candidate_receipt_dry_run receipt_kind=dream_candidate_admission receipt_target=shadow_receipt_log receipt_mode=append_only_dry_run receipt_shape=candidate_contract_provenance' "$RUN_LOG" || die "writer receipt target line missing"
grep -q 'receipt_persisted=true shadow_write_allowed=true body_target=none append_only=true rollback_required=true writer_ready=true writer_impl=true ledger_impl=false rollback_impl=false' "$RUN_LOG" || die "writer receipt readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_receipt_id=writer-receipt-' "$RUN_LOG" || die "writer receipt verdict line missing"
grep -q 'passed=true reason=shadow writer receipt appended as dry-run; body write remains disabled' "$RUN_LOG" || die "writer receipt reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke] pass: writer_implementation=$WRITER_IMPL_LOG writer_receipt=$WRITER_RECEIPT_LOG"
