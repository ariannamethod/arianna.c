#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_writer_implementation_smoke.sh - real nano direct -> writer implementation dry-run.
#
# Runs the nano-direct chat shadow chain through the admission ledger and then
# drafts the append-only writer/ledger/rollback implementation contract without
# enabling body mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_WRITER_IMPLEMENTATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-writer-implementation.XXXXXX")}"
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
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 560 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with writer implementation failed"
fi

[[ -s "$LEDGER_LOG" ]] || die "candidate admission ledger JSONL log not written"
[[ -s "$WRITER_IMPL_LOG" ]] || die "candidate admission writer implementation JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger.v1"' "$LEDGER_LOG" || die "ledger schema missing"
grep -q '"admission_ledger_id":"admission-ledger-' "$LEDGER_LOG" || die "ledger id missing"
grep -q '"passed":true' "$LEDGER_LOG" || die "ledger did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_implementation.v1"' "$WRITER_IMPL_LOG" || die "writer implementation schema missing"
grep -q '"implementation_state":"implementation_contract_drafted_dry_run"' "$WRITER_IMPL_LOG" || die "implementation state missing"
grep -q '"implementation_action":"define_append_only_writer_ledger_rollback"' "$WRITER_IMPL_LOG" || die "implementation action missing"
grep -q '"writer_entrypoint":"append_shadow_candidate_receipt_dry_run"' "$WRITER_IMPL_LOG" || die "writer entrypoint missing"
grep -q '"ledger_entrypoint":"append_admission_ledger_receipt_dry_run"' "$WRITER_IMPL_LOG" || die "ledger entrypoint missing"
grep -q '"rollback_entrypoint":"remove_exact_shadow_candidate_receipt_dry_run"' "$WRITER_IMPL_LOG" || die "rollback entrypoint missing"
grep -q '"write_target":"shadow_receipt_log"' "$WRITER_IMPL_LOG" || die "write target missing"
grep -q '"body_target":"none"' "$WRITER_IMPL_LOG" || die "body target missing"
grep -q '"append_only":true' "$WRITER_IMPL_LOG" || die "append-only flag missing"
grep -q '"rollback_required":true' "$WRITER_IMPL_LOG" || die "rollback required flag missing"
grep -q '"implementation_contract_ready":true' "$WRITER_IMPL_LOG" || die "implementation contract readiness missing"
grep -q '"writer_implementation_ready":false' "$WRITER_IMPL_LOG" || die "writer implementation must remain absent"
grep -q '"rollback_implementation_ready":false' "$WRITER_IMPL_LOG" || die "rollback implementation must remain absent"
grep -q '"ledger_implementation_ready":false' "$WRITER_IMPL_LOG" || die "ledger implementation must remain absent"
grep -q '"contracts_ready":false' "$WRITER_IMPL_LOG" || die "contracts must not be ready"
grep -q '"source_ledger_passed":true' "$WRITER_IMPL_LOG" || die "writer implementation did not consume a passed ledger"
grep -q '"admission_ledger_id":"admission-ledger-' "$WRITER_IMPL_LOG" || die "writer implementation ledger id missing"
grep -q '"live_ready":true' "$WRITER_IMPL_LOG" || die "writer implementation live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$WRITER_IMPL_LOG" || die "writer implementation should not enable live admission"
grep -q '"admission_allowed":false' "$WRITER_IMPL_LOG" || die "writer implementation should not allow admission"
grep -q '"write_allowed":false' "$WRITER_IMPL_LOG" || die "writer implementation must not allow writes"
grep -q '"mutates_state":false' "$WRITER_IMPL_LOG" || die "writer implementation must not mutate state"
grep -q '"writer_implementation_id":"writer-implementation-' "$WRITER_IMPL_LOG" || die "writer implementation id missing"
grep -q '"passed":true' "$WRITER_IMPL_LOG" || die "writer implementation did not pass dry-run"

grep -q 'live-route candidate admission writer implementation dry-run: class=dream route=direct source=direct ledger=admission-ledger-' "$RUN_LOG" || die "writer implementation chat line missing"
grep -q 'implementation=implementation_contract_drafted_dry_run implementation_action=define_append_only_writer_ledger_rollback writer_entrypoint=append_shadow_candidate_receipt_dry_run ledger_entrypoint=append_admission_ledger_receipt_dry_run rollback_entrypoint=remove_exact_shadow_candidate_receipt_dry_run' "$RUN_LOG" || die "writer implementation entrypoint line missing"
grep -q 'write_target=shadow_receipt_log body_target=none append_only=true rollback_required=true implementation_contract=true' "$RUN_LOG" || die "writer implementation target line missing"
grep -q 'writer_impl=false ledger_impl=false rollback_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_implementation_id=writer-implementation-' "$RUN_LOG" || die "writer implementation verdict line missing"
grep -q 'passed=true reason=writer implementation contract drafted; append-only log boundary only' "$RUN_LOG" || die "writer implementation reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke] pass: ledger=$LEDGER_LOG writer_implementation=$WRITER_IMPL_LOG"
