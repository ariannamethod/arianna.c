#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_writer_contract_smoke.sh - real nano direct -> writer contract dry-run.
#
# Runs the nano-direct chat shadow chain through writer inventory, then records
# the non-mutating writer/rollback/ledger contract shape. The implementation is
# still absent and live admission remains closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_WRITER_CONTRACT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-writer-contract.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
LIVE_STAGE_LOG="$WORKDIR/live_route_candidate_admission_live_stage_nano_direct.jsonl"
WRITER_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_writer_preflight_nano_direct.jsonl"
WRITER_INVENTORY_LOG="$WORKDIR/live_route_candidate_admission_writer_inventory_nano_direct.jsonl"
WRITER_CONTRACT_LOG="$WORKDIR/live_route_candidate_admission_writer_contract_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-writer-contract-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 480 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with writer contract failed"
fi

[[ -s "$WRITER_INVENTORY_LOG" ]] || die "candidate admission writer inventory JSONL log not written"
[[ -s "$WRITER_CONTRACT_LOG" ]] || die "candidate admission writer contract JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_inventory.v1"' "$WRITER_INVENTORY_LOG" || die "writer inventory schema missing"
grep -q '"inventory_state":"contracts_absent"' "$WRITER_INVENTORY_LOG" || die "writer inventory should record absent contracts"
grep -q '"contracts_ready":false' "$WRITER_INVENTORY_LOG" || die "inventory contracts must not be ready"
grep -q '"writer_inventory_id":"writer-inventory-' "$WRITER_INVENTORY_LOG" || die "writer inventory id missing"
grep -q '"passed":true' "$WRITER_INVENTORY_LOG" || die "writer inventory did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_contract.v1"' "$WRITER_CONTRACT_LOG" || die "writer contract schema missing"
grep -q '"contract_state":"shape_drafted_dry_run"' "$WRITER_CONTRACT_LOG" || die "writer contract should only draft shape"
grep -q '"contract_action":"define_writer_rollback_ledger_contract"' "$WRITER_CONTRACT_LOG" || die "writer contract action missing"
grep -q '"writer_contract":"live_admission_writer.v1"' "$WRITER_CONTRACT_LOG" || die "writer contract missing"
grep -q '"rollback_contract":"live_admission_rollback.v1"' "$WRITER_CONTRACT_LOG" || die "rollback contract missing"
grep -q '"admission_ledger_contract":"live_admission_ledger.v1"' "$WRITER_CONTRACT_LOG" || die "ledger contract missing"
grep -q '"writer_contract_shape":"append_shadow_candidate_receipt"' "$WRITER_CONTRACT_LOG" || die "writer contract shape missing"
grep -q '"rollback_contract_shape":"remove_exact_writer_receipt"' "$WRITER_CONTRACT_LOG" || die "rollback contract shape missing"
grep -q '"ledger_contract_shape":"append_only_receipt_log"' "$WRITER_CONTRACT_LOG" || die "ledger contract shape missing"
grep -q '"write_scope":"dream_candidate_admission"' "$WRITER_CONTRACT_LOG" || die "write scope missing"
grep -q '"rollback_scope":"single_writer_receipt"' "$WRITER_CONTRACT_LOG" || die "rollback scope missing"
grep -q '"ledger_mode":"append_only_dry_run"' "$WRITER_CONTRACT_LOG" || die "ledger mode missing"
grep -q '"contract_shape_ready":true' "$WRITER_CONTRACT_LOG" || die "contract shape should be ready"
grep -q '"source_writer_contract_present":false' "$WRITER_CONTRACT_LOG" || die "source writer contract should be absent"
grep -q '"source_rollback_contract_present":false' "$WRITER_CONTRACT_LOG" || die "source rollback contract should be absent"
grep -q '"source_ledger_contract_present":false' "$WRITER_CONTRACT_LOG" || die "source ledger contract should be absent"
grep -q '"writer_implementation_ready":false' "$WRITER_CONTRACT_LOG" || die "writer implementation must be absent"
grep -q '"rollback_implementation_ready":false' "$WRITER_CONTRACT_LOG" || die "rollback implementation must be absent"
grep -q '"ledger_implementation_ready":false' "$WRITER_CONTRACT_LOG" || die "ledger implementation must be absent"
grep -q '"contracts_ready":false' "$WRITER_CONTRACT_LOG" || die "contracts must not be ready"
grep -q '"admission_writer_inventory_id":"writer-inventory-' "$WRITER_CONTRACT_LOG" || die "writer contract inventory id missing"
grep -q '"source_writer_inventory_passed":true' "$WRITER_CONTRACT_LOG" || die "writer contract did not consume a passed inventory"
grep -q '"live_ready":true' "$WRITER_CONTRACT_LOG" || die "writer contract live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$WRITER_CONTRACT_LOG" || die "writer contract should not enable live admission"
grep -q '"admission_allowed":false' "$WRITER_CONTRACT_LOG" || die "writer contract should not allow admission"
grep -q '"write_allowed":false' "$WRITER_CONTRACT_LOG" || die "writer contract must not allow writes"
grep -q '"mutates_state":false' "$WRITER_CONTRACT_LOG" || die "writer contract must not mutate state"
grep -q '"writer_contract_id":"writer-contract-' "$WRITER_CONTRACT_LOG" || die "writer contract id missing"
grep -q '"passed":true' "$WRITER_CONTRACT_LOG" || die "writer contract did not pass dry-run"

grep -q 'live-route candidate admission writer contract dry-run: class=dream route=direct source=direct writer_inventory=writer-inventory-' "$RUN_LOG" || die "writer contract chat line missing"
grep -q 'contract=shape_drafted_dry_run contract_action=define_writer_rollback_ledger_contract writer_contract=live_admission_writer.v1 rollback_contract=live_admission_rollback.v1 ledger_contract=live_admission_ledger.v1' "$RUN_LOG" || die "writer contract contract line missing"
grep -q 'writer_shape=append_shadow_candidate_receipt rollback_shape=remove_exact_writer_receipt ledger_shape=append_only_receipt_log' "$RUN_LOG" || die "writer contract shape line missing"
grep -q 'shape_ready=true writer_impl=false rollback_impl=false ledger_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_contract_id=writer-contract-' "$RUN_LOG" || die "writer contract verdict line missing"
grep -q 'passed=true reason=writer contract shape drafted; implementation and ledger remain absent' "$RUN_LOG" || die "writer contract reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-writer-contract-smoke] pass: inventory=$WRITER_INVENTORY_LOG writer_contract=$WRITER_CONTRACT_LOG"
