#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_writer_inventory_smoke.sh - real nano direct -> writer inventory dry-run.
#
# Runs the nano-direct chat shadow chain through writer-preflight and then
# records the missing writer/rollback/ledger contracts. The inventory is still
# closed: no live admission, write permission, or organism-state mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_WRITER_INVENTORY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-writer-inventory.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
LIVE_STAGE_LOG="$WORKDIR/live_route_candidate_admission_live_stage_nano_direct.jsonl"
WRITER_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_writer_preflight_nano_direct.jsonl"
WRITER_INVENTORY_LOG="$WORKDIR/live_route_candidate_admission_writer_inventory_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 420 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with writer inventory failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"
[[ -s "$PROMOTION_LOG" ]] || die "candidate admission promotion JSONL log not written"
[[ -s "$SWITCH_LOG" ]] || die "candidate admission switch JSONL log not written"
[[ -s "$ENABLE_GATE_LOG" ]] || die "candidate admission enable gate JSONL log not written"
[[ -s "$LIVE_STAGE_LOG" ]] || die "candidate admission live stage JSONL log not written"
[[ -s "$WRITER_PREFLIGHT_LOG" ]] || die "candidate admission writer preflight JSONL log not written"
[[ -s "$WRITER_INVENTORY_LOG" ]] || die "candidate admission writer inventory JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_preflight.v1"' "$WRITER_PREFLIGHT_LOG" || die "writer preflight schema missing"
grep -q '"writer_state":"absent"' "$WRITER_PREFLIGHT_LOG" || die "writer should be absent"
grep -q '"rollback_state":"absent"' "$WRITER_PREFLIGHT_LOG" || die "rollback should be absent"
grep -q '"write_allowed":false' "$WRITER_PREFLIGHT_LOG" || die "writer preflight must not allow writes"
grep -q '"mutates_state":false' "$WRITER_PREFLIGHT_LOG" || die "writer preflight must not mutate state"
grep -q '"passed":true' "$WRITER_PREFLIGHT_LOG" || die "writer preflight did not pass dry-run"
grep -q '"writer_preflight_id":"writer-' "$WRITER_PREFLIGHT_LOG" || die "writer preflight id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_inventory.v1"' "$WRITER_INVENTORY_LOG" || die "writer inventory schema missing"
grep -q '"inventory_state":"contracts_absent"' "$WRITER_INVENTORY_LOG" || die "writer inventory should record absent contracts"
grep -q '"inventory_action":"name_required_contracts"' "$WRITER_INVENTORY_LOG" || die "writer inventory action missing"
grep -q '"writer_contract":"live_admission_writer.v1"' "$WRITER_INVENTORY_LOG" || die "writer contract missing"
grep -q '"rollback_contract":"live_admission_rollback.v1"' "$WRITER_INVENTORY_LOG" || die "rollback contract missing"
grep -q '"admission_ledger_contract":"live_admission_ledger.v1"' "$WRITER_INVENTORY_LOG" || die "ledger contract missing"
grep -q '"writer_contract_present":false' "$WRITER_INVENTORY_LOG" || die "writer contract must remain absent"
grep -q '"rollback_contract_present":false' "$WRITER_INVENTORY_LOG" || die "rollback contract must remain absent"
grep -q '"ledger_contract_present":false' "$WRITER_INVENTORY_LOG" || die "ledger contract must remain absent"
grep -q '"contracts_ready":false' "$WRITER_INVENTORY_LOG" || die "contracts must not be ready"
grep -q '"admission_writer_preflight_id":"writer-' "$WRITER_INVENTORY_LOG" || die "writer inventory preflight id missing"
grep -q '"source_writer_preflight_passed":true' "$WRITER_INVENTORY_LOG" || die "writer inventory did not consume a passed preflight"
grep -q '"live_ready":true' "$WRITER_INVENTORY_LOG" || die "writer inventory live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$WRITER_INVENTORY_LOG" || die "writer inventory should not enable live admission"
grep -q '"admission_allowed":false' "$WRITER_INVENTORY_LOG" || die "writer inventory should not allow admission"
grep -q '"manual_enable_requested":true' "$WRITER_INVENTORY_LOG" || die "writer inventory should preserve manual enable"
grep -q '"enable_key_matched":true' "$WRITER_INVENTORY_LOG" || die "writer inventory should preserve key match"
grep -q '"requires_writer":true' "$WRITER_INVENTORY_LOG" || die "writer inventory should require writer"
grep -q '"writer_ready":false' "$WRITER_INVENTORY_LOG" || die "writer inventory must keep writer absent"
grep -q '"requires_rollback":true' "$WRITER_INVENTORY_LOG" || die "writer inventory should require rollback"
grep -q '"rollback_ready":false' "$WRITER_INVENTORY_LOG" || die "writer inventory must keep rollback absent"
grep -q '"write_allowed":false' "$WRITER_INVENTORY_LOG" || die "writer inventory must not allow writes"
grep -q '"mutates_state":false' "$WRITER_INVENTORY_LOG" || die "writer inventory must not mutate state"
grep -q '"writer_inventory_id":"writer-inventory-' "$WRITER_INVENTORY_LOG" || die "writer inventory id missing"
grep -q '"passed":true' "$WRITER_INVENTORY_LOG" || die "writer inventory did not pass dry-run"

grep -q 'live-route candidate admission writer inventory dry-run: class=dream route=direct source=direct writer_preflight=writer-' "$RUN_LOG" || die "writer inventory chat line missing"
grep -q 'inventory=contracts_absent inventory_action=name_required_contracts writer_contract=live_admission_writer.v1 rollback_contract=live_admission_rollback.v1 ledger_contract=live_admission_ledger.v1' "$RUN_LOG" || die "writer inventory contracts line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false writer_inventory_id=writer-inventory-' "$RUN_LOG" || die "writer inventory verdict line missing"
grep -q 'passed=true reason=writer inventory recorded required contracts; live admission remains blocked' "$RUN_LOG" || die "writer inventory reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke] pass: decision=$DECISION_LOG promotion=$PROMOTION_LOG switch=$SWITCH_LOG enable_gate=$ENABLE_GATE_LOG live_stage=$LIVE_STAGE_LOG writer_preflight=$WRITER_PREFLIGHT_LOG writer_inventory=$WRITER_INVENTORY_LOG"
