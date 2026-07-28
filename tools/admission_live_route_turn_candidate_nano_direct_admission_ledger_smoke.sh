#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_admission_ledger_smoke.sh - real nano direct -> admission ledger dry-run.
#
# Runs the nano-direct chat shadow chain through the writer contract and then
# records the append-only ledger receipt shape without persisting any body state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_ADMISSION_LEDGER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-admission-ledger.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
LIVE_STAGE_LOG="$WORKDIR/live_route_candidate_admission_live_stage_nano_direct.jsonl"
WRITER_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_writer_preflight_nano_direct.jsonl"
WRITER_INVENTORY_LOG="$WORKDIR/live_route_candidate_admission_writer_inventory_nano_direct.jsonl"
WRITER_CONTRACT_LOG="$WORKDIR/live_route_candidate_admission_writer_contract_nano_direct.jsonl"
LEDGER_LOG="$WORKDIR/live_route_candidate_admission_ledger_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 520 "$RUN_LOG" >&2 || true
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
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with admission ledger failed"
fi

[[ -s "$WRITER_CONTRACT_LOG" ]] || die "candidate admission writer contract JSONL log not written"
[[ -s "$LEDGER_LOG" ]] || die "candidate admission ledger JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_writer_contract.v1"' "$WRITER_CONTRACT_LOG" || die "writer contract schema missing"
grep -q '"writer_contract_id":"writer-contract-' "$WRITER_CONTRACT_LOG" || die "writer contract id missing"
grep -q '"passed":true' "$WRITER_CONTRACT_LOG" || die "writer contract did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger.v1"' "$LEDGER_LOG" || die "ledger schema missing"
grep -q '"ledger_state":"receipt_drafted_dry_run"' "$LEDGER_LOG" || die "ledger state missing"
grep -q '"ledger_action":"append_candidate_admission_receipt_dry_run"' "$LEDGER_LOG" || die "ledger action missing"
grep -q '"ledger_contract":"live_admission_ledger.v1"' "$LEDGER_LOG" || die "ledger contract missing"
grep -q '"ledger_mode":"append_only_dry_run"' "$LEDGER_LOG" || die "ledger mode missing"
grep -q '"ledger_entry_kind":"dream_candidate_admission"' "$LEDGER_LOG" || die "ledger entry kind missing"
grep -q '"ledger_entry_status":"shadow_candidate_receipt"' "$LEDGER_LOG" || die "ledger entry status missing"
grep -q '"ledger_receipt_shape":"candidate_contract_provenance"' "$LEDGER_LOG" || die "ledger receipt shape missing"
grep -q '"ledger_append_ready":true' "$LEDGER_LOG" || die "ledger append should be ready"
grep -q '"ledger_receipt_persisted":false' "$LEDGER_LOG" || die "ledger receipt must not persist"
grep -q '"ledger_implementation_ready":false' "$LEDGER_LOG" || die "ledger implementation must be absent"
grep -q '"contracts_ready":false' "$LEDGER_LOG" || die "contracts must not be ready"
grep -q '"admission_writer_contract_id":"writer-contract-' "$LEDGER_LOG" || die "ledger writer contract id missing"
grep -q '"source_writer_contract_passed":true' "$LEDGER_LOG" || die "ledger did not consume a passed writer contract"
grep -q '"live_ready":true' "$LEDGER_LOG" || die "ledger live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$LEDGER_LOG" || die "ledger should not enable live admission"
grep -q '"admission_allowed":false' "$LEDGER_LOG" || die "ledger should not allow admission"
grep -q '"write_allowed":false' "$LEDGER_LOG" || die "ledger must not allow writes"
grep -q '"mutates_state":false' "$LEDGER_LOG" || die "ledger must not mutate state"
grep -q '"admission_ledger_id":"admission-ledger-' "$LEDGER_LOG" || die "ledger id missing"
grep -q '"passed":true' "$LEDGER_LOG" || die "ledger did not pass dry-run"

grep -q 'live-route candidate admission ledger dry-run: class=dream route=direct source=direct writer_contract=writer-contract-' "$RUN_LOG" || die "ledger chat line missing"
grep -q 'ledger=receipt_drafted_dry_run ledger_action=append_candidate_admission_receipt_dry_run ledger_contract=live_admission_ledger.v1 ledger_mode=append_only_dry_run' "$RUN_LOG" || die "ledger contract line missing"
grep -q 'ledger_entry=dream_candidate_admission entry_status=shadow_candidate_receipt receipt_shape=candidate_contract_provenance' "$RUN_LOG" || die "ledger entry line missing"
grep -q 'append_ready=true persisted=false ledger_impl=false contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_ledger_id=admission-ledger-' "$RUN_LOG" || die "ledger verdict line missing"
grep -q 'passed=true reason=admission ledger dry-run receipt drafted; no live write occurred' "$RUN_LOG" || die "ledger reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke] pass: writer_contract=$WRITER_CONTRACT_LOG ledger=$LEDGER_LOG"
