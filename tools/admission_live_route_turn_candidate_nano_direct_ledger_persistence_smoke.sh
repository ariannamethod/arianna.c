#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_ledger_persistence_smoke.sh - real nano direct -> ledger persistence dry-run.
#
# Runs the nano-direct chat shadow chain through ledger implementation, then
# persists the append-only admission ledger receipt as a dry-run proof while the
# live body remains closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_LEDGER_PERSISTENCE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-ledger-persistence.XXXXXX")}"
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
LEDGER_PERSISTENCE_LOG="$WORKDIR/live_route_candidate_admission_ledger_persistence_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke] FAIL: $*" >&2
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
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_PERSISTENCE_LOG="$LEDGER_PERSISTENCE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with ledger persistence failed"
fi

[[ -s "$LEDGER_IMPL_LOG" ]] || die "candidate admission ledger implementation JSONL log not written"
[[ -s "$LEDGER_PERSISTENCE_LOG" ]] || die "candidate admission ledger persistence JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger_implementation.v1"' "$LEDGER_IMPL_LOG" || die "ledger implementation schema missing"
grep -q '"ledger_implementation_id":"ledger-implementation-' "$LEDGER_IMPL_LOG" || die "ledger implementation id missing"
grep -q '"ledger_implementation_ready":true' "$LEDGER_IMPL_LOG" || die "ledger implementation readiness missing"
grep -q '"ledger_implementation_receipt_persisted":false' "$LEDGER_IMPL_LOG" || die "ledger implementation must remain non-persistent in its own layer"
grep -q '"passed":true' "$LEDGER_IMPL_LOG" || die "ledger implementation did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger_persistence.v1"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence schema missing"
grep -q '"ledger_persistence_state":"ledger_receipt_persisted_dry_run"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence state missing"
grep -q '"ledger_persistence_action":"append_admission_ledger_receipt_dry_run"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence action missing"
grep -q '"ledger_persistence_target":"admission_ledger"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence target missing"
grep -q '"ledger_persistence_target_kind":"dream_candidate_admission"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence target kind missing"
grep -q '"ledger_persistence_target_mode":"append_only_dry_run"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence target mode missing"
grep -q '"ledger_persistence_receipt_shape":"candidate_contract_provenance"' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence receipt shape missing"
grep -q '"ledger_persistence_append_only":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence append-only flag missing"
grep -q '"ledger_persistence_dry_run_only":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence dry-run flag missing"
grep -q '"ledger_persistence_receipt_persisted":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence receipt did not persist"
grep -q '"ledger_persistence_ready":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence readiness missing"
grep -q '"source_ledger_implementation_schema":"arianna.live_route_turn_candidate_admission_ledger_implementation.v1"' "$LEDGER_PERSISTENCE_LOG" || die "source ledger implementation schema missing"
grep -q '"source_ledger_implementation_passed":true' "$LEDGER_PERSISTENCE_LOG" || die "source ledger implementation did not pass"
grep -q '"source_ledger_implementation_id":"ledger-implementation-' "$LEDGER_PERSISTENCE_LOG" || die "source ledger implementation id missing"
grep -q '"source_ledger_implementation_action":"append_admission_ledger_receipt_dry_run"' "$LEDGER_PERSISTENCE_LOG" || die "source ledger implementation action missing"
grep -q '"source_ledger_implementation_ready":true' "$LEDGER_PERSISTENCE_LOG" || die "source ledger implementation readiness missing"
grep -q '"source_admission_ledger_id_for_persistence":"admission-ledger-' "$LEDGER_PERSISTENCE_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_ledger":"rollback-implementation-' "$LEDGER_PERSISTENCE_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_ledger_persistence":"writer-receipt-' "$LEDGER_PERSISTENCE_LOG" || die "source writer receipt id missing"
grep -q '"writer_implementation_ready":true' "$LEDGER_PERSISTENCE_LOG" || die "writer implementation readiness missing"
grep -q '"rollback_implementation_ready":true' "$LEDGER_PERSISTENCE_LOG" || die "rollback implementation readiness missing"
grep -q '"ledger_implementation_ready":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger implementation readiness missing from persistence"
grep -q '"contracts_ready":false' "$LEDGER_PERSISTENCE_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$LEDGER_PERSISTENCE_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$LEDGER_PERSISTENCE_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$LEDGER_PERSISTENCE_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence must not mutate organism state"
grep -q '"ledger_persistence_id":"ledger-persistence-' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence id missing"
grep -q '"passed":true' "$LEDGER_PERSISTENCE_LOG" || die "ledger persistence did not pass dry-run"

grep -q 'live-route candidate admission ledger persistence dry-run: class=dream route=direct source=direct ledger_implementation=ledger-implementation-' "$RUN_LOG" || die "ledger persistence chat line missing"
grep -q 'admission_ledger=admission-ledger-' "$RUN_LOG" || die "ledger persistence admission ledger id missing"
grep -q 'writer_receipt=writer-receipt-' "$RUN_LOG" || die "ledger persistence writer receipt id missing"
grep -q 'rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "ledger persistence rollback implementation id missing"
grep -q 'persistence=ledger_receipt_persisted_dry_run persistence_action=append_admission_ledger_receipt_dry_run' "$RUN_LOG" || die "ledger persistence action line missing"
grep -q 'persistence_target=admission_ledger persistence_target_kind=dream_candidate_admission persistence_target_mode=append_only_dry_run receipt_shape=candidate_contract_provenance' "$RUN_LOG" || die "ledger persistence target line missing"
grep -q 'append_only=true dry_run_only=true receipt_persisted=true persistence_ready=true writer_ready=true rollback_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "ledger persistence readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false ledger_persistence_id=ledger-persistence-' "$RUN_LOG" || die "ledger persistence verdict line missing"
grep -q 'passed=true reason=ledger receipt persisted to append-only dry-run log; live admission remains disabled' "$RUN_LOG" || die "ledger persistence reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke] pass: ledger_implementation=$LEDGER_IMPL_LOG ledger_persistence=$LEDGER_PERSISTENCE_LOG"
