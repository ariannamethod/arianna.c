#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_permit_smoke.sh - real nano direct -> closed admission permit dry-run.
#
# Runs the nano-direct chat shadow chain through closed readiness, then accepts
# an operator permit receipt without enabling live admission or body writes.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_PERMIT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-permit.XXXXXX")}"
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
LEDGER_VERIFICATION_LOG="$WORKDIR/live_route_candidate_admission_ledger_verification_nano_direct.jsonl"
READINESS_LOG="$WORKDIR/live_route_candidate_admission_readiness_nano_direct.jsonl"
PERMIT_LOG="$WORKDIR/live_route_candidate_admission_permit_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-permit-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 900 "$RUN_LOG" >&2 || true
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
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG="$LEDGER_VERIFICATION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG="$READINESS_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_LOG="$PERMIT_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PERMIT_KEY=ARIANNA_LIVE_ADMISSION_PERMIT_DRY_RUN_ONLY \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with admission permit failed"
fi

[[ -s "$READINESS_LOG" ]] || die "candidate admission readiness JSONL log not written"
[[ -s "$PERMIT_LOG" ]] || die "candidate admission permit JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_readiness.v1"' "$READINESS_LOG" || die "admission readiness schema missing"
grep -q '"admission_readiness_state":"verified_closed_dry_run"' "$READINESS_LOG" || die "admission readiness state missing"
grep -q '"admission_readiness_ready":true' "$READINESS_LOG" || die "admission readiness flag missing"
grep -q '"admission_readiness_id":"admission-readiness-' "$READINESS_LOG" || die "admission readiness id missing"
grep -q '"contracts_ready":false' "$READINESS_LOG" || die "readiness contracts must remain disabled"
grep -q '"write_allowed":false' "$READINESS_LOG" || die "readiness body write must remain disabled"
grep -q '"admission_allowed":false' "$READINESS_LOG" || die "readiness admission must remain disabled"
grep -q '"live_admission_enabled":false' "$READINESS_LOG" || die "readiness live admission must remain disabled"
grep -q '"mutates_state":false' "$READINESS_LOG" || die "readiness must not mutate organism state"
grep -q '"passed":true' "$READINESS_LOG" || die "admission readiness did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_permit.v1"' "$PERMIT_LOG" || die "admission permit schema missing"
grep -q '"admission_permit_state":"operator_permitted_closed_dry_run"' "$PERMIT_LOG" || die "admission permit state missing"
grep -q '"admission_permit_action":"acknowledge_verified_live_admission_readiness_dry_run"' "$PERMIT_LOG" || die "admission permit action missing"
grep -q '"admission_permit_target":"live_admission"' "$PERMIT_LOG" || die "admission permit target missing"
grep -q '"admission_permit_target_kind":"dream_candidate_admission"' "$PERMIT_LOG" || die "admission permit target kind missing"
grep -q '"admission_permit_target_mode":"permit_closed_dry_run"' "$PERMIT_LOG" || die "admission permit target mode missing"
grep -q '"admission_permit_dry_run_only":true' "$PERMIT_LOG" || die "admission permit dry-run flag missing"
grep -q '"admission_permit_readiness_verified":true' "$PERMIT_LOG" || die "admission permit readiness verification flag missing"
grep -q '"admission_permit_ledger_verified":true' "$PERMIT_LOG" || die "admission permit ledger verification flag missing"
grep -q '"admission_permit_writer_ready":true' "$PERMIT_LOG" || die "admission permit writer flag missing"
grep -q '"admission_permit_rollback_ready":true' "$PERMIT_LOG" || die "admission permit rollback flag missing"
grep -q '"admission_permit_ledger_ready":true' "$PERMIT_LOG" || die "admission permit ledger flag missing"
grep -q '"admission_permit_ready":true' "$PERMIT_LOG" || die "admission permit flag missing"
grep -q '"manual_permit_requested":true' "$PERMIT_LOG" || die "admission permit manual request missing"
grep -q '"permit_key_matched":true' "$PERMIT_LOG" || die "admission permit key match missing"
grep -q '"source_admission_readiness_schema":"arianna.live_route_turn_candidate_admission_readiness.v1"' "$PERMIT_LOG" || die "source admission readiness schema missing"
grep -q '"source_admission_readiness_passed":true' "$PERMIT_LOG" || die "source admission readiness did not pass"
grep -q '"source_admission_readiness_id":"admission-readiness-' "$PERMIT_LOG" || die "source admission readiness id missing"
grep -q '"source_admission_readiness_action":"declare_verified_live_admission_readiness_dry_run"' "$PERMIT_LOG" || die "source admission readiness action missing"
grep -q '"source_admission_readiness_ready":true' "$PERMIT_LOG" || die "source admission readiness flag missing"
grep -q '"source_admission_readiness_ledger_verified":true' "$PERMIT_LOG" || die "source admission readiness ledger flag missing"
grep -q '"source_ledger_verification_id_for_permit":"ledger-verification-' "$PERMIT_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_persistence_id_for_permit":"ledger-persistence-' "$PERMIT_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_permit":"ledger-implementation-' "$PERMIT_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_permit":"admission-ledger-' "$PERMIT_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_permit":"rollback-implementation-' "$PERMIT_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_permit":"writer-receipt-' "$PERMIT_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$PERMIT_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$PERMIT_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$PERMIT_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$PERMIT_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$PERMIT_LOG" || die "admission permit must not mutate organism state"
grep -q '"admission_permit_id":"admission-permit-' "$PERMIT_LOG" || die "admission permit id missing"
grep -q '"passed":true' "$PERMIT_LOG" || die "admission permit did not pass dry-run"

grep -q 'live-route candidate admission permit dry-run: class=dream route=direct source=direct readiness=admission-readiness-' "$RUN_LOG" || die "admission permit chat line missing"
grep -q 'ledger_verification=ledger-verification-' "$RUN_LOG" || die "admission permit ledger verification id missing"
grep -q 'ledger_persistence=ledger-persistence-' "$RUN_LOG" || die "admission permit ledger persistence id missing"
grep -q 'ledger_implementation=ledger-implementation-' "$RUN_LOG" || die "admission permit ledger implementation id missing"
grep -q 'admission_ledger=admission-ledger-' "$RUN_LOG" || die "admission permit admission ledger id missing"
grep -q 'writer_receipt=writer-receipt-' "$RUN_LOG" || die "admission permit writer receipt id missing"
grep -q 'rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "admission permit rollback implementation id missing"
grep -q 'permit=operator_permitted_closed_dry_run permit_action=acknowledge_verified_live_admission_readiness_dry_run' "$RUN_LOG" || die "admission permit action line missing"
grep -q 'permit_target=live_admission permit_target_kind=dream_candidate_admission permit_target_mode=permit_closed_dry_run' "$RUN_LOG" || die "admission permit target line missing"
grep -q 'dry_run_only=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true permit_ready=true manual_requested=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "admission permit readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_permit_id=admission-permit-' "$RUN_LOG" || die "admission permit verdict line missing"
grep -q 'passed=true reason=operator permit accepted for verified readiness; live admission remains disabled' "$RUN_LOG" || die "admission permit reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-permit-smoke] pass: readiness=$READINESS_LOG permit=$PERMIT_LOG"
