#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_readiness_smoke.sh - real nano direct -> closed admission readiness dry-run.
#
# Runs the nano-direct chat shadow chain through ledger verification, then
# declares closed admission readiness only after the read-back proof is verified.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_READINESS_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-readiness.XXXXXX")}"
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
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-readiness-smoke] FAIL: $*" >&2
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
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LEDGER_VERIFICATION_LOG="$LEDGER_VERIFICATION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_READINESS_LOG="$READINESS_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with admission readiness failed"
fi

[[ -s "$LEDGER_VERIFICATION_LOG" ]] || die "candidate admission ledger verification JSONL log not written"
[[ -s "$READINESS_LOG" ]] || die "candidate admission readiness JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_ledger_verification.v1"' "$LEDGER_VERIFICATION_LOG" || die "ledger verification schema missing"
grep -q '"ledger_verification_id":"ledger-verification-' "$LEDGER_VERIFICATION_LOG" || die "ledger verification id missing"
grep -q '"ledger_verification_receipt_verified":true' "$LEDGER_VERIFICATION_LOG" || die "ledger verification did not verify receipt"
grep -q '"ledger_verification_ready":true' "$LEDGER_VERIFICATION_LOG" || die "ledger verification readiness missing"
grep -q '"passed":true' "$LEDGER_VERIFICATION_LOG" || die "ledger verification did not pass dry-run"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_readiness.v1"' "$READINESS_LOG" || die "admission readiness schema missing"
grep -q '"admission_readiness_state":"verified_closed_dry_run"' "$READINESS_LOG" || die "admission readiness state missing"
grep -q '"admission_readiness_action":"declare_verified_live_admission_readiness_dry_run"' "$READINESS_LOG" || die "admission readiness action missing"
grep -q '"admission_readiness_target":"live_admission"' "$READINESS_LOG" || die "admission readiness target missing"
grep -q '"admission_readiness_target_kind":"dream_candidate_admission"' "$READINESS_LOG" || die "admission readiness target kind missing"
grep -q '"admission_readiness_target_mode":"closed_verified_dry_run"' "$READINESS_LOG" || die "admission readiness target mode missing"
grep -q '"admission_readiness_dry_run_only":true' "$READINESS_LOG" || die "admission readiness dry-run flag missing"
grep -q '"admission_readiness_ledger_verified":true' "$READINESS_LOG" || die "admission readiness ledger verification flag missing"
grep -q '"admission_readiness_writer_ready":true' "$READINESS_LOG" || die "admission readiness writer flag missing"
grep -q '"admission_readiness_rollback_ready":true' "$READINESS_LOG" || die "admission readiness rollback flag missing"
grep -q '"admission_readiness_ledger_ready":true' "$READINESS_LOG" || die "admission readiness ledger flag missing"
grep -q '"admission_readiness_ready":true' "$READINESS_LOG" || die "admission readiness flag missing"
grep -q '"source_ledger_verification_schema":"arianna.live_route_turn_candidate_admission_ledger_verification.v1"' "$READINESS_LOG" || die "source ledger verification schema missing"
grep -q '"source_ledger_verification_passed":true' "$READINESS_LOG" || die "source ledger verification did not pass"
grep -q '"source_ledger_verification_id":"ledger-verification-' "$READINESS_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_verification_action":"verify_persisted_admission_ledger_receipt_dry_run"' "$READINESS_LOG" || die "source ledger verification action missing"
grep -q '"source_ledger_verification_ready":true' "$READINESS_LOG" || die "source ledger verification readiness missing"
grep -q '"source_ledger_verification_receipt_verified":true' "$READINESS_LOG" || die "source ledger verification verified flag missing"
grep -q '"source_ledger_persistence_id_for_readiness":"ledger-persistence-' "$READINESS_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_readiness":"ledger-implementation-' "$READINESS_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_readiness":"admission-ledger-' "$READINESS_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_readiness":"rollback-implementation-' "$READINESS_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_readiness":"writer-receipt-' "$READINESS_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$READINESS_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$READINESS_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$READINESS_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$READINESS_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$READINESS_LOG" || die "admission readiness must not mutate organism state"
grep -q '"admission_readiness_id":"admission-readiness-' "$READINESS_LOG" || die "admission readiness id missing"
grep -q '"passed":true' "$READINESS_LOG" || die "admission readiness did not pass dry-run"

grep -q 'live-route candidate admission readiness dry-run: class=dream route=direct source=direct ledger_verification=ledger-verification-' "$RUN_LOG" || die "admission readiness chat line missing"
grep -q 'ledger_persistence=ledger-persistence-' "$RUN_LOG" || die "admission readiness ledger persistence id missing"
grep -q 'ledger_implementation=ledger-implementation-' "$RUN_LOG" || die "admission readiness ledger implementation id missing"
grep -q 'admission_ledger=admission-ledger-' "$RUN_LOG" || die "admission readiness admission ledger id missing"
grep -q 'writer_receipt=writer-receipt-' "$RUN_LOG" || die "admission readiness writer receipt id missing"
grep -q 'rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "admission readiness rollback implementation id missing"
grep -q 'readiness=verified_closed_dry_run readiness_action=declare_verified_live_admission_readiness_dry_run' "$RUN_LOG" || die "admission readiness action line missing"
grep -q 'readiness_target=live_admission readiness_target_kind=dream_candidate_admission readiness_target_mode=closed_verified_dry_run' "$RUN_LOG" || die "admission readiness target line missing"
grep -q 'dry_run_only=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "admission readiness readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_readiness_id=admission-readiness-' "$RUN_LOG" || die "admission readiness verdict line missing"
grep -q 'passed=true reason=verified ledger and writer boundaries are ready; live admission remains disabled' "$RUN_LOG" || die "admission readiness reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-readiness-smoke] pass: ledger_verification=$LEDGER_VERIFICATION_LOG readiness=$READINESS_LOG"
