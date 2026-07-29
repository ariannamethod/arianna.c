#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_seal_smoke.sh - real nano direct -> sealed admission permit dry-run.
#
# Runs the permit smoke with one additional seal receipt. The seal freezes the
# operator permit provenance, but still does not enable live admission or body writes.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_SEAL_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-seal.XXXXXX")}"
SEAL_LOG="$WORKDIR/live_route_candidate_admission_seal_nano_direct.jsonl"
PERMIT_LOG="$WORKDIR/live_route_candidate_admission_permit_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-seal-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 900 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_PERMIT_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SEAL_LOG="$SEAL_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_permit_smoke.sh"; then
    die "nano-direct permit smoke with admission seal failed"
fi

[[ -s "$PERMIT_LOG" ]] || die "candidate admission permit JSONL log not written"
[[ -s "$SEAL_LOG" ]] || die "candidate admission seal JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_seal.v1"' "$SEAL_LOG" || die "admission seal schema missing"
grep -q '"admission_seal_state":"sealed_closed_dry_run"' "$SEAL_LOG" || die "admission seal state missing"
grep -q '"admission_seal_action":"seal_operator_permit_provenance_dry_run"' "$SEAL_LOG" || die "admission seal action missing"
grep -q '"admission_seal_target":"live_admission"' "$SEAL_LOG" || die "admission seal target missing"
grep -q '"admission_seal_target_kind":"dream_candidate_admission"' "$SEAL_LOG" || die "admission seal target kind missing"
grep -q '"admission_seal_target_mode":"sealed_closed_dry_run"' "$SEAL_LOG" || die "admission seal target mode missing"
grep -q '"admission_seal_receipt_shape":"candidate_contract_provenance"' "$SEAL_LOG" || die "admission seal receipt shape missing"
grep -q '"admission_seal_dry_run_only":true' "$SEAL_LOG" || die "admission seal dry-run flag missing"
grep -q '"admission_seal_permit_verified":true' "$SEAL_LOG" || die "admission seal permit verification flag missing"
grep -q '"admission_seal_readiness_verified":true' "$SEAL_LOG" || die "admission seal readiness verification flag missing"
grep -q '"admission_seal_ledger_verified":true' "$SEAL_LOG" || die "admission seal ledger verification flag missing"
grep -q '"admission_seal_writer_ready":true' "$SEAL_LOG" || die "admission seal writer flag missing"
grep -q '"admission_seal_rollback_ready":true' "$SEAL_LOG" || die "admission seal rollback flag missing"
grep -q '"admission_seal_ledger_ready":true' "$SEAL_LOG" || die "admission seal ledger flag missing"
grep -q '"admission_seal_ready":true' "$SEAL_LOG" || die "admission seal ready flag missing"
grep -q '"source_admission_permit_schema":"arianna.live_route_turn_candidate_admission_permit.v1"' "$SEAL_LOG" || die "source admission permit schema missing"
grep -q '"source_admission_permit_passed":true' "$SEAL_LOG" || die "source admission permit did not pass"
grep -q '"source_admission_permit_id":"admission-permit-' "$SEAL_LOG" || die "source admission permit id missing"
grep -q '"source_admission_permit_action":"acknowledge_verified_live_admission_readiness_dry_run"' "$SEAL_LOG" || die "source admission permit action missing"
grep -q '"source_admission_permit_ready":true' "$SEAL_LOG" || die "source admission permit ready flag missing"
grep -q '"source_admission_permit_key_matched":true' "$SEAL_LOG" || die "source admission permit key match missing"
grep -q '"source_admission_readiness_id_for_seal":"admission-readiness-' "$SEAL_LOG" || die "source admission readiness id missing"
grep -q '"source_ledger_verification_id_for_seal":"ledger-verification-' "$SEAL_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_persistence_id_for_seal":"ledger-persistence-' "$SEAL_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_seal":"ledger-implementation-' "$SEAL_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_seal":"admission-ledger-' "$SEAL_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_seal":"rollback-implementation-' "$SEAL_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_seal":"writer-receipt-' "$SEAL_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$SEAL_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$SEAL_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$SEAL_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$SEAL_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$SEAL_LOG" || die "admission seal must not mutate organism state"
grep -q '"admission_seal_id":"admission-seal-' "$SEAL_LOG" || die "admission seal id missing"
grep -q '"passed":true' "$SEAL_LOG" || die "admission seal did not pass dry-run"

grep -q 'live-route candidate admission seal dry-run: class=dream route=direct source=direct permit=admission-permit-' "$RUN_LOG" || die "admission seal chat line missing"
grep -q 'readiness=admission-readiness-' "$RUN_LOG" || die "admission seal readiness id missing"
grep -q 'ledger_verification=ledger-verification-' "$RUN_LOG" || die "admission seal ledger verification id missing"
grep -q 'ledger_persistence=ledger-persistence-' "$RUN_LOG" || die "admission seal ledger persistence id missing"
grep -q 'ledger_implementation=ledger-implementation-' "$RUN_LOG" || die "admission seal ledger implementation id missing"
grep -q 'admission_ledger=admission-ledger-' "$RUN_LOG" || die "admission seal admission ledger id missing"
grep -q 'writer_receipt=writer-receipt-' "$RUN_LOG" || die "admission seal writer receipt id missing"
grep -q 'rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "admission seal rollback implementation id missing"
grep -q 'seal=sealed_closed_dry_run seal_action=seal_operator_permit_provenance_dry_run' "$RUN_LOG" || die "admission seal action line missing"
grep -q 'seal_target=live_admission seal_target_kind=dream_candidate_admission seal_target_mode=sealed_closed_dry_run receipt_shape=candidate_contract_provenance' "$RUN_LOG" || die "admission seal target line missing"
grep -q 'dry_run_only=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true seal_ready=true permit_ready=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "admission seal readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_seal_id=admission-seal-' "$RUN_LOG" || die "admission seal verdict line missing"
grep -q 'passed=true reason=operator permit sealed as immutable dry-run receipt; live admission remains disabled' "$RUN_LOG" || die "admission seal reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-seal-smoke] pass: permit=$PERMIT_LOG seal=$SEAL_LOG"
