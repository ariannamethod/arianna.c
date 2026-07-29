#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_final_gate_smoke.sh - real nano direct -> final closed admission gate dry-run.
#
# Runs the seal smoke with one additional final gate receipt. The gate accepts
# only sealed provenance and still keeps live admission and body writes closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_FINAL_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-final-gate.XXXXXX")}"
FINAL_GATE_LOG="$WORKDIR/live_route_candidate_admission_final_gate_nano_direct.jsonl"
SEAL_LOG="$WORKDIR/live_route_candidate_admission_seal_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-final-gate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 900 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_SEAL_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_FINAL_GATE_LOG="$FINAL_GATE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_seal_smoke.sh"; then
    die "nano-direct seal smoke with admission final gate failed"
fi

[[ -s "$SEAL_LOG" ]] || die "candidate admission seal JSONL log not written"
[[ -s "$FINAL_GATE_LOG" ]] || die "candidate admission final gate JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_final_gate.v1"' "$FINAL_GATE_LOG" || die "admission final gate schema missing"
grep -q '"admission_final_gate_state":"ready_closed_dry_run"' "$FINAL_GATE_LOG" || die "admission final gate state missing"
grep -q '"admission_final_gate_action":"verify_sealed_admission_provenance_dry_run"' "$FINAL_GATE_LOG" || die "admission final gate action missing"
grep -q '"admission_final_gate_target":"live_admission"' "$FINAL_GATE_LOG" || die "admission final gate target missing"
grep -q '"admission_final_gate_target_kind":"dream_candidate_admission"' "$FINAL_GATE_LOG" || die "admission final gate target kind missing"
grep -q '"admission_final_gate_target_mode":"final_gate_closed_dry_run"' "$FINAL_GATE_LOG" || die "admission final gate target mode missing"
grep -q '"admission_final_gate_receipt_shape":"sealed_candidate_contract_provenance"' "$FINAL_GATE_LOG" || die "admission final gate receipt shape missing"
grep -q '"admission_final_gate_dry_run_only":true' "$FINAL_GATE_LOG" || die "admission final gate dry-run flag missing"
grep -q '"admission_final_gate_seal_verified":true' "$FINAL_GATE_LOG" || die "admission final gate seal flag missing"
grep -q '"admission_final_gate_permit_verified":true' "$FINAL_GATE_LOG" || die "admission final gate permit flag missing"
grep -q '"admission_final_gate_readiness_verified":true' "$FINAL_GATE_LOG" || die "admission final gate readiness flag missing"
grep -q '"admission_final_gate_ledger_verified":true' "$FINAL_GATE_LOG" || die "admission final gate ledger flag missing"
grep -q '"admission_final_gate_writer_ready":true' "$FINAL_GATE_LOG" || die "admission final gate writer flag missing"
grep -q '"admission_final_gate_rollback_ready":true' "$FINAL_GATE_LOG" || die "admission final gate rollback flag missing"
grep -q '"admission_final_gate_ledger_ready":true' "$FINAL_GATE_LOG" || die "admission final gate ledger ready flag missing"
grep -q '"admission_final_gate_ready":true' "$FINAL_GATE_LOG" || die "admission final gate ready flag missing"
grep -q '"source_admission_seal_schema":"arianna.live_route_turn_candidate_admission_seal.v1"' "$FINAL_GATE_LOG" || die "source admission seal schema missing"
grep -q '"source_admission_seal_passed":true' "$FINAL_GATE_LOG" || die "source admission seal did not pass"
grep -q '"source_admission_seal_id":"admission-seal-' "$FINAL_GATE_LOG" || die "source admission seal id missing"
grep -q '"source_admission_seal_action":"seal_operator_permit_provenance_dry_run"' "$FINAL_GATE_LOG" || die "source admission seal action missing"
grep -q '"source_admission_seal_ready":true' "$FINAL_GATE_LOG" || die "source admission seal ready flag missing"
grep -q '"source_admission_permit_id_for_final_gate":"admission-permit-' "$FINAL_GATE_LOG" || die "source admission permit id missing"
grep -q '"source_admission_readiness_id_for_final_gate":"admission-readiness-' "$FINAL_GATE_LOG" || die "source admission readiness id missing"
grep -q '"source_ledger_verification_id_for_final_gate":"ledger-verification-' "$FINAL_GATE_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_persistence_id_for_final_gate":"ledger-persistence-' "$FINAL_GATE_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_final_gate":"ledger-implementation-' "$FINAL_GATE_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_final_gate":"admission-ledger-' "$FINAL_GATE_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_final_gate":"rollback-implementation-' "$FINAL_GATE_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_final_gate":"writer-receipt-' "$FINAL_GATE_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$FINAL_GATE_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$FINAL_GATE_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$FINAL_GATE_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$FINAL_GATE_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$FINAL_GATE_LOG" || die "admission final gate must not mutate organism state"
grep -q '"admission_final_gate_id":"admission-final-gate-' "$FINAL_GATE_LOG" || die "admission final gate id missing"
grep -q '"passed":true' "$FINAL_GATE_LOG" || die "admission final gate did not pass dry-run"

grep -q 'live-route candidate admission final gate dry-run: class=dream route=direct source=direct seal=admission-seal-' "$RUN_LOG" || die "admission final gate chat line missing"
grep -q 'permit=admission-permit-' "$RUN_LOG" || die "admission final gate permit id missing"
grep -q 'readiness=admission-readiness-' "$RUN_LOG" || die "admission final gate readiness id missing"
grep -q 'ledger_verification=ledger-verification-' "$RUN_LOG" || die "admission final gate ledger verification id missing"
grep -q 'ledger_persistence=ledger-persistence-' "$RUN_LOG" || die "admission final gate ledger persistence id missing"
grep -q 'ledger_implementation=ledger-implementation-' "$RUN_LOG" || die "admission final gate ledger implementation id missing"
grep -q 'admission_ledger=admission-ledger-' "$RUN_LOG" || die "admission final gate admission ledger id missing"
grep -q 'writer_receipt=writer-receipt-' "$RUN_LOG" || die "admission final gate writer receipt id missing"
grep -q 'rollback_implementation=rollback-implementation-' "$RUN_LOG" || die "admission final gate rollback implementation id missing"
grep -q 'final_gate=ready_closed_dry_run final_gate_action=verify_sealed_admission_provenance_dry_run' "$RUN_LOG" || die "admission final gate action line missing"
grep -q 'final_gate_target=live_admission final_gate_target_kind=dream_candidate_admission final_gate_target_mode=final_gate_closed_dry_run receipt_shape=sealed_candidate_contract_provenance' "$RUN_LOG" || die "admission final gate target line missing"
grep -q 'dry_run_only=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true final_gate_ready=true seal_ready=true permit_ready=true key_matched=true readiness_ready=true verification_ready=true persistence_ready=true writer_impl=true rollback_impl=true ledger_impl=true' "$RUN_LOG" || die "admission final gate readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_final_gate_id=admission-final-gate-' "$RUN_LOG" || die "admission final gate verdict line missing"
grep -q 'passed=true reason=sealed admission provenance cleared final gate; live admission remains disabled' "$RUN_LOG" || die "admission final gate reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-final-gate-smoke] pass: seal=$SEAL_LOG final_gate=$FINAL_GATE_LOG"
