#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke.sh - real nano direct -> Resonance-only intent receipt.
#
# Runs the final-gate smoke with one additional dry-run receipt. The receipt
# says Resonance is the first receiver, while Janus, cooc/delta learning, and
# body mutation remain closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_INTENT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-intent.XXXXXX")}"
FINAL_GATE_LOG="$WORKDIR/live_route_candidate_admission_final_gate_nano_direct.jsonl"
RESONANCE_INTENT_LOG="$WORKDIR/live_route_candidate_admission_resonance_intent_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 900 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_FINAL_GATE_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_INTENT_LOG="$RESONANCE_INTENT_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_final_gate_smoke.sh"; then
    die "nano-direct final gate smoke with admission resonance intent failed"
fi

[[ -s "$FINAL_GATE_LOG" ]] || die "candidate admission final gate JSONL log not written"
[[ -s "$RESONANCE_INTENT_LOG" ]] || die "candidate admission resonance intent JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_intent.v1"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent schema missing"
grep -q '"admission_resonance_intent_state":"resonance_intent_drafted_dry_run"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent state missing"
grep -q '"admission_resonance_intent_action":"draft_resonance_direction_intent_dry_run"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent action missing"
grep -q '"admission_resonance_intent_target":"resonance"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent target missing"
grep -q '"admission_resonance_intent_target_kind":"first_live_receiver"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent target kind missing"
grep -q '"admission_resonance_intent_target_mode":"bounded_direction_dry_run"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent target mode missing"
grep -q '"admission_resonance_intent_receipt_shape":"sealed_candidate_contract_provenance"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent receipt shape missing"
grep -q '"admission_resonance_intent_dry_run_only":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent dry-run flag missing"
grep -q '"admission_resonance_intent_final_gate_verified":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent final gate flag missing"
grep -q '"admission_resonance_intent_seal_verified":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent seal flag missing"
grep -q '"admission_resonance_intent_permit_verified":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent permit flag missing"
grep -q '"admission_resonance_intent_readiness_verified":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent readiness flag missing"
grep -q '"admission_resonance_intent_ledger_verified":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent ledger flag missing"
grep -q '"admission_resonance_intent_writer_ready":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent writer flag missing"
grep -q '"admission_resonance_intent_rollback_ready":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent rollback flag missing"
grep -q '"admission_resonance_intent_ledger_ready":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent ledger ready flag missing"
grep -q '"admission_resonance_intent_receiver":"resonance"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent receiver missing"
grep -q '"admission_resonance_intent_receiver_kind":"internal_world"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent receiver kind missing"
grep -q '"admission_resonance_intent_influence_kind":"bounded_direction"' "$RESONANCE_INTENT_LOG" || die "admission resonance intent influence kind missing"
grep -q '"admission_resonance_intent_max_influence":0.05' "$RESONANCE_INTENT_LOG" || die "admission resonance intent influence cap missing"
grep -q '"admission_resonance_intent_ttl_turns":1' "$RESONANCE_INTENT_LOG" || die "admission resonance intent ttl missing"
grep -q '"admission_resonance_intent_causal_id":"resonance-intent-causal-' "$RESONANCE_INTENT_LOG" || die "admission resonance intent causal id missing"
grep -q '"admission_resonance_intent_raw_dream_text_allowed":false' "$RESONANCE_INTENT_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_intent_janus_surface_allowed":false' "$RESONANCE_INTENT_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_intent_cooc_learning_allowed":false' "$RESONANCE_INTENT_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_intent_delta_harvest_allowed":false' "$RESONANCE_INTENT_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_intent_rollback_required":true' "$RESONANCE_INTENT_LOG" || die "rollback requirement missing"
grep -q '"admission_resonance_intent_pre_state_hash_required":true' "$RESONANCE_INTENT_LOG" || die "pre-state hash requirement missing"
grep -q '"admission_resonance_intent_post_state_hash_required":true' "$RESONANCE_INTENT_LOG" || die "post-state hash requirement missing"
grep -q '"admission_resonance_intent_ready":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent ready flag missing"
grep -q '"source_admission_final_gate_schema":"arianna.live_route_turn_candidate_admission_final_gate.v1"' "$RESONANCE_INTENT_LOG" || die "source final gate schema missing"
grep -q '"source_admission_final_gate_passed":true' "$RESONANCE_INTENT_LOG" || die "source final gate did not pass"
grep -q '"source_admission_final_gate_id":"admission-final-gate-' "$RESONANCE_INTENT_LOG" || die "source final gate id missing"
grep -q '"source_admission_final_gate_action":"verify_sealed_admission_provenance_dry_run"' "$RESONANCE_INTENT_LOG" || die "source final gate action missing"
grep -q '"source_admission_final_gate_ready":true' "$RESONANCE_INTENT_LOG" || die "source final gate ready flag missing"
grep -q '"source_admission_seal_id_for_resonance_intent":"admission-seal-' "$RESONANCE_INTENT_LOG" || die "source seal id missing"
grep -q '"source_admission_permit_id_for_resonance_intent":"admission-permit-' "$RESONANCE_INTENT_LOG" || die "source permit id missing"
grep -q '"source_admission_readiness_id_for_resonance_intent":"admission-readiness-' "$RESONANCE_INTENT_LOG" || die "source readiness id missing"
grep -q '"source_ledger_verification_id_for_resonance_intent":"ledger-verification-' "$RESONANCE_INTENT_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_persistence_id_for_resonance_intent":"ledger-persistence-' "$RESONANCE_INTENT_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_resonance_intent":"ledger-implementation-' "$RESONANCE_INTENT_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_resonance_intent":"admission-ledger-' "$RESONANCE_INTENT_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_resonance_intent":"rollback-implementation-' "$RESONANCE_INTENT_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_resonance_intent":"writer-receipt-' "$RESONANCE_INTENT_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$RESONANCE_INTENT_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_INTENT_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_INTENT_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_INTENT_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_INTENT_LOG" || die "resonance intent must not mutate organism state"
grep -q '"admission_resonance_intent_id":"resonance-intent-' "$RESONANCE_INTENT_LOG" || die "admission resonance intent id missing"
grep -q '"passed":true' "$RESONANCE_INTENT_LOG" || die "admission resonance intent did not pass dry-run"

grep -q 'live-route candidate admission resonance intent dry-run: class=dream route=direct source=direct final_gate=admission-final-gate-' "$RUN_LOG" || die "admission resonance intent chat line missing"
grep -q 'receiver=resonance receiver_kind=internal_world influence_kind=bounded_direction max_influence=0.05 ttl_turns=1 causal_id=resonance-intent-causal-' "$RUN_LOG" || die "admission resonance intent receiver line missing"
grep -q 'raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false rollback_required=true pre_hash_required=true post_hash_required=true' "$RUN_LOG" || die "admission resonance intent guard line missing"
grep -q 'intent=resonance_intent_drafted_dry_run intent_action=draft_resonance_direction_intent_dry_run intent_target=resonance intent_target_kind=first_live_receiver intent_target_mode=bounded_direction_dry_run receipt_shape=sealed_candidate_contract_provenance' "$RUN_LOG" || die "admission resonance intent shape line missing"
grep -q 'dry_run_only=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true intent_ready=true' "$RUN_LOG" || die "admission resonance intent readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_intent_id=resonance-intent-' "$RUN_LOG" || die "admission resonance intent verdict line missing"
grep -q 'passed=true reason=resonance intent drafted from final gate; live admission remains disabled' "$RUN_LOG" || die "admission resonance intent reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke] pass: final_gate=$FINAL_GATE_LOG resonance_intent=$RESONANCE_INTENT_LOG"
