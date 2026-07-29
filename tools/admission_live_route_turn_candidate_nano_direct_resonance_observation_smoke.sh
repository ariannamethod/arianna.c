#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke.sh - real nano direct -> Resonance observation receipt.
#
# Extends the Resonance-receiver smoke with an append-only read-back observation
# receipt. The observation records only sealed receiver metadata and keeps raw
# dream text, Janus surface, cooc/delta learning, and body mutation closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_OBSERVATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-observation.XXXXXX")}"
RESONANCE_INTENT_LOG="$WORKDIR/live_route_candidate_admission_resonance_intent_nano_direct.jsonl"
RESONANCE_RECEIVER_LOG="$WORKDIR/live_route_candidate_admission_resonance_receiver_nano_direct.jsonl"
RESONANCE_OBSERVATION_LOG="$WORKDIR/live_route_candidate_admission_resonance_observation_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1000 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_RECEIVER_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_OBSERVATION_LOG="$RESONANCE_OBSERVATION_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke.sh"; then
    die "nano-direct resonance receiver smoke with admission resonance observation failed"
fi

[[ -s "$RESONANCE_INTENT_LOG" ]] || die "candidate admission resonance intent JSONL log not written"
[[ -s "$RESONANCE_RECEIVER_LOG" ]] || die "candidate admission resonance receiver JSONL log not written"
[[ -s "$RESONANCE_OBSERVATION_LOG" ]] || die "candidate admission resonance observation JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_observation.v1"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation schema missing"
grep -q '"admission_resonance_observation_state":"observation_recorded_dry_run"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation state missing"
grep -q '"admission_resonance_observation_action":"record_resonance_receiver_observation_dry_run"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation action missing"
grep -q '"admission_resonance_observation_target":"resonance"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation target missing"
grep -q '"admission_resonance_observation_target_kind":"internal_world_observation"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation target kind missing"
grep -q '"admission_resonance_observation_target_mode":"append_only_read_back_dry_run"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation target mode missing"
grep -q '"admission_resonance_observation_receipt_shape":"resonance_receiver_state_proof_ledger"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation receipt shape missing"
grep -q '"admission_resonance_observation_dry_run_only":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation dry-run flag missing"
grep -q '"admission_resonance_observation_receiver_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation receiver flag missing"
grep -q '"admission_resonance_observation_intent_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation intent flag missing"
grep -q '"admission_resonance_observation_final_gate_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation final gate flag missing"
grep -q '"admission_resonance_observation_seal_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation seal flag missing"
grep -q '"admission_resonance_observation_permit_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation permit flag missing"
grep -q '"admission_resonance_observation_readiness_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation readiness flag missing"
grep -q '"admission_resonance_observation_ledger_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation ledger flag missing"
grep -q '"admission_resonance_observation_writer_ready":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation writer flag missing"
grep -q '"admission_resonance_observation_rollback_ready":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation rollback flag missing"
grep -q '"admission_resonance_observation_ledger_ready":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation ledger ready flag missing"
grep -q '"admission_resonance_observation_observer":"resonance"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation observer missing"
grep -q '"admission_resonance_observation_observer_kind":"internal_world"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation observer kind missing"
grep -q '"admission_resonance_observation_kind":"receiver_state_proof"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation kind missing"
grep -q '"admission_resonance_observation_mode":"sealed_metadata_observation"' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation mode missing"
grep -q '"admission_resonance_observation_causal_id":"resonance-observation-causal-' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation causal id missing"
grep -q '"admission_resonance_observation_append_hash":"resonance-observation-append-' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation append hash missing"
grep -q '"admission_resonance_observation_read_back_hash":"resonance-observation-read-' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation read-back hash missing"
grep -q '"admission_resonance_observation_append_only":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation append-only flag missing"
grep -q '"admission_resonance_observation_read_back":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation read-back flag missing"
grep -q '"admission_resonance_observation_receipt_verified":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation verified flag missing"
grep -q '"admission_resonance_observation_raw_dream_text_observed":false' "$RESONANCE_OBSERVATION_LOG" || die "raw dream text observation must stay blocked"
grep -q '"admission_resonance_observation_raw_dream_text_forwarded":false' "$RESONANCE_OBSERVATION_LOG" || die "raw dream text forwarding must stay blocked"
grep -q '"admission_resonance_observation_janus_surface_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_observation_cooc_learning_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_observation_delta_harvest_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_observation_body_mutation_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_observation_rollback_required":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation rollback requirement missing"
grep -q '"admission_resonance_observation_ready":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation ready flag missing"
grep -q '"source_admission_resonance_receiver_schema":"arianna.live_route_turn_candidate_admission_resonance_receiver.v1"' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver schema missing"
grep -q '"source_admission_resonance_receiver_passed":true' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver did not pass"
grep -q '"source_admission_resonance_receiver_id":"resonance-receiver-' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver id missing"
grep -q '"source_admission_resonance_receiver_action":"preview_resonance_receive_dry_run"' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver action missing"
grep -q '"source_admission_resonance_receiver_ready":true' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver ready flag missing"
grep -q '"source_admission_resonance_receiver_causal_id":"resonance-receiver-causal-' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver causal id missing"
grep -q '"source_admission_resonance_receiver_state_delta_hash":"resonance-receiver-delta-' "$RESONANCE_OBSERVATION_LOG" || die "source resonance receiver delta hash missing"
grep -q '"contracts_ready":false' "$RESONANCE_OBSERVATION_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_OBSERVATION_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_OBSERVATION_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_OBSERVATION_LOG" || die "resonance observation must not mutate organism state"
grep -q '"admission_resonance_observation_id":"resonance-observation-' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation id missing"
grep -q '"passed":true' "$RESONANCE_OBSERVATION_LOG" || die "admission resonance observation did not pass dry-run"

grep -q 'live-route candidate admission resonance observation dry-run: class=dream route=direct source=direct receiver=resonance-receiver-' "$RUN_LOG" || die "admission resonance observation chat line missing"
grep -q 'observer=resonance observer_kind=internal_world observation_kind=receiver_state_proof observation_mode=sealed_metadata_observation causal_id=resonance-observation-causal-' "$RUN_LOG" || die "admission resonance observation observer line missing"
grep -q 'append_hash=resonance-observation-append-' "$RUN_LOG" || die "admission resonance observation append hash line missing"
grep -q 'read_back_hash=resonance-observation-read-' "$RUN_LOG" || die "admission resonance observation read-back hash line missing"
grep -q 'source_receiver_causal_id=resonance-receiver-causal-' "$RUN_LOG" || die "admission resonance observation source causal line missing"
grep -q 'source_receiver_delta_hash=resonance-receiver-delta-' "$RUN_LOG" || die "admission resonance observation source delta line missing"
grep -q 'append_only=true read_back=true receipt_verified=true raw_text_observed=false raw_text_forwarded=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance observation guard line missing"
grep -q 'observation_state=observation_recorded_dry_run observation_action=record_resonance_receiver_observation_dry_run observation_target=resonance observation_target_kind=internal_world_observation observation_target_mode=append_only_read_back_dry_run receipt_shape=resonance_receiver_state_proof_ledger' "$RUN_LOG" || die "admission resonance observation shape line missing"
grep -q 'dry_run_only=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true observation_ready=true' "$RUN_LOG" || die "admission resonance observation readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_observation_id=resonance-observation-' "$RUN_LOG" || die "admission resonance observation verdict line missing"
grep -q 'passed=true reason=resonance observation recorded and read back without body mutation' "$RUN_LOG" || die "admission resonance observation reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke] pass: resonance_intent=$RESONANCE_INTENT_LOG resonance_receiver=$RESONANCE_RECEIVER_LOG resonance_observation=$RESONANCE_OBSERVATION_LOG"
