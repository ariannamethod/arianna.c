#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_receiver_smoke.sh - real nano direct -> Resonance receiver preview receipt.
#
# Extends the Resonance-intent smoke with the first dry-run receiver receipt.
# Resonance receives only sealed metadata: no raw dream text, no Janus surface,
# no cooc/delta learning, and no body mutation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_RECEIVER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-receiver.XXXXXX")}"
RESONANCE_INTENT_LOG="$WORKDIR/live_route_candidate_admission_resonance_intent_nano_direct.jsonl"
RESONANCE_RECEIVER_LOG="$WORKDIR/live_route_candidate_admission_resonance_receiver_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 900 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_INTENT_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_RECEIVER_LOG="$RESONANCE_RECEIVER_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_intent_smoke.sh"; then
    die "nano-direct resonance intent smoke with admission resonance receiver failed"
fi

[[ -s "$RESONANCE_INTENT_LOG" ]] || die "candidate admission resonance intent JSONL log not written"
[[ -s "$RESONANCE_RECEIVER_LOG" ]] || die "candidate admission resonance receiver JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_receiver.v1"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver schema missing"
grep -q '"admission_resonance_receiver_state":"receiver_previewed_dry_run"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver state missing"
grep -q '"admission_resonance_receiver_action":"preview_resonance_receive_dry_run"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver action missing"
grep -q '"admission_resonance_receiver_target":"resonance"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver target missing"
grep -q '"admission_resonance_receiver_target_kind":"first_live_receiver"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver target kind missing"
grep -q '"admission_resonance_receiver_target_mode":"bounded_direction_preview_dry_run"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver target mode missing"
grep -q '"admission_resonance_receiver_receipt_shape":"resonance_receiver_state_proof"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver receipt shape missing"
grep -q '"admission_resonance_receiver_dry_run_only":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver dry-run flag missing"
grep -q '"admission_resonance_receiver_intent_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver intent flag missing"
grep -q '"admission_resonance_receiver_final_gate_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver final gate flag missing"
grep -q '"admission_resonance_receiver_seal_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver seal flag missing"
grep -q '"admission_resonance_receiver_permit_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver permit flag missing"
grep -q '"admission_resonance_receiver_readiness_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver readiness flag missing"
grep -q '"admission_resonance_receiver_ledger_verified":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver ledger flag missing"
grep -q '"admission_resonance_receiver_writer_ready":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver writer flag missing"
grep -q '"admission_resonance_receiver_rollback_ready":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver rollback flag missing"
grep -q '"admission_resonance_receiver_ledger_ready":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver ledger ready flag missing"
grep -q '"admission_resonance_receiver_receiver":"resonance"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver missing"
grep -q '"admission_resonance_receiver_receiver_kind":"internal_world"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver kind missing"
grep -q '"admission_resonance_receiver_influence_kind":"bounded_direction"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver influence kind missing"
grep -q '"admission_resonance_receiver_max_influence":0.05' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver influence cap missing"
grep -q '"admission_resonance_receiver_ttl_turns":1' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver ttl missing"
grep -q '"admission_resonance_receiver_causal_id":"resonance-receiver-causal-' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver causal id missing"
grep -q '"admission_resonance_receiver_pre_state_hash":"resonance-receiver-pre-' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver pre-state hash missing"
grep -q '"admission_resonance_receiver_post_state_hash":"resonance-receiver-post-' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver post-state hash missing"
grep -q '"admission_resonance_receiver_state_delta_hash":"resonance-receiver-delta-' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver delta hash missing"
grep -q '"admission_resonance_receiver_state_hash_mode":"sealed_metadata_preview"' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver hash mode missing"
grep -q '"admission_resonance_receiver_raw_dream_text_observed":false' "$RESONANCE_RECEIVER_LOG" || die "raw dream text observation must stay blocked"
grep -q '"admission_resonance_receiver_raw_dream_text_forwarded":false' "$RESONANCE_RECEIVER_LOG" || die "raw dream text forwarding must stay blocked"
grep -q '"admission_resonance_receiver_janus_surface_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_receiver_cooc_learning_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_receiver_delta_harvest_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_receiver_body_mutation_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_receiver_rollback_required":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver rollback requirement missing"
grep -q '"admission_resonance_receiver_ready":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver ready flag missing"
grep -q '"source_admission_resonance_intent_schema":"arianna.live_route_turn_candidate_admission_resonance_intent.v1"' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent schema missing"
grep -q '"source_admission_resonance_intent_passed":true' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent did not pass"
grep -q '"source_admission_resonance_intent_id":"resonance-intent-' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent id missing"
grep -q '"source_admission_resonance_intent_action":"draft_resonance_direction_intent_dry_run"' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent action missing"
grep -q '"source_admission_resonance_intent_ready":true' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent ready flag missing"
grep -q '"source_admission_resonance_intent_causal_id":"resonance-intent-causal-' "$RESONANCE_RECEIVER_LOG" || die "source resonance intent causal id missing"
grep -q '"source_admission_final_gate_id_for_resonance_receiver":"admission-final-gate-' "$RESONANCE_RECEIVER_LOG" || die "source final gate id missing"
grep -q '"source_admission_seal_id_for_resonance_receiver":"admission-seal-' "$RESONANCE_RECEIVER_LOG" || die "source seal id missing"
grep -q '"source_admission_permit_id_for_resonance_receiver":"admission-permit-' "$RESONANCE_RECEIVER_LOG" || die "source permit id missing"
grep -q '"source_admission_readiness_id_for_resonance_receiver":"admission-readiness-' "$RESONANCE_RECEIVER_LOG" || die "source readiness id missing"
grep -q '"source_ledger_verification_id_for_resonance_receiver":"ledger-verification-' "$RESONANCE_RECEIVER_LOG" || die "source ledger verification id missing"
grep -q '"source_ledger_persistence_id_for_resonance_receiver":"ledger-persistence-' "$RESONANCE_RECEIVER_LOG" || die "source ledger persistence id missing"
grep -q '"source_ledger_implementation_id_for_resonance_receiver":"ledger-implementation-' "$RESONANCE_RECEIVER_LOG" || die "source ledger implementation id missing"
grep -q '"source_admission_ledger_id_for_resonance_receiver":"admission-ledger-' "$RESONANCE_RECEIVER_LOG" || die "source admission ledger id missing"
grep -q '"source_rollback_implementation_id_for_resonance_receiver":"rollback-implementation-' "$RESONANCE_RECEIVER_LOG" || die "source rollback implementation id missing"
grep -q '"source_writer_receipt_id_for_resonance_receiver":"writer-receipt-' "$RESONANCE_RECEIVER_LOG" || die "source writer receipt id missing"
grep -q '"contracts_ready":false' "$RESONANCE_RECEIVER_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_RECEIVER_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_RECEIVER_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_RECEIVER_LOG" || die "resonance receiver must not mutate organism state"
grep -q '"admission_resonance_receiver_id":"resonance-receiver-' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver id missing"
grep -q '"passed":true' "$RESONANCE_RECEIVER_LOG" || die "admission resonance receiver did not pass dry-run"

grep -q 'live-route candidate admission resonance receiver dry-run: class=dream route=direct source=direct intent=resonance-intent-' "$RUN_LOG" || die "admission resonance receiver chat line missing"
grep -q 'receiver=resonance receiver_kind=internal_world influence_kind=bounded_direction max_influence=0.05 ttl_turns=1 causal_id=resonance-receiver-causal-' "$RUN_LOG" || die "admission resonance receiver field line missing"
grep -q 'source_causal_id=resonance-intent-causal-' "$RUN_LOG" || die "admission resonance receiver source causal line missing"
grep -q 'pre_state_hash=resonance-receiver-pre-' "$RUN_LOG" || die "admission resonance receiver pre-state line missing"
grep -q 'post_state_hash=resonance-receiver-post-' "$RUN_LOG" || die "admission resonance receiver post-state line missing"
grep -q 'delta_hash=resonance-receiver-delta-' "$RUN_LOG" || die "admission resonance receiver delta line missing"
grep -q 'state_hash_mode=sealed_metadata_preview raw_text_observed=false raw_text_forwarded=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance receiver guard line missing"
grep -q 'receiver_state=receiver_previewed_dry_run receiver_action=preview_resonance_receive_dry_run receiver_target=resonance receiver_target_kind=first_live_receiver receiver_target_mode=bounded_direction_preview_dry_run receipt_shape=resonance_receiver_state_proof' "$RUN_LOG" || die "admission resonance receiver shape line missing"
grep -q 'dry_run_only=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true receiver_ready=true' "$RUN_LOG" || die "admission resonance receiver readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_receiver_id=resonance-receiver-' "$RUN_LOG" || die "admission resonance receiver verdict line missing"
grep -q 'passed=true reason=resonance receiver previewed sealed intent without body mutation' "$RUN_LOG" || die "admission resonance receiver reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke] pass: resonance_intent=$RESONANCE_INTENT_LOG resonance_receiver=$RESONANCE_RECEIVER_LOG"
