#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_admission_proof_smoke.sh - real nano direct -> Resonance shadow graft admission proof receipt.
#
# Replays the closed Resonance shadow-graft candidate store reader receipt into a
# proof-only admission receipt. This proves the reader can become a future graft
# input while keeping contracts, writes, live admission, Janus surface, cooc/delta
# learning, raw dream text, and body mutation closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_ADMISSION_PROOF_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-admission-proof.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RESONANCE_GRAFT_GATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_gate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_STORE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_store_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_store_reader_nano_direct.jsonl"
RESONANCE_GRAFT_ADMISSION_PROOF_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_admission_proof_nano_direct.jsonl"
BOUNDARY_REPORT="$WORKDIR/live_route_boundary_report.json"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"
BOUNDARY_REPORT_STAGES=(
    rollback_implementation
    ledger_implementation
    ledger_persistence
    ledger_verification
    admission_readiness
    admission_permit
    admission_seal
    final_gate
    resonance_intent
    resonance_receiver
    resonance_observation
    resonance_graft_boundary
    resonance_graft_preflight
    resonance_graft_gate
    resonance_graft_candidate
    resonance_graft_candidate_store
    resonance_graft_candidate_store_reader
    resonance_graft_admission_proof
)

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1800 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_STORE_READER_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_LOG="$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" \
    AM_LIVE_ROUTE_BOUNDARY_REPORT="$BOUNDARY_REPORT" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke.sh"; then
    die "nano-direct resonance graft candidate store reader smoke with admission proof failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"
[[ -s "$RESONANCE_GRAFT_GATE_LOG" ]] || die "candidate admission resonance graft gate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_LOG" ]] || die "candidate admission resonance graft candidate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" ]] || die "candidate admission resonance graft candidate store JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" ]] || die "candidate admission resonance graft candidate store reader JSONL log not written"
[[ -s "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" ]] || die "candidate admission resonance graft admission proof JSONL log not written"

bash "$ROOT/tools/admission_live_route_boundary_report_assert.sh" \
    "$BOUNDARY_REPORT" \
    "${#BOUNDARY_REPORT_STAGES[@]}" \
    "${BOUNDARY_REPORT_STAGES[@]}"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_admission_proof.v1"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof schema missing"
grep -q '"timing":"live_admission_resonance_graft_admission_proof"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof timing missing"
grep -q '"admission_resonance_graft_admission_proof_state":"shadow_graft_admission_proved_dry_run"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof state missing"
grep -q '"admission_resonance_graft_admission_proof_action":"prove_resonance_shadow_graft_admission_dry_run"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof action missing"
grep -q '"admission_resonance_graft_admission_proof_target":"resonance"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof target missing"
grep -q '"admission_resonance_graft_admission_proof_target_kind":"internal_world_shadow_graft_admission_proof"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof target kind missing"
grep -q '"admission_resonance_graft_admission_proof_target_mode":"verified_replay_closed_dry_run"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof target mode missing"
grep -q '"admission_resonance_graft_admission_proof_receipt_shape":"resonance_shadow_graft_admission_proof_receipt"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof receipt shape missing"
grep -q '"admission_resonance_graft_admission_proof_dry_run_only":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof dry-run flag missing"
grep -q '"admission_resonance_graft_admission_proof_reader_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof reader flag missing"
grep -q '"admission_resonance_graft_admission_proof_store_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof store flag missing"
grep -q '"admission_resonance_graft_admission_proof_candidate_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof candidate flag missing"
grep -q '"admission_resonance_graft_admission_proof_ledger_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof ledger flag missing"
grep -q '"admission_resonance_graft_admission_proof_replay_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof replay flag missing"
grep -q '"admission_resonance_graft_admission_proof_read_back_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof read-back flag missing"
grep -q '"admission_resonance_graft_admission_proof_hash_verified":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof hash flag missing"
grep -q '"admission_resonance_graft_admission_proof_kind":"shadow_graft_admission_proof"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof kind missing"
grep -q '"admission_resonance_graft_admission_proof_mode":"verified_replay_closed"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof mode missing"
grep -q '"admission_resonance_graft_admission_proof_stage":"pre_live_graft_admission_proof"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof stage missing"
grep -q '"admission_resonance_graft_admission_proof_causal_id":"resonance-graft-admission-proof-causal-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof causal id missing"
grep -q '"admission_resonance_graft_admission_proof_hash":"resonance-graft-admission-proof-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof hash missing"
grep -q '"admission_resonance_graft_admission_proof_replay_hash":"resonance-graft-admission-proof-replay-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof replay hash missing"
grep -q '"admission_resonance_graft_admission_proof_read_back_hash":"resonance-graft-admission-proof-read-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof read-back hash missing"
grep -q '"admission_resonance_graft_admission_proof_admission_required":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof admission-required flag missing"
grep -q '"admission_resonance_graft_admission_proof_shadow_only":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof shadow flag missing"
grep -q '"admission_resonance_graft_admission_proof_graft_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_janus_surface_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_cooc_learning_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_delta_harvest_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_body_mutation_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_admission_proof_rollback_required":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "rollback-required flag missing"
grep -q '"admission_resonance_graft_admission_proof_ready":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate_store_reader.v1"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader schema missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_passed":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader did not pass"
grep -q '"source_admission_resonance_graft_candidate_store_reader_id":"resonance-graft-candidate-store-reader-id-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader id missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_action":"read_resonance_shadow_graft_candidate_store_dry_run"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader action missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_ready":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_causal_id":"resonance-graft-candidate-store-reader-causal-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader causal id missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_hash":"resonance-graft-candidate-store-reader-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader hash missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_replay_hash":"resonance-graft-candidate-store-reader-replay-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader replay hash missing"
grep -q '"source_admission_resonance_graft_candidate_store_reader_read_back_hash":"resonance-graft-candidate-store-reader-read-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store reader read-back hash missing"
grep -q '"source_admission_resonance_graft_candidate_store_id_for_admission_proof":"resonance-graft-candidate-store-id-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate store id missing"
grep -q '"source_admission_resonance_graft_candidate_id_for_admission_proof":"resonance-graft-candidate-id-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft candidate id missing"
grep -q '"source_admission_resonance_graft_gate_id_for_admission_proof":"resonance-graft-gate-id-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance graft gate id missing"
grep -q '"source_admission_resonance_observation_id_for_admission_proof":"resonance-observation-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source resonance observation id missing"
grep -q '"source_admission_final_gate_id_for_admission_proof":"admission-final-gate-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source admission final gate id missing"
grep -q '"source_ledger_verification_id_for_admission_proof":"ledger-verification-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "source ledger verification id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "resonance graft admission proof must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_admission_proof_id":"resonance-graft-admission-proof-id-' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_ADMISSION_PROOF_LOG" || die "admission resonance graft admission proof did not pass dry-run"

grep -q 'live-route candidate admission resonance graft admission proof dry-run: class=dream route=direct source=direct reader=resonance-graft-candidate-store-reader-id-' "$RUN_LOG" || die "admission resonance graft admission proof chat line missing"
grep -q 'proof_kind=shadow_graft_admission_proof proof_mode=verified_replay_closed proof_stage=pre_live_graft_admission_proof causal_id=resonance-graft-admission-proof-causal-' "$RUN_LOG" || die "admission resonance graft admission proof kind line missing"
grep -q 'proof_hash=resonance-graft-admission-proof-' "$RUN_LOG" || die "admission resonance graft admission proof hash line missing"
grep -q 'replay_hash=resonance-graft-admission-proof-replay-' "$RUN_LOG" || die "admission resonance graft admission proof replay line missing"
grep -q 'read_back_hash=resonance-graft-admission-proof-read-' "$RUN_LOG" || die "admission resonance graft admission proof read-back line missing"
grep -q 'source_reader_causal_id=resonance-graft-candidate-store-reader-causal-' "$RUN_LOG" || die "admission resonance graft admission proof source causal line missing"
grep -q 'source_reader_read_back_hash=resonance-graft-candidate-store-reader-read-' "$RUN_LOG" || die "admission resonance graft admission proof source read-back line missing"
grep -q 'admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance graft admission proof guard line missing"
grep -q 'proof_state=shadow_graft_admission_proved_dry_run proof_action=prove_resonance_shadow_graft_admission_dry_run proof_target=resonance proof_target_kind=internal_world_shadow_graft_admission_proof proof_target_mode=verified_replay_closed_dry_run receipt_shape=resonance_shadow_graft_admission_proof_receipt' "$RUN_LOG" || die "admission resonance graft admission proof shape line missing"
grep -q 'dry_run_only=true reader_verified=true store_verified=true candidate_verified=true ledger_verified=true replay_verified=true hash_verified=true proof_read_back_verified=true proof_ready=true' "$RUN_LOG" || die "admission resonance graft admission proof readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_admission_proof_id=resonance-graft-admission-proof-id-' "$RUN_LOG" || die "admission resonance graft admission proof verdict line missing"
grep -q 'passed=true reason=resonance shadow graft admission proved from read-back store without opening body' "$RUN_LOG" || die "admission resonance graft admission proof reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG resonance_graft_gate=$RESONANCE_GRAFT_GATE_LOG resonance_graft_candidate=$RESONANCE_GRAFT_CANDIDATE_LOG resonance_graft_candidate_store=$RESONANCE_GRAFT_CANDIDATE_STORE_LOG resonance_graft_candidate_store_reader=$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG resonance_graft_admission_proof=$RESONANCE_GRAFT_ADMISSION_PROOF_LOG boundary_report=$BOUNDARY_REPORT"
