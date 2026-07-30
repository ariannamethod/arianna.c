#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_reader_smoke.sh - real nano direct -> Resonance shadow graft candidate store reader receipt.
#
# Replays the append-only Resonance shadow-graft candidate store receipt through a
# read-only reader. This proves the stored candidate can be read back without
# opening raw dream text, Janus surface, cooc/delta learning, body mutation, or
# live admission.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_STORE_READER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-candidate-store-reader.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RESONANCE_GRAFT_GATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_gate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_STORE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_store_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_store_reader_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1800 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_STORE_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG="$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke.sh"; then
    die "nano-direct resonance graft candidate store smoke with store reader failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"
[[ -s "$RESONANCE_GRAFT_GATE_LOG" ]] || die "candidate admission resonance graft gate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_LOG" ]] || die "candidate admission resonance graft candidate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" ]] || die "candidate admission resonance graft candidate store JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" ]] || die "candidate admission resonance graft candidate store reader JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate_store_reader.v1"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader schema missing"
grep -q '"timing":"live_admission_resonance_graft_candidate_store_reader"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader timing missing"
grep -q '"admission_resonance_graft_candidate_store_reader_state":"shadow_graft_candidate_store_read_back_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader state missing"
grep -q '"admission_resonance_graft_candidate_store_reader_action":"read_resonance_shadow_graft_candidate_store_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader action missing"
grep -q '"admission_resonance_graft_candidate_store_reader_target":"resonance"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader target missing"
grep -q '"admission_resonance_graft_candidate_store_reader_target_kind":"internal_world_shadow_graft_candidate_store_reader"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader target kind missing"
grep -q '"admission_resonance_graft_candidate_store_reader_target_mode":"read_only_replay_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader target mode missing"
grep -q '"admission_resonance_graft_candidate_store_reader_receipt_shape":"resonance_shadow_graft_candidate_store_reader_receipt"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader receipt shape missing"
grep -q '"admission_resonance_graft_candidate_store_reader_dry_run_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader dry-run flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_store_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader store flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_candidate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader candidate flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_ledger_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader ledger flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_read_back_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader read-back flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_hash_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader hash flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_kind":"shadow_graft_candidate_store_reader"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader kind missing"
grep -q '"admission_resonance_graft_candidate_store_reader_mode":"read_only_replay"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader mode missing"
grep -q '"admission_resonance_graft_candidate_store_reader_stage":"pre_live_graft_candidate_store_reader"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader stage missing"
grep -q '"admission_resonance_graft_candidate_store_reader_causal_id":"resonance-graft-candidate-store-reader-causal-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader causal id missing"
grep -q '"admission_resonance_graft_candidate_store_reader_hash":"resonance-graft-candidate-store-reader-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader hash missing"
grep -q '"admission_resonance_graft_candidate_store_reader_replay_hash":"resonance-graft-candidate-store-reader-replay-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader replay hash missing"
grep -q '"admission_resonance_graft_candidate_store_reader_read_back_hash":"resonance-graft-candidate-store-reader-read-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader read-back hash missing"
grep -q '"admission_resonance_graft_candidate_store_reader_read_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader read-only flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_replay_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader replay-only flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_source_append_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader source append flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_source_read_back":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader source read-back flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_source_receipt_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader source receipt flag missing"
grep -q '"admission_resonance_graft_candidate_store_reader_graft_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_janus_surface_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_cooc_learning_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_delta_harvest_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_body_mutation_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_reader_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_store_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate_store.v1"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store schema missing"
grep -q '"source_admission_resonance_graft_candidate_store_passed":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store did not pass"
grep -q '"source_admission_resonance_graft_candidate_store_id":"resonance-graft-candidate-store-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store id missing"
grep -q '"source_admission_resonance_graft_candidate_store_action":"store_resonance_shadow_graft_candidate_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store action missing"
grep -q '"source_admission_resonance_graft_candidate_store_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_store_causal_id":"resonance-graft-candidate-store-causal-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store causal id missing"
grep -q '"source_admission_resonance_graft_candidate_store_hash":"resonance-graft-candidate-store-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store hash missing"
grep -q '"source_admission_resonance_graft_candidate_store_read_back_hash":"resonance-graft-candidate-store-read-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate store read-back hash missing"
grep -q '"source_admission_resonance_graft_candidate_id_for_store_reader":"resonance-graft-candidate-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft candidate id missing"
grep -q '"source_admission_resonance_graft_gate_id_for_store_reader":"resonance-graft-gate-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance graft gate id missing"
grep -q '"source_admission_resonance_observation_id_for_store_reader":"resonance-observation-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source resonance observation id missing"
grep -q '"source_admission_final_gate_id_for_store_reader":"admission-final-gate-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source admission final gate id missing"
grep -q '"source_ledger_verification_id_for_store_reader":"ledger-verification-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "source ledger verification id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "resonance graft candidate store reader must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_candidate_store_reader_id":"resonance-graft-candidate-store-reader-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG" || die "admission resonance graft candidate store reader did not pass dry-run"

grep -q 'live-route candidate admission resonance graft candidate store reader dry-run: class=dream route=direct source=direct store=resonance-graft-candidate-store-id-' "$RUN_LOG" || die "admission resonance graft candidate store reader chat line missing"
grep -q 'reader_kind=shadow_graft_candidate_store_reader reader_mode=read_only_replay reader_stage=pre_live_graft_candidate_store_reader causal_id=resonance-graft-candidate-store-reader-causal-' "$RUN_LOG" || die "admission resonance graft candidate store reader kind line missing"
grep -q 'reader_hash=resonance-graft-candidate-store-reader-' "$RUN_LOG" || die "admission resonance graft candidate store reader hash line missing"
grep -q 'replay_hash=resonance-graft-candidate-store-reader-replay-' "$RUN_LOG" || die "admission resonance graft candidate store reader replay line missing"
grep -q 'read_back_hash=resonance-graft-candidate-store-reader-read-' "$RUN_LOG" || die "admission resonance graft candidate store reader read-back line missing"
grep -q 'source_store_causal_id=resonance-graft-candidate-store-causal-' "$RUN_LOG" || die "admission resonance graft candidate store reader source causal line missing"
grep -q 'source_store_read_back_hash=resonance-graft-candidate-store-read-' "$RUN_LOG" || die "admission resonance graft candidate store reader source read-back line missing"
grep -q 'read_only=true replay_only=true source_append_only=true source_read_back=true source_receipt_verified=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false' "$RUN_LOG" || die "admission resonance graft candidate store reader guard line missing"
grep -q 'reader_state=shadow_graft_candidate_store_read_back_dry_run reader_action=read_resonance_shadow_graft_candidate_store_dry_run reader_target=resonance reader_target_kind=internal_world_shadow_graft_candidate_store_reader reader_target_mode=read_only_replay_dry_run receipt_shape=resonance_shadow_graft_candidate_store_reader_receipt' "$RUN_LOG" || die "admission resonance graft candidate store reader shape line missing"
grep -q 'dry_run_only=true store_verified=true candidate_verified=true ledger_verified=true hash_verified=true reader_read_back_verified=true reader_ready=true' "$RUN_LOG" || die "admission resonance graft candidate store reader readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_store_reader_id=resonance-graft-candidate-store-reader-id-' "$RUN_LOG" || die "admission resonance graft candidate store reader verdict line missing"
grep -q 'passed=true reason=resonance shadow graft candidate store read back without opening body' "$RUN_LOG" || die "admission resonance graft candidate store reader reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG resonance_graft_gate=$RESONANCE_GRAFT_GATE_LOG resonance_graft_candidate=$RESONANCE_GRAFT_CANDIDATE_LOG resonance_graft_candidate_store=$RESONANCE_GRAFT_CANDIDATE_STORE_LOG resonance_graft_candidate_store_reader=$RESONANCE_GRAFT_CANDIDATE_STORE_READER_LOG"
