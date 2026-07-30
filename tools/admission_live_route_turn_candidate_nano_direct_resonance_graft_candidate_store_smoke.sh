#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_store_smoke.sh - real nano direct -> Resonance shadow graft candidate store receipt.
#
# Extends the closed Resonance shadow-graft candidate with an append-only,
# read-back-verified store receipt. This still opens no raw dream text, Janus
# surface, cooc/delta learning, body mutation, or live admission.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_STORE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-candidate-store.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RESONANCE_GRAFT_GATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_gate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_STORE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_store_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1600 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_LOG="$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke.sh"; then
    die "nano-direct resonance graft candidate smoke with candidate store failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"
[[ -s "$RESONANCE_GRAFT_GATE_LOG" ]] || die "candidate admission resonance graft gate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_LOG" ]] || die "candidate admission resonance graft candidate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" ]] || die "candidate admission resonance graft candidate store JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate_store.v1"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store schema missing"
grep -q '"admission_resonance_graft_candidate_store_state":"shadow_graft_candidate_stored_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store state missing"
grep -q '"admission_resonance_graft_candidate_store_action":"store_resonance_shadow_graft_candidate_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store action missing"
grep -q '"admission_resonance_graft_candidate_store_target":"resonance"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store target missing"
grep -q '"admission_resonance_graft_candidate_store_target_kind":"internal_world_shadow_graft_candidate_store"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store target kind missing"
grep -q '"admission_resonance_graft_candidate_store_target_mode":"append_only_read_back_store_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store target mode missing"
grep -q '"admission_resonance_graft_candidate_store_receipt_shape":"resonance_shadow_graft_candidate_store_receipt"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store receipt shape missing"
grep -q '"admission_resonance_graft_candidate_store_dry_run_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store dry-run flag missing"
grep -q '"admission_resonance_graft_candidate_store_candidate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store candidate flag missing"
grep -q '"admission_resonance_graft_candidate_store_gate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store gate flag missing"
grep -q '"admission_resonance_graft_candidate_store_preflight_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store preflight flag missing"
grep -q '"admission_resonance_graft_candidate_store_boundary_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store boundary flag missing"
grep -q '"admission_resonance_graft_candidate_store_observation_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store observation flag missing"
grep -q '"admission_resonance_graft_candidate_store_receiver_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store receiver flag missing"
grep -q '"admission_resonance_graft_candidate_store_intent_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store intent flag missing"
grep -q '"admission_resonance_graft_candidate_store_final_gate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store final gate flag missing"
grep -q '"admission_resonance_graft_candidate_store_seal_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store seal flag missing"
grep -q '"admission_resonance_graft_candidate_store_permit_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store permit flag missing"
grep -q '"admission_resonance_graft_candidate_store_readiness_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store readiness flag missing"
grep -q '"admission_resonance_graft_candidate_store_ledger_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store ledger flag missing"
grep -q '"admission_resonance_graft_candidate_store_writer_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store writer flag missing"
grep -q '"admission_resonance_graft_candidate_store_rollback_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store rollback flag missing"
grep -q '"admission_resonance_graft_candidate_store_ledger_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store ledger ready flag missing"
grep -q '"admission_resonance_graft_candidate_store_kind":"shadow_graft_candidate_store"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store kind missing"
grep -q '"admission_resonance_graft_candidate_store_mode":"append_only_read_back_store"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store mode missing"
grep -q '"admission_resonance_graft_candidate_store_stage":"pre_live_graft_candidate_store"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store stage missing"
grep -q '"admission_resonance_graft_candidate_store_causal_id":"resonance-graft-candidate-store-causal-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store causal id missing"
grep -q '"admission_resonance_graft_candidate_store_hash":"resonance-graft-candidate-store-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store hash missing"
grep -q '"admission_resonance_graft_candidate_store_read_back_hash":"resonance-graft-candidate-store-read-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store read-back hash missing"
grep -q '"admission_resonance_graft_candidate_store_admission_required":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store admission flag missing"
grep -q '"admission_resonance_graft_candidate_store_shadow_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store shadow flag missing"
grep -q '"admission_resonance_graft_candidate_store_graft_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_janus_surface_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_cooc_learning_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_delta_harvest_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_body_mutation_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_candidate_store_rollback_required":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store rollback flag missing"
grep -q '"admission_resonance_graft_candidate_store_append_only":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store append-only flag missing"
grep -q '"admission_resonance_graft_candidate_store_read_back":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store read-back flag missing"
grep -q '"admission_resonance_graft_candidate_store_receipt_persisted":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store persisted flag missing"
grep -q '"admission_resonance_graft_candidate_store_receipt_verified":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store verified flag missing"
grep -q '"admission_resonance_graft_candidate_store_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate.v1"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate schema missing"
grep -q '"source_admission_resonance_graft_candidate_passed":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate did not pass"
grep -q '"source_admission_resonance_graft_candidate_id":"resonance-graft-candidate-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate id missing"
grep -q '"source_admission_resonance_graft_candidate_action":"draft_resonance_shadow_graft_candidate_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate action missing"
grep -q '"source_admission_resonance_graft_candidate_ready":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate ready flag missing"
grep -q '"source_admission_resonance_graft_candidate_causal_id":"resonance-graft-candidate-causal-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate causal id missing"
grep -q '"source_admission_resonance_graft_candidate_hash":"resonance-graft-candidate-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate hash missing"
grep -q '"source_admission_resonance_graft_candidate_read_back_hash":"resonance-graft-candidate-read-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft candidate read-back hash missing"
grep -q '"source_admission_resonance_graft_gate_id_for_graft_candidate_store":"resonance-graft-gate-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft gate id missing"
grep -q '"source_admission_resonance_graft_preflight_id_for_graft_candidate_store":"resonance-graft-preflight-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft preflight id missing"
grep -q '"source_admission_resonance_graft_boundary_id_for_graft_candidate_store":"resonance-graft-boundary-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance graft boundary id missing"
grep -q '"source_admission_resonance_observation_id_for_graft_candidate_store":"resonance-observation-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "source resonance observation id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "resonance graft candidate store must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_candidate_store_id":"resonance-graft-candidate-store-id-' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_CANDIDATE_STORE_LOG" || die "admission resonance graft candidate store did not pass dry-run"

grep -q 'live-route candidate admission resonance graft candidate store dry-run: class=dream route=direct source=direct candidate=resonance-graft-candidate-id-' "$RUN_LOG" || die "admission resonance graft candidate store chat line missing"
grep -q 'store_kind=shadow_graft_candidate_store store_mode=append_only_read_back_store store_stage=pre_live_graft_candidate_store causal_id=resonance-graft-candidate-store-causal-' "$RUN_LOG" || die "admission resonance graft candidate store kind line missing"
grep -q 'store_hash=resonance-graft-candidate-store-' "$RUN_LOG" || die "admission resonance graft candidate store hash line missing"
grep -q 'read_back_hash=resonance-graft-candidate-store-read-' "$RUN_LOG" || die "admission resonance graft candidate store read-back line missing"
grep -q 'source_candidate_causal_id=resonance-graft-candidate-causal-' "$RUN_LOG" || die "admission resonance graft candidate store source causal line missing"
grep -q 'source_candidate_read_back_hash=resonance-graft-candidate-read-' "$RUN_LOG" || die "admission resonance graft candidate store source read-back line missing"
grep -q 'admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true append_only=true read_back=true receipt_persisted=true receipt_verified=true' "$RUN_LOG" || die "admission resonance graft candidate store guard line missing"
grep -q 'store_state=shadow_graft_candidate_stored_dry_run store_action=store_resonance_shadow_graft_candidate_dry_run store_target=resonance store_target_kind=internal_world_shadow_graft_candidate_store store_target_mode=append_only_read_back_store_dry_run receipt_shape=resonance_shadow_graft_candidate_store_receipt' "$RUN_LOG" || die "admission resonance graft candidate store shape line missing"
grep -q 'dry_run_only=true candidate_verified=true gate_verified=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true store_ready=true' "$RUN_LOG" || die "admission resonance graft candidate store readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_store_id=resonance-graft-candidate-store-id-' "$RUN_LOG" || die "admission resonance graft candidate store verdict line missing"
grep -q 'passed=true reason=resonance shadow graft candidate stored and read back without body mutation' "$RUN_LOG" || die "admission resonance graft candidate store reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG resonance_graft_gate=$RESONANCE_GRAFT_GATE_LOG resonance_graft_candidate=$RESONANCE_GRAFT_CANDIDATE_LOG resonance_graft_candidate_store=$RESONANCE_GRAFT_CANDIDATE_STORE_LOG"
