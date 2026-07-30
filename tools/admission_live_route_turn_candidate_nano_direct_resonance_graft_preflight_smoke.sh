#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke.sh - real nano direct -> Resonance shadow graft preflight receipt.
#
# Extends the closed Resonance shadow-graft boundary with a typed preflight
# receipt. This still opens no raw dream text, Janus surface, cooc/delta
# learning, body mutation, or live admission.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_PREFLIGHT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-preflight.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1000 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_BOUNDARY_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_LOG="$RESONANCE_GRAFT_PREFLIGHT_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke.sh"; then
    die "nano-direct resonance graft boundary smoke with graft preflight failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_preflight.v1"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight schema missing"
grep -q '"admission_resonance_graft_preflight_state":"shadow_graft_preflight_ready_dry_run"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight state missing"
grep -q '"admission_resonance_graft_preflight_action":"prepare_resonance_shadow_graft_preflight_dry_run"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight action missing"
grep -q '"admission_resonance_graft_preflight_target":"resonance"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight target missing"
grep -q '"admission_resonance_graft_preflight_target_kind":"internal_world_shadow_graft_preflight"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight target kind missing"
grep -q '"admission_resonance_graft_preflight_target_mode":"receipt_only_closed_preflight_dry_run"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight target mode missing"
grep -q '"admission_resonance_graft_preflight_receipt_shape":"resonance_shadow_graft_preflight_contract"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight receipt shape missing"
grep -q '"admission_resonance_graft_preflight_dry_run_only":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight dry-run flag missing"
grep -q '"admission_resonance_graft_preflight_boundary_verified":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight boundary flag missing"
grep -q '"admission_resonance_graft_preflight_observation_verified":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight observation flag missing"
grep -q '"admission_resonance_graft_preflight_receiver_verified":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight receiver flag missing"
grep -q '"admission_resonance_graft_preflight_intent_verified":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight intent flag missing"
grep -q '"admission_resonance_graft_preflight_final_gate_verified":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight final gate flag missing"
grep -q '"admission_resonance_graft_preflight_kind":"shadow_graft_preflight"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight kind missing"
grep -q '"admission_resonance_graft_preflight_mode":"no_mutation_preflight"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight mode missing"
grep -q '"admission_resonance_graft_preflight_stage":"pre_live_graft_admission"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight stage missing"
grep -q '"admission_resonance_graft_preflight_causal_id":"resonance-graft-preflight-causal-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight causal id missing"
grep -q '"admission_resonance_graft_preflight_hash":"resonance-graft-preflight-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight hash missing"
grep -q '"admission_resonance_graft_preflight_read_back_hash":"resonance-graft-preflight-read-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight read-back hash missing"
grep -q '"admission_resonance_graft_preflight_admission_required":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight admission flag missing"
grep -q '"admission_resonance_graft_preflight_shadow_only":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight shadow flag missing"
grep -q '"admission_resonance_graft_preflight_graft_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_preflight_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_preflight_janus_surface_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_preflight_cooc_learning_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_preflight_delta_harvest_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_preflight_body_mutation_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_preflight_rollback_required":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight rollback flag missing"
grep -q '"admission_resonance_graft_preflight_ready":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight ready flag missing"
grep -q '"source_admission_resonance_graft_boundary_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_boundary.v1"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary schema missing"
grep -q '"source_admission_resonance_graft_boundary_passed":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary did not pass"
grep -q '"source_admission_resonance_graft_boundary_id":"resonance-graft-boundary-id-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary id missing"
grep -q '"source_admission_resonance_graft_boundary_action":"declare_resonance_shadow_graft_boundary_dry_run"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary action missing"
grep -q '"source_admission_resonance_graft_boundary_ready":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary ready flag missing"
grep -q '"source_admission_resonance_graft_boundary_causal_id":"resonance-graft-boundary-causal-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary causal id missing"
grep -q '"source_admission_resonance_graft_boundary_hash":"resonance-graft-boundary-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary hash missing"
grep -q '"source_admission_resonance_graft_boundary_read_back_hash":"resonance-graft-boundary-read-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "source resonance graft boundary read-back hash missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "resonance graft preflight must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_preflight_id":"resonance-graft-preflight-id-' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_PREFLIGHT_LOG" || die "admission resonance graft preflight did not pass dry-run"

grep -q 'live-route candidate admission resonance graft preflight dry-run: class=dream route=direct source=direct boundary=resonance-graft-boundary-id-' "$RUN_LOG" || die "admission resonance graft preflight chat line missing"
grep -q 'preflight_kind=shadow_graft_preflight preflight_mode=no_mutation_preflight preflight_stage=pre_live_graft_admission causal_id=resonance-graft-preflight-causal-' "$RUN_LOG" || die "admission resonance graft preflight kind line missing"
grep -q 'preflight_hash=resonance-graft-preflight-' "$RUN_LOG" || die "admission resonance graft preflight hash line missing"
grep -q 'read_back_hash=resonance-graft-preflight-read-' "$RUN_LOG" || die "admission resonance graft preflight read-back line missing"
grep -q 'source_boundary_causal_id=resonance-graft-boundary-causal-' "$RUN_LOG" || die "admission resonance graft preflight source causal line missing"
grep -q 'source_boundary_read_back_hash=resonance-graft-boundary-read-' "$RUN_LOG" || die "admission resonance graft preflight source read-back line missing"
grep -q 'admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance graft preflight guard line missing"
grep -q 'preflight_state=shadow_graft_preflight_ready_dry_run preflight_action=prepare_resonance_shadow_graft_preflight_dry_run preflight_target=resonance preflight_target_kind=internal_world_shadow_graft_preflight preflight_target_mode=receipt_only_closed_preflight_dry_run receipt_shape=resonance_shadow_graft_preflight_contract' "$RUN_LOG" || die "admission resonance graft preflight shape line missing"
grep -q 'dry_run_only=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true preflight_ready=true' "$RUN_LOG" || die "admission resonance graft preflight readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_preflight_id=resonance-graft-preflight-id-' "$RUN_LOG" || die "admission resonance graft preflight verdict line missing"
grep -q 'passed=true reason=resonance shadow graft preflight prepared without body mutation' "$RUN_LOG" || die "admission resonance graft preflight reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG"
