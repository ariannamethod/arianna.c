#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke.sh - real nano direct -> Resonance shadow graft gate receipt.
#
# Extends the closed Resonance shadow-graft preflight with a typed gate receipt.
# This still opens no raw dream text, Janus surface, cooc/delta learning, body
# mutation, or live admission.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-gate.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RESONANCE_GRAFT_GATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_gate_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1200 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_PREFLIGHT_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_GATE_LOG="$RESONANCE_GRAFT_GATE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_preflight_smoke.sh"; then
    die "nano-direct resonance graft preflight smoke with graft gate failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"
[[ -s "$RESONANCE_GRAFT_GATE_LOG" ]] || die "candidate admission resonance graft gate JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_gate.v1"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate schema missing"
grep -q '"admission_resonance_graft_gate_state":"shadow_graft_gate_ready_dry_run"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate state missing"
grep -q '"admission_resonance_graft_gate_action":"gate_resonance_shadow_graft_dry_run"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate action missing"
grep -q '"admission_resonance_graft_gate_target":"resonance"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate target missing"
grep -q '"admission_resonance_graft_gate_target_kind":"internal_world_shadow_graft_gate"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate target kind missing"
grep -q '"admission_resonance_graft_gate_target_mode":"receipt_only_closed_gate_dry_run"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate target mode missing"
grep -q '"admission_resonance_graft_gate_receipt_shape":"resonance_shadow_graft_gate_contract"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate receipt shape missing"
grep -q '"admission_resonance_graft_gate_dry_run_only":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate dry-run flag missing"
grep -q '"admission_resonance_graft_gate_preflight_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate preflight flag missing"
grep -q '"admission_resonance_graft_gate_boundary_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate boundary flag missing"
grep -q '"admission_resonance_graft_gate_observation_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate observation flag missing"
grep -q '"admission_resonance_graft_gate_receiver_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate receiver flag missing"
grep -q '"admission_resonance_graft_gate_intent_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate intent flag missing"
grep -q '"admission_resonance_graft_gate_final_gate_verified":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate final gate flag missing"
grep -q '"admission_resonance_graft_gate_kind":"shadow_graft_gate"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate kind missing"
grep -q '"admission_resonance_graft_gate_mode":"no_mutation_gate"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate mode missing"
grep -q '"admission_resonance_graft_gate_stage":"pre_live_graft_gate"' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate stage missing"
grep -q '"admission_resonance_graft_gate_causal_id":"resonance-graft-gate-causal-' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate causal id missing"
grep -q '"admission_resonance_graft_gate_hash":"resonance-graft-gate-' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate hash missing"
grep -q '"admission_resonance_graft_gate_read_back_hash":"resonance-graft-gate-read-' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate read-back hash missing"
grep -q '"admission_resonance_graft_gate_admission_required":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate admission flag missing"
grep -q '"admission_resonance_graft_gate_shadow_only":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate shadow flag missing"
grep -q '"admission_resonance_graft_gate_graft_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_gate_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_gate_janus_surface_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_gate_cooc_learning_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_gate_delta_harvest_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_gate_body_mutation_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_gate_rollback_required":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate rollback flag missing"
grep -q '"admission_resonance_graft_gate_ready":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate ready flag missing"
grep -q '"source_admission_resonance_graft_preflight_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_preflight.v1"' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight schema missing"
grep -q '"source_admission_resonance_graft_preflight_passed":true' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight did not pass"
grep -q '"source_admission_resonance_graft_preflight_id":"resonance-graft-preflight-id-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight id missing"
grep -q '"source_admission_resonance_graft_preflight_action":"prepare_resonance_shadow_graft_preflight_dry_run"' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight action missing"
grep -q '"source_admission_resonance_graft_preflight_ready":true' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight ready flag missing"
grep -q '"source_admission_resonance_graft_preflight_causal_id":"resonance-graft-preflight-causal-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight causal id missing"
grep -q '"source_admission_resonance_graft_preflight_hash":"resonance-graft-preflight-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight hash missing"
grep -q '"source_admission_resonance_graft_preflight_read_back_hash":"resonance-graft-preflight-read-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft preflight read-back hash missing"
grep -q '"source_admission_resonance_graft_boundary_id_for_graft_gate":"resonance-graft-boundary-id-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance graft boundary id missing"
grep -q '"source_admission_resonance_observation_id_for_graft_gate":"resonance-observation-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance observation id missing"
grep -q '"source_admission_resonance_receiver_id_for_graft_gate":"resonance-receiver-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance receiver id missing"
grep -q '"source_admission_resonance_intent_id_for_graft_gate":"resonance-intent-' "$RESONANCE_GRAFT_GATE_LOG" || die "source resonance intent id missing"
grep -q '"source_admission_final_gate_id_for_graft_gate":"admission-final-gate-' "$RESONANCE_GRAFT_GATE_LOG" || die "source final gate id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_GATE_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_GATE_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_GATE_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_GATE_LOG" || die "resonance graft gate must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_GATE_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_gate_id":"resonance-graft-gate-id-' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_GATE_LOG" || die "admission resonance graft gate did not pass dry-run"

grep -q 'live-route candidate admission resonance graft gate dry-run: class=dream route=direct source=direct preflight=resonance-graft-preflight-id-' "$RUN_LOG" || die "admission resonance graft gate chat line missing"
grep -q 'gate_kind=shadow_graft_gate gate_mode=no_mutation_gate gate_stage=pre_live_graft_gate causal_id=resonance-graft-gate-causal-' "$RUN_LOG" || die "admission resonance graft gate kind line missing"
grep -q 'gate_hash=resonance-graft-gate-' "$RUN_LOG" || die "admission resonance graft gate hash line missing"
grep -q 'read_back_hash=resonance-graft-gate-read-' "$RUN_LOG" || die "admission resonance graft gate read-back line missing"
grep -q 'source_preflight_causal_id=resonance-graft-preflight-causal-' "$RUN_LOG" || die "admission resonance graft gate source causal line missing"
grep -q 'source_preflight_read_back_hash=resonance-graft-preflight-read-' "$RUN_LOG" || die "admission resonance graft gate source read-back line missing"
grep -q 'admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance graft gate guard line missing"
grep -q 'gate_state=shadow_graft_gate_ready_dry_run gate_action=gate_resonance_shadow_graft_dry_run gate_target=resonance gate_target_kind=internal_world_shadow_graft_gate gate_target_mode=receipt_only_closed_gate_dry_run receipt_shape=resonance_shadow_graft_gate_contract' "$RUN_LOG" || die "admission resonance graft gate shape line missing"
grep -q 'dry_run_only=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true gate_ready=true' "$RUN_LOG" || die "admission resonance graft gate readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_gate_id=resonance-graft-gate-id-' "$RUN_LOG" || die "admission resonance graft gate verdict line missing"
grep -q 'passed=true reason=resonance shadow graft gate prepared without body mutation' "$RUN_LOG" || die "admission resonance graft gate reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG resonance_graft_gate=$RESONANCE_GRAFT_GATE_LOG"
