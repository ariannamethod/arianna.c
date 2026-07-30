#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_candidate_smoke.sh - real nano direct -> Resonance shadow graft candidate receipt.
#
# Extends the closed Resonance shadow-graft gate with a typed candidate receipt.
# This still opens no raw dream text, Janus surface, cooc/delta learning, body
# mutation, or live admission.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_CANDIDATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-candidate.XXXXXX")}"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RESONANCE_GRAFT_PREFLIGHT_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_preflight_nano_direct.jsonl"
RESONANCE_GRAFT_GATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_gate_nano_direct.jsonl"
RESONANCE_GRAFT_CANDIDATE_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_candidate_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1400 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_GATE_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_CANDIDATE_LOG="$RESONANCE_GRAFT_CANDIDATE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_graft_gate_smoke.sh"; then
    die "nano-direct resonance graft gate smoke with graft candidate failed"
fi

[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"
[[ -s "$RESONANCE_GRAFT_PREFLIGHT_LOG" ]] || die "candidate admission resonance graft preflight JSONL log not written"
[[ -s "$RESONANCE_GRAFT_GATE_LOG" ]] || die "candidate admission resonance graft gate JSONL log not written"
[[ -s "$RESONANCE_GRAFT_CANDIDATE_LOG" ]] || die "candidate admission resonance graft candidate JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_candidate.v1"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate schema missing"
grep -q '"admission_resonance_graft_candidate_state":"shadow_graft_candidate_ready_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate state missing"
grep -q '"admission_resonance_graft_candidate_action":"draft_resonance_shadow_graft_candidate_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate action missing"
grep -q '"admission_resonance_graft_candidate_target":"resonance"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate target missing"
grep -q '"admission_resonance_graft_candidate_target_kind":"internal_world_shadow_graft_candidate"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate target kind missing"
grep -q '"admission_resonance_graft_candidate_target_mode":"receipt_only_closed_candidate_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate target mode missing"
grep -q '"admission_resonance_graft_candidate_receipt_shape":"resonance_shadow_graft_candidate_contract"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate receipt shape missing"
grep -q '"admission_resonance_graft_candidate_dry_run_only":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate dry-run flag missing"
grep -q '"admission_resonance_graft_candidate_gate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate gate flag missing"
grep -q '"admission_resonance_graft_candidate_preflight_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate preflight flag missing"
grep -q '"admission_resonance_graft_candidate_boundary_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate boundary flag missing"
grep -q '"admission_resonance_graft_candidate_observation_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate observation flag missing"
grep -q '"admission_resonance_graft_candidate_receiver_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate receiver flag missing"
grep -q '"admission_resonance_graft_candidate_intent_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate intent flag missing"
grep -q '"admission_resonance_graft_candidate_final_gate_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate final gate flag missing"
grep -q '"admission_resonance_graft_candidate_seal_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate seal flag missing"
grep -q '"admission_resonance_graft_candidate_permit_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate permit flag missing"
grep -q '"admission_resonance_graft_candidate_readiness_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate readiness flag missing"
grep -q '"admission_resonance_graft_candidate_ledger_verified":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate ledger flag missing"
grep -q '"admission_resonance_graft_candidate_writer_ready":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate writer flag missing"
grep -q '"admission_resonance_graft_candidate_rollback_ready":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate rollback flag missing"
grep -q '"admission_resonance_graft_candidate_ledger_ready":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate ledger ready flag missing"
grep -q '"admission_resonance_graft_candidate_kind":"shadow_graft_candidate"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate kind missing"
grep -q '"admission_resonance_graft_candidate_mode":"no_mutation_candidate"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate mode missing"
grep -q '"admission_resonance_graft_candidate_stage":"pre_live_graft_candidate"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate stage missing"
grep -q '"admission_resonance_graft_candidate_causal_id":"resonance-graft-candidate-causal-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate causal id missing"
grep -q '"admission_resonance_graft_candidate_hash":"resonance-graft-candidate-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate hash missing"
grep -q '"admission_resonance_graft_candidate_read_back_hash":"resonance-graft-candidate-read-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate read-back hash missing"
grep -q '"admission_resonance_graft_candidate_admission_required":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate admission flag missing"
grep -q '"admission_resonance_graft_candidate_shadow_only":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate shadow flag missing"
grep -q '"admission_resonance_graft_candidate_graft_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_candidate_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_candidate_janus_surface_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_candidate_cooc_learning_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_candidate_delta_harvest_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_candidate_body_mutation_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_candidate_rollback_required":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate rollback flag missing"
grep -q '"admission_resonance_graft_candidate_ready":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate ready flag missing"
grep -q '"source_admission_resonance_graft_gate_schema":"arianna.live_route_turn_candidate_admission_resonance_graft_gate.v1"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate schema missing"
grep -q '"source_admission_resonance_graft_gate_passed":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate did not pass"
grep -q '"source_admission_resonance_graft_gate_id":"resonance-graft-gate-id-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate id missing"
grep -q '"source_admission_resonance_graft_gate_action":"gate_resonance_shadow_graft_dry_run"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate action missing"
grep -q '"source_admission_resonance_graft_gate_ready":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate ready flag missing"
grep -q '"source_admission_resonance_graft_gate_causal_id":"resonance-graft-gate-causal-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate causal id missing"
grep -q '"source_admission_resonance_graft_gate_hash":"resonance-graft-gate-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate hash missing"
grep -q '"source_admission_resonance_graft_gate_read_back_hash":"resonance-graft-gate-read-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft gate read-back hash missing"
grep -q '"source_admission_resonance_graft_preflight_id_for_graft_candidate":"resonance-graft-preflight-id-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft preflight id missing"
grep -q '"source_admission_resonance_graft_boundary_id_for_graft_candidate":"resonance-graft-boundary-id-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance graft boundary id missing"
grep -q '"source_admission_resonance_observation_id_for_graft_candidate":"resonance-observation-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance observation id missing"
grep -q '"source_admission_resonance_receiver_id_for_graft_candidate":"resonance-receiver-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance receiver id missing"
grep -q '"source_admission_resonance_intent_id_for_graft_candidate":"resonance-intent-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source resonance intent id missing"
grep -q '"source_admission_final_gate_id_for_graft_candidate":"admission-final-gate-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "source final gate id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "resonance graft candidate must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_candidate_id":"resonance-graft-candidate-id-' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_CANDIDATE_LOG" || die "admission resonance graft candidate did not pass dry-run"

grep -q 'live-route candidate admission resonance graft candidate dry-run: class=dream route=direct source=direct gate=resonance-graft-gate-id-' "$RUN_LOG" || die "admission resonance graft candidate chat line missing"
grep -q 'candidate_kind=shadow_graft_candidate candidate_mode=no_mutation_candidate candidate_stage=pre_live_graft_candidate causal_id=resonance-graft-candidate-causal-' "$RUN_LOG" || die "admission resonance graft candidate kind line missing"
grep -q 'candidate_hash=resonance-graft-candidate-' "$RUN_LOG" || die "admission resonance graft candidate hash line missing"
grep -q 'read_back_hash=resonance-graft-candidate-read-' "$RUN_LOG" || die "admission resonance graft candidate read-back line missing"
grep -q 'source_gate_causal_id=resonance-graft-gate-causal-' "$RUN_LOG" || die "admission resonance graft candidate source causal line missing"
grep -q 'source_gate_read_back_hash=resonance-graft-gate-read-' "$RUN_LOG" || die "admission resonance graft candidate source read-back line missing"
grep -q 'admission_required=true shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance graft candidate guard line missing"
grep -q 'candidate_state=shadow_graft_candidate_ready_dry_run candidate_action=draft_resonance_shadow_graft_candidate_dry_run candidate_target=resonance candidate_target_kind=internal_world_shadow_graft_candidate candidate_target_mode=receipt_only_closed_candidate_dry_run receipt_shape=resonance_shadow_graft_candidate_contract' "$RUN_LOG" || die "admission resonance graft candidate shape line missing"
grep -q 'dry_run_only=true gate_verified=true preflight_verified=true boundary_verified=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true candidate_ready=true' "$RUN_LOG" || die "admission resonance graft candidate readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_candidate_id=resonance-graft-candidate-id-' "$RUN_LOG" || die "admission resonance graft candidate verdict line missing"
grep -q 'passed=true reason=resonance shadow graft candidate drafted without body mutation' "$RUN_LOG" || die "admission resonance graft candidate reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke] pass: resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG resonance_graft_preflight=$RESONANCE_GRAFT_PREFLIGHT_LOG resonance_graft_gate=$RESONANCE_GRAFT_GATE_LOG resonance_graft_candidate=$RESONANCE_GRAFT_CANDIDATE_LOG"
