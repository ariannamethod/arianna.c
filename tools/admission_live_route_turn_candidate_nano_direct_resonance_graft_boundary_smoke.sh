#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_resonance_graft_boundary_smoke.sh - real nano direct -> Resonance shadow graft boundary receipt.
#
# Extends the Resonance-observation smoke with a closed shadow-graft boundary.
# This is still receipt-only: no raw dream text, Janus surface, cooc/delta
# learning, body mutation, or live admission is opened.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_GRAFT_BOUNDARY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-resonance-graft-boundary.XXXXXX")}"
RESONANCE_OBSERVATION_LOG="$WORKDIR/live_route_candidate_admission_resonance_observation_nano_direct.jsonl"
RESONANCE_GRAFT_BOUNDARY_LOG="$WORKDIR/live_route_candidate_admission_resonance_graft_boundary_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 1000 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RESONANCE_OBSERVATION_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_RESONANCE_GRAFT_BOUNDARY_LOG="$RESONANCE_GRAFT_BOUNDARY_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_resonance_observation_smoke.sh"; then
    die "nano-direct resonance observation smoke with admission resonance graft boundary failed"
fi

[[ -s "$RESONANCE_OBSERVATION_LOG" ]] || die "candidate admission resonance observation JSONL log not written"
[[ -s "$RESONANCE_GRAFT_BOUNDARY_LOG" ]] || die "candidate admission resonance graft boundary JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_boundary.v1"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary schema missing"
grep -q '"admission_resonance_graft_boundary_state":"shadow_graft_boundary_declared_dry_run"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary state missing"
grep -q '"admission_resonance_graft_boundary_action":"declare_resonance_shadow_graft_boundary_dry_run"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary action missing"
grep -q '"admission_resonance_graft_boundary_target":"resonance"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary target missing"
grep -q '"admission_resonance_graft_boundary_target_kind":"internal_world_shadow_graft"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary target kind missing"
grep -q '"admission_resonance_graft_boundary_target_mode":"receipt_only_closed_dry_run"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary target mode missing"
grep -q '"admission_resonance_graft_boundary_receipt_shape":"resonance_observation_shadow_graft_boundary"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary receipt shape missing"
grep -q '"admission_resonance_graft_boundary_dry_run_only":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary dry-run flag missing"
grep -q '"admission_resonance_graft_boundary_observation_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary observation flag missing"
grep -q '"admission_resonance_graft_boundary_receiver_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary receiver flag missing"
grep -q '"admission_resonance_graft_boundary_intent_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary intent flag missing"
grep -q '"admission_resonance_graft_boundary_final_gate_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary final gate flag missing"
grep -q '"admission_resonance_graft_boundary_seal_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary seal flag missing"
grep -q '"admission_resonance_graft_boundary_permit_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary permit flag missing"
grep -q '"admission_resonance_graft_boundary_readiness_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary readiness flag missing"
grep -q '"admission_resonance_graft_boundary_ledger_verified":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary ledger flag missing"
grep -q '"admission_resonance_graft_boundary_writer_ready":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary writer flag missing"
grep -q '"admission_resonance_graft_boundary_rollback_ready":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary rollback flag missing"
grep -q '"admission_resonance_graft_boundary_ledger_ready":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary ledger ready flag missing"
grep -q '"admission_resonance_graft_boundary_kind":"shadow_graft_boundary"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary kind missing"
grep -q '"admission_resonance_graft_boundary_mode":"no_mutation_receipt"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary mode missing"
grep -q '"admission_resonance_graft_boundary_stage":"pre_live_graft"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary stage missing"
grep -q '"admission_resonance_graft_boundary_causal_id":"resonance-graft-boundary-causal-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary causal id missing"
grep -q '"admission_resonance_graft_boundary_hash":"resonance-graft-boundary-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary hash missing"
grep -q '"admission_resonance_graft_boundary_read_back_hash":"resonance-graft-boundary-read-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary read-back hash missing"
grep -q '"admission_resonance_graft_boundary_shadow_only":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary shadow flag missing"
grep -q '"admission_resonance_graft_boundary_graft_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "graft must stay blocked"
grep -q '"admission_resonance_graft_boundary_raw_dream_text_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "raw dream text must stay blocked"
grep -q '"admission_resonance_graft_boundary_janus_surface_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "Janus surface must stay blocked"
grep -q '"admission_resonance_graft_boundary_cooc_learning_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "cooc learning must stay blocked"
grep -q '"admission_resonance_graft_boundary_delta_harvest_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "delta harvest must stay blocked"
grep -q '"admission_resonance_graft_boundary_body_mutation_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "body mutation must stay blocked"
grep -q '"admission_resonance_graft_boundary_rollback_required":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary rollback flag missing"
grep -q '"admission_resonance_graft_boundary_ready":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary ready flag missing"
grep -q '"source_admission_resonance_observation_schema":"arianna.live_route_turn_candidate_admission_resonance_observation.v1"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation schema missing"
grep -q '"source_admission_resonance_observation_passed":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation did not pass"
grep -q '"source_admission_resonance_observation_id":"resonance-observation-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation id missing"
grep -q '"source_admission_resonance_observation_action":"record_resonance_receiver_observation_dry_run"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation action missing"
grep -q '"source_admission_resonance_observation_ready":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation ready flag missing"
grep -q '"source_admission_resonance_observation_causal_id":"resonance-observation-causal-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation causal id missing"
grep -q '"source_admission_resonance_observation_append_hash":"resonance-observation-append-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation append hash missing"
grep -q '"source_admission_resonance_observation_read_back_hash":"resonance-observation-read-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance observation read-back hash missing"
grep -q '"source_admission_resonance_receiver_id_for_graft_boundary":"resonance-receiver-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "source resonance receiver id missing"
grep -q '"contracts_ready":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "contracts must remain disabled"
grep -q '"write_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "body write must remain disabled"
grep -q '"admission_allowed":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission must remain disabled"
grep -q '"live_admission_enabled":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "live admission must remain disabled"
grep -q '"mutates_state":false' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "resonance graft boundary must not mutate organism state"
grep -q '"body_target":"none"' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "body target must remain none"
grep -q '"admission_resonance_graft_boundary_id":"resonance-graft-boundary-id-' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary id missing"
grep -q '"passed":true' "$RESONANCE_GRAFT_BOUNDARY_LOG" || die "admission resonance graft boundary did not pass dry-run"

grep -q 'live-route candidate admission resonance graft boundary dry-run: class=dream route=direct source=direct observation=resonance-observation-' "$RUN_LOG" || die "admission resonance graft boundary chat line missing"
grep -q 'boundary_kind=shadow_graft_boundary boundary_mode=no_mutation_receipt boundary_stage=pre_live_graft causal_id=resonance-graft-boundary-causal-' "$RUN_LOG" || die "admission resonance graft boundary kind line missing"
grep -q 'boundary_hash=resonance-graft-boundary-' "$RUN_LOG" || die "admission resonance graft boundary hash line missing"
grep -q 'read_back_hash=resonance-graft-boundary-read-' "$RUN_LOG" || die "admission resonance graft boundary read-back line missing"
grep -q 'source_observation_causal_id=resonance-observation-causal-' "$RUN_LOG" || die "admission resonance graft boundary source causal line missing"
grep -q 'source_observation_read_back_hash=resonance-observation-read-' "$RUN_LOG" || die "admission resonance graft boundary source read-back line missing"
grep -q 'shadow_only=true graft_allowed=false raw_text_allowed=false janus_surface_allowed=false cooc_learning_allowed=false delta_harvest_allowed=false body_mutation_allowed=false rollback_required=true' "$RUN_LOG" || die "admission resonance graft boundary guard line missing"
grep -q 'boundary_state=shadow_graft_boundary_declared_dry_run boundary_action=declare_resonance_shadow_graft_boundary_dry_run boundary_target=resonance boundary_target_kind=internal_world_shadow_graft boundary_target_mode=receipt_only_closed_dry_run receipt_shape=resonance_observation_shadow_graft_boundary' "$RUN_LOG" || die "admission resonance graft boundary shape line missing"
grep -q 'dry_run_only=true observation_verified=true receiver_verified=true intent_verified=true final_gate_verified=true seal_verified=true permit_verified=true readiness_verified=true ledger_verified=true writer_ready=true rollback_ready=true ledger_ready=true boundary_ready=true' "$RUN_LOG" || die "admission resonance graft boundary readiness line missing"
grep -q 'contracts_ready=false write_allowed=false admission_allowed=false live_ready=true live_enabled=false mutates=false admission_resonance_graft_boundary_id=resonance-graft-boundary-id-' "$RUN_LOG" || die "admission resonance graft boundary verdict line missing"
grep -q 'passed=true reason=resonance shadow graft boundary declared without body mutation' "$RUN_LOG" || die "admission resonance graft boundary reason missing"

echo "[admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke] pass: resonance_observation=$RESONANCE_OBSERVATION_LOG resonance_graft_boundary=$RESONANCE_GRAFT_BOUNDARY_LOG"
