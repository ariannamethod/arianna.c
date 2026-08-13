#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_boundary_smoke.sh - declare weighted Resonance shadow graft boundary.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-boundary.XXXXXX")}"
OBSERVATION_WORKDIR="$WORKDIR/observation"
OBSERVATION_REPORT="$OBSERVATION_WORKDIR/live_route_weighted_admission_resonance_observation.json"
GRAFT_BOUNDARY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_boundary.json}"
OBSERVATION_LOG="$WORKDIR/weighted_admission_resonance_observation.log"
GRAFT_BOUNDARY_LOG="$WORKDIR/weighted_admission_resonance_graft_boundary.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-boundary-smoke] FAIL: $*" >&2
    if [[ -f "$OBSERVATION_LOG" ]]; then
        tail -n 500 "$OBSERVATION_LOG" >&2 || true
    fi
    if [[ -f "$GRAFT_BOUNDARY_LOG" ]]; then
        tail -n 180 "$GRAFT_BOUNDARY_LOG" >&2 || true
    fi
    exit 1
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        die "$label missing in $file"
    fi
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_WORKDIR="$OBSERVATION_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_REPORT="$OBSERVATION_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_observation_smoke.sh" >"$OBSERVATION_LOG" 2>&1; then
    die "weighted admission resonance observation producer failed"
fi

[[ -s "$OBSERVATION_REPORT" ]] || die "weighted admission resonance observation report not written: $OBSERVATION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_boundary.sh" "$OBSERVATION_REPORT" "$GRAFT_BOUNDARY_REPORT" >"$GRAFT_BOUNDARY_LOG" 2>&1; then
    die "weighted admission resonance graft boundary rejected observation report"
fi

[[ -s "$GRAFT_BOUNDARY_REPORT" ]] || die "weighted admission resonance graft boundary report not written: $GRAFT_BOUNDARY_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary schema"
require_grep '"status": "shadow_graft_boundary_declared_dry_run"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary status"
require_grep '"target": "resonance"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary target kind"
require_grep '"target_mode": "receipt_only_closed_dry_run"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary target mode"
require_grep '"action": "declare_weighted_resonance_shadow_graft_boundary_dry_run"' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary action"
require_grep '"weighted_admission_resonance_graft_boundary_ready": true' "$GRAFT_BOUNDARY_REPORT" "graft-boundary ready flag"
require_grep '"weighted_admission_resonance_observation_consumed": true' "$GRAFT_BOUNDARY_REPORT" "observation consumed flag"
require_grep '"weighted_admission_resonance_observation_required": true' "$GRAFT_BOUNDARY_REPORT" "observation required flag"
require_grep '"next_step_blocked_without_resonance_graft_boundary": true' "$GRAFT_BOUNDARY_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_BOUNDARY_REPORT" "graft-boundary id"
require_grep '"receipt_shape": "weighted_resonance_observation_shadow_graft_boundary"' "$GRAFT_BOUNDARY_REPORT" "receipt shape"
require_grep '"boundary_kind": "shadow_graft_boundary"' "$GRAFT_BOUNDARY_REPORT" "boundary kind"
require_grep '"boundary_mode": "no_mutation_receipt"' "$GRAFT_BOUNDARY_REPORT" "boundary mode"
require_grep '"boundary_stage": "pre_live_graft"' "$GRAFT_BOUNDARY_REPORT" "boundary stage"
require_grep '"causal_id": "weighted-resonance-graft-boundary-causal-' "$GRAFT_BOUNDARY_REPORT" "causal id"
require_grep '"boundary_hash": "weighted-resonance-graft-boundary-' "$GRAFT_BOUNDARY_REPORT" "boundary hash"
require_grep '"read_back_hash": "weighted-resonance-graft-boundary-read-' "$GRAFT_BOUNDARY_REPORT" "read-back hash"
require_grep '"shadow_only": true' "$GRAFT_BOUNDARY_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_BOUNDARY_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_BOUNDARY_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_BOUNDARY_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_BOUNDARY_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_BOUNDARY_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_BOUNDARY_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_BOUNDARY_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_BOUNDARY_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_BOUNDARY_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_BOUNDARY_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_BOUNDARY_REPORT" "rollback requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_observation.v1"' "$GRAFT_BOUNDARY_REPORT" "source schema"
require_grep '"source_status": "observation_recorded_dry_run"' "$GRAFT_BOUNDARY_REPORT" "source status"
require_grep '"source_target": "resonance"' "$GRAFT_BOUNDARY_REPORT" "source target"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_BOUNDARY_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_observation_ready": true' "$GRAFT_BOUNDARY_REPORT" "source observation-ready flag"
require_grep '"source_weighted_admission_resonance_observation_causal_id": "weighted-resonance-observation-causal-' "$GRAFT_BOUNDARY_REPORT" "source observation causal id"
require_grep '"source_weighted_admission_resonance_observation_append_hash": "weighted-resonance-observation-append-' "$GRAFT_BOUNDARY_REPORT" "source observation append hash"
require_grep '"source_weighted_admission_resonance_observation_read_back_hash": "weighted-resonance-observation-read-' "$GRAFT_BOUNDARY_REPORT" "source observation read-back hash"
require_grep '"source_observer": "resonance"' "$GRAFT_BOUNDARY_REPORT" "source observer"
require_grep '"source_observer_kind": "internal_world"' "$GRAFT_BOUNDARY_REPORT" "source observer kind"
require_grep '"source_observation_kind": "weighted_receiver_state_proof"' "$GRAFT_BOUNDARY_REPORT" "source observation kind"
require_grep '"source_observation_mode": "sealed_metadata_observation"' "$GRAFT_BOUNDARY_REPORT" "source observation mode"
require_grep '"source_append_only": true' "$GRAFT_BOUNDARY_REPORT" "source append-only flag"
require_grep '"source_read_back": true' "$GRAFT_BOUNDARY_REPORT" "source read-back flag"
require_grep '"source_receipt_verified": true' "$GRAFT_BOUNDARY_REPORT" "source receipt flag"
require_grep '"source_dry_run_only": true' "$GRAFT_BOUNDARY_REPORT" "source dry-run flag"
require_grep '"source_observation_raw_dream_text_observed": false' "$GRAFT_BOUNDARY_REPORT" "source observation raw observe guard"
require_grep '"source_observation_raw_dream_text_forwarded": false' "$GRAFT_BOUNDARY_REPORT" "source observation raw forward guard"
require_grep '"source_observation_janus_surface_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source observation Janus guard"
require_grep '"source_observation_cooc_learning_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source observation cooc guard"
require_grep '"source_observation_delta_harvest_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source observation delta guard"
require_grep '"source_observation_body_mutation_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source observation body guard"
require_grep '"source_observation_rollback_required": true' "$GRAFT_BOUNDARY_REPORT" "source observation rollback"
require_grep '"source_resonance_receiver_report": "' "$GRAFT_BOUNDARY_REPORT" "source receiver report"
require_grep '"source_resonance_intent_report": "' "$GRAFT_BOUNDARY_REPORT" "source intent report"
require_grep '"source_final_gate_report": "' "$GRAFT_BOUNDARY_REPORT" "source final gate report"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_BOUNDARY_REPORT" "source receiver id"
require_grep '"source_weighted_admission_resonance_receiver_ready": true' "$GRAFT_BOUNDARY_REPORT" "source receiver-ready flag"
require_grep '"source_weighted_admission_resonance_receiver_causal_id": "weighted-resonance-receiver-causal-' "$GRAFT_BOUNDARY_REPORT" "source receiver causal id"
require_grep '"source_receiver_pre_state_hash": "weighted-resonance-receiver-pre-' "$GRAFT_BOUNDARY_REPORT" "source receiver pre-state hash"
require_grep '"source_receiver_post_state_hash": "weighted-resonance-receiver-post-' "$GRAFT_BOUNDARY_REPORT" "source receiver post-state hash"
require_grep '"source_receiver_state_delta_hash": "weighted-resonance-receiver-delta-' "$GRAFT_BOUNDARY_REPORT" "source receiver state-delta hash"
require_grep '"source_weighted_admission_resonance_intent_consumed": true' "$GRAFT_BOUNDARY_REPORT" "source intent consumed flag"
require_grep '"source_weighted_admission_resonance_intent_required": true' "$GRAFT_BOUNDARY_REPORT" "source intent required flag"
require_grep '"source_weighted_admission_resonance_intent_ready": true' "$GRAFT_BOUNDARY_REPORT" "source intent-ready flag"
require_grep '"source_weighted_admission_final_gate_consumed": true' "$GRAFT_BOUNDARY_REPORT" "source final gate consumed flag"
require_grep '"source_weighted_admission_final_gate_required": true' "$GRAFT_BOUNDARY_REPORT" "source final gate required flag"
require_grep '"source_weighted_admission_final_gate_ready": true' "$GRAFT_BOUNDARY_REPORT" "source final-gate-ready flag"
require_grep '"source_weighted_admission_seal_consumed": true' "$GRAFT_BOUNDARY_REPORT" "source seal consumed flag"
require_grep '"source_weighted_admission_seal_required": true' "$GRAFT_BOUNDARY_REPORT" "source seal required flag"
require_grep '"source_weighted_admission_seal_ready": true' "$GRAFT_BOUNDARY_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_consumed": true' "$GRAFT_BOUNDARY_REPORT" "source permit consumed flag"
require_grep '"source_weighted_admission_permit_required": true' "$GRAFT_BOUNDARY_REPORT" "source permit required flag"
require_grep '"source_weighted_admission_permit_ready": true' "$GRAFT_BOUNDARY_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$GRAFT_BOUNDARY_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$GRAFT_BOUNDARY_REPORT" "source authority required flag"
require_grep '"source_raw_dream_text_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source raw guard"
require_grep '"source_raw_dream_text_observed": false' "$GRAFT_BOUNDARY_REPORT" "source raw observe guard"
require_grep '"source_raw_dream_text_forwarded": false' "$GRAFT_BOUNDARY_REPORT" "source raw forward guard"
require_grep '"source_janus_surface_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source Janus guard"
require_grep '"source_cooc_learning_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source cooc guard"
require_grep '"source_delta_harvest_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source delta guard"
require_grep '"source_body_mutation_allowed": false' "$GRAFT_BOUNDARY_REPORT" "source body mutation guard"
require_grep '"source_rollback_required": true' "$GRAFT_BOUNDARY_REPORT" "source rollback requirement"
require_grep '"source_pre_state_hash_required": true' "$GRAFT_BOUNDARY_REPORT" "source pre-state requirement"
require_grep '"source_post_state_hash_required": true' "$GRAFT_BOUNDARY_REPORT" "source post-state requirement"
require_grep '"body_smoke_weighted": true' "$GRAFT_BOUNDARY_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_BOUNDARY_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_BOUNDARY_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_BOUNDARY_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_BOUNDARY_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_BOUNDARY_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_BOUNDARY_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_BOUNDARY_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_BOUNDARY_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_BOUNDARY_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_BOUNDARY_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_BOUNDARY_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_BOUNDARY_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_BOUNDARY_REPORT" "resonance-graft-boundary pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-boundary\] pass:' "$GRAFT_BOUNDARY_LOG" "resonance-graft-boundary pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-boundary-smoke] pass: resonance_observation_report=$OBSERVATION_REPORT resonance_graft_boundary_report=$GRAFT_BOUNDARY_REPORT"
