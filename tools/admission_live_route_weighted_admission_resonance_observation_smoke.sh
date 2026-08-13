#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_observation_smoke.sh - record weighted Resonance receiver observation.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-observation.XXXXXX")}"
RECEIVER_WORKDIR="$WORKDIR/receiver"
RECEIVER_REPORT="$RECEIVER_WORKDIR/live_route_weighted_admission_resonance_receiver.json"
OBSERVATION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_OBSERVATION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_observation.json}"
RECEIVER_LOG="$WORKDIR/weighted_admission_resonance_receiver.log"
OBSERVATION_LOG="$WORKDIR/weighted_admission_resonance_observation.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-observation-smoke] FAIL: $*" >&2
    if [[ -f "$RECEIVER_LOG" ]]; then
        tail -n 500 "$RECEIVER_LOG" >&2 || true
    fi
    if [[ -f "$OBSERVATION_LOG" ]]; then
        tail -n 180 "$OBSERVATION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_WORKDIR="$RECEIVER_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_REPORT="$RECEIVER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_receiver_smoke.sh" >"$RECEIVER_LOG" 2>&1; then
    die "weighted admission resonance receiver producer failed"
fi

[[ -s "$RECEIVER_REPORT" ]] || die "weighted admission resonance receiver report not written: $RECEIVER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_observation.sh" "$RECEIVER_REPORT" "$OBSERVATION_REPORT" >"$OBSERVATION_LOG" 2>&1; then
    die "weighted admission resonance observation writer rejected receiver report"
fi

[[ -s "$OBSERVATION_REPORT" ]] || die "weighted admission resonance observation report not written: $OBSERVATION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_observation.v1"' "$OBSERVATION_REPORT" "resonance-observation schema"
require_grep '"status": "observation_recorded_dry_run"' "$OBSERVATION_REPORT" "resonance-observation status"
require_grep '"target": "resonance"' "$OBSERVATION_REPORT" "resonance-observation target"
require_grep '"target_kind": "weighted_internal_world_observation"' "$OBSERVATION_REPORT" "resonance-observation target kind"
require_grep '"target_mode": "append_only_read_back_dry_run"' "$OBSERVATION_REPORT" "resonance-observation target mode"
require_grep '"action": "record_weighted_resonance_receiver_observation_dry_run"' "$OBSERVATION_REPORT" "resonance-observation action"
require_grep '"weighted_admission_resonance_observation_ready": true' "$OBSERVATION_REPORT" "resonance-observation ready flag"
require_grep '"weighted_admission_resonance_receiver_consumed": true' "$OBSERVATION_REPORT" "receiver consumed flag"
require_grep '"weighted_admission_resonance_receiver_required": true' "$OBSERVATION_REPORT" "receiver required flag"
require_grep '"next_step_blocked_without_resonance_observation": true' "$OBSERVATION_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$OBSERVATION_REPORT" "observation id"
require_grep '"observer": "resonance"' "$OBSERVATION_REPORT" "observer"
require_grep '"observer_kind": "internal_world"' "$OBSERVATION_REPORT" "observer kind"
require_grep '"observation_kind": "weighted_receiver_state_proof"' "$OBSERVATION_REPORT" "observation kind"
require_grep '"observation_mode": "sealed_metadata_observation"' "$OBSERVATION_REPORT" "observation mode"
require_grep '"causal_id": "weighted-resonance-observation-causal-' "$OBSERVATION_REPORT" "causal id"
require_grep '"append_hash": "weighted-resonance-observation-append-' "$OBSERVATION_REPORT" "append hash"
require_grep '"read_back_hash": "weighted-resonance-observation-read-' "$OBSERVATION_REPORT" "read-back hash"
require_grep '"append_only": true' "$OBSERVATION_REPORT" "append-only flag"
require_grep '"read_back": true' "$OBSERVATION_REPORT" "read-back flag"
require_grep '"receipt_verified": true' "$OBSERVATION_REPORT" "receipt verification flag"
require_grep '"dry_run_only": true' "$OBSERVATION_REPORT" "dry-run flag"
require_grep '"raw_dream_text_observed": false' "$OBSERVATION_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$OBSERVATION_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$OBSERVATION_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$OBSERVATION_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$OBSERVATION_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$OBSERVATION_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$OBSERVATION_REPORT" "rollback requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_receiver.v1"' "$OBSERVATION_REPORT" "source schema"
require_grep '"source_status": "receiver_previewed_dry_run"' "$OBSERVATION_REPORT" "source status"
require_grep '"source_target": "resonance"' "$OBSERVATION_REPORT" "source target"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$OBSERVATION_REPORT" "source receiver id"
require_grep '"source_weighted_admission_resonance_receiver_ready": true' "$OBSERVATION_REPORT" "source receiver-ready flag"
require_grep '"source_weighted_admission_resonance_receiver_causal_id": "weighted-resonance-receiver-causal-' "$OBSERVATION_REPORT" "source receiver causal id"
require_grep '"source_receiver_pre_state_hash": "weighted-resonance-receiver-pre-' "$OBSERVATION_REPORT" "source receiver pre-state hash"
require_grep '"source_receiver_post_state_hash": "weighted-resonance-receiver-post-' "$OBSERVATION_REPORT" "source receiver post-state hash"
require_grep '"source_receiver_state_delta_hash": "weighted-resonance-receiver-delta-' "$OBSERVATION_REPORT" "source receiver state-delta hash"
require_grep '"source_weighted_admission_resonance_intent_consumed": true' "$OBSERVATION_REPORT" "source intent consumed flag"
require_grep '"source_weighted_admission_resonance_intent_required": true' "$OBSERVATION_REPORT" "source intent required flag"
require_grep '"source_weighted_admission_resonance_intent_ready": true' "$OBSERVATION_REPORT" "source intent-ready flag"
require_grep '"source_weighted_admission_final_gate_consumed": true' "$OBSERVATION_REPORT" "source final gate consumed flag"
require_grep '"source_weighted_admission_final_gate_required": true' "$OBSERVATION_REPORT" "source final gate required flag"
require_grep '"source_weighted_admission_final_gate_ready": true' "$OBSERVATION_REPORT" "source final-gate-ready flag"
require_grep '"source_weighted_admission_seal_consumed": true' "$OBSERVATION_REPORT" "source seal consumed flag"
require_grep '"source_weighted_admission_seal_required": true' "$OBSERVATION_REPORT" "source seal required flag"
require_grep '"source_weighted_admission_seal_ready": true' "$OBSERVATION_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_consumed": true' "$OBSERVATION_REPORT" "source permit consumed flag"
require_grep '"source_weighted_admission_permit_required": true' "$OBSERVATION_REPORT" "source permit required flag"
require_grep '"source_weighted_admission_permit_ready": true' "$OBSERVATION_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$OBSERVATION_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$OBSERVATION_REPORT" "source authority required flag"
require_grep '"source_raw_dream_text_allowed": false' "$OBSERVATION_REPORT" "source raw guard"
require_grep '"source_raw_dream_text_observed": false' "$OBSERVATION_REPORT" "source raw observe guard"
require_grep '"source_raw_dream_text_forwarded": false' "$OBSERVATION_REPORT" "source raw forward guard"
require_grep '"source_janus_surface_allowed": false' "$OBSERVATION_REPORT" "source Janus guard"
require_grep '"source_cooc_learning_allowed": false' "$OBSERVATION_REPORT" "source cooc guard"
require_grep '"source_delta_harvest_allowed": false' "$OBSERVATION_REPORT" "source delta guard"
require_grep '"source_body_mutation_allowed": false' "$OBSERVATION_REPORT" "source body mutation guard"
require_grep '"source_rollback_required": true' "$OBSERVATION_REPORT" "source rollback requirement"
require_grep '"source_pre_state_hash_required": true' "$OBSERVATION_REPORT" "source pre-state requirement"
require_grep '"source_post_state_hash_required": true' "$OBSERVATION_REPORT" "source post-state requirement"
require_grep '"body_smoke_weighted": true' "$OBSERVATION_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$OBSERVATION_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$OBSERVATION_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$OBSERVATION_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$OBSERVATION_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$OBSERVATION_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$OBSERVATION_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$OBSERVATION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$OBSERVATION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$OBSERVATION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$OBSERVATION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$OBSERVATION_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$OBSERVATION_REPORT" "body target"
require_grep '"passed": true' "$OBSERVATION_REPORT" "resonance-observation pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-observation\] pass:' "$OBSERVATION_LOG" "resonance-observation pass line"

echo "[admission-live-route-weighted-admission-resonance-observation-smoke] pass: resonance_receiver_report=$RECEIVER_REPORT resonance_observation_report=$OBSERVATION_REPORT"
