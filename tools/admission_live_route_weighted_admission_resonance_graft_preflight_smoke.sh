#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_preflight_smoke.sh - prepare weighted Resonance shadow graft preflight.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-preflight.XXXXXX")}"
BOUNDARY_WORKDIR="$WORKDIR/boundary"
GRAFT_BOUNDARY_REPORT="$BOUNDARY_WORKDIR/live_route_weighted_admission_resonance_graft_boundary.json"
GRAFT_PREFLIGHT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_PREFLIGHT_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_preflight.json}"
BOUNDARY_LOG="$WORKDIR/weighted_admission_resonance_graft_boundary.log"
GRAFT_PREFLIGHT_LOG="$WORKDIR/weighted_admission_resonance_graft_preflight.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-preflight-smoke] FAIL: $*" >&2
    if [[ -f "$BOUNDARY_LOG" ]]; then
        tail -n 500 "$BOUNDARY_LOG" >&2 || true
    fi
    if [[ -f "$GRAFT_PREFLIGHT_LOG" ]]; then
        tail -n 220 "$GRAFT_PREFLIGHT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_WORKDIR="$BOUNDARY_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_BOUNDARY_REPORT="$GRAFT_BOUNDARY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_boundary_smoke.sh" >"$BOUNDARY_LOG" 2>&1; then
    die "weighted admission resonance graft boundary producer failed"
fi

[[ -s "$GRAFT_BOUNDARY_REPORT" ]] || die "weighted admission resonance graft boundary report not written: $GRAFT_BOUNDARY_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_preflight.sh" "$GRAFT_BOUNDARY_REPORT" "$GRAFT_PREFLIGHT_REPORT" >"$GRAFT_PREFLIGHT_LOG" 2>&1; then
    die "weighted admission resonance graft preflight rejected boundary report"
fi

[[ -s "$GRAFT_PREFLIGHT_REPORT" ]] || die "weighted admission resonance graft preflight report not written: $GRAFT_PREFLIGHT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v1"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight schema"
require_grep '"status": "shadow_graft_preflight_ready_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight status"
require_grep '"target": "resonance"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_preflight"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight target kind"
require_grep '"target_mode": "receipt_only_closed_preflight_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight target mode"
require_grep '"action": "prepare_weighted_resonance_shadow_graft_preflight_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight action"
require_grep '"weighted_admission_resonance_graft_preflight_ready": true' "$GRAFT_PREFLIGHT_REPORT" "graft-preflight ready flag"
require_grep '"weighted_admission_resonance_graft_boundary_consumed": true' "$GRAFT_PREFLIGHT_REPORT" "graft-boundary consumed flag"
require_grep '"weighted_admission_resonance_graft_boundary_required": true' "$GRAFT_PREFLIGHT_REPORT" "graft-boundary required flag"
require_grep '"next_step_blocked_without_resonance_graft_preflight": true' "$GRAFT_PREFLIGHT_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_PREFLIGHT_REPORT" "graft-preflight id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_preflight_contract"' "$GRAFT_PREFLIGHT_REPORT" "receipt shape"
require_grep '"preflight_kind": "shadow_graft_preflight"' "$GRAFT_PREFLIGHT_REPORT" "preflight kind"
require_grep '"preflight_mode": "no_mutation_preflight"' "$GRAFT_PREFLIGHT_REPORT" "preflight mode"
require_grep '"preflight_stage": "pre_live_graft_admission"' "$GRAFT_PREFLIGHT_REPORT" "preflight stage"
require_grep '"causal_id": "weighted-resonance-graft-preflight-causal-' "$GRAFT_PREFLIGHT_REPORT" "causal id"
require_grep '"preflight_hash": "weighted-resonance-graft-preflight-' "$GRAFT_PREFLIGHT_REPORT" "preflight hash"
require_grep '"read_back_hash": "weighted-resonance-graft-preflight-read-' "$GRAFT_PREFLIGHT_REPORT" "read-back hash"
require_grep '"boundary_verified": true' "$GRAFT_PREFLIGHT_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_PREFLIGHT_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_PREFLIGHT_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_PREFLIGHT_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_PREFLIGHT_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_PREFLIGHT_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_PREFLIGHT_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_PREFLIGHT_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_PREFLIGHT_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_PREFLIGHT_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_PREFLIGHT_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_PREFLIGHT_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_PREFLIGHT_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_PREFLIGHT_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_PREFLIGHT_REPORT" "rollback requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"' "$GRAFT_PREFLIGHT_REPORT" "source boundary schema"
require_grep '"source_status": "shadow_graft_boundary_declared_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "source boundary status"
require_grep '"source_target": "resonance"' "$GRAFT_PREFLIGHT_REPORT" "source boundary target"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_PREFLIGHT_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_graft_boundary_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source boundary ready"
require_grep '"source_weighted_admission_resonance_graft_boundary_causal_id": "weighted-resonance-graft-boundary-causal-' "$GRAFT_PREFLIGHT_REPORT" "source boundary causal"
require_grep '"source_weighted_admission_resonance_graft_boundary_hash": "weighted-resonance-graft-boundary-' "$GRAFT_PREFLIGHT_REPORT" "source boundary hash"
require_grep '"source_weighted_admission_resonance_graft_boundary_read_back_hash": "weighted-resonance-graft-boundary-read-' "$GRAFT_PREFLIGHT_REPORT" "source boundary read-back"
require_grep '"source_boundary_action": "declare_weighted_resonance_shadow_graft_boundary_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "source boundary action"
require_grep '"source_boundary_receipt_shape": "weighted_resonance_observation_shadow_graft_boundary"' "$GRAFT_PREFLIGHT_REPORT" "source boundary receipt"
require_grep '"source_boundary_kind": "shadow_graft_boundary"' "$GRAFT_PREFLIGHT_REPORT" "source boundary kind"
require_grep '"source_boundary_mode": "no_mutation_receipt"' "$GRAFT_PREFLIGHT_REPORT" "source boundary mode"
require_grep '"source_boundary_stage": "pre_live_graft"' "$GRAFT_PREFLIGHT_REPORT" "source boundary stage"
require_grep '"source_boundary_shadow_only": true' "$GRAFT_PREFLIGHT_REPORT" "source boundary shadow"
require_grep '"source_boundary_graft_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary graft guard"
require_grep '"source_boundary_dry_run_only": true' "$GRAFT_PREFLIGHT_REPORT" "source boundary dry-run"
require_grep '"source_boundary_live_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source boundary live-ready"
require_grep '"source_boundary_raw_dream_text_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary raw guard"
require_grep '"source_boundary_janus_surface_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary Janus guard"
require_grep '"source_boundary_cooc_learning_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary cooc guard"
require_grep '"source_boundary_delta_harvest_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary delta guard"
require_grep '"source_boundary_body_mutation_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source boundary body mutation guard"
require_grep '"source_boundary_rollback_required": true' "$GRAFT_PREFLIGHT_REPORT" "source boundary rollback"
require_grep '"source_observation_schema": "arianna.live_route_weighted_admission_resonance_observation.v1"' "$GRAFT_PREFLIGHT_REPORT" "source observation schema"
require_grep '"source_observation_status": "observation_recorded_dry_run"' "$GRAFT_PREFLIGHT_REPORT" "source observation status"
require_grep '"source_observation_target": "resonance"' "$GRAFT_PREFLIGHT_REPORT" "source observation target"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_PREFLIGHT_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_observation_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source observation-ready flag"
require_grep '"source_weighted_admission_resonance_observation_causal_id": "weighted-resonance-observation-causal-' "$GRAFT_PREFLIGHT_REPORT" "source observation causal id"
require_grep '"source_weighted_admission_resonance_observation_append_hash": "weighted-resonance-observation-append-' "$GRAFT_PREFLIGHT_REPORT" "source observation append hash"
require_grep '"source_weighted_admission_resonance_observation_read_back_hash": "weighted-resonance-observation-read-' "$GRAFT_PREFLIGHT_REPORT" "source observation read-back hash"
require_grep '"source_observer": "resonance"' "$GRAFT_PREFLIGHT_REPORT" "source observer"
require_grep '"source_observer_kind": "internal_world"' "$GRAFT_PREFLIGHT_REPORT" "source observer kind"
require_grep '"source_observation_kind": "weighted_receiver_state_proof"' "$GRAFT_PREFLIGHT_REPORT" "source observation kind"
require_grep '"source_observation_mode": "sealed_metadata_observation"' "$GRAFT_PREFLIGHT_REPORT" "source observation mode"
require_grep '"source_append_only": true' "$GRAFT_PREFLIGHT_REPORT" "source append-only flag"
require_grep '"source_read_back": true' "$GRAFT_PREFLIGHT_REPORT" "source read-back flag"
require_grep '"source_receipt_verified": true' "$GRAFT_PREFLIGHT_REPORT" "source receipt flag"
require_grep '"source_dry_run_only": true' "$GRAFT_PREFLIGHT_REPORT" "source dry-run flag"
require_grep '"source_observation_raw_dream_text_observed": false' "$GRAFT_PREFLIGHT_REPORT" "source observation raw observe guard"
require_grep '"source_observation_body_mutation_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source observation body guard"
require_grep '"source_observation_rollback_required": true' "$GRAFT_PREFLIGHT_REPORT" "source observation rollback"
require_grep '"source_resonance_receiver_report": "' "$GRAFT_PREFLIGHT_REPORT" "source receiver report"
require_grep '"source_resonance_intent_report": "' "$GRAFT_PREFLIGHT_REPORT" "source intent report"
require_grep '"source_final_gate_report": "' "$GRAFT_PREFLIGHT_REPORT" "source final gate report"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_PREFLIGHT_REPORT" "source receiver id"
require_grep '"source_weighted_admission_resonance_receiver_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source receiver-ready flag"
require_grep '"source_weighted_admission_resonance_receiver_causal_id": "weighted-resonance-receiver-causal-' "$GRAFT_PREFLIGHT_REPORT" "source receiver causal id"
require_grep '"source_receiver_pre_state_hash": "weighted-resonance-receiver-pre-' "$GRAFT_PREFLIGHT_REPORT" "source receiver pre-state hash"
require_grep '"source_receiver_post_state_hash": "weighted-resonance-receiver-post-' "$GRAFT_PREFLIGHT_REPORT" "source receiver post-state hash"
require_grep '"source_receiver_state_delta_hash": "weighted-resonance-receiver-delta-' "$GRAFT_PREFLIGHT_REPORT" "source receiver state-delta hash"
require_grep '"source_weighted_admission_resonance_intent_consumed": true' "$GRAFT_PREFLIGHT_REPORT" "source intent consumed flag"
require_grep '"source_weighted_admission_resonance_intent_required": true' "$GRAFT_PREFLIGHT_REPORT" "source intent required flag"
require_grep '"source_weighted_admission_resonance_intent_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source intent-ready flag"
require_grep '"source_weighted_admission_final_gate_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source final-gate-ready flag"
require_grep '"source_weighted_admission_seal_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_ready": true' "$GRAFT_PREFLIGHT_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$GRAFT_PREFLIGHT_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$GRAFT_PREFLIGHT_REPORT" "source authority required flag"
require_grep '"source_raw_dream_text_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source raw guard"
require_grep '"source_raw_dream_text_observed": false' "$GRAFT_PREFLIGHT_REPORT" "source raw observe guard"
require_grep '"source_raw_dream_text_forwarded": false' "$GRAFT_PREFLIGHT_REPORT" "source raw forward guard"
require_grep '"source_janus_surface_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source Janus guard"
require_grep '"source_cooc_learning_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source cooc guard"
require_grep '"source_delta_harvest_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source delta guard"
require_grep '"source_body_mutation_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "source body mutation guard"
require_grep '"source_rollback_required": true' "$GRAFT_PREFLIGHT_REPORT" "source rollback requirement"
require_grep '"source_pre_state_hash_required": true' "$GRAFT_PREFLIGHT_REPORT" "source pre-state requirement"
require_grep '"source_post_state_hash_required": true' "$GRAFT_PREFLIGHT_REPORT" "source post-state requirement"
require_grep '"body_smoke_weighted": true' "$GRAFT_PREFLIGHT_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_PREFLIGHT_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_PREFLIGHT_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_PREFLIGHT_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_PREFLIGHT_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_PREFLIGHT_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_PREFLIGHT_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_PREFLIGHT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_PREFLIGHT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_PREFLIGHT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_PREFLIGHT_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_PREFLIGHT_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_PREFLIGHT_REPORT" "resonance-graft-preflight pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-preflight\] pass:' "$GRAFT_PREFLIGHT_LOG" "resonance-graft-preflight pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-preflight-smoke] pass: resonance_graft_boundary_report=$GRAFT_BOUNDARY_REPORT resonance_graft_preflight_report=$GRAFT_PREFLIGHT_REPORT"
