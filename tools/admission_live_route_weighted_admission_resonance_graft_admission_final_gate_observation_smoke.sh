#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_smoke.sh - record final-gate observation from compact weighted graft admission final-gate receiver.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation.XXXXXX")}"
RECEIVER_WORKDIR="$WORKDIR/final_gate_receiver"
GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.json"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation.json}"
RECEIVER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_receiver.log"
OBSERVATION_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-smoke] FAIL: $*" >&2
    if [[ -f "$RECEIVER_LOG" ]]; then
        tail -n 500 "$RECEIVER_LOG" >&2 || true
    fi
    if [[ -f "$OBSERVATION_LOG" ]]; then
        tail -n 260 "$OBSERVATION_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_RECEIVER_WORKDIR="$RECEIVER_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT="$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_receiver_smoke.sh" >"$RECEIVER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate receiver producer failed"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" ]] || die "weighted admission resonance graft admission final gate receiver report not written: $GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation.sh" "$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" >"$OBSERVATION_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation rejected receiver report"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" ]] || die "weighted admission resonance graft admission final gate observation report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_recorded_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation target kind"
require_grep '"target_mode": "append_only_read_back_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation target mode"
require_grep '"action": "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation action"
require_grep '"writer_action": "reject_blocked_admission_final_gate_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_final_gate_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_final_gate_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger action"
require_grep '"ledger_contract": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger contract"
require_grep '"ledger_entrypoint": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger entrypoint"
require_grep '"ledger_receipt_shape": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger receipt shape"
require_grep '"ledger_write_scope": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "ledger write scope"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "receipt shape"
require_grep '"admission_final_gate_observation_state": "recorded"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation state"
require_grep '"admission_final_gate_observation_action": "record_blocked_final_gate_receiver_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation action field"
require_grep '"admission_final_gate_observation_target": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation target field"
require_grep '"admission_final_gate_observation_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation dry-run"
require_grep '"admission_final_gate_observation_append_only": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "append-only"
require_grep '"admission_final_gate_observation_read_back": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "read-back"
require_grep '"admission_final_gate_observation_receipt_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "receipt verified"
require_grep '"admission_final_gate_observation_receiver_verified": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "receiver verified guard"
require_grep '"admission_final_gate_observation_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation ready guard"
require_grep '"final_gate_observation_observer": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observer"
require_grep '"final_gate_observation_observer_kind": "internal_world"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observer kind"
require_grep '"final_gate_observation_kind": "blocked_final_gate_receiver_state_proof"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation kind"
require_grep '"final_gate_observation_mode": "sealed_metadata_observation"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation mode"
require_grep '"final_gate_observation_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "raw observed guard"
require_grep '"final_gate_observation_raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "raw forwarded guard"
require_grep '"final_gate_observation_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "raw allowed guard"
require_grep '"final_gate_observation_janus_surface_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "Janus guard"
require_grep '"final_gate_observation_cooc_learning_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "cooc guard"
require_grep '"final_gate_observation_delta_harvest_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "delta guard"
require_grep '"final_gate_observation_body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "body mutation guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "weighted observation ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_receiver_consumed": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "receiver consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_receiver_required": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "receiver required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_id": "weighted-resonance-graft-admission-final-gate-observation-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-causal-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation causal"
require_grep '"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-append-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "append hash"
require_grep '"admission_final_gate_observation_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-read-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_receiver.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_receiver_previewed_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver status"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id": "weighted-resonance-graft-admission-final-gate-receiver-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver ready"
require_grep '"source_admission_final_gate_receiver_pre_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-pre-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source pre hash"
require_grep '"source_admission_final_gate_receiver_post_state_hash": "weighted-resonance-graft-admission-final-gate-receiver-post-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source post hash"
require_grep '"source_admission_final_gate_receiver_state_delta_hash": "weighted-resonance-graft-admission-final-gate-receiver-delta-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source delta hash"
require_grep '"source_admission_final_gate_receiver_state": "previewed"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver state"
require_grep '"source_admission_final_gate_receiver_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver ready guard"
require_grep '"source_final_gate_receiver": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source receiver"
require_grep '"source_final_gate_receiver_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "source raw observed guard"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation recorded from blocked receiver; live admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT" "observation reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation\] pass:' "$OBSERVATION_LOG" "observation pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-smoke] pass: resonance_graft_admission_final_gate_receiver_report=$GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT resonance_graft_admission_final_gate_observation_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT"
