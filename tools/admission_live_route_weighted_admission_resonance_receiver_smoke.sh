#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_receiver_smoke.sh - preview weighted Resonance receiver after intent.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-receiver.XXXXXX")}"
INTENT_WORKDIR="$WORKDIR/intent"
INTENT_REPORT="$INTENT_WORKDIR/live_route_weighted_admission_resonance_intent.json"
RECEIVER_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_RECEIVER_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_receiver.json}"
INTENT_LOG="$WORKDIR/weighted_admission_resonance_intent.log"
RECEIVER_LOG="$WORKDIR/weighted_admission_resonance_receiver.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-receiver-smoke] FAIL: $*" >&2
    if [[ -f "$INTENT_LOG" ]]; then
        tail -n 500 "$INTENT_LOG" >&2 || true
    fi
    if [[ -f "$RECEIVER_LOG" ]]; then
        tail -n 180 "$RECEIVER_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_WORKDIR="$INTENT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_REPORT="$INTENT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_intent_smoke.sh" >"$INTENT_LOG" 2>&1; then
    die "weighted admission resonance intent producer failed"
fi

[[ -s "$INTENT_REPORT" ]] || die "weighted admission resonance intent report not written: $INTENT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_receiver.sh" "$INTENT_REPORT" "$RECEIVER_REPORT" >"$RECEIVER_LOG" 2>&1; then
    die "weighted admission resonance receiver writer rejected intent report"
fi

[[ -s "$RECEIVER_REPORT" ]] || die "weighted admission resonance receiver report not written: $RECEIVER_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_receiver.v1"' "$RECEIVER_REPORT" "resonance-receiver schema"
require_grep '"status": "receiver_previewed_dry_run"' "$RECEIVER_REPORT" "resonance-receiver status"
require_grep '"target": "resonance"' "$RECEIVER_REPORT" "resonance-receiver target"
require_grep '"target_kind": "weighted_live_route_first_receiver"' "$RECEIVER_REPORT" "resonance-receiver target kind"
require_grep '"target_mode": "bounded_direction_preview_dry_run"' "$RECEIVER_REPORT" "resonance-receiver target mode"
require_grep '"action": "preview_weighted_resonance_receive_dry_run"' "$RECEIVER_REPORT" "resonance-receiver action"
require_grep '"weighted_admission_resonance_receiver_ready": true' "$RECEIVER_REPORT" "resonance-receiver ready flag"
require_grep '"weighted_admission_resonance_intent_consumed": true' "$RECEIVER_REPORT" "intent consumed flag"
require_grep '"weighted_admission_resonance_intent_required": true' "$RECEIVER_REPORT" "intent required flag"
require_grep '"next_step_blocked_without_resonance_receiver": true' "$RECEIVER_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$RECEIVER_REPORT" "receiver id"
require_grep '"receiver": "resonance"' "$RECEIVER_REPORT" "receiver"
require_grep '"receiver_kind": "internal_world"' "$RECEIVER_REPORT" "receiver kind"
require_grep '"influence_kind": "bounded_direction"' "$RECEIVER_REPORT" "influence kind"
require_grep '"max_influence": 0.05' "$RECEIVER_REPORT" "influence cap"
require_grep '"ttl_turns": 1' "$RECEIVER_REPORT" "ttl"
require_grep '"causal_id": "weighted-resonance-receiver-causal-' "$RECEIVER_REPORT" "causal id"
require_grep '"pre_state_hash": "weighted-resonance-receiver-pre-' "$RECEIVER_REPORT" "pre-state hash"
require_grep '"post_state_hash": "weighted-resonance-receiver-post-' "$RECEIVER_REPORT" "post-state hash"
require_grep '"state_delta_hash": "weighted-resonance-receiver-delta-' "$RECEIVER_REPORT" "state-delta hash"
require_grep '"state_hash_mode": "sealed_metadata_preview"' "$RECEIVER_REPORT" "state hash mode"
require_grep '"dry_run_only": true' "$RECEIVER_REPORT" "dry-run flag"
require_grep '"raw_dream_text_observed": false' "$RECEIVER_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$RECEIVER_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$RECEIVER_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$RECEIVER_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$RECEIVER_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$RECEIVER_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$RECEIVER_REPORT" "rollback requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_intent.v1"' "$RECEIVER_REPORT" "source schema"
require_grep '"source_status": "resonance_intent_drafted_dry_run"' "$RECEIVER_REPORT" "source status"
require_grep '"source_target": "resonance"' "$RECEIVER_REPORT" "source target"
require_grep '"source_weighted_admission_resonance_intent_ready": true' "$RECEIVER_REPORT" "source intent-ready flag"
require_grep '"source_weighted_admission_final_gate_consumed": true' "$RECEIVER_REPORT" "source final gate consumed flag"
require_grep '"source_weighted_admission_final_gate_required": true' "$RECEIVER_REPORT" "source final gate required flag"
require_grep '"source_weighted_admission_final_gate_ready": true' "$RECEIVER_REPORT" "source final-gate-ready flag"
require_grep '"source_weighted_admission_seal_consumed": true' "$RECEIVER_REPORT" "source seal consumed flag"
require_grep '"source_weighted_admission_seal_required": true' "$RECEIVER_REPORT" "source seal required flag"
require_grep '"source_weighted_admission_seal_ready": true' "$RECEIVER_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_consumed": true' "$RECEIVER_REPORT" "source permit consumed flag"
require_grep '"source_weighted_admission_permit_required": true' "$RECEIVER_REPORT" "source permit required flag"
require_grep '"source_weighted_admission_permit_ready": true' "$RECEIVER_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$RECEIVER_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$RECEIVER_REPORT" "source authority required flag"
require_grep '"source_raw_dream_text_allowed": false' "$RECEIVER_REPORT" "source raw guard"
require_grep '"source_janus_surface_allowed": false' "$RECEIVER_REPORT" "source Janus guard"
require_grep '"source_cooc_learning_allowed": false' "$RECEIVER_REPORT" "source cooc guard"
require_grep '"source_delta_harvest_allowed": false' "$RECEIVER_REPORT" "source delta guard"
require_grep '"source_rollback_required": true' "$RECEIVER_REPORT" "source rollback requirement"
require_grep '"source_pre_state_hash_required": true' "$RECEIVER_REPORT" "source pre-state requirement"
require_grep '"source_post_state_hash_required": true' "$RECEIVER_REPORT" "source post-state requirement"
require_grep '"body_smoke_weighted": true' "$RECEIVER_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$RECEIVER_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$RECEIVER_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$RECEIVER_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$RECEIVER_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$RECEIVER_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$RECEIVER_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$RECEIVER_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$RECEIVER_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$RECEIVER_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$RECEIVER_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$RECEIVER_REPORT" "non-mutation flag"
require_grep '"passed": true' "$RECEIVER_REPORT" "resonance-receiver pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-receiver\] pass:' "$RECEIVER_LOG" "resonance-receiver pass line"

echo "[admission-live-route-weighted-admission-resonance-receiver-smoke] pass: resonance_intent_report=$INTENT_REPORT resonance_receiver_report=$RECEIVER_REPORT"
