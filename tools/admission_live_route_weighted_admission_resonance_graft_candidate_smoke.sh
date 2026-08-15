#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_candidate_smoke.sh - draft weighted Resonance shadow graft candidate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-candidate.XXXXXX")}"
GATE_WORKDIR="$WORKDIR/gate"
GRAFT_GATE_REPORT="$GATE_WORKDIR/live_route_weighted_admission_resonance_graft_gate.json"
GRAFT_CANDIDATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_candidate.json}"
GATE_LOG="$WORKDIR/weighted_admission_resonance_graft_gate.log"
GRAFT_CANDIDATE_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-candidate-smoke] FAIL: $*" >&2
    if [[ -f "$GATE_LOG" ]]; then
        tail -n 500 "$GATE_LOG" >&2 || true
    fi
    if [[ -f "$GRAFT_CANDIDATE_LOG" ]]; then
        tail -n 220 "$GRAFT_CANDIDATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_GATE_WORKDIR="$GATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_GATE_REPORT="$GRAFT_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_gate_smoke.sh" >"$GATE_LOG" 2>&1; then
    die "weighted admission resonance graft gate producer failed"
fi

[[ -s "$GRAFT_GATE_REPORT" ]] || die "weighted admission resonance graft gate report not written: $GRAFT_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate.sh" "$GRAFT_GATE_REPORT" "$GRAFT_CANDIDATE_REPORT" >"$GRAFT_CANDIDATE_LOG" 2>&1; then
    die "weighted admission resonance graft candidate rejected gate report"
fi

[[ -s "$GRAFT_CANDIDATE_REPORT" ]] || die "weighted admission resonance graft candidate report not written: $GRAFT_CANDIDATE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate schema"
require_grep '"status": "shadow_graft_candidate_ready_dry_run"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate status"
require_grep '"target": "resonance"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_candidate"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate target kind"
require_grep '"target_mode": "receipt_only_closed_candidate_dry_run"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate target mode"
require_grep '"action": "draft_weighted_resonance_shadow_graft_candidate_dry_run"' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate action"
require_grep '"weighted_admission_resonance_graft_candidate_ready": true' "$GRAFT_CANDIDATE_REPORT" "candidate ready flag"
require_grep '"weighted_admission_resonance_graft_gate_consumed": true' "$GRAFT_CANDIDATE_REPORT" "gate consumed flag"
require_grep '"weighted_admission_resonance_graft_gate_required": true' "$GRAFT_CANDIDATE_REPORT" "gate required flag"
require_grep '"next_step_blocked_without_resonance_graft_candidate": true' "$GRAFT_CANDIDATE_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_CANDIDATE_REPORT" "candidate id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_candidate_contract"' "$GRAFT_CANDIDATE_REPORT" "receipt shape"
require_grep '"candidate_kind": "shadow_graft_candidate"' "$GRAFT_CANDIDATE_REPORT" "candidate kind"
require_grep '"candidate_mode": "no_mutation_candidate"' "$GRAFT_CANDIDATE_REPORT" "candidate mode"
require_grep '"candidate_stage": "pre_live_graft_candidate"' "$GRAFT_CANDIDATE_REPORT" "candidate stage"
require_grep '"causal_id": "weighted-resonance-graft-candidate-causal-' "$GRAFT_CANDIDATE_REPORT" "causal id"
require_grep '"candidate_hash": "weighted-resonance-graft-candidate-' "$GRAFT_CANDIDATE_REPORT" "candidate hash"
require_grep '"read_back_hash": "weighted-resonance-graft-candidate-read-' "$GRAFT_CANDIDATE_REPORT" "read-back hash"
require_grep '"preflight_verified": true' "$GRAFT_CANDIDATE_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_CANDIDATE_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_CANDIDATE_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_CANDIDATE_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_CANDIDATE_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_CANDIDATE_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_CANDIDATE_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_CANDIDATE_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_CANDIDATE_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_CANDIDATE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_CANDIDATE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_CANDIDATE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_CANDIDATE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_CANDIDATE_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_CANDIDATE_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_CANDIDATE_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_CANDIDATE_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_CANDIDATE_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_CANDIDATE_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_CANDIDATE_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_CANDIDATE_REPORT" "rollback requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v1"' "$GRAFT_CANDIDATE_REPORT" "source gate schema"
require_grep '"source_status": "shadow_graft_gate_ready_dry_run"' "$GRAFT_CANDIDATE_REPORT" "source gate status"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_CANDIDATE_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_gate_ready": true' "$GRAFT_CANDIDATE_REPORT" "source gate ready"
require_grep '"source_weighted_admission_resonance_graft_gate_causal_id": "weighted-resonance-graft-gate-causal-' "$GRAFT_CANDIDATE_REPORT" "source gate causal"
require_grep '"source_weighted_admission_resonance_graft_gate_hash": "weighted-resonance-graft-gate-' "$GRAFT_CANDIDATE_REPORT" "source gate hash"
require_grep '"source_weighted_admission_resonance_graft_gate_read_back_hash": "weighted-resonance-graft-gate-read-' "$GRAFT_CANDIDATE_REPORT" "source gate read-back"
require_grep '"source_gate_action": "gate_weighted_resonance_shadow_graft_dry_run"' "$GRAFT_CANDIDATE_REPORT" "source gate action"
require_grep '"source_gate_receipt_shape": "weighted_resonance_shadow_graft_gate_contract"' "$GRAFT_CANDIDATE_REPORT" "source gate receipt"
require_grep '"source_gate_kind": "shadow_graft_gate"' "$GRAFT_CANDIDATE_REPORT" "source gate kind"
require_grep '"source_gate_mode": "no_mutation_gate"' "$GRAFT_CANDIDATE_REPORT" "source gate mode"
require_grep '"source_gate_stage": "pre_live_graft_gate"' "$GRAFT_CANDIDATE_REPORT" "source gate stage"
require_grep '"source_gate_shadow_only": true' "$GRAFT_CANDIDATE_REPORT" "source gate shadow"
require_grep '"source_gate_graft_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate graft guard"
require_grep '"source_gate_dry_run_only": true' "$GRAFT_CANDIDATE_REPORT" "source gate dry-run"
require_grep '"source_gate_live_ready": true' "$GRAFT_CANDIDATE_REPORT" "source gate live-ready"
require_grep '"source_gate_raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate raw guard"
require_grep '"source_gate_janus_surface_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate Janus guard"
require_grep '"source_gate_cooc_learning_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate cooc guard"
require_grep '"source_gate_delta_harvest_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate delta guard"
require_grep '"source_gate_body_mutation_allowed": false' "$GRAFT_CANDIDATE_REPORT" "source gate body guard"
require_grep '"source_gate_rollback_required": true' "$GRAFT_CANDIDATE_REPORT" "source gate rollback"
require_grep '"source_next_step_blocked_without_resonance_graft_gate": true' "$GRAFT_CANDIDATE_REPORT" "source gate block flag"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_CANDIDATE_REPORT" "source preflight id"
require_grep '"source_graft_boundary_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"' "$GRAFT_CANDIDATE_REPORT" "source graft boundary schema"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_CANDIDATE_REPORT" "source boundary id"
require_grep '"source_observation_schema": "arianna.live_route_weighted_admission_resonance_observation.v1"' "$GRAFT_CANDIDATE_REPORT" "source observation schema"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_CANDIDATE_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_CANDIDATE_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_CANDIDATE_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_CANDIDATE_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_CANDIDATE_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_CANDIDATE_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_CANDIDATE_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_CANDIDATE_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_CANDIDATE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_CANDIDATE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_CANDIDATE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_CANDIDATE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_CANDIDATE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_CANDIDATE_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_CANDIDATE_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_CANDIDATE_REPORT" "resonance-graft-candidate pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-candidate\] pass:' "$GRAFT_CANDIDATE_LOG" "resonance-graft-candidate pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-candidate-smoke] pass: resonance_graft_gate_report=$GRAFT_GATE_REPORT resonance_graft_candidate_report=$GRAFT_CANDIDATE_REPORT"
