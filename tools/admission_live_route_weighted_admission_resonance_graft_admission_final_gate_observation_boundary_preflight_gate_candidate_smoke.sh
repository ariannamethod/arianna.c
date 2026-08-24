#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_smoke.sh - draft final-gate observation boundary preflight gate candidate from compact weighted graft admission final-gate observation boundary preflight gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate.XXXXXX")}"
GATE_WORKDIR="$WORKDIR/final_gate_observation_boundary_preflight_gate"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.json"
GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.json}"
GATE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.log"
CANDIDATE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-smoke] FAIL: $*" >&2
    if [[ -f "$GATE_LOG" ]]; then
        tail -n 500 "$GATE_LOG" >&2 || true
    fi
    if [[ -f "$CANDIDATE_LOG" ]]; then
        tail -n 260 "$CANDIDATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_WORKDIR="$GATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT="$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_smoke.sh" >"$GATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate producer failed"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.sh" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT" "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" >"$CANDIDATE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate rejected gate report"
fi

[[ -s "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate report not written: $GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate status"
require_grep '"target": "live_route_admission_next_step"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate target kind"
require_grep '"target_mode": "closed_preflight_gate_candidate_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate target mode"
require_grep '"action": "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate action"
require_grep '"writer_action": "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "writer action"
require_grep '"rollback_action": "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "rollback action"
require_grep '"ledger_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "ledger state"
require_grep '"ledger_action": "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "ledger action"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_receipt"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "receipt shape"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate state"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_action": "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate action field"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_target": "resonance"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate target field"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run_only": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate dry-run"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_gate_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate gate verified"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_preflight_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate preflight verified"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_boundary_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate boundary verified"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_observation_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate observation verified"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_verified": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate read-back verified"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate ready guard"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_kind": "blocked_final_gate_observation_boundary_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate kind"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_mode": "no_mutation_preflight_gate_candidate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate mode"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_stage": "post_preflight_gate_pre_live_admission"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate stage"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_observed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "raw observed guard"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "raw forwarded guard"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "raw allowed guard"
require_grep '"final_gate_observation_boundary_preflight_gate_candidate_body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "body mutation guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "weighted candidate ready"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_consumed": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "gate consumed"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_required": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "gate required"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "next-step block"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate id"
require_grep '"causal_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-causal-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate causal"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate hash"
require_grep '"admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-read-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate read-back hash"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v1"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate schema"
require_grep '"source_status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate status"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready": true' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source weighted gate ready"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_state": "blocked"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate state"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate ready guard"
require_grep '"source_final_gate_observation_boundary_preflight_gate_kind": "blocked_final_gate_observation_boundary_preflight_gate"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "source gate kind"
require_grep '"ledger_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed ledger flag"
require_grep '"ledger_append_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed ledger append flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "non-mutation flag"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "authority guard"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "body target"
require_grep '"reason": "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed"' "$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT" "candidate reason"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate\] pass:' "$CANDIDATE_LOG" "candidate pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_report=$GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT"
