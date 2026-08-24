#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_smoke.sh - read blocked weighted Resonance admission final-gate observation-boundary preflight-gate candidate store.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader.XXXXXX")}"
STORE_WORKDIR="$WORKDIR/store"
STORE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.json"
READER_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.json}"
STORE_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.log"
READER_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-smoke] FAIL: $*" >&2
    if [[ -f "$STORE_LOG" ]]; then
        tail -n 500 "$STORE_LOG" >&2 || true
    fi
    if [[ -f "$READER_LOG" ]]; then
        tail -n 260 "$READER_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_WORKDIR="$STORE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT="$STORE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_smoke.sh" >"$STORE_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store producer failed"
fi

[[ -s "$STORE_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store report not written: $STORE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.sh" "$STORE_REPORT" "$READER_REPORT" >"$READER_LOG" 2>&1; then
    die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader rejected store report"
fi

[[ -s "$READER_REPORT" ]] || die "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader report not written: $READER_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"' "$READER_REPORT" "reader schema"
require_grep '"status": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run"' "$READER_REPORT" "reader status"
require_grep '"target": "live_route_admission_next_step"' "$READER_REPORT" "reader target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader"' "$READER_REPORT" "reader target kind"
require_grep '"target_mode": "read_only_replay_dry_run"' "$READER_REPORT" "reader target mode"
require_grep '"action": "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run"' "$READER_REPORT" "reader action"
require_grep '"ledger_append_allowed": false' "$READER_REPORT" "ledger append guard"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready": true' "$READER_REPORT" "reader ready flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_consumed": true' "$READER_REPORT" "store consumed flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader": true' "$READER_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-' "$READER_REPORT" "reader id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt"' "$READER_REPORT" "receipt shape"
require_grep '"reader_kind": "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader"' "$READER_REPORT" "reader kind"
require_grep '"reader_mode": "read_only_replay"' "$READER_REPORT" "reader mode"
require_grep '"reader_stage": "post_preflight_gate_candidate_store_pre_live_admission_reader"' "$READER_REPORT" "reader stage"
require_grep '"reader_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-' "$READER_REPORT" "reader hash"
require_grep '"replay_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-replay-' "$READER_REPORT" "replay hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-read-' "$READER_REPORT" "read-back hash"
require_grep '"store_verified": true' "$READER_REPORT" "store verification"
require_grep '"candidate_verified": true' "$READER_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$READER_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$READER_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$READER_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$READER_REPORT" "observation verification"
require_grep '"store_hash_verified": true' "$READER_REPORT" "store hash verification"
require_grep '"store_read_back_verified": true' "$READER_REPORT" "store read-back verification"
require_grep '"read_only": true' "$READER_REPORT" "read-only flag"
require_grep '"replay_only": true' "$READER_REPORT" "replay-only flag"
require_grep '"raw_dream_text_allowed": false' "$READER_REPORT" "raw dream text guard"
require_grep '"body_mutation_allowed": false' "$READER_REPORT" "body mutation guard"
require_grep '"authority_granted": false' "$READER_REPORT" "authority guard"
require_grep '"write_allowed": false' "$READER_REPORT" "writer guard"
require_grep '"admission_allowed": false' "$READER_REPORT" "admission guard"
require_grep '"live_admission_enabled": false' "$READER_REPORT" "live guard"
require_grep '"mutates_state": false' "$READER_REPORT" "mutation guard"
require_grep '"body_target": "none"' "$READER_REPORT" "body target"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store.v1"' "$READER_REPORT" "source store schema"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-' "$READER_REPORT" "source store id"
require_grep '"source_store_ledger_append_allowed": false' "$READER_REPORT" "source ledger append guard"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-' "$READER_REPORT" "source candidate id"
require_grep '"source_candidate_opened": false' "$READER_REPORT" "source candidate closed flag"
require_grep '"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-' "$READER_REPORT" "source gate id"
require_grep '"source_admission_final_gate_observation_boundary_preflight_gate_ready": false' "$READER_REPORT" "source gate closed"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader\] pass:' "$READER_LOG" "reader pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-smoke] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_report=$STORE_REPORT resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_report=$READER_REPORT"
