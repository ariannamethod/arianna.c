#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_candidate_store_reader_smoke.sh - read weighted Resonance shadow graft candidate store.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-candidate-store-reader.XXXXXX")}"
STORE_WORKDIR="$WORKDIR/store"
GRAFT_CANDIDATE_STORE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_candidate_store.json"
GRAFT_CANDIDATE_STORE_READER_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_candidate_store_reader.json}"
STORE_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_store.log"
READER_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_store_reader.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-smoke] FAIL: $*" >&2
    if [[ -f "$STORE_LOG" ]]; then
        tail -n 500 "$STORE_LOG" >&2 || true
    fi
    if [[ -f "$READER_LOG" ]]; then
        tail -n 220 "$READER_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_WORKDIR="$STORE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_REPORT="$GRAFT_CANDIDATE_STORE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_store_smoke.sh" >"$STORE_LOG" 2>&1; then
    die "weighted admission resonance graft candidate store producer failed"
fi

[[ -s "$GRAFT_CANDIDATE_STORE_REPORT" ]] || die "weighted admission resonance graft candidate store report not written: $GRAFT_CANDIDATE_STORE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_store_reader.sh" "$GRAFT_CANDIDATE_STORE_REPORT" "$GRAFT_CANDIDATE_STORE_READER_REPORT" >"$READER_LOG" 2>&1; then
    die "weighted admission resonance graft candidate store reader rejected store report"
fi

[[ -s "$GRAFT_CANDIDATE_STORE_READER_REPORT" ]] || die "weighted admission resonance graft candidate store reader report not written: $GRAFT_CANDIDATE_STORE_READER_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader schema"
require_grep '"status": "shadow_graft_candidate_store_read_back_dry_run"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader status"
require_grep '"target": "resonance"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_candidate_store_reader"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader target kind"
require_grep '"target_mode": "read_only_replay_dry_run"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader target mode"
require_grep '"action": "read_weighted_resonance_shadow_graft_candidate_store_dry_run"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader action"
require_grep '"weighted_admission_resonance_graft_candidate_store_reader_ready": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader ready flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_consumed": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "store consumed flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_required": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "store required flag"
require_grep '"next_step_blocked_without_resonance_graft_candidate_store_reader": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_candidate_store_reader_receipt"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "receipt shape"
require_grep '"reader_kind": "shadow_graft_candidate_store_reader"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader kind"
require_grep '"reader_mode": "read_only_replay"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader mode"
require_grep '"reader_stage": "pre_live_graft_candidate_store_reader"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader stage"
require_grep '"causal_id": "weighted-resonance-graft-candidate-store-reader-causal-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "causal id"
require_grep '"reader_hash": "weighted-resonance-graft-candidate-store-reader-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "reader hash"
require_grep '"replay_hash": "weighted-resonance-graft-candidate-store-reader-replay-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "replay hash"
require_grep '"read_back_hash": "weighted-resonance-graft-candidate-store-reader-read-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "read-back hash"
require_grep '"store_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "store verification"
require_grep '"candidate_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "authority verification"
require_grep '"store_hash_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "store hash verification"
require_grep '"store_read_back_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "store read-back verification"
require_grep '"admission_required": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "replay-only flag"
require_grep '"source_append_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source append flag"
require_grep '"source_read_back": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source read-back flag"
require_grep '"source_receipt_persisted": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source persisted flag"
require_grep '"source_receipt_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source verified flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store schema"
require_grep '"source_status": "shadow_graft_candidate_stored_dry_run"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store status"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_ready": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store ready"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_causal_id": "weighted-resonance-graft-candidate-store-causal-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store causal"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_hash": "weighted-resonance-graft-candidate-store-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store hash"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_read_back_hash": "weighted-resonance-graft-candidate-store-read-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store read-back"
require_grep '"source_store_action": "store_weighted_resonance_shadow_graft_candidate_dry_run"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store action"
require_grep '"source_store_receipt_shape": "weighted_resonance_shadow_graft_candidate_store_receipt"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store receipt"
require_grep '"source_store_kind": "shadow_graft_candidate_store"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store kind"
require_grep '"source_store_mode": "append_only_read_back_store"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store mode"
require_grep '"source_store_stage": "pre_live_graft_candidate_store"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store stage"
require_grep '"source_store_append_only": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store append"
require_grep '"source_store_read_back": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store read-back"
require_grep '"source_store_receipt_persisted": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store persisted"
require_grep '"source_store_receipt_verified": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store verified"
require_grep '"source_store_graft_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store graft guard"
require_grep '"source_store_raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store raw guard"
require_grep '"source_store_janus_surface_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store Janus guard"
require_grep '"source_store_cooc_learning_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store cooc guard"
require_grep '"source_store_delta_harvest_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store delta guard"
require_grep '"source_store_body_mutation_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store body guard"
require_grep '"source_store_rollback_required": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source store rollback"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_CANDIDATE_STORE_READER_REPORT" "resonance-graft-candidate-store-reader pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-candidate-store-reader\] pass:' "$READER_LOG" "resonance-graft-candidate-store-reader pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-smoke] pass: resonance_graft_candidate_store_report=$GRAFT_CANDIDATE_STORE_REPORT resonance_graft_candidate_store_reader_report=$GRAFT_CANDIDATE_STORE_READER_REPORT"
