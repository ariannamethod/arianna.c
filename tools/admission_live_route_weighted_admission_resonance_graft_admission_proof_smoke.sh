#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_proof_smoke.sh - prove weighted Resonance shadow graft admission from reader.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-admission-proof.XXXXXX")}"
READER_WORKDIR="$WORKDIR/reader"
GRAFT_CANDIDATE_STORE_READER_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_candidate_store_reader.json"
GRAFT_ADMISSION_PROOF_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_ADMISSION_PROOF_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_admission_proof.json}"
READER_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_store_reader.log"
PROOF_LOG="$WORKDIR/weighted_admission_resonance_graft_admission_proof.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-proof-smoke] FAIL: $*" >&2
    if [[ -f "$READER_LOG" ]]; then
        tail -n 500 "$READER_LOG" >&2 || true
    fi
    if [[ -f "$PROOF_LOG" ]]; then
        tail -n 220 "$PROOF_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_WORKDIR="$READER_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT="$GRAFT_CANDIDATE_STORE_READER_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_store_reader_smoke.sh" >"$READER_LOG" 2>&1; then
    die "weighted admission resonance graft candidate store reader producer failed"
fi

[[ -s "$GRAFT_CANDIDATE_STORE_READER_REPORT" ]] || die "weighted admission resonance graft candidate store reader report not written: $GRAFT_CANDIDATE_STORE_READER_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_admission_proof.sh" "$GRAFT_CANDIDATE_STORE_READER_REPORT" "$GRAFT_ADMISSION_PROOF_REPORT" >"$PROOF_LOG" 2>&1; then
    die "weighted admission resonance graft admission proof rejected reader report"
fi

[[ -s "$GRAFT_ADMISSION_PROOF_REPORT" ]] || die "weighted admission resonance graft admission proof report not written: $GRAFT_ADMISSION_PROOF_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof schema"
require_grep '"status": "shadow_graft_admission_proof_ready_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof status"
require_grep '"target": "resonance"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_admission_proof"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof target kind"
require_grep '"target_mode": "receipt_only_closed_admission_proof_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof target mode"
require_grep '"action": "prove_weighted_resonance_shadow_graft_admission_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof action"
require_grep '"weighted_admission_resonance_graft_admission_proof_ready": true' "$GRAFT_ADMISSION_PROOF_REPORT" "admission proof ready flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_reader_consumed": true' "$GRAFT_ADMISSION_PROOF_REPORT" "reader consumed flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_reader_required": true' "$GRAFT_ADMISSION_PROOF_REPORT" "reader required flag"
require_grep '"next_step_blocked_without_resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_PROOF_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_admission_proof_id": "weighted-resonance-graft-admission-proof-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "admission proof id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_admission_proof_receipt"' "$GRAFT_ADMISSION_PROOF_REPORT" "receipt shape"
require_grep '"proof_kind": "shadow_graft_admission_proof"' "$GRAFT_ADMISSION_PROOF_REPORT" "proof kind"
require_grep '"proof_mode": "closed_read_back_admission_proof"' "$GRAFT_ADMISSION_PROOF_REPORT" "proof mode"
require_grep '"proof_stage": "pre_live_graft_admission_proof"' "$GRAFT_ADMISSION_PROOF_REPORT" "proof stage"
require_grep '"causal_id": "weighted-resonance-graft-admission-proof-causal-' "$GRAFT_ADMISSION_PROOF_REPORT" "causal id"
require_grep '"proof_hash": "weighted-resonance-graft-admission-proof-' "$GRAFT_ADMISSION_PROOF_REPORT" "proof hash"
require_grep '"read_back_hash": "weighted-resonance-graft-admission-proof-read-' "$GRAFT_ADMISSION_PROOF_REPORT" "read-back hash"
require_grep '"store_reader_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "store-reader verification"
require_grep '"store_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "store verification"
require_grep '"candidate_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "authority verification"
require_grep '"reader_hash_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "reader hash verification"
require_grep '"reader_replay_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "reader replay verification"
require_grep '"reader_read_back_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "reader read-back verification"
require_grep '"admission_required": true' "$GRAFT_ADMISSION_PROOF_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_ADMISSION_PROOF_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_ADMISSION_PROOF_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_ADMISSION_PROOF_REPORT" "rollback requirement"
require_grep '"read_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "read-only flag"
require_grep '"replay_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "replay-only flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader schema"
require_grep '"source_status": "shadow_graft_candidate_store_read_back_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader status"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_id": "weighted-resonance-graft-candidate-store-reader-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_ready": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader ready"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_causal_id": "weighted-resonance-graft-candidate-store-reader-causal-' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader causal"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_hash": "weighted-resonance-graft-candidate-store-reader-' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader hash"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_replay_hash": "weighted-resonance-graft-candidate-store-reader-replay-' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader replay"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_reader_read_back_hash": "weighted-resonance-graft-candidate-store-reader-read-' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader read-back"
require_grep '"source_reader_action": "read_weighted_resonance_shadow_graft_candidate_store_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader action"
require_grep '"source_reader_receipt_shape": "weighted_resonance_shadow_graft_candidate_store_reader_receipt"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader receipt"
require_grep '"source_reader_kind": "shadow_graft_candidate_store_reader"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader kind"
require_grep '"source_reader_mode": "read_only_replay"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader mode"
require_grep '"source_reader_stage": "pre_live_graft_candidate_store_reader"' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader stage"
require_grep '"source_reader_read_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader read-only"
require_grep '"source_reader_replay_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader replay-only"
require_grep '"source_reader_graft_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader graft guard"
require_grep '"source_reader_raw_dream_text_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader raw guard"
require_grep '"source_reader_janus_surface_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader Janus guard"
require_grep '"source_reader_cooc_learning_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader cooc guard"
require_grep '"source_reader_delta_harvest_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader delta guard"
require_grep '"source_reader_body_mutation_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader body guard"
require_grep '"source_reader_live_admission_enabled": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source reader live guard"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source store id"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_hash": "weighted-resonance-graft-candidate-store-' "$GRAFT_ADMISSION_PROOF_REPORT" "source store hash"
require_grep '"source_weighted_admission_resonance_graft_candidate_store_read_back_hash": "weighted-resonance-graft-candidate-store-read-' "$GRAFT_ADMISSION_PROOF_REPORT" "source store read-back"
require_grep '"source_store_action": "store_weighted_resonance_shadow_graft_candidate_dry_run"' "$GRAFT_ADMISSION_PROOF_REPORT" "source store action"
require_grep '"source_store_append_only": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source store append"
require_grep '"source_store_read_back": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source store read-back"
require_grep '"source_store_receipt_persisted": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source store persisted"
require_grep '"source_store_receipt_verified": true' "$GRAFT_ADMISSION_PROOF_REPORT" "source store verified"
require_grep '"source_store_graft_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "source store graft guard"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_ADMISSION_PROOF_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_ADMISSION_PROOF_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_ADMISSION_PROOF_REPORT" "source receiver id"
require_grep '"body_smoke_weighted": true' "$GRAFT_ADMISSION_PROOF_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_ADMISSION_PROOF_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_ADMISSION_PROOF_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_ADMISSION_PROOF_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_ADMISSION_PROOF_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_ADMISSION_PROOF_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_ADMISSION_PROOF_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_ADMISSION_PROOF_REPORT" "resonance-graft-admission-proof pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-admission-proof\] pass:' "$PROOF_LOG" "resonance-graft-admission-proof pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-admission-proof-smoke] pass: resonance_graft_candidate_store_reader_report=$GRAFT_CANDIDATE_STORE_READER_REPORT resonance_graft_admission_proof_report=$GRAFT_ADMISSION_PROOF_REPORT"
