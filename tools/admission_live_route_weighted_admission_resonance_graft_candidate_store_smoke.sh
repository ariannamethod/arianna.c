#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_candidate_store_smoke.sh - store weighted Resonance shadow graft candidate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-graft-candidate-store.XXXXXX")}"
CANDIDATE_WORKDIR="$WORKDIR/candidate"
GRAFT_CANDIDATE_REPORT="$WORKDIR/live_route_weighted_admission_resonance_graft_candidate.json"
GRAFT_CANDIDATE_STORE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_STORE_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_graft_candidate_store.json}"
CANDIDATE_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate.log"
STORE_LOG="$WORKDIR/weighted_admission_resonance_graft_candidate_store.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-graft-candidate-store-smoke] FAIL: $*" >&2
    if [[ -f "$CANDIDATE_LOG" ]]; then
        tail -n 500 "$CANDIDATE_LOG" >&2 || true
    fi
    if [[ -f "$STORE_LOG" ]]; then
        tail -n 220 "$STORE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_WORKDIR="$CANDIDATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_GRAFT_CANDIDATE_REPORT="$GRAFT_CANDIDATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_smoke.sh" >"$CANDIDATE_LOG" 2>&1; then
    die "weighted admission resonance graft candidate producer failed"
fi

[[ -s "$GRAFT_CANDIDATE_REPORT" ]] || die "weighted admission resonance graft candidate report not written: $GRAFT_CANDIDATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_graft_candidate_store.sh" "$GRAFT_CANDIDATE_REPORT" "$GRAFT_CANDIDATE_STORE_REPORT" >"$STORE_LOG" 2>&1; then
    die "weighted admission resonance graft candidate store rejected candidate report"
fi

[[ -s "$GRAFT_CANDIDATE_STORE_REPORT" ]] || die "weighted admission resonance graft candidate store report not written: $GRAFT_CANDIDATE_STORE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store schema"
require_grep '"status": "shadow_graft_candidate_stored_dry_run"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store status"
require_grep '"target": "resonance"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store target"
require_grep '"target_kind": "weighted_internal_world_shadow_graft_candidate_store"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store target kind"
require_grep '"target_mode": "append_only_read_back_store_dry_run"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store target mode"
require_grep '"action": "store_weighted_resonance_shadow_graft_candidate_dry_run"' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store action"
require_grep '"weighted_admission_resonance_graft_candidate_store_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "store ready flag"
require_grep '"weighted_admission_resonance_graft_candidate_consumed": true' "$GRAFT_CANDIDATE_STORE_REPORT" "candidate consumed flag"
require_grep '"weighted_admission_resonance_graft_candidate_required": true' "$GRAFT_CANDIDATE_STORE_REPORT" "candidate required flag"
require_grep '"next_step_blocked_without_resonance_graft_candidate_store": true' "$GRAFT_CANDIDATE_STORE_REPORT" "next-step block flag"
require_grep '"weighted_admission_resonance_graft_candidate_store_id": "weighted-resonance-graft-candidate-store-id-' "$GRAFT_CANDIDATE_STORE_REPORT" "store id"
require_grep '"receipt_shape": "weighted_resonance_shadow_graft_candidate_store_receipt"' "$GRAFT_CANDIDATE_STORE_REPORT" "receipt shape"
require_grep '"store_kind": "shadow_graft_candidate_store"' "$GRAFT_CANDIDATE_STORE_REPORT" "store kind"
require_grep '"store_mode": "append_only_read_back_store"' "$GRAFT_CANDIDATE_STORE_REPORT" "store mode"
require_grep '"store_stage": "pre_live_graft_candidate_store"' "$GRAFT_CANDIDATE_STORE_REPORT" "store stage"
require_grep '"causal_id": "weighted-resonance-graft-candidate-store-causal-' "$GRAFT_CANDIDATE_STORE_REPORT" "causal id"
require_grep '"store_hash": "weighted-resonance-graft-candidate-store-' "$GRAFT_CANDIDATE_STORE_REPORT" "store hash"
require_grep '"read_back_hash": "weighted-resonance-graft-candidate-store-read-' "$GRAFT_CANDIDATE_STORE_REPORT" "read-back hash"
require_grep '"candidate_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "candidate verification"
require_grep '"gate_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "gate verification"
require_grep '"preflight_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "preflight verification"
require_grep '"boundary_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "boundary verification"
require_grep '"observation_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "observation verification"
require_grep '"receiver_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "receiver verification"
require_grep '"intent_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "intent verification"
require_grep '"final_gate_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "final-gate verification"
require_grep '"seal_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "seal verification"
require_grep '"permit_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "permit verification"
require_grep '"authority_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "authority verification"
require_grep '"admission_required": true' "$GRAFT_CANDIDATE_STORE_REPORT" "admission requirement"
require_grep '"shadow_only": true' "$GRAFT_CANDIDATE_STORE_REPORT" "shadow flag"
require_grep '"graft_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "graft guard"
require_grep '"dry_run_only": true' "$GRAFT_CANDIDATE_STORE_REPORT" "dry-run flag"
require_grep '"live_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "live-ready flag"
require_grep '"raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "raw dream text allow guard"
require_grep '"raw_dream_text_observed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "raw dream text observe guard"
require_grep '"raw_dream_text_forwarded": false' "$GRAFT_CANDIDATE_STORE_REPORT" "raw dream text forward guard"
require_grep '"janus_surface_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "delta guard"
require_grep '"body_mutation_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "body mutation guard"
require_grep '"rollback_required": true' "$GRAFT_CANDIDATE_STORE_REPORT" "rollback requirement"
require_grep '"append_only": true' "$GRAFT_CANDIDATE_STORE_REPORT" "append-only flag"
require_grep '"read_back": true' "$GRAFT_CANDIDATE_STORE_REPORT" "read-back flag"
require_grep '"receipt_persisted": true' "$GRAFT_CANDIDATE_STORE_REPORT" "receipt persisted flag"
require_grep '"receipt_verified": true' "$GRAFT_CANDIDATE_STORE_REPORT" "receipt verified flag"
require_grep '"source_schema": "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate schema"
require_grep '"source_status": "shadow_graft_candidate_ready_dry_run"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate status"
require_grep '"source_weighted_admission_resonance_graft_candidate_id": "weighted-resonance-graft-candidate-id-' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate id"
require_grep '"source_weighted_admission_resonance_graft_candidate_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate ready"
require_grep '"source_weighted_admission_resonance_graft_candidate_causal_id": "weighted-resonance-graft-candidate-causal-' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate causal"
require_grep '"source_weighted_admission_resonance_graft_candidate_hash": "weighted-resonance-graft-candidate-' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate hash"
require_grep '"source_weighted_admission_resonance_graft_candidate_read_back_hash": "weighted-resonance-graft-candidate-read-' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate read-back"
require_grep '"source_candidate_action": "draft_weighted_resonance_shadow_graft_candidate_dry_run"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate action"
require_grep '"source_candidate_receipt_shape": "weighted_resonance_shadow_graft_candidate_contract"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate receipt"
require_grep '"source_candidate_kind": "shadow_graft_candidate"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate kind"
require_grep '"source_candidate_mode": "no_mutation_candidate"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate mode"
require_grep '"source_candidate_stage": "pre_live_graft_candidate"' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate stage"
require_grep '"source_candidate_shadow_only": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate shadow"
require_grep '"source_candidate_graft_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate graft guard"
require_grep '"source_candidate_dry_run_only": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate dry-run"
require_grep '"source_candidate_live_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate live-ready"
require_grep '"source_candidate_raw_dream_text_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate raw guard"
require_grep '"source_candidate_janus_surface_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate Janus guard"
require_grep '"source_candidate_cooc_learning_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate cooc guard"
require_grep '"source_candidate_delta_harvest_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate delta guard"
require_grep '"source_candidate_body_mutation_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate body guard"
require_grep '"source_candidate_rollback_required": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source candidate rollback"
require_grep '"source_weighted_admission_resonance_graft_gate_id": "weighted-resonance-graft-gate-id-' "$GRAFT_CANDIDATE_STORE_REPORT" "source gate id"
require_grep '"source_weighted_admission_resonance_graft_preflight_id": "weighted-resonance-graft-preflight-id-' "$GRAFT_CANDIDATE_STORE_REPORT" "source preflight id"
require_grep '"source_weighted_admission_resonance_graft_boundary_id": "weighted-resonance-graft-boundary-id-' "$GRAFT_CANDIDATE_STORE_REPORT" "source boundary id"
require_grep '"source_weighted_admission_resonance_observation_id": "weighted-resonance-observation-' "$GRAFT_CANDIDATE_STORE_REPORT" "source observation id"
require_grep '"source_weighted_admission_resonance_receiver_id": "weighted-resonance-receiver-' "$GRAFT_CANDIDATE_STORE_REPORT" "source receiver id"
require_grep '"source_weighted_admission_resonance_intent_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source intent flag"
require_grep '"source_weighted_admission_final_gate_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source final gate flag"
require_grep '"source_weighted_admission_seal_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source seal flag"
require_grep '"source_weighted_admission_permit_ready": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source permit flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$GRAFT_CANDIDATE_STORE_REPORT" "source authority required flag"
require_grep '"body_smoke_weighted": true' "$GRAFT_CANDIDATE_STORE_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$GRAFT_CANDIDATE_STORE_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$GRAFT_CANDIDATE_STORE_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$GRAFT_CANDIDATE_STORE_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$GRAFT_CANDIDATE_STORE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$GRAFT_CANDIDATE_STORE_REPORT" "non-mutation flag"
require_grep '"body_target": "none"' "$GRAFT_CANDIDATE_STORE_REPORT" "body target"
require_grep '"passed": true' "$GRAFT_CANDIDATE_STORE_REPORT" "resonance-graft-candidate-store pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-graft-candidate-store\] pass:' "$STORE_LOG" "resonance-graft-candidate-store pass line"

echo "[admission-live-route-weighted-admission-resonance-graft-candidate-store-smoke] pass: resonance_graft_candidate_report=$GRAFT_CANDIDATE_REPORT resonance_graft_candidate_store_report=$GRAFT_CANDIDATE_STORE_REPORT"
