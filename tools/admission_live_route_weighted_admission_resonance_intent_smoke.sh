#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_intent_smoke.sh - draft weighted Resonance intent after final gate.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-resonance-intent.XXXXXX")}"
FINAL_GATE_WORKDIR="$WORKDIR/final_gate"
FINAL_GATE_REPORT="$FINAL_GATE_WORKDIR/live_route_weighted_admission_final_gate.json"
RESONANCE_INTENT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_RESONANCE_INTENT_REPORT:-$WORKDIR/live_route_weighted_admission_resonance_intent.json}"
FINAL_GATE_LOG="$WORKDIR/weighted_admission_final_gate.log"
RESONANCE_INTENT_LOG="$WORKDIR/weighted_admission_resonance_intent.log"

die() {
    echo "[admission-live-route-weighted-admission-resonance-intent-smoke] FAIL: $*" >&2
    if [[ -f "$FINAL_GATE_LOG" ]]; then
        tail -n 500 "$FINAL_GATE_LOG" >&2 || true
    fi
    if [[ -f "$RESONANCE_INTENT_LOG" ]]; then
        tail -n 160 "$RESONANCE_INTENT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_WORKDIR="$FINAL_GATE_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_REPORT="$FINAL_GATE_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_final_gate_smoke.sh" >"$FINAL_GATE_LOG" 2>&1; then
    die "weighted admission final gate producer failed"
fi

[[ -s "$FINAL_GATE_REPORT" ]] || die "weighted admission final gate report not written: $FINAL_GATE_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_resonance_intent.sh" "$FINAL_GATE_REPORT" "$RESONANCE_INTENT_REPORT" >"$RESONANCE_INTENT_LOG" 2>&1; then
    die "weighted admission resonance intent writer rejected final gate report"
fi

[[ -s "$RESONANCE_INTENT_REPORT" ]] || die "weighted admission resonance intent report not written: $RESONANCE_INTENT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_resonance_intent.v1"' "$RESONANCE_INTENT_REPORT" "resonance-intent schema"
require_grep '"status": "resonance_intent_drafted_dry_run"' "$RESONANCE_INTENT_REPORT" "resonance-intent status"
require_grep '"target": "resonance"' "$RESONANCE_INTENT_REPORT" "resonance-intent target"
require_grep '"target_kind": "weighted_live_route_first_receiver"' "$RESONANCE_INTENT_REPORT" "resonance-intent target kind"
require_grep '"target_mode": "bounded_direction_dry_run"' "$RESONANCE_INTENT_REPORT" "resonance-intent target mode"
require_grep '"action": "draft_weighted_resonance_direction_intent_dry_run"' "$RESONANCE_INTENT_REPORT" "resonance-intent action"
require_grep '"weighted_admission_resonance_intent_ready": true' "$RESONANCE_INTENT_REPORT" "resonance-intent ready flag"
require_grep '"weighted_admission_final_gate_consumed": true' "$RESONANCE_INTENT_REPORT" "final gate consumed flag"
require_grep '"weighted_admission_final_gate_required": true' "$RESONANCE_INTENT_REPORT" "final gate required flag"
require_grep '"next_step_blocked_without_resonance_intent": true' "$RESONANCE_INTENT_REPORT" "next-step block flag"
require_grep '"receiver": "resonance"' "$RESONANCE_INTENT_REPORT" "receiver"
require_grep '"receiver_kind": "internal_world"' "$RESONANCE_INTENT_REPORT" "receiver kind"
require_grep '"influence_kind": "bounded_direction"' "$RESONANCE_INTENT_REPORT" "influence kind"
require_grep '"max_influence": 0.05' "$RESONANCE_INTENT_REPORT" "influence cap"
require_grep '"ttl_turns": 1' "$RESONANCE_INTENT_REPORT" "ttl"
require_grep '"raw_dream_text_allowed": false' "$RESONANCE_INTENT_REPORT" "raw dream text guard"
require_grep '"janus_surface_allowed": false' "$RESONANCE_INTENT_REPORT" "Janus surface guard"
require_grep '"cooc_learning_allowed": false' "$RESONANCE_INTENT_REPORT" "cooc guard"
require_grep '"delta_harvest_allowed": false' "$RESONANCE_INTENT_REPORT" "delta guard"
require_grep '"rollback_required": true' "$RESONANCE_INTENT_REPORT" "rollback requirement"
require_grep '"pre_state_hash_required": true' "$RESONANCE_INTENT_REPORT" "pre-state hash requirement"
require_grep '"post_state_hash_required": true' "$RESONANCE_INTENT_REPORT" "post-state hash requirement"
require_grep '"source_schema": "arianna.live_route_weighted_admission_final_gate.v1"' "$RESONANCE_INTENT_REPORT" "source schema"
require_grep '"source_status": "ready_closed_dry_run"' "$RESONANCE_INTENT_REPORT" "source status"
require_grep '"source_target": "live_route_admission_final_gate"' "$RESONANCE_INTENT_REPORT" "source target"
require_grep '"source_weighted_admission_final_gate_ready": true' "$RESONANCE_INTENT_REPORT" "source final-gate-ready flag"
require_grep '"source_weighted_admission_seal_consumed": true' "$RESONANCE_INTENT_REPORT" "source seal consumed flag"
require_grep '"source_weighted_admission_seal_required": true' "$RESONANCE_INTENT_REPORT" "source seal required flag"
require_grep '"source_weighted_admission_seal_ready": true' "$RESONANCE_INTENT_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_consumed": true' "$RESONANCE_INTENT_REPORT" "source permit consumed flag"
require_grep '"source_weighted_admission_permit_required": true' "$RESONANCE_INTENT_REPORT" "source permit required flag"
require_grep '"source_weighted_admission_permit_ready": true' "$RESONANCE_INTENT_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$RESONANCE_INTENT_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$RESONANCE_INTENT_REPORT" "source authority required flag"
require_grep '"source_manual_permit_requested": true' "$RESONANCE_INTENT_REPORT" "source manual permit flag"
require_grep '"source_permit_key_matched": true' "$RESONANCE_INTENT_REPORT" "source permit key flag"
require_grep '"body_smoke_weighted": true' "$RESONANCE_INTENT_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$RESONANCE_INTENT_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$RESONANCE_INTENT_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$RESONANCE_INTENT_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$RESONANCE_INTENT_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$RESONANCE_INTENT_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$RESONANCE_INTENT_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$RESONANCE_INTENT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$RESONANCE_INTENT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$RESONANCE_INTENT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$RESONANCE_INTENT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$RESONANCE_INTENT_REPORT" "non-mutation flag"
require_grep '"passed": true' "$RESONANCE_INTENT_REPORT" "resonance-intent pass flag"
require_grep '\[admission-live-route-weighted-admission-resonance-intent\] pass:' "$RESONANCE_INTENT_LOG" "resonance-intent pass line"

echo "[admission-live-route-weighted-admission-resonance-intent-smoke] pass: final_gate_report=$FINAL_GATE_REPORT resonance_intent_report=$RESONANCE_INTENT_REPORT"
