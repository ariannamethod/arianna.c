#!/usr/bin/env bash
# admission_live_route_weighted_admission_final_gate_smoke.sh - keep weighted admission closed after seal.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-final-gate.XXXXXX")}"
SEAL_WORKDIR="$WORKDIR/seal"
SEAL_REPORT="$SEAL_WORKDIR/live_route_weighted_admission_seal.json"
FINAL_GATE_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_FINAL_GATE_REPORT:-$WORKDIR/live_route_weighted_admission_final_gate.json}"
SEAL_LOG="$WORKDIR/weighted_admission_seal.log"
FINAL_GATE_LOG="$WORKDIR/weighted_admission_final_gate.log"

die() {
    echo "[admission-live-route-weighted-admission-final-gate-smoke] FAIL: $*" >&2
    if [[ -f "$SEAL_LOG" ]]; then
        tail -n 500 "$SEAL_LOG" >&2 || true
    fi
    if [[ -f "$FINAL_GATE_LOG" ]]; then
        tail -n 160 "$FINAL_GATE_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_SEAL_WORKDIR="$SEAL_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_SEAL_REPORT="$SEAL_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_seal_smoke.sh" >"$SEAL_LOG" 2>&1; then
    die "weighted admission seal producer failed"
fi

[[ -s "$SEAL_REPORT" ]] || die "weighted admission seal report not written: $SEAL_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_final_gate.sh" "$SEAL_REPORT" "$FINAL_GATE_REPORT" >"$FINAL_GATE_LOG" 2>&1; then
    die "weighted admission final gate writer rejected seal report"
fi

[[ -s "$FINAL_GATE_REPORT" ]] || die "weighted admission final gate report not written: $FINAL_GATE_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_final_gate.v1"' "$FINAL_GATE_REPORT" "final-gate schema"
require_grep '"status": "ready_closed_dry_run"' "$FINAL_GATE_REPORT" "final-gate status"
require_grep '"target": "live_route_admission_final_gate"' "$FINAL_GATE_REPORT" "final-gate target"
require_grep '"weighted_admission_final_gate_ready": true' "$FINAL_GATE_REPORT" "final-gate-ready flag"
require_grep '"weighted_admission_seal_consumed": true' "$FINAL_GATE_REPORT" "seal consumed flag"
require_grep '"weighted_admission_seal_required": true' "$FINAL_GATE_REPORT" "seal required flag"
require_grep '"next_step_blocked_without_final_gate": true' "$FINAL_GATE_REPORT" "final-gate next-step block flag"
require_grep '"source_weighted_admission_seal_ready": true' "$FINAL_GATE_REPORT" "source seal-ready flag"
require_grep '"source_weighted_admission_permit_consumed": true' "$FINAL_GATE_REPORT" "source permit consumed flag"
require_grep '"source_weighted_admission_permit_required": true' "$FINAL_GATE_REPORT" "source permit required flag"
require_grep '"source_weighted_admission_permit_ready": true' "$FINAL_GATE_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$FINAL_GATE_REPORT" "source authority consumed flag"
require_grep '"source_weighted_admission_authority_required": true' "$FINAL_GATE_REPORT" "source authority required flag"
require_grep '"source_manual_permit_requested": true' "$FINAL_GATE_REPORT" "source manual permit flag"
require_grep '"source_permit_key_matched": true' "$FINAL_GATE_REPORT" "source permit key flag"
require_grep '"body_smoke_weighted": true' "$FINAL_GATE_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$FINAL_GATE_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$FINAL_GATE_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$FINAL_GATE_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$FINAL_GATE_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$FINAL_GATE_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$FINAL_GATE_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$FINAL_GATE_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$FINAL_GATE_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$FINAL_GATE_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$FINAL_GATE_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$FINAL_GATE_REPORT" "non-mutation flag"
require_grep '"passed": true' "$FINAL_GATE_REPORT" "final-gate pass flag"
require_grep '\[admission-live-route-weighted-admission-final-gate\] pass:' "$FINAL_GATE_LOG" "final-gate pass line"

echo "[admission-live-route-weighted-admission-final-gate-smoke] pass: seal_report=$SEAL_REPORT final_gate_report=$FINAL_GATE_REPORT"
