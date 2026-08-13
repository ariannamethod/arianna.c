#!/usr/bin/env bash
# admission_live_route_weighted_admission_seal_smoke.sh - keep weighted admission seal closed after permit.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_SEAL_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-seal.XXXXXX")}"
PERMIT_WORKDIR="$WORKDIR/permit"
PERMIT_REPORT="$PERMIT_WORKDIR/live_route_weighted_admission_permit.json"
SEAL_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_SEAL_REPORT:-$WORKDIR/live_route_weighted_admission_seal.json}"
PERMIT_LOG="$WORKDIR/weighted_admission_permit.log"
SEAL_LOG="$WORKDIR/weighted_admission_seal.log"

die() {
    echo "[admission-live-route-weighted-admission-seal-smoke] FAIL: $*" >&2
    if [[ -f "$PERMIT_LOG" ]]; then
        tail -n 500 "$PERMIT_LOG" >&2 || true
    fi
    if [[ -f "$SEAL_LOG" ]]; then
        tail -n 160 "$SEAL_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_PERMIT_WORKDIR="$PERMIT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_PERMIT_REPORT="$PERMIT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_permit_smoke.sh" >"$PERMIT_LOG" 2>&1; then
    die "weighted admission permit producer failed"
fi

[[ -s "$PERMIT_REPORT" ]] || die "weighted admission permit report not written: $PERMIT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_seal.sh" "$PERMIT_REPORT" "$SEAL_REPORT" >"$SEAL_LOG" 2>&1; then
    die "weighted admission seal writer rejected permit report"
fi

[[ -s "$SEAL_REPORT" ]] || die "weighted admission seal report not written: $SEAL_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_seal.v1"' "$SEAL_REPORT" "seal schema"
require_grep '"status": "sealed_closed_dry_run"' "$SEAL_REPORT" "seal status"
require_grep '"target": "live_route_admission_seal"' "$SEAL_REPORT" "seal target"
require_grep '"weighted_admission_seal_ready": true' "$SEAL_REPORT" "seal-ready flag"
require_grep '"weighted_admission_permit_consumed": true' "$SEAL_REPORT" "permit consumed flag"
require_grep '"weighted_admission_permit_required": true' "$SEAL_REPORT" "permit required flag"
require_grep '"next_step_blocked_without_seal": true' "$SEAL_REPORT" "seal next-step block flag"
require_grep '"source_weighted_admission_permit_ready": true' "$SEAL_REPORT" "source permit-ready flag"
require_grep '"source_weighted_admission_authority_consumed": true' "$SEAL_REPORT" "source authority consumed flag"
require_grep '"source_manual_permit_requested": true' "$SEAL_REPORT" "source manual permit flag"
require_grep '"source_permit_key_matched": true' "$SEAL_REPORT" "source permit key flag"
require_grep '"body_smoke_weighted": true' "$SEAL_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$SEAL_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$SEAL_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$SEAL_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$SEAL_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$SEAL_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$SEAL_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$SEAL_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$SEAL_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$SEAL_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$SEAL_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$SEAL_REPORT" "non-mutation flag"
require_grep '"passed": true' "$SEAL_REPORT" "seal pass flag"
require_grep '\[admission-live-route-weighted-admission-seal\] pass:' "$SEAL_LOG" "seal pass line"

echo "[admission-live-route-weighted-admission-seal-smoke] pass: permit_report=$PERMIT_REPORT seal_report=$SEAL_REPORT"
