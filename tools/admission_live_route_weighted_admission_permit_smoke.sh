#!/usr/bin/env bash
# admission_live_route_weighted_admission_permit_smoke.sh - keep weighted admission permit closed after authority.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_PERMIT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-permit.XXXXXX")}"
AUTHORITY_WORKDIR="$WORKDIR/authority"
AUTHORITY_REPORT="$AUTHORITY_WORKDIR/live_route_weighted_admission_authority.json"
PERMIT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_PERMIT_REPORT:-$WORKDIR/live_route_weighted_admission_permit.json}"
AUTHORITY_LOG="$WORKDIR/weighted_admission_authority.log"
PERMIT_LOG="$WORKDIR/weighted_admission_permit.log"

die() {
    echo "[admission-live-route-weighted-admission-permit-smoke] FAIL: $*" >&2
    if [[ -f "$AUTHORITY_LOG" ]]; then
        tail -n 500 "$AUTHORITY_LOG" >&2 || true
    fi
    if [[ -f "$PERMIT_LOG" ]]; then
        tail -n 160 "$PERMIT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_WORKDIR="$AUTHORITY_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_REPORT="$AUTHORITY_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_authority_smoke.sh" >"$AUTHORITY_LOG" 2>&1; then
    die "weighted admission authority producer failed"
fi

[[ -s "$AUTHORITY_REPORT" ]] || die "weighted admission authority report not written: $AUTHORITY_REPORT"

if ! A2A_WEIGHTED_ADMISSION_PERMIT_KEY=ARIANNA_WEIGHTED_ADMISSION_PERMIT_DRY_RUN_ONLY \
    bash "$ROOT/tools/admission_live_route_weighted_admission_permit.sh" "$AUTHORITY_REPORT" "$PERMIT_REPORT" >"$PERMIT_LOG" 2>&1; then
    die "weighted admission permit writer rejected authority report"
fi

[[ -s "$PERMIT_REPORT" ]] || die "weighted admission permit report not written: $PERMIT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_permit.v1"' "$PERMIT_REPORT" "permit schema"
require_grep '"status": "operator_permitted_closed_dry_run"' "$PERMIT_REPORT" "permit status"
require_grep '"target": "live_route_admission_permit"' "$PERMIT_REPORT" "permit target"
require_grep '"weighted_admission_permit_ready": true' "$PERMIT_REPORT" "permit-ready flag"
require_grep '"weighted_admission_authority_consumed": true' "$PERMIT_REPORT" "authority consumed flag"
require_grep '"weighted_admission_authority_required": true' "$PERMIT_REPORT" "authority required flag"
require_grep '"manual_permit_requested": true' "$PERMIT_REPORT" "manual permit flag"
require_grep '"permit_key_matched": true' "$PERMIT_REPORT" "permit key flag"
require_grep '"next_step_blocked_without_permit": true' "$PERMIT_REPORT" "permit next-step block flag"
require_grep '"body_smoke_weighted": true' "$PERMIT_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$PERMIT_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$PERMIT_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$PERMIT_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$PERMIT_REPORT" "boundary full-chain flag"
require_grep '"source_authority_granted": false' "$PERMIT_REPORT" "closed source authority flag"
require_grep '"authority_granted": false' "$PERMIT_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$PERMIT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$PERMIT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$PERMIT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$PERMIT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$PERMIT_REPORT" "non-mutation flag"
require_grep '"passed": true' "$PERMIT_REPORT" "permit pass flag"
require_grep '\[admission-live-route-weighted-admission-permit\] pass:' "$PERMIT_LOG" "permit pass line"

echo "[admission-live-route-weighted-admission-permit-smoke] pass: authority_report=$AUTHORITY_REPORT permit_report=$PERMIT_REPORT"
