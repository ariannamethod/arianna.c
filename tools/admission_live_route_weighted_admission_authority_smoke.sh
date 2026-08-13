#!/usr/bin/env bash
# admission_live_route_weighted_admission_authority_smoke.sh - keep weighted admission authority closed after contract consumption.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-authority.XXXXXX")}"
CONTRACT_WORKDIR="$WORKDIR/contract"
CONTRACT_REPORT="$CONTRACT_WORKDIR/live_route_weighted_admission_contract.json"
AUTHORITY_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_AUTHORITY_REPORT:-$WORKDIR/live_route_weighted_admission_authority.json}"
CONTRACT_LOG="$WORKDIR/weighted_admission_contract.log"
AUTHORITY_LOG="$WORKDIR/weighted_admission_authority.log"

die() {
    echo "[admission-live-route-weighted-admission-authority-smoke] FAIL: $*" >&2
    if [[ -f "$CONTRACT_LOG" ]]; then
        tail -n 500 "$CONTRACT_LOG" >&2 || true
    fi
    if [[ -f "$AUTHORITY_LOG" ]]; then
        tail -n 160 "$AUTHORITY_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_WORKDIR="$CONTRACT_WORKDIR" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_REPORT="$CONTRACT_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_admission_contract_smoke.sh" >"$CONTRACT_LOG" 2>&1; then
    die "weighted admission contract producer failed"
fi

[[ -s "$CONTRACT_REPORT" ]] || die "weighted admission contract report not written: $CONTRACT_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_authority.sh" "$CONTRACT_REPORT" "$AUTHORITY_REPORT" >"$AUTHORITY_LOG" 2>&1; then
    die "weighted admission authority writer rejected contract report"
fi

[[ -s "$AUTHORITY_REPORT" ]] || die "weighted admission authority report not written: $AUTHORITY_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_authority.v1"' "$AUTHORITY_REPORT" "authority schema"
require_grep '"status": "authority_receipt_closed_dry_run"' "$AUTHORITY_REPORT" "authority status"
require_grep '"target": "live_route_admission_authority"' "$AUTHORITY_REPORT" "authority target"
require_grep '"weighted_admission_authority_receipt_ready": true' "$AUTHORITY_REPORT" "authority receipt-ready flag"
require_grep '"weighted_admission_contract_consumed": true' "$AUTHORITY_REPORT" "contract consumed flag"
require_grep '"weighted_admission_contract_required": true' "$AUTHORITY_REPORT" "contract required flag"
require_grep '"next_step_blocked_without_authority": true' "$AUTHORITY_REPORT" "authority next-step block flag"
require_grep '"body_smoke_weighted": true' "$AUTHORITY_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_runner": true' "$AUTHORITY_REPORT" "nano direct runner flag"
require_grep '"nano_direct_final_gate": true' "$AUTHORITY_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$AUTHORITY_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$AUTHORITY_REPORT" "boundary full-chain flag"
require_grep '"authority_granted": false' "$AUTHORITY_REPORT" "closed authority flag"
require_grep '"contracts_ready": false' "$AUTHORITY_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$AUTHORITY_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$AUTHORITY_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$AUTHORITY_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$AUTHORITY_REPORT" "non-mutation flag"
require_grep '"passed": true' "$AUTHORITY_REPORT" "authority pass flag"
require_grep '\[admission-live-route-weighted-admission-authority\] pass:' "$AUTHORITY_LOG" "authority pass line"

echo "[admission-live-route-weighted-admission-authority-smoke] pass: contract_report=$CONTRACT_REPORT authority_report=$AUTHORITY_REPORT"
