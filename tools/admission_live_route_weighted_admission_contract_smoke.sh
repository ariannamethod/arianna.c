#!/usr/bin/env bash
# admission_live_route_weighted_admission_contract_smoke.sh - require weighted precondition before the admission contract layer.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-admission-contract.XXXXXX")}"
PRECONDITION_WORKDIR="$WORKDIR/precondition"
PRECONDITION_REPORT="$PRECONDITION_WORKDIR/live_route_weighted_readiness_precondition.json"
CONTRACT_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_ADMISSION_CONTRACT_REPORT:-$WORKDIR/live_route_weighted_admission_contract.json}"
PRECONDITION_LOG="$WORKDIR/weighted_readiness_precondition.log"
CONTRACT_LOG="$WORKDIR/weighted_admission_contract.log"

die() {
    echo "[admission-live-route-weighted-admission-contract-smoke] FAIL: $*" >&2
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 500 "$PRECONDITION_LOG" >&2 || true
    fi
    if [[ -f "$CONTRACT_LOG" ]]; then
        tail -n 160 "$CONTRACT_LOG" >&2 || true
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

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_PRECONDITION_WORKDIR="$PRECONDITION_WORKDIR" \
    bash "$ROOT/tools/admission_live_route_weighted_readiness_precondition_smoke.sh" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted readiness precondition producer failed"
fi

[[ -s "$PRECONDITION_REPORT" ]] || die "weighted readiness precondition report not written: $PRECONDITION_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_admission_contract.sh" "$PRECONDITION_REPORT" "$CONTRACT_REPORT" >"$CONTRACT_LOG" 2>&1; then
    die "weighted admission contract writer rejected precondition report"
fi

[[ -s "$CONTRACT_REPORT" ]] || die "weighted admission contract report not written: $CONTRACT_REPORT"

require_grep '"schema": "arianna.live_route_weighted_admission_contract.v1"' "$CONTRACT_REPORT" "contract schema"
require_grep '"status": "contract_ready_closed_dry_run"' "$CONTRACT_REPORT" "contract status"
require_grep '"target": "live_route_admission"' "$CONTRACT_REPORT" "contract target"
require_grep '"weighted_admission_contract_ready": true' "$CONTRACT_REPORT" "contract-ready flag"
require_grep '"weighted_readiness_precondition_consumed": true' "$CONTRACT_REPORT" "precondition consumed flag"
require_grep '"weighted_readiness_precondition_required": true' "$CONTRACT_REPORT" "precondition required flag"
require_grep '"next_step_blocked_without_precondition": true' "$CONTRACT_REPORT" "next-step block flag"
require_grep '"body_smoke_weighted": true' "$CONTRACT_REPORT" "weighted body-smoke flag"
require_grep '"nano_direct_final_gate": true' "$CONTRACT_REPORT" "nano final-gate flag"
require_grep '"resonance_graft_admission_proof": true' "$CONTRACT_REPORT" "resonance graft proof flag"
require_grep '"boundary_report_full_chain": true' "$CONTRACT_REPORT" "boundary full-chain flag"
require_grep '"contracts_ready": false' "$CONTRACT_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$CONTRACT_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$CONTRACT_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$CONTRACT_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$CONTRACT_REPORT" "non-mutation flag"
require_grep '"passed": true' "$CONTRACT_REPORT" "contract pass flag"
require_grep '\[admission-live-route-weighted-admission-contract\] pass:' "$CONTRACT_LOG" "contract pass line"

echo "[admission-live-route-weighted-admission-contract-smoke] pass: precondition_report=$PRECONDITION_REPORT contract_report=$CONTRACT_REPORT"
