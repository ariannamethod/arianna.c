#!/usr/bin/env bash
# admission_live_route_boundary_report_assert_smoke.sh - negative fixture for report-level pass checks.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_BOUNDARY_REPORT_ASSERT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary-report-assert.XXXXXX")}"
ASSERT="$ROOT/tools/admission_live_route_boundary_report_assert.sh"
GOOD_REPORT="$WORKDIR/live_route_boundary_report.good.json"
BAD_TOP_LEVEL_REPORT="$WORKDIR/live_route_boundary_report.bad_top_level.json"
BAD_RECEIPTS_REPORT="$WORKDIR/live_route_boundary_report.bad_receipts.json"
BAD_TOP_LEVEL_LOG="$WORKDIR/bad_top_level.log"
BAD_RECEIPTS_LOG="$WORKDIR/bad_receipts.log"

die() {
    echo "[admission-live-route-boundary-report-assert-smoke] FAIL: $*" >&2
    exit 1
}

write_report() {
    local report="$1"
    local top_passed="$2"
    local receipts_checked="$3"

    {
        printf '%s\n' '{'
        printf '%s\n' '  "schema": "arianna.live_route_boundary_report.v1",'
        printf '  "passed": %s,\n' "$top_passed"
        printf '  "receipts_checked": %s,\n' "$receipts_checked"
        printf '%s\n' '  "boundary": {},'
        printf '%s\n' '  "stages": ['
        printf '%s\n' '    {'
        printf '%s\n' '      "name": "final_gate",'
        printf '%s\n' '      "passed": true'
        printf '%s\n' '    },'
        printf '%s\n' '    {'
        printf '%s\n' '      "name": "resonance_graft_admission_proof",'
        printf '%s\n' '      "passed": true'
        printf '%s\n' '    }'
        printf '%s\n' '  ]'
        printf '%s\n' '}'
    } > "$report"
}

mkdir -p "$WORKDIR"

write_report "$GOOD_REPORT" true 2
bash "$ASSERT" "$GOOD_REPORT" 2 final_gate resonance_graft_admission_proof || die "valid boundary report rejected"

write_report "$BAD_TOP_LEVEL_REPORT" false 2
if bash "$ASSERT" "$BAD_TOP_LEVEL_REPORT" 2 final_gate resonance_graft_admission_proof > "$BAD_TOP_LEVEL_LOG" 2>&1; then
    die "top-level false boundary report passed assertion"
fi
grep -q --fixed-strings "boundary report did not pass" -- "$BAD_TOP_LEVEL_LOG" || die "top-level false failure reason missing"

write_report "$BAD_RECEIPTS_REPORT" true 3
if bash "$ASSERT" "$BAD_RECEIPTS_REPORT" 2 final_gate resonance_graft_admission_proof > "$BAD_RECEIPTS_LOG" 2>&1; then
    die "wrong receipt count boundary report passed assertion"
fi
grep -q --fixed-strings "boundary report receipt count mismatch" -- "$BAD_RECEIPTS_LOG" || die "receipt count failure reason missing"

echo "[admission-live-route-boundary-report-assert-smoke] pass: workdir=$WORKDIR"
