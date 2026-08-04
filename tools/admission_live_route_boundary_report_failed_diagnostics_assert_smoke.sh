#!/usr/bin/env bash
# admission_live_route_boundary_report_failed_diagnostics_assert_smoke.sh - negative fixtures for failed report diagnostics.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_BOUNDARY_REPORT_FAILED_DIAGNOSTICS_ASSERT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary-report-failed-diagnostics.XXXXXX")}"
ASSERT="$ROOT/tools/admission_live_route_boundary_report_failed_diagnostics_assert.sh"
GOOD_REPORT="$WORKDIR/live_route_boundary_report.failed_good.json"
BAD_TOP_LEVEL_REPORT="$WORKDIR/live_route_boundary_report.bad_top_level.json"
BAD_STAGE_REPORT="$WORKDIR/live_route_boundary_report.bad_stage.json"
BAD_MISMATCH_REPORT="$WORKDIR/live_route_boundary_report.bad_mismatch.json"
BAD_REASON_REPORT="$WORKDIR/live_route_boundary_report.bad_reason.json"
BAD_TOP_LEVEL_LOG="$WORKDIR/bad_top_level.log"
BAD_STAGE_LOG="$WORKDIR/bad_stage.log"
BAD_MISMATCH_LOG="$WORKDIR/bad_mismatch.log"
BAD_REASON_LOG="$WORKDIR/bad_reason.log"

die() {
    echo "[admission-live-route-boundary-report-failed-diagnostics-assert-smoke] FAIL: $*" >&2
    exit 1
}

write_failed_report() {
    local report="$1"
    local top_passed="$2"
    local stage_passed="$3"
    local include_route_missing="${4:-true}"
    local reason="${5:-boundary_mismatch:writer_receipt}"

    {
        printf '%s\n' '{'
        printf '%s\n' '  "schema": "arianna.live_route_boundary_report.v1",'
        printf '  "passed": %s,\n' "$top_passed"
        printf '%s\n' '  "receipts_checked": 1,'
        printf '%s\n' '  "boundary": {},'
        printf '%s\n' '  "stages": ['
        printf '%s\n' '    {'
        printf '%s\n' '      "name": "writer_receipt",'
        printf '      "passed": %s,\n' "$stage_passed"
        printf '%s\n' '      "mismatches": ['
        printf '%s\n' '        "body_inventory_status",'
        printf '%s\n' '        "route_availability_status",'
        if [[ "$include_route_missing" == "true" ]]; then
            printf '%s\n' '        "route_availability_reason",'
            printf '%s\n' '        "route_missing_organs"'
        else
            printf '%s\n' '        "route_availability_reason"'
        fi
        printf '%s\n' '      ]'
        printf '%s\n' '    }'
        printf '%s\n' '  ],'
        printf '%s\n' '  "reasons": ['
        printf '    "%s"\n' "$reason"
        printf '%s\n' '  ]'
        printf '%s\n' '}'
    } > "$report"
}

mkdir -p "$WORKDIR"

write_failed_report "$GOOD_REPORT" false false true
bash "$ASSERT" "$GOOD_REPORT" writer_receipt \
    body_inventory_status \
    route_availability_status \
    route_availability_reason \
    route_missing_organs || die "valid failed boundary report diagnostics rejected"

write_failed_report "$BAD_TOP_LEVEL_REPORT" true false true
if bash "$ASSERT" "$BAD_TOP_LEVEL_REPORT" writer_receipt route_missing_organs > "$BAD_TOP_LEVEL_LOG" 2>&1; then
    die "top-level passing boundary report passed failed-diagnostics assertion"
fi
grep -q --fixed-strings "boundary report did not fail" -- "$BAD_TOP_LEVEL_LOG" || die "top-level failure reason missing"

write_failed_report "$BAD_STAGE_REPORT" false true true
if bash "$ASSERT" "$BAD_STAGE_REPORT" writer_receipt route_missing_organs > "$BAD_STAGE_LOG" 2>&1; then
    die "passing stage boundary report passed failed-diagnostics assertion"
fi
grep -q --fixed-strings "boundary report stage did not fail: writer_receipt" -- "$BAD_STAGE_LOG" || die "stage failure reason missing"

write_failed_report "$BAD_MISMATCH_REPORT" false false false
if bash "$ASSERT" "$BAD_MISMATCH_REPORT" writer_receipt route_missing_organs > "$BAD_MISMATCH_LOG" 2>&1; then
    die "missing mismatch boundary report passed failed-diagnostics assertion"
fi
grep -q --fixed-strings "boundary report stage mismatch missing: writer_receipt/route_missing_organs" -- "$BAD_MISMATCH_LOG" || die "missing mismatch failure reason missing"

write_failed_report "$BAD_REASON_REPORT" false false true "other_failure"
if bash "$ASSERT" "$BAD_REASON_REPORT" writer_receipt route_missing_organs > "$BAD_REASON_LOG" 2>&1; then
    die "missing boundary mismatch reason report passed failed-diagnostics assertion"
fi
grep -q --fixed-strings "boundary mismatch reason missing: writer_receipt" -- "$BAD_REASON_LOG" || die "missing reason failure reason missing"

echo "[admission-live-route-boundary-report-failed-diagnostics-assert-smoke] pass: workdir=$WORKDIR"
