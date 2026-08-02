#!/usr/bin/env bash
# admission_live_route_boundary_report_assert.sh - compact boundary report contract.

set -euo pipefail
export LC_ALL=C

die() {
    echo "[admission-live-route-boundary-report-assert] FAIL: $*" >&2
    exit 1
}

if [[ $# -lt 2 ]]; then
    die "usage: $0 REPORT EXPECTED_RECEIPTS [STAGE...]"
fi

report="$1"
expected_receipts="$2"
shift 2

[[ -n "$report" ]] || die "boundary report path missing"
[[ -s "$report" ]] || die "boundary report not written"

case "$expected_receipts" in
    ''|*[!0-9]*)
        die "expected receipt count must be numeric"
        ;;
esac

grep -q '^  "schema": "arianna.live_route_boundary_report.v1",$' -- "$report" || die "boundary report schema missing"
grep -q '^  "passed": true,$' -- "$report" || die "boundary report did not pass"
grep -q "^  \"receipts_checked\": ${expected_receipts},$" -- "$report" || die "boundary report receipt count mismatch"
grep -q '^  "boundary": {' -- "$report" || die "boundary report projection missing"

for stage in "$@"; do
    [[ -n "$stage" ]] || die "empty boundary report stage name"
    stage_pattern="\"name\": \"${stage}\""
    grep -q --fixed-strings "$stage_pattern" -- "$report" || die "boundary report stage missing: $stage"
done
