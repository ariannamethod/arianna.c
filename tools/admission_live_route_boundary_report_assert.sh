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

grep -q '^  "schema":[[:space:]]*"arianna.live_route_boundary_report.v1",' -- "$report" || die "boundary report schema missing"
grep -q '^  "passed":[[:space:]]*true,' -- "$report" || die "boundary report did not pass"
grep -q "^  \"receipts_checked\":[[:space:]]*${expected_receipts}," -- "$report" || die "boundary report receipt count mismatch"
grep -q '^  "boundary":[[:space:]]*{' -- "$report" || die "boundary report projection missing"

stage_passed() {
    local stage="$1"
    awk -v stage="$stage" '
        $0 ~ /^[[:space:]]*"name":[[:space:]]*"/ {
            value = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            if (value == stage) { found = 1; next }
        }
        found && $0 ~ /^[[:space:]]*"passed":[[:space:]]*true,?$/ { ok = 1; exit 0 }
        found && $0 ~ /^[[:space:]]*},?$/ { exit 1 }
        END { if (!ok) exit 1 }
    ' "$report"
}

for stage in "$@"; do
    [[ -n "$stage" ]] || die "empty boundary report stage name"
    stage_count="$(awk -v stage="$stage" '
        $0 ~ /^[[:space:]]*"name":[[:space:]]*"/ {
            value = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            if (value == stage) count++
        }
        END { print count + 0 }
    ' "$report")"
    [[ "$stage_count" != "0" ]] || die "boundary report stage missing: $stage"
    [[ "$stage_count" == "1" ]] || die "boundary report stage duplicated: $stage"
    stage_passed "$stage" || die "boundary report stage did not pass: $stage"
done
