#!/usr/bin/env bash
# admission_live_route_boundary_report_failed_diagnostics_assert.sh - failed compact boundary report diagnostics contract.

set -euo pipefail
export LC_ALL=C

die() {
    echo "[admission-live-route-boundary-report-failed-diagnostics-assert] FAIL: $*" >&2
    exit 1
}

if [[ $# -lt 3 ]]; then
    die "usage: $0 REPORT STAGE EXPECTED_MISMATCH [EXPECTED_MISMATCH...]"
fi

report="$1"
stage="$2"
shift 2

[[ -n "$report" ]] || die "boundary report path missing"
[[ -s "$report" ]] || die "boundary report not written"
[[ -n "$stage" ]] || die "boundary report stage name missing"

grep -q '^  "schema":[[:space:]]*"arianna.live_route_boundary_report.v1",' -- "$report" || die "boundary report schema missing"
grep -q '^  "passed":[[:space:]]*false,' -- "$report" || die "boundary report did not fail"
grep -q --fixed-strings "\"boundary_mismatch:${stage}\"" -- "$report" || die "boundary mismatch reason missing: $stage"

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

stage_has_passed_false() {
    awk -v stage="$stage" '
        $0 ~ /^[[:space:]]*"name":[[:space:]]*"/ {
            value = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            if (value == stage) { found = 1; next }
        }
        found && $0 ~ /^[[:space:]]*"passed":[[:space:]]*false,?$/ { ok = 1; exit 0 }
        found && $0 ~ /^[[:space:]]*},?$/ { exit 1 }
        END { if (!ok) exit 1 }
    ' "$report"
}

stage_has_mismatch() {
    local mismatch="$1"
    awk -v stage="$stage" -v mismatch="$mismatch" '
        $0 ~ /^[[:space:]]*"name":[[:space:]]*"/ {
            value = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            if (value == stage) { found = 1; next }
        }
        found && $0 ~ "^[[:space:]]*\"" mismatch "\",?$" { ok = 1; exit 0 }
        found && $0 ~ /^[[:space:]]*},?$/ { exit 1 }
        END { if (!ok) exit 1 }
    ' "$report"
}

stage_has_passed_false || die "boundary report stage did not fail: $stage"
stage_has_line_regex() {
    awk -v stage="$stage" '
        $0 ~ /^[[:space:]]*"name":[[:space:]]*"/ {
            value = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", value)
            sub(/".*$/, "", value)
            if (value == stage) { found = 1; next }
        }
        found && $0 ~ /^[[:space:]]*"mismatches":[[:space:]]*\[/ { ok = 1; exit 0 }
        found && $0 ~ /^[[:space:]]*},?$/ { exit 1 }
        END { if (!ok) exit 1 }
    ' "$report"
}
stage_has_line_regex || die "boundary report stage mismatches missing: $stage"

for mismatch in "$@"; do
    [[ -n "$mismatch" ]] || die "empty boundary mismatch name"
    stage_has_mismatch "$mismatch" || die "boundary report stage mismatch missing: $stage/$mismatch"
done
