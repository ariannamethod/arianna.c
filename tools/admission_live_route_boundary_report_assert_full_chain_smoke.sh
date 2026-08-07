#!/usr/bin/env bash
# admission_live_route_boundary_report_assert_full_chain_smoke.sh - full-chain boundary report consumer smoke.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_BOUNDARY_REPORT_ASSERT_FULL_CHAIN_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary-report-assert-full-chain.XXXXXX")}"
ASSERT_FULL="$ROOT/tools/admission_live_route_boundary_report_assert_full_chain.sh"
STAGE_CHAIN="$ROOT/tools/admission_live_route_boundary_report_stage_chain.sh"
GOOD_REPORT="$WORKDIR/live_route_boundary_report.full_chain.good.json"
BAD_STAGE_REPORT="$WORKDIR/live_route_boundary_report.full_chain.bad_stage.json"
MISSING_REPORT="$WORKDIR/live_route_boundary_report.full_chain.missing.json"
USAGE_LOG="$WORKDIR/usage.log"
BAD_STAGE_LOG="$WORKDIR/bad_stage.log"
MISSING_LOG="$WORKDIR/missing.log"

die() {
    echo "[admission-live-route-boundary-report-assert-full-chain-smoke] FAIL: $*" >&2
    exit 1
}

load_stages() {
    STAGES=()
    while IFS= read -r stage_name; do
        [[ -n "$stage_name" ]] || die "stage chain exported empty stage"
        STAGES+=("$stage_name")
    done < <(bash "$STAGE_CHAIN")
    [[ ${#STAGES[@]} -gt 0 ]] || die "stage chain export was empty"
}

write_full_report() {
    local report="$1"
    local failed_stage="${2:-}"
    local stage
    local index=0

    {
        printf '%s\n' '{'
        printf '%s\n' '  "schema": "arianna.live_route_boundary_report.v1",'
        printf '%s\n' '  "passed": true,'
        printf '  "receipts_checked": %s,\n' "${#STAGES[@]}"
        printf '%s\n' '  "boundary": {},'
        printf '%s\n' '  "stages": ['
        for stage in "${STAGES[@]}"; do
            printf '%s\n' '    {'
            printf '      "name": "%s",\n' "$stage"
            if [[ "$stage" == "$failed_stage" ]]; then
                printf '%s\n' '      "passed": false'
            else
                printf '%s\n' '      "passed": true'
            fi
            index=$((index + 1))
            if [[ $index -lt ${#STAGES[@]} ]]; then
                printf '%s\n' '    },'
            else
                printf '%s\n' '    }'
            fi
        done
        printf '%s\n' '  ]'
        printf '%s\n' '}'
    } > "$report"
}

mkdir -p "$WORKDIR"
load_stages

write_full_report "$GOOD_REPORT"
bash "$ASSERT_FULL" "$GOOD_REPORT" || die "valid full-chain boundary report rejected"

if bash "$ASSERT_FULL" "$GOOD_REPORT" extra > "$USAGE_LOG" 2>&1; then
    die "extra arg passed full-chain assertion"
fi
grep -q --fixed-strings "usage:" -- "$USAGE_LOG" || die "usage failure reason missing"

write_full_report "$BAD_STAGE_REPORT" "${STAGES[0]}"
if bash "$ASSERT_FULL" "$BAD_STAGE_REPORT" > "$BAD_STAGE_LOG" 2>&1; then
    die "failed stage passed full-chain assertion"
fi
grep -q --fixed-strings "boundary report stage did not pass: ${STAGES[0]}" -- "$BAD_STAGE_LOG" || die "failed stage reason missing"

if bash "$ASSERT_FULL" "$MISSING_REPORT" > "$MISSING_LOG" 2>&1; then
    die "missing boundary report passed full-chain assertion"
fi
grep -q --fixed-strings "boundary report not written" -- "$MISSING_LOG" || die "missing report failure reason missing"

echo "[admission-live-route-boundary-report-assert-full-chain-smoke] pass: workdir=$WORKDIR stages=${#STAGES[@]}"
