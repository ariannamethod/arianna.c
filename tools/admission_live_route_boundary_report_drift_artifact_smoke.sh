#!/usr/bin/env bash
# admission_live_route_boundary_report_drift_artifact_smoke.sh - Go-written failed boundary report consumer smoke.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_BOUNDARY_REPORT_DRIFT_ARTIFACT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary-report-drift-artifact.XXXXXX")}"
BOUNDARY_REPORT="$WORKDIR/live_route_boundary_report.drift.json"
RUN_LOG="$WORKDIR/admission_live_route_boundary_report_drift_artifact.log"
PASS_ASSERT_LOG="$WORKDIR/pass_assert.log"

die() {
    echo "[admission-live-route-boundary-report-drift-artifact-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-boundary-report-drift-artifact-smoke"

if ! AM_LIVE_ROUTE_BOUNDARY_REPORT="$BOUNDARY_REPORT" \
    "$ROOT/metabolism" --admission-live-route-boundary-report-drift-artifact-smoke >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-boundary-report-drift-artifact-smoke failed"
fi

[[ -s "$BOUNDARY_REPORT" ]] || die "drift boundary report not written"

bash "$ROOT/tools/admission_live_route_boundary_report_failed_diagnostics_assert.sh" \
    "$BOUNDARY_REPORT" \
    writer_receipt \
    body_inventory_status \
    route_availability_status \
    route_availability_reason \
    route_missing_organs || die "Go-written drift boundary report failed diagnostic assertion"

if bash "$ROOT/tools/admission_live_route_boundary_report_assert.sh" \
    "$BOUNDARY_REPORT" \
    1 \
    writer_receipt >"$PASS_ASSERT_LOG" 2>&1; then
    die "Go-written drift boundary report passed the pass assertion"
fi
grep -q --fixed-strings "boundary report did not pass" -- "$PASS_ASSERT_LOG" || die "pass assertion rejected drift artifact for the wrong reason"

grep -q '\[admission-live-route-boundary-report-drift-artifact-smoke\] pass: report=' "$RUN_LOG" || die "pass sentinel missing"
grep -q 'mismatches=body_inventory_status,route_availability_status,route_availability_reason,route_missing_organs' "$RUN_LOG" || die "mismatch summary missing"

echo "[admission-live-route-boundary-report-drift-artifact-smoke] pass: report=$BOUNDARY_REPORT"
