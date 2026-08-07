#!/usr/bin/env bash
# admission_live_route_boundary_report_failed_diagnostics_assert_smoke.sh - negative fixtures for failed report diagnostics.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-boundary-report-failed-diagnostics-assert-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-boundary-report-failed-diagnostics-assert-smoke"

exec "$METABOLISM" --admission-live-route-boundary-report-failed-diagnostics-assert-smoke "$@"
