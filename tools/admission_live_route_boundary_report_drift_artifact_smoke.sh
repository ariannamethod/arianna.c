#!/usr/bin/env bash
# admission_live_route_boundary_report_drift_artifact_smoke.sh - failed boundary report artifact smoke launcher.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-boundary-report-drift-artifact-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-boundary-report-drift-artifact-smoke"

exec "$METABOLISM" --admission-live-route-boundary-report-drift-artifact-smoke "$@"
