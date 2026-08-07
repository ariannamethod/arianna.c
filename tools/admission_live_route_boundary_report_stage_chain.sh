#!/usr/bin/env bash
# admission_live_route_boundary_report_stage_chain.sh - canonical boundary report stage chain.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-boundary-report-stage-chain] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-boundary-report-assert-smoke"

exec "$METABOLISM" --admission-live-route-boundary-report-stage-chain "$@"
