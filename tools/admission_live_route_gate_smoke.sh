#!/usr/bin/env bash
# admission_live_route_gate_smoke.sh - live route-plan admission gate launcher.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-gate-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-gate-smoke"

exec "$METABOLISM" --admission-live-route-gate-smoke "$@"
