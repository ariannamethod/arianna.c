#!/usr/bin/env bash
# admission_live_route_weighted_readiness_precondition.sh - weighted readiness admission precondition writer.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-weighted-readiness-precondition] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-weighted-readiness-precondition-smoke"

exec "$METABOLISM" --admission-live-route-weighted-readiness-precondition "$@"
