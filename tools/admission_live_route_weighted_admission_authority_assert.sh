#!/usr/bin/env bash
# admission_live_route_weighted_admission_authority_assert.sh - assert weighted admission authority JSON.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

if [[ ! -x "$METABOLISM" ]]; then
    echo "[admission-live-route-weighted-admission-authority-assert] FAIL: missing executable metabolism; run make metabolism" >&2
    exit 1
fi

exec "$METABOLISM" --admission-live-route-weighted-admission-authority-assert "$@"
