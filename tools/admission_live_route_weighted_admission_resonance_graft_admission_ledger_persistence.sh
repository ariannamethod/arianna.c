#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_graft_admission_ledger_persistence.sh - block weighted Resonance graft admission ledger persistence behind blocked ledger implementation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="$ROOT/metabolism"

if [[ ! -x "$METABOLISM" ]]; then
    echo "[admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence] FAIL: missing executable metabolism; run make metabolism" >&2
    exit 1
fi

exec "$METABOLISM" --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence "$@"
