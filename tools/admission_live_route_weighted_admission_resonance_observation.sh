#!/usr/bin/env bash
# admission_live_route_weighted_admission_resonance_observation.sh - record weighted Resonance observation metadata.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="$ROOT/metabolism"

if [[ ! -x "$METABOLISM" ]]; then
    echo "[admission-live-route-weighted-admission-resonance-observation] FAIL: missing executable metabolism; run make metabolism" >&2
    exit 1
fi

exec "$METABOLISM" --admission-live-route-weighted-admission-resonance-observation "$@"
