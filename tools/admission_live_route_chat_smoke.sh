#!/usr/bin/env bash
# admission_live_route_chat_smoke.sh - live route chat dry-run launcher.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METABOLISM="${A2A_METABOLISM_BIN:-$ROOT/metabolism}"

die() {
    echo "[admission-live-route-chat-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$METABOLISM" ]] || die "missing executable metabolism; run make admission-live-route-chat-smoke"

exec "$METABOLISM" --admission-live-route-chat-smoke "$@"
