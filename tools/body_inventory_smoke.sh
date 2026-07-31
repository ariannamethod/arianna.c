#!/usr/bin/env bash
# body_inventory_smoke.sh — read-only organ/weight availability receipt.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_BODY_INVENTORY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-body-inventory.XXXXXX")}"
LOG="$WORKDIR/body_inventory.jsonl"
RUN_LOG="$WORKDIR/body_inventory.log"

die() {
    echo "[body-inventory-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make body-inventory-smoke from repo root"

mkdir -p "$WORKDIR"
if ! (cd "$WORKDIR" && \
    AM_BODY_INVENTORY_ROOT="$ROOT" \
    AM_BODY_INVENTORY_LOG="$LOG" \
    "$ROOT/metabolism" --body-inventory-smoke) >"$RUN_LOG" 2>&1; then
    tail -n 80 "$RUN_LOG" >&2 || true
    die "metabolism body inventory smoke failed; log: $RUN_LOG"
fi

grep -q '"schema":"arianna.body_inventory.v1"' "$LOG" || die "receipt schema missing"
grep -q '"mutates_state":false' "$LOG" || die "receipt must be non-mutating"
grep -q '"continue_allowed":true' "$LOG" || die "receipt must keep inspection alive"
grep -q '"route_availability":' "$LOG" || die "route availability missing"
grep -q 'body-inventory: status=' "$RUN_LOG" || die "summary line missing"

for organ in janus-binary janus-weight resonance-binary resonance-weight nano-binary nano-weight; do
    grep -q "\"name\":\"$organ\"" "$LOG" || die "organ missing from receipt: $organ"
done

for route in direct chorus qloop qloop_hint_qa qloop_target user_bridge; do
    grep -q "\"route\":\"$route\"" "$LOG" || die "route missing from availability receipt: $route"
done

if grep -q '"status":"ready"' "$LOG"; then
    grep -q '"live_trio_allowed":true' "$LOG" || die "ready receipt must allow live trio"
else
    grep -Eq '"status":"(blocked|degraded)"' "$LOG" || die "receipt status must be ready/degraded/blocked"
fi

echo "[body-inventory-smoke] pass: receipt=$LOG run=$RUN_LOG"
