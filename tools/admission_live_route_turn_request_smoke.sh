#!/usr/bin/env bash
# admission_live_route_turn_request_smoke.sh - bounded live route generator request.
#
# Converts human-turn route choices into route-generator request receipts without
# running generation or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_REQUEST_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn-request.XXXXXX")}"
LOG="$WORKDIR/live_route_turn_request.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn_request.log"

die() {
    echo "[admission-live-route-turn-request-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-request-smoke"

echo "[admission-live-route-turn-request-smoke] root=$ROOT"
echo "[admission-live-route-turn-request-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_REQUEST_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_REQUEST_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-request-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-request-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn request JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_request.v1"' "$LOG" || die "turn request schema missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "identity class missing"
grep -q '"prompt_class":"cold-reader"' "$LOG" || die "cold-reader class missing"
grep -q '"prompt_class":"recipient-lock"' "$LOG" || die "recipient-lock class missing"
grep -q '"prompt_class":"dream"' "$LOG" || die "dream class missing"
grep -q '"candidate_trigger":"chorus-identity"' "$LOG" || die "chorus identity trigger missing"
grep -q '"candidate_trigger":"user_bridge-cold-reader"' "$LOG" || die "user_bridge cold-reader trigger missing"
grep -q '"candidate_trigger":"qloop_target-recipient-lock"' "$LOG" || die "qloop_target recipient-lock trigger missing"
grep -q '"candidate_trigger":"direct-dream"' "$LOG" || die "direct dream trigger missing"
grep -q '"candidate_seed":"turn-' "$LOG" || die "turn-derived candidate seed missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed request missing"
grep -q 'live-route turn request dry-run: class=identity route=chorus source=chorus trigger=chorus-identity seed=turn-' "$RUN_LOG" || die "identity request line missing"
grep -q '\[admission-live-route-turn-request-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn request smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-request-smoke] pass: log=$LOG"
