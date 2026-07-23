#!/usr/bin/env bash
# admission_live_route_turn_bridge_smoke.sh - type nano/human-turn candidates from the human turn receipt.
#
# This remains receipt-only. It proves AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN can
# turn source=nano, trigger=human-turn into a prompt-class-aware candidate route
# for review while leaving source=nano bounded by the live route map.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_BRIDGE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn-bridge.XXXXXX")}"
LOG="$WORKDIR/live_route_turn_bridge.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn_bridge.log"

die() {
    echo "[admission-live-route-turn-bridge-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 100 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-bridge-smoke"

echo "[admission-live-route-turn-bridge-smoke] root=$ROOT"
echo "[admission-live-route-turn-bridge-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-bridge-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-bridge-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn bridge JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$LOG" || die "review schema missing"
grep -q '"candidate_source":"nano"' "$LOG" || die "nano source missing"
grep -q '"candidate_trigger":"human-turn"' "$LOG" || die "original human-turn trigger missing"
grep -q '"candidate_bridge_applied":true' "$LOG" || die "bridge-applied receipt missing"
grep -q '"candidate_bridge_trigger":"human-turn-identity"' "$LOG" || die "identity bridge trigger missing"
grep -q '"candidate_bridge_trigger":"human-turn-direct-user"' "$LOG" || die "direct-user bridge trigger missing"
grep -q '"candidate_prompt_class":"identity"' "$LOG" || die "identity candidate class missing"
grep -q '"candidate_prompt_class":"direct-user"' "$LOG" || die "direct-user candidate class missing"
grep -q '"candidate_route":"chorus"' "$LOG" || die "chorus candidate route missing"
grep -q '"candidate_route":"user_bridge"' "$LOG" || die "user_bridge candidate route missing"
grep -q 'source nano does not match live route chorus for prompt class identity' "$LOG" || die "identity nano source-bound reason missing"
grep -q 'source nano does not match live route user_bridge for prompt class direct-user' "$LOG" || die "direct-user nano source-bound reason missing"
grep -q 'live-route turn/candidate review: turn_class=identity expected=chorus candidate_source=nano candidate_class=identity candidate_route=chorus matched=false bridge=human-turn-identity' "$RUN_LOG" || die "identity bridge line missing"
grep -q 'live-route turn/candidate review: turn_class=direct-user expected=user_bridge candidate_source=nano candidate_class=direct-user candidate_route=user_bridge matched=false bridge=human-turn-direct-user' "$RUN_LOG" || die "direct-user bridge line missing"
grep -q '\[admission-live-route-turn-bridge-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn bridge smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-bridge-smoke] pass: log=$LOG"
