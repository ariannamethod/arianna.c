#!/usr/bin/env bash
# admission_live_route_turn_choice_smoke.sh - bounded live route chooser proposal.
#
# Turns human-turn observations into route-prefixed candidate proposals without
# running generation or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CHOICE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn-choice.XXXXXX")}"
LOG="$WORKDIR/live_route_turn_choice.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn_choice.log"

die() {
    echo "[admission-live-route-turn-choice-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-choice-smoke"

echo "[admission-live-route-turn-choice-smoke] root=$ROOT"
echo "[admission-live-route-turn-choice-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CHOICE_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-choice-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-choice-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn choice JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_choice.v1"' "$LOG" || die "turn choice schema missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "identity class missing"
grep -q '"prompt_class":"cold-reader"' "$LOG" || die "cold-reader class missing"
grep -q '"prompt_class":"recipient-lock"' "$LOG" || die "recipient-lock class missing"
grep -q '"prompt_class":"dream"' "$LOG" || die "dream class missing"
grep -q '"candidate_trigger":"chorus-identity"' "$LOG" || die "chorus identity trigger missing"
grep -q '"candidate_trigger":"user_bridge-cold-reader"' "$LOG" || die "user_bridge cold-reader trigger missing"
grep -q '"candidate_trigger":"qloop_target-recipient-lock"' "$LOG" || die "qloop_target recipient-lock trigger missing"
grep -q '"candidate_trigger":"direct-dream"' "$LOG" || die "direct dream trigger missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed choice missing"
grep -q 'live-route turn choice dry-run: class=identity route=chorus source=chorus trigger=chorus-identity passed=true' "$RUN_LOG" || die "identity choice line missing"
grep -q '\[admission-live-route-turn-choice-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn choice smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-choice-smoke] pass: log=$LOG"
