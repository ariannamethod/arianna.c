#!/usr/bin/env bash
# admission_live_route_turn_review_smoke.sh - compare prompt route observation to dream candidate route.
#
# This is receipt-only: it proves the review can see a matched typed candidate,
# a wrong-source typed candidate, and the current untyped nano human-turn
# candidate without granting any route power to the live voices.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_REVIEW_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn-review.XXXXXX")}"
LOG="$WORKDIR/live_route_turn_review.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn_review.log"

die() {
    echo "[admission-live-route-turn-review-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 100 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-review-smoke"

echo "[admission-live-route-turn-review-smoke] root=$ROOT"
echo "[admission-live-route-turn-review-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-review-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-review-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn/candidate review JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$LOG" || die "review schema missing"
grep -q '"timing":"async_subconscious"' "$LOG" || die "async timing marker missing"
grep -q '"turn_prompt_class":"identity"' "$LOG" || die "identity turn class missing"
grep -q '"turn_expected_source":"chorus"' "$LOG" || die "turn expected chorus missing"
grep -q '"candidate_source":"chorus"' "$LOG" || die "matched chorus candidate missing"
grep -q '"candidate_source":"direct"' "$LOG" || die "wrong-source direct candidate missing"
grep -q '"candidate_source":"nano"' "$LOG" || die "untyped nano candidate missing"
grep -q '"candidate_prompt_class":"human-turn"' "$LOG" || die "human-turn candidate class missing"
grep -q '"matched":true' "$LOG" || die "matched review missing"
grep -q '"matched":false' "$LOG" || die "failed review missing"
grep -q 'candidate_route_failed: live route plan failed: unknown_prompt_class' "$LOG" || die "untyped nano reason missing"
grep -q 'turn_route_failed: live route plan failed: unknown_prompt_class' "$LOG" || die "unknown turn reason missing"
grep -q 'live-route turn/candidate review: turn_class=identity expected=chorus candidate_source=chorus candidate_class=identity candidate_route=chorus matched=true' "$RUN_LOG" || die "matched review line missing"
grep -q '\[admission-live-route-turn-review-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn/candidate review smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-review-smoke] pass: log=$LOG"
