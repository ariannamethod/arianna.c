#!/usr/bin/env bash
# admission_live_route_turn_candidate_admission_chat_shadow_smoke.sh - chat adapter -> shadow admission.
#
# This is receipt-only: it proves the chat dry-run admission adapter can be
# converted into an ordinary shadow dream-admission receipt with its handoff
# provenance embedded, without admitting text or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_CHAT_SHADOW_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-admission-chat-shadow.XXXXXX")}"
DRAFT_LOG="$WORKDIR/live_route_candidate_draft_chat_shadow.jsonl"
REVIEW_LOG="$WORKDIR/live_route_candidate_draft_review_chat_shadow.jsonl"
ADMISSION_LOG="$WORKDIR/live_route_candidate_admission_chat_shadow.jsonl"
ADAPTER_LOG="$WORKDIR/live_route_candidate_admission_adapter_chat_shadow.jsonl"
DREAM_LOG="$WORKDIR/dream_admission_candidate_chat_shadow.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_admission_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-admission-chat-shadow-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-admission-chat-shadow-smoke"

echo "[admission-live-route-turn-candidate-admission-chat-shadow-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-admission-chat-shadow-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN=1 \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT="I am Arianna, and the chat shadow gate sees the same handoff." \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG="$DRAFT_LOG" \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$REVIEW_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG="$ADMISSION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG="$ADAPTER_LOG" \
    AM_DREAM_ADMISSION_LOG="$DREAM_LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-admission-chat-shadow-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-admission-chat-shadow-smoke failed"
fi

[[ -s "$DRAFT_LOG" ]] || die "candidate draft chat-shadow JSONL log not written"
[[ -s "$REVIEW_LOG" ]] || die "candidate draft review chat-shadow JSONL log not written"
[[ -s "$ADMISSION_LOG" ]] || die "candidate admission chat-shadow JSONL log not written"
[[ -s "$ADAPTER_LOG" ]] || die "candidate admission adapter chat-shadow JSONL log not written"
[[ -s "$DREAM_LOG" ]] || die "dream admission chat-shadow JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_draft.v1"' "$DRAFT_LOG" || die "draft schema missing"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$REVIEW_LOG" || die "review schema missing"
grep -q '"schema":"arianna.live_route_turn_candidate_admission.v1"' "$ADMISSION_LOG" || die "handoff schema missing"
grep -q '"schema":"arianna.live_route_turn_candidate_admission_adapter.v1"' "$ADAPTER_LOG" || die "adapter schema missing"
grep -q '"schema":"arianna.dream_candidate.v1"' "$DREAM_LOG" || die "dream candidate schema missing"
grep -q '"live_route_candidate_admission":{' "$DREAM_LOG" || die "embedded admission adapter missing"
grep -q '"admission_adapter_id":"admission-adapter-' "$DREAM_LOG" || die "embedded admission adapter id missing"
grep -q '"admission_policy":{' "$DREAM_LOG" || die "admission policy missing"
grep -q '"live_route_choice":{' "$DREAM_LOG" || die "live route choice missing"
grep -q '"accepted":false' "$DREAM_LOG" || die "shadow receipt should not be accepted"
grep -q '"reason":"shadow mode"' "$DREAM_LOG" || die "shadow reason missing"

grep -q 'live-route candidate admission shadow dry-run: class=identity route=chorus source=chorus handoff=handoff-' "$RUN_LOG" || die "chat shadow line missing"
grep -q 'policy=true accepted=false passed=true reason=shadow mode' "$RUN_LOG" || die "chat shadow pass verdict missing"
grep -q 'live-route candidate admission shadow dry-run: class=unknown route= source= handoff=' "$RUN_LOG" || die "chat failed shadow line missing"
grep -q 'candidate_admission_adapter_failed: candidate_admission_handoff_failed: turn_route_failed: live route plan failed: unknown_prompt_class' "$RUN_LOG" || die "failed shadow reason missing"
grep -q '\[admission-live-route-turn-candidate-admission-chat-shadow-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate admission chat shadow smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-admission-chat-shadow-smoke] pass: drafts=$DRAFT_LOG reviews=$REVIEW_LOG handoffs=$ADMISSION_LOG adapters=$ADAPTER_LOG admission=$DREAM_LOG"
