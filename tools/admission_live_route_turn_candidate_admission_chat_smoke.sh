#!/usr/bin/env bash
# admission_live_route_turn_candidate_admission_chat_smoke.sh - chat surface for handoff -> adapter.
#
# This is receipt-only: it proves the default-off chat dry-run formatter can
# carry an adapter-backed draft through admission handoff and admission-adapter
# receipts without starting the GGUF voices or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_CHAT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-admission-chat.XXXXXX")}"
DRAFT_LOG="$WORKDIR/live_route_candidate_draft_chat.jsonl"
REVIEW_LOG="$WORKDIR/live_route_candidate_draft_review_chat.jsonl"
ADMISSION_LOG="$WORKDIR/live_route_candidate_admission_chat.jsonl"
ADAPTER_LOG="$WORKDIR/live_route_candidate_admission_adapter_chat.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_admission_chat.log"

die() {
    echo "[admission-live-route-turn-candidate-admission-chat-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-admission-chat-smoke"

echo "[admission-live-route-turn-candidate-admission-chat-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-admission-chat-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT="I am Arianna, and the chat handoff keeps the candidate named." \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG="$DRAFT_LOG" \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$REVIEW_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG="$ADMISSION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG="$ADAPTER_LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-admission-chat-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-admission-chat-smoke failed"
fi

[[ -s "$DRAFT_LOG" ]] || die "candidate draft chat JSONL log not written"
[[ -s "$REVIEW_LOG" ]] || die "candidate draft review chat JSONL log not written"
[[ -s "$ADMISSION_LOG" ]] || die "candidate admission chat JSONL log not written"
[[ -s "$ADAPTER_LOG" ]] || die "candidate admission adapter chat JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_draft.v1"' "$DRAFT_LOG" || die "draft schema missing"
grep -q '"draft_id":"draft-' "$DRAFT_LOG" || die "passed draft id missing"
grep -q '"passed":false' "$DRAFT_LOG" || die "failed draft missing"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$REVIEW_LOG" || die "review schema missing"
grep -q '"matched":true' "$REVIEW_LOG" || die "matched review missing"
grep -q '"matched":false' "$REVIEW_LOG" || die "failed review missing"
grep -q '"schema":"arianna.live_route_turn_candidate_admission.v1"' "$ADMISSION_LOG" || die "handoff schema missing"
grep -q '"handoff_id":"handoff-' "$ADMISSION_LOG" || die "passed handoff id missing"
grep -q '"passed":false' "$ADMISSION_LOG" || die "failed handoff missing"
grep -q '"schema":"arianna.live_route_turn_candidate_admission_adapter.v1"' "$ADAPTER_LOG" || die "adapter schema missing"
grep -q '"admission_adapter_id":"admission-adapter-' "$ADAPTER_LOG" || die "passed adapter id missing"
grep -q '"passed":false' "$ADAPTER_LOG" || die "failed adapter missing"

grep -q 'live-route candidate draft dry-run: class=identity route=chorus source=chorus' "$RUN_LOG" || die "chat draft line missing"
grep -q 'live-route candidate admission handoff dry-run: class=identity route=chorus source=chorus draft=draft-' "$RUN_LOG" || die "chat handoff line missing"
grep -q 'live-route candidate admission adapter dry-run: class=identity route=chorus source=chorus handoff=handoff-' "$RUN_LOG" || die "chat adapter line missing"
grep -q 'live-route candidate admission adapter dry-run: class=unknown route= source= handoff=' "$RUN_LOG" || die "chat failed adapter line missing"
grep -q '\[admission-live-route-turn-candidate-admission-chat-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate admission chat smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-admission-chat-smoke] pass: drafts=$DRAFT_LOG reviews=$REVIEW_LOG handoffs=$ADMISSION_LOG adapters=$ADAPTER_LOG"
