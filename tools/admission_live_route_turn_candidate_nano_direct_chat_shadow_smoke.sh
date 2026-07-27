#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh - real nano direct -> shadow chain.
#
# Runs one live nano-Arianna direct generation, then carries that exact execution
# through adapter, draft, review, handoff, admission adapter, and shadow admission
# receipts without admitting text or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_CHAT_SHADOW_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-chat-shadow.XXXXXX")}"
EXECUTION_LOG="$WORKDIR/live_route_candidate_execution_nano_direct_chat_shadow.jsonl"
GENERATOR_ADAPTER_LOG="$WORKDIR/live_route_generator_adapter_nano_direct_chat_shadow.jsonl"
DRAFT_LOG="$WORKDIR/live_route_candidate_draft_nano_direct_chat_shadow.jsonl"
REVIEW_LOG="$WORKDIR/live_route_candidate_draft_review_nano_direct_chat_shadow.jsonl"
ADMISSION_LOG="$WORKDIR/live_route_candidate_admission_nano_direct_chat_shadow.jsonl"
ADAPTER_LOG="$WORKDIR/live_route_candidate_admission_adapter_nano_direct_chat_shadow.jsonl"
DREAM_LOG="$WORKDIR/dream_admission_candidate_nano_direct_chat_shadow.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 160 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

resolve_root_path() {
    local path="$1"
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$ROOT" "$path" ;;
    esac
}

resolve_model_path() {
    local raw="$1"
    local path
    path="$(resolve_root_path "$raw")"
    if [[ -f "$path" ]]; then
        printf '%s\n' "$path"
        return
    fi
    if [[ "$raw" != /* && "$ROOT" == */.worktrees/* ]]; then
        local main_root="${ROOT%%/.worktrees/*}"
        local alt="$main_root/$raw"
        if [[ -f "$alt" ]]; then
            printf '%s\n' "$alt"
            return
        fi
    fi
    printf '%s\n' "$path"
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke"
[[ -x "$ROOT/nano-arianna" ]] || die "missing executable nano-arianna; run make admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke"

model_file="$(resolve_model_path "${A2A_NANO_MODEL:-weights/nano_arianna_f16.gguf}")"
[[ -f "$model_file" ]] || die "model file missing: $model_file"

echo "[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] scratch=$WORKDIR"
echo "[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] model=$model_file"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER=nano-direct \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS="${A2A_NANO_DIRECT_RUNNER_TIMEOUT_MS:-30000}" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TEXT="${A2A_NANO_DIRECT_CHAT_SHADOW_PROMPT:-What does the dream remember in the field?}" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG="$EXECUTION_LOG" \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_TEXT= \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG="$GENERATOR_ADAPTER_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_TEXT= \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG="$DRAFT_LOG" \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$REVIEW_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG="$ADMISSION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG="$ADAPTER_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SHADOW_DRY_RUN=1 \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN=1 \
    AM_DREAM_ADMISSION_LOG="$DREAM_LOG" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_BIN="$ROOT/nano-arianna" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_MODEL="$model_file" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_MAX_TOKENS="${A2A_NANO_DIRECT_MAX_TOKENS:-24}" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_TEMP="${A2A_NANO_DIRECT_TEMP:-0.9}" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_TOP_P="${A2A_NANO_DIRECT_TOP_P:-0.92}" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke failed"
fi

[[ -s "$EXECUTION_LOG" ]] || die "candidate execution JSONL log not written"
[[ -s "$GENERATOR_ADAPTER_LOG" ]] || die "generator adapter JSONL log not written"
[[ -s "$DRAFT_LOG" ]] || die "candidate draft JSONL log not written"
[[ -s "$REVIEW_LOG" ]] || die "candidate review JSONL log not written"
[[ -s "$ADMISSION_LOG" ]] || die "candidate admission JSONL log not written"
[[ -s "$ADAPTER_LOG" ]] || die "candidate admission adapter JSONL log not written"
[[ -s "$DREAM_LOG" ]] || die "dream admission JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_execution.v1"' "$EXECUTION_LOG" || die "candidate execution schema missing"
grep -q '"runner":"nano-direct"' "$EXECUTION_LOG" || die "nano-direct runner missing"
grep -q '"runner_status":"succeeded"' "$EXECUTION_LOG" || die "nano-direct runner did not succeed"
grep -q '"schema":"arianna.live_route_turn_generator_adapter.v1"' "$GENERATOR_ADAPTER_LOG" || die "generator adapter schema missing"
grep -q '"candidate_execution_id":"execution-' "$GENERATOR_ADAPTER_LOG" || die "generator adapter lost execution id"
grep -q '"schema":"arianna.live_route_turn_candidate_draft.v1"' "$DRAFT_LOG" || die "draft schema missing"
grep -q '"candidate_execution_id":"execution-' "$DRAFT_LOG" || die "draft lost execution id"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$REVIEW_LOG" || die "review schema missing"
grep -q '"matched":true' "$REVIEW_LOG" || die "review did not match"
grep -q '"schema":"arianna.live_route_turn_candidate_admission.v1"' "$ADMISSION_LOG" || die "handoff schema missing"
grep -q '"schema":"arianna.live_route_turn_candidate_admission_adapter.v1"' "$ADAPTER_LOG" || die "admission adapter schema missing"
grep -q '"schema":"arianna.dream_candidate.v1"' "$DREAM_LOG" || die "dream candidate schema missing"
grep -q '"live_route_candidate_admission":{' "$DREAM_LOG" || die "embedded admission adapter missing"
grep -q '"accepted":false' "$DREAM_LOG" || die "shadow receipt should not be accepted"
grep -q '"reason":"shadow mode"' "$DREAM_LOG" || die "shadow reason missing"

grep -q 'live-route candidate execution dry-run: class=dream route=direct backend=nano-arianna entry=direct frame=q_a' "$RUN_LOG" || die "execution chat line missing"
grep -q 'runner=nano-direct runner_status=succeeded passed=true' "$RUN_LOG" || die "runner success verdict missing"
grep -q 'live-route candidate admission shadow dry-run: class=dream route=direct source=direct handoff=handoff-' "$RUN_LOG" || die "shadow chat line missing"
grep -q 'policy=true accepted=false passed=true reason=shadow mode' "$RUN_LOG" || die "shadow pass verdict missing"
grep -q '\[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "nano-direct chat shadow smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke] pass: execution=$EXECUTION_LOG adapter=$GENERATOR_ADAPTER_LOG drafts=$DRAFT_LOG reviews=$REVIEW_LOG handoffs=$ADMISSION_LOG admission_adapters=$ADAPTER_LOG admission=$DREAM_LOG"
