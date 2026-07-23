#!/usr/bin/env bash
# admission_live_route_chat_smoke.sh - bounded dry-run route choice chat surface.
#
# Runs a synthetic human-turn dream candidate through the same admission and
# chat dry-run formatter without starting the GGUF voice daemons. This proves a
# typed route-prefixed candidate can cross the human-turn boundary without being
# clobbered into an untyped outer trigger.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_CHAT_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-chat.XXXXXX")}"
LOG="$WORKDIR/dream_admission_live_route_chat.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_chat.log"

die() {
    echo "[admission-live-route-chat-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-chat-smoke"

echo "[admission-live-route-chat-smoke] root=$ROOT"
echo "[admission-live-route-chat-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_ALLOWED_SOURCES= \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_DREAM_ADMISSION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-chat-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-chat-smoke failed"
fi

[[ -s "$LOG" ]] || die "admission JSONL log not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"mode":"shadow"' "$LOG" || die "shadow mode missing"
grep -q '"trigger":"chorus-identity"' "$LOG" || die "route-prefixed trigger missing"
grep -q '"live_route_choice_dry_run":true' "$LOG" || die "dry-run marker missing"
grep -q '"live_route_plan":{' "$LOG" || die "live route plan missing"
grep -q '"live_route_choice":{' "$LOG" || die "live route choice missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "identity prompt class missing"
grep -q '"route":"chorus"' "$LOG" || die "chorus route missing"
grep -q '"accepted":false' "$LOG" || die "shadow candidate was not rejected"
grep -q 'live-route dry-run: class=identity route=chorus source=chorus expected=chorus passed=true' "$RUN_LOG" || die "chat dry-run line missing"
grep -q '\[admission-live-route-chat-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "chat dry-run smoke wrote durable organism state"
fi

echo "[admission-live-route-chat-smoke] pass: log=$LOG"
