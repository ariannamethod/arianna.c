#!/usr/bin/env bash
# admission_live_route_turn_smoke.sh - bounded live human-turn route observation.
#
# Classifies representative human turns into typed live-route observations and
# records them without asking any voice daemon to generate. This keeps the
# route chooser visible before it can choose runtime power.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn.XXXXXX")}"
LOG="$WORKDIR/live_route_turn.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn.log"

die() {
    echo "[admission-live-route-turn-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-smoke"

echo "[admission-live-route-turn-smoke] root=$ROOT"
echo "[admission-live-route-turn-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_observation.v1"' "$LOG" || die "turn observation schema missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "identity class missing"
grep -q '"prompt_class":"cold-reader"' "$LOG" || die "cold-reader class missing"
grep -q '"prompt_class":"recipient-lock"' "$LOG" || die "recipient-lock class missing"
grep -q '"prompt_class":"format"' "$LOG" || die "format class missing"
grep -q '"prompt_class":"dream"' "$LOG" || die "dream class missing"
grep -q '"prompt_class":"unknown"' "$LOG" || die "unknown class missing"
grep -q '"route":"chorus"' "$LOG" || die "chorus route missing"
grep -q '"route":"user_bridge"' "$LOG" || die "user_bridge route missing"
grep -q '"route":"qloop_target"' "$LOG" || die "qloop_target route missing"
grep -q '"route":"direct"' "$LOG" || die "direct route missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed turn missing"
grep -q 'live-route turn dry-run: class=identity route=chorus expected=chorus passed=true' "$RUN_LOG" || die "identity turn line missing"
grep -q '\[admission-live-route-turn-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn observation smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-smoke] pass: log=$LOG"
