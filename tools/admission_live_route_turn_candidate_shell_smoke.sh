#!/usr/bin/env bash
# admission_live_route_turn_candidate_shell_smoke.sh - bounded live route candidate envelope.
#
# Converts live-route generation jobs into pending candidate shell receipts
# without running generation or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_SHELL_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-shell.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_shell.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_shell.log"

die() {
    echo "[admission-live-route-turn-candidate-shell-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-shell-smoke"

echo "[admission-live-route-turn-candidate-shell-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-shell-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-shell-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-shell-smoke failed"
fi

[[ -s "$LOG" ]] || die "candidate shell JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_shell.v1"' "$LOG" || die "candidate shell schema missing"
grep -q '"candidate_schema":"arianna.dream_candidate.v1"' "$LOG" || die "dream candidate schema missing"
grep -q '"candidate_text_status":"pending_generation"' "$LOG" || die "pending generation status missing"
grep -q '"candidate_kind":"chorus"' "$LOG" || die "chorus candidate kind missing"
grep -q '"candidate_kind":"user_bridge"' "$LOG" || die "user bridge candidate kind missing"
grep -q '"candidate_kind":"qloop_target"' "$LOG" || die "qloop target candidate kind missing"
grep -q '"candidate_kind":"direct"' "$LOG" || die "direct candidate kind missing"
grep -q '"candidate_trigger":"chorus-identity"' "$LOG" || die "chorus identity trigger missing"
grep -q '"candidate_trigger":"user_bridge-cold-reader"' "$LOG" || die "user_bridge cold-reader trigger missing"
grep -q '"candidate_trigger":"qloop_target-recipient-lock"' "$LOG" || die "qloop_target recipient-lock trigger missing"
grep -q '"candidate_trigger":"direct-dream"' "$LOG" || die "direct dream trigger missing"
grep -q '"candidate_seed":"turn-' "$LOG" || die "turn-derived candidate seed missing"
grep -q '"job_id":"job-' "$LOG" || die "generation job id missing"
grep -q '"shell_id":"shell-' "$LOG" || die "candidate shell id missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed candidate shell missing"
grep -q 'live-route candidate shell dry-run: class=identity route=chorus source=chorus trigger=chorus-identity seed=turn-' "$RUN_LOG" || die "identity candidate shell line missing"
grep -q '\[admission-live-route-turn-candidate-shell-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate shell smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-shell-smoke] pass: log=$LOG"
