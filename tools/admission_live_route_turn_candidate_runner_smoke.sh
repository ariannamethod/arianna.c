#!/usr/bin/env bash
# admission_live_route_turn_candidate_runner_smoke.sh - bounded candidate runner receipt path.
#
# Proves the execution layer can run a named bounded process, capture the output,
# and fail closed on timeout without touching durable organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_RUNNER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-runner.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_runner.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_runner.log"

die() {
    echo "[admission-live-route-turn-candidate-runner-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-runner-smoke"

echo "[admission-live-route-turn-candidate-runner-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-runner-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-runner-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-runner-smoke failed"
fi

[[ -s "$LOG" ]] || die "candidate runner JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_execution.v1"' "$LOG" || die "candidate execution schema missing"
grep -q '"runner":"metabolism-self-emit"' "$LOG" || die "runner name missing"
grep -q '"runner_status":"succeeded"' "$LOG" || die "runner success missing"
grep -q '"runner_status":"timed_out"' "$LOG" || die "runner timeout missing"
grep -q '"runner_timed_out":true' "$LOG" || die "runner timed-out flag missing"
grep -q '"runner_stdout_hash":"' "$LOG" || die "runner stdout hash missing"
grep -q '"generated_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"execution_id":"execution-' "$LOG" || die "candidate execution id missing"
grep -q '"passed":false' "$LOG" || die "timeout fail-closed receipt missing"
grep -q 'reason":"candidate runner timed out for shell shell-' "$LOG" || die "timeout reason missing"
grep -q 'runner=metabolism-self-emit runner_status=succeeded passed=true' "$RUN_LOG" || die "runner success line missing"
grep -q 'runner=metabolism-self-emit runner_status=timed_out passed=false' "$RUN_LOG" || die "runner timeout line missing"
grep -q '\[admission-live-route-turn-candidate-runner-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate runner smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-runner-smoke] pass: log=$LOG"
