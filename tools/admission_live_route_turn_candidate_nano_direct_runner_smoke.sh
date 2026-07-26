#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_runner_smoke.sh - real nano direct bounded runner.
#
# Runs one live nano-Arianna direct generation behind the candidate execution
# receipt, then proves a non-direct shell fails closed before the nano starts.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_RUNNER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-runner.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_nano_direct_runner.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_runner.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-runner-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
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
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-nano-direct-runner-smoke"
[[ -x "$ROOT/nano-arianna" ]] || die "missing executable nano-arianna; run make admission-live-route-turn-candidate-nano-direct-runner-smoke"

model_file="$(resolve_model_path "${A2A_NANO_MODEL:-weights/nano_arianna_f16.gguf}")"
[[ -f "$model_file" ]] || die "model file missing: $model_file"

echo "[admission-live-route-turn-candidate-nano-direct-runner-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-nano-direct-runner-smoke] scratch=$WORKDIR"
echo "[admission-live-route-turn-candidate-nano-direct-runner-smoke] model=$model_file"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_RUNNER=nano-direct \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_TIMEOUT_MS="${A2A_NANO_DIRECT_RUNNER_TIMEOUT_MS:-30000}" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG="$LOG" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_BIN="$ROOT/nano-arianna" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_MODEL="$model_file" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_MAX_TOKENS="${A2A_NANO_DIRECT_MAX_TOKENS:-24}" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_TEMP="${A2A_NANO_DIRECT_TEMP:-0.9}" \
    AM_LIVE_ROUTE_TURN_NANO_DIRECT_TOP_P="${A2A_NANO_DIRECT_TOP_P:-0.92}" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-nano-direct-runner-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-nano-direct-runner-smoke failed"
fi

[[ -s "$LOG" ]] || die "nano-direct candidate runner JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_execution.v1"' "$LOG" || die "candidate execution schema missing"
grep -q '"runner":"nano-direct"' "$LOG" || die "runner name missing"
grep -q '"runner_status":"succeeded"' "$LOG" || die "runner success missing"
grep -q '"runner_status":"failed"' "$LOG" || die "runner fail-closed route reject missing"
grep -q '"backend":"nano-arianna"' "$LOG" || die "nano backend missing"
grep -q '"route":"direct"' "$LOG" || die "direct route missing"
grep -q '"generated_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"execution_id":"execution-' "$LOG" || die "candidate execution id missing"
grep -q '"runner_stdout_hash":"' "$LOG" || die "runner stdout hash missing"
grep -q 'candidate nano-direct runner only supports direct route' "$LOG" || die "non-direct route rejection missing"
grep -q 'runner=nano-direct runner_status=succeeded passed=true' "$RUN_LOG" || die "runner success line missing"
grep -q 'runner=nano-direct runner_status=failed passed=false' "$RUN_LOG" || die "runner fail-closed line missing"
grep -q '\[admission-live-route-turn-candidate-nano-direct-runner-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "nano-direct runner smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-nano-direct-runner-smoke] pass: log=$LOG"
