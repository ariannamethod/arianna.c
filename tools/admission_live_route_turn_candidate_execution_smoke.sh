#!/usr/bin/env bash
# admission_live_route_turn_candidate_execution_smoke.sh - bounded live route candidate execution.
#
# Converts pending candidate shell receipts into execution receipts without
# running weights or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-execution.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_execution.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_execution.log"

die() {
    echo "[admission-live-route-turn-candidate-execution-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-execution-smoke"

echo "[admission-live-route-turn-candidate-execution-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-execution-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-execution-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-execution-smoke failed"
fi

[[ -s "$LOG" ]] || die "candidate execution JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_execution.v1"' "$LOG" || die "candidate execution schema missing"
grep -q '"candidate_schema":"arianna.dream_candidate.v1"' "$LOG" || die "dream candidate schema missing"
grep -q '"candidate_text_status":"pending_generation"' "$LOG" || die "pending shell status missing"
grep -q '"generated_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"generated_text_hash":"' "$LOG" || die "generated text hash missing"
grep -q '"timeout_ms":12000' "$LOG" || die "default timeout missing"
grep -q '"candidate_kind":"chorus"' "$LOG" || die "chorus candidate kind missing"
grep -q '"candidate_kind":"user_bridge"' "$LOG" || die "user bridge candidate kind missing"
grep -q '"candidate_kind":"qloop_target"' "$LOG" || die "qloop target candidate kind missing"
grep -q '"candidate_kind":"direct"' "$LOG" || die "direct candidate kind missing"
grep -q '"executor":"chorus-arianna:field:q_a"' "$LOG" || die "chorus executor missing"
grep -q '"executor":"chorus-arianna:repl_user_bridge:user_arianna"' "$LOG" || die "user bridge executor missing"
grep -q '"executor":"chorus-arianna:qloop_target:user_arianna_target"' "$LOG" || die "qloop target executor missing"
grep -q '"executor":"nano-arianna:direct:q_a"' "$LOG" || die "direct executor missing"
grep -q '"candidate_trigger":"chorus-identity"' "$LOG" || die "chorus identity trigger missing"
grep -q '"candidate_trigger":"user_bridge-cold-reader"' "$LOG" || die "user_bridge cold-reader trigger missing"
grep -q '"candidate_trigger":"qloop_target-recipient-lock"' "$LOG" || die "qloop_target recipient-lock trigger missing"
grep -q '"candidate_trigger":"direct-dream"' "$LOG" || die "direct dream trigger missing"
grep -q '"candidate_seed":"turn-' "$LOG" || die "turn-derived candidate seed missing"
grep -q '"job_id":"job-' "$LOG" || die "generation job id missing"
grep -q '"shell_id":"shell-' "$LOG" || die "candidate shell id missing"
grep -q '"execution_id":"execution-' "$LOG" || die "candidate execution id missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed candidate execution missing"
grep -q 'live-route candidate execution dry-run: class=identity route=chorus backend=chorus-arianna entry=field frame=q_a executor=chorus-arianna:field:q_a timeout_ms=12000 shell=shell-' "$RUN_LOG" || die "identity candidate execution line missing"
grep -q '\[admission-live-route-turn-candidate-execution-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate execution smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-execution-smoke] pass: log=$LOG"
