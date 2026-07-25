#!/usr/bin/env bash
# admission_live_route_turn_generator_adapter_smoke.sh - bounded live route generator text boundary.
#
# Binds generator-produced text to a frozen candidate shell receipt without
# running weights or mutating organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-generator-adapter.XXXXXX")}"
LOG="$WORKDIR/live_route_generator_adapter.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_generator_adapter.log"

die() {
    echo "[admission-live-route-turn-generator-adapter-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-generator-adapter-smoke"

echo "[admission-live-route-turn-generator-adapter-smoke] root=$ROOT"
echo "[admission-live-route-turn-generator-adapter-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-generator-adapter-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-generator-adapter-smoke failed"
fi

[[ -s "$LOG" ]] || die "generator adapter JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_generator_adapter.v1"' "$LOG" || die "generator adapter schema missing"
grep -q '"candidate_schema":"arianna.dream_candidate.v1"' "$LOG" || die "dream candidate schema missing"
grep -q '"candidate_text_status":"pending_generation"' "$LOG" || die "pending shell status missing"
grep -q '"generated_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"generated_text_hash":"' "$LOG" || die "generated text hash missing"
grep -q '"candidate_kind":"chorus"' "$LOG" || die "chorus candidate kind missing"
grep -q '"candidate_kind":"user_bridge"' "$LOG" || die "user bridge candidate kind missing"
grep -q '"candidate_kind":"qloop_target"' "$LOG" || die "qloop target candidate kind missing"
grep -q '"candidate_kind":"direct"' "$LOG" || die "direct candidate kind missing"
grep -q '"backend":"chorus-arianna"' "$LOG" || die "chorus backend missing"
grep -q '"backend":"nano-arianna"' "$LOG" || die "nano backend missing"
grep -q '"entrypoint":"field"' "$LOG" || die "field entrypoint missing"
grep -q '"entrypoint":"repl_user_bridge"' "$LOG" || die "user bridge entrypoint missing"
grep -q '"entrypoint":"qloop_target"' "$LOG" || die "qloop target entrypoint missing"
grep -q '"entrypoint":"direct"' "$LOG" || die "direct entrypoint missing"
grep -q '"prompt_frame":"user_arianna_target"' "$LOG" || die "qloop target prompt frame missing"
grep -q '"prompt_frame":"user_arianna"' "$LOG" || die "user bridge prompt frame missing"
grep -q '"candidate_trigger":"chorus-identity"' "$LOG" || die "chorus identity trigger missing"
grep -q '"candidate_trigger":"user_bridge-cold-reader"' "$LOG" || die "user_bridge cold-reader trigger missing"
grep -q '"candidate_trigger":"qloop_target-recipient-lock"' "$LOG" || die "qloop_target recipient-lock trigger missing"
grep -q '"candidate_trigger":"direct-dream"' "$LOG" || die "direct dream trigger missing"
grep -q '"candidate_seed":"turn-' "$LOG" || die "turn-derived candidate seed missing"
grep -q '"job_id":"job-' "$LOG" || die "generation job id missing"
grep -q '"shell_id":"shell-' "$LOG" || die "candidate shell id missing"
grep -q '"adapter_id":"adapter-' "$LOG" || die "generator adapter id missing"
grep -q '"passed":false' "$LOG" || die "unknown fail-closed generator adapter missing"
grep -q 'live-route generator adapter dry-run: class=identity route=chorus backend=chorus-arianna entry=field frame=q_a shell=shell-' "$RUN_LOG" || die "identity generator adapter line missing"
grep -q '\[admission-live-route-turn-generator-adapter-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "generator adapter smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-generator-adapter-smoke] pass: log=$LOG"
