#!/usr/bin/env bash
# admission_live_route_turn_route_boundary_smoke.sh - route-unavailable boundary propagation.
#
# Proves the inventory-gated route refusal is carried as typed receipt fields
# through generation, draft, review, and admission dry-runs without assigning
# runnable ids.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_ROUTE_BOUNDARY_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary.XXXXXX")}"
BODYDIR="$WORKDIR/body"
JOB_LOG="$WORKDIR/live_route_generation_job.jsonl"
SHELL_LOG="$WORKDIR/live_route_candidate_shell.jsonl"
EXECUTION_LOG="$WORKDIR/live_route_candidate_execution.jsonl"
ADAPTER_LOG="$WORKDIR/live_route_generator_adapter.jsonl"
DRAFT_LOG="$WORKDIR/live_route_candidate_draft.jsonl"
REVIEW_LOG="$WORKDIR/live_route_candidate_review.jsonl"
ADMISSION_LOG="$WORKDIR/live_route_candidate_admission.jsonl"
ADMISSION_ADAPTER_LOG="$WORKDIR/live_route_candidate_admission_adapter.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_boundary.log"

die() {
    echo "[admission-live-route-turn-route-boundary-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$BODYDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-route-boundary-smoke"

echo "[admission-live-route-turn-route-boundary-smoke] root=$ROOT"
echo "[admission-live-route-turn-route-boundary-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_BODY_INVENTORY_ROOT="$BODYDIR" \
    A2A_JANUS_MODEL="weights/arianna_v4_sft_f16.gguf" \
    A2A_RESONANCE_MODEL="weights/arianna_resonance_v3_f16.gguf" \
    A2A_NANO_MODEL="weights/nano_arianna_f16.gguf" \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_INVENTORY_GATE=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG="$JOB_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_SHELL_LOG="$SHELL_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_EXECUTION_LOG="$EXECUTION_LOG" \
    AM_LIVE_ROUTE_TURN_GENERATOR_ADAPTER_LOG="$ADAPTER_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_LOG="$DRAFT_LOG" \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$REVIEW_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG="$ADMISSION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG="$ADMISSION_ADAPTER_LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-route-boundary-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-route-boundary-smoke failed"
fi

for log in "$JOB_LOG" "$SHELL_LOG" "$EXECUTION_LOG" "$ADAPTER_LOG" "$DRAFT_LOG" "$REVIEW_LOG" "$ADMISSION_LOG" "$ADMISSION_ADAPTER_LOG"; do
    [[ -s "$log" ]] || die "receipt log not written: $log"
    [[ "$(wc -l < "$log" | tr -d ' ')" == "1" ]] || die "expected one receipt in $log"
    grep -q '"body_inventory_status":"blocked"' "$log" || die "body inventory status missing in $log"
    grep -q '"route_availability_status":"unavailable"' "$log" || die "route availability status missing in $log"
    grep -q '"route_availability_reason":"missing_route_organs:chorus-binary,nano-weight"' "$log" || die "route availability reason missing in $log"
    grep -q '"route_missing_organs":\["chorus-binary","nano-weight"\]' "$log" || die "missing route organs not propagated in $log"
    if [[ "$log" == "$REVIEW_LOG" ]]; then
        grep -q '"matched":false' "$log" || die "review receipt should fail closed in $log"
    else
        grep -q '"passed":false' "$log" || die "receipt should fail closed in $log"
    fi
    if grep -q '"job_id":"job-' "$log"; then die "failed route boundary must not name job id in $log"; fi
    if grep -q '"shell_id":"shell-' "$log"; then die "failed route boundary must not name shell id in $log"; fi
    if grep -q '"execution_id":"execution-' "$log"; then die "failed route boundary must not name execution id in $log"; fi
    if grep -q '"candidate_execution_id":"execution-' "$log"; then die "failed route boundary must not name candidate execution id in $log"; fi
    if grep -q '"adapter_id":"adapter-' "$log"; then die "failed route boundary must not name adapter id in $log"; fi
    if grep -q '"generator_adapter_id":"adapter-' "$log"; then die "failed route boundary must not name generator adapter id in $log"; fi
    if grep -q '"draft_id":"draft-' "$log"; then die "failed route boundary must not name draft id in $log"; fi
    if grep -q '"candidate_draft_id":"draft-' "$log"; then die "failed route boundary must not name candidate draft id in $log"; fi
    if grep -q '"handoff_id":"handoff-' "$log"; then die "failed route boundary must not name handoff id in $log"; fi
    if grep -q '"admission_adapter_id":"admission-adapter-' "$log"; then die "failed route boundary must not name admission adapter id in $log"; fi
done

grep -q 'live-route generation job dry-run: class=identity route=chorus' "$RUN_LOG" || die "generation job line missing"
grep -q 'live-route candidate shell dry-run: class=identity route=chorus' "$RUN_LOG" || die "candidate shell line missing"
grep -q 'live-route candidate execution dry-run: class=identity route=chorus' "$RUN_LOG" || die "candidate execution line missing"
grep -q 'live-route generator adapter dry-run: class=identity route=chorus' "$RUN_LOG" || die "generator adapter line missing"
grep -q 'live-route candidate draft dry-run: class=identity route=chorus' "$RUN_LOG" || die "candidate draft line missing"
grep -q 'live-route candidate admission handoff dry-run: class=identity route=chorus' "$RUN_LOG" || die "candidate admission handoff line missing"
grep -q 'live-route candidate admission adapter dry-run: class=identity route=chorus' "$RUN_LOG" || die "candidate admission adapter line missing"
grep -q '\[admission-live-route-turn-route-boundary-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "route boundary smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-route-boundary-smoke] pass: job=$JOB_LOG shell=$SHELL_LOG execution=$EXECUTION_LOG adapter=$ADAPTER_LOG draft=$DRAFT_LOG review=$REVIEW_LOG admission=$ADMISSION_LOG admission_adapter=$ADMISSION_ADAPTER_LOG"
