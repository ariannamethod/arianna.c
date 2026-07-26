#!/usr/bin/env bash
# admission_live_route_turn_candidate_admission_smoke.sh - hand reviewed drafts to admission.
#
# This is receipt-only: it proves a matched draft review can name an admission
# handoff without mutating organism state or bypassing draft/adapter provenance.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-admission.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_admission.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_admission.log"

die() {
    echo "[admission-live-route-turn-candidate-admission-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 100 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-admission-smoke"

echo "[admission-live-route-turn-candidate-admission-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-admission-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-admission-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-admission-smoke failed"
fi

[[ -s "$LOG" ]] || die "candidate admission JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_admission.v1"' "$LOG" || die "candidate admission schema missing"
grep -q '"timing":"pre_admission_handoff"' "$LOG" || die "handoff timing missing"
grep -q '"candidate_schema":"arianna.dream_candidate.v1"' "$LOG" || die "dream candidate schema missing"
grep -q '"candidate_draft_id":"draft-' "$LOG" || die "candidate draft id missing"
grep -q '"generator_adapter_id":"adapter-' "$LOG" || die "generator adapter id missing"
grep -q '"candidate_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"candidate_text_hash":"' "$LOG" || die "candidate text hash missing"
grep -q '"handoff_id":"handoff-' "$LOG" || die "handoff id missing"
grep -q '"review_matched":true' "$LOG" || die "matched review flag missing"
grep -q '"passed":true' "$LOG" || die "passed handoff missing"
grep -q '"passed":false' "$LOG" || die "failed handoff missing"
grep -q 'candidate_review_failed: candidate_source_mismatch' "$LOG" || die "review mismatch reason missing"
grep -q 'candidate_draft_failed: generator adapter failed' "$LOG" || die "failed draft reason missing"
grep -q 'live-route candidate admission handoff: class=identity route=chorus source=chorus draft=draft-' "$RUN_LOG" || die "matched admission handoff line missing"
grep -q '\[admission-live-route-turn-candidate-admission-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate admission smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-admission-smoke] pass: log=$LOG"
