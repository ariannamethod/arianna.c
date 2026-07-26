#!/usr/bin/env bash
# admission_live_route_turn_candidate_draft_review_smoke.sh - review a generated draft receipt.
#
# This is receipt-only: it proves a candidate draft can be reviewed against the
# human-turn route without converting the draft back into free-form surfaced text.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_REVIEW_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-draft-review.XXXXXX")}"
LOG="$WORKDIR/live_route_candidate_draft_review.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_draft_review.log"

die() {
    echo "[admission-live-route-turn-candidate-draft-review-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 100 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-draft-review-smoke"

echo "[admission-live-route-turn-candidate-draft-review-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-draft-review-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_REVIEW_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-draft-review-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-draft-review-smoke failed"
fi

[[ -s "$LOG" ]] || die "candidate draft review JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_candidate_review.v1"' "$LOG" || die "review schema missing"
grep -q '"candidate_draft_id":"draft-' "$LOG" || die "candidate draft id missing"
grep -q '"generator_adapter_id":"adapter-' "$LOG" || die "generator adapter id missing"
grep -q '"candidate_text_status":"generated"' "$LOG" || die "generated text status missing"
grep -q '"candidate_text_hash":"' "$LOG" || die "candidate text hash missing"
grep -q '"turn_prompt_class":"identity"' "$LOG" || die "identity turn class missing"
grep -q '"turn_expected_source":"chorus"' "$LOG" || die "turn expected chorus missing"
grep -q '"candidate_source":"chorus"' "$LOG" || die "matched chorus draft missing"
grep -q '"candidate_source":"direct"' "$LOG" || die "mismatched direct draft missing"
grep -q '"matched":true' "$LOG" || die "matched draft review missing"
grep -q '"matched":false' "$LOG" || die "failed draft review missing"
grep -q 'candidate_source_mismatch: source direct does not match turn expected chorus for prompt class identity' "$LOG" || die "draft mismatch reason missing"
grep -q 'candidate_draft_failed: generator adapter failed' "$LOG" || die "failed draft reason missing"
grep -q 'live-route candidate draft review: turn_class=identity expected=chorus draft=draft-' "$RUN_LOG" || die "matched draft review line missing"
grep -q '\[admission-live-route-turn-candidate-draft-review-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate draft review smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-draft-review-smoke] pass: log=$LOG"
