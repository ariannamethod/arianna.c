#!/usr/bin/env bash
# admission_live_route_turn_candidate_admission_adapter_smoke.sh - adapt handoffs into admission candidates.
#
# This remains shadow-only: it proves a verified candidate admission handoff can
# become the exact dreamCandidate consumed by the normal admission policy.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-candidate-admission-adapter.XXXXXX")}"
ADAPTER_LOG="$WORKDIR/live_route_candidate_admission_adapter.jsonl"
ADMISSION_LOG="$WORKDIR/dream_admission_from_handoff.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_admission_adapter.log"

die() {
    echo "[admission-live-route-turn-candidate-admission-adapter-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-candidate-admission-adapter-smoke"

echo "[admission-live-route-turn-candidate-admission-adapter-smoke] root=$ROOT"
echo "[admission-live-route-turn-candidate-admission-adapter-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_LIVE_ROUTE_TURN_CANDIDATE_DRAFT_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ADAPTER_LOG="$ADAPTER_LOG" \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN=1 \
    AM_DREAM_ADMISSION_LOG="$ADMISSION_LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-candidate-admission-adapter-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-candidate-admission-adapter-smoke failed"
fi

[[ -s "$ADAPTER_LOG" ]] || die "candidate admission adapter JSONL log not written"
[[ -s "$ADMISSION_LOG" ]] || die "shadow admission JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_adapter.v1"' "$ADAPTER_LOG" || die "candidate admission adapter schema missing"
grep -q '"timing":"admission_candidate_adapter"' "$ADAPTER_LOG" || die "adapter timing missing"
grep -q '"handoff_id":"handoff-' "$ADAPTER_LOG" || die "handoff id missing"
grep -q '"admission_adapter_id":"admission-adapter-' "$ADAPTER_LOG" || die "admission adapter id missing"
grep -q '"dream_candidate_run_id":"' "$ADAPTER_LOG" || die "dream candidate run id missing"
grep -q '"candidate_text_status":"generated"' "$ADAPTER_LOG" || die "generated text status missing"
grep -q '"candidate_text_hash":"' "$ADAPTER_LOG" || die "candidate text hash missing"
grep -q '"passed":true' "$ADAPTER_LOG" || die "passed adapter missing"
grep -q '"passed":false' "$ADAPTER_LOG" || die "failed adapter missing"
grep -q 'candidate_admission_handoff_failed: candidate_review_failed' "$ADAPTER_LOG" || die "failed handoff reason missing"
grep -q 'candidate_admission_handoff_id_mismatch' "$ADAPTER_LOG" || die "tampered handoff reason missing"

grep -q 'live-route candidate admission adapter: class=identity route=chorus source=chorus handoff=handoff-' "$RUN_LOG" || die "matched adapter line missing"
grep -q '\[admission-live-route-turn-candidate-admission-adapter-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

grep -q '"schema":"arianna.dream_candidate.v1"' "$ADMISSION_LOG" || die "dream candidate schema missing"
grep -q '"accepted":false' "$ADMISSION_LOG" || die "shadow admission should not accept"
grep -q '"reason":"shadow mode"' "$ADMISSION_LOG" || die "shadow mode reason missing"
grep -q '"live_route_candidate_admission":{' "$ADMISSION_LOG" || die "candidate admission provenance missing from dream receipt"
grep -q '"schema":"arianna.live_route_turn_candidate_admission_adapter.v1"' "$ADMISSION_LOG" || die "adapter schema missing from dream receipt"
grep -q '"admission_policy":{' "$ADMISSION_LOG" || die "admission policy missing from dream receipt"
grep -q '"live_route_choice":{' "$ADMISSION_LOG" || die "live route choice missing from dream receipt"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "candidate admission adapter smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-candidate-admission-adapter-smoke] pass: log=$ADAPTER_LOG admission_log=$ADMISSION_LOG"
