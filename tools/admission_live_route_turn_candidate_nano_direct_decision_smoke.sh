#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_decision_smoke.sh - real nano direct -> shadow-ready decision.
#
# Runs the existing nano-direct chat shadow chain, then requires the extra
# admission decision receipt that proves the candidate is live-ready while still
# mutating no organism state.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_DECISION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-decision.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-decision-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 160 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_CHAT_SHADOW_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG="$DECISION_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with decision failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_decision.v1"' "$DECISION_LOG" || die "decision schema missing"
grep -q '"decision":"shadow_ready"' "$DECISION_LOG" || die "shadow-ready decision missing"
grep -q '"live_ready":true' "$DECISION_LOG" || die "live-ready verdict missing"
grep -q '"mutates_state":false' "$DECISION_LOG" || die "decision must not mutate state"
grep -q '"passed":true' "$DECISION_LOG" || die "decision did not pass"
grep -q '"decision_id":"decision-' "$DECISION_LOG" || die "decision id missing"

grep -q 'live-route candidate admission decision dry-run: class=dream route=direct source=direct handoff=handoff-' "$RUN_LOG" || die "decision chat line missing"
grep -q 'decision=shadow_ready decision_id=decision-' "$RUN_LOG" || die "decision id line missing"
grep -q 'live_ready=true mutates=false passed=true reason=shadow ready; live mutation still disabled' "$RUN_LOG" || die "decision pass verdict missing"

echo "[admission-live-route-turn-candidate-nano-direct-decision-smoke] pass: decision=$DECISION_LOG"
