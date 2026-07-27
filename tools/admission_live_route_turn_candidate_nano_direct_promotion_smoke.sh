#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_promotion_smoke.sh - real nano direct -> pending live admission.
#
# Runs the existing nano-direct chat shadow chain, then requires both the
# shadow-ready decision receipt and its non-mutating promotion consumer receipt.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_PROMOTION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-promotion.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-promotion-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 180 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_CHAT_SHADOW_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG="$DECISION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG="$PROMOTION_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with promotion failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"
[[ -s "$PROMOTION_LOG" ]] || die "candidate admission promotion JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_decision.v1"' "$DECISION_LOG" || die "decision schema missing"
grep -q '"decision":"shadow_ready"' "$DECISION_LOG" || die "shadow-ready decision missing"
grep -q '"decision_id":"decision-' "$DECISION_LOG" || die "decision id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_promotion.v1"' "$PROMOTION_LOG" || die "promotion schema missing"
grep -q '"promotion":"pending_live_admission"' "$PROMOTION_LOG" || die "pending promotion missing"
grep -q '"admission_decision_id":"decision-' "$PROMOTION_LOG" || die "promotion decision id missing"
grep -q '"source_decision_passed":true' "$PROMOTION_LOG" || die "promotion did not consume a passed decision"
grep -q '"live_ready":true' "$PROMOTION_LOG" || die "promotion live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$PROMOTION_LOG" || die "live admission should remain disabled"
grep -q '"mutates_state":false' "$PROMOTION_LOG" || die "promotion must not mutate state"
grep -q '"passed":true' "$PROMOTION_LOG" || die "promotion did not pass"
grep -q '"promotion_id":"promotion-' "$PROMOTION_LOG" || die "promotion id missing"

grep -q 'live-route candidate admission promotion dry-run: class=dream route=direct source=direct decision=shadow_ready decision_id=decision-' "$RUN_LOG" || die "promotion chat line missing"
grep -q 'promotion=pending_live_admission promotion_id=promotion-' "$RUN_LOG" || die "promotion id line missing"
grep -q 'live_ready=true live_enabled=false mutates=false passed=true reason=shadow decision consumed; live admission still disabled' "$RUN_LOG" || die "promotion pass verdict missing"

echo "[admission-live-route-turn-candidate-nano-direct-promotion-smoke] pass: decision=$DECISION_LOG promotion=$PROMOTION_LOG"
