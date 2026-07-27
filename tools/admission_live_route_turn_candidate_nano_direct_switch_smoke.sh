#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_switch_smoke.sh - real nano direct -> disabled live switch guard.
#
# Runs the existing nano-direct chat shadow chain, then requires decision,
# promotion, and switch receipts. The switch may acknowledge readiness but must
# keep live admission disabled and non-mutating.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_SWITCH_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-switch.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-switch-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 220 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_CHAT_SHADOW_WORKDIR="$WORKDIR" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_DECISION_LOG="$DECISION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_PROMOTION_LOG="$PROMOTION_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_SWITCH_LOG="$SWITCH_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with switch failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"
[[ -s "$PROMOTION_LOG" ]] || die "candidate admission promotion JSONL log not written"
[[ -s "$SWITCH_LOG" ]] || die "candidate admission switch JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_decision.v1"' "$DECISION_LOG" || die "decision schema missing"
grep -q '"decision":"shadow_ready"' "$DECISION_LOG" || die "shadow-ready decision missing"
grep -q '"decision_id":"decision-' "$DECISION_LOG" || die "decision id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_promotion.v1"' "$PROMOTION_LOG" || die "promotion schema missing"
grep -q '"promotion":"pending_live_admission"' "$PROMOTION_LOG" || die "pending promotion missing"
grep -q '"promotion_id":"promotion-' "$PROMOTION_LOG" || die "promotion id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_switch.v1"' "$SWITCH_LOG" || die "switch schema missing"
grep -q '"switch_state":"disabled"' "$SWITCH_LOG" || die "switch should stay disabled"
grep -q '"switch_action":"hold_pending_live_admission"' "$SWITCH_LOG" || die "switch action missing"
grep -q '"admission_promotion_id":"promotion-' "$SWITCH_LOG" || die "switch promotion id missing"
grep -q '"source_promotion_passed":true' "$SWITCH_LOG" || die "switch did not consume a passed promotion"
grep -q '"live_ready":true' "$SWITCH_LOG" || die "switch live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$SWITCH_LOG" || die "live admission should remain disabled"
grep -q '"admission_allowed":false' "$SWITCH_LOG" || die "switch should not allow admission"
grep -q '"mutates_state":false' "$SWITCH_LOG" || die "switch must not mutate state"
grep -q '"passed":true' "$SWITCH_LOG" || die "switch guard did not pass"
grep -q '"switch_id":"switch-' "$SWITCH_LOG" || die "switch id missing"

grep -q 'live-route candidate admission switch dry-run: class=dream route=direct source=direct promotion=pending_live_admission promotion_id=promotion-' "$RUN_LOG" || die "switch chat line missing"
grep -q 'switch=disabled switch_action=hold_pending_live_admission switch_id=switch-' "$RUN_LOG" || die "switch id line missing"
grep -q 'admission_allowed=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission switch disabled; pending promotion held without mutation' "$RUN_LOG" || die "switch pass verdict missing"

echo "[admission-live-route-turn-candidate-nano-direct-switch-smoke] pass: decision=$DECISION_LOG promotion=$PROMOTION_LOG switch=$SWITCH_LOG"
