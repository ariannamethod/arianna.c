#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_enable_gate_smoke.sh - real nano direct -> closed live enable gate.
#
# Runs the nano-direct chat shadow chain through decision, promotion, switch, and
# enable-gate receipts. The enable gate must stay closed without the operator
# confirmation key, and it must remain non-mutating.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_ENABLE_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-enable-gate.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-enable-gate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 260 "$RUN_LOG" >&2 || true
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
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_LOG="$ENABLE_GATE_LOG" \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY= \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with enable gate failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"
[[ -s "$PROMOTION_LOG" ]] || die "candidate admission promotion JSONL log not written"
[[ -s "$SWITCH_LOG" ]] || die "candidate admission switch JSONL log not written"
[[ -s "$ENABLE_GATE_LOG" ]] || die "candidate admission enable gate JSONL log not written"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_decision.v1"' "$DECISION_LOG" || die "decision schema missing"
grep -q '"decision":"shadow_ready"' "$DECISION_LOG" || die "shadow-ready decision missing"
grep -q '"decision_id":"decision-' "$DECISION_LOG" || die "decision id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_promotion.v1"' "$PROMOTION_LOG" || die "promotion schema missing"
grep -q '"promotion":"pending_live_admission"' "$PROMOTION_LOG" || die "pending promotion missing"
grep -q '"promotion_id":"promotion-' "$PROMOTION_LOG" || die "promotion id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_switch.v1"' "$SWITCH_LOG" || die "switch schema missing"
grep -q '"switch_state":"disabled"' "$SWITCH_LOG" || die "switch should stay disabled"
grep -q '"switch_action":"hold_pending_live_admission"' "$SWITCH_LOG" || die "switch action missing"
grep -q '"switch_id":"switch-' "$SWITCH_LOG" || die "switch id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_enable_gate.v1"' "$ENABLE_GATE_LOG" || die "enable gate schema missing"
grep -q '"enable_state":"disabled"' "$ENABLE_GATE_LOG" || die "enable gate should stay disabled"
grep -q '"enable_action":"require_operator_key"' "$ENABLE_GATE_LOG" || die "enable gate action missing"
grep -q '"admission_switch_id":"switch-' "$ENABLE_GATE_LOG" || die "enable gate switch id missing"
grep -q '"source_switch_passed":true' "$ENABLE_GATE_LOG" || die "enable gate did not consume a passed switch"
grep -q '"live_ready":true' "$ENABLE_GATE_LOG" || die "enable gate live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$ENABLE_GATE_LOG" || die "live admission should remain disabled"
grep -q '"admission_allowed":false' "$ENABLE_GATE_LOG" || die "enable gate should not allow admission"
grep -q '"manual_enable_requested":false' "$ENABLE_GATE_LOG" || die "manual enable should not be requested"
grep -q '"enable_key_matched":false' "$ENABLE_GATE_LOG" || die "enable key should not match"
grep -q '"mutates_state":false' "$ENABLE_GATE_LOG" || die "enable gate must not mutate state"
grep -q '"passed":true' "$ENABLE_GATE_LOG" || die "enable gate did not pass closed"
grep -q '"enable_gate_id":"enable-' "$ENABLE_GATE_LOG" || die "enable gate id missing"

grep -q 'live-route candidate admission enable gate dry-run: class=dream route=direct source=direct switch=disabled switch_id=switch-' "$RUN_LOG" || die "enable gate chat line missing"
grep -q 'enable=disabled enable_action=require_operator_key enable_id=enable-' "$RUN_LOG" || die "enable gate id line missing"
grep -q 'admission_allowed=false manual_enable=false key_matched=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission enable gate closed; operator key absent' "$RUN_LOG" || die "enable gate closed verdict missing"

echo "[admission-live-route-turn-candidate-nano-direct-enable-gate-smoke] pass: decision=$DECISION_LOG promotion=$PROMOTION_LOG switch=$SWITCH_LOG enable_gate=$ENABLE_GATE_LOG"
