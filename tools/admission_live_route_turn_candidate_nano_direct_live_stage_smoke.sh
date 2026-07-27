#!/usr/bin/env bash
# admission_live_route_turn_candidate_nano_direct_live_stage_smoke.sh - real nano direct -> live stage dry-run.
#
# Runs the nano-direct chat shadow chain through decision, promotion, switch,
# enable-gate, and live-stage receipts. The explicit confirmation key may only
# stage a candidate as dry-run; writer/rollback are still absent and no live
# admission or state mutation is allowed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_CANDIDATE_NANO_DIRECT_LIVE_STAGE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-nano-direct-live-stage.XXXXXX")}"
DECISION_LOG="$WORKDIR/live_route_candidate_admission_decision_nano_direct.jsonl"
PROMOTION_LOG="$WORKDIR/live_route_candidate_admission_promotion_nano_direct.jsonl"
SWITCH_LOG="$WORKDIR/live_route_candidate_admission_switch_nano_direct.jsonl"
ENABLE_GATE_LOG="$WORKDIR/live_route_candidate_admission_enable_gate_nano_direct.jsonl"
LIVE_STAGE_LOG="$WORKDIR/live_route_candidate_admission_live_stage_nano_direct.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_candidate_nano_direct_chat_shadow.log"

die() {
    echo "[admission-live-route-turn-candidate-nano-direct-live-stage-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 300 "$RUN_LOG" >&2 || true
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
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_ENABLE_GATE_KEY=ARIANNA_LIVE_ADMISSION_ENABLE_DRY_RUN_ONLY \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_CANDIDATE_ADMISSION_LIVE_STAGE_LOG="$LIVE_STAGE_LOG" \
    bash "$ROOT/tools/admission_live_route_turn_candidate_nano_direct_chat_shadow_smoke.sh"; then
    die "nano-direct chat shadow smoke with live stage failed"
fi

[[ -s "$DECISION_LOG" ]] || die "candidate admission decision JSONL log not written"
[[ -s "$PROMOTION_LOG" ]] || die "candidate admission promotion JSONL log not written"
[[ -s "$SWITCH_LOG" ]] || die "candidate admission switch JSONL log not written"
[[ -s "$ENABLE_GATE_LOG" ]] || die "candidate admission enable gate JSONL log not written"
[[ -s "$LIVE_STAGE_LOG" ]] || die "candidate admission live stage JSONL log not written"

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
grep -q '"enable_state":"armed_dry_run"' "$ENABLE_GATE_LOG" || die "enable gate should only arm dry-run"
grep -q '"enable_action":"would_enable_live_admission_dry_run"' "$ENABLE_GATE_LOG" || die "enable gate armed action missing"
grep -q '"admission_switch_id":"switch-' "$ENABLE_GATE_LOG" || die "enable gate switch id missing"
grep -q '"source_switch_passed":true' "$ENABLE_GATE_LOG" || die "enable gate did not consume a passed switch"
grep -q '"live_ready":true' "$ENABLE_GATE_LOG" || die "enable gate live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$ENABLE_GATE_LOG" || die "live admission should remain disabled"
grep -q '"admission_allowed":false' "$ENABLE_GATE_LOG" || die "enable gate should not allow admission"
grep -q '"manual_enable_requested":true' "$ENABLE_GATE_LOG" || die "manual enable should be requested"
grep -q '"enable_key_matched":true' "$ENABLE_GATE_LOG" || die "enable key should match"
grep -q '"mutates_state":false' "$ENABLE_GATE_LOG" || die "enable gate must not mutate state"
grep -q '"passed":true' "$ENABLE_GATE_LOG" || die "enable gate did not pass armed dry-run"
grep -q '"enable_gate_id":"enable-' "$ENABLE_GATE_LOG" || die "enable gate id missing"

grep -q '"schema":"arianna.live_route_turn_candidate_admission_live_stage.v1"' "$LIVE_STAGE_LOG" || die "live stage schema missing"
grep -q '"stage_state":"staged_dry_run"' "$LIVE_STAGE_LOG" || die "live stage should only stage dry-run"
grep -q '"stage_action":"stage_live_candidate_dry_run"' "$LIVE_STAGE_LOG" || die "live stage action missing"
grep -q '"admission_enable_gate_id":"enable-' "$LIVE_STAGE_LOG" || die "live stage enable gate id missing"
grep -q '"source_enable_passed":true' "$LIVE_STAGE_LOG" || die "live stage did not consume a passed enable gate"
grep -q '"live_ready":true' "$LIVE_STAGE_LOG" || die "live stage live-ready verdict missing"
grep -q '"live_admission_enabled":false' "$LIVE_STAGE_LOG" || die "live stage should not enable live admission"
grep -q '"admission_allowed":false' "$LIVE_STAGE_LOG" || die "live stage should not allow admission"
grep -q '"manual_enable_requested":true' "$LIVE_STAGE_LOG" || die "live stage should preserve manual enable"
grep -q '"enable_key_matched":true' "$LIVE_STAGE_LOG" || die "live stage should preserve key match"
grep -q '"requires_writer":true' "$LIVE_STAGE_LOG" || die "live stage should require writer"
grep -q '"writer_ready":false' "$LIVE_STAGE_LOG" || die "writer must remain absent"
grep -q '"requires_rollback":true' "$LIVE_STAGE_LOG" || die "live stage should require rollback"
grep -q '"rollback_ready":false' "$LIVE_STAGE_LOG" || die "rollback must remain absent"
grep -q '"mutates_state":false' "$LIVE_STAGE_LOG" || die "live stage must not mutate state"
grep -q '"passed":true' "$LIVE_STAGE_LOG" || die "live stage did not pass dry-run"
grep -q '"live_stage_id":"stage-' "$LIVE_STAGE_LOG" || die "live stage id missing"

grep -q 'live-route candidate admission enable gate dry-run: class=dream route=direct source=direct switch=disabled switch_id=switch-' "$RUN_LOG" || die "enable gate chat line missing"
grep -q 'enable=armed_dry_run enable_action=would_enable_live_admission_dry_run enable_id=enable-' "$RUN_LOG" || die "armed enable gate id line missing"
grep -q 'admission_allowed=false manual_enable=true key_matched=true live_ready=true live_enabled=false mutates=false passed=true reason=live admission enable key matched; dry-run still refuses mutation' "$RUN_LOG" || die "armed enable gate verdict missing"
grep -q 'live-route candidate admission live stage dry-run: class=dream route=direct source=direct enable=armed_dry_run enable_id=enable-' "$RUN_LOG" || die "live stage chat line missing"
grep -q 'stage=staged_dry_run stage_action=stage_live_candidate_dry_run stage_id=stage-' "$RUN_LOG" || die "live stage id line missing"
grep -q 'admission_allowed=false writer_ready=false rollback_ready=false live_ready=true live_enabled=false mutates=false passed=true reason=live admission candidate staged as dry-run; writer and rollback remain absent' "$RUN_LOG" || die "live stage dry-run verdict missing"

echo "[admission-live-route-turn-candidate-nano-direct-live-stage-smoke] pass: decision=$DECISION_LOG promotion=$PROMOTION_LOG switch=$SWITCH_LOG enable_gate=$ENABLE_GATE_LOG live_stage=$LIVE_STAGE_LOG"
