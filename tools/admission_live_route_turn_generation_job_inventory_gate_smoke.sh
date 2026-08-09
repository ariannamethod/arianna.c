#!/usr/bin/env bash
# admission_live_route_turn_generation_job_inventory_gate_smoke.sh - inventory-gated generation job refusal.
#
# Proves route generation job dry-run consults body inventory and fails closed
# before assigning a runnable job id when the selected route organs are absent.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_GENERATION_JOB_INVENTORY_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-generation-job-inventory.XXXXXX")}"
BODYDIR="$WORKDIR/body"
LOG="$WORKDIR/live_route_generation_job_inventory_gate.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_generation_job_inventory_gate.log"

die() {
    echo "[admission-live-route-turn-generation-job-inventory-gate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$BODYDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-generation-job-inventory-gate-smoke"

echo "[admission-live-route-turn-generation-job-inventory-gate-smoke] root=$ROOT"
echo "[admission-live-route-turn-generation-job-inventory-gate-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_BODY_INVENTORY_ROOT="$BODYDIR" \
    A2A_JANUS_MODEL="weights/arianna_v4_sft_f16.gguf" \
    A2A_RESONANCE_MODEL="weights/arianna_resonance_v3_f16.gguf" \
    A2A_NANO_MODEL="weights/nano_arianna_f16.gguf" \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_INVENTORY_GATE=1 \
    AM_LIVE_ROUTE_TURN_GENERATION_JOB_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-generation-job-inventory-gate-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-generation-job-inventory-gate-smoke failed"
fi

[[ -s "$LOG" ]] || die "generation job inventory gate JSONL log not written"
grep -q '"schema":"arianna.live_route_turn_generation_job.v1"' "$LOG" || die "generation job schema missing"
grep -q '"body_inventory_status":"blocked"' "$LOG" || die "body inventory status missing"
grep -q '"route_availability_status":"unavailable"' "$LOG" || die "route availability status missing"
grep -q '"route_missing_organs":\["chorus-binary","nano-weight"\]' "$LOG" || die "missing route organs not named"
grep -q '"reason":"route chorus unavailable in body inventory: missing_route_organs:chorus-binary,nano-weight"' "$LOG" || die "route refusal reason missing"
grep -q '"passed":false' "$LOG" || die "inventory-gated job must fail closed"
if grep -q '"job_id":"job-' "$LOG"; then
    die "inventory-gated failed job must not name a runnable job id"
fi
grep -q 'live-route generation job dry-run: class=identity route=chorus backend=chorus-arianna entry=field' "$RUN_LOG" || die "dry-run line missing"
grep -q 'passed=false reason=route chorus unavailable in body inventory' "$RUN_LOG" || die "dry-run line did not report inventory refusal"
grep -q '\[admission-live-route-turn-generation-job-inventory-gate-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "inventory-gated generation job smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-generation-job-inventory-gate-smoke] pass: log=$LOG"
