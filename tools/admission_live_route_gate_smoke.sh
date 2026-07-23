#!/usr/bin/env bash
# admission_live_route_gate_smoke.sh - non-mutating route-plan admission gate check.
#
# Runs the metabolism binary in shadow admission mode with the live route-plan
# gate enabled, then verifies both a matching prompt-class route and a wrong
# source fail-closed receipt.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_GATE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-gate.XXXXXX")}"
LOG="$WORKDIR/dream_admission_live_route_gate.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_gate.log"

die() {
    echo "[admission-live-route-gate-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-gate-smoke"

echo "[admission-live-route-gate-smoke] root=$ROOT"
echo "[admission-live-route-gate-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_ALLOWED_SOURCES= \
    AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN=1 \
    AM_DREAM_ADMISSION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-gate-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-gate-smoke failed"
fi

[[ -s "$LOG" ]] || die "admission JSONL log not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"mode":"shadow"' "$LOG" || die "shadow mode missing"
grep -q '"accepted":false' "$LOG" || die "shadow candidates were not rejected"
grep -q '"live_route_plan":{' "$LOG" || die "live route plan missing"
grep -q '"schema":"arianna.live_route_plan.v1"' "$LOG" || die "live route plan schema missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "identity prompt class missing"
grep -q '"route":"chorus"' "$LOG" || die "chorus route missing"
grep -q '"passed":true' "$LOG" || die "matching route policy did not pass"
grep -q '"passed":false' "$LOG" || die "wrong-source route policy did not fail"
grep -q '"source direct does not match live route chorus for prompt class identity"' "$LOG" || die "wrong-source route-plan reason missing"
grep -q '\[admission-live-route-gate-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "live route gate smoke wrote durable organism state"
fi

echo "[admission-live-route-gate-smoke] pass: log=$LOG"
