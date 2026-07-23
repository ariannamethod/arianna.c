#!/usr/bin/env bash
# admission_live_route_turn_bridge_admission_smoke.sh - prove bridged turn class reaches admission receipts.
#
# Review receipts already show the typed nano/human-turn debt. This smoke proves
# the dream admission receipt gets the same bridged live_route_choice while the
# original candidate remains source=nano, trigger=human-turn.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_TURN_BRIDGE_ADMISSION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-turn-bridge-admission.XXXXXX")}"
LOG="$WORKDIR/dream_admission_turn_bridge.jsonl"
RUN_LOG="$WORKDIR/admission_live_route_turn_bridge_admission.log"

die() {
    echo "[admission-live-route-turn-bridge-admission-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 100 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-live-route-turn-bridge-admission-smoke"

echo "[admission-live-route-turn-bridge-admission-smoke] root=$ROOT"
echo "[admission-live-route-turn-bridge-admission-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN=1 \
    AM_LIVE_ROUTE_TURN_BRIDGE_DRY_RUN=1 \
    AM_DREAM_ADMISSION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-live-route-turn-bridge-admission-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-live-route-turn-bridge-admission-smoke failed"
fi

[[ -s "$LOG" ]] || die "turn bridge admission JSONL log not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"source":"nano"' "$LOG" || die "original nano source missing"
grep -q '"trigger":"human-turn"' "$LOG" || die "original human-turn trigger missing"
grep -q '"live_route_choice_dry_run":true' "$LOG" || die "live route dry-run marker missing"
grep -q '"live_route_turn_bridge_applied":true' "$LOG" || die "turn bridge marker missing"
grep -q '"live_route_turn_bridge_trigger":"human-turn-identity"' "$LOG" || die "turn bridge trigger missing"
grep -q '"prompt_class":"identity"' "$LOG" || die "bridged identity prompt class missing"
grep -q '"route":"chorus"' "$LOG" || die "bridged chorus route missing"
grep -q '"source":"nano"' "$LOG" || die "choice nano source missing"
grep -q '"expected_source":"chorus"' "$LOG" || die "choice expected chorus missing"
grep -q 'source nano does not match live route chorus for prompt class identity' "$LOG" || die "source-bound route reason missing"
grep -q '\[admission-live-route-turn-bridge-admission-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "turn bridge admission smoke wrote durable organism state"
fi

echo "[admission-live-route-turn-bridge-admission-smoke] pass: log=$LOG"
