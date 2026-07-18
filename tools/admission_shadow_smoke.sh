#!/usr/bin/env bash
# admission_shadow_smoke.sh - runtime check for shadow dream admission receipts.
#
# Runs the built metabolism binary in an isolated scratch directory with
# AM_DREAM_ADMISSION=shadow and AM_DREAM_ADMISSION_LOG set, then verifies that a
# typed JSONL receipt appears while no durable organism state is written.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_SMOKE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-admission-smoke.XXXXXX")}"
LOG="$WORKDIR/dream_admission.jsonl"
RUN_LOG="$WORKDIR/admission_smoke.log"

die() {
    echo "[admission-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-shadow-smoke or make body-smoke"

echo "[admission-smoke] root=$ROOT"
echo "[admission-smoke] scratch=$WORKDIR"

if ! (cd "$WORKDIR" && \
    AM_DREAM_ADMISSION=shadow \
    AM_DREAM_ADMISSION_LOG="$LOG" \
    "$ROOT/metabolism" --admission-smoke) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-smoke failed"
fi

[[ -s "$LOG" ]] || die "admission JSONL log not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"mode":"shadow"' "$LOG" || die "shadow mode missing"
grep -q '"accepted":false' "$LOG" || die "shadow candidate was not rejected"
grep -q '"counterfactual":{' "$LOG" || die "counterfactual missing"
grep -q '"pre_state_hash":"' "$LOG" || die "pre_state_hash missing"
grep -q '"post_state_hash":"' "$LOG" || die "post_state_hash missing"
grep -q '"trauma_level":' "$LOG" || die "trauma delta missing"
grep -q '"replay":{' "$LOG" || die "replay guard missing"
grep -q '"schema":"arianna.dream_replay_guard.v1"' "$LOG" || die "replay guard schema missing"
grep -q '"matched":true' "$LOG" || die "replay guard did not match"
grep -q '"passes":2' "$LOG" || die "replay guard did not run both passes"
grep -q '"admission_policy":{' "$LOG" || die "admission policy missing"
grep -q '"schema":"arianna.dream_admission_policy.v1"' "$LOG" || die "admission policy schema missing"
grep -q '"passed":true' "$LOG" || die "admission policy did not pass"
grep -q '\[admission-smoke\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "shadow admission smoke wrote durable organism state"
fi

echo "[admission-smoke] pass: log=$LOG"
