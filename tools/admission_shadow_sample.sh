#!/usr/bin/env bash
# admission_shadow_sample.sh - collect shadow dream admission receipts and summary.
#
# Runs the metabolism sampler in an isolated scratch directory. The sampler only
# works in shadow admission mode, writes typed receipts plus an aggregate summary,
# and verifies that no durable organism state is written.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ADMISSION_SAMPLE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-admission-sample.XXXXXX")}"
LOG="$WORKDIR/dream_admission_samples.jsonl"
SUMMARY="$WORKDIR/dream_admission_summary.json"
RUN_LOG="$WORKDIR/admission_sample.log"

die() {
    echo "[admission-sample] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 80 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-shadow-sample or make body-smoke"

echo "[admission-sample] root=$ROOT"
echo "[admission-sample] scratch=$WORKDIR"

env_args=(
    AM_DREAM_ADMISSION=shadow
    AM_DREAM_ADMISSION_LOG="$LOG"
    AM_DREAM_ADMISSION_SUMMARY="$SUMMARY"
)
if [[ -n "${A2A_ADMISSION_SAMPLE_FILE:-}" ]]; then
    [[ -f "$A2A_ADMISSION_SAMPLE_FILE" ]] || die "sample file missing: $A2A_ADMISSION_SAMPLE_FILE"
    env_args+=(AM_DREAM_ADMISSION_SAMPLE_FILE="$A2A_ADMISSION_SAMPLE_FILE")
fi

if ! (cd "$WORKDIR" && env "${env_args[@]}" "$ROOT/metabolism" --admission-sample) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-sample failed"
fi

[[ -s "$LOG" ]] || die "admission sample JSONL log not written"
[[ -s "$SUMMARY" ]] || die "admission sample summary not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"mode":"shadow"' "$LOG" || die "shadow mode missing"
grep -q '"accepted":false' "$LOG" || die "shadow candidate was not rejected"
grep -q '"counterfactual":{' "$LOG" || die "counterfactual missing"
grep -q '"replay":{' "$LOG" || die "replay guard missing"
grep -q '"matched":true' "$LOG" || die "replay guard did not match"
grep -q '"admission_policy":{' "$LOG" || die "admission policy missing"
grep -q '"schema": "arianna.dream_admission_sample_summary.v1"' "$SUMMARY" || die "summary schema missing"
grep -q '"replay_failed": 0' "$SUMMARY" || die "sampler replay failures found"
if [[ -z "${A2A_ADMISSION_SAMPLE_FILE:-}" ]]; then
    grep -Eq '"policy_failed": [1-9][0-9]*' "$SUMMARY" || die "builtin sample did not exercise policy failure"
fi
grep -q '\[admission-sample\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "shadow admission sampler wrote durable organism state"
fi

echo "[admission-sample] pass: log=$LOG summary=$SUMMARY"
