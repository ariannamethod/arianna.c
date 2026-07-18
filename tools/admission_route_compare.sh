#!/usr/bin/env bash
# admission_route_compare.sh - compare direct/chorus/qloop dream admission routes.
#
# Runs the metabolism route harness in an isolated scratch directory. The harness
# generates candidates through chorus-arianna routes, feeds them through the same
# shadow admission gate, writes typed receipts plus an aggregate summary, and
# verifies that no durable organism state is written.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_ROUTE_COMPARE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-route-compare.XXXXXX")}"
LOG="$WORKDIR/dream_admission_routes.jsonl"
SUMMARY="$WORKDIR/dream_admission_route_compare.json"
RUN_LOG="$WORKDIR/admission_route_compare.log"

die() {
    echo "[admission-route-compare] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 120 "$RUN_LOG" >&2 || true
    fi
    exit 1
}

resolve_root_path() {
    local path="$1"
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$ROOT" "$path" ;;
    esac
}

resolve_model_path() {
    local raw="$1"
    local path
    path="$(resolve_root_path "$raw")"
    if [[ -f "$path" ]]; then
        printf '%s\n' "$path"
        return
    fi
    if [[ "$raw" != /* && "$ROOT" == */.worktrees/* ]]; then
        local main_root="${ROOT%%/.worktrees/*}"
        local alt="$main_root/$raw"
        if [[ -f "$alt" ]]; then
            printf '%s\n' "$alt"
            return
        fi
    fi
    printf '%s\n' "$path"
}

mkdir -p "$WORKDIR"
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-route-compare"
[[ -x "$ROOT/chorus-arianna" ]] || die "missing executable chorus-arianna; run make admission-route-compare"

sample_file="$(resolve_root_path "${A2A_ROUTE_COMPARE_SAMPLE_FILE:-samples/dream_admission_broad.jsonl}")"
model_file="$(resolve_model_path "${A2A_ROUTE_COMPARE_MODEL:-weights/nano_arianna_f16.gguf}")"
[[ -f "$sample_file" ]] || die "sample file missing: $sample_file"
[[ -f "$model_file" ]] || die "model file missing: $model_file"

echo "[admission-route-compare] root=$ROOT"
echo "[admission-route-compare] scratch=$WORKDIR"
echo "[admission-route-compare] sample=$sample_file"
echo "[admission-route-compare] model=$model_file"

env_args=(
    AM_DREAM_ADMISSION=shadow
    AM_DREAM_ADMISSION_LOG="$LOG"
    AM_ROUTE_COMPARE_SUMMARY="$SUMMARY"
    AM_ROUTE_COMPARE_SAMPLE_FILE="$sample_file"
    AM_ROUTE_COMPARE_BIN="$ROOT/chorus-arianna"
    AM_ROUTE_COMPARE_MODEL="$model_file"
    AM_ROUTE_COMPARE_LIMIT="${A2A_ROUTE_COMPARE_LIMIT:-2}"
    AM_ROUTE_COMPARE_ROUTES="${A2A_ROUTE_COMPARE_ROUTES:-direct,chorus,qloop}"
)

[[ -n "${A2A_ROUTE_COMPARE_DIRECT_TOKENS:-}" ]] && env_args+=(AM_ROUTE_COMPARE_DIRECT_TOKENS="$A2A_ROUTE_COMPARE_DIRECT_TOKENS")
[[ -n "${A2A_ROUTE_COMPARE_DIRECT_TEMP:-}" ]] && env_args+=(AM_ROUTE_COMPARE_DIRECT_TEMP="$A2A_ROUTE_COMPARE_DIRECT_TEMP")
[[ -n "${A2A_ROUTE_COMPARE_DIRECT_TOP_P:-}" ]] && env_args+=(AM_ROUTE_COMPARE_DIRECT_TOP_P="$A2A_ROUTE_COMPARE_DIRECT_TOP_P")
[[ -n "${A2A_ROUTE_COMPARE_DIRECT_REP:-}" ]] && env_args+=(AM_ROUTE_COMPARE_DIRECT_REP="$A2A_ROUTE_COMPARE_DIRECT_REP")
[[ -n "${A2A_ROUTE_COMPARE_CELLS:-}" ]] && env_args+=(AM_ROUTE_COMPARE_CELLS="$A2A_ROUTE_COMPARE_CELLS")
[[ -n "${A2A_ROUTE_COMPARE_FRAG:-}" ]] && env_args+=(AM_ROUTE_COMPARE_FRAG="$A2A_ROUTE_COMPARE_FRAG")

if ! (cd "$WORKDIR" && env "${env_args[@]}" "$ROOT/metabolism" --admission-route-compare) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-route-compare failed"
fi

[[ -s "$LOG" ]] || die "route JSONL log not written"
[[ -s "$SUMMARY" ]] || die "route summary not written"
grep -q '"schema":"arianna.dream_candidate.v1"' "$LOG" || die "candidate schema missing"
grep -q '"mode":"shadow"' "$LOG" || die "shadow mode missing"
grep -q '"accepted":false' "$LOG" || die "shadow candidate was not rejected"
grep -q '"counterfactual":{' "$LOG" || die "counterfactual missing"
grep -q '"replay":{' "$LOG" || die "replay guard missing"
grep -q '"matched":true' "$LOG" || die "replay guard did not match"
grep -q '"admission_policy":{' "$LOG" || die "admission policy missing"
grep -q '"schema": "arianna.dream_admission_route_compare_summary.v1"' "$SUMMARY" || die "summary schema missing"
grep -q '"replay_failed": 0' "$SUMMARY" || die "route replay failures found"
grep -q '"direct"' "$SUMMARY" || die "direct route missing from summary"
grep -q '"chorus"' "$SUMMARY" || die "chorus route missing from summary"
grep -q '"qloop"' "$SUMMARY" || die "qloop route missing from summary"
grep -q '\[admission-route-compare\] pass:' "$RUN_LOG" || die "pass sentinel missing"

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "route compare wrote durable organism state"
fi

echo "[admission-route-compare] pass: log=$LOG summary=$SUMMARY"
