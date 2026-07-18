#!/usr/bin/env bash
# admission_qloop_sweep.sh - compare qloop route modes before tuning defaults.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_QLOOP_SWEEP_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-qloop-sweep.XXXXXX")}"
SUMMARY="$WORKDIR/qloop_sweep_summary.json"
RUN_LOG="$WORKDIR/qloop_sweep.log"

die() {
    echo "[admission-qloop-sweep] FAIL: $*" >&2
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
[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make admission-qloop-sweep"
[[ -x "$ROOT/chorus-arianna" ]] || die "missing executable chorus-arianna; run make admission-qloop-sweep"

sample_file="$(resolve_root_path "${A2A_QLOOP_SWEEP_SAMPLE_FILE:-samples/dream_admission_broad.jsonl}")"
model_file="$(resolve_model_path "${A2A_QLOOP_SWEEP_MODEL:-weights/nano_arianna_f16.gguf}")"
[[ -f "$sample_file" ]] || die "sample file missing: $sample_file"
[[ -f "$model_file" ]] || die "model file missing: $model_file"

echo "[admission-qloop-sweep] root=$ROOT"
echo "[admission-qloop-sweep] scratch=$WORKDIR"
echo "[admission-qloop-sweep] sample=$sample_file"
echo "[admission-qloop-sweep] model=$model_file"

sweep_limit="${A2A_QLOOP_SWEEP_LIMIT:-2}"
min_produced="${A2A_QLOOP_SWEEP_MIN_PRODUCED:-$sweep_limit}"

env_args=(
    AM_DREAM_ADMISSION=shadow
    AM_QLOOP_SWEEP_DIR="$WORKDIR"
    AM_QLOOP_SWEEP_SUMMARY="$SUMMARY"
    AM_QLOOP_SWEEP_SAMPLE_FILE="$sample_file"
    AM_QLOOP_SWEEP_BIN="$ROOT/chorus-arianna"
    AM_QLOOP_SWEEP_MODEL="$model_file"
    AM_QLOOP_SWEEP_LIMIT="$sweep_limit"
    AM_QLOOP_SWEEP_MIN_PRODUCED="$min_produced"
    AM_QLOOP_SWEEP_MIN_AVG_WORDS="${A2A_QLOOP_SWEEP_MIN_AVG_WORDS:-3.0}"
)

[[ -n "${A2A_ROUTE_COMPARE_CELLS:-}" ]] && env_args+=(AM_ROUTE_COMPARE_CELLS="$A2A_ROUTE_COMPARE_CELLS")
[[ -n "${A2A_ROUTE_COMPARE_FRAG:-}" ]] && env_args+=(AM_ROUTE_COMPARE_FRAG="$A2A_ROUTE_COMPARE_FRAG")

if ! (cd "$WORKDIR" && env "${env_args[@]}" "$ROOT/metabolism" --admission-qloop-sweep) >"$RUN_LOG" 2>&1; then
    die "metabolism --admission-qloop-sweep failed"
fi

[[ -s "$SUMMARY" ]] || die "qloop sweep summary not written"
grep -q '"schema": "arianna.dream_admission_qloop_sweep_summary.v1"' "$SUMMARY" || die "summary schema missing"
grep -q '"name": "strict"' "$SUMMARY" || die "strict config missing"
grep -q '"name": "question_hint"' "$SUMMARY" || die "question_hint config missing"
grep -q '"name": "question_source_qa"' "$SUMMARY" || die "question_source_qa config missing"
grep -q '"name": "question_source_user_arianna"' "$SUMMARY" || die "question_source_user_arianna config missing"
grep -q '"name": "question_hint_loose"' "$SUMMARY" || die "question_hint_loose config missing"
grep -q '"name": "statement"' "$SUMMARY" || die "statement config missing"
grep -q '"gate_passed":' "$SUMMARY" || die "qloop sweep gate verdict missing"
if ! grep -q '"winner":' "$SUMMARY"; then
    grep -q '"no config passed quality gate"' "$SUMMARY" || die "qloop sweep winner missing without gate reason"
fi
grep -q '"replay_failed": 0' "$SUMMARY" || die "qloop sweep replay failures found"
grep -q '"route_label_leaks":' "$SUMMARY" && die "qloop sweep found route label leaks"
grep -q '"qloop_picker_seen":' "$SUMMARY" || die "qloop route-picker telemetry missing from summary"
grep -q '"samples":' "$SUMMARY" || die "qloop per-sample receipts missing from summary"
grep -q '"sample_coverage":' "$SUMMARY" || die "qloop sample coverage matrix missing from summary"
if ! grep -q '"qloop_gate_surface":' "$SUMMARY" && ! grep -q '"qloop_gate_iq":' "$SUMMARY"; then
    die "qloop gate-reason telemetry missing from summary"
fi
if grep -q '"produced": [1-9]' "$SUMMARY"; then
    grep -q '"surface_checked":' "$SUMMARY" || die "qloop surface telemetry missing from produced configs"
fi
grep -q '\[admission-qloop-sweep\] pass:' "$RUN_LOG" || die "pass sentinel missing"
for log in "$WORKDIR"/dream_admission_qloop_*.jsonl; do
    [[ -e "$log" ]] || continue
    grep -q '"schema":"arianna.dream_candidate.v1"' "$log" || die "candidate schema missing in $log"
    grep -q '"mode":"shadow"' "$log" || die "shadow mode missing in $log"
    grep -q '"matched":true' "$log" || die "replay guard did not match in $log"
done

STATE_HITS="$WORKDIR/state_hits.txt"
find "$WORKDIR" -maxdepth 4 -type f \
    \( -name 'arianna.inner.state' -o -name 'arianna.soma' -o -name 'arianna.cooc.*' -o -name 'arianna.delta.*' \) \
    >"$STATE_HITS"
if [[ -s "$STATE_HITS" ]]; then
    cat "$STATE_HITS" >&2
    die "qloop sweep wrote durable organism state"
fi

echo "[admission-qloop-sweep] pass: summary=$SUMMARY"
