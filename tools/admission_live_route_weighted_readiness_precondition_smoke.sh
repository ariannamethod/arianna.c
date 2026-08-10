#!/usr/bin/env bash
# admission_live_route_weighted_readiness_precondition_smoke.sh - require weighted readiness before the next admission layer.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_PRECONDITION_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-readiness-precondition.XXXXXX")}"
READINESS_REPORT="$WORKDIR/live_route_weighted_readiness.json"
PRECONDITION_REPORT="$WORKDIR/live_route_weighted_readiness_precondition.json"
RUN_LOG="$WORKDIR/weighted_readiness.log"
PRECONDITION_LOG="$WORKDIR/weighted_readiness_precondition.log"

die() {
    echo "[admission-live-route-weighted-readiness-precondition-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 500 "$RUN_LOG" >&2 || true
    fi
    if [[ -f "$PRECONDITION_LOG" ]]; then
        tail -n 120 "$PRECONDITION_LOG" >&2 || true
    fi
    exit 1
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        die "$label missing in $file"
    fi
}

mkdir -p "$WORKDIR"

if ! A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_WORKDIR="$WORKDIR/producer" \
    A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_REPORT="$READINESS_REPORT" \
    bash "$ROOT/tools/admission_live_route_weighted_readiness_smoke.sh" >"$RUN_LOG" 2>&1; then
    die "weighted readiness producer failed"
fi

[[ -s "$READINESS_REPORT" ]] || die "weighted readiness report not written: $READINESS_REPORT"

if ! bash "$ROOT/tools/admission_live_route_weighted_readiness_precondition.sh" "$READINESS_REPORT" "$PRECONDITION_REPORT" >"$PRECONDITION_LOG" 2>&1; then
    die "weighted readiness precondition writer rejected producer report"
fi

[[ -s "$PRECONDITION_REPORT" ]] || die "weighted readiness precondition report not written: $PRECONDITION_REPORT"

require_grep '"schema": "arianna.live_route_weighted_readiness_precondition.v1"' "$PRECONDITION_REPORT" "precondition schema"
require_grep '"status": "precondition_satisfied_closed_dry_run"' "$PRECONDITION_REPORT" "precondition status"
require_grep '"target": "live_route_admission_next_step"' "$PRECONDITION_REPORT" "precondition target"
require_grep '"weighted_readiness_consumed": true' "$PRECONDITION_REPORT" "weighted readiness consumed flag"
require_grep '"weighted_readiness_required": true' "$PRECONDITION_REPORT" "weighted readiness required flag"
require_grep '"next_step_blocked_without_readiness": true' "$PRECONDITION_REPORT" "next-step block flag"
require_grep '"contracts_ready": false' "$PRECONDITION_REPORT" "closed contracts flag"
require_grep '"write_allowed": false' "$PRECONDITION_REPORT" "closed writer flag"
require_grep '"admission_allowed": false' "$PRECONDITION_REPORT" "closed admission flag"
require_grep '"live_admission_enabled": false' "$PRECONDITION_REPORT" "closed live flag"
require_grep '"mutates_state": false' "$PRECONDITION_REPORT" "non-mutation flag"
require_grep '"passed": true' "$PRECONDITION_REPORT" "precondition pass flag"
require_grep '\[admission-live-route-weighted-readiness-precondition\] pass:' "$PRECONDITION_LOG" "precondition pass line"

echo "[admission-live-route-weighted-readiness-precondition-smoke] pass: readiness_report=$READINESS_REPORT precondition_report=$PRECONDITION_REPORT"
