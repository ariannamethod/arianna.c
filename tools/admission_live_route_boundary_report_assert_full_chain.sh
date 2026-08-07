#!/usr/bin/env bash
# admission_live_route_boundary_report_assert_full_chain.sh - compact boundary report full-chain contract.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_CHAIN="$ROOT/tools/admission_live_route_boundary_report_stage_chain.sh"
ASSERT="$ROOT/tools/admission_live_route_boundary_report_assert.sh"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="$(mktemp -d "${tmp_root%/}/arianna-live-route-boundary-report-full-chain.XXXXXX")"
STAGE_FILE="$WORKDIR/live_route_boundary_report_stages.txt"

die() {
    echo "[admission-live-route-boundary-report-assert-full-chain] FAIL: $*" >&2
    exit 1
}

cleanup() {
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

[[ $# -eq 1 ]] || die "usage: $0 <boundary-report.json>"

if ! bash "$STAGE_CHAIN" >"$STAGE_FILE"; then
    die "boundary report stage chain export failed"
fi

BOUNDARY_REPORT_STAGES=()
while IFS= read -r stage_name; do
    [[ -n "$stage_name" ]] || die "boundary report stage chain exported an empty stage"
    BOUNDARY_REPORT_STAGES+=("$stage_name")
done <"$STAGE_FILE"
[[ ${#BOUNDARY_REPORT_STAGES[@]} -gt 0 ]] || die "boundary report stage chain export was empty"

bash "$ASSERT" "$1" "${#BOUNDARY_REPORT_STAGES[@]}" "${BOUNDARY_REPORT_STAGES[@]}"
