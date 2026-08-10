#!/usr/bin/env bash
# admission_live_route_weighted_readiness_smoke.sh - one weighted pre-live gate.
#
# This wraps the portable body smoke plus its GGUF-required nano-direct tail into
# a named pre-live readiness surface. It proves the heavy path through final gate
# and Resonance graft admission proof while keeping live admission and mutation
# closed.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

WORKDIR="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-live-route-weighted-readiness.XXXXXX")}"
BODY_WORKDIR="$WORKDIR/body-smoke-weighted"
RUN_LOG="$WORKDIR/body_smoke_weighted.log"
ASSERT_LOG="$WORKDIR/boundary_report_full_chain.log"
READINESS_REPORT="${A2A_ADMISSION_LIVE_ROUTE_WEIGHTED_READINESS_REPORT:-$WORKDIR/live_route_weighted_readiness.json}"

JANUS_MODEL="${A2A_JANUS_MODEL:-$ROOT/weights/arianna_v4_sft_f16.gguf}"
RESONANCE_MODEL="${A2A_RESONANCE_MODEL:-$ROOT/weights/arianna_resonance_v3_f16.gguf}"
NANO_MODEL="${A2A_NANO_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"

PROOF_DIR="$BODY_WORKDIR/nano-direct-resonance-graft-admission-proof"
FINAL_GATE_DIR="$BODY_WORKDIR/nano-direct-final-gate"
BOUNDARY_REPORT="$PROOF_DIR/live_route_boundary_report.json"
PROOF_LOG="$PROOF_DIR/live_route_candidate_admission_resonance_graft_admission_proof_nano_direct.jsonl"
FINAL_GATE_LOG="$FINAL_GATE_DIR/live_route_candidate_admission_final_gate_nano_direct.jsonl"

die() {
    echo "[admission-live-route-weighted-readiness-smoke] FAIL: $*" >&2
    if [[ -f "$RUN_LOG" ]]; then
        tail -n 500 "$RUN_LOG" >&2 || true
    fi
    if [[ -f "$ASSERT_LOG" ]]; then
        tail -n 120 "$ASSERT_LOG" >&2 || true
    fi
    exit 1
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "$path" ]] || die "missing $label: $path"
}

require_nonempty() {
    local path="$1"
    local label="$2"
    [[ -s "$path" ]] || die "$label not written: $path"
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

require_file "$JANUS_MODEL" "Janus GGUF"
require_file "$RESONANCE_MODEL" "Resonance GGUF"
require_file "$NANO_MODEL" "nano GGUF"

if ! A2A_JANUS_MODEL="$JANUS_MODEL" \
    A2A_RESONANCE_MODEL="$RESONANCE_MODEL" \
    A2A_NANO_MODEL="$NANO_MODEL" \
    A2A_BODY_SMOKE_WORKDIR="$BODY_WORKDIR" \
    A2A_BODY_SMOKE_REQUIRE_WEIGHTS=1 \
    A2A_BODY_SMOKE_NANO_DIRECT=1 \
    bash "$ROOT/tools/body_smoke.sh" >"$RUN_LOG" 2>&1; then
    die "weighted body-smoke failed"
fi

require_nonempty "$BOUNDARY_REPORT" "weighted boundary report"
require_nonempty "$PROOF_LOG" "weighted Resonance graft admission proof"
require_nonempty "$FINAL_GATE_LOG" "weighted final gate"

if ! bash "$ROOT/tools/admission_live_route_boundary_report_assert_full_chain.sh" "$BOUNDARY_REPORT" >"$ASSERT_LOG" 2>&1; then
    die "weighted boundary report failed full-chain assertion"
fi

require_grep '"schema":"arianna.live_route_turn_candidate_admission_resonance_graft_admission_proof.v1"' "$PROOF_LOG" "proof schema"
require_grep '"admission_resonance_graft_admission_proof_ready":true' "$PROOF_LOG" "proof readiness"
require_grep '"admission_resonance_graft_admission_proof_dry_run_only":true' "$PROOF_LOG" "proof dry-run guard"
require_grep '"contracts_ready":false' "$PROOF_LOG" "proof closed contracts"
require_grep '"write_allowed":false' "$PROOF_LOG" "proof closed writer"
require_grep '"admission_allowed":false' "$PROOF_LOG" "proof closed admission"
require_grep '"live_admission_enabled":false' "$PROOF_LOG" "proof closed live admission"
require_grep '"mutates_state":false' "$PROOF_LOG" "proof non-mutation"

require_grep '"schema":"arianna.live_route_turn_candidate_admission_final_gate.v1"' "$FINAL_GATE_LOG" "final gate schema"
require_grep '"admission_final_gate_ready":true' "$FINAL_GATE_LOG" "final gate readiness"
require_grep '"live_admission_enabled":false' "$FINAL_GATE_LOG" "final gate closed live admission"
require_grep '"mutates_state":false' "$FINAL_GATE_LOG" "final gate non-mutation"

require_grep '\[body-smoke\] weighted nano-direct Resonance graft admission proof' "$RUN_LOG" "weighted graft proof lane"
require_grep '\[body-smoke\] pass: runtime scratch=' "$RUN_LOG" "weighted body smoke pass"

cat >"$READINESS_REPORT" <<EOF
{
  "schema": "arianna.live_route_weighted_readiness.v1",
  "status": "ready_closed_dry_run",
  "target": "live_admission",
  "body_smoke_weighted": true,
  "nano_direct_runner": true,
  "nano_direct_final_gate": true,
  "resonance_graft_admission_proof": true,
  "boundary_report_full_chain": true,
  "contracts_ready": false,
  "write_allowed": false,
  "admission_allowed": false,
  "live_admission_enabled": false,
  "mutates_state": false,
  "body_workdir": "$BODY_WORKDIR",
  "boundary_report": "$BOUNDARY_REPORT",
  "proof_log": "$PROOF_LOG",
  "final_gate_log": "$FINAL_GATE_LOG"
}
EOF

require_grep '"schema": "arianna.live_route_weighted_readiness.v1"' "$READINESS_REPORT" "readiness report schema"
require_grep '"status": "ready_closed_dry_run"' "$READINESS_REPORT" "readiness report status"
require_grep '"boundary_report_full_chain": true' "$READINESS_REPORT" "readiness report full-chain flag"
require_grep '"live_admission_enabled": false' "$READINESS_REPORT" "readiness report closed live flag"
require_grep '"mutates_state": false' "$READINESS_REPORT" "readiness report non-mutation flag"

echo "[admission-live-route-weighted-readiness-smoke] pass: readiness_report=$READINESS_REPORT boundary_report=$BOUNDARY_REPORT proof=$PROOF_LOG final_gate=$FINAL_GATE_LOG"
