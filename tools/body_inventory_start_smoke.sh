#!/usr/bin/env bash
# body_inventory_start_smoke.sh - prove live startup blocks on missing required organs.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi

if [[ -n "${A2A_BODY_INVENTORY_START_WORKDIR:-}" ]]; then
    mkdir -p "$A2A_BODY_INVENTORY_START_WORKDIR"
    WORKDIR="$(mktemp -d "${A2A_BODY_INVENTORY_START_WORKDIR%/}/arianna-body-inventory-start.XXXXXX")"
else
    WORKDIR="$(mktemp -d "${tmp_root%/}/arianna-body-inventory-start.XXXXXX")"
fi
BODYDIR="$WORKDIR/body"
LOG="$WORKDIR/body_inventory_start.jsonl"
RUN_LOG="$WORKDIR/metabolism.log"

die() {
    echo "[body-inventory-start-smoke] FAIL: $*" >&2
    exit 1
}

[[ -x "$ROOT/metabolism" ]] || die "missing executable metabolism; run make body-inventory-start-smoke from repo root"

mkdir -p "$BODYDIR"
if ! (cd "$BODYDIR" && \
    A2A_JANUS_MODEL="weights/arianna_v4_sft_f16.gguf" \
    A2A_RESONANCE_MODEL="weights/arianna_resonance_v3_f16.gguf" \
    A2A_NANO_MODEL="weights/nano_arianna_f16.gguf" \
    AM_BODY_INVENTORY_LOG="$LOG" \
    "$ROOT/metabolism" "body inventory blocked smoke") >"$RUN_LOG" 2>&1; then
    tail -n 80 "$RUN_LOG" >&2 || true
    die "metabolism missing-body probe crashed; log: $RUN_LOG"
fi

grep -q 'metabolism: body inventory blocked: required organs missing:' "$RUN_LOG" || die "blocked startup line missing"
grep -q '"schema":"arianna.body_inventory.v1"' "$LOG" || die "receipt schema missing"
grep -q '"status":"blocked"' "$LOG" || die "receipt status must be blocked"
grep -q '"live_trio_allowed":false' "$LOG" || die "blocked receipt must deny live trio"
grep -q '"continue_allowed":true' "$LOG" || die "blocked receipt must keep inspection alive"
grep -q '"mutates_state":false' "$LOG" || die "receipt must be non-mutating"

for organ in janus-binary janus-weight resonance-binary resonance-weight; do
    grep -q "\"name\":\"$organ\"" "$LOG" || die "organ missing from receipt: $organ"
    grep -q "$organ" "$RUN_LOG" || die "required missing organ not named in startup block: $organ"
done

if grep -Eq 'janus daemon:|resonance daemon:|\[high\]|arianna-metabolism' "$RUN_LOG"; then
    tail -n 80 "$RUN_LOG" >&2 || true
    die "startup advanced past inventory gate"
fi

echo "[body-inventory-start-smoke] pass: receipt=$LOG run=$RUN_LOG"
