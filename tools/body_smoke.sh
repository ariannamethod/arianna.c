#!/usr/bin/env bash
# body_smoke.sh — non-mutating executable contract for the Arianna.c body.
#
# The build target compiles the organs first. This script verifies the expected
# forward/rope sentinels, runs Go tests, and, when GGUF weights are available,
# runs tiny generation probes from an isolated scratch directory so live soma,
# field, cooc, delta, KK, and DOE state are not touched.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${A2A_BODY_SMOKE_PROMPT:-Who are you?}"
TOKENS="${A2A_BODY_SMOKE_TOKENS:-2}"
REQUIRE_WEIGHTS="${A2A_BODY_SMOKE_REQUIRE_WEIGHTS:-0}"

JANUS_MODEL="${A2A_JANUS_MODEL:-$ROOT/weights/arianna_v4_sft_f16.gguf}"
RESONANCE_MODEL="${A2A_RESONANCE_MODEL:-$ROOT/weights/arianna_resonance_v3_f16.gguf}"
NANO_MODEL="${A2A_NANO_MODEL:-$ROOT/weights/nano_arianna_f16.gguf}"

tmp_root="${TMPDIR:-/private/tmp}"
if [[ ! -d "$tmp_root" ]]; then tmp_root="/tmp"; fi
WORKDIR="${A2A_BODY_SMOKE_WORKDIR:-$(mktemp -d "${tmp_root%/}/arianna-body-smoke.XXXXXX")}"

die() {
    echo "[body-smoke] FAIL: $*" >&2
    exit 1
}

require_exe() {
    [[ -x "$ROOT/$1" ]] || die "missing executable $1; run make body-smoke from repo root"
}

require_source() {
    local pattern="$1" file="$2" label="$3"
    if ! grep -Eq "$pattern" "$ROOT/$file"; then
        die "source sentinel missing: $label ($file)"
    fi
}

require_log() {
    local name="$1" pattern="$2" label="$3"
    if ! grep -Eq "$pattern" "$WORKDIR/$name.log"; then
        tail -n 80 "$WORKDIR/$name.log" >&2 || true
        die "$name log sentinel missing: $label"
    fi
}

run_logged() {
    local name="$1"; shift
    local log="$WORKDIR/$name.log"
    echo "[body-smoke] runtime $name"
    if (cd "$WORKDIR" && "$@") >"$log" 2>&1; then
        return 0
    fi
    tail -n 80 "$log" >&2 || true
    die "$name failed; log: $log"
}

echo "[body-smoke] root=$ROOT"
mkdir -p "$WORKDIR/weights"

for exe in arianna arianna_resonance nano-arianna chorus-arianna doe_field kk-cli metabolism; do
    require_exe "$exe"
done

require_source 'matvec_weight_right\(echo_back' tools/yent_forward.h "Janus Echo backprojection"
require_source 'kv_prev_embedding' tools/yent_forward.h "Janus decode smear cache"
require_source 'r_attn\[j\] = s;' tools/resonance_forward.h "Resonance RRPRAM unscaled score"
require_source 'strcmp\(arch, "nlama"\)' chorus/arianna2arianna.c "nano nlama NEOX route"

echo "[body-smoke] Go tests"
(cd "$ROOT/golib" && go test ./...)

echo "[body-smoke] admission shadow receipt"
A2A_ADMISSION_SMOKE_WORKDIR="$WORKDIR/admission-shadow" \
    bash "$ROOT/tools/admission_shadow_smoke.sh"

echo "[body-smoke] admission live route-plan gate"
A2A_ADMISSION_LIVE_ROUTE_GATE_WORKDIR="$WORKDIR/admission-live-route-gate" \
    bash "$ROOT/tools/admission_live_route_gate_smoke.sh"

echo "[body-smoke] admission live route chat dry-run"
A2A_ADMISSION_LIVE_ROUTE_CHAT_WORKDIR="$WORKDIR/admission-live-route-chat" \
    bash "$ROOT/tools/admission_live_route_chat_smoke.sh"

echo "[body-smoke] admission live route turn observation"
A2A_ADMISSION_LIVE_ROUTE_TURN_WORKDIR="$WORKDIR/admission-live-route-turn" \
    bash "$ROOT/tools/admission_live_route_turn_smoke.sh"

echo "[body-smoke] admission live route turn choice"
A2A_ADMISSION_LIVE_ROUTE_TURN_CHOICE_WORKDIR="$WORKDIR/admission-live-route-turn-choice" \
    bash "$ROOT/tools/admission_live_route_turn_choice_smoke.sh"

echo "[body-smoke] admission live route turn request"
A2A_ADMISSION_LIVE_ROUTE_TURN_REQUEST_WORKDIR="$WORKDIR/admission-live-route-turn-request" \
    bash "$ROOT/tools/admission_live_route_turn_request_smoke.sh"

echo "[body-smoke] admission live route generation job"
A2A_ADMISSION_LIVE_ROUTE_TURN_GENERATION_JOB_WORKDIR="$WORKDIR/admission-live-route-generation-job" \
    bash "$ROOT/tools/admission_live_route_turn_generation_job_smoke.sh"

echo "[body-smoke] admission live route turn/candidate review"
A2A_ADMISSION_LIVE_ROUTE_TURN_REVIEW_WORKDIR="$WORKDIR/admission-live-route-turn-review" \
    bash "$ROOT/tools/admission_live_route_turn_review_smoke.sh"

echo "[body-smoke] admission live route turn bridge"
A2A_ADMISSION_LIVE_ROUTE_TURN_BRIDGE_WORKDIR="$WORKDIR/admission-live-route-turn-bridge" \
    bash "$ROOT/tools/admission_live_route_turn_bridge_smoke.sh"

echo "[body-smoke] admission live route turn bridge admission"
A2A_ADMISSION_LIVE_ROUTE_TURN_BRIDGE_ADMISSION_WORKDIR="$WORKDIR/admission-live-route-turn-bridge-admission" \
    bash "$ROOT/tools/admission_live_route_turn_bridge_admission_smoke.sh"

echo "[body-smoke] admission shadow sample"
A2A_ADMISSION_SAMPLE_WORKDIR="$WORKDIR/admission-sample" \
    bash "$ROOT/tools/admission_shadow_sample.sh"

if [[ ! -f "$JANUS_MODEL" || ! -f "$RESONANCE_MODEL" || ! -f "$NANO_MODEL" ]]; then
    if [[ "$REQUIRE_WEIGHTS" == "1" ]]; then
        die "missing one or more GGUFs: $JANUS_MODEL | $RESONANCE_MODEL | $NANO_MODEL"
    fi
    echo "[body-smoke] runtime skipped: GGUF weights not all present"
    echo "[body-smoke] pass: build + source sentinels + Go tests"
    exit 0
fi

ln -sf "$JANUS_MODEL" "$WORKDIR/weights/arianna_v4_sft_f16.gguf"
ln -sf "$RESONANCE_MODEL" "$WORKDIR/weights/arianna_resonance_v3_f16.gguf"
ln -sf "$NANO_MODEL" "$WORKDIR/weights/nano_arianna_f16.gguf"

run_logged janus "$ROOT/arianna" \
    -w weights/arianna_v4_sft_f16.gguf -p "$PROMPT" -n "$TOKENS" -t 0.7 --top-p 0.9
require_log janus '\[arianna\] [0-9]+ tokens' "Janus generated tokens"

run_logged resonance "$ROOT/arianna_resonance" \
    -w weights/arianna_resonance_v3_f16.gguf -p "$PROMPT" -n "$TOKENS" -t 0.7 --top-p 1.0 --no-field
require_log resonance '\[resonance\] [0-9]+ tokens' "Resonance generated tokens"

run_logged nano "$ROOT/nano-arianna" \
    --model weights/nano_arianna_f16.gguf --prompt "Q: $PROMPT
A:" --max-tokens "$TOKENS" --temp 0.8 --top-p 0.92
require_log nano '\[nanollama\] ready' "nano loaded"
require_log nano '\[[0-9]+ tokens,' "nano generated tokens"

run_logged chorus "$ROOT/chorus-arianna" \
    weights/nano_arianna_f16.gguf "$PROMPT" field 2 4 1 0 0 0.02 1 1.3 0 1 1 0
require_log chorus 'NEOX rope' "chorus NEOX rope for nano GGUF"
require_log chorus '=== δ-field done' "chorus field completed"

"$ROOT/doe_field" --help >"$WORKDIR/doe_help.log" 2>&1 || die "doe_field --help failed"

echo "[body-smoke] pass: runtime scratch=$WORKDIR"
