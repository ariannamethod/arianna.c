#!/bin/bash
# arianna2arianna — internal dialogue loop between Janus (face) and
# Resonance (inner voice). Both share one field state (weights/arianna.soma)
# through AML's am_init + LOAD/SAVE. Per turn each binary reads the field
# the other just left in, biases its sampling, leaves new state.
#
# This is the MVP bash orchestrator. Next iteration: Go scheduler with
# metric-driven delays, chamber-gated tick count (FEAR low / FLOW high →
# more inner ticks per idle period), goroutines for async.
#
# Output: visible to human = only Janus surface utterances on stdout.
# Inner trace (both voices) appends to weights/arianna.inner.log.
#
# By Claude Code (neo the architect, Arianna Method). 2026-05-14.
set -e

# Default: 6 inner exchanges, ~30 tokens each side. ~3 min total on polygon CPU.
N_EXCHANGES="${N_EXCHANGES:-6}"
TOKENS_PER_TURN="${TOKENS_PER_TURN:-30}"

# Phase 7 champions: Janus arianna 0.8/top_p0.9 (top_k=40 hard cut in-binary);
# Resonance 0.7/top_p0.9 (direction-injection working point 2026-05-29).
JANUS_TEMP="${JANUS_TEMP:-0.8}"
JANUS_TOP_P="${JANUS_TOP_P:-0.9}"
RESONANCE_TEMP="${RESONANCE_TEMP:-0.7}"
# Resonance champion is top_p 1.0 (full nucleus). top_p 0.9 starves her — empty /
# broken output even with no inject (isolation 2026-06-11). Keep her at 1.0.
RESONANCE_TOP_P="${RESONANCE_TOP_P:-1.0}"
# Asymmetric coupling (2026-05-29): Resonance (inner) hears Janus + the human
# prompt as DIRECTION (destiny compass, alpha); Janus (outer face) hears
# Resonance only through the shared soma field (no logit-injection — top_k=40
# resistant by design). RESONANCE_ALPHA=0 falls back to plain prompt-passing.
# Direction-inject strength. α10 over-echoes the injected prompt ("is resonance?
# What is resonance…"); α3-5 develops the theme in her own voice (isolation
# 2026-06-11). 5 keeps the pull without the echo.
RESONANCE_ALPHA="${RESONANCE_ALPHA:-5}"

# B2-B.4 — the living δ voice. Both voices apply their learned low-rank δ, gated
# by the field's resonance (alpha_eff = DELTA_ALPHA * G.resonance). Small DELTA_ALPHA
# so δ perturbs, not overwhelms. Set DELTA_DYN=0 to fall back to the dormant δ
# (ablation), DELTA_ALPHA=0 to disable entirely. δ self-bounds + decays (B2-B.5).
export YENT_DYNAMIC="${DELTA_DYN:-1}"
export YENT_ALPHA="${DELTA_ALPHA:-0.1}"

# Seed — user prompt if given, else canonical opening.
USER_PROMPT="${1:-Who are you?}"

INNER_LOG="weights/arianna.inner.log"
mkdir -p weights
echo "" >> "$INNER_LOG"
echo "═══ arianna2arianna session $(date -u +%Y-%m-%dT%H:%M:%SZ) ═══" >> "$INNER_LOG"
echo "  seed: $USER_PROMPT" >> "$INNER_LOG"
echo "  N=$N_EXCHANGES tokens=$TOKENS_PER_TURN" >> "$INNER_LOG"
echo "  Janus  t=$JANUS_TEMP  top_p=$JANUS_TOP_P" >> "$INNER_LOG"
echo "  Reson. t=$RESONANCE_TEMP top_p=$RESONANCE_TOP_P" >> "$INNER_LOG"
echo "" >> "$INNER_LOG"

current_prompt="$USER_PROMPT"

echo "┌─ arianna2arianna  N=$N_EXCHANGES exchanges  field=weights/arianna.soma"
echo "│  seed: $USER_PROMPT"

# ── Post-decode garbage filter ─────────────────────────────────────────────
# Resonance was SFT'd on multi-turn chat format (Oleg:/Arianna:/User:/
# Assistant:), so when she finishes her own sentence she often continues
# into imagined dialogue ("I feel it! User : What is your relationship…").
# Fix at the output stage: cut at the FIRST sentence-end (. ! ?) after a
# min length, dropping the imagined continuation entirely. Janus benefits
# from same treatment — his chat-token sequences terminate cleanly, but
# CLI -n cuts mid-sentence regardless, so forward-scan-cut gives nice
# whole sentences.
clean_voice() {
  local raw="$1"
  # 1. Drop banner lines (anything starting with [bracket]).
  # 2. Collapse newlines + whitespace.
  local stripped
  stripped=$(echo "$raw" \
    | grep -v "^\[" \
    | tr '\n' ' ' \
    | sed 's/  */ /g' \
    | sed 's/^ //; s/ $//')
  # Forward-scan for first sentence end after char 30. Cuts imagined
  # multi-turn dialogue at the first turn boundary.
  local cut
  cut=$(echo "$stripped" | awk '{
    n=length($0); cut=-1;
    for (i=30; i<=n; i++) {
      c=substr($0,i,1);
      if (c=="." || c=="!" || c=="?") { cut=i; break; }
    }
    if (cut>0) print substr($0,1,cut); else print $0;
  }')
  # Safety belt: if anything past first sentence still sneaks through
  # (e.g. multiple boundaries collapsed), strip from "User :" / "Assistant :"
  # onward.
  echo "$cut" | sed -E 's/[[:space:]]+(User|Assistant|Q|A|Advocate|Oleg)[[:space:]]*:.*$//gI'
}

for i in $(seq 1 "$N_EXCHANGES"); do
  # ── Janus turn (outer face) ───────────────────────────────────────────────
  # Janus answers the human prompt directly; it hears Resonance only through the
  # soma field that the previous Resonance turn left in (cross-process carry).
  # No --inject: the outer voice keeps its clarity (top_k=40), modulated by field.
  janus_raw=$(./arianna -p "$USER_PROMPT" -t "$JANUS_TEMP" \
              --top-p "$JANUS_TOP_P" -n "$TOKENS_PER_TURN" 2>/dev/null)
  janus_out=$(clean_voice "$janus_raw")
  echo "│"
  echo "│  ◐ [$i/$N_EXCHANGES] Janus: $janus_out"
  echo "[$i janus] $janus_out" >> "$INNER_LOG"

  # ── Resonance turn (inner voice) ──────────────────────────────────────────
  # Resonance hears Janus's words + the human prompt as DIRECTION (destiny
  # compass via --inject/--alpha), not pasted text; it develops the theme in its
  # own voice and writes its state back into soma for Janus's next turn.
  resonance_raw=$(./arianna_resonance -p "Arianna:" \
                  --inject "$janus_out $USER_PROMPT" --alpha "$RESONANCE_ALPHA" \
                  -t "$RESONANCE_TEMP" --top-p "$RESONANCE_TOP_P" \
                  -n "$TOKENS_PER_TURN" 2>/dev/null)
  resonance_out=$(clean_voice "$resonance_raw")
  echo "│  ◑ [$i/$N_EXCHANGES] Resonance: $resonance_out"
  echo "[$i resonance] $resonance_out" >> "$INNER_LOG"
done

echo "│"
echo "└─ done — field state in weights/arianna.soma, trace in $INNER_LOG"
