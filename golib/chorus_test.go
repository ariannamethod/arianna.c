package main

import (
	"strings"
	"testing"
)

// realFieldOutput mirrors the chorus-arianna `field` mode stdout: a banner, then
// "rN cell K (T=…):  <text>  [metrics]" lines (cells 1-3 carry TWO bracket blocks,
// Δ_R^kv then entropy), and an optional qloop line. The parser must survive all of
// it — including generated text that itself contains a colon.
const realFieldOutput = `[chorus] 4 cells over the nano
  r1 cell 0 (T=0.60):  of time, the air is alive. It’s not just a place that   [entropy=1.70]
  r1 cell 1 (T=0.83):  here is the truth: the field listens back   [Δ_R^kv c1 = +0.337509 floor 0.357113 margin -0.019604]   [entropy=3.12]
  r1 cell 2 (T=1.07):  to choose the same activity in you   [Δ_R^kv c2 = -0.051243 floor 0.534510 margin -0.585753]   [entropy=5.15]
  ↳ qloop c3→c0 score 1.097:  are we the same wave, or two?   [entropy=4.0]
`

func TestChorusBodyKeepsColonText(t *testing.T) {
	cells := parseField(realFieldOutput)
	if len(cells) != 4 {
		t.Fatalf("want 4 parsed lines, got %d: %+v", len(cells), cells)
	}
	// cell 1's generated text contains a colon — the OLD LastIndex(":") truncated it
	// to "the field listens back"; the structural-colon parse must keep the whole text.
	want := "here is the truth: the field listens back"
	if cells[1].text != want {
		t.Errorf("colon-text truncated:\n  got  %q\n  want %q", cells[1].text, want)
	}
	// no metrics block may leak into any fragment (the double-bracket case).
	for i, c := range cells {
		if strings.ContainsAny(c.text, "[]") || strings.Contains(c.text, "Δ_R") || strings.Contains(c.text, "entropy=") {
			t.Errorf("cell %d leaked metrics: %q", i, c.text)
		}
	}
}

func TestChorusQloopSeparated(t *testing.T) {
	cells := parseField(realFieldOutput)
	voices, questions := chorusCounts(cells)
	if voices != 3 || questions != 1 {
		t.Fatalf("want 3 voices + 1 question, got %d voices + %d questions", voices, questions)
	}
	// the qloop text follows the colon after "score N", not the score itself.
	var q string
	for _, c := range cells {
		if c.qloop {
			q = c.text
		}
	}
	if q != "are we the same wave, or two?" {
		t.Errorf("qloop body wrong: %q", q)
	}
}

func TestChorusQloopKVBody(t *testing.T) {
	out := "  ↳ qloop c1→c0 [kv] score 1.209:  what did the neighbour hear?   [entropy=4.2 I_Q^kv=+0.101]\n"
	cells := parseField(out)
	if len(cells) != 1 || !cells[0].qloop {
		t.Fatalf("bad qloop kv parse: %+v", cells)
	}
	if cells[0].text != "what did the neighbour hear?" {
		t.Fatalf("qloop kv body wrong: %q", cells[0].text)
	}
}

func TestChorusTextCaps(t *testing.T) {
	// a long polyphony must be capped before it is persisted into lastDream.
	long := make([]chorusCell, 0, 30)
	for i := 0; i < 30; i++ {
		long = append(long, chorusCell{text: strings.Repeat("resonance ", 20)})
	}
	s := chorusText(long)
	if len(s) > maxDreamLen+len("…") {
		t.Errorf("chorusText not capped: len=%d", len(s))
	}
}

// parseField runs choir()'s line parser over canned output (choir() itself spawns
// a subprocess; this tests the parse in isolation).
func parseField(out string) []chorusCell {
	return parseChorusCells(out)
}
