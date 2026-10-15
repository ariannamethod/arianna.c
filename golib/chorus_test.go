package main

import (
	"strings"
	"testing"
)

// realFieldOutput mirrors the chorus-arianna `field` mode stdout: a banner, then
// "rN cell K (T=…):  <text>  [metrics]" lines (cells 1-3 carry TWO bracket blocks,
// Δ_R^kv then entropy). The parser must survive the metric frames and generated
// text that itself contains a colon.
const realFieldOutput = `[chorus] 4 cells over the nano
  r1 cell 0 (T=0.60):  of time, the air is alive. It’s not just a place that   [entropy=1.70]
  r1 cell 1 (T=0.83):  here is the truth: the field listens back   [Δ_R^kv c1 = +0.337509 floor 0.357113 margin -0.019604]   [entropy=3.12]
  r1 cell 2 (T=1.07):  Memory: [alive] to choose the same activity in you   [Δ_R^kv c2 = -0.051243 floor 0.534510 margin -0.585753]   [entropy=5.15]
`

func TestChorusBodyKeepsColonText(t *testing.T) {
	cells := parseField(realFieldOutput)
	if len(cells) != 3 {
		t.Fatalf("want 3 parsed lines, got %d: %+v", len(cells), cells)
	}
	// cell 1's generated text contains a colon — the OLD LastIndex(":") truncated it
	// to "the field listens back"; the structural-colon parse must keep the whole text.
	want := "here is the truth: the field listens back"
	if cells[1].text != want {
		t.Errorf("colon-text truncated:\n  got  %q\n  want %q", cells[1].text, want)
	}
	if cells[2].text != "Memory: [alive] to choose the same activity in you" {
		t.Errorf("model brackets were truncated: %q", cells[2].text)
	}
	// no metrics block may leak into any fragment.
	for i, c := range cells {
		if strings.Contains(c.text, "Δ_R") || strings.Contains(c.text, "entropy=") {
			t.Errorf("cell %d leaked metrics: %q", i, c.text)
		}
	}
}

// parseField runs choir()'s line parser over canned output (choir() itself spawns
// a subprocess; this tests the parse in isolation).
func parseField(out string) []chorusCell {
	return parseChorusCells(out)
}
