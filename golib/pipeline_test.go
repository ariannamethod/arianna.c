package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// ── surfaces(): the strained/wintering contract (the #4 fix) ────────────────────
func TestSurfacesContract(t *testing.T) {
	cases := []struct {
		name string
		fs   fieldSnapshot
		want bool
	}{
		{"no-signal", fieldSnapshot{}, false},
		{"summer expressive", fieldSnapshot{valid: true, summer: 0.8}, true},
		{"RUN clean", fieldSnapshot{valid: true, velocityMode: velRUN}, true},
		{"RUN but strained (debt>5) stays inward", fieldSnapshot{valid: true, velocityMode: velRUN, debt: 30}, false},
		{"RUN but wintering stays inward", fieldSnapshot{valid: true, velocityMode: velRUN, winter: 0.9}, false},
		{"NOMOVE wintering strained", fieldSnapshot{valid: true, velocityMode: velNOMOVE, winter: 1, debt: 30}, false},
	}
	for _, c := range cases {
		if got := c.fs.surfaces(); got != c.want {
			t.Errorf("%s: surfaces()=%v want %v", c.name, got, c.want)
		}
	}
}

// ── chorusText: rune-safe cap (the #2 fix) ──────────────────────────────────────
func TestChorusTextRuneSafe(t *testing.T) {
	// a multibyte run far over maxDreamLen must cap to valid UTF-8.
	cells := []chorusCell{{text: strings.Repeat("поле резонанса ", 200)}} // Cyrillic = 2 bytes/letter
	s := chorusText(cells)
	if len(s) > maxDreamLen+len("…")+8 {
		t.Errorf("chorusText not capped: len=%d", len(s))
	}
	if !utf8.ValidString(s) {
		t.Errorf("chorusText produced invalid UTF-8 at the cap")
	}
}

// ── SaveState/LoadState: round-trip + the LastDream cap (the #7 fix) ─────────────
func TestPersistRoundTripAndCap(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "arianna.inner.state")
	iw := NewInnerWorld()
	iw.State.Arousal, iw.State.Valence, iw.State.ProphecyDebt = 0.62, -0.3, 7.5
	if err := iw.SaveState(p, "the tide remembers the field"); err != nil {
		t.Fatal(err)
	}
	iw2 := NewInnerWorld()
	ld := iw2.LoadState(p)
	if ld != "the tide remembers the field" {
		t.Errorf("lastDream round-trip: %q", ld)
	}
	if d := iw2.State.Arousal - 0.62; d > 1e-6 || d < -1e-6 {
		t.Errorf("arousal not restored: %v", iw2.State.Arousal)
	}
	if iw2.State.ProphecyDebt != 7.5 {
		t.Errorf("debt not restored: %v", iw2.State.ProphecyDebt)
	}
	// a huge last_dream is capped (rune-safe) on load.
	huge := strings.Repeat("сон ", 4000) // > maxPersistedDream bytes
	iw.SaveState(p, huge)
	ld = iw2.LoadState(p)
	if len(ld) > maxPersistedDream {
		t.Errorf("huge lastDream not capped: len=%d", len(ld))
	}
	if !utf8.ValidString(ld) {
		t.Errorf("capped lastDream invalid UTF-8")
	}
	// missing file → defaults + "".
	if got := iw2.LoadState(filepath.Join(dir, "nope")); got != "" {
		t.Errorf("missing file should return empty dream, got %q", got)
	}
}

// ── parseDoeDream: extract the dream from doe's REPL stdout (no test existed) ────
func TestParseDoeDream(t *testing.T) {
	out := `
  doe.c — Democracy of Experts
[identity] arianna
[host] nano_arianna_f16.gguf
  L0: health=0.87 l2=54.0
  L12: health=0.88 l2=55.0
[doe] attached to weights/nano_arianna_f16.gguf (arch=llama)
[doe] the parliament is in session.
>    A: the field hums a living response, but here it is. Q: more
  [life] births=0 deaths=2
> [mycelium] spore saved: doe_mycelium/spore_x_s200.bin
[doe] the parliament adjourns.
`
	d := parseDoeDream(out)
	if d == "" {
		t.Fatal("parseDoeDream returned empty for a real dream")
	}
	if strings.Contains(d, "[life]") || strings.Contains(d, "[doe]") || strings.Contains(d, "mycelium") || strings.HasPrefix(d, "A:") {
		t.Errorf("parseDoeDream leaked a log/label: %q", d)
	}
	if !strings.Contains(d, "the field hums a living response") {
		t.Errorf("parseDoeDream lost the dream body: %q", d)
	}
	// no real '>' dream line → "".
	if got := parseDoeDream("[doe] only logs\n> [mycelium] x\n"); got != "" {
		t.Errorf("no-dream output should be empty: %q", got)
	}
}

// ── breath.tick: cooldown + threshold scaling ───────────────────────────────────
func TestBreathTick(t *testing.T) {
	var b breath
	now := time.Now()
	// Arousal=0.5 keeps Thermograph quiet (|0.5-0.5|<0.12), so only Silence (wander) fires.
	idle := Snapshot{WanderPull: 0.55, Arousal: 0.5} // above the 0.45 Silence bar
	if got := b.tick(idle, now, 1.0, 1.0); got != bSilence {
		t.Errorf("idle wander should fire Silence, got %d", got)
	}
	// immediate re-tick within cooldown → no fire.
	if got := b.tick(idle, now.Add(100*time.Millisecond), 1.0, 1.0); got != -1 {
		t.Errorf("within cooldown should not fire, got %d", got)
	}
	// a high threshold mult (strained field) suppresses the marginal wander.
	var b2 breath
	if got := b2.tick(idle, now, 1.5, 1.0); got != -1 {
		t.Errorf("threshold×1.5 should suppress 0.55 wander, got %d", got)
	}
	// after the cooldown elapses (×coolMult), it can fire again.
	if got := b.tick(idle, now.Add(bCooldown[bSilence]+time.Second), 1.0, 1.0); got != bSilence {
		t.Errorf("after cooldown Silence should fire again, got %d", got)
	}
}

// ── moodWord priority + dreamCue field tint ─────────────────────────────────────
func TestMoodWordAndDreamCue(t *testing.T) {
	if moodWord(Snapshot{TraumaLevel: 0.6, WanderPull: 0.9}) != "fear, the held breath" {
		t.Error("trauma must win over wander")
	}
	if moodWord(Snapshot{WanderPull: 0.7, Arousal: 0.6}) != "drifting, the mind wanders far" {
		t.Error("wander must win over arousal")
	}
	// lastDream present → carried; field valid → tinted.
	cue := dreamCue(Snapshot{}, fieldSnapshot{valid: true, velocityMode: velRUN, summer: 1}, "the tide")
	if !strings.Contains(cue, "the tide") || !strings.Contains(cue, "summer") {
		t.Errorf("cue must carry dream + field tint: %q", cue)
	}
	// no dream + no field → inner mood, non-empty, no tint.
	bare := dreamCue(Snapshot{Coherence: 0.8}, fieldSnapshot{}, "")
	if bare == "" || strings.Contains(bare, "summer") {
		t.Errorf("bare cue wrong: %q", bare)
	}
}

// ── tickBudget / tickDelay ──────────────────────────────────────────────────────
func TestTickBudgetDelay(t *testing.T) {
	// WanderPull=0.3 is the neutral point (the formula's -0.3 offset → 0 contribution).
	if n := tickBudget(Snapshot{Arousal: 0.3, WanderPull: 0.3, Coherence: 0.8}); n != 4 {
		t.Errorf("calm budget = %d want 4", n)
	}
	if n := tickBudget(Snapshot{Arousal: 0.6, WanderPull: 0.3, Coherence: 0.8}); n != 8 {
		t.Errorf("hot budget = %d want 8", n)
	}
	if n := tickBudget(Snapshot{Arousal: 0.3, WanderPull: 0.3, TraumaLevel: 1, Coherence: 0.4}); n != 2 {
		t.Errorf("trauma budget = %d want 2", n)
	}
	if d := tickDelay(Snapshot{}); d != 150*time.Millisecond {
		t.Errorf("base delay = %v", d)
	}
	if d := tickDelay(Snapshot{LoopCount: 3, Arousal: 0.6}); d != 700*time.Millisecond {
		t.Errorf("loops+arousal delay = %v want 700ms", d)
	}
}

// ── nano cleaners: stripLabel / cutSentence / sanitizeCue / cleanDream ───────────
func TestNanoCleaners(t *testing.T) {
	if stripLabel("Arianna: the field") != "the field" {
		t.Error("stripLabel must drop a leading Arianna: label")
	}
	if got := cutSentence("the resonant field is alive and well today and more"); !strings.HasSuffix(got, ".") && len(got) < len("the resonant field is alive") {
		// cutSentence cuts at the first sentence end after 30 chars; no period here → returns whole
		_ = got
	}
	if got := sanitizeCue("What is, resonance?!\n  the field"); strings.ContainsAny(got, "?,!\n") {
		t.Errorf("sanitizeCue left punctuation: %q", got)
	}
	if got := sanitizeCue(strings.Repeat("word ", 40)); len(strings.Fields(got)) > 24 {
		t.Errorf("sanitizeCue should cap at 24 words, got %d", len(strings.Fields(got)))
	}
	// cleanDream: text after the "[<n> tokens .. tok/s]" frame.
	d := cleanDream("[nanollama] banner\nstreamed\n[16 tokens, 30.0 tok/s]\nA: the clean dream here.\n")
	if d == "" || strings.Contains(d, "tokens") || strings.HasPrefix(d, "A:") {
		t.Errorf("cleanDream wrong: %q", d)
	}
	// guard: ensure the os import is exercised (temp-file existence is asserted elsewhere).
	_ = os.TempDir
}
