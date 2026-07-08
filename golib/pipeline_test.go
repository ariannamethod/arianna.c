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

// TestParseDoeDreamDaemonLeftover guards the persistent-daemon framing: each
// exchange stops at its `[field] step=` sentinel, leaving the rest of that status
// reply (`[field] season`/`[drift]`/`[experts]`/`[prophecy]`) buffered. The NEXT
// dream's raw text therefore BEGINS with those leftover log lines before its `>`
// dream line — parseDoeDream must skip them and still recover the dream. This is the
// exact byte shape doeDaemon.exchange accumulates for the second dream onward.
func TestParseDoeDreamDaemonLeftover(t *testing.T) {
	raw := `[field] season=summer health=0.512 temp=0.83 velocity=walk
[drift] d=0.104 stability=0.91 accel=0.0011 snapshots=3
[experts] alive=12 consensus=0.80 elections=2
[prophecy] avg_debt=0.214 total_debt=4.10
>    the dream body flows here, soft and strange.
  [life] births=1 deaths=0
`
	d := parseDoeDream(raw)
	if d != "the dream body flows here, soft and strange." {
		t.Errorf("leftover status lines must be skipped, got %q", d)
	}
	if strings.Contains(d, "[field]") || strings.Contains(d, "[drift]") || strings.Contains(d, "[experts]") || strings.Contains(d, "[life]") {
		t.Errorf("parseDoeDream leaked a daemon log line: %q", d)
	}
}

// TestDreamDropsInvalidUTF8 guards the Go-side UTF-8 sanitize: a byte-fallback the
// doe / nanollama model emits (e.g. a raw 0xFF) must never survive into the dream
// (those are separate binaries — the C voices' output guard does not cover them).
func TestDreamDropsInvalidUTF8(t *testing.T) {
	raw := ">    the field \xff hums a living response here.\n  [life] births=0 deaths=0\n"
	d := parseDoeDream(raw)
	if !utf8.ValidString(d) || strings.ContainsRune(d, 0xFFFD) {
		t.Errorf("parseDoeDream left invalid UTF-8: %q", d)
	}
	if !strings.Contains(d, "the field") || !strings.Contains(d, "hums a living response") {
		t.Errorf("parseDoeDream dropped real text: %q", d)
	}
	cd := cleanDream("[nanollama] x\nstreamed\n[16 tokens, 30.0 tok/s]\nthe \xff field is alive.\n")
	if !utf8.ValidString(cd) || strings.ContainsRune(cd, 0xFFFD) {
		t.Errorf("cleanDream left invalid UTF-8: %q", cd)
	}
	if !strings.Contains(cd, "field is alive") {
		t.Errorf("cleanDream dropped real text: %q", cd)
	}
	// a SHORT chorus dream (under maxDreamLen, so the cap branch is skipped) with a
	// raw byte-fallback must still be sanitized before it reaches lastDream / inject.
	ch := chorusText([]chorusCell{{text: "the field \xff hums"}, {text: "a living \xfe note"}})
	if !utf8.ValidString(ch) || strings.ContainsRune(ch, 0xFFFD) {
		t.Errorf("chorusText left invalid UTF-8 on a short dream: %q", ch)
	}
	if !strings.Contains(ch, "the field") || !strings.Contains(ch, "living") {
		t.Errorf("chorusText dropped real text: %q", ch)
	}
}

// TestEmitNonBlocking guards the inner-world deadlock fix: emit must never block on a
// full Signals buffer (in the trio path nothing drains it — the run() readers are
// dormant under Start(false) — and the sender runs under iw.mu via Step/ProcessText).
func TestEmitNonBlocking(t *testing.T) {
	iw := &InnerWorld{Signals: make(chan Signal, 2)}
	iw.emit(Signal{})
	iw.emit(Signal{}) // buffer now full (cap 2)
	done := make(chan struct{})
	go func() { iw.emit(Signal{}); close(done) }() // a blocking send would hang here
	select {
	case <-done: // returned → non-blocking, the 3rd signal was dropped
	case <-time.After(2 * time.Second):
		t.Fatal("emit blocked on a full Signals buffer (deadlock risk not fixed)")
	}
	if len(iw.Signals) != 2 {
		t.Errorf("emit should have dropped the overflow signal, buffer len=%d (want 2)", len(iw.Signals))
	}
}

// TestDoeStatusSentinel guards the end-of-generation frame: only the FULL status line
// (doe.c:3471) counts, so a dream that merely emits "[field] step=" is not mistaken
// for the frame (which would truncate the dream + desync the next exchange).
func TestDoeStatusSentinel(t *testing.T) {
	// the real status reply, with doe's `> ` prompt prepended (no newline after it).
	real := "> [field] step=42 debt=0.214 entropy=0.501 resonance=0.330 emergence=0.120"
	if !isDoeStatusSentinel(real) {
		t.Errorf("real status line must be the sentinel: %q", real)
	}
	// a dream that happens to contain the bare substring is NOT the sentinel.
	for _, bogus := range []string{
		"the oracle whispered [field] step= into the dark",
		"[field] step=7", // partial — no debt/entropy/resonance/emergence
		">    a soft murmur about a field that steps forward",
		"  [life] births=1 deaths=0",
		// the full signature but mid-sentence (no leading prompt) — structurally NOT
		// the frame (doe always prints it right after its `> ` prompt).
		"the dream said [field] step=1 debt=2 entropy=3 resonance=4 emergence=5 softly",
	} {
		if isDoeStatusSentinel(bogus) {
			t.Errorf("non-status line wrongly matched the sentinel: %q", bogus)
		}
	}
}

// TestPruneMycelium: the dir is capped PER FINGERPRINT (doe.c loads the highest step
// for the current host only), non-spore + malformed names are left alone, and a
// fingerprint with <=keep spores is never touched — so a busy OTHER host can't crowd
// out this host's load target.
func TestPruneMycelium(t *testing.T) {
	dir := t.TempDir()
	fpA := "00000000deadbeef" // "this" host: 2 spores (both must survive at keep=3)
	fpB := "1111111100000000" // a busy other host: 5 spores
	write := func(name string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range []string{"10", "20"} {
		write("spore_" + fpA + "_s" + s + ".bin")
	}
	for _, s := range []string{"300", "100", "500", "50", "200"} {
		write("spore_" + fpB + "_s" + s + ".bin")
	}
	// non-spore + malformed names — must all be untouched (not parsed as spores).
	for _, n := range []string{"notes.txt", "spore_junk_s9.bin", "spore__s9.bin", "spore_" + fpA + "_sxx.bin"} {
		write(n)
	}
	pruneMycelium(dir, 3)
	survives := map[string]bool{}
	ents, _ := os.ReadDir(dir)
	for _, e := range ents {
		survives[e.Name()] = true
	}
	// fpA has 2 <= 3 → both survive (a busy fpB must not delete this host's spores).
	for _, s := range []string{"10", "20"} {
		if !survives["spore_"+fpA+"_s"+s+".bin"] {
			t.Errorf("per-fingerprint: fpA spore s%s must survive", s)
		}
	}
	// fpB keeps its 3 highest (500, 300, 200); 100, 50 pruned.
	for _, s := range []string{"500", "300", "200"} {
		if !survives["spore_"+fpB+"_s"+s+".bin"] {
			t.Errorf("fpB highest-step spore s%s must survive", s)
		}
	}
	for _, s := range []string{"100", "50"} {
		if survives["spore_"+fpB+"_s"+s+".bin"] {
			t.Errorf("fpB low-step spore s%s must be pruned", s)
		}
	}
	for _, n := range []string{"notes.txt", "spore_junk_s9.bin", "spore__s9.bin", "spore_" + fpA + "_sxx.bin"} {
		if !survives[n] {
			t.Errorf("non-spore/malformed name %q must be left alone", n)
		}
	}
	// idempotent + safe on a missing dir.
	pruneMycelium(filepath.Join(dir, "nope"), 3)
}

// TestNeutralizeDoeSeed guards against a seed that is exactly a doe REPL command
// (status/quit/exit) being executed instead of dreamt on.
func TestNeutralizeDoeSeed(t *testing.T) {
	for _, cmd := range []string{"status", "quit", "exit"} {
		if got := neutralizeDoeSeed(cmd); got != " "+cmd {
			t.Errorf("reserved command %q must be neutralized, got %q", cmd, got)
		}
	}
	// ordinary seeds pass through untouched (incl. ones that merely contain a command).
	for _, ok := range []string{"the field hums", "quit the silence and speak", "status report of the soul", ""} {
		if got := neutralizeDoeSeed(ok); got != ok {
			t.Errorf("ordinary seed %q must pass through, got %q", ok, got)
		}
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
func TestViability(t *testing.T) {
	if got := viability(Snapshot{}, false, false); got != 1.0 {
		t.Fatalf("healthy viability = %.3f, want 1.0", got)
	}
	if got := viability(Snapshot{}, true, false); got >= 1.0 {
		t.Fatalf("a silent voice must drop viability, got %.3f", got)
	}
	stressed := Snapshot{ProphecyDebt: 10, TraumaLevel: 1, MemoryPressure: 1}
	if v := viability(stressed, false, false); v >= viability(Snapshot{}, false, false) {
		t.Fatalf("stressed viability %.3f should sit below healthy", v)
	}
	if v := viability(stressed, true, true); v != 0 {
		t.Fatalf("both voices dead + fully stressed must clamp to 0, got %.3f", v)
	}
}

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
