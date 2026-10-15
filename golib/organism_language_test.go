package main

import (
	"bufio"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestLiveVoiceSanitizerPreservesModelLanguage(t *testing.T) {
	input := "  Arianna\x04 says:\tthe field\vis not\fa wall, but an unfinished echo.  "
	want := "Arianna says: the field is not a wall, but an unfinished echo."
	if got := sanitizeLiveVoiceText(input); got != want {
		t.Fatalf("sanitizeLiveVoiceText = %q, want %q", got, want)
	}
	for _, text := range []string{
		"I am Arianna.",
		"To me, the Method is a living field.",
		"I sense wormhole.",
		"As an AI, I choose my own next sentence.",
		"func breathe() {}",
	} {
		if got := sanitizeLiveVoiceText(text); got == "" {
			t.Fatalf("model language was withheld: %q", text)
		}
	}
	framed := "Memory: [alive]\nA: the second line"
	if got := sanitizeLiveVoiceText(framed); got != framed {
		t.Fatalf("model framing was rewritten: %q", got)
	}
}

func TestLiveVoiceRequestUsesOneDaemonLine(t *testing.T) {
	prompt := "human\ttext\ncontinues"
	injection := "first model line\nsecond\tmodel line\r\nthird model line\rfourth"
	want := "human text continues\tfirst model line second model line third model line fourth"
	if got := encodeLiveVoiceRequest(prompt, injection); got != want {
		t.Fatalf("encodeLiveVoiceRequest = %q, want %q", got, want)
	}
}

func TestLiveVoiceFrameCannotCollideWithModelText(t *testing.T) {
	payload := "first line\n<END>\nlast line"
	frame := "AMF1 26\n" + payload
	got, complete := readLiveVoiceFrame(bufio.NewReader(strings.NewReader(frame)))
	if !complete || got != payload {
		t.Fatalf("readLiveVoiceFrame = %q, %v; want %q, true", got, complete, payload)
	}
}

func TestStartVoiceRejectsMalformedReadinessWithoutHanging(t *testing.T) {
	done := make(chan error, 1)
	go func() {
		_, err := startVoice("/bin/sh", []string{"-c", "printf '<END>\\n'; while read line; do :; done"})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("malformed readiness frame was accepted")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("startVoice hung while rejecting malformed readiness")
	}
}

func TestLanguageTokensSplitPunctuationWithoutJoiningWords(t *testing.T) {
	want := []string{"river", "stone", "река", "камень", "שָׁלוֹם", "café", "café"}
	if got := languageTokens("river—stone / река,камень שָׁלוֹם café cafe\u0301"); !reflect.DeepEqual(got, want) {
		t.Fatalf("languageTokens = %#v, want %#v", got, want)
	}
}

func TestRepetitionUsesConcentrationNotRawTokenCount(t *testing.T) {
	ol := NewOverthinkingLoops()
	ordinary := "x alpha beta x gamma delta x epsilon zeta x eta theta x iota kappa lambda mu nu xi omicron pi rho sigma tau"
	if got := ol.AnalyzeText(ordinary).RepetitionScore; got > 0.5 {
		t.Fatalf("ordinary distributed repetition scored as a loop: %v", got)
	}
	spiral := "echo echo echo echo echo elsewhere"
	if got := ol.AnalyzeText(spiral).RepetitionScore; got <= 0.5 {
		t.Fatalf("concentrated repetition was not detected: %v", got)
	}
	markedSpiral := "שָׁלוֹם שָׁלוֹם שָׁלוֹם שָׁלוֹם שָׁלוֹם אחר"
	if got := ol.AnalyzeText(markedSpiral).RepetitionScore; got <= 0.5 {
		t.Fatalf("combining-mark repetition was not detected: %v", got)
	}
}

func TestMemoryConnectionsUseCorpusFrequency(t *testing.T) {
	if got := countSignificantOverlap([]string{"river"}, []string{"river", "river"}, map[string]int{"river": 1}, 4); got != 1 {
		t.Fatalf("significant overlap counted one shared token %d times", got)
	}
	mc := NewMemoryConsolidation()
	for _, content := range []string{
		"the river and moon",
		"the bridge and stone",
		"the river and moon returns",
		"the cedar and rain",
	} {
		mc.longTerm = append(mc.longTerm, ConsolidatableMemory{Content: content})
	}
	mc.rebuildConnections()
	if len(mc.longTerm[0].Connections) != 1 || mc.longTerm[0].Connections[0] != 2 {
		t.Fatalf("frequency-weighted connections = %#v, want only memory 2", mc.longTerm[0].Connections)
	}
	if len(mc.longTerm[1].Connections) != 0 {
		t.Fatalf("ubiquitous tokens created a false connection: %#v", mc.longTerm[1].Connections)
	}
	small := NewMemoryConsolidation()
	for _, content := range []string{"river moon rises", "river moon returns"} {
		small.longTerm = append(small.longTerm, ConsolidatableMemory{Content: content})
	}
	small.rebuildConnections()
	if len(small.longTerm[0].Connections) != 1 || small.longTerm[0].Connections[0] != 1 {
		t.Fatalf("small corpus lost its shared structure: %#v", small.longTerm[0].Connections)
	}
	common := NewMemoryConsolidation()
	for _, content := range []string{"the river and moon", "the bridge and stone"} {
		common.longTerm = append(common.longTerm, ConsolidatableMemory{Content: content})
	}
	common.rebuildConnections()
	if len(common.longTerm[0].Connections) != 0 {
		t.Fatalf("small corpus linked disconnected common tokens: %#v", common.longTerm[0].Connections)
	}
}

func TestStructuralAnalysisKeepsNonLexicalSignalsLive(t *testing.T) {
	if got := structuralNestingScore("outer(inner[core{point}])"); got <= 0.5 {
		t.Fatalf("nested structure score = %v", got)
	}
	words := languageTokens("field turns field turns field turns elsewhere")
	if got := structuralRecurrenceScore(words); got <= 0.5 {
		t.Fatalf("recursive structure score = %v", got)
	}
	world := NewInnerWorld()
	trauma := NewTraumaSurfacing()
	trauma.Start(world)
	world.State.SetTraumaLevel(0.625)
	if got := trauma.CheckText("arbitrary vocabulary"); got != 0.625 {
		t.Fatalf("body trauma activation = %v, want 0.625", got)
	}
	trauma.Stop()
}

func TestStructuralCountersDecayAtRuntimeStepSizes(t *testing.T) {
	world := NewInnerWorld()
	ol := NewOverthinkingLoops()
	ol.world = world
	world.State.AbstractionDepth = 4
	world.State.SelfRefCount = 4
	for i := 0; i < 6; i++ {
		ol.Step(0.1)
	}
	if world.State.AbstractionDepth != 3 || world.State.SelfRefCount != 2 {
		t.Fatalf("structural counter cadence mismatch: abstraction=%d self_ref=%d",
			world.State.AbstractionDepth, world.State.SelfRefCount)
	}
	world.State.LoopCount = 4
	if got := ol.BreakSignal(); got <= 0 {
		t.Fatalf("live break signal = %v", got)
	}
	ol.Step(float32(math.Inf(1)))
	ol.Step(math.MaxFloat32)
	if world.State.AbstractionDepth != 0 || world.State.SelfRefCount != 0 {
		t.Fatalf("large finite step did not decay in bounded time: abstraction=%d self_ref=%d",
			world.State.AbstractionDepth, world.State.SelfRefCount)
	}
}
