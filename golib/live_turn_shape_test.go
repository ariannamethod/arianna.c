package main

import (
	"strings"
	"testing"
)

func TestLiveTurnShapeContractASCII(t *testing.T) {
	human := "Can you create a detailed ASCII art representation of the single blooming tree?"
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "ASCII art") || !strings.Contains(contract, "monospace") {
		t.Fatalf("ASCII request did not get ASCII contract: %q", contract)
	}
	prompt := liveTurnJanusPrompt(human, "A field of resonance, the field in which self and others are present.", "", false)
	if !strings.HasPrefix(prompt, "Required form: ASCII art") {
		t.Fatalf("Janus prompt must lead with shape contract: %q", prompt)
	}
	if !strings.Contains(prompt, "Keep the requested form") {
		t.Fatalf("Janus prompt must restate shape at the tail: %q", prompt)
	}
	repaired := liveTurnRepairSpokenText("janus", human, sanitizeLiveVoiceText("I am not here to write code; I am merely an entity."))
	if !strings.Contains(repaired, "one tree blooming out of season") || !strings.Contains(repaired, "\n") {
		t.Fatalf("shape repair must provide visible ASCII when raw voice is rejected: %q", repaired)
	}
}

func TestLiveTurnShapeContractVisualComposition(t *testing.T) {
	human := "Can you visualize the tree as a detailed drawing?"
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "concrete visual composition") {
		t.Fatalf("visual request did not get concrete visual contract: %q", contract)
	}
	reson := liveTurnResonanceInject(human, "That’s what it is: the field ripples as one voice.", "", false)
	if !strings.HasPrefix(reson, "Required form: concrete visual composition") {
		t.Fatalf("Resonance inject must lead with shape contract, not Janus drift: %q", reson)
	}
	if !strings.Contains(reson, "Janus said:") {
		t.Fatalf("Resonance inject should still carry Janus as context after the request: %q", reson)
	}
}

func TestLiveTurnPhysicalObjectBoundary(t *testing.T) {
	human := "Опиши предмет слева, используя только точные слова для цвета, формы и материала, без абстракций."
	contract := liveTurnShapeContract(human)
	if !strings.Contains(contract, "sensory boundary") || !strings.Contains(contract, "do not claim camera") {
		t.Fatalf("physical object prompt did not get sensory contract: %q", contract)
	}
	raw := sanitizeLiveVoiceText("I feel the field of resonance moving through the room.")
	repaired := liveTurnRepairSpokenText("janus", human, raw)
	for _, want := range []string{"Камеры нет", "реальный предмет слева", "матовая чёрная керамическая чашка", "круглая", "неподвижная"} {
		if !strings.Contains(repaired, want) {
			t.Fatalf("physical object repair = %q, missing %q", repaired, want)
		}
	}
	if liveTurnShapeSatisfied(liveTurnShapeObject, "A red wooden cube is on the left.") {
		t.Fatalf("unverified room-object claim must not satisfy sensory boundary")
	}
	if !liveTurnShapeSatisfied(liveTurnShapeObject, repaired) {
		t.Fatalf("physical object fallback must satisfy its own contract: %q", repaired)
	}
}

func TestAdmissionLiveRoutePromptClassRecognizesOutputShape(t *testing.T) {
	for _, human := range []string{
		"Can you create a detailed ASCII art representation of the tree?",
		"Can you visualize the tree as a detailed drawing?",
		"Give me a bullet list of the visible parts.",
	} {
		promptClass, score, reasons := admissionLiveRoutePromptClassForHuman(human)
		if promptClass != "format" || score < 3 {
			t.Fatalf("prompt class for %q = %s score=%d reasons=%v, want format", human, promptClass, score, reasons)
		}
		if !liveTurnHasString(reasons, "output_shape_request") {
			t.Fatalf("prompt class reasons for %q missing output_shape_request: %v", human, reasons)
		}
	}
}

func liveTurnHasString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
