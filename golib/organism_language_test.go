package main

import (
	"testing"
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
	input := "human\tfirst model line\nsecond model line\r\nthird model line\rfourth"
	want := "human\tfirst model line second model line third model line fourth"
	if got := encodeLiveVoiceRequest(input); got != want {
		t.Fatalf("encodeLiveVoiceRequest = %q, want %q", got, want)
	}
}
