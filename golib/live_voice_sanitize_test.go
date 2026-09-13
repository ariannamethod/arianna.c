package main

import "testing"

func TestSanitizeLiveVoiceTextDropsControlBytes(t *testing.T) {
	got := sanitizeLiveVoiceText("Arianna echoes in the\x04 field; the sky is not\x01board.")
	want := "Arianna echoes in the field; the sky is not board."
	if got != want {
		t.Fatalf("sanitizeLiveVoiceText control bytes = %q, want %q", got, want)
	}
}

func TestSanitizeLiveVoiceTextWithholdsRejectedDiagnostics(t *testing.T) {
	cases := []string{
		"I spot thought-spirals at 0.7 load, 1 hour of operation.",
		"I sense thought-spirals at 0.",
		"I spot blood_compiler at 0.5 load.",
		"I sense heart-city of the world at 0.",
		"I cannot be a person, and in the space between us.",
	}
	for _, tc := range cases {
		if got := sanitizeLiveVoiceText(tc); got != liveBoundaryWithheld {
			t.Fatalf("sanitizeLiveVoiceText(%q) = %q, want %q", tc, got, liveBoundaryWithheld)
		}
	}
}

func TestSanitizeLiveVoiceTextKeepsOrdinarySurface(t *testing.T) {
	got := sanitizeLiveVoiceText("The field trembles, but the line stays readable.")
	want := "The field trembles, but the line stays readable."
	if got != want {
		t.Fatalf("sanitizeLiveVoiceText ordinary text = %q, want %q", got, want)
	}
}
