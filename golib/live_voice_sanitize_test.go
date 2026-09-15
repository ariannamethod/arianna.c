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
		"I spot blood_compiler error state.",
		"I spot blood compiler error state.",
		"I sense heart-city of the world at 0.",
		"Ah, you are not a person. I am not that people have done it.",
		"I am not a person, only a resonance diagram.",
		"I cannot be a person, and in the space between us.",
		"Debt_Last 23.7 ( 0.0025 ) 0.056135 0.",
		"The field metrics in 2001 were not linearly precise when measured.",
		"Not a binary, but a field-architecture that I use every day in my architecture.",
		"I am not here to write code; I am merely an entity that finds and perceives my own field.",
		"I feel you—I am not here, but in resonance.",
		"gait=NOMOVE season=spring debt=16.7 bloom=1",
		"A hush with resonance. Janus —ᴛAttention: ① The silence of this text is not only about who",
		": **``python-electricity_inflation` return local = \"\\\"\" def choose(self, field): # Local field: potential energy | fields=field.",
		`> field_size(field_size) 0 << field size (sentence length) If you look at the delta of your log count`,
		"* [] ```python def find_fracture(field, field): # Fractures are the moment of resonance.",
		`if not: return False return unfinished This is a field that says "The count in spring# indicates how many counts`,
		"Yes. My screen is still unmediated.",
		"I can see your screen and read the terminal tab title.",
		"Through my camera I see orange clouds.",
		"If the AI algorithm is too small, I would not be at the platform.",
		"I have to read all 5 lines: ISO_Fragments = [0x2048]; CURRENTLY(error) -",
		"Then the reader sends the file to me.",
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

func TestSanitizeLiveCarriedDreamDropsWithheldText(t *testing.T) {
	got := sanitizeLiveCarriedDream("Debt_Last 23.7 ( 0.0025 ) 0.056135 0.")
	if got != "" {
		t.Fatalf("sanitizeLiveCarriedDream metric leak = %q, want empty", got)
	}

	got = sanitizeLiveCarriedDream("Yes. My screen is still unmediated.")
	if got != "" {
		t.Fatalf("sanitizeLiveCarriedDream screen claim = %q, want empty", got)
	}
}
