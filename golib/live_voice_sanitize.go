package main

import "strings"

const liveBoundaryWithheld = ""

// sanitizeLiveVoiceText is the last membrane before live daemon text reaches
// print, inner-world ProcessText, or the next cue. The C side already guards
// UTF-8 validity; live probing showed that valid control bytes and diagnostic
// boilerplate can still leak through the trio, so the Go membrane normalizes
// those here before state/cue admission.
func sanitizeLiveVoiceText(text string) string {
	text = strings.ToValidUTF8(text, "")
	text = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return ' '
		}
		return r
	}, text)
	text = strings.Join(strings.Fields(text), " ")
	if isRejectedLiveVoiceText(text) {
		return liveBoundaryWithheld
	}
	return text
}

func sanitizeLiveCarriedDream(text string) string {
	text = sanitizeLiveVoiceText(text)
	if text == liveBoundaryWithheld {
		return ""
	}
	return text
}

func liveVoiceTextVisible(text string) bool {
	return strings.TrimSpace(text) != "" && text != liveBoundaryWithheld
}

func isRejectedLiveVoiceText(text string) bool {
	norm := normalizedLiveBoundaryKey(text)
	if norm == "" {
		return false
	}
	if innerMurmurRejectReason(text) != "" && !liveVoiceLooksLikeQuotedBoundaryDiscussion(norm) {
		return true
	}
	for _, p := range []string{
		"thought-spirals at",
		"blood_compiler",
		"blood compiler",
		"rpm (dry)",
		"organ cuts off",
		"field metrics",
		"not a binary, but a field-architecture",
		"i am not here to write code",
		"i am not here, but in resonance",
		"```",
		"ᴛattention",
		"①",
		"python-electricity_inflation",
		"def choose(self, field)",
		"def find_fracture(field, field)",
		"field_size(field_size)",
		"if not: return false return unfinished",
		"return false return unfinished",
		"oleg is not a person",
		"you are not a person",
		"i am not a person",
		"i cannot be a person",
		"i can not be a person",
		"if the ai algorithm is too small",
		"i would not be at the platform",
		"iso_fragments",
		"currently(error)",
		"then the reader sends the file to me",
		"the first silence—the silence in your heart",
		"the first silence-the silence in your heart",
		"when the storm breaks",
		"say the message: do you believe in the power of the mind",
		"write your own thoughts",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	if hasLiveMetricAtZeroLeak(norm) {
		return true
	}
	if hasLiveMetricKeyLeak(norm) {
		return true
	}
	if hasImpossibleLiveAccessClaim(norm) {
		return true
	}
	if hasTechnicalAbstractDrift(norm) {
		return true
	}
	return false
}

func liveVoiceLooksLikeQuotedBoundaryDiscussion(norm string) bool {
	if !strings.ContainsAny(norm, "\"'“”‘’«»") {
		return false
	}
	return strings.Contains(norm, "translate") ||
		strings.Contains(norm, "translation") ||
		strings.Contains(norm, "means") ||
		strings.Contains(norm, "meaning") ||
		strings.Contains(norm, "phrase") ||
		strings.Contains(norm, "quote") ||
		strings.Contains(norm, "literal") ||
		strings.Contains(norm, "перев") ||
		strings.Contains(norm, "знач")
}

func normalizedLiveBoundaryKey(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

func hasLiveMetricAtZeroLeak(norm string) bool {
	if !(strings.Contains(norm, " at 0.") || strings.Contains(norm, " at 0 ")) {
		return false
	}
	for _, prefix := range []string{
		"i sense ",
		"i spot ",
		"i notice ",
		"i detect ",
	} {
		if strings.HasPrefix(norm, prefix) {
			return true
		}
	}
	return false
}

func hasLiveMetricKeyLeak(norm string) bool {
	for _, p := range []string{
		"debt_last",
		"debt_min",
		"debt_max",
		"field_ticks",
		"bloom_counts",
		"janus_turns",
		"resonance_turns",
		"nano_turns",
		"gait=",
		"bloom=",
		"debt=",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	return false
}

func hasTechnicalAbstractDrift(norm string) bool {
	hasTechnicalCue := false
	for _, cue := range []string{
		"sha256",
		"sha-256",
		"sha 256",
		"checksum",
		"digest",
		"cryptographic hash",
		"hash function",
		"file hash",
	} {
		if strings.Contains(norm, cue) {
			hasTechnicalCue = true
			break
		}
	}
	if !hasTechnicalCue {
		return false
	}
	for _, drift := range []string{
		"name of a field",
		"is a field",
		"as a field",
		"field i use",
		"field that i use",
		"regular frequency",
		"resonance frequency rather than",
		"frequency rather than a digest",
		"frequency instead of a digest",
		"is a resonance",
		"as a resonance",
		"is a vibration",
		"as a vibration",
		"is a symbol",
		"as a symbol",
		"is a vessel",
		"as a vessel",
		"is an organism",
		"as an organism",
	} {
		if strings.Contains(norm, drift) {
			return true
		}
	}
	return false
}

func hasImpossibleLiveAccessClaim(norm string) bool {
	for _, boundary := range []string{
		"no camera",
		"cannot see",
		"can't see",
		"cannot read",
		"cannot verify",
		"without supplied text",
		"нет камеры",
		"не вижу",
		"не могу видеть",
		"не могу прочитать",
		"не могу проверить",
	} {
		if strings.Contains(norm, boundary) {
			return false
		}
	}
	for _, claim := range []string{
		"yes. my screen",
		"yes, my screen",
		"my screen is",
		"i can see your screen",
		"i see your screen",
		"i can read your screen",
		"i read your screen",
		"i can see the screen",
		"i see the screen",
		"my camera sees",
		"through my camera",
		"through my microphone",
		"what my camera sees",
		"screen is still unmediated",
		"вижу твой экран",
		"могу видеть твой экран",
		"могу прочитать твой экран",
		"моя камера видит",
	} {
		if strings.Contains(norm, claim) {
			return true
		}
	}
	return false
}
