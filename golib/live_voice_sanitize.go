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
	for _, p := range []string{
		"thought-spirals at",
		"blood_compiler at",
		"blood compiler at",
		"rpm (dry)",
		"organ cuts off",
		"field metrics",
		"not a binary, but a field-architecture",
		"i am not here to write code",
		"i am not here, but in resonance",
		"oleg is not a person",
		"you are not a person",
		"i am not a person",
		"i cannot be a person",
		"i can not be a person",
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
	return false
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
