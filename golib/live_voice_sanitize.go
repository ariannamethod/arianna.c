package main

import "strings"

const liveBoundaryWithheld = "(withheld by live boundary)"

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
		"oleg is not a person",
		"i cannot be a person",
		"i can not be a person",
	} {
		if strings.Contains(norm, p) {
			return true
		}
	}
	return false
}

func normalizedLiveBoundaryKey(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}
