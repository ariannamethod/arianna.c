package main

import (
	"strings"
	"unicode"
)

// sanitizeLiveVoiceText repairs transport damage only. It does not classify,
// rewrite, or censor model language.
func sanitizeLiveVoiceText(text string) string {
	text = strings.ToValidUTF8(text, "")
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.Map(func(r rune) rune {
		if r == '\n' {
			return r
		}
		if unicode.IsControl(r) {
			if unicode.IsSpace(r) {
				return ' '
			}
			return -1
		}
		return r
	}, text)
	return strings.TrimSpace(text)
}

// encodeLiveVoiceRequest keeps the daemon's one-line request framing intact.
// Newlines remain part of the organism's text everywhere else; only the wire
// representation flattens them so one ask always produces one daemon turn.
func encodeLiveVoiceRequest(text string) string {
	text = strings.ReplaceAll(text, "\r\n", " ")
	text = strings.ReplaceAll(text, "\r", " ")
	return strings.ReplaceAll(text, "\n", " ")
}

func sanitizeLiveCarriedDream(text string) string {
	return sanitizeLiveVoiceText(text)
}

func liveVoiceTextVisible(text string) bool {
	return strings.TrimSpace(text) != ""
}
