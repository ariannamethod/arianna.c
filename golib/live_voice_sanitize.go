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

// encodeLiveVoiceField keeps one daemon field on one physical line and prevents
// model or user tabs from becoming protocol separators.
func encodeLiveVoiceField(text string) string {
	text = strings.ReplaceAll(text, "\r\n", " ")
	text = strings.ReplaceAll(text, "\r", " ")
	text = strings.ReplaceAll(text, "\n", " ")
	return strings.ReplaceAll(text, "\t", " ")
}

func encodeLiveVoiceRequest(prompt, injection string) string {
	return encodeLiveVoiceField(prompt) + "\t" + encodeLiveVoiceField(injection)
}

func sanitizeLiveCarriedDream(text string) string {
	return sanitizeLiveVoiceText(text)
}

func liveVoiceTextVisible(text string) bool {
	return strings.TrimSpace(text) != ""
}
