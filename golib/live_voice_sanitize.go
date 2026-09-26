package main

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

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
	remainder, found := liveVoiceStripQuotedRejectedBoundarySpans(norm)
	return found && liveVoiceHasDiscussionCue(remainder) && innerMurmurRejectReason(remainder) == ""
}

func liveVoiceHasDiscussionCue(norm string) bool {
	for _, word := range strings.FieldsFunc(norm, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		switch word {
		case "translate", "translation", "means", "meaning", "phrase", "quote", "quoted", "literal":
			return true
		}
		if strings.HasPrefix(word, "перев") ||
			strings.HasPrefix(word, "знач") ||
			strings.HasPrefix(word, "означ") {
			return true
		}
	}
	return false
}

func liveVoiceStripQuotedRejectedBoundarySpans(norm string) (string, bool) {
	remainder := norm
	found := false
	for _, pair := range [][2]string{
		{`"`, `"`},
		{"“", "”"},
		{"‘", "’"},
		{"«", "»"},
	} {
		next, stripped := liveVoiceStripRejectedSpansForPair(remainder, pair[0], pair[1])
		remainder = next
		found = found || stripped
	}
	next, stripped := liveVoiceStripRejectedStandaloneSingleQuoteSpans(remainder)
	remainder = next
	found = found || stripped
	return remainder, found
}

func liveVoiceStripRejectedSpansForPair(s, open, close string) (string, bool) {
	var b strings.Builder
	rest := s
	stripped := false
	for {
		start := liveVoiceIndexOpeningQuote(rest, open)
		if start < 0 {
			b.WriteString(rest)
			break
		}
		b.WriteString(rest[:start])
		afterStart := rest[start+len(open):]
		end := liveVoiceIndexClosingQuote(afterStart, close)
		if end < 0 {
			b.WriteString(rest[start:])
			break
		}
		span := strings.TrimSpace(afterStart[:end])
		fullEnd := start + len(open) + end + len(close)
		if span != "" && innerMurmurRejectReason(span) != "" {
			b.WriteByte(' ')
			stripped = true
		} else {
			b.WriteString(rest[start:fullEnd])
		}
		rest = rest[fullEnd:]
	}
	return normalizedLiveBoundaryKey(b.String()), stripped
}

func liveVoiceIndexOpeningQuote(s, quote string) int {
	if quote != `"` {
		return strings.Index(s, quote)
	}
	for offset := 0; offset < len(s); {
		idx := strings.Index(s[offset:], quote)
		if idx < 0 {
			return -1
		}
		idx += offset
		prev, hasPrev := liveVoicePrevRune(s, idx)
		if !hasPrev || !liveVoiceRuneIsWord(prev) {
			return idx
		}
		offset = idx + len(quote)
	}
	return -1
}

func liveVoiceIndexClosingQuote(s, quote string) int {
	if quote != `"` {
		return strings.Index(s, quote)
	}
	for offset := 0; offset < len(s); {
		idx := strings.Index(s[offset:], quote)
		if idx < 0 {
			return -1
		}
		idx += offset
		next, hasNext := liveVoiceNextRune(s, idx+len(quote))
		if !hasNext || !liveVoiceRuneIsWord(next) {
			return idx
		}
		offset = idx + len(quote)
	}
	return -1
}

func liveVoiceStripRejectedStandaloneSingleQuoteSpans(s string) (string, bool) {
	var b strings.Builder
	rest := s
	stripped := false
	for {
		start := liveVoiceIndexOpeningSingleQuote(rest)
		if start < 0 {
			b.WriteString(rest)
			break
		}
		b.WriteString(rest[:start])
		afterStart := rest[start+1:]
		end := liveVoiceIndexClosingSingleQuote(afterStart)
		if end < 0 {
			b.WriteString(rest[start:])
			break
		}
		span := strings.TrimSpace(afterStart[:end])
		if span != "" && innerMurmurRejectReason(span) != "" {
			b.WriteByte(' ')
			stripped = true
		} else {
			b.WriteString(rest[start : start+1+end+1])
		}
		rest = afterStart[end+1:]
	}
	return normalizedLiveBoundaryKey(b.String()), stripped
}

func liveVoiceIndexOpeningSingleQuote(s string) int {
	for offset := 0; offset < len(s); {
		idx := strings.IndexByte(s[offset:], '\'')
		if idx < 0 {
			return -1
		}
		idx += offset
		prev, hasPrev := liveVoicePrevRune(s, idx)
		if !hasPrev || !liveVoiceRuneIsWord(prev) {
			return idx
		}
		offset = idx + 1
	}
	return -1
}

func liveVoiceIndexClosingSingleQuote(s string) int {
	for offset := 0; offset < len(s); {
		idx := strings.IndexByte(s[offset:], '\'')
		if idx < 0 {
			return -1
		}
		idx += offset
		next, hasNext := liveVoiceNextRune(s, idx+1)
		if !hasNext || !liveVoiceRuneIsWord(next) {
			return idx
		}
		offset = idx + 1
	}
	return -1
}

func liveVoicePrevRune(s string, before int) (rune, bool) {
	var last rune
	found := false
	for _, r := range s[:before] {
		last = r
		found = true
	}
	return last, found
}

func liveVoiceNextRune(s string, at int) (rune, bool) {
	if at >= len(s) {
		return 0, false
	}
	r, size := utf8.DecodeRuneInString(s[at:])
	if r == utf8.RuneError && size == 0 {
		return 0, false
	}
	return r, true
}

func liveVoiceRuneIsWord(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
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
