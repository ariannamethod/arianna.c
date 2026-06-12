package main

// nano.go — the subconscious (the third voice) in the metabolism.
//
// The nano (88M, the deepest layer, the origin-seed) runs as an async background
// dreamer: the metabolism hands it the latest context as a seed, the nano dreams
// on it via a one-shot spawn of the nanollama Go inference, and the dream
// surfaces a turn or two late — the lag IS the design, the subconscious trailing
// the conscious duet. One-shot spawn (not a hot daemon) keeps the nanollama
// scaffold untouched; the ~1.6s load is hidden by the subconscious being async
// and occasional. Absent binary or GGUF => the metabolism runs the duet without
// the subconscious (graceful degradation to two voices).

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"unicode"
)

// nano spawns the nanollama Go inference one-shot per dream.
type nano struct {
	bin    string
	gguf   string
	maxTok string
	temp   string
	topP   string
}

// newNano returns a nano if the binary and the GGUF are both present, else nil.
func newNano(bin, gguf string) *nano {
	if _, err := os.Stat(bin); err != nil {
		return nil
	}
	if _, err := os.Stat(gguf); err != nil {
		return nil
	}
	// Low max-tokens + warm temp: a murmur, not a monologue. top-p slightly open
	// so the associative drift (the dream-logic) is not clamped to the mode.
	return &nano{bin: bin, gguf: gguf, maxTok: "32", temp: "0.9", topP: "0.92"}
}

// dream spawns the nano on a seed and returns its murmur (the text after the
// "[<n> tokens ...]" frame line, label-stripped and sentence-cut). "" on failure.
func (n *nano) dream(seed string) string {
	cmd := exec.Command(n.bin,
		"--model", n.gguf,
		"--prompt", seed,
		"--max-tokens", n.maxTok,
		"--temp", n.temp,
		"--top-p", n.topP,
	)
	out, err := cmd.Output() // stdout only
	if err != nil {
		return ""
	}
	return cleanDream(string(out))
}

// cleanDream extracts the murmur from the nano's one-shot stdout. The nano prints
// its banner ("[nanollama] ...", "[gguf] ...") and the streamed text, then a
// "[<n> tokens, <tps> tok/s]" frame, then the result a second time un-streamed.
// We take everything after that frame line — the clean copy.
func cleanDream(out string) string {
	lines := strings.Split(out, "\n")
	marker := -1
	for i, l := range lines {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "[") && strings.Contains(t, " tokens") && strings.Contains(t, "tok/s") {
			marker = i
			break
		}
	}
	if marker < 0 || marker+1 >= len(lines) {
		return ""
	}
	murmur := stripLabel(strings.Join(lines[marker+1:], " "))
	return cutSentence(strings.Join(strings.Fields(murmur), " "))
}

// stripLabel drops a leading SFT chat-label ("A:", "Arianna:", "Q:") — the
// subconscious murmurs, it does not answer as an assistant.
func stripLabel(s string) string {
	s = strings.TrimSpace(s)
	for _, p := range []string{"A:", "Arianna:", "Q:"} {
		if strings.HasPrefix(s, p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	return s
}

// dreamResult is what a single round of the subconscious produces: the KK
// fragment that seeded it (the resonant book-passage, "" if none) and the dream.
type dreamResult struct {
	frag  string
	dream string
}

// kkPacket is the minimal shape of the Knowledge Kernel's compressed-mode JSON.
type kkPacket struct {
	Results []struct {
		Text  string  `json:"text"`
		Score float64 `json:"score"`
	} `json:"results"`
}

// kkRetrieve asks the Knowledge Kernel for the book-fragment most resonant with
// the cue (the resonant spiral) and returns its text — the dream-seed. "" if the
// kk-cli or the DB is absent or there is no hit; the nano then dreams on the cue
// itself (the Phase-1b behaviour).
func kkRetrieve(cli, db, cue string) string {
	if _, err := os.Stat(cli); err != nil {
		return ""
	}
	if _, err := os.Stat(db); err != nil {
		return ""
	}
	q := sanitizeCue(cue)
	if q == "" {
		return ""
	}
	out, err := exec.Command(cli, "query", db, q, "public", "1", "compressed").Output()
	if err != nil {
		return ""
	}
	var p kkPacket
	if json.Unmarshal(out, &p) != nil || len(p.Results) == 0 {
		return ""
	}
	return strings.TrimSpace(p.Results[0].Text)
}

// sanitizeCue reduces the conversation cue to a clean bag of words: a plain FTS
// query that won't trip on punctuation (the "?" / "," of live speech), capped so
// the query stays a focused signal rather than a paragraph.
func sanitizeCue(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == ' ' {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	words := strings.Fields(b.String())
	if len(words) > 24 {
		words = words[:24]
	}
	return strings.Join(words, " ")
}

// runSubconscious hosts the nano as a background dreamer: it pulls the latest
// cue, retrieves the resonant book-fragment for it (the KK injection), dreams on
// the fragment, and publishes the result. seedCh and dreamCh are each single-slot
// (buffered 1) with one producer and one consumer, so neither side blocks and the
// dream naturally lags the duet. Exits + closes dreamCh when seedCh closes.
func runSubconscious(n *nano, cli, db string, seedCh <-chan string, dreamCh chan<- dreamResult) {
	for cue := range seedCh {
		frag := kkRetrieve(cli, db, cue)
		seed := cue
		if frag != "" {
			seed = frag // dream ON the resonant fragment (subscription), not the chatter
		}
		d := n.dream(seed)
		if d == "" {
			continue
		}
		// publish (non-blocking). The consumer drains every turn, so the single slot
		// is normally free; an unread dream is dropped rather than queued — the
		// subconscious does not back up.
		select {
		case dreamCh <- dreamResult{frag: frag, dream: d}:
		default:
		}
	}
	close(dreamCh)
}

// sendLatest pushes a seed into a single-slot channel, replacing any unread one
// (keep only the latest — stale context is dropped, not queued).
func sendLatest(ch chan string, v string) {
	select {
	case ch <- v:
	default:
		select {
		case <-ch:
		default:
		}
		select {
		case ch <- v:
		default:
		}
	}
}

// recvDream returns the latest dream if one is ready (non-blocking).
func recvDream(ch chan dreamResult) (dreamResult, bool) {
	select {
	case r := <-ch:
		return r, true
	default:
		return dreamResult{}, false
	}
}
