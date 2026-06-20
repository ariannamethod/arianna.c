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
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
	"unicode"
)

// subprocess deadlines (F-3): a hung nanollama / kk-cli must not wedge the
// subconscious goroutine or orphan a child at shutdown.
const (
	dreamTimeout = 25 * time.Second
	kkTimeout    = 10 * time.Second
)

// nano spawns the subconscious inference one-shot per dream — the vendored
// notorch-native doe engine (the LoRA parliament) when present, else the nanollama
// Go inference. The body (gguf) is the same Arianna nano either way; doe just lets
// the parliament seat on it (#3).
type nano struct {
	bin    string
	gguf   string
	maxTok string
	temp   string
	topP   string
	// #3 parliament: when doeBin is set, the dream runs through doe (doeAlpha = the
	// LoRA-parliament strength: "0" = dormant / plain notorch-native forward, "0.1"
	// = the parliament seats). doeTrain ("1") turns on the proven online expert
	// learning (notorch_step Oja, debt-driven) so the experts learn from the dream;
	// "0" (default) = static experts (the proven yent config). Empty doeBin => the
	// nanollama path above.
	doeBin   string
	doeAlpha string
	doeTrain string
	// doeD is the persistent doe REPL (the parliament stays awake — one model load,
	// the field evolving across dreams). nil => doeDream falls back to a one-shot spawn
	// per dream (graceful degradation). Talked to under mu (one generation at a time).
	doeD *doeDaemon
	// serialize the one-shot spawn: runSubconscious (the human-turn dream) and the
	// autonomous breathing fallback both call dream() on the shared nano — without
	// this, two full doe/nanollama model-loads could run at once (RSS/CPU spike).
	mu sync.Mutex
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

// dream spawns the nano on a seed and returns its murmur (label-stripped and
// sentence-cut). "" on failure. Routes through the doe parliament engine when set
// (the SAME nano body, notorch-native), else the nanollama one-shot below.
func (n *nano) dream(parent context.Context, seed string) string {
	n.mu.Lock() // one model-load at a time across the turn dream + the breathing fallback
	defer n.mu.Unlock()
	select {
	case <-parent.Done(): // cancelled while waiting for the lock (e.g. /quit) — don't spawn
		return ""
	default:
	}
	if n.doeBin != "" {
		return n.doeDream(parent, seed)
	}
	ctx, cancel := context.WithTimeout(parent, dreamTimeout) // derive from parent so a cancel kills the spawn
	defer cancel()
	cmd := exec.CommandContext(ctx, n.bin,
		"--model", n.gguf,
		"--prompt", seed,
		"--max-tokens", n.maxTok,
		"--temp", n.temp,
		"--top-p", n.topP,
	)
	out, err := cmd.Output() // stdout only; a hung nano is killed at the deadline
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
	ctx, cancel := context.WithTimeout(context.Background(), kkTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, cli, "query", db, q, "public", "1", "compressed").Output()
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
// dream naturally lags the duet. On close(seedCh) it finishes any in-flight dream
// (bounded by dreamTimeout), closes dreamCh, and closes done so stop() can join.
func runSubconscious(n *nano, cli, db string, seedCh <-chan string, dreamCh chan dreamResult, done chan<- struct{}) {
	defer close(done)
	for cue := range seedCh {
		frag := kkRetrieve(cli, db, cue)
		seed := cue
		if frag != "" {
			seed = frag // dream ON the resonant fragment (subscription), not the chatter
		}
		d := n.dream(context.Background(), seed) // the human-turn dream runs to its own deadline; stop() joins it
		if d == "" {
			continue
		}
		// publish the LATEST dream (F-4): drain any unread previous one, then send —
		// the subconscious surfaces with its newest state, not a stale backlog.
		r := dreamResult{frag: frag, dream: d}
		select {
		case dreamCh <- r:
		default:
			select {
			case <-dreamCh:
			default:
			}
			select {
			case dreamCh <- r:
			default:
			}
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

// recvDream returns the latest dream if one is ready (non-blocking). A closed
// channel reports ok=false (F-9) — it is not a fresh empty dream.
func recvDream(ch chan dreamResult) (dreamResult, bool) {
	select {
	case r, ok := <-ch:
		return r, ok
	default:
		return dreamResult{}, false
	}
}
