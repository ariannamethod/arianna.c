package main

// doe.go — the subconscious's notorch-native engine (the LoRA parliament). The
// nano (Arianna's 88M body) runs through the vendored doe.c (doe_field) so the
// living parliament can seat on it (#3). doe is a REPL: the seed is piped on stdin,
// the dream is read from stdout. doe.c is NOT a replacement for the nano — it is
// the engine + parliament over the same body. doeAlpha gates the parliament:
// "0" = dormant (plain notorch-native forward, the step-1 bridge), "0.1" = it seats.

import (
	"context"
	"os/exec"
	"strings"
	"time"
)

// doe loads the model fresh + profiles + may generate up to its 200-token REPL cap,
// so it is slower than the nanollama one-shot — give it headroom.
const doeDreamTimeout = 45 * time.Second

// maxDoeSeedBytes keeps the piped seed under doe's REPL line buffer (input[1024])
// so a long KK fragment can't split into a second prompt.
const maxDoeSeedBytes = 1000

// doeDream runs the seed through doe_field (the nano body + the parliament at
// doeAlpha) and returns the dream murmur. "" on failure / empty seed. The seed is
// fed on stdin (doe's REPL has no --prompt); stderr (timing) is discarded.
func (n *nano) doeDream(seed string) string {
	// collapse to ONE line — doe's REPL is line-oriented, so embedded newlines in a
	// KK fragment would become multiple prompts (extra generations, only the first
	// line dreamt on). One line = one dream.
	seed = strings.Join(strings.Fields(seed), " ")
	if seed == "" {
		return ""
	}
	// cap below doe's fgets line buffer (input[1024]) — a longer KK fragment would
	// split into a second REPL prompt. Trim at a word boundary so no UTF-8 rune is
	// cut (spaces are ASCII, safe to slice at).
	if len(seed) > maxDoeSeedBytes {
		cut := maxDoeSeedBytes
		if sp := strings.LastIndexByte(seed[:cut], ' '); sp > 0 {
			cut = sp
		}
		// ToValidUTF8 drops a multibyte rune split by the hard byte cut (when no
		// space was found before the cap — rare for word cues, but kept correct).
		seed = strings.ToValidUTF8(seed[:cut], "")
	}
	ctx, cancel := context.WithTimeout(context.Background(), doeDreamTimeout)
	defer cancel()
	train := n.doeTrain
	if train == "" {
		train = "0" // proven default: static experts (no online weight drift)
	}
	cmd := exec.CommandContext(ctx, n.doeBin, "--model", n.gguf, "--lora-alpha", n.doeAlpha, "--train", train)
	cmd.Stdin = strings.NewReader(seed + "\n")
	out, err := cmd.Output() // stdout only; a hung doe is killed at the deadline
	if err != nil {
		return ""
	}
	return parseDoeDream(string(out))
}

// parseDoeDream extracts the dream from doe's REPL stdout. doe prints a banner +
// index logs (`[identity]`/`[host]`/`[sonar]`/`[mycelium]`/`[doe] …` and per-layer
// `  L#: health=…` lines), then the prompt `> ` immediately followed by the
// streamed dream on the same line, then a `  [life] …` footer. The dream is the
// first `>`-line that carries real text (not a `> [mycelium]`/`> [doe]` log line).
func parseDoeDream(out string) string {
	var b strings.Builder
	capturing := false
	for _, line := range strings.Split(out, "\n") {
		t := strings.TrimSpace(line)
		if !capturing {
			if !strings.HasPrefix(t, ">") { // not the generation prompt yet
				continue
			}
			body := strings.TrimSpace(strings.TrimPrefix(t, ">"))
			if body == "" || strings.HasPrefix(body, "[") { // empty prompt / "> [mycelium]"/"> [doe]" log
				continue
			}
			capturing = true // the dream starts here (may be just a label like "Arianna:" — continue below)
			b.WriteString(body)
			b.WriteByte(' ')
			continue
		}
		// capturing the dream: stop at a footer / log / the next prompt
		if t == "" || strings.HasPrefix(t, "[") || strings.HasPrefix(t, ">") {
			break
		}
		b.WriteString(t)
		b.WriteByte(' ')
	}
	dream := strings.TrimSpace(b.String())
	if i := strings.Index(dream, "[life]"); i >= 0 { // a footer that leaked onto the line
		dream = dream[:i]
	}
	dream = stripLabel(strings.TrimSpace(dream)) // drop a leading SFT label (A:/Q:/Arianna:)
	return cutSentence(strings.Join(strings.Fields(dream), " "))
}
