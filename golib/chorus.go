package main

// chorus.go — the subconscious as a POLYPHONY (non-binarity). Instead of one
// murmur, the nano answers as a chorus: N cells over the ONE frozen body, each
// from its own angle (temperature/seed), hearing each other's hidden state
// (cross-cell), never echoing, sometimes asking each other resonant questions
// (qloop). Borrowed from the twin arianna2arianna.c (a single-file chorus engine
// over the same nanoArianna 89M). The metabolism spawns it like the nano; the
// cells become the subconscious's dream, and Resonance murmurs to the chorus.

import (
	"context"
	"os/exec"
	"strings"
	"time"
)

const chorusTimeout = 40 * time.Second

// choir spawns the chorus engine in field mode (4 cells, cross-cell on) and
// returns the cells' fragments — the polyphonic dream. "" / nil on failure.
func choir(bin, gguf, seed string) []string {
	seed = strings.TrimSpace(seed)
	if bin == "" || gguf == "" || seed == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), chorusTimeout)
	defer cancel()
	// field <cells> <frag> <rounds> <alpha> <leap> <xcell>: 4 voices, 16 tokens
	// each, one round, soma-alpha 0 (it does not earn resonance), cross-cell 0.3
	// (the cells hear each other).
	out, err := exec.CommandContext(ctx, bin, gguf, seed, "field", "4", "16", "1", "0", "0", "0.3").Output()
	if err != nil {
		return nil
	}
	var cells []string
	for _, line := range strings.Split(string(out), "\n") {
		// cell lines: "  r1 cell 0 (T=0.60):  <text>  [entropy=…]"
		// qloop lines: "  ↳ qloop c3→c0 score 1.097:  <text>  [entropy=…]"
		isCell := strings.Contains(line, "(T=")
		isQloop := strings.Contains(line, "qloop")
		if !isCell && !isQloop {
			continue
		}
		p := line
		if b := strings.Index(p, "["); b >= 0 { // drop the trailing metrics
			p = p[:b]
		}
		c := strings.LastIndex(p, ":") // text follows the last colon (cell "):" or "score N:")
		if c < 0 {
			continue
		}
		frag := strings.TrimSpace(p[c+1:])
		frag = strings.TrimSpace(strings.TrimPrefix(frag, "A:"))
		frag = strings.TrimSpace(strings.TrimPrefix(frag, "-"))
		frag = strings.Join(strings.Fields(frag), " ")
		if len(frag) > 3 {
			cells = append(cells, frag)
		}
	}
	return cells
}

// chorusText folds the cells into one line for the inject / lastDream — the
// chorus heard as a single resonant murmur by the inner voice.
func chorusText(cells []string) string {
	return strings.Join(cells, " / ")
}
