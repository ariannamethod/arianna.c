package main

// chorus.go — the subconscious as a POLYPHONY (non-binarity). Instead of one
// murmur, the nano answers as a chorus: N cells over the ONE frozen body, each
// from its own angle (temperature/seed), hearing each other's hidden state
// (cross-cell). Vendored from the twin into chorus/arianna2arianna.c (a single-file
// chorus engine over the same nanoArianna 89M, built into ./chorus-arianna — a
// byte-exact in-repo copy, no external dependency). The metabolism spawns it like
// the nano; the cells become the subconscious's dream, and Resonance murmurs to it.

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	chorusTimeout  = 40 * time.Second
	maxChorusCells = 8
)

type chorusCell struct {
	text string
}

// choir spawns the chorus engine in field mode (nCells voices, cross-cell on) and
// returns the parsed cells — the polyphonic dream. nil on failure. The ctx lets
// the caller cancel a slow chorus at shutdown so it never outlives the join. nCells
// is the live field's bloom↔collapse knob (the engine's own n_cells axis); it is
// clamped to the engine's [1,8] range here.
func choir(ctx context.Context, bin, gguf, seed string, nCells int) []chorusCell {
	seed = strings.TrimSpace(seed)
	if bin == "" || gguf == "" || seed == "" {
		return nil
	}
	if nCells < 1 {
		nCells = 1
	}
	if nCells > 8 {
		nCells = 8 // engine hard cap (POP_MAX, arianna2arianna.c:1135)
	}
	cctx, cancel := context.WithTimeout(ctx, chorusTimeout)
	defer cancel()
	// field <cells> <frag> <rounds> <alpha> <leap> <xcell>: nCells voices, 16 tokens
	// each, one round, soma-alpha 0 (it does not earn resonance), cross-cell 0.3
	// (the cells hear each other).
	out, err := exec.CommandContext(cctx, bin, gguf, nanoPrompt(seed), "field", strconv.Itoa(nCells), "16", "1", "0", "0", "0.3").Output()
	if err != nil {
		return nil
	}
	return parseChorusCells(string(out))
}

func parseChorusCells(out string) []chorusCell {
	var cells []chorusCell
	for _, line := range strings.Split(out, "\n") {
		// cell lines: "  r1 cell 0 (T=0.60):  <text>  [entropy=…]"
		isCell := strings.Contains(line, "(T=")
		if !isCell {
			continue
		}
		frag := chorusBody(line)
		if frag != "" {
			cells = append(cells, chorusCell{text: frag})
			if len(cells) >= maxChorusCells { // cap the parsed run
				break
			}
		}
	}
	return cells
}

// chorusBody extracts the spoken text from one chorus line, cutting the leading
// "cell N (T=…):" / "score N:" frame and the trailing "[entropy=…]" metrics. It
// keys on the STRUCTURAL colon (the frame's), not the last colon, so generated
// text that itself contains a colon is not truncated.
func chorusBody(line string) string {
	p := line
	head := 0
	if c := strings.Index(p, "):"); c >= 0 {
		// "… (T=0.60):  <text>" — the colon that closes the temperature.
		head = c + 2
	}
	p = p[head:]
	cut := len(p)
	for _, marker := range []string{"   [Δ_R^kv ", "   [entropy="} {
		if i := strings.Index(p, marker); i >= 0 && i < cut {
			cut = i
		}
	}
	p = p[:cut]
	frag := strings.TrimSpace(p)
	// sanitize per cell at the source — the chorus engine's SPM <0xXX> byte fallback
	// decodes to raw bytes, and the cell is shown raw in the breathing display AND
	// folded into the dream, so clean it here for both.
	return strings.ToValidUTF8(strings.Join(strings.Fields(frag), " "), "")
}

// chorusText folds the cells into one line for the inject / lastDream.
func chorusText(cells []chorusCell) string {
	parts := make([]string, 0, len(cells))
	for _, c := range cells {
		parts = append(parts, c.text)
	}
	// Drop any byte-fallback / invalid UTF-8 the chorus engine emitted (chorus is a
	// separate binary; its SPM <0xXX> byte fallback decodes to raw bytes the C voices'
	// output guard does not cover), ALWAYS — not only on the capped path — so a short
	// chorus dream that lands in lastDream / the Resonance inject is also valid UTF-8.
	return strings.ToValidUTF8(strings.Join(parts, " / "), "")
}
