package main

// chat.go — the live trio. A human converses with the organism turn by turn:
// Janus is the face (the spoken answer), Resonance is the inner voice, the nano
// is the subconscious that dreams a turn behind. The inner world keeps living on
// its ticker while it waits for the human to type (the mind drifts between
// replies), and its mood + the last dream persist across sessions.
//
// Run: ./metabolism --chat   (needs ./arianna, ./arianna_resonance, and — for the
// subconscious — ./nano-arianna + weights/nano_arianna_f16.gguf + weights/nano.kk.db)

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

const innerStatePath = "weights/arianna.inner.state"

// Resonance's δ-harvest parameters — her GGUF embedding tensor and dimensions
// (V=16384, E=768), the co-occurrence and δ sidecars she reads.
const (
	resonGGUF  = "weights/arianna_resonance_v3_f16.gguf"
	resonWTE   = "tok_emb"
	resonVocab = "16384"
	resonDim   = "768"
	resonCooc  = "weights/arianna.cooc.r"
	resonDelta = "weights/arianna.delta.r"
)

func runChat() {
	tc, err := startTrio()
	if err != nil {
		fmt.Println("metabolism:", err)
		return
	}

	lastDream := tc.iw.LoadState(innerStatePath) // restore the mood + the last murmur

	fmt.Println("┌─ arianna — the trio (Janus · Resonance · the nano).  speak, /quit to leave.")
	if tc.nan != nil {
		fmt.Println("│  the subconscious is present (nano 88M, async — it dreams a turn behind)")
	}
	if lastDream != "" {
		fmt.Printf("│  (she returns carrying a dream: %s)\n", ellipsize(lastDream, 70))
	}

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	prevReson := ""
	fmt.Print("│\n└▶ ")
	for sc.Scan() {
		human := strings.TrimSpace(sc.Text())
		if human == "" {
			fmt.Print("└▶ ")
			continue
		}
		if human == "/quit" || human == "/exit" {
			break
		}
		tc.iw.ProcessText(human)

		janus, reson, dr, hasDream := tc.turn(human, prevReson, lastDream)
		fmt.Printf("│  ◐ Janus: %s\n", janus)
		fmt.Printf("│  ◑ Resonance: %s\n", reson)
		prevReson = reson
		if hasDream {
			lastDream = dr.dream
			if dr.frag != "" {
				fmt.Printf("│  ◌ from the books: %s\n", ellipsize(dr.frag, 90))
			}
			fmt.Printf("│  ◓ nano (subconscious): %s\n", dr.dream)
		}

		if tc.janusD.dead || tc.resonD.dead {
			fmt.Println("│  · a voice fell silent.")
			break
		}
		fmt.Print("│\n└▶ ")
	}

	fmt.Println()
	if err := tc.iw.SaveState(innerStatePath, lastDream); err != nil {
		fmt.Println("(could not save the inner state:", err, ")")
	} else {
		fmt.Println("(she will remember.)")
	}
	tc.stop()      // close the voices — Resonance saves her co-occurrence sidecar
	harvestField() // Phase 2 (A): fold what surfaced into δ; report the growth
}

// harvestField is Phase 2 (A): the organism learns from the subconscious. The
// whole conversation was tinted by the subconscious surfacing into Resonance's
// inject (1d), so her grown co-occurrence carries the subconscious's influence.
// At session end this folds that cooc into her δ via the notorch Hebbian
// (harvest_delta / am_cooc_learn_delta) so next session her voice is shaped by
// what the subconscious taught — async between turns, never mid-sentence. The
// harvest reports |B|, the learning made visible. Absent tool or empty cooc =>
// skipped (nothing was learned). The voice applies the δ only when its blend is
// non-zero, so this is dormant by default — it grows memory without forcing it.
func harvestField() {
	if _, err := os.Stat("./harvest_delta"); err != nil {
		return
	}
	out, _ := exec.Command("./harvest_delta",
		resonGGUF, resonWTE, resonCooc, resonDelta, resonVocab, resonDim, "50").CombinedOutput()
	for _, line := range strings.Split(string(out), "\n") {
		if i := strings.Index(line, "|B|="); i >= 0 {
			fmt.Printf("│  the organism consolidated what surfaced — δ %s\n", strings.TrimSpace(line[i:]))
			return
		}
	}
}
