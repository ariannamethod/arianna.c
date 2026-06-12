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
	"strings"
)

const innerStatePath = "weights/arianna.inner.state"

func runChat() {
	tc, err := startTrio()
	if err != nil {
		fmt.Println("metabolism:", err)
		return
	}
	defer tc.stop()

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
}
