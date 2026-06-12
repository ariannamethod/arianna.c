package main

// arianna-metabolism — the Go orchestrator. Hosts the inner-world's async
// goroutines continuously and runs the Janus<->Resonance duet over HOT --daemon
// voices, with the inner world in the loop and gating the rhythm:
//
//   hot daemons (4b.2)        : each voice runs once as a persistent --daemon
//     process; the metabolism talks to it over stdin/stdout framed by <END>, so
//     there is no 5-6s spawn per turn — the organism stays responsive.
//   conversation -> inner world : each voice's words feed ProcessText.
//   inner world -> conversation : Resonance's per-turn inject (Janus's words, sent
//     after a tab) carries the texture; the larynx-α lives in the forward.
//   inner world -> rhythm (4b.1) : tickBudget / tickDelay gate how long and how
//     fast they talk. (Temperature is fixed at the daemon's launch value; the
//     inner-world coupling rides the rhythm, the stronger channel.)
//
// This is also the package's main(); -buildmode=c-shared ignores the body, so
// libarianna still builds. Build: go build -o ../metabolism ./golib

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
)

// voice is a persistent --daemon process talked to over stdin/stdout, framed by
// a "<END>" line after each reply.
type voice struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Scanner
	dead bool // set when the daemon stops responding (EOF before the <END> frame)
}

func startVoice(bin string, args []string) (*voice, error) {
	cmd := exec.Command(bin, append(args, "--daemon")...)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = nil // discard banners + larynx prints
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(outPipe)
	sc.Buffer(make([]byte, 1<<20), 1<<20) // tolerate long lines
	return &voice{cmd: cmd, in: in, out: sc}, nil
}

// ask sends one request line and reads the reply up to the <END> frame. If the
// daemon dies (stdin closed, or EOF before <END>), it marks the voice dead so the
// caller can stop instead of looping over silent empty turns. (Mythos M3.)
func (v *voice) ask(line string) string {
	if _, err := fmt.Fprintln(v.in, line); err != nil {
		v.dead = true
		return ""
	}
	var b strings.Builder
	sawEnd := false
	for v.out.Scan() {
		t := v.out.Text()
		if strings.TrimSpace(t) == "<END>" {
			sawEnd = true
			break
		}
		if strings.HasPrefix(t, "[") {
			continue
		}
		b.WriteString(t)
		b.WriteByte(' ')
	}
	if !sawEnd {
		v.dead = true // Scan returned false before the <END> frame — the daemon is gone
	}
	return cutSentence(strings.Join(strings.Fields(b.String()), " "))
}

func (v *voice) close() {
	v.in.Close()
	v.cmd.Wait()
}

func main() {
	prompt := "What is resonance?"
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}

	// Host the inner-world: start the async processes + step them on a ticker.
	iw := Global()
	iw.Start(false) // sync: the metabolism's ticker is the only clock (no per-process self-tick)
	defer iw.Stop()
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				iw.Step(0.1)
			}
		}
	}()

	// Start both voices hot, once.
	janusD, err := startVoice("./arianna", []string{"-t", "0.8", "--top-p", "0.9", "-n", "28"})
	if err != nil {
		fmt.Println("janus daemon failed:", err)
		return
	}
	defer janusD.close()
	resonD, err := startVoice("./arianna_resonance", []string{"--alpha", "5", "-t", "0.7", "--top-p", "1.0", "-n", "28"})
	if err != nil {
		fmt.Println("resonance daemon failed:", err)
		return
	}
	defer resonD.close()

	// The subconscious — the third voice. The nano (88M) dreams in the background
	// on the duet's context and surfaces a turn or two late. Absent binary/GGUF =>
	// the duet runs without it.
	nan := newNano("./nano-arianna", "weights/nano_arianna_f16.gguf")
	var seedCh chan string
	var dreamCh chan dreamResult
	if nan != nil {
		seedCh = make(chan string, 1)
		dreamCh = make(chan dreamResult, 1)
		go runSubconscious(nan, "./kk-cli", "weights/nano.kk.db", seedCh, dreamCh)
		defer close(seedCh)
	}

	calm := Snapshot{Arousal: 0.30, Coherence: 0.80}
	hot := Snapshot{Arousal: 0.60, Coherence: 0.80}
	fmt.Printf("┌─ arianna-metabolism  (hot daemons + inner-world in the loop + gating the rhythm)\n")
	fmt.Printf("│  rhythm map: budget(calm=0.30)=%d  budget(aroused=0.60)=%d\n", tickBudget(calm), tickBudget(hot))
	if nan != nil {
		fmt.Printf("│  the subconscious is present (nano 88M, async — it dreams a turn behind)\n")
	}

	iw.ProcessText(prompt)
	time.Sleep(400 * time.Millisecond)
	nExch := tickBudget(iw.GetSnapshot())
	fmt.Printf("│  seed: %s\n│  exchange budget (from state): %d\n", prompt, nExch)

	// The direct human→nano channel: the human's raw words hit the subconscious
	// before the face has formed, so the first dream is the subconscious reacting
	// to the human directly — Janus learns of it later, surfaced through Resonance.
	if nan != nil {
		sendLatest(seedCh, prompt)
	}

	prevReson := ""
	lastDream := ""
	for i := 1; i <= nExch; i++ {
		// E1: Janus hears Resonance's last line as CONTEXT in his prompt (not an
		// inject — Janus resists injection by design), so the duet is a dialogue
		// instead of Janus answering the same seed every turn.
		janusPrompt := prompt
		if prevReson != "" {
			janusPrompt = prompt + " " + prevReson
		}
		janus := janusD.ask(janusPrompt)
		iw.ProcessText(janus)
		fmt.Printf("│\n│  ◐ [%d/%d] Janus: %s\n", i, nExch, janus)

		// per-turn inject: "<prompt>\t<Janus words>[ <the subconscious undertone>]".
		// Resonance is a receiver by design, so the last dream surfaces into the
		// inner voice as an undertone (Janus, who resists injection, gets it only
		// indirectly — weaker — through the field and Resonance's reply).
		resonInject := janus + " " + prompt
		if lastDream != "" {
			resonInject += " " + lastDream
		}
		reson := resonD.ask("Arianna:\t" + resonInject)
		iw.ProcessText(reson)
		fmt.Printf("│  ◑ [%d/%d] Resonance: %s\n", i, nExch, reson)
		prevReson = reson

		// The subconscious: hand it this turn's context as a cue (non-blocking). The
		// background dreamer retrieves the most resonant book-fragment for it (the
		// KK injection — the resonant spiral) and dreams on that. Surface any dream
		// that finished from an earlier turn — the fragment that seeded it (◌) and
		// the murmur (◓) — and feed the murmur into the inner world so it tints the
		// field.
		if nan != nil {
			// The cue is normally the turn's context; but when the attention wanders
			// inward (high WanderPull) the direct human→nano channel re-opens — the
			// mind drops the conversation and returns to the human's raw words.
			cue := prompt + " " + janus + " " + reson
			if iw.GetSnapshot().WanderPull > 0.55 {
				cue = prompt
			}
			sendLatest(seedCh, cue)
			if r, ok := recvDream(dreamCh); ok {
				iw.ProcessText(r.dream)
				lastDream = r.dream // surfaces into the next turn's inner voice (1d)
				if r.frag != "" {
					fmt.Printf("│  ◌ [%d/%d] from the books: %s\n", i, nExch, ellipsize(r.frag, 90))
				}
				fmt.Printf("│  ◓ [%d/%d] nano (subconscious): %s\n", i, nExch, r.dream)
			}
		}

		// M3: if a voice fell silent, stop instead of looping over empty turns.
		if janusD.dead || resonD.dead {
			fmt.Println("│  · a voice fell silent — ending the duet")
			break
		}

		s := iw.GetSnapshot()
		d := tickDelay(s)
		fmt.Printf("│  · inner-world: arousal=%.3f coher=%.3f trauma=%.3f wander=%.3f loops=%d  | settle %v\n",
			s.Arousal, s.Coherence, s.TraumaLevel, s.WanderPull, s.LoopCount, d)
		// E3: re-read the budget from the live state — trauma mid-duet can cut it
		// short ("traumatised => terse" is the system's own claim).
		if i >= tickBudget(s) {
			fmt.Println("│  · the field settled — ending early")
			break
		}
		if i < nExch {
			time.Sleep(d)
		}
	}
	close(done)
	fmt.Println("└─ done — hot daemons, inner world in the loop, rhythm gated by it")
}

// tickBudget maps the inner-world state to how many exchanges the duet runs —
// aroused + coherent => generative, traumatised => terse.
func tickBudget(s Snapshot) int {
	b := 4.0
	b += float64(s.Arousal-0.3) * 12.0
	b += float64(s.WanderPull-0.3) * 3.0
	b -= float64(s.TraumaLevel) * 6.0
	if s.Coherence < 0.5 {
		b -= 2.0
	}
	n := int(b + 0.5)
	if n < 2 {
		n = 2
	}
	if n > 8 {
		n = 8
	}
	return n
}

// tickDelay maps the inner-world state to the inter-turn pause.
func tickDelay(s Snapshot) time.Duration {
	d := 150 * time.Millisecond
	if s.LoopCount > 2 {
		d += 350 * time.Millisecond
	}
	if s.Arousal > 0.5 {
		d += 200 * time.Millisecond
	}
	return d
}

// ellipsize trims a string to n runes for display, appending an ellipsis.
func ellipsize(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// cutSentence cuts at the first sentence end after a minimum length (the bash
// clean_voice essential, the banner-strip already done in ask()).
func cutSentence(t string) string {
	for i := 30; i < len(t); i++ {
		if c := t[i]; c == '.' || c == '!' || c == '?' {
			return t[:i+1]
		}
	}
	return t
}
