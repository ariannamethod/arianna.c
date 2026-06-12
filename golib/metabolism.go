package main

// arianna-metabolism — the Go orchestrator. Hosts the inner-world's async
// goroutines continuously and runs the Janus<->Resonance duet with the inner
// world IN THE LOOP and the conversation's RHYTHM gated by the inner world.
//
//   conversation -> inner world : each voice's words are fed through ProcessText
//     (trauma / overthinking / attention / prophecy react to what was said).
//   inner world -> conversation : arousal tilts each voice's sampling temperature.
//   inner world -> rhythm (4b)  : the exchange budget (how long they talk) and the
//     inter-turn delay (how fast) are gated by the inner-world state — aroused +
//     coherent => generative (more, snappier); traumatised => terse; overthinking
//     => settle (longer pauses). The legacy chamber-gated scheduler, driven by our
//     in-loop inner world instead of the AML field's chambers.
//
// Still spawn-per-turn; hot --daemon voices are 4b.2, the shared mmap nerve is 4d.
//
// This is also the package's main(); -buildmode=c-shared ignores the body, so
// libarianna still builds. Build: go build -o ../metabolism ./golib

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func main() {
	prompt := "What is resonance?"
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}

	// Host the inner-world: start the async processes + step them on a ticker.
	iw := Global()
	iw.Start()
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

	// Proof that the rhythm is state-driven: the same scheduler maps a calm state
	// and an aroused state to different exchange budgets.
	calm := Snapshot{Arousal: 0.30, Coherence: 0.80}
	hot := Snapshot{Arousal: 0.60, Coherence: 0.80}
	fmt.Printf("┌─ arianna-metabolism  (inner-world in the loop + gating the rhythm)\n")
	fmt.Printf("│  rhythm map: budget(calm arousal=0.30)=%d  budget(aroused arousal=0.60)=%d\n",
		tickBudget(calm), tickBudget(hot))

	iw.ProcessText(prompt)       // the topic enters the inner world
	time.Sleep(400 * time.Millisecond) // let the goroutines digest the seed
	nExch := tickBudget(iw.GetSnapshot())
	fmt.Printf("│  seed: %s\n│  exchange budget (from state): %d\n", prompt, nExch)

	for i := 1; i <= nExch; i++ {
		// inner-world -> Janus: arousal tilts the sampling temperature.
		s := iw.GetSnapshot()
		jTemp := clampf(0.8+(s.Arousal-0.3)*0.5, 0.6, 1.1)
		janus := voiceClean(runVoice("./arianna",
			[]string{"-p", prompt, "-t", ftoa(jTemp), "--top-p", "0.9", "-n", "28"}))
		iw.ProcessText(janus)
		fmt.Printf("│\n│  ◐ [%d/%d] Janus(t=%.2f): %s\n", i, nExch, jTemp, janus)

		s = iw.GetSnapshot()
		rTemp := clampf(0.7+(s.Arousal-0.3)*0.4, 0.5, 1.0)
		inject := janus + " " + prompt
		reson := voiceClean(runVoice("./arianna_resonance",
			[]string{"-p", "Arianna:", "--inject", inject, "--alpha", "5",
				"-t", ftoa(rTemp), "--top-p", "1.0", "-n", "28"}))
		iw.ProcessText(reson)
		fmt.Printf("│  ◑ [%d/%d] Resonance(t=%.2f): %s\n", i, nExch, rTemp, reson)

		s = iw.GetSnapshot()
		d := tickDelay(s)
		fmt.Printf("│  · inner-world: arousal=%.3f coher=%.3f trauma=%.3f wander=%.3f loops=%d  | settle %v\n",
			s.Arousal, s.Coherence, s.TraumaLevel, s.WanderPull, s.LoopCount, d)
		if i < nExch {
			time.Sleep(d) // the inter-turn pace, gated by the inner-world state
		}
	}
	close(done)
	fmt.Println("└─ done — inner world in the loop, rhythm gated by it")
}

// tickBudget maps the inner-world state to how many exchanges the duet runs —
// aroused + coherent => generative, traumatised => terse. (Legacy chamber-gated
// scheduler, driven by our in-loop inner world.)
func tickBudget(s Snapshot) int {
	b := 4.0
	b += float64(s.Arousal-0.3) * 12.0    // aroused -> talk more
	b += float64(s.WanderPull-0.3) * 3.0  // wandering -> explore more
	b -= float64(s.TraumaLevel) * 6.0     // trauma -> don't chatter in pain
	if s.Coherence < 0.5 {
		b -= 2.0 // incoherent -> shorter
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

// tickDelay maps the inner-world state to the inter-turn pause — settle (longer)
// when overthinking or highly aroused, snappy when calm.
func tickDelay(s Snapshot) time.Duration {
	d := 150 * time.Millisecond
	if s.LoopCount > 2 {
		d += 350 * time.Millisecond // overthinking -> settle
	}
	if s.Arousal > 0.5 {
		d += 200 * time.Millisecond
	}
	return d
}

// runVoice spawns a voice binary and returns its stdout; stderr is discarded.
func runVoice(bin string, args []string) string {
	cmd := exec.Command(bin, args...)
	out, _ := cmd.Output()
	return string(out)
}

// voiceClean drops banner lines, collapses whitespace, and cuts at the first
// sentence end after a minimum length (the bash clean_voice essentials).
func voiceClean(raw string) string {
	var b strings.Builder
	for _, l := range strings.Split(raw, "\n") {
		if strings.HasPrefix(l, "[") {
			continue
		}
		b.WriteString(l)
		b.WriteByte(' ')
	}
	t := strings.Join(strings.Fields(b.String()), " ")
	for i := 30; i < len(t); i++ {
		if c := t[i]; c == '.' || c == '!' || c == '?' {
			return t[:i+1]
		}
	}
	return t
}

func clampf(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func ftoa(v float32) string {
	return strconv.FormatFloat(float64(v), 'f', 3, 64)
}
