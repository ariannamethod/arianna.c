package main

// arianna-metabolism — the Go orchestrator (Stage 4a). Hosts the inner-world's
// async goroutines continuously and runs the Janus<->Resonance duet, stepping the
// inner world between turns so it LIVES alongside the conversation rather than
// only existing inside a test.
//
// Stage 4a uses spawn-per-turn voices (like the bash orchestrator). Hot --daemon
// voices + a per-turn inject protocol come in 4b, the chamber-gated rhythm in 4b,
// and surfacing the inner-world's signals into the nerve in 4c.
//
// This is also the package's main(); -buildmode=c-shared ignores the body, so
// libarianna still builds. Build the orchestrator binary with:
//   go build -o ../metabolism ./golib

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

func main() {
	prompt := "What is resonance?"
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}
	nExch := 4

	// Host the inner-world: start the async processes + step them on a ticker so
	// the goroutines keep breathing between (and during) the voices' turns.
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

	fmt.Printf("┌─ arianna-metabolism  N=%d exchanges  (inner-world hosted)\n│  seed: %s\n", nExch, prompt)
	for i := 1; i <= nExch; i++ {
		janus := voiceClean(runVoice("./arianna",
			[]string{"-p", prompt, "-t", "0.8", "--top-p", "0.9", "-n", "28"}))
		fmt.Printf("│\n│  ◐ [%d/%d] Janus: %s\n", i, nExch, janus)

		inject := janus + " " + prompt
		reson := voiceClean(runVoice("./arianna_resonance",
			[]string{"-p", "Arianna:", "--inject", inject, "--alpha", "5",
				"-t", "0.7", "--top-p", "1.0", "-n", "28"}))
		fmt.Printf("│  ◑ [%d/%d] Resonance: %s\n", i, nExch, reson)

		s := iw.GetSnapshot()
		fmt.Printf("│  · inner-world: arousal=%.3f coher=%.3f trauma=%.3f debt=%.3f wander_pull=%.3f loops=%d\n",
			s.Arousal, s.Coherence, s.TraumaLevel, s.ProphecyDebt, s.WanderPull, s.LoopCount)
	}
	close(done)
	fmt.Println("└─ done — inner-world ran alongside the duet")
}

// runVoice spawns a voice binary and returns its stdout (the spoken text);
// stderr (banners, larynx prints) is discarded.
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
