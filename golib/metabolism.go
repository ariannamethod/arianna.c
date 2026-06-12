package main

// arianna-metabolism — the Go orchestrator. Hosts the inner-world's async
// goroutines continuously and runs the Janus<->Resonance duet with the inner
// world IN THE LOOP (Stage 4c):
//
//   conversation -> inner world : each voice's words are fed through
//     ProcessText (trauma / overthinking / attention / prophecy react to what was
//     actually said).
//   inner world -> conversation : the inner world's arousal tilts each voice's
//     sampling temperature before it speaks (energetic when aroused, calm when
//     settled).
//
// That closes the resonant loop — the dialogue shapes the inner life, the inner
// life colours the dialogue. Stage 4a hosted the inner world alongside; 4c puts
// it in the circuit. Still spawn-per-turn; hot --daemon voices + the chamber-
// gated rhythm are 4b, the shared mmap nerve is 4d.
//
// This is also the package's main(); -buildmode=c-shared ignores the body, so
// libarianna still builds. Build the orchestrator binary with:
//   go build -o ../metabolism ./golib

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

	fmt.Printf("┌─ arianna-metabolism  N=%d exchanges  (inner-world in the loop)\n│  seed: %s\n", nExch, prompt)
	iw.ProcessText(prompt) // the topic enters the inner world

	for i := 1; i <= nExch; i++ {
		// inner-world -> Janus: arousal tilts the sampling temperature.
		s := iw.GetSnapshot()
		jTemp := clampf(0.8+(s.Arousal-0.3)*0.5, 0.6, 1.1)
		janus := voiceClean(runVoice("./arianna",
			[]string{"-p", prompt, "-t", ftoa(jTemp), "--top-p", "0.9", "-n", "28"}))
		iw.ProcessText(janus) // Janus's words -> the inner world reacts
		fmt.Printf("│\n│  ◐ [%d/%d] Janus(t=%.2f): %s\n", i, nExch, jTemp, janus)

		// inner-world -> Resonance: same arousal tilt on her temperature.
		s = iw.GetSnapshot()
		rTemp := clampf(0.7+(s.Arousal-0.3)*0.4, 0.5, 1.0)
		inject := janus + " " + prompt
		reson := voiceClean(runVoice("./arianna_resonance",
			[]string{"-p", "Arianna:", "--inject", inject, "--alpha", "5",
				"-t", ftoa(rTemp), "--top-p", "1.0", "-n", "28"}))
		iw.ProcessText(reson) // Resonance's words -> the inner world reacts
		fmt.Printf("│  ◑ [%d/%d] Resonance(t=%.2f): %s\n", i, nExch, rTemp, reson)

		s = iw.GetSnapshot()
		fmt.Printf("│  · inner-world: arousal=%.3f coher=%.3f trauma=%.3f debt=%.3f wander_pull=%.3f loops=%d\n",
			s.Arousal, s.Coherence, s.TraumaLevel, s.ProphecyDebt, s.WanderPull, s.LoopCount)
	}
	close(done)
	fmt.Println("└─ done — the inner world was in the loop")
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
