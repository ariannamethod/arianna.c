package main

// arianna-metabolism — the Go orchestrator. Hosts the inner-world's async
// goroutines continuously and runs the Janus<->Resonance duet over HOT --daemon
// voices, with the inner world in the loop and gating the rhythm:
//
//   hot daemons (4b.2)        : each voice runs once as a persistent --daemon
//     process; the metabolism talks to it over stdin/stdout with length-prefixed
//     reply frames, so
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
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// voice is a persistent --daemon process talked to over stdin/stdout, framed by
// an AMF1 byte length before each reply.
type voice struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Reader
	dead bool   // set when the daemon stops responding or truncates a frame
	bin  string // remembered so a fallen voice can be respawned in place
	args []string
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
	reader := bufio.NewReaderSize(outPipe, 1<<20)
	if _, err := fmt.Fprintln(in); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, err
	}
	ready := make(chan bool, 1)
	go func() {
		_, ok := readLiveVoiceFrame(reader)
		ready <- ok
	}()
	select {
	case ok := <-ready:
		if !ok {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			return nil, fmt.Errorf("daemon exited before ready")
		}
	case <-time.After(voiceTimeout):
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("daemon readiness timeout")
	}
	return &voice{cmd: cmd, in: in, out: reader, bin: bin, args: args}, nil
}

// respawn revives a fallen voice in place: kill and reap the old daemon, then start a fresh one
// with the same bin+args. A hot voice daemon can stop completing frames after a turn or two; reviving
// it lets the trio survive one voice's death instead of ending the whole session. Caller holds
// voiceMu (the voices are single-stream — a revive must not race a concurrent ask or the breath).
func (v *voice) respawn() error {
	if v.cmd != nil && v.cmd.Process != nil {
		_ = v.cmd.Process.Kill()
		go func(c *exec.Cmd) { _ = c.Wait() }(v.cmd) // reap without blocking the turn
	}
	nv, err := startVoice(v.bin, v.args)
	if err != nil {
		return err
	}
	v.cmd, v.in, v.out, v.dead = nv.cmd, nv.in, nv.out, false
	return nil
}

// voiceTimeout bounds one ask: our voices emit a bounded token budget. On an idle
// machine that completes a frame in seconds, but a 176M CPU voice under heavy contention
// (other jobs saturating the cores) can legitimately take far longer — 30s treated a
// merely-slow voice as wedged and killed it, silencing the trio. The ceiling is
// generous so a slow-but-alive voice finishes its turn; a genuine wedge is still
// caught (and a real death is handled by respawn). Tunable via AM_VOICE_TIMEOUT
// (seconds).
var voiceTimeout = func() time.Duration {
	if s := os.Getenv("AM_VOICE_TIMEOUT"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 && n <= 3600 {
			return time.Duration(n) * time.Second // cap at 1h — no time.Duration overflow
		}
	}
	return 120 * time.Second
}()

// ask sends one request line and reads one length-prefixed reply. If the daemon
// dies (stdin closed, or EOF before the full frame), it marks the voice dead so the
// caller can stop instead of looping over silent empty turns (Mythos M3). The read
// runs under a deadline: a daemon that wedges mid-frame would otherwise hold
// voiceMu forever (a human turn or the autonomous breathing stuck for good), so on
// timeout the process is killed (which unblocks the read with EOF) and marked dead.
func (v *voice) ask(line string) string {
	return v.askWire(encodeLiveVoiceField(line))
}

func (v *voice) askInjected(prompt, injection string) string {
	return v.askWire(encodeLiveVoiceRequest(prompt, injection))
}

func (v *voice) askWire(line string) string {
	if _, err := fmt.Fprintln(v.in, line); err != nil {
		v.dead = true
		return ""
	}
	type reply struct {
		text     string
		complete bool
	}
	ch := make(chan reply, 1)
	go func() {
		text, complete := readLiveVoiceFrame(v.out)
		ch <- reply{text, complete}
	}()
	select {
	case r := <-ch:
		if !r.complete {
			v.dead = true
		}
		return r.text
	case <-time.After(voiceTimeout):
		v.dead = true
		if v.cmd.Process != nil {
			_ = v.cmd.Process.Kill() // closes the pipe → the read goroutine hits EOF and exits (no leak)
		}
		<-ch
		return ""
	}
}

const (
	liveVoiceFramePrefix = "AMF1 "
	liveVoiceFrameMax    = 1 << 20
)

func readLiveVoiceFrame(reader *bufio.Reader) (string, bool) {
	header, err := reader.ReadString('\n')
	if err != nil {
		return "", false
	}
	header = strings.TrimSuffix(strings.TrimSuffix(header, "\n"), "\r")
	if !strings.HasPrefix(header, liveVoiceFramePrefix) {
		return "", false
	}
	size, err := strconv.Atoi(strings.TrimPrefix(header, liveVoiceFramePrefix))
	if err != nil || size < 0 || size > liveVoiceFrameMax {
		return "", false
	}
	payload := make([]byte, size)
	if _, err := io.ReadFull(reader, payload); err != nil {
		return "", false
	}
	return sanitizeLiveVoiceText(string(payload)), true
}

func (v *voice) close() {
	v.in.Close() // EOF → the daemon saves its sidecars and exits
	// F-3: don't hang the shutdown on a wedged daemon — wait briefly, then kill.
	done := make(chan struct{})
	go func() { v.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = v.cmd.Process.Kill()
		<-done
	}
}

// trioCtx holds the three voices, the subconscious channels, and the inner world
// the metabolism drives — shared by the demo (runDemo) and the live chat (runChat).
type trioCtx struct {
	janusD, resonD *voice
	nan            *nano
	seedCh         chan string
	dreamCh        chan dreamResult
	subDone        chan struct{} // closed by runSubconscious on exit (F-3 join)
	chorusBin      string        // ./chorus-arianna, if present — the subconscious as a polyphony
	chorusGGUF     string        // the nano GGUF the chorus runs over
	iw             *InnerWorld
	tickerDone     chan struct{}
	lastMoved      float32 // being-moved: core-affect displacement of the last turn (telemetry, no feedback)
	humanTurns     int64
	pendingHumans  atomic.Int64 // received input, including turns waiting for voiceMu
	janusTurns     int64
	resonanceTurns int64
	nanoTurns      int64
	dreams         int64
	chorusDreams   int64
	innerLines     int64
	fieldTicks     int64
}

func filePresent(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func executablePresent(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Mode()&0111 != 0
}

// startTrio brings the organism up: the inner world on a 100ms ticker (its single
// clock), both voices hot as --daemon processes, and the subconscious as a
// background dreamer (absent binary/GGUF => the duet runs without it).
func startTrio() (*trioCtx, error) {
	iw := Global()
	iw.Start(false) // sync: the metabolism's ticker is the only clock (no per-process self-tick)
	// Warm the High brain (boot libjulia + JIT once) up front, so the first conversational
	// turn doesn't pay the ~1s Julia init under the inner-world lock. If Julia is unavailable,
	// the inner world falls back to its heuristics — the trio still runs.
	if err := highStart(); err != nil {
		fmt.Printf("  [high] Julia brain unavailable (%v) — inner world uses heuristic fallback\n", err)
	}
	tickerDone := make(chan struct{})
	go func() {
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-tickerDone:
				return
			case <-t.C:
				iw.Step(0.1)
			}
		}
	}()

	janusN := strconv.Itoa(liveVoiceTokenBudget("janus"))
	resonanceN := strconv.Itoa(liveVoiceTokenBudget("resonance"))
	janusD, err := startVoice("./arianna", []string{"-t", "0.8", "--top-p", "0.9", "-n", janusN})
	if err != nil {
		close(tickerDone)
		iw.Stop()
		return nil, fmt.Errorf("janus daemon: %w", err)
	}
	resonD, err := startVoice("./arianna_resonance", []string{"--alpha", "5", "-t", "0.7", "--top-p", "1.0", "-n", resonanceN})
	if err != nil {
		janusD.close()
		close(tickerDone)
		iw.Stop()
		return nil, fmt.Errorf("resonance daemon: %w", err)
	}

	tc := &trioCtx{janusD: janusD, resonD: resonD, iw: iw, tickerDone: tickerDone}
	// The subconscious body is the nano GGUF; it runs through the vendored doe engine
	// (the LoRA parliament, #3) and/or the nanollama one-shot. It is present if the
	// GGUF + at least one engine exists — so doe alone (no nanollama) still dreams.
	const nanoGGUF = "weights/nano_arianna_f16.gguf"
	doePresent := executablePresent("./doe_field")
	nanoWeightPresent := filePresent(nanoGGUF)
	if executablePresent("./nano-arianna") && nanoWeightPresent {
		tc.nan = newNano("./nano-arianna", nanoGGUF) // nanollama path (nil if its binary is absent)
	}
	if tc.nan == nil && doePresent && nanoWeightPresent {
		tc.nan = &nano{gguf: nanoGGUF, maxTok: "32", temp: "0.9", topP: "0.92"} // doe-only: dream through doe without nanollama
	}
	if tc.nan != nil {
		// #3: doe is the parliament engine over the SAME body. Step-2: the parliament
		// SEATS by default (--lora-alpha 0.1 = election + per-layer LoRA inject,
		// experts vote / mitosis / apoptosis). The AM_LORA_ALPHA env var is the debug
		// knob — set it to 0 to silence the parliament (plain notorch-native forward),
		// or to any α to tune it. nanollama stays the fallback when doe is absent.
		if doePresent {
			tc.nan.doeBin = "./doe_field"
			tc.nan.doeAlpha = "0.1"
			if a := os.Getenv("AM_LORA_ALPHA"); a != "" {
				tc.nan.doeAlpha = a
			}
			// step-3: the experts LEARN from the dream (notorch_step Oja, debt-driven)
			// only when opted in — default off (the proven yent config; no weight drift).
			// The mycelium spore persists the learned experts across dreams.
			tc.nan.doeTrain = "0"
			if os.Getenv("AM_DOE_TRAIN") == "1" {
				tc.nan.doeTrain = "1"
			}
			// cap the mycelium spore dir before the parliament loads (crash-safe: bounds
			// it every startup regardless of a clean prior shutdown). Keeps the highest-
			// step spores, so the one the parliament loads is never pruned.
			pruneMycelium(myceliumDir, myceliumKeep)
			// persistent daemon (perf + the parliament stays awake across dreams): start
			// the REPL once and reuse it. AM_DOE_DAEMON=0 forces the one-shot spawn (the
			// A/B knob). startDoeDaemon primes single-threaded here, BEFORE runSubconscious
			// and the breathing goroutine can call dream(); nil on failure => one-shot.
			if os.Getenv("AM_DOE_DAEMON") != "0" {
				tc.nan.doeD = startDoeDaemon(tc.nan.doeBin, tc.nan.gguf, tc.nan.doeAlpha, tc.nan.doeTrain)
			}
		}
		tc.seedCh = make(chan string, 1)
		tc.dreamCh = make(chan dreamResult, 1)
		tc.subDone = make(chan struct{})
		go runSubconscious(tc.nan, "./kk-cli", "weights/nano.kk.db", tc.seedCh, tc.dreamCh, tc.subDone)
	}
	// The subconscious can dream as a POLYPHONY (the chorus over the same nano body)
	// when ./chorus-arianna is built — used by the autonomous breathing.
	if os.Getenv("AM_CHORUS") != "0" && executablePresent("./chorus-arianna") && nanoWeightPresent {
		tc.chorusBin = "./chorus-arianna"
		tc.chorusGGUF = "weights/nano_arianna_f16.gguf"
	}
	return tc, nil
}

// stop tears the organism down in order: signal the subconscious goroutine
// (close seedCh) and join it (F-3 — bounded, so an in-flight dream can finish or
// hit its own deadline without wedging shutdown), then the voices, the ticker,
// the inner world.
func (tc *trioCtx) stop() {
	if tc.seedCh != nil {
		close(tc.seedCh)
		// wait past a FULL in-flight subconscious cycle so it finishes (or hits its
		// own ctx-kill) before teardown. The cycle is kkRetrieve (kkTimeout) THEN the
		// dream (dreamTimeout, or doeDreamTimeout when doe is the engine — longer),
		// sequential — so the join must budget for both.
		join := dreamTimeout
		if tc.nan != nil && tc.nan.doeBin != "" && doeDreamTimeout > join {
			join = doeDreamTimeout
		}
		select {
		case <-tc.subDone:
		case <-time.After(join + kkTimeout + 5*time.Second):
		}
	}
	// close the doe daemon — it saves its mycelium spore and exits before the rest tears
	// down. Under nano.mu so it can NOT race an in-flight generate(): if the subDone
	// join above timed out (a buffered cue can extend runSubconscious past the budget),
	// a dream may still hold the daemon; mu serializes close behind it (generate runs
	// under the same mu), and that dream's own ctx-deadline releases it.
	if tc.nan != nil {
		tc.nan.mu.Lock()
		tc.nan.doeD.close() // nil-safe (daemon may never have started)
		tc.nan.mu.Unlock()
	}
	// F-8 palliative (until the 4d-mmap nerve merges the field for real): both
	// daemons rewrite the shared soma at exit, so the last to close wins. Close
	// Janus (the face, which holds form) FIRST, so Resonance (the inner voice — the
	// field's carrier, whom the subconscious teaches) writes the soma LAST and keeps
	// the field overnight.
	tc.janusD.close()
	tc.resonD.close()
	close(tc.tickerDone)
	tc.iw.Stop()
}

// turn runs one trio exchange. Janus answers (the human line + the rolling
// context — he resists injection, so context is a hint, not a directive).
// Resonance murmurs with the last dream as an undertone (she is a receiver). The
// subconscious is seeded (the direct human→nano channel re-opens when the
// attention wanders inward) and any earlier dream surfaces. Each voice feeds the
// inner world. Returns the words; the caller prints and controls the loop.
func (tc *trioCtx) turn(human, context, lastDream string, surfaceDream bool) (janus, reson string, dr dreamResult, hasDream bool) {
	return tc.turnUntil(human, context, lastDream, surfaceDream, nil)
}

// turnUntil lets the long-lived chat stop between voice generations without
// killing a daemon that is currently saving its own evolving state. Batch mode
// passes no stop function and retains the ordinary full exchange.
func (tc *trioCtx) turnUntil(human, context, lastDream string, surfaceDream bool, stopping func() bool) (janus, reson string, dr dreamResult, hasDream bool) {
	before := tc.iw.GetSnapshot()
	if stopping != nil && stopping() {
		return
	}

	janusParts := []string{human}
	if strings.TrimSpace(context) != "" {
		janusParts = append(janusParts, context)
	}
	if surfaceDream && strings.TrimSpace(lastDream) != "" {
		janusParts = append(janusParts, lastDream)
	}
	janus = tc.janusD.ask(strings.Join(janusParts, " "))
	if liveVoiceTextVisible(janus) {
		tc.iw.ProcessText(janus)
	}
	if stopping != nil && stopping() {
		return
	}

	resonParts := []string{human}
	if liveVoiceTextVisible(janus) {
		resonParts = append(resonParts, janus)
	}
	if surfaceDream && strings.TrimSpace(lastDream) != "" {
		resonParts = append(resonParts, lastDream)
	}
	reson = tc.resonD.askInjected(human, strings.Join(resonParts, " "))
	if liveVoiceTextVisible(reson) {
		tc.iw.ProcessText(reson)
	}
	if stopping != nil && stopping() {
		return
	}

	if tc.nan != nil {
		cueParts := []string{human}
		if liveVoiceTextVisible(janus) {
			cueParts = append(cueParts, janus)
		}
		if liveVoiceTextVisible(reson) {
			cueParts = append(cueParts, reson)
		}
		cue := strings.Join(cueParts, " ")
		if tc.iw.GetSnapshot().WanderPull > 0.55 {
			cue = human
		}
		sendLatest(tc.seedCh, cue)
		if r, ok := recvDream(tc.dreamCh); ok {
			if carried := sanitizeLiveCarriedDream(r.dream); carried != "" {
				r.dream = carried
				tc.iw.ProcessText(carried)
				dr, hasDream = r, true
			}
		}
	}
	// How far her core affect (valence/arousal/coherence) traveled across the
	// exchange — the magnitude of "being moved" by the object.
	after := tc.iw.GetSnapshot()
	dV := after.Valence - before.Valence
	dA := after.Arousal - before.Arousal
	dC := after.Coherence - before.Coherence
	tc.lastMoved = float32(math.Sqrt(float64(dV*dV + dA*dA + dC*dC)))
	return
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--chat" {
		runChat()
		return
	}
	if len(os.Args) == 1 {
		fmt.Println("usage: metabolism --chat | <prompt>")
		return
	}
	runDemo(strings.Join(os.Args[1:], " "))
}

// runDemo runs the fixed self-duet on one seed — the smoke + race-verify path.
func runDemo(prompt string) {
	tc, err := startTrio()
	if err != nil {
		fmt.Println("metabolism:", err)
		return
	}
	defer tc.stop()

	calm := Snapshot{Arousal: 0.30, Coherence: 0.80}
	hot := Snapshot{Arousal: 0.60, Coherence: 0.80}
	fmt.Printf("┌─ arianna-metabolism  (hot daemons + inner-world in the loop + gating the rhythm)\n")
	fmt.Printf("│  rhythm map: budget(calm=0.30)=%d  budget(aroused=0.60)=%d\n", tickBudget(calm), tickBudget(hot))
	if tc.nan != nil {
		fmt.Printf("│  the subconscious is present (nano 88M, async — it dreams a turn behind)\n")
	}

	tc.iw.ProcessText(prompt)
	time.Sleep(400 * time.Millisecond)
	nExch := tickBudget(tc.iw.GetSnapshot())
	fmt.Printf("│  seed: %s\n│  exchange budget (from state): %d\n", prompt, nExch)

	// The direct human→nano channel: the human's raw words hit the subconscious
	// before the face has formed, so the first dream reacts to the human directly.
	if tc.nan != nil {
		sendLatest(tc.seedCh, prompt)
	}

	prevReson, lastDream := "", ""
	for i := 1; i <= nExch; i++ {
		janus, reson, dr, hasDream := tc.turn(prompt, prevReson, lastDream, false)
		fmt.Print("│\n")
		if liveVoiceTextVisible(janus) {
			fmt.Printf("│  ◐ [%d/%d] Janus: %s\n", i, nExch, janus)
		}
		if liveVoiceTextVisible(reson) {
			fmt.Printf("│  ◑ [%d/%d] Resonance: %s\n", i, nExch, reson)
			prevReson = reson
		} else {
			prevReson = ""
		}
		if hasDream {
			displayDream := sanitizeLiveVoiceText(dr.dream)
			if liveVoiceTextVisible(displayDream) {
				lastDream = displayDream
			}
			if displayFrag := sanitizeLiveVoiceText(dr.frag); liveVoiceTextVisible(displayFrag) {
				fmt.Printf("│  ◌ [%d/%d] from the books: %s\n", i, nExch, ellipsize(displayFrag, 90))
			}
			if liveVoiceTextVisible(displayDream) {
				fmt.Printf("│  ◓ [%d/%d] nano (subconscious): %s\n", i, nExch, displayDream)
			}
		}

		// M3: if a voice fell silent, stop instead of looping over empty turns.
		if tc.janusD.dead || tc.resonD.dead {
			fmt.Println("│  · a voice fell silent — ending the duet")
			break
		}
		s := tc.iw.GetSnapshot()
		d := tickDelay(s)
		fmt.Printf("│  · inner-world: arousal=%.3f coher=%.3f trauma=%.3f wander=%.3f loops=%d moved=%.3f viab=%.3f  | settle %v\n",
			s.Arousal, s.Coherence, s.TraumaLevel, s.WanderPull, s.LoopCount, tc.lastMoved, viability(s, tc.janusD.dead, tc.resonD.dead), d)
		// E3: re-read the budget — trauma mid-duet can cut it short.
		if i >= tickBudget(s) {
			fmt.Println("│  · the field settled — ending early")
			break
		}
		if i < nExch {
			time.Sleep(d)
		}
	}
	fmt.Println("└─ done — hot daemons, inner world in the loop, rhythm gated by it")
}

// viability collapses her self-preservation signals into one scalar in [0,1] —
// how alive/whole she is, distinct from field health (which drives learning, not
// self-preservation, per Damasio). A silent voice, saturated prophecy debt, trauma,
// or memory pressure each pull it down. Read-only telemetry (measure-first) — it
// feeds nothing back into generation yet; wiring it to the vagus is a deliberate
// step with Oleg.
func viability(s Snapshot, janusDead, resonDead bool) float32 {
	v := 1.0
	if janusDead {
		v -= 0.5
	}
	if resonDead {
		v -= 0.5
	}
	v -= float64(s.ProphecyDebt) / 10.0 * 0.3 // debt clamps at 10 (AddProphecyDebt)
	v -= float64(s.TraumaLevel) * 0.3
	v -= float64(s.MemoryPressure) * 0.2
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return float32(v)
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
