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
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// voice is a persistent --daemon process talked to over stdin/stdout, framed by
// a "<END>" line after each reply.
type voice struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Scanner
	dead bool   // set when the daemon stops responding (EOF before the <END> frame)
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
	sc := bufio.NewScanner(outPipe)
	sc.Buffer(make([]byte, 1<<20), 1<<20) // tolerate long lines
	return &voice{cmd: cmd, in: in, out: sc, bin: bin, args: args}, nil
}

// respawn revives a fallen voice in place: kill and reap the old daemon, then start a fresh one
// with the same bin+args. A hot voice daemon can stop framing <END> after a turn or two; reviving
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

// voiceTimeout bounds one ask: our voices emit a fixed token budget (-n 28). On an idle
// machine that frames <END> in seconds, but a 176M CPU voice under heavy contention (other
// jobs saturating the cores) can legitimately take far longer — 30s treated a merely-slow
// voice as wedged and killed it, silencing the trio. The ceiling is generous so a slow-but-
// alive voice finishes its turn; a genuine wedge is still caught (and a real death is handled
// by respawn). Tunable via AM_VOICE_TIMEOUT (seconds).
var voiceTimeout = func() time.Duration {
	if s := os.Getenv("AM_VOICE_TIMEOUT"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 && n <= 3600 {
			return time.Duration(n) * time.Second // cap at 1h — no time.Duration overflow
		}
	}
	return 120 * time.Second
}()

// ask sends one request line and reads the reply up to the <END> frame. If the
// daemon dies (stdin closed, or EOF before <END>), it marks the voice dead so the
// caller can stop instead of looping over silent empty turns (Mythos M3). The read
// runs under a deadline: a daemon that wedges before <END> would otherwise hold
// voiceMu forever (a human turn or the autonomous breathing stuck for good), so on
// timeout the process is killed (which unblocks the read with EOF) and marked dead.
func (v *voice) ask(line string) string {
	if _, err := fmt.Fprintln(v.in, line); err != nil {
		v.dead = true
		return ""
	}
	type reply struct {
		text   string
		sawEnd bool
	}
	ch := make(chan reply, 1)
	go func() {
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
		ch <- reply{cutSentence(strings.Join(strings.Fields(b.String()), " ")), sawEnd}
	}()
	select {
	case r := <-ch:
		if !r.sawEnd {
			v.dead = true // Scan returned false before the <END> frame — the daemon is gone
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
	bodyInventory  bodyInventoryReceipt
	iw             *InnerWorld
	tickerDone     chan struct{}
	lastMoved      float32 // being-moved: core-affect displacement of the last turn (telemetry, no feedback)
}

// startTrio brings the organism up: the inner world on a 100ms ticker (its single
// clock), both voices hot as --daemon processes, and the subconscious as a
// background dreamer (absent binary/GGUF => the duet runs without it).
func startTrio() (*trioCtx, error) {
	inventory := inspectBodyInventory(bodyInventoryRoot())
	if err := writeBodyInventoryReceipt(os.Getenv("AM_BODY_INVENTORY_LOG"), inventory); err != nil {
		return nil, fmt.Errorf("body inventory receipt: %w", err)
	}
	if err := requireBodyInventoryLiveTrio(inventory); err != nil {
		return nil, err
	}

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

	janusD, err := startVoice("./arianna", []string{"-t", "0.8", "--top-p", "0.9", "-n", "28"})
	if err != nil {
		close(tickerDone)
		iw.Stop()
		return nil, fmt.Errorf("janus daemon: %w", err)
	}
	resonD, err := startVoice("./arianna_resonance", []string{"--alpha", "5", "-t", "0.7", "--top-p", "1.0", "-n", "28"})
	if err != nil {
		janusD.close()
		close(tickerDone)
		iw.Stop()
		return nil, fmt.Errorf("resonance daemon: %w", err)
	}

	tc := &trioCtx{janusD: janusD, resonD: resonD, bodyInventory: inventory, iw: iw, tickerDone: tickerDone}
	// The subconscious body is the nano GGUF; it runs through the vendored doe engine
	// (the LoRA parliament, #3) and/or the nanollama one-shot. It is present if the
	// GGUF + at least one engine exists — so doe alone (no nanollama) still dreams.
	const nanoGGUF = "weights/nano_arianna_f16.gguf"
	doePresent := inventory.organPresent("doe-binary")
	nanoWeightPresent := inventory.organPresent("nano-weight")
	if inventory.organPresent("nano-binary") && nanoWeightPresent {
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
	if inventory.organPresent("chorus-binary") && nanoWeightPresent {
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
func (tc *trioCtx) turn(human, context, lastDream string, surfaceDream bool, turnRouteObs admissionLiveRouteTurnObservation) (janus, reson string, dr dreamResult, hasDream bool) {
	// Core-self (Damasio) instrumentation: her felt state before this exchange's
	// object touches it. Read-only telemetry — the being-moved delta computed at
	// the end feeds nothing back into generation (that wiring is a separate,
	// deliberate step).
	before := tc.iw.GetSnapshot()
	janusPrompt := human
	if context != "" {
		janusPrompt = human + " " + context
	}
	// When the field is expressive (summer / running), the inner dream lightly
	// SURFACES to the face — a faint undertone in Janus's prompt (ellipsized), not a
	// directive. Janus resists injection by design, so it stays a trace; a quiet /
	// wintering field keeps the dream inward (only Resonance hears it below).
	if surfaceDream && lastDream != "" {
		janusPrompt += " " + ellipsize(lastDream, 60)
	}
	janus = tc.janusD.ask(janusPrompt)
	tc.iw.ProcessText(janus)

	resonInject := janus + " " + human
	if lastDream != "" {
		resonInject += " " + lastDream
	}
	reson = tc.resonD.ask("Arianna:\t" + resonInject)
	tc.iw.ProcessText(reson)

	if tc.nan != nil {
		cue := human + " " + janus + " " + reson
		if tc.iw.GetSnapshot().WanderPull > 0.55 {
			cue = human // the direct human→nano channel: the mind returns to the raw words
		}
		sendLatest(tc.seedCh, cue)
		if r, ok := recvDream(tc.dreamCh); ok {
			admitDreamToInnerWorldWithTurnObservation(tc.iw, &r, "human-turn", turnRouteObs)
			dr, hasDream = r, true
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
	if len(os.Args) > 1 && os.Args[1] == "--body-inventory-smoke" {
		if err := runBodyInventorySmoke(); err != nil {
			fmt.Println("body-inventory-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-smoke" {
		if err := runAdmissionSmoke(); err != nil {
			fmt.Println("admission-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-gate-smoke" {
		if err := runAdmissionLiveRouteGateSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-gate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-chat-smoke" {
		if err := runAdmissionLiveRouteChatSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-chat-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-smoke" {
		if err := runAdmissionLiveRouteTurnSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-choice-smoke" {
		if err := runAdmissionLiveRouteTurnChoiceSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-choice-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-request-smoke" {
		if err := runAdmissionLiveRouteTurnRequestSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-request-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-generation-job-smoke" {
		if err := runAdmissionLiveRouteTurnGenerationJobSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-generation-job-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-generation-job-inventory-gate-smoke" {
		if err := runAdmissionLiveRouteTurnGenerationJobInventoryGateSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-generation-job-inventory-gate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-route-boundary-smoke" {
		if err := runAdmissionLiveRouteTurnRouteBoundarySmoke(); err != nil {
			fmt.Println("admission-live-route-turn-route-boundary-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-drift-artifact-smoke" {
		if err := runAdmissionLiveRouteBoundaryReportDriftArtifactSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-drift-artifact-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-assert-smoke" {
		if err := runAdmissionLiveRouteBoundaryReportAssertSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-assert-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-assert-full-chain-smoke" {
		if err := runAdmissionLiveRouteBoundaryReportAssertFullChainSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-assert-full-chain-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-failed-diagnostics-assert-smoke" {
		if err := runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssertSmoke(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-failed-diagnostics-assert-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-stage-chain" {
		if err := runAdmissionLiveRouteBoundaryReportStageChain(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-stage-chain:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-assert" {
		if err := runAdmissionLiveRouteBoundaryReportAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-assert-full-chain" {
		if err := runAdmissionLiveRouteBoundaryReportAssertFullChain(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-assert-full-chain:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-readiness-assert" {
		if err := runAdmissionLiveRouteWeightedReadinessAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-readiness-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-readiness-precondition" {
		if err := runAdmissionLiveRouteWeightedReadinessPrecondition(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-readiness-precondition:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-contract" {
		if err := runAdmissionLiveRouteWeightedAdmissionContract(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-contract:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-contract-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionContractAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-contract-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-authority" {
		if err := runAdmissionLiveRouteWeightedAdmissionAuthority(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-authority:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-authority-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionAuthorityAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-authority-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-permit" {
		if err := runAdmissionLiveRouteWeightedAdmissionPermit(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-permit:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-permit-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionPermitAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-permit-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-seal" {
		if err := runAdmissionLiveRouteWeightedAdmissionSeal(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-seal:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-seal-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionSealAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-seal-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-final-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionFinalGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-final-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-final-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionFinalGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-final-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-intent" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntent(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-intent:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-intent-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-intent-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-receiver" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiver(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-receiver:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-receiver-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-receiver-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-observation" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservation(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-observation:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-observation-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-observation-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-boundary" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-boundary:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-boundary-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-boundary-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-preflight" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-preflight:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-preflight-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-preflight-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate-store" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate-store:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate-store-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate-store-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate-store-reader" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate-store-reader:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-proof" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-proof:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-proof-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-proof-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-decision" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-decision:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-decision-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-decision-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-promotion" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-promotion:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-promotion-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-promotion-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-switch" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-switch:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-switch-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-switch-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-enable-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-enable-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-enable-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-live-stage" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-live-stage:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-live-stage-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-live-stage-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-contract" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-contract:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-writer-contract-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-writer-contract-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-readiness" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-readiness:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-readiness-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-readiness-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-permit" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-permit:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-permit-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-permit-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-authority" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-authority:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-authority-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-authority-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-seal" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-seal:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-seal-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-seal-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiver(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-receiver-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPrecondition(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotion(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitch(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStage(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContract(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementation(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerImplementationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-implementation-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistence(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerPersistenceAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-persistence-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerification(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerVerificationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-verification-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadiness(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-readiness-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermit(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-permit-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthority(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-authority-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSeal(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-seal-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntent(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-intent-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiver(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservation(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundary(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflight(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGate(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-assert" {
		if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-boundary-report-failed-diagnostics-assert" {
		if err := runAdmissionLiveRouteBoundaryReportFailedDiagnosticsAssert(os.Args[2:]); err != nil {
			fmt.Println("admission-live-route-boundary-report-failed-diagnostics-assert:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-shell-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateShellSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-shell-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-execution-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateExecutionSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-execution-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-runner-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateRunnerSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-runner-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-runner-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectRunnerSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-runner-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-chat-shadow-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-decision-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-decision-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-promotion-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-promotion-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-switch-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-switch-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-enable-gate-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-enable-gate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-live-stage-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-live-stage-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-writer-preflight-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-writer-preflight-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-writer-inventory-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-writer-contract-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-writer-contract-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-admission-ledger-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-writer-implementation-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-writer-receipt-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-rollback-implementation-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-ledger-implementation-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-ledger-persistence-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-ledger-verification-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-ledger-verification-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-readiness-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-readiness-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-permit-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-permit-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-seal-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-seal-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-final-gate-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-final-gate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-intent-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-receiver-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-observation-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-boundary-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-preflight-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-gate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-candidate-store-reader-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateNanoDirectChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-nano-direct-resonance-graft-admission-proof-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-runner-emit" {
		if err := runAdmissionLiveRouteTurnCandidateRunnerEmit(); err != nil {
			fmt.Fprintln(os.Stderr, "admission-live-route-turn-candidate-runner-emit:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-generator-adapter-smoke" {
		if err := runAdmissionLiveRouteTurnGeneratorAdapterSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-generator-adapter-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-draft-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateDraftSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-draft-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-draft-review-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateDraftReviewSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-draft-review-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-admission-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateAdmissionSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-admission-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-admission-adapter-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateAdmissionAdapterSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-admission-adapter-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-admission-chat-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateAdmissionChatSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-admission-chat-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-candidate-admission-chat-shadow-smoke" {
		if err := runAdmissionLiveRouteTurnCandidateAdmissionChatShadowSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-candidate-admission-chat-shadow-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-review-smoke" {
		if err := runAdmissionLiveRouteTurnReviewSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-review-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-bridge-smoke" {
		if err := runAdmissionLiveRouteTurnBridgeSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-bridge-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-live-route-turn-bridge-admission-smoke" {
		if err := runAdmissionLiveRouteTurnBridgeAdmissionSmoke(); err != nil {
			fmt.Println("admission-live-route-turn-bridge-admission-smoke:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-sample" {
		if err := runAdmissionSample(); err != nil {
			fmt.Println("admission-sample:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-route-compare" {
		if err := runAdmissionRouteCompare(); err != nil {
			fmt.Println("admission-route-compare:", err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--admission-qloop-sweep" {
		if err := runAdmissionQloopSweep(); err != nil {
			fmt.Println("admission-qloop-sweep:", err)
			os.Exit(1)
		}
		return
	}
	prompt := "What is resonance?"
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}
	runDemo(prompt)
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
		janus, reson, dr, hasDream := tc.turn(prompt, prevReson, lastDream, false, admissionLiveRouteTurnObservation{})
		fmt.Printf("│\n│  ◐ [%d/%d] Janus: %s\n", i, nExch, janus)
		fmt.Printf("│  ◑ [%d/%d] Resonance: %s\n", i, nExch, reson)
		prevReson = reson
		if hasDream {
			if dr.admitted() {
				lastDream = dr.dream
				if dr.frag != "" {
					fmt.Printf("│  ◌ [%d/%d] from the books: %s\n", i, nExch, ellipsize(dr.frag, 90))
				}
				fmt.Printf("│  ◓ [%d/%d] nano (subconscious): %s\n", i, nExch, dr.dream)
			} else {
				fmt.Printf("│  ◓ [%d/%d] nano candidate (%s): %s\n", i, nExch, dr.admissionLabel(), ellipsize(dr.dream, 90))
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
