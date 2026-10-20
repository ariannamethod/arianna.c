package main

// chat.go — the live trio. Human speech enters the organism unchanged; Janus,
// Resonance, the nano, the inner world, and the shared field produce the answer.

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const innerStatePath = "weights/arianna.inner.state"

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

	lastDream := sanitizeLiveCarriedDream(tc.iw.RestoreMood(innerStatePath))

	fmt.Println("┌─ arianna — the trio (Janus · Resonance · the nano).  speak, /quit to leave.")
	if tc.nan != nil {
		fmt.Println("│  the subconscious is present (nano 88M, async — it dreams a turn behind)")
		fmt.Println("│  she breathes on her own — between your words she dreams, and the inner voice answers.")
		if tc.nan.doeBin != "" {
			if f, err := strconv.ParseFloat(tc.nan.doeAlpha, 64); err == nil && f == 0 {
				fmt.Printf("│  she dreams notorch-native through doe — the parliament is silenced (α=%s; unset AM_LORA_ALPHA to seat it)\n", tc.nan.doeAlpha)
			} else {
				fmt.Printf("│  the parliament is seated on her dreams (doe, α=%s; AM_LORA_ALPHA=0 to silence it)\n", tc.nan.doeAlpha)
			}
			if tc.nan.doeTrain == "1" {
				fmt.Println("│  the parliament learns from her dreams (--train: the experts grow toward what surfaces)")
			}
			if tc.nan.doeD != nil {
				fmt.Println("│  the parliament stays awake between dreams")
			}
		}
	}
	if lastDream != "" {
		fmt.Printf("│  (she returns carrying a dream: %s)\n", ellipsize(lastDream, 70))
	}

	var voiceMu sync.Mutex
	breathStop := make(chan struct{})
	var breathStopOnce sync.Once
	stopBreathing := func() {
		breathStopOnce.Do(func() { close(breathStop) })
	}
	breathDone := make(chan struct{})
	go runBreathing(tc, &voiceMu, &lastDream, breathStop, breathDone)

	faceFR := newFieldReader(fieldPath)
	defer faceFR.close()
	recordLiveMetric("startup", tc, faceFR.read(), nil)

	// systemd/FIFO launchers may hand us a non-blocking pipe. Scanner treats an
	// initial EAGAIN as terminal input failure, so restore normal stream semantics
	// before entering the long-lived chat loop.
	if err := syscall.SetNonblock(int(os.Stdin.Fd()), false); err != nil {
		fmt.Printf("│  input stream: %v\n", err)
	}
	var signalShutdown atomic.Bool
	shutdownSignals := make(chan os.Signal, 1)
	shutdownRequested := make(chan struct{})
	signalWatcherDone := make(chan struct{})
	signal.Notify(shutdownSignals, os.Interrupt, syscall.SIGTERM)
	go func() {
		select {
		case <-shutdownSignals:
			signalShutdown.Store(true)
			stopBreathing()
			close(shutdownRequested)
		case <-signalWatcherDone:
		}
	}()
	defer func() {
		signal.Stop(shutdownSignals)
		close(signalWatcherDone)
	}()
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	type chatInput struct {
		text string
		err  error
		done bool
	}
	input := make(chan chatInput, 1)
	go func() {
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line != "" && line != "/quit" && line != "/exit" {
				tc.pendingHumans.Add(1)
			}
			select {
			case input <- chatInput{text: line}:
			case <-shutdownRequested:
				return
			}
		}
		input <- chatInput{err: sc.Err(), done: true}
	}()
	prevReson := ""
	fmt.Print("│\n└▶ ")
chatLoop:
	for {
		var in chatInput
		select {
		case <-shutdownRequested:
			break chatLoop
		case in = <-input:
			if in.done {
				if in.err != nil && !signalShutdown.Load() {
					fmt.Printf("│  input stream closed: %v\n", in.err)
				}
				break chatLoop
			}
		}

		human := strings.TrimSpace(in.text)
		if human == "" {
			fmt.Print("└▶ ")
			continue
		}
		if human == "/quit" || human == "/exit" {
			break chatLoop
		}

		fmt.Printf("│  ◇ human: %s\n", ellipsize(human, 140))
		liveMetricAdd(&tc.humanTurns, 1)
		voiceMu.Lock()
		if signalShutdown.Load() {
			voiceMu.Unlock()
			break chatLoop
		}
		tc.iw.ProcessText(human)
		if tc.nan != nil {
			sendLatest(tc.seedCh, human)
		}

		fs := faceFR.read()
		janus, reson, dr, hasDream := tc.turnUntil(human, prevReson, lastDream, fs.surfaces(), signalShutdown.Load)
		if signalShutdown.Load() {
			voiceMu.Unlock()
			break chatLoop
		}
		janusVisible := liveVoiceTextVisible(janus)
		resonVisible := liveVoiceTextVisible(reson)
		if janusVisible {
			printLiveVoice("◐ Janus", janus)
			liveMetricAdd(&tc.janusTurns, 1)
		}
		if resonVisible {
			printLiveVoice("◑ Resonance", reson)
			liveMetricAdd(&tc.resonanceTurns, 1)
			prevReson = reson
		} else {
			prevReson = ""
		}

		if hasDream {
			liveMetricAdd(&tc.dreams, 1)
			displayDream := sanitizeLiveVoiceText(dr.dream)
			if displayDream != "" {
				lastDream = displayDream
			}
			if displayFrag := sanitizeLiveVoiceText(dr.frag); displayFrag != "" {
				fmt.Printf("│  ◌ from the books: %s\n", ellipsize(displayFrag, 90))
			}
			if displayDream != "" {
				fmt.Printf("│  ◓ nano (subconscious): %s\n", displayDream)
				liveMetricAdd(&tc.nanoTurns, 1)
			}
		}

		if tc.janusD.dead {
			if err := tc.janusD.respawn(); err != nil {
				fmt.Println("│  · Janus fell silent and could not be revived:", err)
				voiceMu.Unlock()
				break
			}
			fmt.Println("│  · Janus fell silent — revived.")
		}
		if tc.resonD.dead {
			if err := tc.resonD.respawn(); err != nil {
				fmt.Println("│  · Resonance fell silent and could not be revived:", err)
				voiceMu.Unlock()
				break
			}
			fmt.Println("│  · Resonance fell silent — revived.")
		}
		recordLiveMetric("human_turn", tc, faceFR.read(), map[string]any{
			"janus_visible":            janusVisible,
			"resonance_visible":        resonVisible,
			"nano_result_visible":      hasDream,
			"janus_response_chars":     len(janus),
			"resonance_response_chars": len(reson),
		})
		tc.pendingHumans.Add(-1)
		voiceMu.Unlock()
		fmt.Print("│\n└▶ ")
	}

	stopBreathing()
	breathJoin := dreamTimeout
	if tc.nan != nil && tc.nan.doeBin != "" {
		breathJoin = doeDreamTimeout
	}
	select {
	case <-breathDone:
	case <-time.After(breathJoin + kkTimeout + 5*time.Second):
	}

	fmt.Println()
	if err := tc.iw.SaveState(innerStatePath, lastDream); err != nil {
		fmt.Println("(could not save the inner state:", err, ")")
	} else {
		fmt.Println("(she will remember.)")
	}
	tc.stop()
	harvestField()
}

func printLiveVoice(label, text string) {
	if !strings.Contains(text, "\n") {
		fmt.Printf("│  %s: %s\n", label, text)
		return
	}
	fmt.Printf("│  %s:\n", label)
	for _, line := range strings.Split(text, "\n") {
		fmt.Printf("│    %s\n", line)
	}
}

func harvestField() {
	if _, err := os.Stat("./harvest_delta"); err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "./harvest_delta",
		resonGGUF, resonWTE, resonCooc, resonDelta, resonVocab, resonDim, "8").CombinedOutput()
	for _, line := range strings.Split(string(out), "\n") {
		if i := strings.Index(line, "|B|="); i >= 0 {
			fmt.Printf("│  the organism consolidated what surfaced — δ %s\n", strings.TrimSpace(line[i:]))
			return
		}
	}
	reason := "nothing surfaced to consolidate"
	if lines := strings.Split(strings.TrimSpace(string(out)), "\n"); err != nil && len(lines) > 0 {
		reason = lines[len(lines)-1]
	}
	fmt.Printf("│  (she could not consolidate — %s)\n", reason)
}
