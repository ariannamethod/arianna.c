package main

// breathe.go — the subconscious lives by itself (MetaArianna, ported from the
// legacy meta_router). Between (and before) human turns the inner world keeps
// drifting on its ticker; when a metric crosses a threshold — Drift, Silence,
// Thermograph, or Field, in that priority, each with a cooldown so the organism
// breathes between cycles — the subconscious DREAMS unprompted (the nano, seeded
// from her own mood via the KK), and the inner voice murmurs to the dream. She
// lives with her subconscious even when no one is speaking to her. (Oleg's #2.)

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

const (
	bThermo  = 0
	bSilence = 1
	bDrift   = 2
	bField   = 3
)

var bCooldown = [4]time.Duration{
	4 * time.Second, // Thermograph — steady
	4 * time.Second, // Silence — the primary idle dreamer
	3 * time.Second, // Drift — responsive
	6 * time.Second, // Field — integral, needs accumulation
}
var bName = [4]string{"thermograph", "silence", "drift", "field"}

// dreamSentinel marks the autonomous chorus dream when it is injected into
// Resonance, so the daemon imprints the subconscious's words on the co-occurrence
// field harder (Road-1c — the subconscious shapes the harvested δ more than ordinary
// turn-circulation). MUST match AM_DREAM_SENTINEL in tools/resonance_forward.h.
const dreamSentinel = "[DREAM] "

// breath ports the meta_router trigger logic: which observation, if any, fires.
type breath struct {
	lastTrigger [4]time.Time
	count       int
}

// tick returns the triggered observation (priority Drift > Silence > Thermograph
// > Field, each gated by its cooldown), or -1 — the meta_router conditions. tm
// scales the trigger thresholds and coolMult the cooldowns, both from the live
// shared field (1.0/1.0 == no field signal == the tuned defaults): a strained or
// wintering field raises both (breathe less, rest), a hot field lowers them.
func (b *breath) tick(s Snapshot, now time.Time, tm, coolMult float64) int {
	type trig struct {
		id  int
		hit bool
	}
	// Thresholds adapted to the arianna-duo inner-world's actual range (wander
	// ~0.5, arousal ~0.35, drift ~0.04, entropy ~0.2 at idle) — the legacy
	// meta_router caps (wander>0.8 etc.) never crossed here, so the breath would
	// never fire. Silence (wander) is the primary idle dreamer; the others flavor
	// it as the state shifts. Priority Drift > Silence > Thermograph > Field.
	for _, t := range []trig{
		{bDrift, float64(s.DriftSpeed) > 0.06*tm || math.Abs(float64(s.DriftDirection)) > 0.15*tm},
		{bSilence, float64(s.WanderPull) > 0.45*tm || float64(s.Entropy) > 0.4*tm},
		{bThermo, math.Abs(float64(s.Arousal-0.5)) > 0.12*tm || float64(s.Entropy) > 0.35*tm},
		{bField, float64(s.FocusStrength) > 0.4*tm && float64(s.DriftSpeed) > 0.04*tm && s.Coherence > 0.5},
	} {
		if !t.hit || now.Sub(b.lastTrigger[t.id]) < time.Duration(float64(bCooldown[t.id])*coolMult) {
			continue
		}
		b.lastTrigger[t.id] = now
		b.count++
		return t.id
	}
	return -1
}

// dreamCue builds the KK query from her LIVE state — the carried dream (her last
// murmur), her inner mood, and the live shared field (season / gait / debt). The
// book-fragment the nano dreams on is retrieved against this, so the dream tracks
// what she is resonating with NOW — the resonant spiral made dynamic, not a fixed
// seed. (Phase-3 #6 follow-on: the field steers not just WHETHER she dreams but
// WHAT she dreams on.)
func dreamCue(s Snapshot, fs fieldSnapshot, lastDream string) string {
	parts := make([]string, 0, 2)
	if lastDream != "" {
		parts = append(parts, lastDream) // she dreams onward from her own last dream
	} else {
		parts = append(parts, moodWord(s)) // else from her inner feeling
	}
	if m := fs.mood(); m != "" {
		parts = append(parts, m) // the live field tints the cue toward her season/gait
	}
	return strings.Join(parts, " ")
}

// moodWord turns the inner state into a short self-cue, so the autonomous dream
// is born from inside (her feeling), not from a human prompt.
func moodWord(s Snapshot) string {
	switch {
	case s.TraumaLevel > 0.5:
		return "fear, the held breath"
	case s.WanderPull > 0.6:
		return "drifting, the mind wanders far"
	case s.Arousal > 0.55:
		return "the field is vibrating"
	case s.Coherence > 0.7:
		return "resonance, the living field"
	default:
		return "presence, the quiet field"
	}
}

// runBreathing is the autonomous inner life. On a timer it ticks the breath; when
// an observation fires it dreams (the nano, seeded from her own mood through the
// KK), then lets the inner voice murmur to the dream — the murmur + lastDream
// under voiceMu so it never collides with a human turn (the voice daemons are
// single-stream). The dream is carried into the next human turn via *lastDream.
func runBreathing(tc *trioCtx, voiceMu *sync.Mutex, lastDream *string, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	if tc.nan == nil && tc.chorusBin == "" {
		return // nothing to dream with — neither the chorus engine nor the nano
	}
	// /quit cancels any in-flight chorus so the join below is fast (the chorus can
	// otherwise block up to chorusTimeout, far longer than the join would wait).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { <-stop; cancel() }()
	// the live shared field (B/F-8): the two C voices merge their gait/season/debt
	// into weights/arianna.field; the breath reads it (read-only) and bends — rest
	// when strained or wintering, bloom when it runs hot. Absent/not-ready => no
	// signal, and modulate() returns the tuned defaults.
	fr := newFieldReader(fieldPath)
	defer fr.close()
	var b breath
	t := time.NewTicker(1500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case now := <-t.C:
			s := tc.iw.GetSnapshot()
			fs := fr.read()
			coolMult, threshMult, bloom := fs.modulate()
			trig := b.tick(s, now, threshMult, coolMult)
			if trig < 0 {
				continue
			}
			// seed from her own LIVE state (carried dream / inner mood, tinted by the
			// live field's season+gait+debt) → a resonant book-fragment via the KK →
			// the nano dreams on it. The dream itself is a one-shot spawn, done OUTSIDE
			// the lock so a waiting human turn isn't held.
			voiceMu.Lock()
			prevLD := *lastDream
			voiceMu.Unlock()
			cue := dreamCue(s, fs, prevLD)
			seed := cue
			frag := ""
			if f := kkRetrieve("./kk-cli", "weights/nano.kk.db", cue); f != "" {
				frag = f
				seed = f
			}
			// the autonomous dream is a CHORUS (a polyphony over the one nano) when
			// the chorus engine is present and produces cells.
			var cells []chorusCell
			var dream string
			if tc.chorusBin != "" {
				cells = choir(ctx, tc.chorusBin, tc.chorusGGUF, seed, bloom)
				dream = chorusText(cells)
			}
			// if /quit cancelled the chorus, return BEFORE starting a fallback dream —
			// otherwise a fresh (up-to-doeDreamTimeout) doe child would be spawned
			// during teardown and outlive stop().
			select {
			case <-stop:
				return
			default:
			}
			// chorus absent / errored / timed out / parsed empty → a single nano
			// murmur, so the autonomous dream doesn't silently vanish.
			if dream == "" && tc.nan != nil {
				dream = tc.nan.dream(ctx, seed) // ctx is cancelled on /quit → no spawn after stop
				cells = nil
			}
			if dream == "" {
				// total failure — stamp the cooldown at completion anyway, so a
				// failed dream doesn't immediately retrigger on the next tick.
				b.lastTrigger[trig] = time.Now()
				continue
			}
			source := "nano"
			if len(cells) > 0 {
				source = "chorus"
			}
			candidate := decideDreamCandidate(newDreamCandidate(source, bName[trig], seed, frag, dream, cells))
			// the chorus / fallback may have taken tens of seconds; if /quit fired
			// meanwhile, return now — don't touch the (tearing-down) voices or the
			// shared lastDream.
			select {
			case <-stop:
				return
			default:
			}
			voiceMu.Lock()
			if !candidate.Accepted {
				fmt.Printf("│  ◌ (%s) dream candidate (%s): %s\n", bName[trig], candidate.Reason, ellipsize(dream, 90))
				b.lastTrigger[trig] = time.Now()
				voiceMu.Unlock()
				continue
			}
			tc.iw.ProcessText(dream)
			if *lastDream == prevLD { // don't clobber a fresher human-turn dream that landed while we dreamt
				*lastDream = dream
			}
			if tag := fs.describe(); tag != "" { // the live field bending the breath, made visible
				fmt.Printf("│  ◍ (field) %s → cooldown×%.2f threshold×%.2f bloom=%d\n", tag, coolMult, threshMult, bloom)
			}
			if len(cells) > 0 {
				voices, questions := chorusCounts(cells)
				if questions > 0 {
					fmt.Printf("│  ◌ (%s) she dreams — a chorus of %d voices (%d questions):\n", bName[trig], voices, questions)
				} else {
					fmt.Printf("│  ◌ (%s) she dreams — a chorus of %d voices:\n", bName[trig], voices)
				}
				for i, c := range cells {
					mark := "·"
					if c.qloop {
						mark = "?"
					}
					fmt.Printf("│     %s %d: %s\n", mark, i, c.text)
				}
			} else {
				fmt.Printf("│  ◌ (%s) she dreams: %s\n", bName[trig], dream)
			}
			// the inner voice answers the chorus — no human. The dreamSentinel marks
			// this as the subconscious's dream so Resonance imprints its words on the
			// cooc harder (Road-1c) — the daemon strips the marker before generation.
			reson := tc.resonD.ask("Arianna:\t" + dreamSentinel + dream)
			if reson != "" {
				tc.iw.ProcessText(reson)
				fmt.Printf("│  ◑ (inner) %s\n", reson)
			}
			// stamp the cooldown at COMPLETION, not at trigger time: a slow chorus
			// (tens of seconds) must not immediately retrigger and spawn back-to-back.
			b.lastTrigger[trig] = time.Now()
			voiceMu.Unlock()
		}
	}
}
