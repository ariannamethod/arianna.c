package main

// breathe.go — the subconscious lives by itself (MetaArianna, ported from the
// legacy meta_router). Between (and before) human turns the inner world keeps
// drifting on its ticker; when a metric crosses a threshold — Drift, Silence,
// Thermograph, or Field, in that priority, each with a cooldown so the organism
// breathes between cycles — the subconscious DREAMS unprompted (the nano, seeded
// from her own mood via the KK), and the inner voice murmurs to the dream. She
// lives with her subconscious even when no one is speaking to her. (Oleg's #2.)

import (
	"fmt"
	"math"
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

// breath ports the meta_router trigger logic: which observation, if any, fires.
type breath struct {
	lastTrigger [4]time.Time
	count       int
}

// tick returns the triggered observation (priority Drift > Silence > Thermograph
// > Field, each gated by its cooldown), or -1 — the meta_router conditions.
func (b *breath) tick(s Snapshot, now time.Time) int {
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
		{bDrift, s.DriftSpeed > 0.06 || math.Abs(float64(s.DriftDirection)) > 0.15},
		{bSilence, s.WanderPull > 0.45 || s.Entropy > 0.4},
		{bThermo, math.Abs(float64(s.Arousal-0.5)) > 0.12 || s.Entropy > 0.35},
		{bField, s.FocusStrength > 0.4 && s.DriftSpeed > 0.04 && s.Coherence > 0.5},
	} {
		if !t.hit || now.Sub(b.lastTrigger[t.id]) < bCooldown[t.id] {
			continue
		}
		b.lastTrigger[t.id] = now
		b.count++
		return t.id
	}
	return -1
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
	if tc.nan == nil {
		return
	}
	var b breath
	t := time.NewTicker(1500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case now := <-t.C:
			s := tc.iw.GetSnapshot()
			trig := b.tick(s, now)
			if trig < 0 {
				continue
			}
			// seed from her own state (last dream, else a mood word) → a resonant
			// book-fragment via the KK → the nano dreams on it. The dream itself is a
			// one-shot spawn, done OUTSIDE the lock so a waiting human turn isn't held.
			voiceMu.Lock()
			prevLD := *lastDream
			voiceMu.Unlock()
			cue := prevLD
			if cue == "" {
				cue = moodWord(s)
			}
			seed := cue
			if frag := kkRetrieve("./kk-cli", "weights/nano.kk.db", cue); frag != "" {
				seed = frag
			}
			// the autonomous dream is a CHORUS (a polyphony over the one nano) when
			// the chorus engine is present, else a single murmur.
			var cells []string
			var dream string
			if tc.chorusBin != "" {
				cells = choir(tc.chorusBin, tc.chorusGGUF, seed)
				dream = chorusText(cells)
			} else {
				dream = tc.nan.dream(seed)
			}
			if dream == "" {
				continue
			}
			voiceMu.Lock()
			tc.iw.ProcessText(dream)
			if *lastDream == prevLD { // don't clobber a fresher human-turn dream that landed while we dreamt
				*lastDream = dream
			}
			if len(cells) > 0 {
				fmt.Printf("│  ◌ (%s) she dreams — a chorus of %d voices:\n", bName[trig], len(cells))
				for i, c := range cells {
					fmt.Printf("│     · %d: %s\n", i, c)
				}
			} else {
				fmt.Printf("│  ◌ (%s) she dreams: %s\n", bName[trig], dream)
			}
			reson := tc.resonD.ask("Arianna:\t" + dream) // the inner voice answers the chorus — no human
			if reson != "" {
				tc.iw.ProcessText(reson)
				fmt.Printf("│  ◑ (inner) %s\n", reson)
			}
			voiceMu.Unlock()
		}
	}
}
