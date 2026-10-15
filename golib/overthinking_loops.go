// inner_world/overthinking_loops.go — Overthinking and spiral detection
// ═══════════════════════════════════════════════════════════════════════════════
// המחשבות סובבות עצמן
// Thoughts circling themselves
// ═══════════════════════════════════════════════════════════════════════════════
//
// Overthinking is detected from observed repetition within and across turns.
// Runtime code does not assign abstraction or self-reference to a word list.
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"time"
)

// OverthinkingLoops detects and manages thought spirals
type OverthinkingLoops struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Detection state
	lastText string // previous utterance, for the cross-turn loop signal (High brain)

	// Thresholds
	repetitionThreshold int // concepts repeated this many times = spiral
}

// NewOverthinkingLoops creates a new overthinking detector
func NewOverthinkingLoops() *OverthinkingLoops {
	return &OverthinkingLoops{
		stop:                make(chan struct{}),
		repetitionThreshold: 3,
	}
}

func (ol *OverthinkingLoops) Name() string {
	return "overthinking_loops"
}

func (ol *OverthinkingLoops) Start(world *InnerWorld) {
	ol.world = world
	ol.running = true
	if world.async {
		go ol.run()
	}
}

func (ol *OverthinkingLoops) Stop() {
	if ol.running {
		close(ol.stop)
		ol.running = false
	}
}

func (ol *OverthinkingLoops) Step(dt float32) {
	if ol.world == nil {
		return
	}

	state := ol.world.State

	// Natural decay of overthink metrics
	state.mu.Lock()
	if state.LoopCount > 0 {
		state.LoopCount = max(0, state.LoopCount-1)
	}
	if state.AbstractionDepth > 0 && dt > 0.5 { // Slower decay for abstraction
		state.AbstractionDepth = max(0, state.AbstractionDepth-1)
	}
	if state.SelfRefCount > 0 && dt > 0.3 {
		state.SelfRefCount = max(0, state.SelfRefCount-1)
	}

	state.mu.Unlock()

	// Process signals
	select {
	case sig := <-ol.world.Signals:
		if sig.Type == SignalOverthink {
			ol.processOverthinkSignal(sig)
		}
	default:
	}
}

func (ol *OverthinkingLoops) run() {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ol.stop:
			return
		case <-ticker.C:
			ol.Step(0.2)
		}
	}
}

// AnalyzeText checks text for overthinking patterns
func (ol *OverthinkingLoops) AnalyzeText(text string) OverthinkResult {
	result := OverthinkResult{}

	words := languageTokens(text)

	// 1. Check for repetition spiral
	result.RepetitionScore = ol.checkRepetition(words)

	// Real cross-turn loop signal: how strongly this utterance echoes the previous one
	// (legacy NgramOverlap, computed in Julia). The bigram overlap of consecutive turns
	// catches a voice circling the same matter; it raises the score, never lowers it, and
	// falls back silently to the heuristic above on any Julia fault.
	if ol.lastText != "" && len(words) >= 2 {
		if overlap, err := HighNgramOverlap(ol.lastText, text, 2); err == nil {
			// clamp to [0,1]: the legacy list-length-union overlap can exceed 1 when n-grams
			// repeat, and this score is averaged into TotalScore and exported to C / prophecy debt.
			if ov := clamp(float32(overlap), 0, 1); ov > result.RepetitionScore {
				result.RepetitionScore = ov
			}
		}
	}
	ol.lastText = text

	// Update state
	if ol.world != nil {
		state := ol.world.State
		state.mu.Lock()

		if result.RepetitionScore > 0.5 {
			state.LoopCount++
		}
		result.TotalScore = result.RepetitionScore

		// High overthink reduces coherence
		if result.TotalScore > 0.6 {
			state.Coherence = clamp(state.Coherence-0.1, 0, 1)
			state.Entropy = clamp(state.Entropy+0.1, 0, 1)
		}

		state.mu.Unlock()

		// Emit signal if overthinking detected
		if result.TotalScore > 0.5 {
			ol.world.emit(Signal{
				Type:      SignalOverthink,
				Value:     result.TotalScore,
				Source:    ol.Name(),
				Timestamp: time.Now(),
				Metadata:  map[string]any{"repetition": result.RepetitionScore},
			})
		}
	}

	return result
}

// OverthinkResult holds analysis results
type OverthinkResult struct {
	RepetitionScore float32
	TotalScore      float32
}

func (ol *OverthinkingLoops) checkRepetition(words []string) float32 {
	if len(words) < 5 {
		return 0
	}

	// Measure concentration rather than raw frequency. A token repeated five
	// times across a long utterance is ordinary grammar; a token occupying most
	// of an utterance is a repetition spiral. This needs no language dictionary.
	localCounts := make(map[string]int)
	for _, w := range words {
		localCounts[w]++
	}

	// Check for high frequency concepts
	var maxCount int
	for _, count := range localCounts {
		if count > maxCount {
			maxCount = count
		}
	}

	if maxCount > 1 {
		return float32(maxCount-1) / float32(len(words)-1)
	}
	return 0
}

func (ol *OverthinkingLoops) processOverthinkSignal(sig Signal) {
	state := ol.world.State
	state.mu.Lock()
	defer state.mu.Unlock()

	// External signal increases loop count
	if sig.Value > 0.5 {
		state.LoopCount++
	}
}

// IsSpiraling returns true if currently in an overthink spiral
func (ol *OverthinkingLoops) IsSpiraling() bool {
	if ol.world == nil {
		return false
	}

	state := ol.world.State
	state.mu.RLock()
	defer state.mu.RUnlock()

	return state.LoopCount > ol.repetitionThreshold
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
