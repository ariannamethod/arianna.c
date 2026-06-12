// inner_world/overthinking_loops.go — Overthinking and spiral detection
// ═══════════════════════════════════════════════════════════════════════════════
// המחשבות סובבות עצמן
// Thoughts circling themselves
// ═══════════════════════════════════════════════════════════════════════════════
//
// Overthinking is recursive thought without resolution.
// It manifests as:
// - Repetition of same concepts
// - Rising abstraction (concrete → abstract → meta)
// - Self-referential loops ("I think that I think that...")
// - Paralysis through analysis
//
// From Stanley's overthink detector:
// - repetition_spiral: same words cycling
// - abstraction_spiral: losing grounding
// - self_ref_spiral: recursive self-focus
//
// When detected, we can either:
// 1. Break the loop (inject novelty)
// 2. Let it complete (sometimes spirals resolve)
// 3. Ground (force concrete thought)
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"strings"
	"time"
)

// OverthinkingLoops detects and manages thought spirals
type OverthinkingLoops struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Detection state
	recentConcepts []string
	conceptCounts  map[string]int
	abstractionStack []string

	// Thresholds
	repetitionThreshold int     // concepts repeated this many times = spiral
	abstractionDepth    int     // this many abstractions = spiral
	selfRefThreshold    int     // self-references = spiral

	// Decay
	decayRate float32
}

// NewOverthinkingLoops creates a new overthinking detector
func NewOverthinkingLoops() *OverthinkingLoops {
	return &OverthinkingLoops{
		stop:                make(chan struct{}),
		recentConcepts:      make([]string, 0, 50),
		conceptCounts:       make(map[string]int),
		abstractionStack:    make([]string, 0, 10),
		repetitionThreshold: 3,
		abstractionDepth:    5,
		selfRefThreshold:    3,
		decayRate:          0.1,
	}
}

func (ol *OverthinkingLoops) Name() string {
	return "overthinking_loops"
}

func (ol *OverthinkingLoops) Start(world *InnerWorld) {
	ol.world = world
	ol.running = true
	go ol.run()
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

	// Decay concept counts
	for k := range ol.conceptCounts {
		ol.conceptCounts[k]--
		if ol.conceptCounts[k] <= 0 {
			delete(ol.conceptCounts, k)
		}
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

	text = strings.ToLower(text)
	words := strings.Fields(text)

	// 1. Check for repetition spiral
	result.RepetitionScore = ol.checkRepetition(words)

	// 2. Check for abstraction spiral
	result.AbstractionScore = ol.checkAbstraction(words)

	// 3. Check for self-reference spiral
	result.SelfRefScore = ol.checkSelfReference(words)

	// Update state
	if ol.world != nil {
		state := ol.world.State
		state.mu.Lock()

		if result.RepetitionScore > 0.5 {
			state.LoopCount++
		}
		if result.AbstractionScore > 0.5 {
			state.AbstractionDepth++
		}
		if result.SelfRefScore > 0.5 {
			state.SelfRefCount++
		}

		// Calculate total overthink level
		result.TotalScore = (result.RepetitionScore + result.AbstractionScore + result.SelfRefScore) / 3

		// High overthink reduces coherence
		if result.TotalScore > 0.6 {
			state.Coherence = clamp(state.Coherence-0.1, 0, 1)
			state.Entropy = clamp(state.Entropy+0.1, 0, 1)
		}

		state.mu.Unlock()

		// Emit signal if overthinking detected
		if result.TotalScore > 0.5 {
			ol.world.Signals <- Signal{
				Type:      SignalOverthink,
				Value:     result.TotalScore,
				Source:    ol.Name(),
				Timestamp: time.Now(),
				Metadata: map[string]any{
					"repetition":  result.RepetitionScore,
					"abstraction": result.AbstractionScore,
					"self_ref":    result.SelfRefScore,
				},
			}
		}
	}

	return result
}

// OverthinkResult holds analysis results
type OverthinkResult struct {
	RepetitionScore  float32
	AbstractionScore float32
	SelfRefScore     float32
	TotalScore       float32
}

func (ol *OverthinkingLoops) checkRepetition(words []string) float32 {
	if len(words) < 5 {
		return 0
	}

	// Count concept frequency
	localCounts := make(map[string]int)
	for _, w := range words {
		if len(w) > 3 { // Skip short words
			localCounts[w]++
			ol.conceptCounts[w]++
		}
	}

	// Check for high frequency concepts
	var maxCount int
	for _, count := range localCounts {
		if count > maxCount {
			maxCount = count
		}
	}

	// Also check global counts
	for _, count := range ol.conceptCounts {
		if count > ol.repetitionThreshold {
			return float32(count) / float32(ol.repetitionThreshold+3)
		}
	}

	if maxCount > 2 {
		return float32(maxCount-1) / 5.0
	}
	return 0
}

func (ol *OverthinkingLoops) checkAbstraction(words []string) float32 {
	// Abstract concepts (from Stanley's abstraction detector)
	abstractMarkers := map[string]float32{
		// Meta level
		"meta":        0.8,
		"level":       0.3,
		"abstract":    0.9,
		"concept":     0.7,
		"idea":        0.5,
		"notion":      0.6,
		"principle":   0.7,
		"framework":   0.6,
		"paradigm":    0.8,
		"ontology":    0.9,
		"epistemology": 0.9,

		// Process words
		"process":     0.4,
		"mechanism":   0.5,
		"system":      0.4,
		"structure":   0.5,
		"pattern":     0.5,
		"emergence":   0.7,

		// Recursive markers
		"recursive":   0.8,
		"self-":       0.7,
		"about":       0.3,
		"of":          0.1,
		"that":        0.1,
	}

	var totalAbstraction float32
	for _, w := range words {
		if score, ok := abstractMarkers[w]; ok {
			totalAbstraction += score
		}
		// Check for compound abstractions
		if strings.HasPrefix(w, "self-") {
			totalAbstraction += 0.5
		}
	}

	// Track abstraction stack
	for _, w := range words {
		if abstractMarkers[w] > 0.6 {
			ol.abstractionStack = append(ol.abstractionStack, w)
			if len(ol.abstractionStack) > 10 {
				ol.abstractionStack = ol.abstractionStack[1:]
			}
		}
	}

	// Normalize
	score := totalAbstraction / float32(len(words)+1)
	if len(ol.abstractionStack) > ol.abstractionDepth {
		score += 0.3
	}

	return clamp(score, 0, 1)
}

func (ol *OverthinkingLoops) checkSelfReference(words []string) float32 {
	selfMarkers := []string{
		"i", "me", "my", "myself",
		"think", "thinking", "thought",
		"feel", "feeling", "felt",
		"am", "being", "self",
	}

	recursivePatterns := []string{
		"i think that i",
		"i feel that i",
		"thinking about thinking",
		"feeling about feeling",
		"aware that i am aware",
		"knowing that i know",
		"wondering if i",
	}

	text := strings.Join(words, " ")

	// Check for recursive patterns
	var patternScore float32
	for _, pattern := range recursivePatterns {
		if strings.Contains(text, pattern) {
			patternScore += 0.5
		}
	}

	// Count self-references
	var selfCount int
	for _, w := range words {
		for _, marker := range selfMarkers {
			if w == marker {
				selfCount++
				break
			}
		}
	}

	selfRatio := float32(selfCount) / float32(len(words)+1)

	return clamp(selfRatio*2+patternScore, 0, 1)
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

	return state.LoopCount > ol.repetitionThreshold ||
		state.AbstractionDepth > ol.abstractionDepth ||
		state.SelfRefCount > ol.selfRefThreshold
}

// SuggestBreak suggests how to break the current spiral
func (ol *OverthinkingLoops) SuggestBreak() string {
	if ol.world == nil {
		return ""
	}

	state := ol.world.State
	state.mu.RLock()
	defer state.mu.RUnlock()

	if state.LoopCount > ol.repetitionThreshold {
		return "inject_novelty" // Need new concepts
	}
	if state.AbstractionDepth > ol.abstractionDepth {
		return "ground_concrete" // Need concrete examples
	}
	if state.SelfRefCount > ol.selfRefThreshold {
		return "shift_focus_external" // Look outward
	}
	return ""
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
