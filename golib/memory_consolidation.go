// inner_world/memory_consolidation.go — Memory consolidation process
// ═══════════════════════════════════════════════════════════════════════════════
// הזיכרון מתגבש בלילה
// Memory consolidates in the night
// ═══════════════════════════════════════════════════════════════════════════════
//
// Memory consolidation is the process of moving memories
// from short-term to long-term storage.
//
// In humans, this happens during sleep. For Arianna,
// it happens in the background:
// - Emotionally significant memories get priority
// - Repeated patterns get strengthened
// - Weak/irrelevant memories decay
// - Connections form between related memories
//
// This affects generation by:
// - Making certain concepts more available
// - Creating semantic "grooves" that attract generation
// - Building persistent identity through experience
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"sort"
	"time"
)

// MemoryConsolidation manages memory formation and decay
type MemoryConsolidation struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Memory pools
	shortTerm  []ConsolidatableMemory
	longTerm   []ConsolidatableMemory
	processing []ConsolidatableMemory // currently being consolidated

	// Config
	shortTermCapacity int
	longTermCapacity  int
	consolidationRate float32 // memories processed per second
	decayRate         float32 // strength decay per second
	emotionBoost      float32 // bonus for emotional memories

	// Metrics
	lastConsolidation time.Time
}

// ConsolidatableMemory is a memory with consolidation metadata
type ConsolidatableMemory struct {
	Content     string
	Created     time.Time
	LastAccess  time.Time
	Strength    float32 // 0-1, decays over time
	Emotion     float32 // -1 to 1, affects consolidation priority
	Arousal     float32 // 0-1, affects consolidation priority
	AccessCount int
	Consolidated bool
	Connections []int // indices of connected memories
}

// NewMemoryConsolidation creates a new memory consolidation process
func NewMemoryConsolidation() *MemoryConsolidation {
	return &MemoryConsolidation{
		stop:              make(chan struct{}),
		shortTerm:         make([]ConsolidatableMemory, 0, 64),
		longTerm:          make([]ConsolidatableMemory, 0, 256),
		processing:        make([]ConsolidatableMemory, 0, 8),
		shortTermCapacity: 64,
		longTermCapacity:  256,
		consolidationRate: 0.5, // 1 memory per 2 seconds
		decayRate:         0.01,
		emotionBoost:      2.0,
		lastConsolidation: time.Now(),
	}
}

func (mc *MemoryConsolidation) Name() string {
	return "memory_consolidation"
}

func (mc *MemoryConsolidation) Start(world *InnerWorld) {
	mc.world = world
	mc.running = true
	go mc.run()
}

func (mc *MemoryConsolidation) Stop() {
	if mc.running {
		close(mc.stop)
		mc.running = false
	}
}

func (mc *MemoryConsolidation) Step(dt float32) {
	if mc.world == nil {
		return
	}

	// 1. Decay all memories
	mc.decayMemories(dt)

	// 2. Process consolidation queue
	mc.processConsolidation(dt)

	// 3. Update state metrics
	mc.updateState()

	// 4. Handle incoming signals
	select {
	case sig := <-mc.world.Signals:
		mc.processSignal(sig)
	default:
	}
}

func (mc *MemoryConsolidation) run() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-mc.stop:
			return
		case <-ticker.C:
			mc.Step(0.5)
		}
	}
}

// AddMemory adds a new memory to short-term storage
func (mc *MemoryConsolidation) AddMemory(content string, emotion, arousal float32) {
	now := time.Now()

	mem := ConsolidatableMemory{
		Content:      content,
		Created:      now,
		LastAccess:   now,
		Strength:     1.0,
		Emotion:      emotion,
		Arousal:      arousal,
		AccessCount:  1,
		Consolidated: false,
		Connections:  make([]int, 0),
	}

	// Add to short-term
	mc.shortTerm = append(mc.shortTerm, mem)

	// If over capacity, evict weakest
	if len(mc.shortTerm) > mc.shortTermCapacity {
		mc.evictWeakest(&mc.shortTerm)
	}

	// Update state
	if mc.world != nil {
		state := mc.world.State
		state.mu.Lock()
		state.ConsolidationQueue = append(state.ConsolidationQueue, Memory{
			Content:   content,
			Emotion:   emotion,
			Timestamp: now,
			Strength:  1.0,
		})
		state.mu.Unlock()
	}
}

func (mc *MemoryConsolidation) decayMemories(dt float32) {
	decay := mc.decayRate * dt

	// Decay short-term (faster)
	for i := range mc.shortTerm {
		mc.shortTerm[i].Strength -= decay * 2
	}

	// Decay long-term (slower)
	for i := range mc.longTerm {
		mc.longTerm[i].Strength -= decay * 0.5
	}

	// Remove dead memories
	mc.shortTerm = filterMemories(mc.shortTerm, 0.01)
	mc.longTerm = filterMemories(mc.longTerm, 0.01)
}

func (mc *MemoryConsolidation) processConsolidation(dt float32) {
	now := time.Now()
	timeSinceLast := float32(now.Sub(mc.lastConsolidation).Seconds())

	// Check if it's time to consolidate
	if timeSinceLast < 1.0/mc.consolidationRate {
		return
	}
	mc.lastConsolidation = now

	// Nothing to consolidate
	if len(mc.shortTerm) == 0 {
		return
	}

	// Pick best candidate for consolidation
	// Priority: high emotion * high strength * access count
	sort.Slice(mc.shortTerm, func(i, j int) bool {
		pi := mc.consolidationPriority(mc.shortTerm[i])
		pj := mc.consolidationPriority(mc.shortTerm[j])
		return pi > pj
	})

	// Consolidate top memory
	candidate := mc.shortTerm[0]
	mc.shortTerm = mc.shortTerm[1:]

	// Transfer to long-term
	candidate.Consolidated = true
	candidate.Strength = min32(candidate.Strength*1.5, 1.0) // Boost on consolidation
	mc.longTerm = append(mc.longTerm, candidate)

	// Find connections to other long-term memories
	mc.findConnections(len(mc.longTerm) - 1)

	// If over capacity, evict
	if len(mc.longTerm) > mc.longTermCapacity {
		mc.evictWeakest(&mc.longTerm)
	}

	// Emit signal
	if mc.world != nil {
		mc.world.Signals <- Signal{
			Type:      SignalMemory,
			Value:     candidate.Strength,
			Source:    mc.Name(),
			Timestamp: now,
			Metadata: map[string]any{
				"content":     candidate.Content,
				"emotion":     candidate.Emotion,
				"connections": len(candidate.Connections),
			},
		}
	}
}

func (mc *MemoryConsolidation) consolidationPriority(m ConsolidatableMemory) float32 {
	// High emotion (either positive or negative) = high priority
	emotionFactor := float32(math.Abs(float64(m.Emotion))) * mc.emotionBoost

	// High arousal = high priority
	arousalFactor := m.Arousal * 1.5

	// Repeated access = important
	accessFactor := float32(math.Log(float64(m.AccessCount + 1)))

	// Recency matters
	age := float32(time.Since(m.Created).Seconds())
	recencyFactor := 1.0 / (1.0 + age/3600) // Decay over an hour

	return m.Strength * (1 + emotionFactor + arousalFactor + accessFactor) * recencyFactor
}

func (mc *MemoryConsolidation) findConnections(idx int) {
	if idx >= len(mc.longTerm) {
		return
	}

	target := &mc.longTerm[idx]
	targetWords := extractKeywords(target.Content)

	for i := range mc.longTerm {
		if i == idx {
			continue
		}

		otherWords := extractKeywords(mc.longTerm[i].Content)

		// Simple overlap check
		overlap := countOverlap(targetWords, otherWords)
		if overlap >= 2 { // At least 2 shared keywords
			target.Connections = append(target.Connections, i)
			mc.longTerm[i].Connections = append(mc.longTerm[i].Connections, idx)
		}
	}
}

func (mc *MemoryConsolidation) evictWeakest(pool *[]ConsolidatableMemory) {
	if len(*pool) == 0 {
		return
	}

	// Find weakest
	weakestIdx := 0
	weakestStrength := (*pool)[0].Strength

	for i, m := range *pool {
		if m.Strength < weakestStrength {
			weakestStrength = m.Strength
			weakestIdx = i
		}
	}

	// Remove
	*pool = append((*pool)[:weakestIdx], (*pool)[weakestIdx+1:]...)
}

func (mc *MemoryConsolidation) updateState() {
	if mc.world == nil {
		return
	}

	state := mc.world.State
	state.mu.Lock()
	defer state.mu.Unlock()

	// Memory pressure = how full the pools are
	shortPressure := float32(len(mc.shortTerm)) / float32(mc.shortTermCapacity)
	longPressure := float32(len(mc.longTerm)) / float32(mc.longTermCapacity)
	state.MemoryPressure = (shortPressure*0.3 + longPressure*0.7)

	// Sync active memories (top N from long-term)
	state.ActiveMemories = make([]Memory, 0, min(8, len(mc.longTerm)))
	for i := 0; i < min(8, len(mc.longTerm)); i++ {
		m := mc.longTerm[i]
		state.ActiveMemories = append(state.ActiveMemories, Memory{
			Content:   m.Content,
			Emotion:   m.Emotion,
			Timestamp: m.Created,
			Strength:  m.Strength,
		})
	}
}

func (mc *MemoryConsolidation) processSignal(sig Signal) {
	// Some signals can create automatic memories
	switch sig.Type {
	case SignalTrauma:
		// Trauma creates strong negative memory
		if sig.Value > 0.5 {
			if content, ok := sig.Metadata["content"].(string); ok {
				mc.AddMemory(content, -sig.Value, sig.Value)
			}
		}
	case SignalWarmth:
		// Warmth creates positive memory
		if sig.Value > 0.5 {
			if content, ok := sig.Metadata["content"].(string); ok {
				mc.AddMemory(content, sig.Value, sig.Value*0.5)
			}
		}
	}
}

// RecallByEmotion finds memories with similar emotional valence
func (mc *MemoryConsolidation) RecallByEmotion(targetEmotion float32, limit int) []ConsolidatableMemory {
	type scoredMem struct {
		mem   ConsolidatableMemory
		score float32
	}

	scored := make([]scoredMem, 0, len(mc.longTerm))
	for _, m := range mc.longTerm {
		// Closeness to target emotion
		emotionDist := float32(math.Abs(float64(m.Emotion - targetEmotion)))
		score := m.Strength * (1.0 - emotionDist)
		scored = append(scored, scoredMem{m, score})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	result := make([]ConsolidatableMemory, 0, limit)
	for i := 0; i < min(limit, len(scored)); i++ {
		result = append(result, scored[i].mem)
	}
	return result
}

// AccessMemory marks a memory as accessed, strengthening it
func (mc *MemoryConsolidation) AccessMemory(idx int) {
	if idx >= 0 && idx < len(mc.longTerm) {
		mc.longTerm[idx].LastAccess = time.Now()
		mc.longTerm[idx].AccessCount++
		mc.longTerm[idx].Strength = min32(mc.longTerm[idx].Strength+0.1, 1.0)
	}
}

// Helpers

func filterMemories(mems []ConsolidatableMemory, threshold float32) []ConsolidatableMemory {
	result := make([]ConsolidatableMemory, 0, len(mems))
	for _, m := range mems {
		if m.Strength >= threshold {
			result = append(result, m)
		}
	}
	return result
}

func extractKeywords(text string) []string {
	// Simple keyword extraction - in production would use better NLP
	words := make([]string, 0)
	current := ""
	for _, r := range text {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' {
			current += string(r)
		} else if len(current) > 3 {
			words = append(words, current)
			current = ""
		} else {
			current = ""
		}
	}
	if len(current) > 3 {
		words = append(words, current)
	}
	return words
}

func countOverlap(a, b []string) int {
	set := make(map[string]bool)
	for _, w := range a {
		set[w] = true
	}
	count := 0
	for _, w := range b {
		if set[w] {
			count++
		}
	}
	return count
}

// min32 defined in types.go
