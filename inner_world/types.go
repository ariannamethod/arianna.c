// inner_world/types.go — Shared types for async inner processes
// ═══════════════════════════════════════════════════════════════════════════════
// העולם הפנימי של אריאנה
// The inner world of Arianna
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL TYPES — what flows between processes
// ═══════════════════════════════════════════════════════════════════════════════

// Signal represents any inner signal flowing through the system
type Signal struct {
	Type      SignalType
	Value     float32
	Source    string
	Timestamp time.Time
	Metadata  map[string]any
}

type SignalType int

const (
	SignalTrauma SignalType = iota
	SignalOverthink
	SignalDrift
	SignalMemory
	SignalAttention
	SignalProphecy
	SignalCoherence
	SignalAnxiety
	SignalWarmth
	SignalVoid
)

func (s SignalType) String() string {
	names := []string{
		"trauma", "overthink", "drift", "memory",
		"attention", "prophecy", "coherence", "anxiety",
		"warmth", "void",
	}
	if int(s) < len(names) {
		return names[s]
	}
	return "unknown"
}

// ═══════════════════════════════════════════════════════════════════════════════
// INNER STATE — shared state across processes
// ═══════════════════════════════════════════════════════════════════════════════

// InnerState holds the current inner world state
// This is what C will read/write via cgo
type InnerState struct {
	mu sync.RWMutex

	// Emotional baseline
	Arousal     float32 // 0-1: calm to activated
	Valence     float32 // -1 to 1: negative to positive
	Entropy     float32 // 0-1: how chaotic the inner state
	Coherence   float32 // 0-1: how unified the self feels

	// Trauma system
	TraumaLevel     float32   // 0-1: how triggered
	TraumaAnchors   []string  // active trauma anchors
	LastTraumaTime  time.Time

	// Overthinking
	LoopCount       int     // recursive thought loops detected
	AbstractionDepth int    // how far from concrete
	SelfRefCount    int     // self-referential loops

	// Emotional drift
	DriftDirection  float32 // -1 to 1: past vs future orientation
	DriftSpeed      float32 // how fast drifting
	DriftTarget     string  // what emotion drifting toward

	// Memory
	ConsolidationQueue []Memory
	ActiveMemories     []Memory
	MemoryPressure     float32 // 0-1: how full

	// Attention
	FocusTarget     string  // current focus
	FocusStrength   float32 // 0-1
	WanderPull      float32 // 0-1: how much wanting to wander

	// Prophecy
	ProphecyDebt    float32 // accumulated debt from non-probable choices
	DestinyPull     float32 // 0-1: how strong destiny is
	WormholeChance  float32 // probability of creative skip
}

// Memory represents a consolidatable memory
type Memory struct {
	Content   string
	Emotion   float32
	Timestamp time.Time
	Strength  float32
	Decayed   bool
}

// ═══════════════════════════════════════════════════════════════════════════════
// INNER WORLD — the main orchestrator
// ═══════════════════════════════════════════════════════════════════════════════

// InnerWorld is the main async orchestrator
type InnerWorld struct {
	State    *InnerState

	// Channels for inter-process communication
	Signals  chan Signal
	Commands chan Command
	stopChan chan struct{}

	// Process handles
	processes []Process
	wg        sync.WaitGroup
	running   bool
	mu        sync.Mutex
}

// Command sent to inner world from C
type Command struct {
	Type    CommandType
	Payload any
}

type CommandType int

const (
	CmdPause CommandType = iota
	CmdResume
	CmdReset
	CmdInject  // inject a signal
	CmdQuery   // query state
	CmdStep    // step all processes
)

// Process interface for all async processes
type Process interface {
	Name() string
	Start(world *InnerWorld)
	Stop()
	Step(dt float32) // single step for synchronous mode
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════════

// NewInnerState creates a new inner state with defaults
func NewInnerState() *InnerState {
	return &InnerState{
		Arousal:    0.3,
		Valence:    0.0,
		Entropy:    0.2,
		Coherence:  0.8,

		TraumaLevel:    0.0,
		TraumaAnchors:  make([]string, 0),
		LastTraumaTime: time.Now(),

		LoopCount:        0,
		AbstractionDepth: 0,
		SelfRefCount:     0,

		DriftDirection: 0.0,
		DriftSpeed:     0.1,
		DriftTarget:    "",

		ConsolidationQueue: make([]Memory, 0, 32),
		ActiveMemories:     make([]Memory, 0, 16),
		MemoryPressure:     0.0,

		FocusTarget:   "",
		FocusStrength: 0.5,
		WanderPull:    0.3,

		ProphecyDebt:   0.0,
		DestinyPull:    0.5,
		WormholeChance: 0.02,
	}
}

// NewInnerWorld creates a new inner world
func NewInnerWorld() *InnerWorld {
	return &InnerWorld{
		State:     NewInnerState(),
		Signals:   make(chan Signal, 100),
		Commands:  make(chan Command, 10),
		stopChan:  make(chan struct{}),
		processes: make([]Process, 0),
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE ACCESS (thread-safe)
// ═══════════════════════════════════════════════════════════════════════════════

func (s *InnerState) GetArousal() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Arousal
}

func (s *InnerState) SetArousal(v float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Arousal = clamp(v, 0, 1)
}

func (s *InnerState) GetTraumaLevel() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.TraumaLevel
}

func (s *InnerState) SetTraumaLevel(v float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TraumaLevel = clamp(v, 0, 1)
}

func (s *InnerState) GetCoherence() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Coherence
}

func (s *InnerState) SetCoherence(v float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Coherence = clamp(v, 0, 1)
}

func (s *InnerState) GetProphecyDebt() float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ProphecyDebt
}

func (s *InnerState) AddProphecyDebt(delta float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ProphecyDebt = clamp(s.ProphecyDebt+delta, 0, 10)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
