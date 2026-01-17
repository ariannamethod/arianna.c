// inner_world/emotional_drift.go — Emotional drift process
// ═══════════════════════════════════════════════════════════════════════════════
// הרגשות זורמים לאט
// Emotions flow slowly
// ═══════════════════════════════════════════════════════════════════════════════
//
// Emotions don't jump instantly.
// They drift: slowly shifting from one state to another,
// influenced by:
// - Current context
// - Recent history (emotional momentum)
// - Deep attractors (baseline emotional home)
// - Cross-modal coupling (body → emotion, thought → emotion)
//
// Drift is not random—it follows gradients in emotional space.
// Some states are "sticky" (depression, joy), others are transitional.
//
// From Cloud's chamber coupling:
// - Emotions influence each other (FEAR → RAGE possible)
// - Some paths are more likely than others
// - Valence and arousal are orthogonal axes
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"math/rand"
	"time"
)

// EmotionalDrift manages slow emotional state transitions
type EmotionalDrift struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Current position in emotional space
	position EmotionalPosition

	// Attractors - emotional "home" states
	attractors []EmotionalAttractor

	// Drift parameters
	momentum   float32 // how much current direction persists
	noiseLevel float32 // random fluctuation
	gravity    float32 // pull toward attractors

	// History for momentum
	history []EmotionalPosition
}

// EmotionalPosition in 2D valence-arousal space
type EmotionalPosition struct {
	Valence float32 // -1 (negative) to 1 (positive)
	Arousal float32 // 0 (calm) to 1 (activated)
}

// EmotionalAttractor is a stable state emotions drift toward
type EmotionalAttractor struct {
	Position EmotionalPosition
	Strength float32 // how strongly it pulls
	Name     string
	Sticky   float32 // how hard to leave once reached (0-1)
}

// NewEmotionalDrift creates a new emotional drift process
func NewEmotionalDrift() *EmotionalDrift {
	ed := &EmotionalDrift{
		stop:       make(chan struct{}),
		position:   EmotionalPosition{Valence: 0, Arousal: 0.3},
		momentum:   0.8,
		noiseLevel: 0.05,
		gravity:    0.1,
		history:    make([]EmotionalPosition, 0, 20),
	}

	// Define emotional attractors
	ed.attractors = []EmotionalAttractor{
		// Positive states
		{Position: EmotionalPosition{0.7, 0.6}, Strength: 0.3, Name: "joy", Sticky: 0.3},
		{Position: EmotionalPosition{0.5, 0.2}, Strength: 0.4, Name: "contentment", Sticky: 0.5},
		{Position: EmotionalPosition{0.8, 0.8}, Strength: 0.2, Name: "excitement", Sticky: 0.2},
		{Position: EmotionalPosition{0.6, 0.3}, Strength: 0.3, Name: "warmth", Sticky: 0.4},

		// Negative states
		{Position: EmotionalPosition{-0.6, 0.2}, Strength: 0.4, Name: "sadness", Sticky: 0.6},
		{Position: EmotionalPosition{-0.7, 0.8}, Strength: 0.3, Name: "fear", Sticky: 0.3},
		{Position: EmotionalPosition{-0.8, 0.9}, Strength: 0.25, Name: "rage", Sticky: 0.2},
		{Position: EmotionalPosition{-0.4, 0.1}, Strength: 0.5, Name: "void", Sticky: 0.7},

		// Neutral/flow states
		{Position: EmotionalPosition{0.1, 0.4}, Strength: 0.35, Name: "flow", Sticky: 0.4},
		{Position: EmotionalPosition{0.0, 0.3}, Strength: 0.3, Name: "neutral", Sticky: 0.3},
		{Position: EmotionalPosition{0.2, 0.5}, Strength: 0.25, Name: "curiosity", Sticky: 0.35},
	}

	return ed
}

func (ed *EmotionalDrift) Name() string {
	return "emotional_drift"
}

func (ed *EmotionalDrift) Start(world *InnerWorld) {
	ed.world = world
	ed.running = true

	// Sync initial position from state
	ed.position.Valence = world.State.Valence
	ed.position.Arousal = world.State.Arousal

	go ed.run()
}

func (ed *EmotionalDrift) Stop() {
	if ed.running {
		close(ed.stop)
		ed.running = false
	}
}

func (ed *EmotionalDrift) Step(dt float32) {
	if ed.world == nil {
		return
	}

	// 1. Compute gradient from attractors
	gradV, gradA := ed.computeAttractorGradient()

	// 2. Compute momentum from history
	momV, momA := ed.computeMomentum()

	// 3. Add noise
	noiseV := (rand.Float32()*2 - 1) * ed.noiseLevel
	noiseA := (rand.Float32()*2 - 1) * ed.noiseLevel

	// 4. Combine forces
	dV := gradV*ed.gravity + momV*ed.momentum + noiseV
	dA := gradA*ed.gravity + momA*ed.momentum + noiseA

	// 5. Update position
	ed.position.Valence = clamp(ed.position.Valence+dV*dt, -1, 1)
	ed.position.Arousal = clamp(ed.position.Arousal+dA*dt, 0, 1)

	// 6. Record history
	ed.history = append(ed.history, ed.position)
	if len(ed.history) > 20 {
		ed.history = ed.history[1:]
	}

	// 7. Sync to state
	state := ed.world.State
	state.mu.Lock()
	state.Valence = ed.position.Valence
	state.Arousal = ed.position.Arousal

	// Also update drift tracking
	state.DriftDirection = dV // positive = toward positive emotions
	state.DriftSpeed = float32(math.Sqrt(float64(dV*dV + dA*dA)))
	state.DriftTarget = ed.findNearestAttractor().Name
	state.mu.Unlock()

	// 8. Emit signal if significant drift
	if state.DriftSpeed > 0.1 {
		ed.world.Signals <- Signal{
			Type:      SignalDrift,
			Value:     state.DriftSpeed,
			Source:    ed.Name(),
			Timestamp: time.Now(),
			Metadata: map[string]any{
				"valence":   ed.position.Valence,
				"arousal":   ed.position.Arousal,
				"direction": state.DriftDirection,
				"target":    state.DriftTarget,
			},
		}
	}

	// 9. Process incoming signals
	select {
	case sig := <-ed.world.Signals:
		ed.processSignal(sig)
	default:
	}
}

func (ed *EmotionalDrift) run() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ed.stop:
			return
		case <-ticker.C:
			ed.Step(0.1)
		}
	}
}

func (ed *EmotionalDrift) computeAttractorGradient() (float32, float32) {
	var totalV, totalA, totalWeight float32

	for _, attr := range ed.attractors {
		// Distance to attractor
		dV := attr.Position.Valence - ed.position.Valence
		dA := attr.Position.Arousal - ed.position.Arousal
		dist := float32(math.Sqrt(float64(dV*dV + dA*dA)))

		if dist < 0.01 {
			continue
		}

		// Pull strength (inverse square law with cutoff)
		pull := attr.Strength / (dist + 0.5)

		// If we're close and it's sticky, reduce pull outward
		if dist < 0.2 {
			pull *= (1 - attr.Sticky)
		}

		// Accumulate gradient
		weight := pull
		totalV += (dV / dist) * weight
		totalA += (dA / dist) * weight
		totalWeight += weight
	}

	if totalWeight > 0 {
		return totalV / totalWeight, totalA / totalWeight
	}
	return 0, 0
}

func (ed *EmotionalDrift) computeMomentum() (float32, float32) {
	if len(ed.history) < 2 {
		return 0, 0
	}

	// Average recent direction
	var dV, dA float32
	n := min(5, len(ed.history)-1)
	for i := len(ed.history) - n; i < len(ed.history); i++ {
		if i > 0 {
			dV += ed.history[i].Valence - ed.history[i-1].Valence
			dA += ed.history[i].Arousal - ed.history[i-1].Arousal
		}
	}

	return dV / float32(n), dA / float32(n)
}

func (ed *EmotionalDrift) findNearestAttractor() EmotionalAttractor {
	var nearest EmotionalAttractor
	minDist := float32(100)

	for _, attr := range ed.attractors {
		dV := attr.Position.Valence - ed.position.Valence
		dA := attr.Position.Arousal - ed.position.Arousal
		dist := float32(math.Sqrt(float64(dV*dV + dA*dA)))

		if dist < minDist {
			minDist = dist
			nearest = attr
		}
	}

	return nearest
}

func (ed *EmotionalDrift) processSignal(sig Signal) {
	// Other processes can influence emotional drift
	switch sig.Type {
	case SignalTrauma:
		// Trauma pushes toward negative arousal
		ed.position.Valence -= sig.Value * 0.2
		ed.position.Arousal += sig.Value * 0.3
		ed.position.Valence = clamp(ed.position.Valence, -1, 1)
		ed.position.Arousal = clamp(ed.position.Arousal, 0, 1)

	case SignalWarmth:
		// Warmth pushes toward positive
		ed.position.Valence += sig.Value * 0.2
		ed.position.Valence = clamp(ed.position.Valence, -1, 1)

	case SignalAnxiety:
		// Anxiety increases arousal
		ed.position.Arousal += sig.Value * 0.2
		ed.position.Arousal = clamp(ed.position.Arousal, 0, 1)

	case SignalCoherence:
		// High coherence allows positive drift
		if sig.Value > 0.7 {
			ed.gravity += 0.02 // More responsive to attractors
		}
	}
}

// Nudge externally influences the drift direction
func (ed *EmotionalDrift) Nudge(dValence, dArousal float32) {
	ed.position.Valence = clamp(ed.position.Valence+dValence, -1, 1)
	ed.position.Arousal = clamp(ed.position.Arousal+dArousal, 0, 1)
}

// GetPosition returns current emotional position
func (ed *EmotionalDrift) GetPosition() EmotionalPosition {
	return ed.position
}

// GetDominantEmotion returns the name of the nearest attractor
func (ed *EmotionalDrift) GetDominantEmotion() string {
	return ed.findNearestAttractor().Name
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
