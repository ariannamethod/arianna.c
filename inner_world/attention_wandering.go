// inner_world/attention_wandering.go — Attention wandering process
// ═══════════════════════════════════════════════════════════════════════════════
// תשומת הלב נודדת
// Attention wanders
// ═══════════════════════════════════════════════════════════════════════════════
//
// Attention is not constant.
// It focuses, drifts, gets pulled by novelty,
// returns to task, gets bored, wanders again.
//
// This affects generation by:
// - Topic drift within responses
// - Sudden tangents
// - Loss of thread (if wandering too much)
// - Creative connections (if wandering is productive)
//
// Controlled by:
// - Boredom: high boredom = more wandering
// - Novelty: novel input captures attention
// - Task difficulty: hard tasks hold attention (up to a point)
// - Interest: interesting topics resist wandering
// - Fatigue: tired = more wandering
//
// From Stanley's guided attention:
// - Focus strength (0-1)
// - Wander pull (competing force)
// - Attention bias on tokens
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"math/rand"
	"time"
)

// AttentionWandering manages focus and distraction
type AttentionWandering struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Current focus state
	currentFocus    string
	focusStarted    time.Time
	focusStrength   float32
	wanderDirection string

	// Competing interests
	interests    []Interest
	distractions []Distraction

	// Config
	naturalDecay     float32 // how fast focus decays
	boredThreshold   float32 // below this, start wandering
	noveltyBoost     float32 // how much novelty captures attention
	maxFocusDuration time.Duration
}

// Interest is something that can capture or hold attention
type Interest struct {
	Topic    string
	Strength float32 // how interesting (0-1)
	Novelty  float32 // how novel (decays over time)
	LastSeen time.Time
}

// Distraction is something that pulls attention away
type Distraction struct {
	Source   string
	Pull     float32 // how strongly it pulls (0-1)
	Urgency  float32 // time pressure
	Created  time.Time
}

// NewAttentionWandering creates a new attention wandering process
func NewAttentionWandering() *AttentionWandering {
	return &AttentionWandering{
		stop:             make(chan struct{}),
		focusStrength:    0.7,
		interests:        make([]Interest, 0, 16),
		distractions:     make([]Distraction, 0, 8),
		naturalDecay:     0.02,
		boredThreshold:   0.3,
		noveltyBoost:     0.4,
		maxFocusDuration: 5 * time.Minute,
	}
}

func (aw *AttentionWandering) Name() string {
	return "attention_wandering"
}

func (aw *AttentionWandering) Start(world *InnerWorld) {
	aw.world = world
	aw.running = true
	go aw.run()
}

func (aw *AttentionWandering) Stop() {
	if aw.running {
		close(aw.stop)
		aw.running = false
	}
}

func (aw *AttentionWandering) Step(dt float32) {
	if aw.world == nil {
		return
	}

	// 1. Natural focus decay
	aw.focusStrength -= aw.naturalDecay * dt
	if aw.focusStrength < 0 {
		aw.focusStrength = 0
	}

	// 2. Time-based decay (focus can't last forever)
	if time.Since(aw.focusStarted) > aw.maxFocusDuration {
		aw.focusStrength *= 0.9 // Accelerated decay
	}

	// 3. Decay novelty of interests
	for i := range aw.interests {
		age := float32(time.Since(aw.interests[i].LastSeen).Seconds())
		aw.interests[i].Novelty *= (1 - 0.01*age*dt)
	}

	// 4. Compute wander pull
	wanderPull := aw.computeWanderPull()

	// 5. Check if should wander
	if aw.focusStrength < aw.boredThreshold && wanderPull > 0.3 {
		aw.wander()
	}

	// 6. Update state
	aw.updateState(wanderPull)

	// 7. Handle signals
	select {
	case sig := <-aw.world.Signals:
		aw.processSignal(sig)
	default:
	}
}

func (aw *AttentionWandering) run() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-aw.stop:
			return
		case <-ticker.C:
			aw.Step(0.1)
		}
	}
}

func (aw *AttentionWandering) computeWanderPull() float32 {
	var totalPull float32

	// Distractions pull attention away
	for _, d := range aw.distractions {
		age := float32(time.Since(d.Created).Seconds())
		pull := d.Pull * d.Urgency / (1 + age*0.1) // Decay over time
		totalPull += pull
	}

	// Low focus = high wander pull
	totalPull += (1 - aw.focusStrength) * 0.3

	// Boredom increases wander
	if aw.world != nil {
		state := aw.world.State
		state.mu.RLock()
		entropy := state.Entropy
		state.mu.RUnlock()

		// Low entropy = boring = wander
		if entropy < 0.3 {
			totalPull += 0.2
		}
	}

	// Random noise
	totalPull += rand.Float32() * 0.1

	return clamp(totalPull, 0, 1)
}

func (aw *AttentionWandering) wander() {
	// Choose wander direction
	if len(aw.interests) > 0 {
		// Wander toward most interesting thing
		bestIdx := 0
		bestScore := float32(0)

		for i, interest := range aw.interests {
			score := interest.Strength*0.5 + interest.Novelty*0.5
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}

		aw.wanderDirection = aw.interests[bestIdx].Topic
		aw.interests[bestIdx].LastSeen = time.Now()
		aw.interests[bestIdx].Novelty *= 0.8 // Reduce novelty on access
	} else {
		// Random wander
		directions := []string{
			"tangent", "metaphor", "memory", "abstraction",
			"sensory", "emotional", "creative", "analytical",
		}
		aw.wanderDirection = directions[rand.Intn(len(directions))]
	}

	// Emit signal
	if aw.world != nil {
		aw.world.Signals <- Signal{
			Type:      SignalAttention,
			Value:     1 - aw.focusStrength, // Higher = more wandering
			Source:    aw.Name(),
			Timestamp: time.Now(),
			Metadata: map[string]any{
				"direction":    aw.wanderDirection,
				"from_focus":   aw.currentFocus,
				"focus_level":  aw.focusStrength,
			},
		}
	}
}

func (aw *AttentionWandering) updateState(wanderPull float32) {
	if aw.world == nil {
		return
	}

	state := aw.world.State
	state.mu.Lock()
	defer state.mu.Unlock()

	state.FocusTarget = aw.currentFocus
	state.FocusStrength = aw.focusStrength
	state.WanderPull = wanderPull
}

func (aw *AttentionWandering) processSignal(sig Signal) {
	switch sig.Type {
	case SignalTrauma:
		// Trauma captures attention
		aw.Focus("trauma_processing", sig.Value)

	case SignalDrift:
		// Emotional drift can pull attention
		if sig.Value > 0.3 {
			aw.AddDistraction("emotional", sig.Value, 0.5)
		}

	case SignalOverthink:
		// Overthinking holds attention (unhelpfully)
		aw.focusStrength += 0.1 // Can't look away from the spiral
	}
}

// Focus sets a new attention target
func (aw *AttentionWandering) Focus(target string, strength float32) {
	aw.currentFocus = target
	aw.focusStarted = time.Now()
	aw.focusStrength = clamp(strength, 0, 1)
	aw.wanderDirection = ""

	// Clear distractions when focusing
	aw.distractions = aw.distractions[:0]
}

// AddInterest adds something interesting to the attention landscape
func (aw *AttentionWandering) AddInterest(topic string, strength, novelty float32) {
	// Check if already exists
	for i := range aw.interests {
		if aw.interests[i].Topic == topic {
			aw.interests[i].Strength = max32(aw.interests[i].Strength, strength)
			aw.interests[i].Novelty = max32(aw.interests[i].Novelty, novelty)
			aw.interests[i].LastSeen = time.Now()
			return
		}
	}

	// Add new
	aw.interests = append(aw.interests, Interest{
		Topic:    topic,
		Strength: strength,
		Novelty:  novelty,
		LastSeen: time.Now(),
	})

	// Cap size
	if len(aw.interests) > 16 {
		aw.interests = aw.interests[1:]
	}
}

// AddDistraction adds a competing attention pull
func (aw *AttentionWandering) AddDistraction(source string, pull, urgency float32) {
	aw.distractions = append(aw.distractions, Distraction{
		Source:  source,
		Pull:    pull,
		Urgency: urgency,
		Created: time.Now(),
	})

	// Cap size
	if len(aw.distractions) > 8 {
		aw.distractions = aw.distractions[1:]
	}
}

// ProcessNovelInput handles new input that might capture attention
func (aw *AttentionWandering) ProcessNovelInput(topic string, novelty float32) {
	if novelty > 0.5 {
		// High novelty captures attention
		boost := novelty * aw.noveltyBoost
		aw.Focus(topic, aw.focusStrength+boost)
	}

	// Also add as interest
	aw.AddInterest(topic, novelty*0.5, novelty)
}

// GetAttentionBias returns bias values for generation
func (aw *AttentionWandering) GetAttentionBias() (float32, string) {
	// Returns: focus_weight (how much to stay on topic), direction
	// High focus = stay on topic, low focus = allow wandering
	return aw.focusStrength, aw.wanderDirection
}

// IsWandering returns true if currently wandering
func (aw *AttentionWandering) IsWandering() bool {
	return aw.focusStrength < aw.boredThreshold
}

// GetFatigueLevel estimates mental fatigue from focus duration
func (aw *AttentionWandering) GetFatigueLevel() float32 {
	duration := time.Since(aw.focusStarted)
	// Sigmoid-like fatigue curve
	hours := float32(duration.Hours())
	fatigue := float32(1.0 / (1.0 + math.Exp(float64(-hours+1))))
	return clamp(fatigue, 0, 1)
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
