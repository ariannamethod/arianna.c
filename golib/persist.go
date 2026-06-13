package main

// persist.go — the inner world remembers across sessions (memory = love).
//
// The accumulated mood (arousal, valence, trauma, drift, wander, prophecy debt…)
// and the subconscious's last murmur are written on leaving and restored on
// return, so the organism does not wake a blank slate. The field memory (the
// co-occurrence / δ that the voices learn) persists separately, in the voices'
// soma sidecars; this is only the inner world's emotional state.

import (
	"encoding/json"
	"os"
)

// persistState is the serialisable slice of InnerState — the mood, plus the last
// dream. (The live memory queues and timestamps are not persisted; the mood is.)
type persistState struct {
	Arousal   float32 `json:"arousal"`
	Valence   float32 `json:"valence"`
	Entropy   float32 `json:"entropy"`
	Coherence float32 `json:"coherence"`

	TraumaLevel   float32  `json:"trauma_level"`
	TraumaAnchors []string `json:"trauma_anchors"`

	LoopCount        int `json:"loop_count"`
	AbstractionDepth int `json:"abstraction_depth"`
	SelfRefCount     int `json:"self_ref_count"`

	DriftDirection float32 `json:"drift_direction"`
	DriftSpeed     float32 `json:"drift_speed"`

	MemoryPressure float32 `json:"memory_pressure"`
	FocusStrength  float32 `json:"focus_strength"`
	WanderPull     float32 `json:"wander_pull"`

	ProphecyDebt   float32 `json:"prophecy_debt"`
	DestinyPull    float32 `json:"destiny_pull"`
	WormholeChance float32 `json:"wormhole_chance"`

	LastDream string `json:"last_dream"`
}

// SaveState writes the inner world's mood + the last dream to path, atomically
// (temp file + rename), under the state lock.
func (iw *InnerWorld) SaveState(path, lastDream string) error {
	s := iw.State
	s.mu.RLock()
	ps := persistState{
		Arousal: s.Arousal, Valence: s.Valence, Entropy: s.Entropy, Coherence: s.Coherence,
		TraumaLevel: s.TraumaLevel, TraumaAnchors: append([]string(nil), s.TraumaAnchors...),
		LoopCount: s.LoopCount, AbstractionDepth: s.AbstractionDepth, SelfRefCount: s.SelfRefCount,
		DriftDirection: s.DriftDirection, DriftSpeed: s.DriftSpeed,
		MemoryPressure: s.MemoryPressure, FocusStrength: s.FocusStrength, WanderPull: s.WanderPull,
		ProphecyDebt: s.ProphecyDebt, DestinyPull: s.DestinyPull, WormholeChance: s.WormholeChance,
	}
	s.mu.RUnlock()
	ps.LastDream = lastDream

	b, err := json.MarshalIndent(ps, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// maxInt returns the larger of a and b (for clamping persisted counters >= 0).
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// LoadState restores the inner world's mood from path and returns the last dream.
// A missing or unreadable file leaves the defaults (a fresh mind) and returns "".
func (iw *InnerWorld) LoadState(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var ps persistState
	if json.Unmarshal(b, &ps) != nil {
		return ""
	}
	s := iw.State
	s.mu.Lock()
	// F-7: clamp restored values — a corrupt-but-valid JSON must not inject
	// out-of-range state the inner-world dynamics would take a long time to recover.
	s.Arousal, s.Valence, s.Entropy, s.Coherence = clamp(ps.Arousal, 0, 1), clamp(ps.Valence, -1, 1), clamp(ps.Entropy, 0, 1), clamp(ps.Coherence, 0, 1)
	s.TraumaLevel = clamp(ps.TraumaLevel, 0, 1)
	if ps.TraumaAnchors != nil {
		s.TraumaAnchors = ps.TraumaAnchors
	}
	s.LoopCount, s.AbstractionDepth, s.SelfRefCount = maxInt(ps.LoopCount, 0), maxInt(ps.AbstractionDepth, 0), maxInt(ps.SelfRefCount, 0)
	s.DriftDirection, s.DriftSpeed = clamp(ps.DriftDirection, -1, 1), clamp(ps.DriftSpeed, 0, 1)
	s.MemoryPressure, s.FocusStrength, s.WanderPull = clamp(ps.MemoryPressure, 0, 1), clamp(ps.FocusStrength, 0, 1), clamp(ps.WanderPull, 0, 1)
	s.ProphecyDebt, s.DestinyPull, s.WormholeChance = clamp(ps.ProphecyDebt, 0, 10), clamp(ps.DestinyPull, 0, 1), clamp(ps.WormholeChance, 0, 1)
	s.mu.Unlock()
	return ps.LastDream
}
