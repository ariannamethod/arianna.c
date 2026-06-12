// inner_world/meta_router.go — MetaArianna Router
// ═══════════════════════════════════════════════════════════════════════════════
// הנשימה — שאיפה → צפייה → נשיפה
// Inhale → observe → exhale. Breathing.
// ═══════════════════════════════════════════════════════════════════════════════
//
// The MetaRouter is permanent. It reads InnerWorld metrics and decides
// when to spawn a FluidTransformer observer. One template at a time.
// Priority: Drift > Silence > Thermograph > Field.
//
// CGO exports:
//   meta_router_init()              — initialize router
//   meta_router_tick() int          — check triggers, return template_id or -1
//   meta_router_get_params(out)     — fill MetaTemplateParams for C
//   meta_router_feed_thermogram(in) — receive observation result
//   meta_router_get_observation_count() int
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

/*
#include <stdlib.h>

// MetaTemplateParams — must match src/meta_arianna.h
typedef struct {
    int   template_type;
    float attention_biases[8];
    float layer_focus[8];
    float temperature;
    int   delta_target;
} MetaTemplateParams;

// MetaThermogram — must match src/meta_arianna.h
typedef struct {
    float warmth;
    float sharpness;
    float silence;
    float uncertainty;
    float drift_rate;
    int   drift_direction;
    float field_vector[8];
    int   valid;
    int   template_used;
} MetaThermogram;
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const (
	metaTemplateThermograph = 0
	metaTemplateSilence     = 1
	metaTemplateDrift       = 2
	metaTemplateField       = 3
	metaNumTemplates        = 4
)

// Cooldown per template (one observation at a time, breathe between cycles)
var metaCooldowns = [metaNumTemplates]time.Duration{
	2 * time.Second, // Thermograph — steady, frequent
	3 * time.Second, // Silence — needs stillness to detect
	2 * time.Second, // Drift — tracks movement, responsive
	5 * time.Second, // Field — integral, needs accumulation
}

// ═══════════════════════════════════════════════════════════════════════════════
// META ROUTER
// ═══════════════════════════════════════════════════════════════════════════════

// MetaRouter watches InnerWorld metrics and triggers FluidTransformer templates.
// Lives as a goroutine-safe singleton alongside InnerWorld.
type MetaRouter struct {
	mu sync.Mutex

	initialized      bool
	lastTrigger      [metaNumTemplates]time.Time
	lastParams       metaRouterParams
	lastThermogram   metaRouterThermo
	observationCount int
}

// Internal Go representations (avoid CGO types in business logic)
type metaRouterParams struct {
	TemplateType    int
	AttentionBiases [8]float32
	LayerFocus      [8]float32
	Temperature     float32
	DeltaTarget     int
}

type metaRouterThermo struct {
	Warmth         float32
	Sharpness      float32
	Silence        float32
	Uncertainty    float32
	DriftRate      float32
	DriftDirection int
	FieldVector    [8]float32
	Valid          bool
	TemplateUsed   int
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL SINGLETON
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalMetaRouter *MetaRouter
	metaRouterMu     sync.Mutex
)

func getMetaRouter() *MetaRouter {
	metaRouterMu.Lock()
	defer metaRouterMu.Unlock()

	if globalMetaRouter == nil {
		globalMetaRouter = &MetaRouter{
			initialized: true,
		}
		fmt.Fprintf(os.Stderr, "[meta_router] initialized\n")
	}
	return globalMetaRouter
}

// ═══════════════════════════════════════════════════════════════════════════════
// TICK — Check metrics, return template_id or -1
// ═══════════════════════════════════════════════════════════════════════════════

func (r *MetaRouter) tick() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.initialized {
		return -1
	}

	now := time.Now()

	// Get thread-safe snapshot of InnerWorld state
	snap := GetSnapshotGlobal()

	// Trigger conditions checked in priority order:
	// Drift > Silence > Thermograph > Field
	type trigger struct {
		id    int
		check func() bool
	}

	triggers := []trigger{
		{metaTemplateDrift, func() bool {
			// Fast drift or strong directional movement
			return snap.DriftSpeed > 0.3 ||
				math.Abs(float64(snap.DriftDirection)) > 0.5
		}},
		{metaTemplateSilence, func() bool {
			// High wandering (attention fragmented) or high entropy
			return snap.WanderPull > 0.8 || snap.Entropy > 0.7
		}},
		{metaTemplateThermograph, func() bool {
			// Temperature difference from baseline or high entropy
			warmthDiff := math.Abs(float64(snap.Arousal - 0.5))
			return warmthDiff > 0.4 || snap.Entropy > 0.6
		}},
		{metaTemplateField, func() bool {
			// Multiple signals converging — integral view needed
			return snap.FocusStrength > 0.6 &&
				snap.DriftSpeed > 0.2 &&
				snap.Coherence > 0.5
		}},
	}

	for _, t := range triggers {
		if !t.check() {
			continue
		}
		// Check cooldown
		if now.Sub(r.lastTrigger[t.id]) < metaCooldowns[t.id] {
			continue
		}

		// Triggered — fill params from current metrics
		r.fillParams(t.id, &snap)
		r.lastTrigger[t.id] = now
		r.observationCount++

		templateNames := [metaNumTemplates]string{
			"THERMO", "SILENCE", "DRIFT", "FIELD",
		}
		fmt.Fprintf(os.Stderr,
			"[meta_router] trigger: %s (observation #%d)\n",
			templateNames[t.id], r.observationCount)

		return t.id
	}

	return -1
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILL PARAMS — Construct template params from current metrics
// ═══════════════════════════════════════════════════════════════════════════════

func (r *MetaRouter) fillParams(templateID int, snap *Snapshot) {
	p := &r.lastParams
	p.TemplateType = templateID

	// Reset to defaults
	for i := 0; i < 8; i++ {
		p.AttentionBiases[i] = 0.0
		p.LayerFocus[i] = 1.0
	}

	switch templateID {
	case metaTemplateThermograph:
		// Steady observer. V-focused. Biases from arousal.
		p.Temperature = 0.5
		p.DeltaTarget = 2 // V
		for i := 0; i < 8; i++ {
			p.AttentionBiases[i] = (snap.Arousal - 0.5) * 0.2
		}

	case metaTemplateSilence:
		// Cool, still. Q-focused. Early layers. Biases from entropy.
		p.Temperature = 0.3
		p.DeltaTarget = 0 // Q
		for i := 0; i < 8; i++ {
			if i < 4 {
				p.LayerFocus[i] = 1.0
			} else {
				p.LayerFocus[i] = 0.3
			}
			p.AttentionBiases[i] = snap.Entropy * 0.1
		}

	case metaTemplateDrift:
		// Warm, tracking. K-focused. Middle layers. Biases from drift.
		p.Temperature = 0.7
		p.DeltaTarget = 1 // K
		for i := 0; i < 8; i++ {
			if i >= 3 && i <= 5 {
				p.LayerFocus[i] = 1.0
			} else {
				p.LayerFocus[i] = 0.4
			}
			p.AttentionBiases[i] = snap.DriftDirection * 0.15
		}

	case metaTemplateField:
		// Hot, wide. All Q/K/V. Late layers. Biases from coherence.
		p.Temperature = 0.9
		p.DeltaTarget = 3 // all
		for i := 0; i < 8; i++ {
			if i >= 5 {
				p.LayerFocus[i] = 1.0
			} else {
				p.LayerFocus[i] = 0.4
			}
			p.AttentionBiases[i] = (snap.Coherence - 0.5) * 0.1
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// CGO EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

//export meta_router_init
func meta_router_init() {
	getMetaRouter()
}

//export meta_router_tick
func meta_router_tick() C.int {
	return C.int(getMetaRouter().tick())
}

//export meta_router_get_params
func meta_router_get_params(out *C.MetaTemplateParams) {
	r := getMetaRouter()
	r.mu.Lock()
	defer r.mu.Unlock()

	p := &r.lastParams
	out.template_type = C.int(p.TemplateType)
	for i := 0; i < 8; i++ {
		out.attention_biases[i] = C.float(p.AttentionBiases[i])
		out.layer_focus[i] = C.float(p.LayerFocus[i])
	}
	out.temperature = C.float(p.Temperature)
	out.delta_target = C.int(p.DeltaTarget)
}

//export meta_router_feed_thermogram
func meta_router_feed_thermogram(thermo *C.MetaThermogram) {
	r := getMetaRouter()
	r.mu.Lock()
	defer r.mu.Unlock()

	r.lastThermogram = metaRouterThermo{
		Warmth:         float32(thermo.warmth),
		Sharpness:      float32(thermo.sharpness),
		Silence:        float32(thermo.silence),
		Uncertainty:    float32(thermo.uncertainty),
		DriftRate:      float32(thermo.drift_rate),
		DriftDirection: int(thermo.drift_direction),
		Valid:          int(thermo.valid) != 0,
		TemplateUsed:   int(thermo.template_used),
	}
	for i := 0; i < 8; i++ {
		r.lastThermogram.FieldVector[i] = float32(thermo.field_vector[i])
	}

	// Feed thermogram back to InnerWorld as a signal
	if r.lastThermogram.Valid {
		// Nudge emotions based on observation
		warmthDelta := (r.lastThermogram.Warmth - 0.5) * 0.1
		Global().State.mu.Lock()
		Global().State.Arousal = clamp(
			Global().State.Arousal+warmthDelta, 0, 1)
		Global().State.mu.Unlock()
	}
}

//export meta_router_get_observation_count
func meta_router_get_observation_count() C.int {
	r := getMetaRouter()
	r.mu.Lock()
	defer r.mu.Unlock()
	return C.int(r.observationCount)
}
