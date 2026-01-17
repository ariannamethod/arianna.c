// inner_world/cgo_bridge.go — CGO bridge for C interop
// ═══════════════════════════════════════════════════════════════════════════════
// הגשר בין C ל-Go
// The bridge between C and Go
// ═══════════════════════════════════════════════════════════════════════════════
//
// This file exports Go functions to C using cgo.
// C code can #include "inner_world.h" and call these functions.
//
// Build with: go build -buildmode=c-shared -o libinner_world.so
// Or:         go build -buildmode=c-archive -o libinner_world.a
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

/*
#include <stdlib.h>
#include <string.h>

// Snapshot struct for C
typedef struct {
    float arousal;
    float valence;
    float entropy;
    float coherence;
    float trauma_level;
    int loop_count;
    int abstraction_depth;
    int self_ref_count;
    float drift_direction;
    float drift_speed;
    float memory_pressure;
    float focus_strength;
    float wander_pull;
    float prophecy_debt;
    float destiny_pull;
    float wormhole_chance;
} InnerWorldSnapshot;

// Text analysis struct for C
typedef struct {
    float trauma_activation;
    float identity_pull;
    float repetition_score;
    float abstraction_score;
    float self_ref_score;
    float overthink_total;
    float focus_strength;
    float destiny_bias;
    int wormhole_active;
    int wormhole_skip;
    float temporal_dissonance;
} InnerWorldTextAnalysis;
*/
import "C"

import (
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_init
func inner_world_init() {
	Init()
}

//export inner_world_shutdown
func inner_world_shutdown() {
	Shutdown()
}

//export inner_world_step
func inner_world_step(dt C.float) {
	StepGlobal(float32(dt))
}

//export inner_world_auto_step
func inner_world_auto_step() {
	AutoStep()
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE GETTERS
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_get_arousal
func inner_world_get_arousal() C.float {
	return C.float(Global().State.GetArousal())
}

//export inner_world_get_trauma
func inner_world_get_trauma() C.float {
	return C.float(Global().State.GetTraumaLevel())
}

//export inner_world_get_coherence
func inner_world_get_coherence() C.float {
	return C.float(Global().State.GetCoherence())
}

//export inner_world_get_prophecy_debt
func inner_world_get_prophecy_debt() C.float {
	return C.float(Global().State.GetProphecyDebt())
}

//export inner_world_get_snapshot
func inner_world_get_snapshot(out *C.InnerWorldSnapshot) {
	snap := GetSnapshotGlobal()

	out.arousal = C.float(snap.Arousal)
	out.valence = C.float(snap.Valence)
	out.entropy = C.float(snap.Entropy)
	out.coherence = C.float(snap.Coherence)
	out.trauma_level = C.float(snap.TraumaLevel)
	out.loop_count = C.int(snap.LoopCount)
	out.abstraction_depth = C.int(snap.AbstractionDepth)
	out.self_ref_count = C.int(snap.SelfRefCount)
	out.drift_direction = C.float(snap.DriftDirection)
	out.drift_speed = C.float(snap.DriftSpeed)
	out.memory_pressure = C.float(snap.MemoryPressure)
	out.focus_strength = C.float(snap.FocusStrength)
	out.wander_pull = C.float(snap.WanderPull)
	out.prophecy_debt = C.float(snap.ProphecyDebt)
	out.destiny_pull = C.float(snap.DestinyPull)
	out.wormhole_chance = C.float(snap.WormholeChance)
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE SETTERS
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_set_arousal
func inner_world_set_arousal(v C.float) {
	Global().State.SetArousal(float32(v))
}

//export inner_world_set_trauma
func inner_world_set_trauma(v C.float) {
	Global().State.SetTraumaLevel(float32(v))
}

//export inner_world_set_coherence
func inner_world_set_coherence(v C.float) {
	Global().State.SetCoherence(float32(v))
}

//export inner_world_add_prophecy_debt
func inner_world_add_prophecy_debt(delta C.float) {
	Global().State.AddProphecyDebt(float32(delta))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEXT PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_process_text
func inner_world_process_text(text *C.char, out *C.InnerWorldTextAnalysis) {
	goText := C.GoString(text)
	analysis := ProcessTextGlobal(goText)

	out.trauma_activation = C.float(analysis.TraumaActivation)
	out.identity_pull = C.float(analysis.IdentityPull)
	out.repetition_score = C.float(analysis.RepetitionScore)
	out.abstraction_score = C.float(analysis.AbstractionScore)
	out.self_ref_score = C.float(analysis.SelfRefScore)
	out.overthink_total = C.float(analysis.OverthinkTotal)
	out.focus_strength = C.float(analysis.FocusStrength)
	out.destiny_bias = C.float(analysis.DestinyBias)
	if analysis.WormholeActive {
		out.wormhole_active = 1
	} else {
		out.wormhole_active = 0
	}
	out.wormhole_skip = C.int(analysis.WormholeSkip)
	out.temporal_dissonance = C.float(analysis.TemporalDissonance)
}

//export inner_world_check_trauma
func inner_world_check_trauma(text *C.char) C.float {
	goText := C.GoString(text)
	if ts := Global().GetTraumaSurfacing(); ts != nil {
		return C.float(ts.CheckText(goText))
	}
	return 0
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPHECY/DESTINY
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_accumulate_prophecy_debt
func inner_world_accumulate_prophecy_debt(probability C.float) {
	if pd := Global().GetProphecyDebt(); pd != nil {
		pd.AccumulateDebt(float32(probability))
	}
}

//export inner_world_check_wormhole
func inner_world_check_wormhole() C.int {
	if pd := Global().GetProphecyDebt(); pd != nil {
		active, skip := pd.CheckWormhole()
		if active {
			return C.int(skip)
		}
	}
	return 0
}

//export inner_world_get_destiny_bias
func inner_world_get_destiny_bias() C.float {
	if pd := Global().GetProphecyDebt(); pd != nil {
		return C.float(pd.GetDestinyBias())
	}
	return 0
}

//export inner_world_get_lookahead
func inner_world_get_lookahead() C.int {
	if pd := Global().GetProphecyDebt(); pd != nil {
		return C.int(pd.GetLookahead())
	}
	return 3
}

//export inner_world_set_lookahead
func inner_world_set_lookahead(n C.int) {
	if pd := Global().GetProphecyDebt(); pd != nil {
		pd.SetLookahead(int(n))
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_focus
func inner_world_focus(target *C.char, strength C.float) {
	goTarget := C.GoString(target)
	if aw := Global().GetAttentionWandering(); aw != nil {
		aw.Focus(goTarget, float32(strength))
	}
}

//export inner_world_is_wandering
func inner_world_is_wandering() C.int {
	if aw := Global().GetAttentionWandering(); aw != nil {
		if aw.IsWandering() {
			return 1
		}
	}
	return 0
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_add_memory
func inner_world_add_memory(content *C.char, emotion C.float, arousal C.float) {
	goContent := C.GoString(content)
	if mc := Global().GetMemoryConsolidation(); mc != nil {
		mc.AddMemory(goContent, float32(emotion), float32(arousal))
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL DRIFT
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_nudge_emotion
func inner_world_nudge_emotion(d_valence C.float, d_arousal C.float) {
	if ed := Global().GetEmotionalDrift(); ed != nil {
		ed.Nudge(float32(d_valence), float32(d_arousal))
	}
}

//export inner_world_get_dominant_emotion
func inner_world_get_dominant_emotion(buf *C.char, buf_size C.int) {
	if ed := Global().GetEmotionalDrift(); ed != nil {
		emotion := ed.GetDominantEmotion()
		cstr := C.CString(emotion)
		defer C.free(unsafe.Pointer(cstr))

		// Safe copy
		size := len(emotion)
		if size >= int(buf_size) {
			size = int(buf_size) - 1
		}
		C.memcpy(unsafe.Pointer(buf), unsafe.Pointer(cstr), C.size_t(size))
		*(*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(buf)) + uintptr(size))) = 0
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// OVERTHINKING
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_is_spiraling
func inner_world_is_spiraling() C.int {
	if ol := Global().GetOverthinkingLoops(); ol != nil {
		if ol.IsSpiraling() {
			return 1
		}
	}
	return 0
}

//export inner_world_suggest_break
func inner_world_suggest_break(buf *C.char, buf_size C.int) {
	if ol := Global().GetOverthinkingLoops(); ol != nil {
		suggestion := ol.SuggestBreak()
		if len(suggestion) > 0 {
			cstr := C.CString(suggestion)
			defer C.free(unsafe.Pointer(cstr))

			size := len(suggestion)
			if size >= int(buf_size) {
				size = int(buf_size) - 1
			}
			C.memcpy(unsafe.Pointer(buf), unsafe.Pointer(cstr), C.size_t(size))
			*(*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(buf)) + uintptr(size))) = 0
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// DSL COMMANDS (from ariannamethod.lang style)
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_dsl_destiny
func inner_world_dsl_destiny(strength C.float) {
	if pd := Global().GetProphecyDebt(); pd != nil {
		pd.destinyStrength = float32(strength)
	}
}

//export inner_world_dsl_prophecy
func inner_world_dsl_prophecy(lookahead C.int) {
	if pd := Global().GetProphecyDebt(); pd != nil {
		pd.SetLookahead(int(lookahead))
	}
}

//export inner_world_dsl_wormhole
func inner_world_dsl_wormhole(chance C.float) {
	if pd := Global().GetProphecyDebt(); pd != nil {
		pd.wormholeChance = clamp(float32(chance), 0, 1)
	}
}

//export inner_world_dsl_velocity
func inner_world_dsl_velocity(mode C.int) {
	// Velocity affects emotional arousal
	// 0 = NOMOVE, 1 = WALK, 2 = RUN, -1 = BACKWARD
	if ed := Global().GetEmotionalDrift(); ed != nil {
		switch mode {
		case 0: // NOMOVE - calm
			ed.Nudge(0, -0.1)
		case 1: // WALK - neutral
			// no change
		case 2: // RUN - activated
			ed.Nudge(0, 0.2)
		case -1: // BACKWARD - tension
			ed.Nudge(-0.1, 0.15)
		}
	}
}

//export inner_world_dsl_pain
func inner_world_dsl_pain(level C.float) {
	// Pain increases trauma and decreases valence
	Global().State.SetTraumaLevel(Global().State.GetTraumaLevel() + float32(level)*0.3)
	if ed := Global().GetEmotionalDrift(); ed != nil {
		ed.Nudge(-float32(level)*0.2, float32(level)*0.1)
	}
}

//export inner_world_dsl_tension
func inner_world_dsl_tension(level C.float) {
	// Tension increases arousal and entropy
	state := Global().State
	state.mu.Lock()
	state.Arousal = clamp(state.Arousal+float32(level)*0.2, 0, 1)
	state.Entropy = clamp(state.Entropy+float32(level)*0.1, 0, 1)
	state.mu.Unlock()
}

//export inner_world_dsl_dissonance
func inner_world_dsl_dissonance(level C.float) {
	// Dissonance decreases coherence
	Global().State.SetCoherence(Global().State.GetCoherence() - float32(level)*0.2)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE PARAMETERS (Linux-like sysctl interface)
// ═══════════════════════════════════════════════════════════════════════════════

//export inner_world_set_param
func inner_world_set_param(name *C.char, value C.float) C.int {
	goName := C.GoString(name)
	err := GetAdaptiveEngine().SetParam(goName, float32(value))
	if err != nil {
		return 0
	}
	return 1
}

//export inner_world_get_param
func inner_world_get_param(name *C.char) C.float {
	goName := C.GoString(name)
	value, err := GetAdaptiveEngine().GetParam(goName)
	if err != nil {
		return -1
	}
	return C.float(value)
}

//export inner_world_load_config
func inner_world_load_config(path *C.char) C.int {
	goPath := C.GoString(path)
	err := GetAdaptiveEngine().LoadConfig(goPath)
	if err != nil {
		return 0
	}
	return 1
}

//export inner_world_save_config
func inner_world_save_config(path *C.char) C.int {
	goPath := C.GoString(path)
	err := GetAdaptiveEngine().SaveConfig(goPath)
	if err != nil {
		return 0
	}
	return 1
}

// main is required for buildmode=c-shared
func main() {}
