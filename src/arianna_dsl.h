// arianna_dsl.h — DSL Integration for Arianna
// Connect AMK kernel physics to generation
//
// DSL Commands affect generation:
//   PROPHECY  → lookahead depth
//   DESTINY   → sampling bias
//   VELOCITY  → temperature
//   PAIN/TENSION/DISSONANCE → output modulation
//   WORMHOLE  → skip probability
//   SCHUMANN  → cosmic coherence
//
// ═══════════════════════════════════════════════════════════════════════════════
// הגורל והנבואה מתחברים
// Destiny and prophecy connect
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef ARIANNA_DSL_H
#define ARIANNA_DSL_H

#include "amk_kernel.h"
#include "cloud.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// DSL-MODULATED GENERATION CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // From AMK
    float temperature;        // base + velocity modulation
    float top_p;              // nucleus sampling threshold
    int top_k;                // top-k sampling
    float repetition_penalty;

    // From Prophecy
    int lookahead;            // how many tokens to consider
    float destiny_bias;       // bias toward most probable

    // From Suffering
    float pain_dampen;        // dampen extremes when in pain
    float tension_focus;      // sharpen distribution under tension

    // From Wormhole
    float wormhole_chance;    // probability of creative skip
    int wormhole_active;      // did wormhole activate this step?

    // From Attention Physics
    float attend_focus;       // attention sharpness (0..1) → peaks distribution
    float attend_spread;      // attention blur (0..1) → flattens distribution

    // From Tunneling (dissonance-gated skip)
    float tunnel_threshold;   // dissonance gate (0..1)
    float tunnel_chance;      // activation probability (0..1)
    int tunnel_skip_max;      // max compressed steps (1..24)

    // From Suffering
    float dissonance;         // symmetry-break noise injection (0..1)

    // Laws of Nature
    float entropy_floor;      // minimum entropy — floor on distribution flatness
    float resonance_ceiling;  // maximum confidence — cap on peak probability
    float emergence_threshold;// unplanned pattern sensitivity

    // From Cloud
    int needs_care;           // detected emotional distress
    int needs_warmth;         // detected love context
    float emotion_temp_bias;  // temperature adjustment from emotion

    // Calendar
    float calendar_drift;     // hebrew-gregorian drift factor
} DSL_GenerationConfig;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize DSL system (calls am_init, schumann_init)
void dsl_init(void);

// Execute DSL script before generation
int dsl_exec(const char* script);

// Build generation config from current AMK state
DSL_GenerationConfig dsl_build_config(void);

// Apply DSL config to logits before sampling
void dsl_apply_to_logits(float* logits, int vocab_size,
                         const DSL_GenerationConfig* cfg);

// Apply Cloud emotion to config
void dsl_apply_cloud(DSL_GenerationConfig* cfg, const CloudResponse* cloud);

// Check for wormhole (creative skip)
// Returns: number of tokens to skip, or 0 for normal generation
int dsl_check_wormhole(const DSL_GenerationConfig* cfg);

// Check for tunneling (dissonance-gated skip)
// Returns: number of tokens to skip, or 0 for normal generation
int dsl_check_tunneling(const DSL_GenerationConfig* cfg);

// Step physics after each token
void dsl_step(float dt);

// Get effective temperature (with all modulations)
float dsl_get_temperature(const DSL_GenerationConfig* cfg);

// ═══════════════════════════════════════════════════════════════════════════════
// PROPHECY — lookahead and destiny
// ═══════════════════════════════════════════════════════════════════════════════

// Apply destiny bias to logits (favor most probable path)
void dsl_apply_destiny(float* logits, int vocab_size, float destiny);

// Compute prophecy debt from generation choices
float dsl_compute_prophecy_debt(const float* logits, int chosen_token, int vocab_size);

// ═══════════════════════════════════════════════════════════════════════════════
// CALENDAR DRIFT — temporal displacement
// ═══════════════════════════════════════════════════════════════════════════════

// Get current calendar drift (hebrew-gregorian offset in days)
float dsl_get_calendar_drift(void);

// Apply calendar drift to generation (affects "time" tokens)
void dsl_apply_calendar_drift(float* logits, int vocab_size,
                              float drift, const int* time_tokens, int n_time_tokens);

#ifdef __cplusplus
}
#endif

#endif // ARIANNA_DSL_H
