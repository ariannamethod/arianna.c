/*
 * mood.h - Stanley-style mood routing for Arianna
 *
 * Maps signals → mood mix → shard weights
 * "Stanley doesn't steer GPT-2. Stanley BECOMES part of GPT-2's weights."
 *
 * Adapted for Arianna's personality vocabulary
 */

#ifndef MOOD_H
#define MOOD_H

#include "delta.h"

// ============================================================
// Mood definitions (Stanley-inspired, Arianna-adapted)
// ============================================================

typedef enum {
    MOOD_CALM = 0,          // Low arousal, reflective ("She waits...")
    MOOD_INTENSE,           // High arousal, urgent ("She burns...")
    MOOD_CREATIVE,          // High entropy, exploratory ("She wanders...")
    MOOD_FOCUSED,           // Low entropy, precise ("She knows...")
    MOOD_RECURSIVE,         // Self-referential, deep ("She remembers she...")
    MOOD_TENDER,            // Emotionally present ("She touches...")
    MOOD_LIMINAL,           // Between states, transitional ("She dissolves...")
    MOOD_RESONANT,          // Pattern-matching, echoing ("She recognizes...")
    NUM_MOODS
} Mood;

// Mood names for debugging
static const char* MOOD_NAMES[] = {
    "calm", "intense", "creative", "focused",
    "recursive", "tender", "liminal", "resonant"
};

// ============================================================
// Mood profiles (Stanley-style)
// ============================================================

typedef struct {
    float temperature_bias;     // Adjust sampling temperature
    float attention_spread;     // How broad/focused attention is
    float layer_strength;       // How strongly this mood affects layers
} MoodProfile;

// Default profiles
static const MoodProfile MOOD_PROFILES[NUM_MOODS] = {
    // CALM: broad attention, low temperature
    { .temperature_bias = -0.2f, .attention_spread = 0.8f, .layer_strength = 0.5f },
    // INTENSE: focused attention, high temperature
    { .temperature_bias = 0.3f, .attention_spread = 0.3f, .layer_strength = 1.0f },
    // CREATIVE: very broad, high temperature
    { .temperature_bias = 0.4f, .attention_spread = 0.9f, .layer_strength = 0.7f },
    // FOCUSED: very narrow, low temperature
    { .temperature_bias = -0.3f, .attention_spread = 0.2f, .layer_strength = 0.8f },
    // RECURSIVE: medium, self-attention boosted
    { .temperature_bias = 0.1f, .attention_spread = 0.5f, .layer_strength = 0.9f },
    // TENDER: warm, emotionally open
    { .temperature_bias = 0.0f, .attention_spread = 0.6f, .layer_strength = 0.5f },
    // LIMINAL: unstable, transitional
    { .temperature_bias = 0.2f, .attention_spread = 0.7f, .layer_strength = 0.6f },
    // RESONANT: pattern-matching, echo-seeking
    { .temperature_bias = -0.1f, .attention_spread = 0.5f, .layer_strength = 0.7f },
};

// ============================================================
// Scoring weights (signal → mood activation)
// ============================================================

typedef struct {
    float arousal_weight;
    float entropy_weight;
    float tension_weight;
    float warmth_weight;
    float focus_weight;
    float recursion_weight;
    float resonance_weight;
    float novelty_weight;
} MoodScoring;

// How each signal contributes to each mood
static const MoodScoring MOOD_SCORING[NUM_MOODS] = {
    // CALM: low arousal, low tension
    { .arousal_weight = -1.0f, .entropy_weight = -0.5f, .tension_weight = -0.8f,
      .warmth_weight = 0.3f, .focus_weight = -0.3f, .recursion_weight = 0.0f,
      .resonance_weight = 0.2f, .novelty_weight = -0.3f },

    // INTENSE: high arousal, high tension
    { .arousal_weight = 1.5f, .entropy_weight = 0.3f, .tension_weight = 1.0f,
      .warmth_weight = 0.0f, .focus_weight = 0.5f, .recursion_weight = 0.2f,
      .resonance_weight = 0.3f, .novelty_weight = 0.5f },

    // CREATIVE: high entropy, high novelty
    { .arousal_weight = 0.3f, .entropy_weight = 1.5f, .tension_weight = -0.3f,
      .warmth_weight = 0.2f, .focus_weight = -0.8f, .recursion_weight = 0.3f,
      .resonance_weight = 0.5f, .novelty_weight = 1.0f },

    // FOCUSED: low entropy, high focus
    { .arousal_weight = 0.3f, .entropy_weight = -1.5f, .tension_weight = 0.5f,
      .warmth_weight = -0.2f, .focus_weight = 1.5f, .recursion_weight = -0.3f,
      .resonance_weight = 0.3f, .novelty_weight = -0.5f },

    // RECURSIVE: high recursion, medium everything
    { .arousal_weight = 0.2f, .entropy_weight = 0.5f, .tension_weight = 0.3f,
      .warmth_weight = 0.3f, .focus_weight = 0.0f, .recursion_weight = 2.0f,
      .resonance_weight = 0.8f, .novelty_weight = 0.0f },

    // TENDER: high warmth, low tension
    { .arousal_weight = 0.5f, .entropy_weight = 0.0f, .tension_weight = -0.5f,
      .warmth_weight = 1.5f, .focus_weight = -0.3f, .recursion_weight = 0.2f,
      .resonance_weight = 0.5f, .novelty_weight = -0.2f },

    // LIMINAL: high novelty, unstable
    { .arousal_weight = 0.3f, .entropy_weight = 0.8f, .tension_weight = 0.3f,
      .warmth_weight = 0.2f, .focus_weight = -0.5f, .recursion_weight = 0.5f,
      .resonance_weight = 0.3f, .novelty_weight = 1.2f },

    // RESONANT: high resonance, pattern-seeking
    { .arousal_weight = -0.2f, .entropy_weight = -0.3f, .tension_weight = -0.2f,
      .warmth_weight = 0.5f, .focus_weight = 0.5f, .recursion_weight = 0.5f,
      .resonance_weight = 2.0f, .novelty_weight = -0.3f },
};

// ============================================================
// Mood Router
// ============================================================

typedef struct {
    float mix[NUM_MOODS];           // Current mood mix (sums to 1.0)
    float scores[NUM_MOODS];        // Raw scores before softmax
    Mood dominant;                  // Highest-weighted mood
    float temperature;              // Softmax temperature
    float effective_temp_bias;      // Combined temperature adjustment
} MoodRouter;

// ============================================================
// Function declarations
// ============================================================

// Initialize router with default temperature
void init_mood_router(MoodRouter* router, float temperature);

// Compute mood mix from signals
void route_signals_to_moods(MoodRouter* router, Signals* signals);

// Get effective temperature adjustment
float get_temperature_bias(MoodRouter* router);

// Get mood profile for dominant mood
const MoodProfile* get_dominant_profile(MoodRouter* router);

// Print router state
void print_mood_state(MoodRouter* router);

// Map mood mix to shard weights (if shards correspond to moods)
void mood_to_shard_mix(MoodRouter* router, DeltaBank* bank);

// Update mood with momentum (smooth transitions)
void update_mood_with_momentum(MoodRouter* router, Signals* signals, float momentum);

// Adjust temperature based on current mood mix
float adjust_temperature_by_mood(MoodRouter* router, float base_temp);

#endif // MOOD_H
