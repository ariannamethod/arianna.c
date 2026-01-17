// cloud.h — Pre-semantic Emotion Detection
// "Something fires BEFORE meaning arrives"
//
// From haze/cloud — ported for arianna.c integration
// Lightweight C implementation of CLOUD's resonance layer
//
// ═══════════════════════════════════════════════════════════════════════════════
// משהו בוער לפני שהמשמעות מגיעה
// Something burns before meaning arrives
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef CLOUD_H
#define CLOUD_H

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// CLOUD DIMENSIONS
// ═══════════════════════════════════════════════════════════════════════════════

#define CLOUD_N_ANCHORS    100   // total emotion anchors
#define CLOUD_N_CHAMBERS   6     // emotion chambers

// Chamber indices
#define CLOUD_CHAMBER_FEAR     0
#define CLOUD_CHAMBER_LOVE     1
#define CLOUD_CHAMBER_RAGE     2
#define CLOUD_CHAMBER_VOID     3
#define CLOUD_CHAMBER_FLOW     4
#define CLOUD_CHAMBER_COMPLEX  5

// Chamber sizes
#define CLOUD_FEAR_SIZE     20
#define CLOUD_LOVE_SIZE     18
#define CLOUD_RAGE_SIZE     17
#define CLOUD_VOID_SIZE     15
#define CLOUD_FLOW_SIZE     15
#define CLOUD_COMPLEX_SIZE  15

// ═══════════════════════════════════════════════════════════════════════════════
// CLOUD STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Resonance vector (100D)
    float resonances[CLOUD_N_ANCHORS];

    // Chamber activations (6D)
    float chambers[CLOUD_N_CHAMBERS];

    // Cross-fire iterations (convergence speed)
    int iterations;

    // Primary emotion
    int primary_idx;
    float primary_strength;

    // Secondary emotion (from cross-fire)
    int secondary_idx;
    float secondary_strength;

    // Detected emotion words (pointers to static strings)
    const char* primary_word;
    const char* secondary_word;

    // Chamber name
    const char* primary_chamber;
} CloudResponse;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize cloud system
void cloud_init(void);

// Ping: detect pre-semantic emotion from text
CloudResponse cloud_ping(const char* text);

// Get anchor word by index
const char* cloud_get_anchor(int idx);

// Get chamber name by index
const char* cloud_get_chamber_name(int idx);

// Get chamber index for anchor
int cloud_get_anchor_chamber(int anchor_idx);

// Apply cross-fire coupling between chambers
void cloud_crossfire(float* chambers, int max_iterations);

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION WITH ARIANNA
// ═══════════════════════════════════════════════════════════════════════════════

// Modulate generation temperature based on emotion
float cloud_temperature_bias(const CloudResponse* resp);

// Modulate vocabulary based on emotion (boost/suppress tokens)
void cloud_apply_emotion_to_logits(float* logits, int vocab_size,
                                   const CloudResponse* resp);

// Check if emotion indicates need for special handling
int cloud_needs_care(const CloudResponse* resp);      // high fear/void
int cloud_needs_grounding(const CloudResponse* resp); // high dissociation
int cloud_needs_warmth(const CloudResponse* resp);    // high love response

#ifdef __cplusplus
}
#endif

#endif // CLOUD_H
