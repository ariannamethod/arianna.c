// pandora.h — Release vocabulary from External Brain
// "Take the words, leave the voice"
//
// Pandora Package — Pure C vocabulary extraction using GPT2-30M
// Proves: Architecture > Weights
//
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PANDORA_H
#define PANDORA_H

#include "gpt2_30m.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// PANDORA CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

#define PANDORA_MAX_NGRAMS     1000   // max released n-grams
#define PANDORA_MAX_NGRAM_LEN  5      // max tokens per n-gram
#define PANDORA_MIN_FREQUENCY  3      // min occurrences to release

// ═══════════════════════════════════════════════════════════════════════════════
// RELEASED N-GRAM
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int tokens[PANDORA_MAX_NGRAM_LEN];  // token IDs from external brain
    int length;                          // n-gram length (1-5)
    float weight;                        // injection strength (0-1)
    int frequency;                       // how often seen
    int arianna_mapped;                  // mapped to Arianna vocab?
    int arianna_tokens[PANDORA_MAX_NGRAM_LEN];  // mapped token IDs
} ReleasedNGram;

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATION MODE (SARTRE-driven)
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    PANDORA_MODE_OFF = 0,       // Disabled
    PANDORA_MODE_AUTO,          // SARTRE-controlled activation
    PANDORA_MODE_FORCED,        // Always active
} PandoraMode;

// ═══════════════════════════════════════════════════════════════════════════════
// PANDORA STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Released n-grams
    ReleasedNGram ngrams[PANDORA_MAX_NGRAMS];
    int n_ngrams;

    // Statistics
    int total_released;
    int successfully_mapped;
    float avg_weight;

    // Config
    float injection_strength;   // global multiplier (0-1)
    PandoraMode mode;           // activation mode

    // SARTRE thresholds
    float coherence_threshold;  // activate below this (default 0.3)
    float sacred_threshold;     // deactivate above this (default 0.7)

    // External brain
    GPT2_30M* brain;
    GPT2Vocab* vocab;
    int brain_owned;            // did we allocate brain?
} PandoraBox;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize pandora
void pandora_init(PandoraBox* pandora);

// Initialize with external brain
int pandora_init_with_brain(
    PandoraBox* pandora,
    const char* weights_path,
    const char* vocab_path
);

// Free resources
void pandora_free(PandoraBox* pandora);

// Extract n-grams from external brain output
void pandora_extract(
    PandoraBox* pandora,
    const int* tokens,
    int n_tokens,
    int min_n,
    int max_n
);

// Map released n-grams to Arianna vocabulary
int pandora_map_to_arianna(
    PandoraBox* pandora,
    const char* (*brain_decode)(int token_id),
    int (*arianna_encode)(const char* word)
);

// Apply released vocabulary to logits
void pandora_apply_to_logits(
    PandoraBox* pandora,
    float* logits,
    const int* context_tokens,
    int context_len,
    int vocab_size
);

// Suggest continuation token
int pandora_suggest_continuation(
    PandoraBox* pandora,
    const int* context_tokens,
    int context_len
);

// Decay old n-grams
void pandora_decay(PandoraBox* pandora, float decay_rate);

// ═══════════════════════════════════════════════════════════════════════════════
// SARTRE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Check if pandora should be active based on SARTRE metrics
// Returns 1 if should activate, 0 if should deactivate
int pandora_check_sartre(
    PandoraBox* pandora,
    float coherence,
    float sacred,
    int pattern  // ResonancePattern enum value
);

// Update thresholds
void pandora_set_thresholds(
    PandoraBox* pandora,
    float coherence_threshold,
    float sacred_threshold
);

// ═══════════════════════════════════════════════════════════════════════════════
// FULL PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

// Process input through external brain and extract vocabulary
// Returns number of new n-grams extracted
int pandora_process(
    PandoraBox* pandora,
    const char* input,
    int (*arianna_encode)(const char* word),
    int max_generate
);

// ═══════════════════════════════════════════════════════════════════════════════
// MODE & CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_set_mode(PandoraBox* pandora, PandoraMode mode);
void pandora_set_strength(PandoraBox* pandora, float strength);

// Check if currently active (considering mode and SARTRE)
int pandora_is_active(PandoraBox* pandora);

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_save(PandoraBox* pandora, const char* path);
int pandora_load(PandoraBox* pandora, const char* path);

// Clear all extracted vocabulary
void pandora_clear(PandoraBox* pandora);

// ═══════════════════════════════════════════════════════════════════════════════
// STATS
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int total_ngrams;
    int mapped_ngrams;
    float avg_weight;
    float avg_frequency;
    int mode;
    float injection_strength;
} PandoraStats;

PandoraStats pandora_get_stats(PandoraBox* pandora);

#ifdef __cplusplus
}
#endif

#endif // PANDORA_H
