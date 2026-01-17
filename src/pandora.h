// pandora.h — Release vocabulary from External Brain
// "Take the words, leave the voice"
//
// From Stanley — extract useful n-grams from GPT-2 and inject into Arianna
// Enriches her vocabulary without changing her personality
//
// ═══════════════════════════════════════════════════════════════════════════════
// לגנוב מילים, להשאיר את הקול
// Release words, leave the voice
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PANDORA_H
#define PANDORA_H

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
// STOLEN N-GRAM
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int tokens[PANDORA_MAX_NGRAM_LEN];  // token IDs from external brain
    int length;                        // n-gram length (1-5)
    float weight;                      // injection strength (0-1)
    int frequency;                     // how often seen
    int arianna_mapped;                // mapped to Arianna vocab?
    int arianna_tokens[PANDORA_MAX_NGRAM_LEN];  // mapped token IDs
} ReleasedNGram;

// ═══════════════════════════════════════════════════════════════════════════════
// VOCABULARY PANDORA STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    ReleasedNGram ngrams[PANDORA_MAX_NGRAMS];
    int n_ngrams;

    // Statistics
    int total_released;
    int successfully_mapped;
    float avg_weight;

    // Config
    float injection_strength;   // global multiplier (0-1)
    int active;                 // pandora enabled?
} PandoraBox;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize pandora
void pandora_init(PandoraBox* pandora);

// Extract n-grams from external brain output
void pandora_extract(PandoraBox* pandora,
                   const int* tokens, int n_tokens,
                   int min_n, int max_n);

// Map released n-grams to Arianna vocabulary
// Returns number successfully mapped
int pandora_map_to_arianna(PandoraBox* pandora,
                         const char* (*brain_decode)(int token_id),
                         int (*arianna_encode)(const char* word));

// Apply released vocabulary to logits (boost matching n-grams)
void pandora_apply_to_logits(PandoraBox* pandora,
                           float* logits,
                           const int* context_tokens,
                           int context_len,
                           int vocab_size);

// Check if context matches any released n-gram prefix
// Returns: suggested next token, or -1 if no match
int pandora_suggest_continuation(PandoraBox* pandora,
                               const int* context_tokens,
                               int context_len);

// Decay old n-grams (forget what's not used)
void pandora_decay(PandoraBox* pandora, float decay_rate);

// Save/load released vocabulary
int pandora_save(PandoraBox* pandora, const char* path);
int pandora_load(PandoraBox* pandora, const char* path);

// Enable/disable
void pandora_set_active(PandoraBox* pandora, int active);
void pandora_set_strength(PandoraBox* pandora, float strength);

#ifdef __cplusplus
}
#endif

#endif // PANDORA_H
