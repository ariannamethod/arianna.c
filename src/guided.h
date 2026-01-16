/*
 * guided.h - Stanley-style Guided Attention for Arianna
 *
 * "Stanley doesn't generate text. Stanley guides attention."
 *
 * Core concepts from Stanley:
 * - Gravity centers: anchor words that pull attention
 * - Hot words: frequent words from context
 * - Spiral topics: recurring themes in overthinking
 * - Attention bias: token-level steering of generation
 */

#ifndef GUIDED_H
#define GUIDED_H

#include "arianna.h"
#include "delta.h"
#include <string.h>

// ============================================================
// Configuration
// ============================================================

#define MAX_KEYWORDS 64          // Max keywords per category
#define MAX_KEYWORD_LEN 32       // Max chars per keyword
#define MAX_SPIRAL_TOPICS 16     // Overthinking spirals
#define MAX_GRAVITY_CENTERS 32   // Identity anchors

// ============================================================
// Pulse - Impact metrics from input (Stanley's subjectivity)
// ============================================================

typedef struct {
    float novelty;      // % words not in corpus [0,1]
    float arousal;      // Intensity (caps, punctuation) [0,1]
    float entropy;      // Word diversity [0,1]
    float valence;      // Positive/negative tone [-1,1]
} Pulse;

// ============================================================
// Identity - Core personality anchors
// ============================================================

typedef struct {
    char gravity_centers[MAX_GRAVITY_CENTERS][MAX_KEYWORD_LEN];
    int n_gravity;

    char warm_words[MAX_KEYWORDS][MAX_KEYWORD_LEN];  // Emotional attractors
    int n_warm;

    char cold_words[MAX_KEYWORDS][MAX_KEYWORD_LEN];  // Emotional repellers
    int n_cold;

    // Lexicon stats
    int lexicon_size;
    float lexicon_coverage;  // % of corpus covered
} Identity;

// ============================================================
// Extended Stanley Signals
// ============================================================

typedef struct {
    // === From Subjectivity ===
    Pulse pulse;
    Identity* identity;  // Pointer to shared identity

    // === Gravity (attention anchors) ===
    char gravity_centers[MAX_GRAVITY_CENTERS][MAX_KEYWORD_LEN];
    int n_gravity;

    // === Hot Words (from cooccurrence) ===
    char hot_words[MAX_KEYWORDS][MAX_KEYWORD_LEN];
    int n_hot;

    // === Surface Keywords (from recent context) ===
    char surface_keywords[MAX_KEYWORDS][MAX_KEYWORD_LEN];
    int n_surface;

    // === Spiral Topics (from overthinking) ===
    char spiral_topics[MAX_SPIRAL_TOPICS][MAX_KEYWORD_LEN];
    int n_spiral;

    // === State Metrics ===
    int overthink_depth;        // Recursive thought depth
    float body_tension;         // Somatic tension [0,1]
    float body_boredom;         // Boredom level [0,1]
    float drift_momentum;       // Direction change rate [0,1]

    // === Expert State ===
    int active_expert;          // 0=structural, 1=semantic, 2=creative, 3=precise
    float expert_temperature;   // Suggested temperature

} StanleySignals;

// ============================================================
// Attention Bias Weights (from Stanley's guided_attention.py)
// ============================================================

typedef struct {
    float gravity_weight;       // 1.0 - strongest anchor
    float memory_weight;        // 0.7 - medium persistence
    float cooccur_weight;       // 0.8 - contextual relevance
    float overthink_weight;     // 0.6 - reflection depth
    float surface_weight;       // 0.5 - recent context
} BiasWeights;

// Default weights
static const BiasWeights DEFAULT_BIAS_WEIGHTS = {
    .gravity_weight = 1.0f,
    .memory_weight = 0.7f,
    .cooccur_weight = 0.8f,
    .overthink_weight = 0.6f,
    .surface_weight = 0.5f
};

// ============================================================
// Attention Bias Computer
// ============================================================

typedef struct {
    BiasWeights weights;

    // Aggregated keywords (merged from all sources)
    char all_keywords[MAX_KEYWORDS * 4][MAX_KEYWORD_LEN];
    float keyword_weights[MAX_KEYWORDS * 4];
    int n_keywords;

    // Token-level bias (computed per generation)
    float* token_bias;      // [vocab_size] bias for each token
    int vocab_size;

    // Modulation
    float arousal_modulation;   // Boost from pulse arousal
    float focus_factor;         // Boost from overthink depth
} AttentionBias;

// ============================================================
// Function Declarations
// ============================================================

// === Pulse computation ===
void init_pulse(Pulse* p);
void compute_pulse(Pulse* p, const char* text, int len, Identity* identity);
float pulse_to_temperature(Pulse* p, float base_temp);

// === Identity management ===
void init_identity(Identity* id);
void add_gravity_center(Identity* id, const char* word);
void add_warm_word(Identity* id, const char* word);
void add_cold_word(Identity* id, const char* word);
int is_warm_word(Identity* id, const char* word);
int is_cold_word(Identity* id, const char* word);

// === Stanley Signals ===
void init_stanley_signals(StanleySignals* sig);
void extract_stanley_signals(StanleySignals* sig, int* tokens, int n_tokens,
                             float* hidden_states, Identity* identity);
void add_hot_word(StanleySignals* sig, const char* word);
void add_spiral_topic(StanleySignals* sig, const char* topic);
void update_overthink_depth(StanleySignals* sig, int depth);

// === Attention Bias ===
void init_attention_bias(AttentionBias* bias, int vocab_size);
void free_attention_bias(AttentionBias* bias);
void aggregate_keywords(AttentionBias* bias, StanleySignals* sig);
void compute_token_bias(AttentionBias* bias, StanleySignals* sig);
float get_token_bias(AttentionBias* bias, int token_id);
void apply_bias_to_logits(AttentionBias* bias, float* logits, int vocab_size);

// === Steering Prompt ===
void generate_steering_prompt(char* buffer, int max_len, StanleySignals* sig);

// === Expert Selection ===
int select_expert(StanleySignals* sig);
float expert_temperature(int expert_id, float base_temp);

// === Overthinking Detection ===
typedef struct {
    // Tracking for repetition detection
    char recent_words[32][MAX_KEYWORD_LEN];
    int n_recent_words;
    int word_repeat_count;

    // Spiral detection
    float repetition_score;     // How much text is repeating [0,1]
    float abstraction_score;    // How abstract/unfocused [0,1]
    float self_ref_score;       // Self-reference level [0,1]

    // Thresholds
    float spiral_threshold;     // When to trigger spiral mode
    float break_threshold;      // When to force topic change
} OverthinkDetector;

void init_overthink_detector(OverthinkDetector* od);
void detect_overthinking(OverthinkDetector* od, StanleySignals* sig,
                        const char* text, int len);
int should_break_spiral(OverthinkDetector* od);
void add_to_spiral(StanleySignals* sig, const char* topic);

#endif // GUIDED_H
