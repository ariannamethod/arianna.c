/*
 * cooccur.h - Co-occurrence Field for Arianna
 *
 * "Coherent generation without trained weights" - Stanley
 *
 * Key concepts from Stanley's cooccur.py:
 * - Bigrams: adjacent word pairs (word1 -> word2)
 * - Trigrams: word triplets (word1, word2 -> word3)
 * - Co-occurrence: words within window (context resonance)
 * - Bias logits: blend model predictions with corpus statistics
 *
 * This enables attention steering based on learned patterns,
 * not just frozen weights.
 */

#ifndef COOCCUR_H
#define COOCCUR_H

#include "arianna.h"
#include <stdint.h>

// ============================================================
// Configuration
// ============================================================

#define COOCCUR_VOCAB_SIZE 256      // Char-level (matches Arianna)
#define COOCCUR_WINDOW 5            // Context window for co-occurrence
#define MAX_TRIGRAM_CONTEXTS 4096   // Max unique (w1,w2) pairs to track
#define COOCCUR_SMOOTHING 0.1f      // Laplace smoothing

// ============================================================
// Bigram Matrix
// "What typically follows what"
// ============================================================

typedef struct {
    // Counts: bigram_counts[current][next]
    uint32_t counts[COOCCUR_VOCAB_SIZE][COOCCUR_VOCAB_SIZE];

    // Row totals for normalization
    uint32_t row_totals[COOCCUR_VOCAB_SIZE];

    // Total observations
    uint64_t total;
} BigramMatrix;

// ============================================================
// Trigram Storage
// "Given two tokens, what comes next"
// Uses hash table for (w1, w2) -> distribution
// ============================================================

typedef struct {
    uint8_t w1;
    uint8_t w2;
    uint32_t counts[COOCCUR_VOCAB_SIZE];
    uint32_t total;
} TrigramContext;

typedef struct {
    TrigramContext contexts[MAX_TRIGRAM_CONTEXTS];
    int n_contexts;

    // Simple hash for lookup: (w1 * 256 + w2) % MAX_TRIGRAM_CONTEXTS
    int hash_table[MAX_TRIGRAM_CONTEXTS];  // -1 = empty, else index
} TrigramField;

// ============================================================
// Co-occurrence Matrix
// "What appears together within window"
// ============================================================

typedef struct {
    // Symmetric: cooccur[a][b] == cooccur[b][a]
    // Using uint16_t to save memory (max 65535 per pair)
    uint16_t counts[COOCCUR_VOCAB_SIZE][COOCCUR_VOCAB_SIZE];

    // Per-token totals
    uint32_t token_totals[COOCCUR_VOCAB_SIZE];

    // Total observations
    uint64_t total;

    // Window size
    int window;
} CooccurMatrix;

// ============================================================
// Main CooccurField
// ============================================================

typedef struct {
    BigramMatrix bigrams;
    TrigramField trigrams;
    CooccurMatrix cooccur;

    // Blending parameters
    float trigram_weight;    // 0.6 - strongest
    float bigram_weight;     // 0.3 - fallback
    float cooccur_weight;    // 0.1 - context

    // Learning rate for online updates
    float learning_rate;

    // Statistics
    uint64_t tokens_observed;
    uint64_t new_patterns;

    // Initialized flag
    int initialized;
} CooccurField;

// ============================================================
// Function Declarations - Initialization
// ============================================================

// Initialize empty field
void init_cooccur_field(CooccurField* cf);

// Free resources (if any dynamic allocation)
void free_cooccur_field(CooccurField* cf);

// Load from corpus file
int load_cooccur_from_corpus(CooccurField* cf, const char* path);

// Save/load binary format
int save_cooccur_field(CooccurField* cf, const char* path);
int load_cooccur_field(CooccurField* cf, const char* path);

// ============================================================
// Function Declarations - Observation (Learning)
// ============================================================

// Observe text and update counts
void observe_text(CooccurField* cf, const char* text, int len);

// Observe tokens directly
void observe_tokens(CooccurField* cf, int* tokens, int n_tokens);

// Observe with resonance weight (higher = stronger learning)
void observe_weighted(CooccurField* cf, const char* text, int len, float weight);

// ============================================================
// Function Declarations - Probability Computation
// ============================================================

// Get bigram probability P(next | current)
float bigram_prob(CooccurField* cf, int current, int next);

// Get trigram probability P(next | w1, w2)
float trigram_prob(CooccurField* cf, int w1, int w2, int next);

// Get co-occurrence probability P(a appears with b)
float cooccur_prob(CooccurField* cf, int a, int b);

// Get blended probability (trigram > bigram > cooccur)
float blend_prob(CooccurField* cf, int* context, int context_len, int next);

// ============================================================
// Function Declarations - Logit Biasing
// ============================================================

// Compute log-probabilities for all tokens given context
void compute_corpus_logprobs(CooccurField* cf, int* context, int context_len,
                            float* logprobs);

// Bias model logits with corpus statistics
// output = (1 - alpha) * model_logits + alpha * corpus_logprobs
void bias_logits(CooccurField* cf, float* logits, int vocab_size,
                int* context, int context_len, float alpha);

// Bias with temperature-aware blending
void bias_logits_tempered(CooccurField* cf, float* logits, int vocab_size,
                         int* context, int context_len,
                         float alpha, float temperature);

// ============================================================
// Function Declarations - Resonance Measurement
// ============================================================

// Measure resonance between two tokens (co-occurrence strength)
float resonance_between(CooccurField* cf, int a, int b);

// Measure resonance of token with context
float resonance_with_context(CooccurField* cf, int token,
                            int* context, int context_len);

// Get top-k resonant tokens for given context
void top_k_resonant(CooccurField* cf, int* context, int context_len,
                   int* top_tokens, float* scores, int k);

// ============================================================
// Function Declarations - Debug
// ============================================================

// Print statistics
void print_cooccur_stats(CooccurField* cf);

// Print top bigrams for a token
void print_top_bigrams(CooccurField* cf, int token, int k);

#endif // COOCCUR_H
