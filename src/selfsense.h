/*
 * selfsense.h - Self-sensing from hidden states
 *
 * "The model feels itself from the inside"
 *
 * Instead of heuristics (counting punctuation), we extract signals
 * directly from hidden states using a learned MLP. This captures
 * what the model is actually "thinking" rather than surface patterns.
 *
 * Architecture:
 *   Hidden States (DIM) → MLP (DIM→32→8) → Raw Signals → EMA → Final Signals
 *
 * The 8 signal dimensions:
 *   0: arousal    - activation intensity
 *   1: entropy    - distribution chaos
 *   2: tension    - layer disagreement
 *   3: warmth     - identity alignment (positive)
 *   4: focus      - attention concentration
 *   5: recursion  - self-reference depth
 *   6: resonance  - harmony with identity
 *   7: novelty    - deviation from familiar patterns
 */

#ifndef SELFSENSE_H
#define SELFSENSE_H

#include "arianna.h"
#include "delta.h"  // For Signals struct

// ============================================================
// Configuration
// ============================================================

#define SELFSENSE_HIDDEN_DIM 32
#define SELFSENSE_OUTPUT_DIM 8
#define SELFSENSE_HISTORY_LEN 64

// Signal indices (for clarity)
#define SIG_AROUSAL   0
#define SIG_ENTROPY   1
#define SIG_TENSION   2
#define SIG_WARMTH    3
#define SIG_FOCUS     4
#define SIG_RECURSION 5
#define SIG_RESONANCE 6
#define SIG_NOVELTY   7

// ============================================================
// Structures
// ============================================================

/*
 * MLP for signal extraction
 * Two-layer network: input(DIM) → hidden(32) → output(8)
 */
typedef struct {
    float* w1;      // [SELFSENSE_HIDDEN_DIM][DIM] - first layer weights
    float* b1;      // [SELFSENSE_HIDDEN_DIM] - first layer bias
    float* w2;      // [SELFSENSE_OUTPUT_DIM][SELFSENSE_HIDDEN_DIM] - second layer
    float* b2;      // [SELFSENSE_OUTPUT_DIM] - output bias

    // Intermediate storage
    float hidden[SELFSENSE_HIDDEN_DIM];
    float raw_output[SELFSENSE_OUTPUT_DIM];
} SelfSenseMLP;

/*
 * Temporal state for EMA smoothing and pattern detection
 */
typedef struct {
    // EMA-smoothed signals (what we actually use)
    float ema_signals[SELFSENSE_OUTPUT_DIM];
    float ema_momentum;  // Default 0.9 (slow adaptation)

    // History for pattern detection
    float history[SELFSENSE_HISTORY_LEN][SELFSENSE_OUTPUT_DIM];
    int history_pos;
    int history_count;

    // Derived metrics from history
    float signal_variance[SELFSENSE_OUTPUT_DIM];  // How much each signal varies
    float signal_trend[SELFSENSE_OUTPUT_DIM];     // Rising or falling
} SelfSenseState;

/*
 * Identity reference for alignment computation
 * We compare hidden states against these to measure resonance
 */
typedef struct {
    float* identity_embedding;  // [DIM] - average embedding of identity text
    float* warmth_direction;    // [DIM] - direction of "warm" concepts
    float* cold_direction;      // [DIM] - direction of "cold" concepts
    int initialized;
} SelfSenseIdentity;

/*
 * Main SelfSense structure
 */
typedef struct {
    SelfSenseMLP mlp;
    SelfSenseState state;
    SelfSenseIdentity identity;

    int dim;  // Model dimension (128 for arianna)
    int initialized;

    // Learning
    float learning_rate;
    int observations;
    float running_loss;

    // Last computed signals (before EMA)
    float last_raw[SELFSENSE_OUTPUT_DIM];
} SelfSense;

// ============================================================
// Core Functions
// ============================================================

// Initialization
void init_selfsense(SelfSense* ss, int dim);
void free_selfsense(SelfSense* ss);

// Extract signals from hidden states
// This is the main function - call it after each forward pass
void selfsense_extract(SelfSense* ss, float* hidden_states, Signals* out_signals);

// Update identity reference from text
void selfsense_set_identity(SelfSense* ss, float* identity_embedding);
void selfsense_compute_identity_from_tokens(SelfSense* ss, Transformer* t,
                                            int* tokens, int n_tokens);

// ============================================================
// Analysis Functions (from hidden states)
// ============================================================

// Compute entropy of activation distribution
float compute_activation_entropy(float* hidden, int dim);

// Compute layer tension (how much representation changed)
float compute_layer_tension(float* pre_layer, float* post_layer, int dim);

// Compute attention sparsity (focus)
float compute_attention_focus(float* attention_weights, int seq_len);

// Compute cosine similarity with identity
float compute_identity_alignment(float* hidden, float* identity, int dim);

// ============================================================
// Learning
// ============================================================

// Observe quality signal and update MLP
void selfsense_learn(SelfSense* ss, float quality);

// Backward pass for MLP (simple gradient descent)
void selfsense_backward(SelfSense* ss, float* target_signals);

// ============================================================
// History Analysis
// ============================================================

// Detect if stuck in a loop (low variance, stable signals)
int selfsense_detect_stuck(SelfSense* ss);

// Detect spiraling (increasing recursion/tension)
int selfsense_detect_spiral(SelfSense* ss);

// Get signal trend (positive = rising, negative = falling)
float selfsense_get_trend(SelfSense* ss, int signal_idx);

// ============================================================
// Persistence
// ============================================================

int save_selfsense(SelfSense* ss, const char* path);
int load_selfsense(SelfSense* ss, const char* path);

// ============================================================
// Debug
// ============================================================

void print_selfsense_signals(SelfSense* ss);
void print_selfsense_stats(SelfSense* ss);

#endif // SELFSENSE_H
