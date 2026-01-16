/*
 * mathbrain.h - Arithmetic through resonance, not memorization
 *
 * From Stanley: numbers have "texture", operations emerge from patterns
 *
 * Key insight: 7 + 5 = 12 not because we memorized it,
 * but because 7 and 5 resonate at a frequency that IS 12.
 *
 * Architecture:
 *   - Number embeddings: each digit 0-99 has a learned "feel"
 *   - Operation heads: +, -, *, / as attention patterns
 *   - Resonance field: numbers that often appear together pull each other
 *   - Quality feedback: correct answers strengthen, wrong weaken
 */

#ifndef MATHBRAIN_H
#define MATHBRAIN_H

#include "arianna.h"

// ============================================================
// Configuration
// ============================================================

#define MATH_DIM 32              // Embedding dimension for numbers
#define MATH_MAX_NUM 100         // 0-99 for now
#define MATH_HIDDEN 64           // Hidden layer for operations
#define MATH_HISTORY 32          // Recent computations for patterns

// ============================================================
// Number embeddings - "texture" of each digit
// ============================================================

typedef struct {
    float embeddings[MATH_MAX_NUM][MATH_DIM];   // Learned number feel
    float magnitude_bias[MATH_DIM];              // Bias for "bigness"
    float parity_bias[MATH_DIM];                 // Bias for even/odd
} NumberField;

// ============================================================
// Operation heads - attention patterns for +, -, *, /
// ============================================================

typedef enum {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
    OP_COUNT = 4
} MathOp;

typedef struct {
    // Each operation is a learned bilinear form:
    // result_embed = W_op @ (a_embed ⊗ b_embed)
    float W[OP_COUNT][MATH_DIM][MATH_DIM];       // Operation weights
    float b[OP_COUNT][MATH_DIM];                  // Operation biases

    // Temperature per operation (some need precision, others creativity)
    float temperature[OP_COUNT];
} OperationHeads;

// ============================================================
// Resonance field - co-occurrence of number pairs
// ============================================================

typedef struct {
    // Pair frequencies: how often do (a, b) appear in same context?
    float pair_freq[MATH_MAX_NUM][MATH_MAX_NUM];

    // Result associations: which results follow which pairs?
    float result_assoc[MATH_MAX_NUM][MATH_MAX_NUM][MATH_MAX_NUM];  // sparse!

    // Total observations for normalization
    int total_observations;
} ResonanceField;

// ============================================================
// Computation history - pattern detection
// ============================================================

typedef struct {
    int a;
    int b;
    MathOp op;
    int result;
    int correct;  // 1 if ground truth matched
    float confidence;
} MathEvent;

typedef struct {
    MathEvent history[MATH_HISTORY];
    int head;
    int count;

    // Running stats
    float accuracy_ema;       // Exponential moving average of correctness
    float confidence_ema;     // Average confidence
    int total_computed;
} MathHistory;

// ============================================================
// MathBrain - complete arithmetic system
// ============================================================

typedef struct {
    NumberField numbers;
    OperationHeads ops;
    ResonanceField resonance;
    MathHistory history;

    // MLP for combining embeddings
    float hidden_w[MATH_DIM * 2][MATH_HIDDEN];  // Concat a,b -> hidden
    float hidden_b[MATH_HIDDEN];
    float output_w[MATH_HIDDEN][MATH_DIM];       // Hidden -> result embed
    float output_b[MATH_DIM];

    // Learning rate and momentum
    float lr;
    float momentum;

    // Cached activations for learning
    float last_a_embed[MATH_DIM];
    float last_b_embed[MATH_DIM];
    float last_hidden[MATH_HIDDEN];
    float last_output[MATH_DIM];
    int last_predicted;
} MathBrain;

// ============================================================
// Core functions
// ============================================================

// Initialize MathBrain with sensible defaults
void init_mathbrain(MathBrain* mb);

// Free MathBrain resources
void free_mathbrain(MathBrain* mb);

// Get number embedding
void get_number_embedding(MathBrain* mb, int n, float* out);

// Compute operation: a op b = ?
// Returns predicted result and confidence
int compute_op(MathBrain* mb, int a, MathOp op, int b, float* confidence);

// Learn from feedback (correct_result is ground truth)
void mathbrain_learn(MathBrain* mb, int a, MathOp op, int b, int correct_result);

// Update resonance field from observation
void observe_math(MathBrain* mb, int a, int b, int result);

// ============================================================
// Text parsing and generation
// ============================================================

// Parse arithmetic expression from text: "7 + 5" -> (7, ADD, 5)
int parse_math_expr(const char* text, int* a, MathOp* op, int* b);

// Generate result text: 12 -> "12"
void result_to_text(int result, char* out, int max_len);

// Full computation: "7 + 5" -> "12" with learning
int compute_from_text(MathBrain* mb, const char* expr, char* result, int max_len);

// ============================================================
// Stats and debugging
// ============================================================

// Print MathBrain stats
void print_mathbrain_stats(MathBrain* mb);

// Get accuracy over recent history
float get_recent_accuracy(MathBrain* mb, int window);

// Visualize number embedding similarity
void print_number_similarities(MathBrain* mb);

// ============================================================
// Save/Load
// ============================================================

int save_mathbrain(MathBrain* mb, const char* path);
int load_mathbrain(MathBrain* mb, const char* path);

#endif // MATHBRAIN_H
