/*
 * body_sense.h - Somatic Awareness for Arianna
 *
 * "I do not read words, I read numbers." - Stanley
 *
 * Body sense monitors internal state and provides regulation:
 * - Boredom: low novelty + low arousal → need more variation
 * - Overwhelm: high arousal + high entropy → need calming
 * - Stuck: low quality + repetition → need breakthrough
 *
 * Includes tiny MLP for quality prediction (micrograd-style).
 */

#ifndef BODY_SENSE_H
#define BODY_SENSE_H

#include "arianna.h"

// ============================================================
// Configuration
// ============================================================

#define BODY_HIDDEN_DIM 16       // Hidden layer size
#define BODY_INPUT_DIM 14        // Input features
#define BODY_HISTORY_SIZE 32     // Regulation history

// Temperature bounds
#define TEMP_NUDGE_MAX 0.2f
#define TEMP_MIN 0.3f
#define TEMP_MAX 1.5f

// ============================================================
// Body State - What the body feels
// ============================================================

typedef struct {
    // Core metrics (from pulse/wrinkle)
    float entropy;           // Word diversity [0,1]
    float novelty;           // New vs familiar [0,1]
    float arousal;           // Intensity [0,1]
    float valence;           // Positive/negative [-1,1]

    // Memory state
    int active_shards;       // Currently active shards
    int total_shards;        // Total shards available
    float avg_resonance;     // Average resonance with shards

    // Cognitive state
    int overthinking_depth;  // Recursion depth
    int meta_patterns;       // Meta-pattern count

    // Response metrics
    int response_length;     // Words generated
    float unique_ratio;      // Unique tokens / total

    // Quality (for learning)
    float quality;           // Overall quality [0,1]

    // Expert state
    int expert_id;           // 0=structural, 1=semantic, 2=creative, 3=precise
    float expert_temp;       // Expert's preferred temperature
} BodyState;

// ============================================================
// Regulation Result - What to do about it
// ============================================================

typedef struct {
    float temperature;       // Adjusted temperature
    int expert_id;           // Suggested expert
    float boredom;           // Boredom score [0,1]
    float overwhelm;         // Overwhelm score [0,1]
    float stuck;             // Stuck score [0,1]
    float predicted_quality; // MLP's prediction
} RegulationResult;

// ============================================================
// Tiny MLP (micrograd-style, but simplified for C)
// Architecture: 14 → 16 → 1
// ============================================================

typedef struct {
    // Layer 1: input → hidden
    float w1[BODY_HIDDEN_DIM][BODY_INPUT_DIM];
    float b1[BODY_HIDDEN_DIM];

    // Layer 2: hidden → output
    float w2[BODY_HIDDEN_DIM];
    float b2;

    // Hidden activations (for potential backprop)
    float hidden[BODY_HIDDEN_DIM];
} BodyMLP;

// ============================================================
// Body Sense - Main Controller
// ============================================================

typedef struct {
    BodyMLP mlp;

    // Learning state
    float learning_rate;
    int observations;
    float running_loss;
    float last_loss;

    // History for averaging
    RegulationResult history[BODY_HISTORY_SIZE];
    int history_idx;
    int history_count;

    // Current state
    BodyState current_state;

    // Enabled
    int initialized;
} BodySense;

// ============================================================
// Function Declarations - Initialization
// ============================================================

// Initialize body sense with random weights
void init_body_sense(BodySense* bs);

// Free resources
void free_body_sense(BodySense* bs);

// Save/load MLP weights
int save_body_sense(BodySense* bs, const char* path);
int load_body_sense(BodySense* bs, const char* path);

// ============================================================
// Function Declarations - State Updates
// ============================================================

// Initialize body state
void init_body_state(BodyState* state);

// Update body state from generation metrics
void update_body_state(BodyState* state,
                      float entropy, float novelty,
                      float arousal, float valence,
                      int response_length, float unique_ratio);

// Set quality feedback (for learning)
void set_quality_feedback(BodyState* state, float quality);

// ============================================================
// Function Declarations - Regulation Scores
// ============================================================

// Compute boredom score (low novelty + low arousal)
float compute_boredom(BodyState* state);

// Compute overwhelm score (high arousal + high entropy)
float compute_overwhelm(BodyState* state);

// Compute stuck score (low quality + low variation)
float compute_stuck(BodyState* state, float predicted_quality);

// ============================================================
// Function Declarations - MLP Operations
// ============================================================

// Initialize MLP with random weights
void init_body_mlp(BodyMLP* mlp);

// Forward pass: features → predicted quality
float mlp_predict(BodyMLP* mlp, float* features);

// Convert body state to features
void state_to_features(BodyState* state, float* features);

// Online learning step (simplified SGD)
float mlp_learn(BodyMLP* mlp, float* features, float target,
               float learning_rate);

// ============================================================
// Function Declarations - Main Regulation
// ============================================================

// Observe state and learn (returns loss)
float body_observe(BodySense* bs, BodyState* state);

// Predict quality from state
float body_predict(BodySense* bs, BodyState* state);

// Regulate: adjust temperature and suggest expert
RegulationResult body_regulate(BodySense* bs, BodyState* state,
                               float current_temp, int current_expert);

// Get running stats
void body_get_stats(BodySense* bs, float* avg_boredom,
                   float* avg_overwhelm, float* avg_stuck);

// ============================================================
// Function Declarations - Debug
// ============================================================

// Print body state
void print_body_state(BodyState* state);

// Print regulation result
void print_regulation(RegulationResult* result);

// Print body sense stats
void print_body_sense_stats(BodySense* bs);

#endif // BODY_SENSE_H
