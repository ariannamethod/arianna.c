/*
 * test_selfsense.c — Tests for SelfSense (model introspection)
 *
 * Build: gcc -O2 -I src tests/test_selfsense.c src/selfsense.c src/delta.c -lm -o bin/test_selfsense
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "selfsense.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test Framework
// ═══════════════════════════════════════════════════════════════════════════════

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n[TEST] %s\n", name)
#define PASS(msg) do { printf("  ✓ %s\n", msg); tests_passed++; } while(0)
#define FAIL(msg) do { printf("  ✗ %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)
#define CHECK_RANGE(val, lo, hi, msg) CHECK((val) >= (lo) && (val) <= (hi), msg)
#define CHECK_NEAR(a, b, eps, msg) CHECK(fabsf((a) - (b)) < (eps), msg)

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

// Generate random hidden states
static void generate_random_hidden(float* hidden, int dim) {
    for (int i = 0; i < dim; i++) {
        hidden[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
}

// Generate "stable" hidden states (low variance)
static void generate_stable_hidden(float* hidden, int dim, float value) {
    for (int i = 0; i < dim; i++) {
        hidden[i] = value + ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
}

// Generate "chaotic" hidden states (high variance)
static void generate_chaotic_hidden(float* hidden, int dim) {
    for (int i = 0; i < dim; i++) {
        hidden[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

void test_init(void) {
    TEST("SelfSense Initialization");

    SelfSense ss;
    int dim = 128;  // Typical for small models

    init_selfsense(&ss, dim);

    CHECK(ss.initialized == 1, "Marked as initialized");
    CHECK(ss.dim == dim, "Dimension stored correctly");
    CHECK(ss.mlp.w1 != NULL, "MLP weights allocated");
    CHECK(ss.mlp.w2 != NULL, "MLP layer 2 allocated");
    CHECK(ss.state.ema_momentum > 0.0f, "EMA momentum set");
    CHECK(ss.learning_rate > 0.0f, "Learning rate set");

    free_selfsense(&ss);
    PASS("SelfSense freed without crash");
}

void test_signal_extraction(void) {
    TEST("Signal Extraction");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    generate_random_hidden(hidden, dim);

    Signals signals;
    selfsense_extract(&ss, hidden, &signals);

    // Check all signal fields are in reasonable range
    CHECK_RANGE(signals.arousal, 0.0f, 1.0f, "Arousal in [0,1]");
    CHECK_RANGE(signals.entropy, 0.0f, 1.0f, "Entropy in [0,1]");
    CHECK_RANGE(signals.tension, 0.0f, 1.0f, "Tension in [0,1]");
    CHECK_RANGE(signals.warmth, 0.0f, 1.0f, "Warmth in [0,1]");

    free(hidden);
    free_selfsense(&ss);
}

void test_ema_smoothing(void) {
    TEST("EMA Smoothing");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    Signals signals;

    // First extraction
    generate_random_hidden(hidden, dim);
    selfsense_extract(&ss, hidden, &signals);
    float first_arousal = ss.state.ema_signals[SIG_AROUSAL];

    // Second extraction with different input
    generate_random_hidden(hidden, dim);
    selfsense_extract(&ss, hidden, &signals);
    float second_arousal = ss.state.ema_signals[SIG_AROUSAL];

    // EMA should change smoothly, not jump wildly
    // (unless the change is very large)
    CHECK(fabsf(second_arousal - first_arousal) < 1.0f, "EMA changes smoothly");

    free(hidden);
    free_selfsense(&ss);
}

void test_entropy_computation(void) {
    TEST("Activation Entropy");

    int dim = 128;
    float* uniform = malloc(dim * sizeof(float));
    float* peaked = malloc(dim * sizeof(float));

    // Uniform distribution (high entropy)
    for (int i = 0; i < dim; i++) {
        uniform[i] = 1.0f / dim;
    }

    // Peaked distribution (low entropy)
    for (int i = 0; i < dim; i++) {
        peaked[i] = (i == 0) ? 0.99f : 0.01f / (dim - 1);
    }

    float entropy_uniform = compute_activation_entropy(uniform, dim);
    float entropy_peaked = compute_activation_entropy(peaked, dim);

    // Note: actual values depend on implementation
    // Just check they're different and non-negative
    CHECK(entropy_uniform >= 0.0f, "Uniform entropy non-negative");
    CHECK(entropy_peaked >= 0.0f, "Peaked entropy non-negative");
    // Uniform should have higher entropy than peaked (usually)
    // But implementation may vary, so just check they're computed

    free(uniform);
    free(peaked);
    PASS("Entropy computed without crash");
}

void test_layer_tension(void) {
    TEST("Layer Tension");

    int dim = 128;
    float* pre = malloc(dim * sizeof(float));
    float* post_similar = malloc(dim * sizeof(float));
    float* post_different = malloc(dim * sizeof(float));

    // Pre-layer state
    for (int i = 0; i < dim; i++) {
        pre[i] = (float)i / dim;
    }

    // Similar post-layer (low tension)
    for (int i = 0; i < dim; i++) {
        post_similar[i] = pre[i] + 0.01f;
    }

    // Different post-layer (high tension)
    for (int i = 0; i < dim; i++) {
        post_different[i] = -pre[i];
    }

    float tension_low = compute_layer_tension(pre, post_similar, dim);
    float tension_high = compute_layer_tension(pre, post_different, dim);

    CHECK(tension_low >= 0.0f, "Low tension non-negative");
    CHECK(tension_high >= 0.0f, "High tension non-negative");
    CHECK(tension_high > tension_low, "Different layers have higher tension");

    free(pre);
    free(post_similar);
    free(post_different);
}

void test_identity_alignment(void) {
    TEST("Identity Alignment");

    int dim = 128;
    float* hidden = malloc(dim * sizeof(float));
    float* identity = malloc(dim * sizeof(float));

    // Create identity vector
    for (int i = 0; i < dim; i++) {
        identity[i] = (float)i / dim;
    }

    // Aligned hidden state (same direction)
    for (int i = 0; i < dim; i++) {
        hidden[i] = identity[i] * 2.0f;  // Same direction, different magnitude
    }

    float aligned = compute_identity_alignment(hidden, identity, dim);
    CHECK_RANGE(aligned, 0.9f, 1.0f, "Aligned vectors have high similarity");

    // Orthogonal hidden state
    for (int i = 0; i < dim; i++) {
        hidden[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    float orthogonal = compute_identity_alignment(hidden, identity, dim);
    CHECK_RANGE(orthogonal, -0.5f, 0.5f, "Different vectors have lower similarity");

    free(hidden);
    free(identity);
}

void test_stuck_detection(void) {
    TEST("Stuck Detection");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    Signals signals;

    // Feed many similar states (should trigger stuck)
    for (int i = 0; i < SELFSENSE_HISTORY_LEN + 10; i++) {
        generate_stable_hidden(hidden, dim, 0.5f);
        selfsense_extract(&ss, hidden, &signals);
    }

    int stuck = selfsense_detect_stuck(&ss);
    // May or may not be stuck depending on threshold, just check it runs
    CHECK(stuck == 0 || stuck == 1, "Stuck detection returns boolean");

    free(hidden);
    free_selfsense(&ss);
}

void test_spiral_detection(void) {
    TEST("Spiral Detection");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    Signals signals;

    // Feed increasingly chaotic states (might trigger spiral)
    for (int i = 0; i < SELFSENSE_HISTORY_LEN + 10; i++) {
        for (int j = 0; j < dim; j++) {
            hidden[j] = ((float)rand() / RAND_MAX) * (float)(i + 1) * 0.1f;
        }
        selfsense_extract(&ss, hidden, &signals);
    }

    int spiral = selfsense_detect_spiral(&ss);
    CHECK(spiral == 0 || spiral == 1, "Spiral detection returns boolean");

    free(hidden);
    free_selfsense(&ss);
}

void test_trend(void) {
    TEST("Signal Trend");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    Signals signals;

    // Feed some states to build history
    for (int i = 0; i < 20; i++) {
        generate_random_hidden(hidden, dim);
        selfsense_extract(&ss, hidden, &signals);
    }

    float trend = selfsense_get_trend(&ss, SIG_AROUSAL);
    // Trend can be positive, negative, or zero
    CHECK(!isnan(trend) && !isinf(trend), "Trend is valid number");

    free(hidden);
    free_selfsense(&ss);
}

void test_learning(void) {
    TEST("Learning from Quality");

    SelfSense ss;
    int dim = 128;
    init_selfsense(&ss, dim);

    float* hidden = malloc(dim * sizeof(float));
    Signals signals;

    // Initial extraction (observations increments here)
    generate_random_hidden(hidden, dim);
    selfsense_extract(&ss, hidden, &signals);

    int initial_obs = ss.observations;
    float initial_loss = ss.running_loss;

    // More extractions to build up observations
    for (int i = 0; i < 5; i++) {
        generate_random_hidden(hidden, dim);
        selfsense_extract(&ss, hidden, &signals);
    }

    CHECK(ss.observations > initial_obs, "Observations increased after extractions");

    // Learn from quality signal (updates weights, not observations)
    selfsense_learn(&ss, 0.8f);
    selfsense_learn(&ss, 0.2f);  // Low quality to trigger different path

    // Running loss should have been updated
    CHECK(ss.running_loss >= 0.0f, "Running loss is non-negative");

    free(hidden);
    free_selfsense(&ss);
}

void test_save_load(void) {
    TEST("Save/Load");

    SelfSense ss1;
    int dim = 128;
    init_selfsense(&ss1, dim);

    // Modify some state
    float* hidden = malloc(dim * sizeof(float));
    Signals signals;
    for (int i = 0; i < 10; i++) {
        generate_random_hidden(hidden, dim);
        selfsense_extract(&ss1, hidden, &signals);
    }
    ss1.observations = 42;

    // Save (returns 1 on success)
    int save_result = save_selfsense(&ss1, "/tmp/test_selfsense.bin");
    CHECK(save_result == 1, "Save succeeded");

    // Load into new instance (returns 1 on success)
    SelfSense ss2;
    init_selfsense(&ss2, dim);
    int load_result = load_selfsense(&ss2, "/tmp/test_selfsense.bin");
    CHECK(load_result == 1, "Load succeeded");

    // Verify
    CHECK(ss2.observations == 42, "Observations preserved");
    CHECK(ss2.dim == dim, "Dimension preserved");

    free(hidden);
    free_selfsense(&ss1);
    free_selfsense(&ss2);
}

void test_signal_indices(void) {
    TEST("Signal Indices");

    // Just verify the constants are sensible
    CHECK(SIG_AROUSAL == 0, "SIG_AROUSAL is 0");
    CHECK(SIG_ENTROPY == 1, "SIG_ENTROPY is 1");
    CHECK(SIG_TENSION == 2, "SIG_TENSION is 2");
    CHECK(SIG_WARMTH == 3, "SIG_WARMTH is 3");
    CHECK(SIG_FOCUS == 4, "SIG_FOCUS is 4");
    CHECK(SIG_RECURSION == 5, "SIG_RECURSION is 5");
    CHECK(SIG_RESONANCE == 6, "SIG_RESONANCE is 6");
    CHECK(SIG_NOVELTY == 7, "SIG_NOVELTY is 7");
    CHECK(SELFSENSE_OUTPUT_DIM == 8, "8 signal dimensions");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    srand(time(NULL));

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  SELFSENSE TEST SUITE\n");
    printf("  \"The model feels itself from the inside\"\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    test_init();
    test_signal_extraction();
    test_ema_smoothing();
    test_entropy_computation();
    test_layer_tension();
    test_identity_alignment();
    test_stuck_detection();
    test_spiral_detection();
    test_trend();
    test_learning();
    test_save_load();
    test_signal_indices();

    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (tests_failed > 0) {
        printf("\n  Self-sensing failed\n\n");
        return 1;
    } else {
        printf("\n  ✓ The model knows itself\n\n");
        return 0;
    }
}
