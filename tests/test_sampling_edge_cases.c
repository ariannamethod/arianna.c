/*
 * test_sampling_edge_cases.c — Edge case tests for sampling functions
 *
 * Tests for temperature=0, temperature=inf, top_p edge cases
 * These tests verify the fixes for CRITICAL bugs found in audit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "../src/arianna.h"

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  ✗ FAILED: %s\n", msg); \
        tests_failed++; \
        return; \
    } else { \
        printf("  ✓ %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("\n[TEST] %s\n", #test_func); \
    test_func(); \
} while(0)

// ============================================================
// HELPER: Create a mock transformer with logits
// ============================================================

static Transformer* create_mock_transformer(int vocab_size) {
    Transformer* t = (Transformer*)calloc(1, sizeof(Transformer));
    t->config.vocab_size = vocab_size;
    t->state.logits = (float*)calloc(vocab_size, sizeof(float));

    // Initialize with reasonable logits
    for (int i = 0; i < vocab_size; i++) {
        t->state.logits[i] = (float)i / vocab_size;
    }

    return t;
}

static void free_mock_transformer(Transformer* t) {
    if (t) {
        free(t->state.logits);
        free(t);
    }
}

// ============================================================
// TEMPERATURE EDGE CASE TESTS
// ============================================================

void test_sample_temperature_zero() {
    Transformer* t = create_mock_transformer(84);

    // Set up logits so one token is clearly better
    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (i == 42) ? 10.0f : -10.0f;
    }

    // Temperature = 0 should NOT crash (was a division by zero bug)
    int token = sample(t, 0.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Temperature 0: should return valid token");

    free_mock_transformer(t);
}

void test_sample_temperature_very_small() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (i == 42) ? 10.0f : -10.0f;
    }

    // Very small temperature (near zero)
    int token = sample(t, 1e-10f);
    TEST_ASSERT(token >= 0 && token < 84, "Temperature 1e-10: should return valid token");

    free_mock_transformer(t);
}

void test_sample_temperature_negative() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (float)i;
    }

    // Negative temperature (pathological input)
    int token = sample(t, -1.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Temperature -1: should return valid token");

    free_mock_transformer(t);
}

void test_sample_temperature_very_high() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (i == 42) ? 100.0f : 0.0f;
    }

    // Very high temperature = more uniform distribution
    int token = sample(t, 100.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Temperature 100: should return valid token");

    free_mock_transformer(t);
}

// ============================================================
// TOP-P EDGE CASE TESTS
// ============================================================

void test_sample_top_p_temperature_zero() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (float)i;
    }

    // Temperature = 0 with top-p
    int token = sample_top_p(t, 0.0f, 0.9f);
    TEST_ASSERT(token >= 0 && token < 84, "Top-p with temp=0: should return valid token");

    free_mock_transformer(t);
}

void test_sample_top_p_zero() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (i == 50) ? 10.0f : 0.0f;
    }

    // top_p = 0 (should still work, maybe just return top token)
    int token = sample_top_p(t, 1.0f, 0.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Top-p=0: should return valid token");

    free_mock_transformer(t);
}

void test_sample_top_p_one() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (float)i;
    }

    // top_p = 1.0 (use all tokens)
    int token = sample_top_p(t, 1.0f, 1.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Top-p=1: should return valid token");

    free_mock_transformer(t);
}

void test_sample_top_p_greater_than_one() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = (float)i;
    }

    // top_p > 1 (pathological input)
    int token = sample_top_p(t, 1.0f, 2.0f);
    TEST_ASSERT(token >= 0 && token < 84, "Top-p=2: should return valid token");

    free_mock_transformer(t);
}

// ============================================================
// LOGITS EDGE CASE TESTS
// ============================================================

void test_sample_all_zero_logits() {
    Transformer* t = create_mock_transformer(84);

    // All zero logits
    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = 0.0f;
    }

    int token = sample(t, 1.0f);
    TEST_ASSERT(token >= 0 && token < 84, "All zero logits: should return valid token");

    free_mock_transformer(t);
}

void test_sample_all_negative_logits() {
    Transformer* t = create_mock_transformer(84);

    // All very negative logits
    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = -100.0f;
    }

    int token = sample(t, 1.0f);
    TEST_ASSERT(token >= 0 && token < 84, "All negative logits: should return valid token");

    free_mock_transformer(t);
}

void test_sample_inf_logit() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = 0.0f;
    }
    t->state.logits[42] = INFINITY;

    int token = sample(t, 1.0f);
    // With INF logit, softmax should give 1.0 to that token
    TEST_ASSERT(token >= 0 && token < 84, "INF logit: should return valid token");

    free_mock_transformer(t);
}

void test_sample_nan_logit() {
    Transformer* t = create_mock_transformer(84);

    for (int i = 0; i < 84; i++) {
        t->state.logits[i] = 1.0f;
    }
    t->state.logits[42] = NAN;

    // This is a pathological case - behavior may be undefined
    // but should at least not crash
    int token = sample(t, 1.0f);
    (void)token; // May be undefined, just check no crash
    printf("  ✓ NAN logit: did not crash\n");
    tests_passed++;
}

// ============================================================
// VOCAB SIZE EDGE CASES
// ============================================================

void test_sample_vocab_size_one() {
    Transformer* t = create_mock_transformer(1);
    t->state.logits[0] = 5.0f;

    int token = sample(t, 1.0f);
    TEST_ASSERT(token == 0, "Vocab size 1: should return token 0");

    free_mock_transformer(t);
}

void test_sample_vocab_size_large() {
    // Test with vocab > 256 (was a buffer overflow bug in top-p)
    Transformer* t = create_mock_transformer(1024);

    for (int i = 0; i < 1024; i++) {
        t->state.logits[i] = (float)(i % 100) / 100.0f;
    }

    int token = sample_top_p(t, 1.0f, 0.9f);
    TEST_ASSERT(token >= 0 && token < 1024, "Vocab 1024 (was buffer overflow): should work");

    free_mock_transformer(t);
}

void test_sample_vocab_size_very_large() {
    // Even larger vocab
    Transformer* t = create_mock_transformer(4096);

    for (int i = 0; i < 4096; i++) {
        t->state.logits[i] = (float)(i % 100) / 100.0f;
    }

    int token = sample_top_p(t, 1.0f, 0.9f);
    TEST_ASSERT(token >= 0 && token < 4096, "Vocab 4096: should work");

    free_mock_transformer(t);
}

// ============================================================
// MAIN
// ============================================================

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  SAMPLING EDGE CASE TESTS\n");
    printf("  Testing CRITICAL bug fixes from Opus audit\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // Initialize RNG
    srand(42);

    // Temperature edge cases
    RUN_TEST(test_sample_temperature_zero);
    RUN_TEST(test_sample_temperature_very_small);
    RUN_TEST(test_sample_temperature_negative);
    RUN_TEST(test_sample_temperature_very_high);

    // Top-p edge cases
    RUN_TEST(test_sample_top_p_temperature_zero);
    RUN_TEST(test_sample_top_p_zero);
    RUN_TEST(test_sample_top_p_one);
    RUN_TEST(test_sample_top_p_greater_than_one);

    // Logits edge cases
    RUN_TEST(test_sample_all_zero_logits);
    RUN_TEST(test_sample_all_negative_logits);
    RUN_TEST(test_sample_inf_logit);
    RUN_TEST(test_sample_nan_logit);

    // Vocab size edge cases
    RUN_TEST(test_sample_vocab_size_one);
    RUN_TEST(test_sample_vocab_size_large);
    RUN_TEST(test_sample_vocab_size_very_large);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
