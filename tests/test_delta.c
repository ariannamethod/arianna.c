/*
 * test_delta.c — Tests for delta.c (experience shards, microtraining)
 *
 * Tests low-rank delta operations, shard I/O, microtrainer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../src/delta.h"

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
// SIGNAL TESTS
// ============================================================

void test_signals_init() {
    Signals sig;
    init_signals(&sig);

    TEST_ASSERT(sig.arousal == 0.5f, "Initial arousal should be 0.5");
    TEST_ASSERT(sig.entropy == 0.5f, "Initial entropy should be 0.5");
    TEST_ASSERT(sig.tension == 0.3f, "Initial tension should be 0.3");
    TEST_ASSERT(sig.warmth == 0.6f, "Initial warmth should be 0.6");
    TEST_ASSERT(sig.focus == 0.5f, "Initial focus should be 0.5");
}

void test_signals_extract_empty() {
    Signals sig;
    int tokens[1] = {0};

    extract_signals(&sig, tokens, 0, NULL);

    // Should not crash with empty input
    TEST_ASSERT(sig.arousal >= 0.0f && sig.arousal <= 1.0f, "Arousal in valid range");
}

void test_signals_extract_exclamation() {
    Signals sig;
    int tokens[5] = {'h', 'e', 'l', 'l', '!'};

    extract_signals(&sig, tokens, 5, NULL);

    TEST_ASSERT(sig.arousal > 0.0f, "Exclamation should increase arousal");
}

void test_signals_extract_question() {
    Signals sig;
    int tokens[5] = {'w', 'h', 'a', 't', '?'};

    extract_signals(&sig, tokens, 5, NULL);

    TEST_ASSERT(sig.tension > 0.0f, "Question mark should increase tension");
}

// ============================================================
// DELTA BANK TESTS
// ============================================================

void test_delta_bank_init() {
    DeltaBank bank;
    init_delta_bank(&bank);

    TEST_ASSERT(bank.n_shards == 0, "Initial shard count should be 0");
    TEST_ASSERT(bank.cache_valid == 0, "Cache should be invalid initially");

    free_delta_bank(&bank);
}

void test_delta_bank_compute_mix() {
    DeltaBank bank;
    init_delta_bank(&bank);

    Signals sig;
    init_signals(&sig);

    // With no shards, compute_mix should not crash
    compute_mix(&bank, &sig);
    TEST_ASSERT(1, "compute_mix with no shards should not crash");

    free_delta_bank(&bank);
}

// ============================================================
// MICROTRAINER TESTS
// ============================================================

void test_microtrainer_init() {
    MicroTrainer mt;
    init_microtrainer(&mt, 64);

    TEST_ASSERT(mt.learning_rate == 0.001f, "Default learning rate should be 0.001");
    TEST_ASSERT(mt.momentum == 0.9f, "Default momentum should be 0.9");
    TEST_ASSERT(mt.decay == 0.999f, "Default decay should be 0.999");
    TEST_ASSERT(mt.dim == 64, "Dim should be 64");
    TEST_ASSERT(mt.pre_trace != NULL, "Pre trace should be allocated");
    TEST_ASSERT(mt.post_trace != NULL, "Post trace should be allocated");

    free_microtrainer(&mt);
}

void test_microtrainer_free() {
    MicroTrainer mt;
    init_microtrainer(&mt, 64);
    free_microtrainer(&mt);

    // Should not crash
    TEST_ASSERT(1, "free_microtrainer should not crash");
}

void test_build_dy_from_probs() {
    MicroTrainer mt;
    init_microtrainer(&mt, 64);
    mt.vocab_size = 10;
    mt.push = 1.0f;
    mt.pull = 0.5f;
    mt.topk = 3;

    float probs[10] = {0.1f, 0.2f, 0.3f, 0.1f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f};
    float dy[10];

    build_dy_from_probs(&mt, dy, probs, 10, 2);

    // Target (index 2) should have positive push
    TEST_ASSERT(dy[2] > 0, "Target should have positive push");

    // Some non-targets should have negative pull
    int has_pull = 0;
    for (int i = 0; i < 10; i++) {
        if (i != 2 && dy[i] < 0) has_pull = 1;
    }
    TEST_ASSERT(has_pull, "Competitors should have negative pull");

    free_microtrainer(&mt);
}

void test_quality_weight() {
    // Test quality weighting function
    float w1 = compute_quality_weight(0.9f, 0.0f, 0.0f);  // High quality, not stuck, not bored
    float w2 = compute_quality_weight(0.1f, 0.0f, 0.0f);  // Low quality
    float w3 = compute_quality_weight(0.9f, 0.9f, 0.0f);  // High quality but stuck

    TEST_ASSERT(w1 > w2, "Higher quality should give higher weight");
    TEST_ASSERT(w1 > w3, "Being stuck should reduce weight");
}

void test_delta_norm() {
    // Create a simple delta
    LowRankDelta d;
    d.in_dim = 4;
    d.out_dim = 4;
    d.rank = 2;
    d.A = (float*)calloc(4 * 2, sizeof(float));
    d.B = (float*)calloc(2 * 4, sizeof(float));

    // Set some values
    d.A[0] = 1.0f;
    d.A[1] = 1.0f;
    d.B[0] = 1.0f;
    d.B[1] = 1.0f;

    float norm = get_delta_norm(&d);
    TEST_ASSERT(norm > 0.0f, "Delta norm should be positive");
    TEST_ASSERT(norm == 2.0f, "Delta norm should be sqrt(1+1+1+1) = 2");

    free(d.A);
    free(d.B);
}

void test_clamp_delta() {
    LowRankDelta d;
    d.in_dim = 4;
    d.out_dim = 4;
    d.rank = 2;
    d.A = (float*)calloc(4 * 2, sizeof(float));
    d.B = (float*)calloc(2 * 4, sizeof(float));

    // Set large values
    for (int i = 0; i < 8; i++) d.A[i] = 10.0f;
    for (int i = 0; i < 8; i++) d.B[i] = 10.0f;

    float norm_before = get_delta_norm(&d);
    clamp_delta(&d, 1.0f);
    float norm_after = get_delta_norm(&d);

    TEST_ASSERT(norm_after <= 1.0f + 0.001f, "Norm should be clamped to max_norm");
    TEST_ASSERT(norm_before > norm_after, "Clamping should reduce norm");

    free(d.A);
    free(d.B);
}

void test_soft_reset_delta() {
    LowRankDelta d;
    d.in_dim = 4;
    d.out_dim = 4;
    d.rank = 2;
    d.A = (float*)calloc(4 * 2, sizeof(float));
    d.B = (float*)calloc(2 * 4, sizeof(float));

    for (int i = 0; i < 8; i++) d.A[i] = 1.0f;
    for (int i = 0; i < 8; i++) d.B[i] = 1.0f;

    float norm_before = get_delta_norm(&d);
    soft_reset_delta(&d, 0.5f);
    float norm_after = get_delta_norm(&d);

    TEST_ASSERT(norm_after < norm_before, "Soft reset should reduce norm");
    TEST_ASSERT(fabs(norm_after - norm_before * 0.5f) < 0.001f, "Norm should be halved");

    free(d.A);
    free(d.B);
}

// ============================================================
// ACCUMULATOR TESTS
// ============================================================

void test_accumulator_init() {
    ExperienceAccumulator acc;
    init_accumulator(&acc, 64, 100);

    TEST_ASSERT(acc.dim == 64, "Dim should be 64");
    TEST_ASSERT(acc.vocab_size == 100, "Vocab size should be 100");
    TEST_ASSERT(acc.experience_count == 0, "Buffer should be empty");
    TEST_ASSERT(acc.mean_x != NULL, "x_buffer should be allocated");
    TEST_ASSERT(acc.baseline_probs != NULL, "baseline_probs should be allocated");

    free_accumulator(&acc);
}

void test_accumulator_tick() {
    ExperienceAccumulator acc;
    init_accumulator(&acc, 64, 100);

    acc.cooldown_remaining = 1.0f;
    accumulator_tick(&acc, 0.5f);
    TEST_ASSERT(acc.cooldown_remaining == 0.5f, "Cooldown should decrease");

    accumulator_tick(&acc, 1.0f);
    TEST_ASSERT(acc.cooldown_remaining == 0.0f, "Cooldown should not go negative");

    free_accumulator(&acc);
}

// ============================================================
// MAIN
// ============================================================

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DELTA MODULE TESTS\n");
    printf("  Testing experience shards, microtraining, accumulators\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // Signal tests
    RUN_TEST(test_signals_init);
    RUN_TEST(test_signals_extract_empty);
    RUN_TEST(test_signals_extract_exclamation);
    RUN_TEST(test_signals_extract_question);

    // Delta bank tests
    RUN_TEST(test_delta_bank_init);
    RUN_TEST(test_delta_bank_compute_mix);

    // Microtrainer tests
    RUN_TEST(test_microtrainer_init);
    RUN_TEST(test_microtrainer_free);
    RUN_TEST(test_build_dy_from_probs);
    RUN_TEST(test_quality_weight);

    // Delta operations
    RUN_TEST(test_delta_norm);
    RUN_TEST(test_clamp_delta);
    RUN_TEST(test_soft_reset_delta);

    // Accumulator tests
    RUN_TEST(test_accumulator_init);
    RUN_TEST(test_accumulator_tick);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
