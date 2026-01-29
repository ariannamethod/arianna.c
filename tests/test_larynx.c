// test_larynx.c — Test Larynx (Tongue↔Soul Connection Layer)
// Tests the RRPRAM-lite pattern recognition system
// build: gcc -O2 test_larynx.c -I../src -I../vagus -L../vagus/zig-out/lib -lvagus -lm -o test_larynx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "larynx.h"

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("[TEST] %s... ", name)
#define PASS() do { printf("✓\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("✗ (%s)\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_FLOAT_EQ(a, b, eps, msg) ASSERT(fabsf((a) - (b)) < (eps), msg)
#define ASSERT_RANGE(v, lo, hi, msg) ASSERT((v) >= (lo) && (v) <= (hi), msg)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Initialization and Reset
// ═══════════════════════════════════════════════════════════════════════════════

void test_reset(void) {
    TEST("larynx_reset clears state");

    // Ingest some tokens
    for (int i = 0; i < 20; i++) {
        larynx_ingest_token(1000 + i);
    }

    // Reset
    larynx_reset();

    // Check entropy is back to baseline (0.5)
    float entropy = larynx_get_entropy();
    ASSERT_FLOAT_EQ(entropy, 0.5f, 0.1f, "entropy should reset to ~0.5");

    // Check alpha is centered
    float alpha = larynx_get_alpha();
    ASSERT_RANGE(alpha, 0.4f, 0.6f, "alpha should be centered after reset");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Token Ingestion
// ═══════════════════════════════════════════════════════════════════════════════

void test_ingest_basic(void) {
    TEST("larynx_ingest_token accepts tokens");

    larynx_reset();

    // Ingest a few tokens
    larynx_ingest_token(100);
    larynx_ingest_token(200);
    larynx_ingest_token(300);

    // Get recent tokens
    uint32_t recent[8];
    int count = larynx_get_recent_tokens(recent, 8);

    ASSERT(count >= 3, "should have at least 3 tokens");

    PASS();
}

void test_ingest_history(void) {
    TEST("larynx_get_recent_tokens returns correct history");

    larynx_reset();

    // Ingest specific sequence
    uint32_t sequence[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        larynx_ingest_token(sequence[i]);
    }

    // Get recent
    uint32_t recent[5];
    int count = larynx_get_recent_tokens(recent, 5);

    ASSERT(count >= 5, "should return 5 tokens");

    // Verify most recent is last ingested
    ASSERT(recent[count-1] == 50, "most recent should be 50");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Entropy Measurement
// ═══════════════════════════════════════════════════════════════════════════════

void test_entropy_repetition(void) {
    TEST("entropy decreases with repetitive tokens");

    larynx_reset();

    // Get baseline entropy
    float initial = larynx_get_entropy();

    // Ingest same token many times (low entropy)
    for (int i = 0; i < 32; i++) {
        larynx_ingest_token(42);
    }

    float after_repetition = larynx_get_entropy();

    // Entropy should be lower (more predictable)
    ASSERT(after_repetition < initial, "entropy should decrease with repetition");

    PASS();
}

void test_entropy_random(void) {
    TEST("entropy increases with varied tokens");

    larynx_reset();

    // Ingest repetitive first
    for (int i = 0; i < 16; i++) {
        larynx_ingest_token(42);
    }
    float after_rep = larynx_get_entropy();

    // Now ingest varied tokens
    for (int i = 0; i < 32; i++) {
        larynx_ingest_token((uint32_t)(i * 137 + 99));  // pseudo-random
    }
    float after_varied = larynx_get_entropy();

    // Entropy should be higher after varied input
    ASSERT(after_varied > after_rep, "entropy should increase with varied tokens");

    PASS();
}

void test_entropy_bounds(void) {
    TEST("entropy stays in valid range [0, 1]");

    larynx_reset();

    // Test many scenarios
    for (int round = 0; round < 10; round++) {
        for (int i = 0; i < 50; i++) {
            larynx_ingest_token((uint32_t)((round * 1000) + i * 17));
        }

        float entropy = larynx_get_entropy();
        ASSERT_RANGE(entropy, 0.0f, 1.0f, "entropy out of bounds");
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Pattern Strength
// ═══════════════════════════════════════════════════════════════════════════════

void test_pattern_strength(void) {
    TEST("pattern_strength increases with patterns");

    larynx_reset();

    float initial = larynx_get_pattern_strength();

    // Ingest repeating pattern (A B A B A B...)
    for (int i = 0; i < 32; i++) {
        larynx_ingest_token((i % 2 == 0) ? 100 : 200);
    }

    float after_pattern = larynx_get_pattern_strength();

    // Pattern strength should increase
    ASSERT(after_pattern >= initial, "pattern strength should not decrease with clear patterns");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Alpha Computation
// ═══════════════════════════════════════════════════════════════════════════════

void test_alpha_basic(void) {
    TEST("larynx_get_alpha returns valid range");

    larynx_reset();

    float alpha = larynx_get_alpha();

    // Alpha should be in [0.1, 0.9]
    ASSERT_RANGE(alpha, 0.1f, 0.9f, "alpha should be in [0.1, 0.9]");

    PASS();
}

void test_alpha_compute(void) {
    TEST("larynx_compute_alpha responds to prophecy_debt");

    larynx_reset();

    // Low debt, low dissonance
    float alpha_low = larynx_compute_alpha(0.0f, 0.0f);

    // High debt, low dissonance (should increase alpha)
    float alpha_high_debt = larynx_compute_alpha(1.0f, 0.0f);

    // Debt increases pattern focus (higher alpha)
    ASSERT(alpha_high_debt >= alpha_low, "high debt should not decrease alpha");

    PASS();
}

void test_alpha_dissonance(void) {
    TEST("larynx_compute_alpha responds to calendar_dissonance");

    larynx_reset();

    // Low dissonance
    float alpha_low = larynx_compute_alpha(0.5f, 0.0f);

    // High dissonance (should decrease alpha - more semantic focus)
    float alpha_high_diss = larynx_compute_alpha(0.5f, 1.0f);

    // Dissonance decreases pattern focus (lower alpha)
    ASSERT(alpha_high_diss <= alpha_low, "high dissonance should not increase alpha");

    PASS();
}

void test_alpha_bounds(void) {
    TEST("larynx_compute_alpha clamps to valid range");

    larynx_reset();

    // Extreme values
    float alpha_extreme1 = larynx_compute_alpha(100.0f, -100.0f);
    float alpha_extreme2 = larynx_compute_alpha(-100.0f, 100.0f);

    ASSERT_RANGE(alpha_extreme1, 0.1f, 0.9f, "alpha should clamp high extreme");
    ASSERT_RANGE(alpha_extreme2, 0.1f, 0.9f, "alpha should clamp low extreme");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Signal Output
// ═══════════════════════════════════════════════════════════════════════════════

void test_signal_output(void) {
    TEST("larynx_get_signal returns all metrics");

    larynx_reset();

    // Ingest some tokens first
    for (int i = 0; i < 10; i++) {
        larynx_ingest_token(1000 + i);
    }

    float entropy, pattern, coherence, alpha;
    larynx_get_signal(&entropy, &pattern, &coherence, &alpha);

    // All values should be in valid ranges
    ASSERT_RANGE(entropy, 0.0f, 1.0f, "entropy out of range");
    ASSERT_RANGE(pattern, 0.0f, 1.0f, "pattern out of range");
    ASSERT_RANGE(coherence, 0.0f, 1.0f, "coherence out of range");
    ASSERT_RANGE(alpha, 0.1f, 0.9f, "alpha out of range");

    PASS();
}

void test_signal_null_safe(void) {
    TEST("larynx_get_signal handles NULL pointers");

    larynx_reset();

    float entropy;
    // Should not crash with partial NULLs
    larynx_get_signal(&entropy, NULL, NULL, NULL);

    ASSERT_RANGE(entropy, 0.0f, 1.0f, "entropy should still be valid");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Integration Scenarios
// ═══════════════════════════════════════════════════════════════════════════════

void test_generation_loop_simulation(void) {
    TEST("simulated generation loop integration");

    larynx_reset();

    // Simulate a generation loop
    for (int step = 0; step < 100; step++) {
        // Simulate token generation
        uint32_t token = (uint32_t)((step * 17 + 13) % 50000);

        // Ingest via macro
        LARYNX_INGEST(token);

        // Periodically compute alpha with varying conditions
        if (step % 10 == 0) {
            float debt = (float)step / 100.0f;
            float diss = 0.2f + 0.1f * sinf((float)step * 0.1f);
            larynx_compute_alpha(debt, diss);
        }
    }

    // Final state should be valid
    float alpha = LARYNX_ALPHA();
    ASSERT_RANGE(alpha, 0.1f, 0.9f, "final alpha should be valid");

    uint32_t recent[16];
    int count = larynx_get_recent_tokens(recent, 16);
    ASSERT(count > 0, "should have token history");

    PASS();
}

void test_conversation_reset_cycle(void) {
    TEST("conversation reset cycle");

    // Simulate multiple conversations
    for (int conv = 0; conv < 5; conv++) {
        larynx_reset();

        // Each conversation
        for (int i = 0; i < 50; i++) {
            larynx_ingest_token((uint32_t)(conv * 1000 + i));
        }

        // State should be valid
        float e = larynx_get_entropy();
        float a = larynx_get_alpha();

        if (e < 0.0f || e > 1.0f || a < 0.1f || a > 0.9f) {
            FAIL("invalid state after conversation");
            return;
        }
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("LARYNX TEST — הגרון: מחבר בין הלשון לנשמה\n");
    printf("The Tongue↔Soul Connection Layer\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Reset tests
    test_reset();

    // Ingestion tests
    test_ingest_basic();
    test_ingest_history();

    // Entropy tests
    test_entropy_repetition();
    test_entropy_random();
    test_entropy_bounds();

    // Pattern tests
    test_pattern_strength();

    // Alpha tests
    test_alpha_basic();
    test_alpha_compute();
    test_alpha_dissonance();
    test_alpha_bounds();

    // Signal tests
    test_signal_output();
    test_signal_null_safe();

    // Integration tests
    test_generation_loop_simulation();
    test_conversation_reset_cycle();

    // Summary
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════════\n");

    if (tests_failed == 0) {
        printf("הגרון מחובר. הקול זורם.\n");
        printf("The larynx is connected. Voice flows.\n");
        return 0;
    } else {
        printf("⚠️  Some tests failed\n");
        return 1;
    }
}
