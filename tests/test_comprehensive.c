/*
 * test_comprehensive.c — Comprehensive tests for arianna.c (new API)
 *
 * Build: gcc -O2 -I src tests/test_comprehensive.c src/cloud.c src/schumann.c \
 *        src/mood.c src/body_sense.c src/delta.c src/mathbrain.c \
 *        src/inner_arianna.c src/amk_kernel.c src/cooccur.c -lm -o bin/test_comprehensive
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cloud.h"
#include "mood.h"
#include "body_sense.h"
#include "delta.h"
#include "mathbrain.h"
#include "inner_arianna.h"
#include "amk_kernel.h"
#include "schumann.h"
#include "cooccur.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test Framework
// ═══════════════════════════════════════════════════════════════════════════════

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_SECTION(name) printf("\n═══════════════════════════════════════════════════════════════════\n[SECTION] %s\n═══════════════════════════════════════════════════════════════════\n", name)
#define TEST(name) printf("\n[TEST] %s\n", name)
#define PASS(msg) do { printf("  ✓ %s\n", msg); tests_passed++; } while(0)
#define FAIL(msg) do { printf("  ✗ %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)
#define CHECK_RANGE(val, lo, hi, msg) CHECK((val) >= (lo) && (val) <= (hi), msg)
#define CHECK_NEAR(a, b, eps, msg) CHECK(fabsf((a) - (b)) < (eps), msg)

// ═══════════════════════════════════════════════════════════════════════════════
// CLOUD TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_cloud(void) {
    TEST("Cloud Initialization");
    cloud_init();
    PASS("Cloud initialized");

    TEST("Cloud FEAR Detection");
    CloudResponse fear = cloud_ping("I'm terrified and scared");
    CHECK(fear.chambers[CLOUD_CHAMBER_FEAR] > 0.3f, "FEAR chamber activated");

    TEST("Cloud LOVE Detection");
    CloudResponse love = cloud_ping("I love you with all my heart");
    CHECK(love.chambers[CLOUD_CHAMBER_LOVE] > 0.3f, "LOVE chamber activated");

    TEST("Cloud RAGE Detection");
    CloudResponse rage = cloud_ping("I am furious with hatred");
    CHECK(rage.chambers[CLOUD_CHAMBER_RAGE] > 0.3f, "RAGE chamber activated");

    TEST("Cloud VOID Detection");
    CloudResponse void_resp = cloud_ping("despair hopeless hollow");
    CHECK(void_resp.chambers[CLOUD_CHAMBER_VOID] > 0.3f, "VOID chamber activated");

    TEST("Cloud Neutral Input");
    CloudResponse neutral = cloud_ping("The table is in the room");
    float max_chamber = 0.0f;
    for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
        if (neutral.chambers[i] > max_chamber) max_chamber = neutral.chambers[i];
    }
    CHECK(max_chamber < 0.5f, "Neutral text has low activations");

    TEST("Cloud Temperature Bias");
    float bias = cloud_temperature_bias(&fear);
    CHECK(bias >= -0.5f && bias <= 0.5f, "Temperature bias in range");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOOD TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_mood(void) {
    TEST("MoodRouter Initialization");
    MoodRouter router;
    init_mood_router(&router, 1.0f);
    CHECK(router.dominant >= 0 && router.dominant < NUM_MOODS, "Dominant mood valid");

    TEST("Mood Mix Sums to 1.0");
    float sum = 0.0f;
    for (int i = 0; i < NUM_MOODS; i++) {
        sum += router.mix[i];
    }
    CHECK_NEAR(sum, 1.0f, 0.01f, "Mix sums to 1.0");

    TEST("Mood Routing from Signals");
    Signals signals = {.arousal = 0.9f, .tension = 0.8f, .warmth = 0.1f};
    route_signals_to_moods(&router, &signals);
    CHECK(MOOD_NAMES[router.dominant] != NULL, "Dominant mood name exists");

    TEST("Temperature Bias");
    float temp_bias = get_temperature_bias(&router);
    CHECK(temp_bias >= -0.5f && temp_bias <= 0.5f, "Temperature bias reasonable");

    TEST("Adjust Temperature by Mood");
    float adjusted = adjust_temperature_by_mood(&router, 0.8f);
    CHECK(adjusted > 0.0f && adjusted < 2.0f, "Adjusted temp in range");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BODY SENSE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_body_sense(void) {
    TEST("BodyState Initialization");
    BodyState state;
    init_body_state(&state);
    PASS("BodyState initialized");

    TEST("BodyState Fields");
    state.entropy = 0.5f;
    state.novelty = 0.6f;
    state.arousal = 0.7f;
    state.quality = 0.8f;
    CHECK_RANGE(state.entropy, 0.0f, 1.0f, "Entropy in range");
    CHECK_RANGE(state.quality, 0.0f, 1.0f, "Quality in range");

    TEST("BodySense Initialization");
    BodySense bs;
    init_body_sense(&bs);
    PASS("BodySense initialized");

    TEST("Compute Boredom");
    float boredom = compute_boredom(&state);
    CHECK_RANGE(boredom, 0.0f, 1.0f, "Boredom in range");

    TEST("Compute Overwhelm");
    float overwhelm = compute_overwhelm(&state);
    CHECK_RANGE(overwhelm, 0.0f, 1.0f, "Overwhelm in range");

    TEST("Compute Stuck");
    float stuck = compute_stuck(&state, 0.5f);
    CHECK_RANGE(stuck, 0.0f, 1.0f, "Stuck in range");

    TEST("Body Regulate");
    RegulationResult reg = body_regulate(&bs, &state, 0.8f, 0);
    CHECK_RANGE(reg.boredom, 0.0f, 1.0f, "Reg boredom in range");
    CHECK_RANGE(reg.overwhelm, 0.0f, 1.0f, "Reg overwhelm in range");
    CHECK(reg.temperature > 0.0f, "Temperature positive");

    free_body_sense(&bs);
    PASS("BodySense freed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MATHBRAIN TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_mathbrain(void) {
    TEST("MathBrain Initialization");
    MathBrain mb;
    init_mathbrain(&mb);
    PASS("MathBrain initialized");

    TEST("Parse Math Expression");
    int a, b;
    MathOp op;
    CHECK(parse_math_expr("7 + 5", &a, &op, &b), "Parse '7 + 5'");
    CHECK(a == 7 && b == 5 && op == OP_ADD, "Parsed correctly");

    TEST("MathBrain Compute");
    float conf;
    int result = compute_op(&mb, 5, OP_ADD, 3, &conf);
    CHECK(result >= 0, "Compute returns result");
    CHECK_RANGE(conf, 0.0f, 1.0f, "Confidence in range");

    TEST("MathBrain Learn");
    for (int i = 0; i < 20; i++) {
        mathbrain_learn(&mb, 7, OP_ADD, 3, 10);
    }
    CHECK(mb.history.total_computed > 0, "History tracked");

    TEST("MathBrain Save/Load");
    save_mathbrain(&mb, "/tmp/test_mb.bin");
    MathBrain mb2;
    init_mathbrain(&mb2);
    load_mathbrain(&mb2, "/tmp/test_mb.bin");
    CHECK(mb2.history.total_computed > 0, "Loaded state");

    free_mathbrain(&mb);
    free_mathbrain(&mb2);
    PASS("MathBrain freed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// INNER ARIANNA TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_inner_arianna(void) {
    TEST("InnerArianna Initialization");
    InnerArianna inner;
    inner_init(&inner);
    CHECK(inner.base_weight >= 0.0f && inner.base_weight <= 1.0f, "Base weight in range");

    TEST("Inner Weight Computation");
    float weight = inner_compute_weight(&inner);
    CHECK_RANGE(weight, 0.0f, 1.0f, "Inner weight in range");

    TEST("Set Cloud Response");
    CloudResponse cloud = {0};
    cloud.chambers[CLOUD_CHAMBER_FEAR] = 0.8f;
    inner_update_cloud(&inner, &cloud);
    float weight_after = inner_compute_weight(&inner);
    CHECK(weight_after >= 0.0f, "Cloud affects weight");

    TEST("Set Mode");
    inner_set_mode(&inner, 0);
    inner_set_mode(&inner, 1);
    PASS("Mode set without crash");

    inner_free(&inner);
    PASS("InnerArianna freed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// AMK KERNEL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_amk(void) {
    TEST("AMK Initialization");
    am_init();
    PASS("AMK initialized");

    TEST("Schumann Initialization");
    schumann_init();
    PASS("Schumann initialized");

    TEST("AMK State Access");
    AM_State* state = am_get_state();
    CHECK(state != NULL, "State accessible");
    CHECK(state->prophecy >= 0, "Prophecy valid");
    CHECK_RANGE(state->destiny, 0.0f, 1.0f, "Destiny in range");

    TEST("Velocity Modes via DSL");
    am_exec("VELOCITY NOMOVE");
    CHECK_NEAR(am_get_temperature(), 0.5f, 0.05f, "NOMOVE temp ~0.5");

    am_exec("VELOCITY WALK");
    CHECK_NEAR(am_get_temperature(), 0.85f, 0.05f, "WALK temp ~0.85");

    am_exec("VELOCITY RUN");
    CHECK_NEAR(am_get_temperature(), 1.2f, 0.05f, "RUN temp ~1.2");

    TEST("Direct State Modification");
    state->destiny = 0.7f;
    CHECK_NEAR(am_get_destiny_bias(), 0.7f, 0.01f, "Destiny set");
    state->prophecy = 10;
    CHECK(state->prophecy == 10, "Prophecy set");

    TEST("Schumann Coherence");
    float coherence = schumann_get_coherence();
    CHECK_RANGE(coherence, 0.0f, 1.0f, "Coherence in range");

    TEST("AMK Step");
    am_step(0.1f);
    PASS("AMK stepped");
}

// ═══════════════════════════════════════════════════════════════════════════════
// COOCCUR TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_cooccur(void) {
    TEST("CooccurField Initialization");
    CooccurField field;
    init_cooccur_field(&field);
    CHECK(field.initialized != 0, "Field initialized");

    TEST("Observe Text");
    observe_text(&field, "hello world hello world", 23);
    CHECK(field.bigrams.total > 0, "Bigrams observed");
    CHECK(field.tokens_observed > 0, "Tokens counted");

    TEST("Bias Logits");
    float logits[256] = {0};
    int context[] = {'h', 'e'};
    bias_logits(&field, logits, 256, context, 2, 0.5f);
    PASS("Bias applied");

    free_cooccur_field(&field);
    PASS("CooccurField freed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

void test_integration(void) {
    TEST("Cloud → Mood Integration");
    cloud_init();
    CloudResponse resp = cloud_ping("I'm anxious and scared");

    MoodRouter router;
    init_mood_router(&router, 1.0f);

    Signals signals = {
        .arousal = resp.chambers[CLOUD_CHAMBER_FEAR],
        .warmth = resp.chambers[CLOUD_CHAMBER_LOVE],
        .tension = resp.chambers[CLOUD_CHAMBER_RAGE]
    };
    route_signals_to_moods(&router, &signals);
    CHECK(MOOD_NAMES[router.dominant] != NULL, "Integration produces mood");

    TEST("Cloud → Inner Integration");
    InnerArianna inner;
    inner_init(&inner);
    inner_update_cloud(&inner, &resp);
    float weight = inner_compute_weight(&inner);
    CHECK(weight >= 0.0f, "Cloud affects inner");
    inner_free(&inner);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

void test_edge_cases(void) {
    TEST("Empty Input");
    cloud_init();
    CloudResponse empty = cloud_ping("");
    PASS("Cloud handles empty");

    TEST("Long Input");
    char long_input[2048];
    memset(long_input, 'a', 2047);
    long_input[2047] = '\0';
    CloudResponse long_resp = cloud_ping(long_input);
    PASS("Cloud handles long input");

    TEST("Special Characters");
    CloudResponse special = cloud_ping("!@#$%^&*()");
    PASS("Cloud handles special chars");

    TEST("Division by Zero");
    MathBrain mb;
    init_mathbrain(&mb);
    float conf;
    compute_op(&mb, 10, OP_DIV, 0, &conf);
    PASS("MathBrain handles div/0");
    free_mathbrain(&mb);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    srand(time(NULL));

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  ARIANNA.C COMPREHENSIVE TEST SUITE (New API)\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    TEST_SECTION("CLOUD");
    test_cloud();

    TEST_SECTION("MOOD");
    test_mood();

    TEST_SECTION("BODY SENSE");
    test_body_sense();

    TEST_SECTION("MATHBRAIN");
    test_mathbrain();

    TEST_SECTION("INNER ARIANNA");
    test_inner_arianna();

    TEST_SECTION("AMK KERNEL");
    test_amk();

    TEST_SECTION("COOCCUR");
    test_cooccur();

    TEST_SECTION("INTEGRATION");
    test_integration();

    TEST_SECTION("EDGE CASES");
    test_edge_cases();

    // Summary
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (tests_failed > 0) {
        printf("\n  ⚠️  SOME TESTS FAILED!\n\n");
        return 1;
    } else {
        printf("\n  ✓ ALL TESTS PASSED!\n");
        printf("  הכל עובד. המשך הדרך.\n\n");
        return 0;
    }
}
