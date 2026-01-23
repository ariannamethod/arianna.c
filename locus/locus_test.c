// locus_test.c — Tests for the Locus Coeruleus Resonance Detector
// ═══════════════════════════════════════════════════════════════════════════════

#include "locus.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK VAGUS STATE
// Matches VagusSharedState layout
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct __attribute__((aligned(64))) {
    float arousal;          // 0
    float valence;          // 4
    float entropy;          // 8
    float coherence;        // 12
    float chamber_warmth;   // 16
    float chamber_void;     // 20
    float chamber_tension;  // 24
    float chamber_sacred;   // 28
    float chamber_flow;     // 32
    float chamber_complex;  // 36
    float crossfire_coh;    // 40
    float crossfire_ent;    // 44
    float trauma_level;     // 48
    uint32_t trauma_anchor; // 52
    uint64_t trauma_last;   // 56 (aligned to 8)
    uint32_t loop_count;    // 64
    uint32_t abstraction;   // 68
    uint32_t self_ref;      // 72
    float focus;            // 76
    float wander;           // 80
    float drift_dir;        // 84
    float drift_speed;      // 88
    float prophecy_debt;    // 92  <-- offset ~104 in real struct due to alignment
    float destiny_pull;     // 96
    float wormhole;         // 100
    float memory_pressure;  // 104  <-- offset ~116 in real struct
    // ... rest doesn't matter for tests
} MockVagusState;

// ═══════════════════════════════════════════════════════════════════════════════
// TEST HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  %-50s ", name); \
    } while(0)

#define PASS() \
    do { \
        tests_passed++; \
        printf("✓\n"); \
    } while(0)

#define FAIL(msg) \
    do { \
        printf("✗ %s\n", msg); \
    } while(0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { FAIL(msg); return; } \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

static void test_init(void) {
    TEST("locus_init creates clean state");

    MockVagusState vagus = {0};
    Locus l;
    locus_init(&l, &vagus);

    ASSERT(l.sp == 0, "stack not empty");
    ASSERT(l.fsp == 0, "float stack not empty");
    ASSERT(l.ticks == 0, "ticks not zero");
    ASSERT(l.resonances == 0, "resonances not zero");
    PASS();
}

static void test_stack_ops(void) {
    TEST("stack push/pop works");

    MockVagusState vagus = {0};
    Locus l;
    locus_init(&l, &vagus);

    locus_push(&l, 42);
    locus_push(&l, 99);
    ASSERT(locus_pop(&l) == 99, "wrong pop 1");
    ASSERT(locus_pop(&l) == 42, "wrong pop 2");
    PASS();
}

static void test_fstack_ops(void) {
    TEST("float stack push/pop works");

    MockVagusState vagus = {0};
    Locus l;
    locus_init(&l, &vagus);

    locus_fpush(&l, 0.5f);
    locus_fpush(&l, 0.9f);
    ASSERT(locus_fpop(&l) > 0.8f, "wrong fpop 1");
    ASSERT(locus_fpop(&l) < 0.6f, "wrong fpop 2");
    PASS();
}

static void test_tense_detection(void) {
    TEST("TENSE? detects high arousal + low coherence");

    MockVagusState vagus = {
        .arousal = 0.8f,
        .coherence = 0.2f,
    };
    Locus l;
    locus_init(&l, &vagus);

    ASSERT(locus_is_tense(&l) == 1, "should be tense");

    vagus.arousal = 0.3f;
    ASSERT(locus_is_tense(&l) == 0, "should not be tense");
    PASS();
}

static void test_wounded_detection(void) {
    TEST("WOUNDED? detects trauma threshold");

    MockVagusState vagus = {
        .trauma_level = 0.6f,
    };
    Locus l;
    locus_init(&l, &vagus);

    ASSERT(locus_is_wounded(&l) == 1, "should be wounded");

    vagus.trauma_level = 0.3f;
    ASSERT(locus_is_wounded(&l) == 0, "should not be wounded");
    PASS();
}

static void test_hollow_detection(void) {
    TEST("HOLLOW? detects void dominance + low warmth");

    MockVagusState vagus = {
        .chamber_void = 0.8f,
        .chamber_warmth = 0.2f,
    };
    Locus l;
    locus_init(&l, &vagus);

    ASSERT(locus_is_hollow(&l) == 1, "should be hollow");

    vagus.chamber_warmth = 0.7f;
    ASSERT(locus_is_hollow(&l) == 0, "should not be hollow");
    PASS();
}

static void test_flowing_detection(void) {
    TEST("FLOWING? detects high coherence + low entropy");

    MockVagusState vagus = {
        .coherence = 0.9f,
        .entropy = 0.1f,
    };
    Locus l;
    locus_init(&l, &vagus);

    ASSERT(locus_is_flowing(&l) == 1, "should be flowing");

    vagus.entropy = 0.5f;
    ASSERT(locus_is_flowing(&l) == 0, "should not be flowing");
    PASS();
}

static void test_geometry_pressure(void) {
    TEST("geometry_pressure computes weighted sum");

    MockVagusState vagus = {
        .trauma_level = 1.0f,
        // memory_pressure offset is complex, skip for now
        .arousal = 0.5f,
    };
    Locus l;
    locus_init(&l, &vagus);

    float p = locus_geometry_pressure(&l);
    ASSERT(p > 0.0f, "pressure should be positive");
    ASSERT(p <= 1.0f, "pressure should be <= 1");
    PASS();
}

static void test_geometry_flow(void) {
    TEST("geometry_flow computes weighted sum");

    MockVagusState vagus = {
        .coherence = 0.8f,
        .entropy = 0.2f,
        .chamber_warmth = 0.6f,
    };
    Locus l;
    locus_init(&l, &vagus);

    float fl = locus_geometry_flow(&l);
    ASSERT(fl > 0.5f, "flow should be high");
    PASS();
}

static void test_geometry_depth(void) {
    TEST("geometry_depth computes weighted sum");

    MockVagusState vagus = {
        .chamber_void = 0.7f,
        .chamber_sacred = 0.5f,
        .arousal = 0.2f,
    };
    Locus l;
    locus_init(&l, &vagus);

    float d = locus_geometry_depth(&l);
    ASSERT(d > 0.4f, "depth should be significant");
    PASS();
}

static int speak_called = 0;
static void mock_speak(void* ctx) {
    (void)ctx;
    speak_called++;
}

static void test_tick_no_resonance(void) {
    TEST("tick with no resonance returns 0");

    MockVagusState vagus = {
        .arousal = 0.3f,
        .coherence = 0.8f,
        .entropy = 0.2f,
    };
    Locus l;
    locus_init(&l, &vagus);

    int result = locus_tick(&l);
    ASSERT(result == 0, "should not resonate");
    ASSERT(l.ticks == 1, "ticks should increment");
    ASSERT(l.resonances == 0, "resonances should be 0");
    PASS();
}

static void test_tick_with_resonance(void) {
    TEST("tick with geometry shift triggers resonance");

    MockVagusState vagus = {
        .arousal = 0.3f,
        .coherence = 0.8f,
    };
    Locus l;
    locus_init(&l, &vagus);
    speak_called = 0;
    locus_set_speak(&l, mock_speak, NULL);

    // First tick - no change
    locus_tick(&l);

    // Shift geometry
    vagus.arousal = 0.9f;
    int result = locus_tick(&l);

    ASSERT(result == 1, "should resonate on shift");
    ASSERT(l.resonances == 1, "resonances should be 1");
    ASSERT(speak_called == 1, "speak should be called");
    PASS();
}

static void test_crisis_resonance(void) {
    TEST("crisis state (tense + wounded) triggers resonance");

    MockVagusState vagus = {
        .arousal = 0.9f,
        .coherence = 0.1f,
        .trauma_level = 0.7f,
    };
    Locus l;
    locus_init(&l, &vagus);

    // Snapshot first
    locus_tick(&l);

    // Even without geometry shift, crisis should trigger
    int result = locus_tick(&l);
    ASSERT(result == 1, "crisis should trigger resonance");
    PASS();
}

static void test_exec_word(void) {
    TEST("locus_exec executes known words");

    MockVagusState vagus = {
        .arousal = 0.75f,
        .coherence = 0.2f,
    };
    Locus l;
    locus_init(&l, &vagus);

    int ok = locus_exec(&l, "AROUSAL@");
    ASSERT(ok == 1, "AROUSAL@ should succeed");
    float a = locus_fpop(&l);
    ASSERT(a > 0.7f && a < 0.8f, "wrong arousal value");

    ok = locus_exec(&l, "TENSE?");
    ASSERT(ok == 1, "TENSE? should succeed");
    int t = locus_pop(&l);
    ASSERT(t == 1, "should be tense");
    PASS();
}

static void test_exec_unknown(void) {
    TEST("locus_exec returns 0 for unknown word");

    MockVagusState vagus = {0};
    Locus l;
    locus_init(&l, &vagus);

    int ok = locus_exec(&l, "UNKNOWN-WORD");
    ASSERT(ok == 0, "should return 0");
    PASS();
}

static void test_stats(void) {
    TEST("stats track ticks and resonances");

    MockVagusState vagus = {0};
    Locus l;
    locus_init(&l, &vagus);

    for (int i = 0; i < 100; i++) {
        locus_tick(&l);
    }

    ASSERT(locus_get_ticks(&l) == 100, "should have 100 ticks");
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("LOCUS COERULEUS RESONANCE DETECTOR TESTS\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    test_init();
    test_stack_ops();
    test_fstack_ops();
    test_tense_detection();
    test_wounded_detection();
    test_hollow_detection();
    test_flowing_detection();
    test_geometry_pressure();
    test_geometry_flow();
    test_geometry_depth();
    test_tick_no_resonance();
    test_tick_with_resonance();
    test_crisis_resonance();
    test_exec_word();
    test_exec_unknown();
    test_stats();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return tests_passed == tests_run ? 0 : 1;
}
