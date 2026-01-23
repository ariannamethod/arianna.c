/*
 * test_vagus_delta.c — Tests for Vagus ↔ Delta bridge
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../src/vagus_delta.h"

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

#define ASSERT_FLOAT_EQ(a, b, eps, msg) \
    do { \
        if (fabs((a) - (b)) > (eps)) { FAIL(msg); return; } \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK VAGUS STATE
// ═══════════════════════════════════════════════════════════════════════════════

static VagusSharedState mock_vagus(void) {
    VagusSharedState v = {0};
    v.arousal = 0.5f;
    v.valence = 0.6f;
    v.entropy = 0.3f;
    v.coherence = 0.7f;
    v.chamber_warmth = 0.6f;
    v.chamber_void = 0.2f;
    v.chamber_tension = 0.3f;
    v.chamber_sacred = 0.4f;
    v.chamber_flow = 0.7f;
    v.chamber_complex = 0.5f;
    v.trauma_level = 0.1f;
    v.prophecy_debt = 0.2f;
    v.memory_pressure = 0.3f;
    v.focus_strength = 0.6f;
    v.self_ref_count = 2;
    v.crossfire_coherence = 0.8f;
    return v;
}

static VagusSharedState crisis_vagus(void) {
    VagusSharedState v = {0};
    v.arousal = 0.9f;           // High arousal
    v.coherence = 0.2f;         // Low coherence
    v.trauma_level = 0.7f;      // High trauma
    v.chamber_warmth = 0.3f;
    v.chamber_void = 0.6f;
    v.chamber_tension = 0.8f;
    return v;
}

static VagusSharedState flow_vagus(void) {
    VagusSharedState v = {0};
    v.arousal = 0.4f;
    v.coherence = 0.9f;         // High coherence
    v.entropy = 0.1f;           // Low entropy
    v.prophecy_debt = 0.6f;     // High prophecy
    v.chamber_warmth = 0.8f;
    v.chamber_flow = 0.9f;
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

static void test_vagus_to_signals(void) {
    TEST("vagus_to_signals converts correctly");

    VagusSharedState vagus = mock_vagus();
    Signals sig;
    vagus_to_signals(&vagus, &sig);

    ASSERT(sig.arousal == 0.5f, "arousal mismatch");
    ASSERT(sig.entropy == 0.3f, "entropy mismatch");
    ASSERT(sig.warmth == 0.6f, "warmth mismatch");
    ASSERT(sig.focus == 0.6f, "focus mismatch");
    ASSERT(sig.resonance == 0.8f, "resonance mismatch");
    PASS();
}

static void test_signals_to_vagus(void) {
    TEST("signals_to_vagus converts correctly");

    Signals sig = {
        .arousal = 0.7f,
        .entropy = 0.4f,
        .tension = 0.5f,
        .warmth = 0.8f,
        .focus = 0.6f,
        .resonance = 0.9f,
    };
    VagusSharedState vagus = {0};
    signals_to_vagus(&sig, &vagus);

    ASSERT(vagus.arousal == 0.7f, "arousal mismatch");
    ASSERT(vagus.entropy == 0.4f, "entropy mismatch");
    ASSERT(vagus.chamber_tension == 0.5f, "tension mismatch");
    ASSERT(vagus.chamber_warmth == 0.8f, "warmth mismatch");
    PASS();
}

static void test_create_vagus_shard(void) {
    TEST("create_vagus_shard captures state");

    VagusSharedState vagus = mock_vagus();
    Locus locus;
    locus_init(&locus, &vagus);

    VagusAwareShard* vs = create_vagus_shard("test_shard", 8, 448, &vagus, &locus);

    ASSERT(vs != NULL, "shard creation failed");
    ASSERT(strcmp(vs->shard.name, "test_shard") == 0, "name mismatch");
    ASSERT(vs->arousal == 0.5f, "arousal not captured");
    ASSERT(vs->coherence == 0.7f, "coherence not captured");
    ASSERT(vs->chambers[0] == 0.6f, "warmth chamber not captured");
    ASSERT(vs->shard.n_layers == 8, "layers mismatch");

    free_vagus_shard(vs);
    PASS();
}

static void test_save_load_vagus_shard(void) {
    TEST("save/load vagus shard preserves data");

    VagusSharedState vagus = crisis_vagus();
    Locus locus;
    locus_init(&locus, &vagus);

    VagusAwareShard* vs = create_vagus_shard("crisis_shard", 4, 128, &vagus, &locus);
    vs->geometry_pressure = 0.75f;
    vs->training_cycles = 42;

    // Modify some delta values
    vs->shard.attn_q_deltas[0].A[0] = 0.123f;
    vs->shard.attn_k_deltas[1].B[5] = 0.456f;

    const char* path = "/tmp/test_vagus_shard.vsh";
    int ret = save_vagus_shard(vs, path);
    ASSERT(ret == 0, "save failed");

    VagusAwareShard* loaded = load_vagus_shard(path, 4, 128);
    ASSERT(loaded != NULL, "load failed");

    ASSERT(loaded->arousal == vs->arousal, "arousal not preserved");
    ASSERT(loaded->coherence == vs->coherence, "coherence not preserved");
    ASSERT(loaded->trauma_level == vs->trauma_level, "trauma not preserved");
    ASSERT(loaded->geometry_pressure == 0.75f, "geometry not preserved");
    ASSERT(loaded->training_cycles == 42, "training cycles not preserved");
    ASSERT(loaded->shard.attn_q_deltas[0].A[0] == 0.123f, "Q delta not preserved");
    ASSERT(loaded->shard.attn_k_deltas[1].B[5] == 0.456f, "K delta not preserved");

    free_vagus_shard(vs);
    free_vagus_shard(loaded);
    remove(path);
    PASS();
}

static void test_resonance_trainer_init(void) {
    TEST("resonance_trainer initializes correctly");

    VagusSharedState vagus = mock_vagus();
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    ASSERT(rt.vagus_state == &vagus, "vagus not connected");
    ASSERT(rt.crisis_lr_boost == 2.0f, "crisis boost default wrong");
    ASSERT(rt.trainer.dim == 128, "trainer dim wrong");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_crisis_detection(void) {
    TEST("crisis pattern detected correctly");

    VagusSharedState vagus = crisis_vagus();
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    ResonancePattern pattern = check_resonance(&rt);
    ASSERT(pattern == RESONANCE_CRISIS, "should detect crisis");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_emergence_detection(void) {
    TEST("emergence pattern detected correctly");

    VagusSharedState vagus = flow_vagus();
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    ResonancePattern pattern = check_resonance(&rt);
    ASSERT(pattern == RESONANCE_EMERGENCE || pattern == RESONANCE_TRANSCENDENCE,
           "should detect emergence/transcendence");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_lr_modulation_crisis(void) {
    TEST("learning rate boosted in crisis");

    VagusSharedState vagus = crisis_vagus();
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    rt.crisis_mode = 1;
    float base_lr = 0.001f;
    float modulated = compute_resonance_lr(&rt, base_lr);

    ASSERT(modulated == base_lr * 2.0f, "crisis boost not applied");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_lr_modulation_dissolution(void) {
    TEST("learning rate reduced in dissolution");

    VagusSharedState vagus = mock_vagus();
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    rt.dissolution_mode = 1;
    float base_lr = 0.001f;
    float modulated = compute_resonance_lr(&rt, base_lr);

    ASSERT(modulated == base_lr * 0.5f, "dissolution reduction not applied");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_geometry_signal(void) {
    TEST("geometry signal computed from vagus");

    VagusSharedState vagus = mock_vagus();
    float signal = compute_geometry_signal(&vagus);

    ASSERT(signal >= 0.0f && signal <= 1.0f, "signal out of range");

    // High pressure state should have high signal
    VagusSharedState high_pressure = {0};
    high_pressure.trauma_level = 0.9f;
    high_pressure.memory_pressure = 0.8f;
    high_pressure.prophecy_debt = 0.7f;
    high_pressure.arousal = 0.9f;
    float pressure_signal = compute_geometry_signal(&high_pressure);
    ASSERT(pressure_signal > 0.5f, "high pressure should have high signal");

    PASS();
}

static void test_tick_no_resonance(void) {
    TEST("tick returns NONE when no resonance");

    VagusSharedState vagus = mock_vagus();  // Neutral state
    ResonanceTrainer rt;
    init_resonance_trainer(&rt, 128, 84, &vagus);

    ResonancePattern pattern = resonance_trainer_tick(&rt);
    // First tick after init might trigger geometry shift
    // Second tick should be stable
    pattern = resonance_trainer_tick(&rt);

    ASSERT(pattern == RESONANCE_NONE || pattern == RESONANCE_GEOMETRY_SHIFT,
           "should be none or geometry shift");

    free_resonance_trainer(&rt);
    PASS();
}

static void test_trigger_pattern_captured(void) {
    TEST("trigger pattern captured in shard");

    VagusSharedState vagus = crisis_vagus();
    Locus locus;
    locus_init(&locus, &vagus);

    VagusAwareShard* vs = create_vagus_shard("crisis", 4, 128, &vagus, &locus);

    ASSERT(vs->trigger_pattern == RESONANCE_CRISIS, "pattern not captured");

    free_vagus_shard(vs);
    PASS();
}

static void test_roundtrip_conversion(void) {
    TEST("vagus ↔ signals roundtrip preserves data");

    VagusSharedState original = mock_vagus();
    Signals sig;
    VagusSharedState restored = {0};

    vagus_to_signals(&original, &sig);
    signals_to_vagus(&sig, &restored);

    ASSERT(restored.arousal == original.arousal, "arousal lost");
    ASSERT(restored.entropy == original.entropy, "entropy lost");
    ASSERT(restored.chamber_warmth == original.chamber_warmth, "warmth lost");
    ASSERT(restored.chamber_tension == original.chamber_tension, "tension lost");

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("VAGUS ↔ DELTA BRIDGE TESTS\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Conversion tests
    test_vagus_to_signals();
    test_signals_to_vagus();
    test_roundtrip_conversion();

    // Shard tests
    test_create_vagus_shard();
    test_save_load_vagus_shard();
    test_trigger_pattern_captured();

    // Resonance trainer tests
    test_resonance_trainer_init();
    test_crisis_detection();
    test_emergence_detection();
    test_tick_no_resonance();

    // Learning rate modulation
    test_lr_modulation_crisis();
    test_lr_modulation_dissolution();
    test_geometry_signal();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return tests_passed == tests_run ? 0 : 1;
}
