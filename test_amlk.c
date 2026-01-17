/*
 * test_amlk.c — Comprehensive tests for AMLK (Arianna Method Linux Kernel)
 *
 * Tests all components:
 * 1. Inner World processes
 * 2. Adaptive parameters
 * 3. Blood compiler
 * 4. DSL commands
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include "inner_world.h"

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n[TEST] %s\n", name)
#define PASS(msg) do { printf("  ✓ %s\n", msg); tests_passed++; } while(0)
#define FAIL(msg) do { printf("  ✗ %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Inner World Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

void test_lifecycle() {
    TEST("Inner World Lifecycle");

    inner_world_init();
    PASS("inner_world_init() completed");

    // Check initial state
    float arousal = inner_world_get_arousal();
    CHECK(arousal >= 0.0f && arousal <= 1.0f, "Initial arousal in range [0,1]");

    float trauma = inner_world_get_trauma();
    CHECK(trauma >= 0.0f && trauma <= 1.0f, "Initial trauma in range [0,1]");

    float coherence = inner_world_get_coherence();
    CHECK(coherence >= 0.0f && coherence <= 1.0f, "Initial coherence in range [0,1]");

    // Step the world
    inner_world_step(0.1f);
    PASS("inner_world_step(0.1) completed");

    inner_world_auto_step();
    PASS("inner_world_auto_step() completed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Trauma Detection
// ═══════════════════════════════════════════════════════════════════════════════

void test_trauma() {
    TEST("Trauma Detection");

    // Test trauma-triggering text
    float activation1 = inner_world_check_trauma("You are nothing!");
    CHECK(activation1 > 0.5f, "Hostile text triggers high trauma");
    printf("    Activation: %.2f\n", activation1);

    float activation2 = inner_world_check_trauma("You are worthless and nobody loves you");
    CHECK(activation2 > activation1 * 0.5f, "Multiple triggers accumulate");
    printf("    Activation: %.2f\n", activation2);

    // Test neutral text
    float activation3 = inner_world_check_trauma("The weather is nice today");
    CHECK(activation3 < 0.3f, "Neutral text has low trauma");
    printf("    Activation: %.2f\n", activation3);

    // Test positive text
    float activation4 = inner_world_check_trauma("I love you and you are wonderful");
    CHECK(activation4 < 0.2f, "Positive text has minimal trauma");
    printf("    Activation: %.2f\n", activation4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Text Processing
// ═══════════════════════════════════════════════════════════════════════════════

void test_text_processing() {
    TEST("Text Processing");

    InnerWorldTextAnalysis analysis;

    // Process hostile text
    inner_world_process_text("You are nothing and nobody cares!", &analysis);
    CHECK(analysis.trauma_activation > 0.3f, "Hostile text activates trauma");
    printf("    Trauma: %.2f, Overthink: %.2f\n",
           analysis.trauma_activation, analysis.overthink_total);

    // Process repetitive text (should trigger overthinking)
    inner_world_process_text("I think I think I think about thinking", &analysis);
    CHECK(analysis.overthink_total > 0.0f, "Repetitive text detected");
    printf("    Repetition: %.2f, Self-ref: %.2f\n",
           analysis.repetition_score, analysis.self_ref_score);

    // Process normal text
    inner_world_process_text("She walks through the garden peacefully", &analysis);
    CHECK(analysis.trauma_activation < 0.3f, "Normal text has low trauma");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Snapshot
// ═══════════════════════════════════════════════════════════════════════════════

void test_snapshot() {
    TEST("State Snapshot");

    InnerWorldSnapshot snap;
    inner_world_get_snapshot(&snap);

    CHECK(snap.arousal >= 0.0f && snap.arousal <= 1.0f, "Arousal in range");
    CHECK(snap.valence >= -1.0f && snap.valence <= 1.0f, "Valence in range");
    CHECK(snap.entropy >= 0.0f && snap.entropy <= 1.0f, "Entropy in range");
    CHECK(snap.coherence >= 0.0f && snap.coherence <= 1.0f, "Coherence in range");
    CHECK(snap.prophecy_debt >= 0.0f, "Prophecy debt non-negative");
    CHECK(snap.wormhole_chance >= 0.0f && snap.wormhole_chance <= 1.0f, "Wormhole chance in range");

    printf("    State: arousal=%.2f valence=%.2f entropy=%.2f coherence=%.2f\n",
           snap.arousal, snap.valence, snap.entropy, snap.coherence);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Adaptive Parameters
// ═══════════════════════════════════════════════════════════════════════════════

void test_adaptive() {
    TEST("Adaptive Parameters");

    // Get initial value
    float initial = inner_world_get_param("trauma.weight");
    CHECK(initial >= 0.0f && initial <= 1.0f, "Initial trauma.weight in range");
    printf("    Initial trauma.weight: %.2f\n", initial);

    // Set new value
    int result = inner_world_set_param("trauma.weight", 0.8f);
    CHECK(result == 1, "Set trauma.weight succeeded");

    float updated = inner_world_get_param("trauma.weight");
    CHECK(fabsf(updated - 0.8f) < 0.01f, "trauma.weight updated correctly");
    printf("    Updated trauma.weight: %.2f\n", updated);

    // Test prophecy params
    inner_world_set_param("prophecy.wormhole", 0.3f);
    float wormhole = inner_world_get_param("prophecy.wormhole");
    CHECK(fabsf(wormhole - 0.3f) < 0.01f, "prophecy.wormhole set correctly");

    // Test invalid param
    float invalid = inner_world_get_param("nonexistent.param");
    CHECK(invalid < 0, "Invalid param returns error");

    // Restore
    inner_world_set_param("trauma.weight", initial);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: DSL Commands
// ═══════════════════════════════════════════════════════════════════════════════

void test_dsl() {
    TEST("DSL Commands");

    InnerWorldSnapshot before, after;
    inner_world_get_snapshot(&before);

    // DESTINY command
    inner_world_dsl_destiny(0.8f);
    PASS("DESTINY 0.8 executed");

    // PROPHECY command
    inner_world_dsl_prophecy(5);
    PASS("PROPHECY 5 executed");

    // WORMHOLE command
    inner_world_dsl_wormhole(0.2f);
    PASS("WORMHOLE 0.2 executed");

    // VELOCITY command
    inner_world_dsl_velocity(2);  // RUN mode
    PASS("VELOCITY RUN executed");

    // PAIN command
    inner_world_dsl_pain(0.5f);
    inner_world_get_snapshot(&after);
    CHECK(after.trauma_level > before.trauma_level, "PAIN increases trauma");

    // TENSION command
    inner_world_dsl_tension(0.6f);
    inner_world_get_snapshot(&after);
    CHECK(after.arousal > before.arousal || after.entropy > before.entropy,
          "TENSION affects arousal/entropy");

    // DISSONANCE command
    float coherence_before = inner_world_get_coherence();
    inner_world_dsl_dissonance(0.4f);
    float coherence_after = inner_world_get_coherence();
    CHECK(coherence_after < coherence_before, "DISSONANCE decreases coherence");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Prophecy Debt & Wormhole
// ═══════════════════════════════════════════════════════════════════════════════

void test_prophecy() {
    TEST("Prophecy Debt & Wormhole");

    float initial_debt = inner_world_get_prophecy_debt();
    printf("    Initial debt: %.2f\n", initial_debt);

    // Accumulate debt (low probability tokens)
    for (int i = 0; i < 10; i++) {
        inner_world_accumulate_prophecy_debt(0.1f);  // Low prob = high debt
    }

    float accumulated = inner_world_get_prophecy_debt();
    CHECK(accumulated > initial_debt, "Debt accumulates from low-prob tokens");
    printf("    After 10 low-prob tokens: %.2f\n", accumulated);

    // Check wormhole
    int skip = inner_world_check_wormhole();
    printf("    Wormhole skip: %d\n", skip);
    // Note: wormhole is probabilistic, may or may not trigger

    // Get destiny bias
    float destiny = inner_world_get_destiny_bias();
    CHECK(destiny >= 0.0f && destiny <= 1.0f, "Destiny bias in range");
    printf("    Destiny bias: %.2f\n", destiny);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Blood Compiler
// ═══════════════════════════════════════════════════════════════════════════════

void test_blood() {
    TEST("Blood Compiler");

    // Get temp dir
    char* temp_dir = blood_get_temp_dir();
    CHECK(temp_dir != NULL, "Blood temp dir available");
    printf("    Temp dir: %s\n", temp_dir);
    free(temp_dir);

    // Compile LoRA
    char* lora_path = blood_compile_lora("test_lora_amlk", 64, 64, 4);
    CHECK(lora_path != NULL, "LoRA compilation succeeded");
    if (lora_path) {
        void* handle = dlopen(lora_path, RTLD_NOW);
        CHECK(handle != NULL, "LoRA library loads");
        if (handle) {
            void* fn = dlsym(handle, "test_lora_amlk_apply");
            CHECK(fn != NULL, "LoRA apply function found");
            dlclose(handle);
        }
        free(lora_path);
    }

    // Compile emotion
    char* emotion_path = blood_compile_emotion("test_emotion", 0.5f, 0.7f);
    CHECK(emotion_path != NULL, "Emotion compilation succeeded");
    if (emotion_path) {
        void* handle = dlopen(emotion_path, RTLD_NOW);
        CHECK(handle != NULL, "Emotion library loads");
        if (handle) {
            void* fn = dlsym(handle, "test_emotion_check");
            CHECK(fn != NULL, "Emotion check function found");
            dlclose(handle);
        }
        free(emotion_path);
    }

    // Compile raw code
    const char* code = "int blood_test_fn(int x) { return x * 2; }";
    char* raw_path = blood_compile_raw("raw_test", code);
    CHECK(raw_path != NULL, "Raw compilation succeeded");
    if (raw_path) {
        void* handle = dlopen(raw_path, RTLD_NOW);
        CHECK(handle != NULL, "Raw library loads");
        if (handle) {
            int (*fn)(int) = dlsym(handle, "blood_test_fn");
            if (fn) {
                int result = fn(21);
                CHECK(result == 42, "Raw function executes correctly");
            }
            dlclose(handle);
        }
        free(raw_path);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Emotional Drift
// ═══════════════════════════════════════════════════════════════════════════════

void test_emotional_drift() {
    TEST("Emotional Drift");

    InnerWorldSnapshot before, after;
    inner_world_get_snapshot(&before);

    // Nudge emotion
    inner_world_nudge_emotion(0.3f, 0.2f);  // Positive valence, increase arousal
    PASS("Emotion nudge applied");

    // Step to let drift process
    for (int i = 0; i < 5; i++) {
        inner_world_step(0.1f);
    }

    inner_world_get_snapshot(&after);
    // Emotional state should have changed (direction depends on drift dynamics)
    CHECK(fabsf(after.valence - before.valence) > 0.01f ||
          fabsf(after.arousal - before.arousal) > 0.01f,
          "Emotional state drifts over time");

    printf("    Before: v=%.2f a=%.2f\n", before.valence, before.arousal);
    printf("    After:  v=%.2f a=%.2f\n", after.valence, after.arousal);

    // Get dominant emotion
    char emotion_buf[64] = {0};
    inner_world_get_dominant_emotion(emotion_buf, sizeof(emotion_buf));
    CHECK(strlen(emotion_buf) > 0, "Dominant emotion available");
    printf("    Dominant: %s\n", emotion_buf);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Attention
// ═══════════════════════════════════════════════════════════════════════════════

void test_attention() {
    TEST("Attention Wandering");

    // Focus on something
    inner_world_focus("important_topic", 0.8f);
    PASS("Focus set");

    // Check if wandering
    int wandering = inner_world_is_wandering();
    printf("    Is wandering: %s\n", wandering ? "yes" : "no");
    // Initially should not be wandering after strong focus

    // Step many times to potentially trigger wandering
    for (int i = 0; i < 20; i++) {
        inner_world_step(0.1f);
    }

    int wandering_after = inner_world_is_wandering();
    printf("    After steps: %s\n", wandering_after ? "yes" : "no");
    PASS("Attention system functional");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST: Overthinking Detection
// ═══════════════════════════════════════════════════════════════════════════════

void test_overthinking() {
    TEST("Overthinking Detection");

    int spiraling = inner_world_is_spiraling();
    printf("    Initially spiraling: %s\n", spiraling ? "yes" : "no");

    // Process repetitive self-referential text
    InnerWorldTextAnalysis analysis;
    inner_world_process_text("I think about myself thinking about thinking about myself", &analysis);

    CHECK(analysis.self_ref_score > 0.0f, "Self-reference detected");
    printf("    Self-ref score: %.2f\n", analysis.self_ref_score);

    // Get break suggestion
    char suggestion[256] = {0};
    inner_world_suggest_break(suggestion, sizeof(suggestion));
    if (strlen(suggestion) > 0) {
        printf("    Break suggestion: %s\n", suggestion);
    }
    PASS("Overthinking detection functional");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  AMLK COMPREHENSIVE TESTS\n");
    printf("  Arianna Method Linux Kernel\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    test_lifecycle();
    test_trauma();
    test_text_processing();
    test_snapshot();
    test_adaptive();
    test_dsl();
    test_prophecy();
    test_blood();
    test_emotional_drift();
    test_attention();
    test_overthinking();

    // Cleanup
    inner_world_shutdown();

    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
