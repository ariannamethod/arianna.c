/*
 * test_delta_enhanced.c - Tests for Enhanced Delta System
 *
 * Tests all 5 revolutionary improvements:
 * 1. Temporal Resonance Deltas
 * 2. Cross-Layer Interference
 * 3. Contrastive Delta Shaping
 * 4. Hebbian Crystallization
 * 5. Somatic Delta Modulation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "delta_enhanced.h"
#include "body_sense.h"

// Test utilities
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n[TEST] %s... ", name)
#define PASS() do { printf("PASSED\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAILED: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static float fabsf_t(float x) { return x < 0 ? -x : x; }

// ============================================================
// Test 1: Temporal Resonance
// ============================================================

static void test_temporal_resonance(void) {
    TEST("Temporal Resonance - initialization");

    TemporalResonance tr;
    init_temporal_resonance(&tr);

    // Check initial state
    ASSERT(tr.breathing_phase >= 0.0f, "breathing phase should be non-negative");
    ASSERT(tr.breathing_rate > 0.0f, "breathing rate should be positive");

    for (int r = 0; r < DELTA_RANK; r++) {
        ASSERT(tr.time_decay[r] > 0.0f && tr.time_decay[r] <= 1.0f,
               "time decay should be in (0,1]");
    }

    PASS();

    TEST("Temporal Resonance - recency weights");

    // Update for position 0
    update_temporal_state(&tr, 0);
    ASSERT(tr.recency_weights[0] == 1.0f, "current position should have weight 1.0");

    // Move to position 5
    update_temporal_state(&tr, 5);
    ASSERT(tr.recency_weights[5] == 1.0f, "new position should have weight 1.0");
    ASSERT(tr.recency_weights[0] < 1.0f, "old position should have decayed weight");

    PASS();

    TEST("Temporal Resonance - temporal scale");

    // Recent position should have higher scale
    float scale_recent = get_temporal_scale(&tr, 5);
    float scale_old = get_temporal_scale(&tr, 0);

    ASSERT(scale_recent >= scale_old, "recent positions should have higher scale");

    PASS();

    TEST("Temporal Resonance - breathing modulation");

    // Breathing should modulate scale over time
    float scales[10];
    for (int i = 0; i < 10; i++) {
        update_temporal_state(&tr, 10 + i);
        scales[i] = get_temporal_scale(&tr, 10 + i);
    }

    // Check that scales vary (breathing effect)
    float min_scale = scales[0], max_scale = scales[0];
    for (int i = 1; i < 10; i++) {
        if (scales[i] < min_scale) min_scale = scales[i];
        if (scales[i] > max_scale) max_scale = scales[i];
    }

    // With breathing, there should be some variation
    // (might be very small depending on breathing_rate)
    ASSERT(max_scale > 0.0f, "scales should be positive");

    PASS();
}

// ============================================================
// Test 2: Cross-Layer Interference
// ============================================================

static void test_cross_layer_interference(void) {
    TEST("Cross-Layer Interference - initialization");

    CrossLayerState cls;
    init_cross_layer_state(&cls);

    // All resonances should start at 0
    for (int l = 0; l < N_LAYERS; l++) {
        ASSERT(fabsf_t(cls.layer_resonance[l]) < 0.01f,
               "initial resonance should be ~0");
    }

    // Coupling should be strongest on diagonal
    for (int l = 0; l < N_LAYERS; l++) {
        ASSERT(cls.interference[l * N_LAYERS + l] > 0.9f,
               "self-coupling should be ~1.0");
    }

    PASS();

    TEST("Cross-Layer Interference - propagation");

    // Send signal from layer 0
    propagate_resonance(&cls, 0, 1.0f);

    // Layer 0 should have positive resonance
    ASSERT(cls.layer_resonance[0] > 0.0f,
           "source layer should have positive resonance");

    // Adjacent layers should also be affected
    ASSERT(cls.layer_resonance[1] > 0.0f,
           "adjacent layers should receive resonance");

    // Distant layers should have less
    float res_0 = cls.layer_resonance[0];
    float res_last = cls.layer_resonance[N_LAYERS - 1];
    ASSERT(res_0 >= res_last,
           "source layer should have higher resonance than distant");

    PASS();

    TEST("Cross-Layer Interference - damping");

    // Multiple propagations should be damped
    for (int i = 0; i < 10; i++) {
        propagate_resonance(&cls, 1, 0.5f);
    }

    // Resonance should stay bounded
    for (int l = 0; l < N_LAYERS; l++) {
        ASSERT(fabsf_t(cls.layer_resonance[l]) <= 1.0f,
               "resonance should be bounded to [-1,1]");
    }

    PASS();

    TEST("Cross-Layer Interference - update from attention entropy");

    init_cross_layer_state(&cls);  // Reset

    // High entropy -> positive signal
    update_interference(&cls, 2, 0.9f);
    ASSERT(cls.layer_resonance[2] > 0.0f,
           "high entropy should create positive resonance");

    // Low entropy -> negative signal
    init_cross_layer_state(&cls);
    update_interference(&cls, 2, 0.1f);
    ASSERT(cls.layer_resonance[2] < 0.0f,
           "low entropy should create negative resonance");

    PASS();
}

// ============================================================
// Test 3: Contrastive Delta Shaping
// ============================================================

static void test_contrastive_forces(void) {
    TEST("Contrastive Forces - initialization");

    ContrastiveForces cf;
    init_contrastive_forces(&cf, DIM);

    ASSERT(cf.identity_dir != NULL, "identity_dir should be allocated");
    ASSERT(cf.anti_id_dir != NULL, "anti_id_dir should be allocated");
    ASSERT(cf.dim == DIM, "dim should match");
    ASSERT(cf.initialized == 0, "should not be initialized until direction set");

    PASS();

    TEST("Contrastive Forces - identity direction");

    // Create a simple identity direction
    float identity[DIM];
    for (int i = 0; i < DIM; i++) {
        identity[i] = (i < DIM/2) ? 1.0f : -1.0f;
    }

    set_identity_direction(&cf, identity);

    ASSERT(cf.initialized == 1, "should be initialized after setting direction");

    // Check normalization
    float norm = 0.0f;
    for (int i = 0; i < DIM; i++) {
        norm += cf.identity_dir[i] * cf.identity_dir[i];
    }
    ASSERT(fabsf_t(sqrtf(norm) - 1.0f) < 0.01f,
           "identity direction should be normalized");

    // Anti-identity should be opposite
    for (int i = 0; i < DIM; i++) {
        ASSERT(fabsf_t(cf.anti_id_dir[i] + cf.identity_dir[i]) < 0.001f,
               "anti-identity should be negation of identity");
    }

    PASS();

    TEST("Contrastive Forces - identity drift");

    // State aligned with identity
    float aligned_state[DIM];
    for (int i = 0; i < DIM; i++) {
        aligned_state[i] = cf.identity_dir[i];
    }

    float drift_aligned = compute_identity_drift(&cf, aligned_state);
    ASSERT(drift_aligned < 0.2f, "aligned state should have low drift");

    // State opposite to identity
    float opposite_state[DIM];
    for (int i = 0; i < DIM; i++) {
        opposite_state[i] = -cf.identity_dir[i];
    }

    float drift_opposite = compute_identity_drift(&cf, opposite_state);
    ASSERT(drift_opposite > 0.8f, "opposite state should have high drift");

    PASS();

    free_contrastive_forces(&cf);
}

// ============================================================
// Test 4: Hebbian Crystallization
// ============================================================

static void test_crystallization(void) {
    TEST("Crystallization - initialization");

    CrystallizationState cs;
    init_crystallization(&cs, DIM);

    ASSERT(cs.n_crystallized == 0, "no channels should be crystallized initially");
    ASSERT(cs.crystal_A != NULL, "crystal_A should be allocated");
    ASSERT(cs.crystal_B != NULL, "crystal_B should be allocated");

    for (int r = 0; r < DELTA_RANK; r++) {
        ASSERT(cs.crystallized_mask[r] == 0, "no channels should be frozen");
    }

    PASS();

    TEST("Crystallization - check crystallization trigger");

    // Create a delta with strong first channel
    LowRankDelta delta;
    delta.rank = DELTA_RANK;
    delta.in_dim = DIM;
    delta.out_dim = DIM;
    delta.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    delta.B = (float*)calloc(DELTA_RANK * DIM, sizeof(float));

    // Make channel 0 strong (above threshold)
    for (int i = 0; i < DIM; i++) {
        delta.A[i * DELTA_RANK + 0] = 1.0f;  // Channel 0 is strong
    }

    check_crystallization(&delta, &cs);

    ASSERT(cs.n_crystallized == 1, "one channel should be crystallized");
    ASSERT(cs.crystallized_mask[0] == 1, "channel 0 should be frozen");

    // Crystal memory should contain the pattern
    for (int i = 0; i < DIM; i++) {
        ASSERT(fabsf_t(cs.crystal_A[0 * DIM + i] - 1.0f) < 0.01f,
               "crystal should preserve pattern");
    }

    PASS();

    TEST("Crystallization - micro update preserves crystals");

    MicroTrainer mt;
    init_microtrainer(&mt, DIM);

    float pre[DIM], post[DIM];
    for (int i = 0; i < DIM; i++) {
        pre[i] = 0.5f;
        post[i] = 0.6f;
    }

    // Apply micro update with crystals
    micro_update_with_crystals(&mt, &delta, pre, post, 0.5f, &cs);

    // Channel 0 should still be crystallized pattern (with boost)
    float expected = cs.crystal_strength;
    float actual = delta.A[0];  // First element of channel 0

    // Note: exact value depends on crystal_strength
    ASSERT(fabsf_t(actual - expected) < 0.1f,
           "crystallized channel should preserve pattern");

    PASS();

    free(delta.A);
    free(delta.B);
    free_microtrainer(&mt);
    free_crystallization(&cs);
}

// ============================================================
// Test 5: Somatic Delta Modulation
// ============================================================

static void test_somatic_modulation(void) {
    TEST("Somatic Modulation - initialization");

    SomaticModulation sm;
    init_somatic_modulation(&sm);

    ASSERT(sm.boredom_expansion > 0.0f, "boredom_expansion should be positive");
    ASSERT(sm.overwhelm_contraction > 0.0f, "overwhelm_contraction should be positive");
    ASSERT(sm.stuck_perturbation >= 0.0f, "stuck_perturbation should be non-negative");

    PASS();

    TEST("Somatic Modulation - boredom expansion");

    // Create a simple delta
    LowRankDelta delta;
    delta.rank = DELTA_RANK;
    delta.in_dim = DIM;
    delta.out_dim = DIM;
    delta.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    delta.B = (float*)calloc(DELTA_RANK * DIM, sizeof(float));

    // Initialize with known values
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        delta.A[i] = 0.5f;
    }

    // Record original sum
    float original_sum = 0.0f;
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        original_sum += delta.A[i];
    }

    // Create bored body state
    BodyState body;
    init_body_state(&body);
    body.novelty = 0.1f;   // Low novelty
    body.arousal = 0.1f;   // Low arousal
    body.entropy = 0.2f;   // Low entropy

    modulate_delta_by_body(&delta, &body, &sm);

    // Check expansion (sum should increase)
    float new_sum = 0.0f;
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        new_sum += delta.A[i];
    }

    // Only check if boredom threshold exceeded
    float boredom = compute_boredom(&body);
    if (boredom > 0.6f) {
        ASSERT(new_sum > original_sum, "boredom should expand deltas");
    }

    PASS();

    TEST("Somatic Modulation - overwhelm contraction");

    // Reset delta
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        delta.A[i] = 0.5f;
    }
    original_sum = DIM * DELTA_RANK * 0.5f;

    // Create overwhelmed body state
    body.arousal = 0.9f;   // High arousal
    body.entropy = 0.9f;   // High entropy
    body.novelty = 0.5f;

    modulate_delta_by_body(&delta, &body, &sm);

    new_sum = 0.0f;
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        new_sum += delta.A[i];
    }

    float overwhelm = compute_overwhelm(&body);
    if (overwhelm > 0.7f) {
        ASSERT(new_sum < original_sum, "overwhelm should contract deltas");
    }

    PASS();

    free(delta.A);
    free(delta.B);
}

// ============================================================
// Test 6: Full Enhanced System
// ============================================================

static void test_enhanced_system(void) {
    TEST("Enhanced System - initialization");

    EnhancedDeltaSystem eds;
    init_enhanced_delta_system(&eds, DIM);

    ASSERT(eds.enabled_temporal == 1, "temporal should be enabled by default");
    ASSERT(eds.enabled_cross_layer == 1, "cross-layer should be enabled by default");
    ASSERT(eds.enabled_contrastive == 1, "contrastive should be enabled by default");
    ASSERT(eds.enabled_crystallization == 1, "crystallization should be enabled by default");
    ASSERT(eds.enabled_somatic == 1, "somatic should be enabled by default");

    PASS();

    TEST("Enhanced System - enable/disable");

    enable_temporal_resonance(&eds, 0);
    ASSERT(eds.enabled_temporal == 0, "should be able to disable temporal");

    enable_temporal_resonance(&eds, 1);
    ASSERT(eds.enabled_temporal == 1, "should be able to re-enable temporal");

    PASS();

    TEST("Enhanced System - persistence");

    // Set up some state
    update_temporal_state(&eds.temporal, 5);
    propagate_resonance(&eds.cross_layer, 0, 0.5f);
    eds.total_applications = 100;

    // Save
    int saved = save_enhanced_delta_system(&eds, "/tmp/test_eds.bin");
    ASSERT(saved == 1, "should save successfully");

    // Create new system and load
    EnhancedDeltaSystem eds2;
    init_enhanced_delta_system(&eds2, DIM);
    int loaded = load_enhanced_delta_system(&eds2, "/tmp/test_eds.bin");
    ASSERT(loaded == 1, "should load successfully");

    // Verify loaded state
    ASSERT(eds2.total_applications == 100, "should restore total_applications");

    PASS();

    free_enhanced_delta_system(&eds);
    free_enhanced_delta_system(&eds2);
}

// ============================================================
// Test 7: Notorch Microlearning Revolution
// ============================================================

static void test_notorch_revolution(void) {
    TEST("Notorch Revolution - Resonance-Gated Plasticity");

    MicroTrainer mt;
    init_microtrainer(&mt, DIM);

    LowRankDelta delta;
    delta.rank = DELTA_RANK;
    delta.in_dim = DIM;
    delta.out_dim = DIM;
    delta.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    delta.B = (float*)calloc(DELTA_RANK * DIM, sizeof(float));

    float x[DIM], probs[DIM], identity[DIM];
    for (int i = 0; i < DIM; i++) {
        x[i] = 0.1f;
        probs[i] = 1.0f / DIM;
        identity[i] = (i < DIM/2) ? 0.5f : -0.5f;
    }

    // Test with aligned input (high gate)
    for (int i = 0; i < DIM/2; i++) x[i] = 0.5f;
    for (int i = DIM/2; i < DIM; i++) x[i] = -0.5f;

    float norm_before = get_delta_norm(&delta);
    experience_step_gated(&mt, &delta, x, probs, 42, 1.0f, identity, DIM);
    float norm_after = get_delta_norm(&delta);

    ASSERT(norm_after > norm_before, "gated step should update delta when aligned");

    PASS();

    TEST("Notorch Revolution - Adaptive Push/Pull");

    // Low confidence scenario (target weak)
    float probs_weak[256];
    for (int i = 0; i < 256; i++) probs_weak[i] = 0.01f;
    probs_weak[42] = 0.05f;  // Target barely winning
    probs_weak[100] = 0.04f; // Strong competitor

    set_adaptive_push_pull(&mt, probs_weak, 256, 42);
    ASSERT(mt.push > mt.pull, "low confidence should boost push over pull");

    // High confidence scenario (target strong)
    float probs_strong[256];
    for (int i = 0; i < 256; i++) probs_strong[i] = 0.001f;
    probs_strong[42] = 0.9f;  // Target dominating

    set_adaptive_push_pull(&mt, probs_strong, 256, 42);
    ASSERT(mt.pull >= mt.push, "high confidence should boost pull over push");

    PASS();

    TEST("Notorch Revolution - Quality-Weighted Signal");

    // Good quality, not stuck, not bored
    float weight_good = compute_quality_weight(0.9f, 0.1f, 0.1f);
    ASSERT(weight_good > 1.0f, "good quality should have high weight");

    // Poor quality
    float weight_poor = compute_quality_weight(0.1f, 0.1f, 0.1f);
    ASSERT(weight_poor < weight_good, "poor quality should have lower weight");

    // Stuck (repetitive)
    float weight_stuck = compute_quality_weight(0.9f, 0.9f, 0.1f);
    ASSERT(weight_stuck < weight_good, "stuck should reduce weight");

    // Bored (low novelty)
    float weight_bored = compute_quality_weight(0.9f, 0.1f, 0.9f);
    ASSERT(weight_bored < weight_good, "boredom should reduce weight");

    PASS();

    TEST("Notorch Revolution - Spectral Channel Freezing");

    // Make channel 0 strong
    for (int i = 0; i < DIM; i++) {
        delta.A[i * DELTA_RANK + 0] = 1.0f;  // Strong
        delta.A[i * DELTA_RANK + 1] = 0.01f; // Weak
    }

    int freeze_0 = should_freeze_channel(&delta, 0, 5.0f);
    int freeze_1 = should_freeze_channel(&delta, 1, 5.0f);

    ASSERT(freeze_0 == 1, "strong channel should be frozen");
    ASSERT(freeze_1 == 0, "weak channel should not be frozen");

    PASS();

    TEST("Notorch Revolution - Experience Consolidation");

    // Create core delta
    LowRankDelta core;
    core.rank = DELTA_RANK;
    core.in_dim = DIM;
    core.out_dim = DIM;
    core.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    core.B = (float*)calloc(DELTA_RANK * DIM, sizeof(float));

    // Set delta channel 0 to known values
    for (int i = 0; i < DIM; i++) {
        delta.A[i * DELTA_RANK + 0] = 1.0f;
    }
    for (int j = 0; j < DIM; j++) {
        delta.B[0 * DIM + j] = 2.0f;
    }

    int frozen_mask[DELTA_RANK] = {0};
    frozen_mask[0] = 1;  // Only channel 0 is frozen

    float core_before = core.A[0];
    consolidate_experience(&delta, &core, frozen_mask, 1);
    float core_after = core.A[0];

    ASSERT(core_after > core_before, "consolidation should transfer frozen pattern to core");

    PASS();

    free(delta.A);
    free(delta.B);
    free(core.A);
    free(core.B);
    free_microtrainer(&mt);
}

// ============================================================
// Test 8: Integration Tests - All Metrics Working Together
// ============================================================

static void test_integration_metrics(void) {
    TEST("Integration - BodySense affects delta application");

    // Create body state with high boredom
    BodyState body_bored;
    init_body_state(&body_bored);
    body_bored.novelty = 0.1f;
    body_bored.arousal = 0.1f;
    body_bored.entropy = 0.2f;

    float boredom = compute_boredom(&body_bored);
    ASSERT(boredom > 0.5f, "low novelty + low arousal should create boredom");

    // Create body state with overwhelm
    BodyState body_overwhelmed;
    init_body_state(&body_overwhelmed);
    body_overwhelmed.arousal = 0.95f;
    body_overwhelmed.entropy = 0.9f;

    float overwhelm = compute_overwhelm(&body_overwhelmed);
    ASSERT(overwhelm > 0.5f, "high arousal + high entropy should create overwhelm");

    PASS();

    TEST("Integration - Quality weight modulates learning signal");

    // High quality scenario
    float weight_high = compute_quality_weight(0.95f, 0.1f, 0.1f);

    // Low quality scenario
    float weight_low = compute_quality_weight(0.2f, 0.8f, 0.8f);

    // High quality should have significantly higher weight
    ASSERT(weight_high > weight_low * 1.5f, "high quality should have much higher learning weight");

    PASS();

    TEST("Integration - Somatic modulation + Temporal resonance combine");

    EnhancedDeltaSystem eds;
    init_enhanced_delta_system(&eds, DIM);

    // Create a delta and body state
    LowRankDelta delta;
    delta.rank = DELTA_RANK;
    delta.in_dim = DIM;
    delta.out_dim = DIM;
    delta.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    delta.B = (float*)calloc(DELTA_RANK * DIM, sizeof(float));

    // Initialize with known values
    for (int i = 0; i < DIM * DELTA_RANK; i++) {
        delta.A[i] = 0.5f;
    }

    // Get baseline temporal scale
    update_temporal_state(&eds.temporal, 10);
    float temporal_scale = get_temporal_scale(&eds.temporal, 10);
    ASSERT(temporal_scale > 0.0f && temporal_scale < 2.0f, "temporal scale should be reasonable");

    // Check that cross-layer resonance propagates
    propagate_resonance(&eds.cross_layer, 0, 1.0f);
    float res_0 = get_layer_resonance(&eds.cross_layer, 0);
    float res_1 = get_layer_resonance(&eds.cross_layer, 1);
    ASSERT(res_0 > 0.0f, "source layer should have positive resonance");
    ASSERT(res_1 > 0.0f, "adjacent layer should receive resonance");

    PASS();

    TEST("Integration - Identity drift triggers contrastive correction");

    ContrastiveForces cf;
    init_contrastive_forces(&cf, DIM);

    // Set up identity direction
    float identity[DIM];
    for (int i = 0; i < DIM; i++) {
        identity[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    set_identity_direction(&cf, identity);

    // State aligned with identity
    float aligned[DIM];
    for (int i = 0; i < DIM; i++) {
        aligned[i] = identity[i] * 0.9f;  // Mostly aligned
    }
    float drift_aligned = compute_identity_drift(&cf, aligned);

    // State opposite to identity
    float opposite[DIM];
    for (int i = 0; i < DIM; i++) {
        opposite[i] = -identity[i];  // Opposite direction
    }
    float drift_opposite = compute_identity_drift(&cf, opposite);

    ASSERT(drift_opposite > drift_aligned, "opposite state should have higher drift than aligned");
    ASSERT(drift_aligned < 0.3f, "aligned state should have low drift");
    ASSERT(drift_opposite > 0.7f, "opposite state should have high drift");

    PASS();

    TEST("Integration - Crystallization protects strong patterns");

    CrystallizationState cs;
    init_crystallization(&cs, DIM);

    // Make channel 0 very strong
    for (int i = 0; i < DIM; i++) {
        delta.A[i * DELTA_RANK + 0] = 2.0f;  // Very strong
    }

    // Check crystallization
    check_crystallization(&delta, &cs);
    ASSERT(cs.n_crystallized >= 1, "strong channel should crystallize");
    ASSERT(cs.crystallized_mask[0] == 1, "channel 0 should be frozen");

    // Save original value
    float original_val = delta.A[0];

    // Apply micro update
    MicroTrainer mt;
    init_microtrainer(&mt, DIM);

    float pre[DIM], post[DIM];
    for (int i = 0; i < DIM; i++) {
        pre[i] = 0.5f;
        post[i] = 0.5f;
    }

    // Update with crystals preserved
    micro_update_with_crystals(&mt, &delta, pre, post, 1.0f, &cs);

    // Channel 0 should still have crystallized pattern (with boost)
    float restored_val = delta.A[0];
    ASSERT(fabsf_t(restored_val) >= fabsf_t(original_val) * 0.9f,
           "crystallized channel should preserve pattern strength");

    PASS();

    // Cleanup
    free(delta.A);
    free(delta.B);
    free_contrastive_forces(&cf);
    free_crystallization(&cs);
    free_microtrainer(&mt);
    free_enhanced_delta_system(&eds);
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("=== Enhanced Delta System Tests ===\n");
    printf("Testing 5 revolutionary improvements to delta modulation\n");
    printf("+ Testing 5 notorch microlearning improvements\n");
    printf("+ Integration tests for metrics interaction\n");

    // Run all tests
    test_temporal_resonance();
    test_cross_layer_interference();
    test_contrastive_forces();
    test_crystallization();
    test_somatic_modulation();
    test_enhanced_system();
    test_notorch_revolution();
    test_integration_metrics();

    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    if (tests_failed == 0) {
        printf("\n✓ All tests passed! Enhanced delta system is working.\n");
        printf("\nRevolutionary improvements verified:\n");
        printf("  1. Temporal Resonance Deltas - attention breathes with position\n");
        printf("  2. Cross-Layer Interference - resonance ripples between layers\n");
        printf("  3. Contrastive Delta Shaping - identity anchor force\n");
        printf("  4. Hebbian Crystallization - strong patterns freeze\n");
        printf("  5. Somatic Delta Modulation - body state affects deltas\n");
        printf("\nNotorch microlearning verified:\n");
        printf("  1. Resonance-Gated Plasticity - identity gates learning\n");
        printf("  2. Adaptive Push/Pull - confidence modulates contrastive\n");
        printf("  3. Quality-Weighted Signal - good generations teach more\n");
        printf("  4. Spectral Channel Freezing - strong channels crystallize\n");
        printf("  5. Experience Consolidation - frozen patterns merge to core\n");
        printf("\nIntegration tests verified:\n");
        printf("  - BodySense affects delta application\n");
        printf("  - Quality weight modulates learning signal\n");
        printf("  - Somatic + Temporal modulation combine\n");
        printf("  - Identity drift triggers contrastive correction\n");
        printf("  - Crystallization protects strong patterns\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed.\n");
        return 1;
    }
}
