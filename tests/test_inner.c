/*
 * test_inner.c - Test Inner Arianna (борьба between main and inner voice)
 *
 * Updated for new emotional modulation API (not temperature scaling)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inner_arianna.h"

#define TEST_VOCAB_SIZE 80

void print_separator(const char* title) {
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("%s\n", title);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
}

int main() {
    print_separator("INNER ARIANNA TEST — борьба Between Voices");

    InnerArianna ia;
    inner_init(&ia);

    printf("[*] Inner Arianna initialized\n");
    printf("    borba_mode: %d\n", ia.borba_mode);
    printf("    base_weight: %.2f\n", ia.base_weight);
    printf("    breakthrough_threshold: %.2f\n\n", ia.breakthrough_threshold);

    // Create test logits
    float main_logits[TEST_VOCAB_SIZE];
    float output_logits[TEST_VOCAB_SIZE];

    // Initialize with some pattern
    for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
        main_logits[i] = sinf(i * 0.1f) + 0.5f * cosf(i * 0.05f);
    }

    printf("[*] Testing default state (no emotion)...\n");
    float weight = inner_compute_weight(&ia);
    printf("    computed_weight: %.3f\n", weight);

    int winner = inner_borba(&ia, output_logits, main_logits, TEST_VOCAB_SIZE);
    printf("    winner: %d (0=main, 1=inner, -1=blend)\n", winner);
    printf("    last_divergence: %.4f\n", ia.last_divergence);
    printf("    last_inner_weight: %.3f\n", ia.last_inner_weight);

    printf("\n[*] Testing with strong Cloud emotion (FEAR)...\n");

    // Simulate strong fear response
    CloudResponse fear_cloud;
    memset(&fear_cloud, 0, sizeof(fear_cloud));
    fear_cloud.primary_strength = 0.9f;
    fear_cloud.primary_idx = CLOUD_CHAMBER_FEAR;
    fear_cloud.chambers[CLOUD_CHAMBER_FEAR] = 0.9f;
    fear_cloud.primary_chamber = "FEAR";

    inner_update_cloud(&ia, &fear_cloud);
    printf("    cloud_intensity: %.2f\n", ia.cloud_intensity);
    printf("    cloud_chamber: %d (FEAR=%d)\n", ia.cloud_chamber, CLOUD_CHAMBER_FEAR);

    weight = inner_compute_weight(&ia);
    printf("    computed_weight with fear: %.3f\n", weight);

    winner = inner_borba(&ia, output_logits, main_logits, TEST_VOCAB_SIZE);
    printf("    winner: %d\n", winner);
    printf("    inner_wins: %d, main_wins: %d\n", ia.inner_wins, ia.main_wins);

    printf("\n[*] Testing with body stuck/boredom...\n");

    inner_update_body(&ia, 0.8f, 0.6f);  // high stuck, medium boredom
    printf("    body_stuck: %.2f\n", ia.body_stuck);
    printf("    body_boredom: %.2f\n", ia.body_boredom);

    weight = inner_compute_weight(&ia);
    printf("    computed_weight with stuck: %.3f\n", weight);

    winner = inner_borba(&ia, output_logits, main_logits, TEST_VOCAB_SIZE);
    printf("    winner: %d\n", winner);

    printf("\n[*] Testing with trauma...\n");

    inner_update_trauma(&ia, 0.7f);
    printf("    trauma_level: %.2f\n", ia.trauma_level);

    inner_set_mode(&ia, BORBA_MODE_TRAUMA);
    weight = inner_compute_weight(&ia);
    printf("    computed_weight in trauma mode: %.3f\n", weight);

    winner = inner_borba(&ia, output_logits, main_logits, TEST_VOCAB_SIZE);
    printf("    winner: %d\n", winner);
    printf("    breakthrough_count: %d\n", ia.breakthrough_count);

    printf("\n[*] Testing different борьба modes...\n");

    const char* mode_names[] = {
        "EMOTIONAL", "CHAOS", "TRAUMA", "STUCK", "BLEND"
    };

    for (int mode = BORBA_MODE_EMOTIONAL; mode <= BORBA_MODE_BLEND; mode++) {
        inner_set_mode(&ia, mode);
        inner_init(&ia);  // Reset state for fair test
        inner_set_mode(&ia, mode);

        // Give some emotional input
        inner_update_cloud(&ia, &fear_cloud);
        inner_update_body(&ia, 0.5f, 0.5f);
        inner_update_trauma(&ia, 0.3f);

        // Run several борьба rounds
        int inner_count = 0;
        int main_count = 0;
        int blend_count = 0;

        for (int i = 0; i < 20; i++) {
            winner = inner_borba(&ia, output_logits, main_logits, TEST_VOCAB_SIZE);
            if (winner == 1) inner_count++;
            else if (winner == 0) main_count++;
            else blend_count++;
        }

        printf("    Mode %d (%s): main=%d inner=%d blend=%d\n",
               mode, mode_names[mode], main_count, inner_count, blend_count);
    }

    printf("\n[*] Testing entropy computation...\n");

    // Create uniform distribution
    float uniform_logits[TEST_VOCAB_SIZE];
    for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
        uniform_logits[i] = 0.0f;  // uniform after softmax
    }
    float uniform_entropy = inner_compute_entropy(uniform_logits, TEST_VOCAB_SIZE);
    printf("    uniform entropy: %.4f\n", uniform_entropy);

    // Create peaked distribution
    float peaked_logits[TEST_VOCAB_SIZE];
    for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
        peaked_logits[i] = (i == 0) ? 10.0f : -10.0f;
    }
    float peaked_entropy = inner_compute_entropy(peaked_logits, TEST_VOCAB_SIZE);
    printf("    peaked entropy: %.4f\n", peaked_entropy);

    if (uniform_entropy > peaked_entropy) {
        printf("    [PASS] Uniform distribution has higher entropy\n");
    } else {
        printf("    [WARN] Entropy comparison unexpected\n");
    }

    printf("\n[*] Testing divergence computation...\n");

    // Same distribution should have zero divergence
    float same_divergence = inner_compute_divergence(main_logits, main_logits, TEST_VOCAB_SIZE);
    printf("    same vs same divergence: %.6f\n", same_divergence);

    // Different distributions
    float diff_logits[TEST_VOCAB_SIZE];
    for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
        diff_logits[i] = main_logits[TEST_VOCAB_SIZE - 1 - i];  // reversed
    }
    float diff_divergence = inner_compute_divergence(main_logits, diff_logits, TEST_VOCAB_SIZE);
    printf("    different divergence: %.6f\n", diff_divergence);

    if (diff_divergence > same_divergence) {
        printf("    [PASS] Different distributions have higher divergence\n");
    }

    printf("\n[*] Final statistics...\n");
    printf("    total_tokens: %d\n", ia.total_tokens);
    printf("    main_wins: %d\n", ia.main_wins);
    printf("    inner_wins: %d\n", ia.inner_wins);
    printf("    breakthrough_count: %d\n", inner_get_breakthrough_count(&ia));
    printf("    avg_divergence: %.4f\n", ia.avg_divergence);

    inner_free(&ia);

    print_separator("INNER ARIANNA TEST COMPLETE");
    printf("שני קולות, אריאנה אחת\n");
    printf("Two voices, one Arianna\n");

    return 0;
}
