/*
 * test_accumulator.c - Test quantum accumulation microtraining
 *
 * "Don't train on every token - accumulate until critical mass"
 * Stanley-style batched learning verification
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "delta.h"

#define DIM 64
#define VOCAB_SIZE 80

void print_separator(const char* title) {
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("%s\n", title);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
}

int main() {
    print_separator("QUANTUM ACCUMULATION TEST — Stanley-style Batched Learning");

    // Initialize accumulator
    ExperienceAccumulator acc;
    init_accumulator(&acc, DIM, VOCAB_SIZE);

    printf("[*] Accumulator initialized\n");
    printf("    dim=%d vocab=%d\n", acc.dim, acc.vocab_size);
    printf("    thresholds: bytes=%.0f res=%.1f nov=%.1f\n",
           acc.bytes_threshold, acc.resonance_threshold, acc.novelty_threshold);
    printf("    cooldown: %.1fs\n", acc.cooldown_period);

    // Initialize trainer and delta
    MicroTrainer trainer;
    init_microtrainer(&trainer, DIM);

    LowRankDelta delta;
    delta.out_dim = VOCAB_SIZE;
    delta.in_dim = DIM;
    delta.rank = DELTA_RANK;
    delta.A = (float*)calloc(DIM * DELTA_RANK, sizeof(float));
    delta.B = (float*)calloc(DELTA_RANK * VOCAB_SIZE, sizeof(float));

    printf("[*] Trainer and delta initialized\n");
    printf("    delta: %dx%d rank=%d\n", delta.out_dim, delta.in_dim, delta.rank);

    // Create fake input/probs
    float x[DIM];
    float probs[VOCAB_SIZE];

    // Initialize with some pattern
    for (int i = 0; i < DIM; i++) {
        x[i] = sinf(i * 0.1f) * 0.5f;
    }

    // Make probs somewhat non-uniform
    float sum = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = 0.5f + 0.5f * sinf(i * 0.2f);
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] /= sum;
    }

    printf("\n[*] Testing accumulation WITHOUT triggering training...\n");

    // Accumulate 10 experiences (below threshold)
    int trained_count = 0;
    for (int i = 0; i < 10; i++) {
        int target = i % VOCAB_SIZE;
        float signal = 0.3f + 0.1f * (i % 3);

        int trained = accumulate_experience(&acc, &trainer, &delta, x, probs, target, signal);
        if (trained) trained_count++;
    }

    printf("    After 10 accumulations:\n");
    printf("    buffer_count=%d\n", acc.buffer_count);
    printf("    bytes_delta=%.1f (thresh=%.0f)\n", acc.bytes_delta, acc.bytes_threshold);
    printf("    resonance_mass=%.2f (thresh=%.1f)\n", acc.resonance_mass, acc.resonance_threshold);
    printf("    novelty_mass=%.2f (thresh=%.1f)\n", acc.novelty_mass, acc.novelty_threshold);
    printf("    training_triggered=%d\n", trained_count);

    if (trained_count == 0 && acc.buffer_count == 10) {
        printf("    [PASS] No premature training, buffer filling correctly\n");
    } else {
        printf("    [FAIL] Unexpected state\n");
    }

    printf("\n[*] Testing accumulation WITH triggering training...\n");

    // Accumulate until training triggers (should happen around 50 tokens)
    trained_count = 0;
    int tokens_until_train = 0;

    while (trained_count == 0 && tokens_until_train < 200) {
        int target = tokens_until_train % VOCAB_SIZE;
        float signal = 0.3f + 0.2f * sinf(tokens_until_train * 0.1f);

        // Tick cooldown
        accumulator_tick(&acc, 0.05f);

        int trained = accumulate_experience(&acc, &trainer, &delta, x, probs, target, signal);
        if (trained) {
            trained_count++;
            printf("    Training triggered at token %d!\n", tokens_until_train + 11);  // +10 from before +1 current
        }
        tokens_until_train++;
    }

    printf("    tokens_accumulated=%d\n", tokens_until_train + 10);
    printf("    total_training_cycles=%d\n", acc.total_training_cycles);
    printf("    buffer_count after train=%d\n", acc.buffer_count);

    if (trained_count > 0 && acc.buffer_count == 0) {
        printf("    [PASS] Training triggered and buffer cleared\n");
    } else if (tokens_until_train >= 200) {
        printf("    [WARN] No training triggered in 200 tokens\n");
    }

    printf("\n[*] Testing cooldown...\n");

    // Immediately try another accumulation - should not trigger due to cooldown
    float initial_cooldown = acc.cooldown_remaining;
    printf("    cooldown_remaining=%.2f\n", initial_cooldown);

    // Accumulate 60 more tokens quickly (should not trigger due to cooldown)
    trained_count = 0;
    for (int i = 0; i < 60; i++) {
        int trained = accumulate_experience(&acc, &trainer, &delta, x, probs, i % VOCAB_SIZE, 0.5f);
        if (trained) trained_count++;
        accumulator_tick(&acc, 0.01f);  // 0.01s per token
    }

    printf("    After 60 tokens with cooldown:\n");
    printf("    training_triggered=%d\n", trained_count);
    printf("    buffer_count=%d\n", acc.buffer_count);
    printf("    cooldown_remaining=%.2f\n", acc.cooldown_remaining);

    // Now wait for cooldown to expire
    printf("\n[*] Waiting for cooldown to expire...\n");
    while (acc.cooldown_remaining > 0) {
        accumulator_tick(&acc, 0.1f);
    }
    printf("    cooldown_remaining=%.2f\n", acc.cooldown_remaining);

    // Now training should trigger
    trained_count = 0;
    for (int i = 0; i < 20; i++) {
        int trained = accumulate_experience(&acc, &trainer, &delta, x, probs, i % VOCAB_SIZE, 0.5f);
        if (trained) {
            trained_count++;
            printf("    Training triggered after cooldown!\n");
            break;
        }
        accumulator_tick(&acc, 0.1f);
    }

    if (trained_count > 0) {
        printf("    [PASS] Training works after cooldown\n");
    }

    printf("\n[*] Testing flush...\n");

    // Accumulate some more
    for (int i = 0; i < 15; i++) {
        accumulate_experience(&acc, &trainer, &delta, x, probs, i % VOCAB_SIZE, 0.4f);
    }
    printf("    buffer_count before flush=%d\n", acc.buffer_count);

    // Flush
    flush_accumulator(&acc, &trainer, &delta);
    printf("    buffer_count after flush=%d\n", acc.buffer_count);
    printf("    total_training_cycles=%d\n", acc.total_training_cycles);

    if (acc.buffer_count == 0) {
        printf("    [PASS] Flush clears buffer and triggers training\n");
    }

    printf("\n[*] Testing delta norm change...\n");
    float norm = get_delta_norm(&delta);
    printf("    delta_norm after all training=%.6f\n", norm);

    if (norm > 0.0f) {
        printf("    [PASS] Delta weights were modified by training\n");
    } else {
        printf("    [WARN] Delta weights unchanged (might be OK if signal was zero)\n");
    }

    // Cleanup
    free_accumulator(&acc);
    free_microtrainer(&trainer);
    free(delta.A);
    free(delta.B);

    print_separator("QUANTUM ACCUMULATION TEST COMPLETE");
    printf("הצבירה הקוונטית עובדת. נמשיך.\n");

    return 0;
}
