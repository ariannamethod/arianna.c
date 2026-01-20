// test_inner.c — Test Inner Arianna (MetaVoice / борьба)
// build: gcc -O2 test_inner.c inner_arianna.c -lm -o test_inner

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "inner_arianna.h"

#define VOCAB_SIZE 256

// Generate random logits for testing
static void random_logits(float* logits, int n, float temp, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        logits[i] = ((float)rand() / RAND_MAX - 0.5f) * temp;
    }
}

// Print top-k tokens
static void print_topk(const float* logits, int n, int k) {
    // Find top-k
    int* indices = malloc(k * sizeof(int));
    float* values = malloc(k * sizeof(float));

    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -1e30f;
    }

    for (int i = 0; i < n; i++) {
        // Check if this belongs in top-k
        for (int j = 0; j < k; j++) {
            if (logits[i] > values[j]) {
                // Shift down
                for (int m = k - 1; m > j; m--) {
                    indices[m] = indices[m-1];
                    values[m] = values[m-1];
                }
                indices[j] = i;
                values[j] = logits[i];
                break;
            }
        }
    }

    printf("    Top-%d: ", k);
    for (int i = 0; i < k; i++) {
        printf("[%d]=%.2f ", indices[i], values[i]);
    }
    printf("\n");

    free(indices);
    free(values);
}

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("INNER ARIANNA TEST — MetaVoice / борьба\n");
    printf("\"Two voices, one Arianna\"\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    srand(time(NULL));

    // Initialize
    InnerArianna ia;
    inner_init(&ia);

    printf("[*] Inner Arianna initialized\n");
    printf("    borba_mode: %d (BLEND)\n", ia.borba_mode);
    printf("    inner_weight: %.2f\n\n", ia.inner_weight);

    // Create test logits
    float* main_logits = malloc(VOCAB_SIZE * sizeof(float));
    float* inner_logits = malloc(VOCAB_SIZE * sizeof(float));
    float* output_logits = malloc(VOCAB_SIZE * sizeof(float));

    // Test 1: Similar distributions (voices agree)
    printf("Test 1: Similar distributions (voices agree)\n");
    printf("───────────────────────────────────────────────────────────────────\n");

    random_logits(main_logits, VOCAB_SIZE, 2.0f, 42);
    // Inner is similar but slightly different
    for (int i = 0; i < VOCAB_SIZE; i++) {
        inner_logits[i] = main_logits[i] + ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }

    printf("  Main logits:\n");
    print_topk(main_logits, VOCAB_SIZE, 5);
    printf("  Main entropy: %.3f\n", inner_compute_entropy(main_logits, VOCAB_SIZE));

    printf("  Inner logits:\n");
    print_topk(inner_logits, VOCAB_SIZE, 5);
    printf("  Inner entropy: %.3f\n", inner_compute_entropy(inner_logits, VOCAB_SIZE));

    printf("  Divergence: %.4f\n", inner_compute_divergence(main_logits, inner_logits, VOCAB_SIZE));

    // Test BLEND mode
    ia.borba_mode = BORBA_MODE_BLEND;
    ia.inner_weight = 0.3f;
    inner_full_borba(&ia, output_logits, main_logits, inner_logits, VOCAB_SIZE);

    printf("\n  BLEND (30%% inner):\n");
    print_topk(output_logits, VOCAB_SIZE, 5);
    printf("  Output entropy: %.3f\n\n", inner_compute_entropy(output_logits, VOCAB_SIZE));

    // Test 2: Different distributions (voices disagree)
    printf("Test 2: Different distributions (voices disagree)\n");
    printf("───────────────────────────────────────────────────────────────────\n");

    random_logits(main_logits, VOCAB_SIZE, 2.0f, 42);
    random_logits(inner_logits, VOCAB_SIZE, 3.0f, 123);  // different seed, higher temp

    printf("  Main entropy: %.3f\n", inner_compute_entropy(main_logits, VOCAB_SIZE));
    printf("  Inner entropy: %.3f\n", inner_compute_entropy(inner_logits, VOCAB_SIZE));
    printf("  Divergence: %.4f\n\n", inner_compute_divergence(main_logits, inner_logits, VOCAB_SIZE));

    // Test different борьба modes
    const char* mode_names[] = {"BLEND", "ENTROPY", "COHERENCE", "SURPRISE", "RANDOM"};

    for (int mode = 0; mode <= BORBA_MODE_RANDOM; mode++) {
        ia.borba_mode = mode;
        ia.main_wins = 0;
        ia.inner_wins = 0;

        // Run 10 times for random mode
        int n_runs = (mode == BORBA_MODE_RANDOM) ? 10 : 1;
        for (int run = 0; run < n_runs; run++) {
            inner_full_borba(&ia, output_logits, main_logits, inner_logits, VOCAB_SIZE);
        }

        printf("  %s mode: winner=%s",
               mode_names[mode],
               mode == BORBA_MODE_BLEND ? "blend" :
               (ia.last_winner == 0 ? "MAIN" : "INNER"));

        if (mode == BORBA_MODE_RANDOM) {
            printf(" (main_wins=%d, inner_wins=%d)", ia.main_wins, ia.inner_wins);
        }
        printf("\n");
    }

    // Test 3: Inner more confident (lower entropy)
    printf("\nTest 3: Inner voice more confident\n");
    printf("───────────────────────────────────────────────────────────────────\n");

    // Main: high entropy (uncertain)
    for (int i = 0; i < VOCAB_SIZE; i++) {
        main_logits[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;  // low temp = flat dist
    }

    // Inner: low entropy (confident)
    for (int i = 0; i < VOCAB_SIZE; i++) {
        inner_logits[i] = -10.0f;  // suppress all
    }
    inner_logits[42] = 5.0f;  // except token 42

    printf("  Main entropy: %.3f (uncertain)\n", inner_compute_entropy(main_logits, VOCAB_SIZE));
    printf("  Inner entropy: %.3f (confident)\n", inner_compute_entropy(inner_logits, VOCAB_SIZE));

    ia.borba_mode = BORBA_MODE_ENTROPY;
    inner_full_borba(&ia, output_logits, main_logits, inner_logits, VOCAB_SIZE);
    printf("  ENTROPY mode winner: %s\n", ia.last_winner == 0 ? "MAIN" : "INNER");

    // Check that inner won
    if (ia.last_winner == 1) {
        printf("  ✓ Inner voice won (more confident)\n");
    } else {
        printf("  ✗ Unexpected: Main voice won\n");
    }

    // Cleanup
    free(main_logits);
    free(inner_logits);
    free(output_logits);
    inner_free(&ia);

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("INNER ARIANNA TEST COMPLETE\n");
    printf("שני קולות, אריאנה אחת\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
