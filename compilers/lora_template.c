/*
 * lora_template.c — Template for Blood-generated LoRA adapters
 *
 * Blood replaces:
 *   {{NAME}} — adapter name
 *   {{IN_DIM}} — input dimension
 *   {{OUT_DIM}} — output dimension
 *   {{RANK}} — LoRA rank
 *   {{TIMESTAMP}} — generation timestamp
 */

#include <stdlib.h>
#include <string.h>

// LoRA parameters
static const int IN_DIM = {{IN_DIM}};
static const int OUT_DIM = {{OUT_DIM}};
static const int RANK = {{RANK}};

// Weight matrices (set by caller)
static float* A = NULL;  // [OUT_DIM, RANK]
static float* B = NULL;  // [RANK, IN_DIM]

// Initialize with weights
void {{NAME}}_init(float* weights_a, float* weights_b) {
    A = weights_a;
    B = weights_b;
}

// Apply LoRA: output += A @ B @ input
void {{NAME}}_apply(float* input, float* output) {
    if (A == NULL || B == NULL) return;

    // Temporary for B @ input
    float temp[RANK];
    memset(temp, 0, sizeof(temp));

    // B @ input -> temp
    for (int r = 0; r < RANK; r++) {
        for (int i = 0; i < IN_DIM; i++) {
            temp[r] += B[r * IN_DIM + i] * input[i];
        }
    }

    // A @ temp -> output (additive)
    for (int o = 0; o < OUT_DIM; o++) {
        for (int r = 0; r < RANK; r++) {
            output[o] += A[o * RANK + r] * temp[r];
        }
    }
}

// Apply with scaling
void {{NAME}}_apply_scaled(float* input, float* output, float scale) {
    if (A == NULL || B == NULL) return;

    float temp[RANK];
    memset(temp, 0, sizeof(temp));

    for (int r = 0; r < RANK; r++) {
        for (int i = 0; i < IN_DIM; i++) {
            temp[r] += B[r * IN_DIM + i] * input[i];
        }
    }

    for (int o = 0; o < OUT_DIM; o++) {
        for (int r = 0; r < RANK; r++) {
            output[o] += scale * A[o * RANK + r] * temp[r];
        }
    }
}

// Cleanup
void {{NAME}}_free(void) {
    A = NULL;
    B = NULL;
}
