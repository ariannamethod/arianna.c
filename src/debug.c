/*
 * Debug: trace forward pass step by step
 */

#include "arianna.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s weights.bin\n", argv[0]);
        return 1;
    }

    Transformer t;
    printf("Loading weights from %s...\n", argv[1]);
    if (load_weights(&t, argv[1]) != 0) {
        printf("Failed to load weights\n");
        return 1;
    }

    Config* c = &t.config;
    printf("Config: dim=%d, layers=%d, heads=%d\n", c->dim, c->n_layers, c->n_heads);

    // Test with single token 'S' = 83
    int token = 83;
    printf("\nToken: '%c' (%d)\n", token, token);

    // Check embedding
    float* emb = t.weights.token_embedding + token * c->dim;
    printf("Embedding first 5: %.6f %.6f %.6f %.6f %.6f\n",
           emb[0], emb[1], emb[2], emb[3], emb[4]);

    // Check first wq
    float* wq = t.weights.wq;  // First layer
    printf("wq[0] first 5: %.6f %.6f %.6f %.6f %.6f\n",
           wq[0], wq[1], wq[2], wq[3], wq[4]);

    // Check ln1
    float* ln1 = t.weights.ln1_weight;
    printf("ln1 first 5: %.6f %.6f %.6f %.6f %.6f\n",
           ln1[0], ln1[1], ln1[2], ln1[3], ln1[4]);

    // Manual RMSNorm
    float rms = 0.0f;
    for (int i = 0; i < c->dim; i++) {
        rms += emb[i] * emb[i];
    }
    rms = sqrtf(rms / c->dim + 1e-5f);

    float xb[128];
    for (int i = 0; i < c->dim; i++) {
        xb[i] = (emb[i] / rms) * ln1[i];
    }
    printf("After ln1 first 5: %.6f %.6f %.6f %.6f %.6f\n",
           xb[0], xb[1], xb[2], xb[3], xb[4]);

    // Manual Q = xb @ wq
    float q[128] = {0};
    for (int j = 0; j < c->dim; j++) {
        for (int l = 0; l < c->dim; l++) {
            q[j] += xb[l] * wq[l * c->dim + j];
        }
    }
    printf("Q first 5: %.6f %.6f %.6f %.6f %.6f\n",
           q[0], q[1], q[2], q[3], q[4]);

    // Expected from Python:
    // After ln1: 0.247379, -1.297098, 0.361640, -1.332417, 1.343552
    // Q: -0.567234, -0.274548, -0.511646, 0.042035, 0.525874

    // Now do a full forward pass for "She "
    printf("\n=== Full forward pass for 'She ' ===\n");
    char* prompt = "She ";
    int tokens[4];
    int n_tokens = 4;
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = (unsigned char)prompt[i];
    }
    // Process each token
    for (int pos = 0; pos < n_tokens; pos++) {
        forward(&t, tokens, n_tokens, pos);
    }

    // Check logits
    printf("Logits first 5: %.4f %.4f %.4f %.4f %.4f\n",
           t.state.logits[0], t.state.logits[1], t.state.logits[2],
           t.state.logits[3], t.state.logits[4]);

    // Find top predictions
    printf("Top predictions:\n");
    for (int top = 0; top < 5; top++) {
        int best_idx = 0;
        float best_val = -1e10;
        for (int i = 0; i < c->vocab_size; i++) {
            if (t.state.logits[i] > best_val) {
                best_val = t.state.logits[i];
                best_idx = i;
            }
        }
        printf("  %d. '%c' (%d): %.4f\n", top+1,
               best_idx >= 32 && best_idx < 127 ? best_idx : '?',
               best_idx, best_val);
        t.state.logits[best_idx] = -1e10;  // mask out
    }

    free_transformer(&t);
    return 0;
}
