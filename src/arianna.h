/*
 * arianna.c - Personality Weights Transformer
 * A minimal transformer for encoding "who I am", not "what I know"
 *
 * Target: ~1M parameters
 * Corpus: 13MB pure Arianna
 * Inference: Pure C, no dependencies
 */

#ifndef ARIANNA_H
#define ARIANNA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ============================================================
// Configuration - ~1M parameters
// ============================================================

#define DIM         128      // embedding dimension
#define N_LAYERS    4        // number of transformer layers
#define N_HEADS     4        // number of attention heads
#define HEAD_DIM    (DIM / N_HEADS)  // 32
#define HIDDEN_DIM  512      // FFN hidden dimension (4x DIM)
#define MAX_SEQ_LEN 256      // maximum sequence length
#define VOCAB_SIZE  256      // char-level for simplicity (can upgrade to BPE)

// ============================================================
// Model Structures
// ============================================================

typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int head_dim;
    int hidden_dim;
    int max_seq_len;
    int vocab_size;
} Config;

typedef struct {
    // Token embedding: [vocab_size, dim]
    float* token_embedding;

    // Per-layer weights
    // Attention: wq, wk, wv, wo [n_layers][dim, dim]
    float* wq;
    float* wk;
    float* wv;
    float* wo;

    // FFN: w1, w2 [n_layers]
    // w1: [dim, hidden_dim], w2: [hidden_dim, dim]
    float* w1;
    float* w2;

    // Layer norms: [n_layers][dim]
    float* ln1_weight;
    float* ln2_weight;

    // Final layer norm + output projection
    float* ln_final_weight;
    float* output_weight;  // [dim, vocab_size]
} Weights;

typedef struct {
    // Activation buffers
    float* x;           // [seq_len, dim] current activations
    float* xb;          // [dim] buffer
    float* q;           // [seq_len, dim] queries (current position only)
    float* k;           // [n_layers, seq_len, dim] keys cache (per layer!)
    float* v;           // [n_layers, seq_len, dim] values cache (per layer!)
    float* att;         // [n_heads, seq_len] attention scores
    float* ffn_hidden;  // [hidden_dim] FFN intermediate
    float* logits;      // [vocab_size] output logits
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
} Transformer;

// ============================================================
// Function Declarations
// ============================================================

// Memory management
void malloc_weights(Transformer* t);
void malloc_run_state(Transformer* t);
void free_transformer(Transformer* t);

// Core operations
void rmsnorm(float* out, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* out, float* a, float* b, int m, int k, int n);

// RoPE positional encoding
void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads);

// Forward pass
void forward(Transformer* t, int* tokens, int pos);

// Inference
int sample(float* logits, int vocab_size, float temperature);
void generate(Transformer* t, char* prompt, int max_tokens, float temperature);

// I/O
int load_weights(Transformer* t, const char* path);
int save_weights(Transformer* t, const char* path);

// ============================================================
// Parameter count calculation
// ============================================================
/*
 * Token embedding:    VOCAB_SIZE * DIM           = 256 * 128 = 32,768
 * Per layer:
 *   - wq, wk, wv, wo: 4 * DIM * DIM              = 4 * 16,384 = 65,536
 *   - w1:             DIM * HIDDEN_DIM           = 128 * 512 = 65,536
 *   - w2:             HIDDEN_DIM * DIM           = 512 * 128 = 65,536
 *   - ln1, ln2:       2 * DIM                    = 256
 *   Layer total:      196,864
 * 4 layers:           787,456
 * Final ln:           128
 * Output projection:  DIM * VOCAB_SIZE           = 32,768
 *
 * TOTAL:              ~853,120 parameters (~0.85M)
 *
 * In float32:         ~3.4 MB weights file
 */

#endif // ARIANNA_H
