/*
 * arianna.c - Personality Weights Transformer (GPT-2 Architecture)
 * "Who I am", not "What I know"
 *
 * Architecture: GPT-2 style (LayerNorm, position embedding, GELU)
 * Tokenization: char-level
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
// Default Constants (for compatibility with arianna_dynamic.c)
// These are defaults; actual values come from Config at runtime
// ============================================================

#ifndef DIM
#define DIM 384
#endif

#ifndef N_LAYERS
#define N_LAYERS 6
#endif

#ifndef N_HEADS
#define N_HEADS 6
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 256
#endif

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 1536
#endif

// ============================================================
// Model Structures (GPT-2 style)
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
    // Embeddings
    float* wte;             // token embedding [vocab_size, dim]
    float* wpe;             // position embedding [max_seq_len, dim]

    // Per-layer weights (GPT-2 style)
    float* ln1_weight;      // [n_layers, dim]
    float* ln1_bias;        // [n_layers, dim]
    float* c_attn_weight;   // [n_layers, dim, 3*dim] combined QKV
    float* c_attn_bias;     // [n_layers, 3*dim]
    float* c_proj_weight;   // [n_layers, dim, dim] attention output
    float* c_proj_bias;     // [n_layers, dim]
    float* ln2_weight;      // [n_layers, dim]
    float* ln2_bias;        // [n_layers, dim]
    float* c_fc_weight;     // [n_layers, dim, hidden_dim]
    float* c_fc_bias;       // [n_layers, hidden_dim]
    float* c_proj2_weight;  // [n_layers, hidden_dim, dim]
    float* c_proj2_bias;    // [n_layers, dim]

    // Final layer norm
    float* ln_f_weight;     // [dim]
    float* ln_f_bias;       // [dim]

    // lm_head (can be tied with wte)
    float* lm_head;         // [dim, vocab_size] or NULL if tied
} Weights;

typedef struct {
    // Activation buffers
    float* x;           // [max_seq_len, dim] hidden states
    float* xb;          // [dim] buffer for layer norm output
    float* qkv;         // [3*dim] QKV projection
    float* attn_out;    // [dim] attention output
    float* ffn_buf;     // [hidden_dim] FFN intermediate
    float* logits;      // [vocab_size] output logits

    // KV cache
    float* key_cache;   // [n_layers, max_seq_len, dim]
    float* value_cache; // [n_layers, max_seq_len, dim]
    int cache_len;
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

// Core operations (GPT-2 style)
void layer_norm(float* out, float* x, float* weight, float* bias, int size);
void gelu(float* x, int size);
void softmax(float* x, int size);
void matmul(float* out, float* x, float* w, int n, int d);
void matmul_add(float* out, float* x, float* w, float* b, int n, int d);

// Forward pass
void forward(Transformer* t, int token, int pos);

// Inference
int sample(Transformer* t, float temperature);
void generate(Transformer* t, const char* prompt, int max_tokens, float temperature);

// I/O
int load_weights(Transformer* t, const char* path);

// Char tokenization
int char_to_token(char c);
char token_to_char(int token);

// Vocab management
int load_vocab(const char* path);

#endif // ARIANNA_H
