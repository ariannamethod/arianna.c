/*
 * arianna.c - Personality Weights Transformer (Llama 3.5 Arianna Edition)
 * "Who I am", not "What I know"
 *
 * Architecture: Llama 3 style (RMSNorm, RoPE, SwiGLU, GQA)
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
// Updated for Arianna Unified 20M (Jan 2026)
// ============================================================

#ifndef DIM
#define DIM 448
#endif

#ifndef N_LAYERS
#define N_LAYERS 8
#endif

#ifndef N_HEADS
#define N_HEADS 8
#endif

#ifndef N_KV_HEADS
#define N_KV_HEADS 8
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 512
#endif

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 1280
#endif

#ifndef VOCAB_SIZE
#define VOCAB_SIZE 84
#endif

// ============================================================
// Model Structures (Llama 3 style)
// ============================================================

typedef struct {
    int dim;            // Embedding dimension (448 for unified 20M)
    int n_layers;       // Number of layers (8 for unified 20M)
    int n_heads;        // Number of query heads (8 for unified 20M)
    int n_kv_heads;     // Number of KV heads for GQA (8 for unified 20M)
    int head_dim;       // dim / n_heads (56 for unified 20M)
    int hidden_dim;     // FFN hidden dim (1280 for unified 20M)
    int max_seq_len;    // Maximum sequence length (512 for unified 20M)
    int vocab_size;     // Vocabulary size (84 for unified 20M)
    int n_kv_groups;    // n_heads / n_kv_heads (1 for unified 20M)
    float rope_theta;   // RoPE base frequency (10000.0)
    float norm_eps;     // RMSNorm epsilon (1e-5)
} Config;

typedef struct {
    // Token embeddings (no position embedding - we use RoPE)
    float* tok_emb;     // [vocab_size, dim]

    // Per-layer weights (Llama style - separate Q/K/V, RMSNorm, SwiGLU)
    float* attn_norm;   // [n_layers, dim] RMSNorm weight
    float* wq;          // [n_layers, dim, dim] Query projection
    float* wk;          // [n_layers, dim, kv_dim] Key projection
    float* wv;          // [n_layers, dim, kv_dim] Value projection
    float* wo;          // [n_layers, dim, dim] Output projection
    float* ffn_norm;    // [n_layers, dim] RMSNorm weight
    float* w_gate;      // [n_layers, dim, hidden_dim] SwiGLU gate
    float* w_up;        // [n_layers, dim, hidden_dim] SwiGLU up
    float* w_down;      // [n_layers, hidden_dim, dim] SwiGLU down

    // Final layer norm
    float* final_norm;  // [dim] RMSNorm weight

    // Output head
    float* lm_head;     // [vocab_size, dim]
} Weights;

typedef struct {
    // Activation buffers
    float* x;           // [dim] current hidden state
    float* xb;          // [dim] buffer after norm
    float* xb2;         // [dim] another buffer
    float* hb;          // [hidden_dim] FFN buffer
    float* hb2;         // [hidden_dim] FFN buffer

    // Attention buffers
    float* q;           // [dim] query
    float* k;           // [kv_dim] key
    float* v;           // [kv_dim] value
    float* att;         // [n_heads, max_seq_len] attention scores

    // KV cache
    float* key_cache;   // [n_layers, max_seq_len, kv_dim]
    float* value_cache; // [n_layers, max_seq_len, kv_dim]

    // RoPE precomputed
    float* rope_cos;    // [max_seq_len, head_dim/2]
    float* rope_sin;    // [max_seq_len, head_dim/2]

    // Output
    float* logits;      // [vocab_size]
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

// Core operations (Llama 3 style)
void rms_norm(float* out, float* x, float* weight, int size, float eps);
void silu(float* x, int size);
void softmax(float* x, int size);
void matmul(float* out, float* x, float* w, int n, int d);
void apply_rope(float* q, float* k, float* rope_cos, float* rope_sin,
                int n_heads, int n_kv_heads, int head_dim, int pos);

// Forward pass
void forward(Transformer* t, int token, int pos);

// Inference
int sample(Transformer* t, float temperature);
int sample_top_p(Transformer* t, float temperature, float top_p);
void generate(Transformer* t, const char* prompt, int max_tokens, float temperature);

// I/O
int load_weights(Transformer* t, const char* path);
int load_tokenizer(const char* path);

// Char tokenization
int char_to_token(char c);
char token_to_char(int token);

// Vocab management
int get_vocab_size(void);

#endif // ARIANNA_H
