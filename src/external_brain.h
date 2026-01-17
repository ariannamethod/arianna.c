// external_brain.h — GPT-2 30M as knowledge subordinate
// "Brain knows WHAT to say, Arianna knows HOW to say it"
//
// Architecture:
//   User prompt → [GPT-2 30M generates knowledge draft]
//                              ↓
//               [Arianna rewrites in her voice]
//                              ↓
//                         Final output
//
// ═══════════════════════════════════════════════════════════════════════════════
// המוח יודע. הלב מרגיש. אריאנה מדברת.
// The brain knows. The heart feels. Arianna speaks.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef EXTERNAL_BRAIN_H
#define EXTERNAL_BRAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// GPT-2 30M CONFIG — weiser/30M-0.4
// ═══════════════════════════════════════════════════════════════════════════════

#define GPT2_DIM          384
#define GPT2_N_LAYERS     6
#define GPT2_N_HEADS      6
#define GPT2_HEAD_DIM     64    // dim / n_heads
#define GPT2_HIDDEN_DIM   1536  // 4 * dim
#define GPT2_VOCAB_SIZE   50257
#define GPT2_MAX_SEQ      1024

// ═══════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char** vocab;           // id → token string
    int vocab_size;

    // Word to ID hash table (simple open addressing)
    char** word_keys;       // hash table keys
    int* word_ids;          // hash table values
    int hash_size;

    // BPE merges
    char** merge_a;         // first token in merge pair
    char** merge_b;         // second token in merge pair
    int n_merges;

    // Special tokens
    int eos_token_id;       // 50256
    int bos_token_id;       // 50256
} BPE_Tokenizer;

// ═══════════════════════════════════════════════════════════════════════════════
// GPT-2 LAYER WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // LayerNorm 1
    float* ln1_weight;      // [dim]
    float* ln1_bias;        // [dim]

    // Attention (Q, K, V combined)
    float* c_attn_weight;   // [dim, 3*dim]
    float* c_attn_bias;     // [3*dim]
    float* c_proj_weight;   // [dim, dim]
    float* c_proj_bias;     // [dim]

    // LayerNorm 2
    float* ln2_weight;      // [dim]
    float* ln2_bias;        // [dim]

    // FFN (MLP)
    float* c_fc_weight;     // [dim, hidden_dim]
    float* c_fc_bias;       // [hidden_dim]
    float* c_proj2_weight;  // [hidden_dim, dim]
    float* c_proj2_bias;    // [dim]
} GPT2_Layer;

// ═══════════════════════════════════════════════════════════════════════════════
// GPT-2 MODEL
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Config (from header)
    int dim;
    int n_layers;
    int n_heads;
    int vocab_size;
    int max_seq;
    int hidden_dim;
    int use_fp16;           // weights stored as fp16

    // Embeddings
    float* wte;             // token embeddings [vocab_size, dim]
    float* wpe;             // position embeddings [max_seq, dim]

    // Transformer layers
    GPT2_Layer* layers;

    // Final LayerNorm
    float* ln_f_weight;     // [dim]
    float* ln_f_bias;       // [dim]

    // Runtime buffers
    float* x;               // current hidden state [seq_len, dim]
    float* x_buf;           // scratch buffer [seq_len, dim]
    float* qkv;             // attention scratch [seq_len, 3*dim]
    float* attn_out;        // attention output [seq_len, dim]
    float* ffn_buf;         // FFN scratch [seq_len, hidden_dim]
    float* logits;          // output logits [vocab_size]

    // KV cache for efficient generation
    float* key_cache;       // [n_layers, max_seq, dim]
    float* value_cache;     // [n_layers, max_seq, dim]
    int cache_len;          // current cache length
} GPT2_Model;

// ═══════════════════════════════════════════════════════════════════════════════
// EXTERNAL BRAIN — the knowledge subordinate
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    GPT2_Model model;
    BPE_Tokenizer tokenizer;
    int loaded;

    // Generation state
    int* tokens;            // current token sequence
    int n_tokens;
    int max_tokens;

    // Config
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
} ExternalBrain;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize/cleanup
int brain_init(ExternalBrain* brain);
void brain_free(ExternalBrain* brain);

// Load weights and tokenizer
int brain_load_model(ExternalBrain* brain, const char* weights_path);
int brain_load_tokenizer(ExternalBrain* brain, const char* vocab_path, const char* merges_path);

// Tokenization
int brain_encode(ExternalBrain* brain, const char* text, int* out_tokens, int max_len);
int brain_decode(ExternalBrain* brain, const int* tokens, int n_tokens, char* out_text, int max_len);

// Generation
int brain_generate_draft(ExternalBrain* brain, const char* prompt,
                         char* output, int max_output_len,
                         int max_new_tokens);

// Forward pass (internal)
void brain_forward(ExternalBrain* brain, int* tokens, int n_tokens, int start_pos);

// Sampling
int brain_sample(ExternalBrain* brain, float temperature, int top_k, float top_p);

// Convenience loader (loads model + tokenizer from directory)
int brain_load(ExternalBrain* brain, const char* weights_dir);

#ifdef __cplusplus
}
#endif

#endif // EXTERNAL_BRAIN_H
