// gpt2_30m.h — Minimal GPT2-30M inference for Pandora
// "External brain for vocabulary extraction"
//
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef GPT2_30M_H
#define GPT2_30M_H

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

#define GPT2_30M_VOCAB_SIZE    50257
#define GPT2_30M_CONTEXT_LEN   1024
#define GPT2_30M_EMBED_DIM     256
#define GPT2_30M_N_HEADS       4
#define GPT2_30M_N_LAYERS      6

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Token embeddings [vocab_size, embed_dim]
    float* wte;
    // Position embeddings [context_len, embed_dim]
    float* wpe;

    // Transformer blocks
    struct {
        // Layer norm 1
        float* ln1_g;  // [embed_dim]
        float* ln1_b;  // [embed_dim]

        // Attention
        float* attn_qkv;  // [3 * embed_dim, embed_dim]
        float* attn_proj; // [embed_dim, embed_dim]

        // Layer norm 2
        float* ln2_g;  // [embed_dim]
        float* ln2_b;  // [embed_dim]

        // MLP
        float* mlp_fc;   // [4 * embed_dim, embed_dim]
        float* mlp_proj; // [embed_dim, 4 * embed_dim]
    } blocks[GPT2_30M_N_LAYERS];

    // Final layer norm
    float* ln_f_g;
    float* ln_f_b;

    // Output projection (shared with wte in GPT-2)
    // Uses wte transposed

    // Loaded flag
    int loaded;

    // Scratch buffers
    float* scratch1;
    float* scratch2;
    float* qkv_buf;
    float* attn_buf;
} GPT2_30M;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize model (allocates memory)
int gpt2_30m_init(GPT2_30M* model);

// Load weights from binary file
int gpt2_30m_load(GPT2_30M* model, const char* path);

// Free model memory
void gpt2_30m_free(GPT2_30M* model);

// Forward pass - compute logits for next token
// Returns logits array [vocab_size]
float* gpt2_30m_forward(GPT2_30M* model, const int* tokens, int n_tokens);

// Generate tokens
// Returns number of tokens generated
int gpt2_30m_generate(
    GPT2_30M* model,
    const int* prompt_tokens,
    int n_prompt,
    int* output_tokens,
    int max_tokens,
    float temperature
);

// Sample from logits
int gpt2_30m_sample(const float* logits, int vocab_size, float temperature);

// ═══════════════════════════════════════════════════════════════════════════════
// VOCAB (BPE tokenizer)
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char** id_to_token;  // [vocab_size]
    // For encoding we'd need a trie/hash, but for Pandora we only decode
    int loaded;
} GPT2Vocab;

int gpt2_vocab_load(GPT2Vocab* vocab, const char* path);
void gpt2_vocab_free(GPT2Vocab* vocab);
const char* gpt2_vocab_decode(GPT2Vocab* vocab, int token_id);

#ifdef __cplusplus
}
#endif

#endif // GPT2_30M_H
