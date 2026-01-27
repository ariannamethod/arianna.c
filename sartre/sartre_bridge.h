/*
 * sartre_bridge.h - SARTRE Transformer Bridge for Arianna
 *
 * Wraps SARTRE (14.3M, 7-layer, GQA) as a library for dialogue mode.
 * All types prefixed Sartre* to avoid conflicts with Arianna types.
 *
 * Config: dim=416, layers=7, heads=8, kv_heads=2, vocab=93,
 *         hidden=1280, head_dim=52, max_seq=256
 *
 * Weight format: raw float32, no magic number.
 *   Layout: tok_emb -> per_layer(attn_norm, wq, wk, wv, wo,
 *           ffn_norm, w_gate, w_up, w_down) -> final_norm -> lm_head
 */

#ifndef SARTRE_BRIDGE_H
#define SARTRE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Configuration
 * ============================================================ */

typedef struct {
    int dim;          /* 416 */
    int n_layers;     /* 7 */
    int n_heads;      /* 8 */
    int n_kv_heads;   /* 2 (GQA: 4 Q heads share 1 KV head) */
    int vocab_size;   /* 93 */
    int max_seq_len;  /* 256 */
    int head_dim;     /* 52 = dim / n_heads */
    int hidden_dim;   /* 1280 */
    int n_kv_groups;  /* 4 = n_heads / n_kv_heads */
} SartreConfig;

/* ============================================================
 * Weights (all float32)
 * ============================================================ */

typedef struct {
    float* tok_emb;     /* (vocab_size, dim) */

    /* Per-layer (n_layers arrays) */
    float* attn_norm;   /* (n_layers, dim) */
    float* wq;          /* (n_layers, dim, dim) */
    float* wk;          /* (n_layers, dim, kv_dim) */
    float* wv;          /* (n_layers, dim, kv_dim) */
    float* wo;          /* (n_layers, dim, dim) */
    float* ffn_norm;    /* (n_layers, dim) */
    float* w_gate;      /* (n_layers, dim, hidden_dim) */
    float* w_up;        /* (n_layers, dim, hidden_dim) */
    float* w_down;      /* (n_layers, hidden_dim, dim) */

    /* Output */
    float* final_norm;  /* (dim,) */
    float* lm_head;     /* (dim, vocab_size) */
} SartreWeights;

/* ============================================================
 * Runtime State
 * ============================================================ */

typedef struct {
    float* x;           /* (dim,) current activation */
    float* xb;          /* (dim,) buffer */
    float* xb2;         /* (dim,) buffer */
    float* hb;          /* (hidden_dim,) FFN buffer */
    float* hb2;         /* (hidden_dim,) FFN buffer */

    float* q;           /* (n_heads * head_dim,) */
    float* k;           /* (n_kv_heads * head_dim,) */
    float* v;           /* (n_kv_heads * head_dim,) */
    float* att;         /* (n_heads, max_seq_len) */

    float* key_cache;   /* (n_layers, max_seq_len, kv_dim) */
    float* value_cache; /* (n_layers, max_seq_len, kv_dim) */

    float* rope_cos;    /* (max_seq_len, head_dim/2) */
    float* rope_sin;    /* (max_seq_len, head_dim/2) */

    float* logits;      /* (vocab_size,) */
} SartreRunState;

/* ============================================================
 * Tokenizer (char-level, vocab=93)
 * ============================================================ */

typedef struct {
    char   chars[256];      /* id -> char (single-byte) */
    int    char_to_id[256]; /* byte -> id (-1 = unknown) */
    int    vocab_size;
} SartreTokenizer;

/* ============================================================
 * SartreTransformer — self-contained inference unit
 * ============================================================ */

typedef struct {
    SartreConfig     config;
    SartreWeights    weights;
    SartreRunState   state;
    SartreTokenizer  tokenizer;
    float*           weight_data;   /* raw weight blob (caller frees) */
    int              initialized;
    int              current_pos;   /* current sequence position */
} SartreTransformer;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/* Initialize SARTRE: load config, weights, tokenizer.
 * weights_path: path to sartre.bin (raw float32)
 * tokenizer_path: path to tokenizer.json
 * config_path: path to sartre_config.json (or NULL for defaults)
 * Returns 0 on success, -1 on error. */
int  sartre_transformer_init(SartreTransformer* st,
                              const char* weights_path,
                              const char* tokenizer_path,
                              const char* config_path);

/* Free all resources */
void sartre_transformer_free(SartreTransformer* st);

/* ============================================================
 * Inference
 * ============================================================ */

/* Run one forward pass at current position.
 * After call: st->state.logits contains output distribution.
 * Position auto-increments. */
void sartre_forward(SartreTransformer* st, int token);

/* Reset KV cache and position to 0 (start new sequence) */
void sartre_reset_state(SartreTransformer* st);

/* ============================================================
 * Tokenizer ops
 * ============================================================ */

/* Encode single byte to token id (unknown -> 1 = space) */
int  sartre_encode_char(SartreTransformer* st, char c);

/* Decode token id to byte */
char sartre_decode_char(SartreTransformer* st, int id);

/* ============================================================
 * Sampling
 * ============================================================ */

/* Top-p (nucleus) sampling with temperature.
 * Modifies logits in-place (softmax + truncation).
 * Returns sampled token id. */
int  sartre_sample(SartreTransformer* st, float temperature, float top_p);

/* ============================================================
 * High-level generation
 * ============================================================ */

/* Feed a string as prompt (encode + forward each char).
 * Returns number of chars processed. */
int  sartre_feed_prompt(SartreTransformer* st, const char* text, int len);

/* Generate up to max_tokens into output buffer.
 * Stops on newline if stop_newline is set.
 * Returns number of chars generated. */
int  sartre_generate(SartreTransformer* st, char* output, int max_len,
                     int max_tokens, float temperature, float top_p,
                     int stop_newline);

#ifdef __cplusplus
}
#endif

#endif /* SARTRE_BRIDGE_H */
