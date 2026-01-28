/*
 * nano.c — Pure C inference engine for nanochat GPT models.
 *
 * Supports:
 *   - FP16 weights (VERSION=1)
 *   - INT8 per-row quantized weights (VERSION=2)
 *   - INT4 per-row quantized weights (VERSION=3)
 *   - HYBRID q4h: embeddings q8, attention/MLP q4 (VERSION=4)
 *
 * Architecture: RoPE, RMSNorm (no params), ReLU², QK-Norm,
 *   Value Embeddings, Bigram Hash Embedding, Softcap (15*tanh),
 *   Sliding Window Attention, per-layer residual lambdas.
 *
 * Usage:
 *   ./nano <weights.bin> <tokenizer.tok> [options]
 *   ./nano weights/d12_arianna.bin tokenizer.tok -p "32759 32760 ... 32762"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Half-float (IEEE 754 binary16) to float conversion
// ---------------------------------------------------------------------------
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define MAGIC 0x4E414E4F
#define HEADER_SIZE 256
#define MAX_LAYERS 64

typedef struct {
    int n_layer;
    int n_embd;
    int n_head;
    int n_kv_head;
    int head_dim;
    int vocab_size;
    int padded_vocab;
    int seq_len;
    int bigram_vocab;
    int n_ve_layers;
    int window_pattern_len;
    uint8_t window_pattern[256];
    int quant_type;       // 0=fp16, 1=q8, 2=q4, 3=q4h (hybrid)
    // Derived
    int embed_quant;      // quant for embeddings (q8 for hybrid)
    int attn_quant;       // quant for attention/MLP (q4 for hybrid)
    int kv_dim;
    int mlp_dim;
    int ve_parity;
    int window_sizes[MAX_LAYERS];
    int has_ve[MAX_LAYERS];
} Config;

// ---------------------------------------------------------------------------
// Weight pointers — void* for data (fp16 or q8), float* for q8 scales
// Small tensors (lambdas, ve_gate) are always fp16 (uint16_t*)
// ---------------------------------------------------------------------------
typedef struct {
    // Embedding tables
    void*     wte;                      // [padded_vocab, n_embd]
    float*    wte_scales;               // q8: [padded_vocab], fp16: NULL
    void*     bigram_embed;             // [bigram_vocab, n_embd]
    float*    bigram_scales;

    // Scalar lambdas (always fp16)
    uint16_t* resid_lambdas;            // [n_layer]
    uint16_t* x0_lambdas;
    uint16_t* bigram_lambdas;

    // Per-layer
    void*     c_q[MAX_LAYERS];          // [n_head*head_dim, n_embd]
    float*    c_q_s[MAX_LAYERS];
    void*     c_k[MAX_LAYERS];
    float*    c_k_s[MAX_LAYERS];
    void*     c_v[MAX_LAYERS];
    float*    c_v_s[MAX_LAYERS];
    void*     c_proj[MAX_LAYERS];
    float*    c_proj_s[MAX_LAYERS];
    uint16_t* ve_gate[MAX_LAYERS];      // always fp16 (tiny), NULL if no VE

    void*     mlp_fc[MAX_LAYERS];       // [4*n_embd, n_embd]
    float*    mlp_fc_s[MAX_LAYERS];
    void*     mlp_proj[MAX_LAYERS];     // [n_embd, 4*n_embd]
    float*    mlp_proj_s[MAX_LAYERS];

    // Value embeddings
    void*     value_embeds[MAX_LAYERS]; // [vocab_size, kv_dim] or NULL
    float*    value_embeds_s[MAX_LAYERS];

    // LM head
    void*     lm_head;                  // [padded_vocab, n_embd]
    float*    lm_head_s;
} Weights;

// ---------------------------------------------------------------------------
// Run-time state
// ---------------------------------------------------------------------------
typedef struct {
    float* x;
    float* x0;
    float* x0_bigram;
    float* xn;
    float* q;
    float* k;
    float* v;
    float* att;
    float* y_att;
    float* hb;
    float* logits;
    float* key_cache;
    float* value_cache;
    float* cos_cache;
    float* sin_cache;
} RunState;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
#define TOK_MAGIC 0x4E544F4B

typedef struct {
    int vocab_size;
    int max_token_len;
    char** tokens;
    int* token_lens;
    int bos_id;
    int user_start_id;
    int user_end_id;
    int assistant_start_id;
    int assistant_end_id;
} Tokenizer;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
typedef struct {
    Config config;
    Weights weights;
    RunState state;
    void* mmap_data;
    size_t mmap_size;
    int fd;
} Model;

// ===== CORE MATH ==========================================================

static void rmsnorm(float* out, const float* x, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / size + 1e-6f);
    for (int i = 0; i < size; i++) out[i] = x[i] * inv_rms;
}

// FP16 matmul: out[rows] = W_f16[rows, cols] @ x[cols]
static void matmul_f16(float* out, const uint16_t* w, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint16_t* row = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            sum += half_to_float(row[j]) * x[j];
        }
        out[i] = sum;
    }
}

// Q8 matmul: out[rows] = (W_i8[rows, cols] * scales[rows]) @ x[cols]
static void matmul_q8(float* out, const int8_t* w, const float* scales, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const int8_t* row = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            sum += (float)row[j] * x[j];
        }
        out[i] = sum * scales[i];
    }
}

// Q4 matmul: out[rows] = (W_q4[rows, cols] * scales[rows]) @ x[cols]
// Data is packed: 2 values per byte, low nibble + high nibble, unsigned 0-15 -> signed -8..7
static void matmul_q4(float* out, const uint8_t* w, const float* scales, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t* row = w + (size_t)i * (cols / 2);
        for (int j = 0; j < cols; j += 2) {
            uint8_t packed = row[j / 2];
            int8_t v0 = (int8_t)(packed & 0x0F) - 8;       // low nibble
            int8_t v1 = (int8_t)((packed >> 4) & 0x0F) - 8; // high nibble
            sum += (float)v0 * x[j] + (float)v1 * x[j + 1];
        }
        out[i] = sum * scales[i];
    }
}

// Dispatched matmul
static void matmul(float* out, const void* w, const float* scales, const float* x,
                    int rows, int cols, int quant) {
    if (quant == 2) {
        matmul_q4(out, (const uint8_t*)w, scales, x, rows, cols);
    } else if (quant == 1) {
        matmul_q8(out, (const int8_t*)w, scales, x, rows, cols);
    } else {
        matmul_f16(out, (const uint16_t*)w, x, rows, cols);
    }
}

// Dispatched embedding lookup: out[dim] = embed[token_id]
static void embed_lookup(float* out, const void* emb, const float* scales,
                          int token, int dim, int quant) {
    if (quant == 2) {
        // Q4: packed 2 values per byte
        const uint8_t* data = (const uint8_t*)emb;
        float scale = scales[token];
        const uint8_t* row = data + (size_t)token * (dim / 2);
        for (int i = 0; i < dim; i += 2) {
            uint8_t packed = row[i / 2];
            out[i]     = (float)((int8_t)(packed & 0x0F) - 8) * scale;
            out[i + 1] = (float)((int8_t)((packed >> 4) & 0x0F) - 8) * scale;
        }
    } else if (quant == 1) {
        const int8_t* data = (const int8_t*)emb;
        float scale = scales[token];
        const int8_t* row = data + (size_t)token * dim;
        for (int i = 0; i < dim; i++) out[i] = (float)row[i] * scale;
    } else {
        const uint16_t* data = (const uint16_t*)emb;
        const uint16_t* row = data + (size_t)token * dim;
        for (int i = 0; i < dim; i++) out[i] = half_to_float(row[i]);
    }
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ===== ROPE ===============================================================

static void precompute_rope(RunState* s, const Config* c) {
    int half = c->head_dim / 2;
    for (int pos = 0; pos < c->seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)c->head_dim);
            float angle = (float)pos * freq;
            s->cos_cache[pos * half + i] = cosf(angle);
            s->sin_cache[pos * half + i] = sinf(angle);
        }
    }
}

static void apply_rope(float* vec, int pos, const RunState* s, int head_dim) {
    int half = head_dim / 2;
    const float* cr = s->cos_cache + pos * half;
    const float* sr = s->sin_cache + pos * half;
    for (int i = 0; i < half; i++) {
        float x1 = vec[i], x2 = vec[i + half];
        vec[i]        = x1 * cr[i] + x2 * sr[i];
        vec[i + half] = x1 * (-sr[i]) + x2 * cr[i];
    }
}

// ===== BIGRAM HASH ========================================================

static int bigram_hash(int curr, int prev, int bigram_vocab) {
    unsigned int h = ((unsigned int)(36313 * curr)) ^ ((unsigned int)(27191 * prev));
    return (int)(h % (unsigned int)(bigram_vocab - 1));
}

// ===== ALLOC STATE ========================================================

static void alloc_state(RunState* s, const Config* c) {
    int n = c->n_embd, kv = c->kv_dim, hd = c->head_dim;
    s->x         = calloc(n, sizeof(float));
    s->x0        = calloc(n, sizeof(float));
    s->x0_bigram = calloc(n, sizeof(float));
    s->xn        = calloc(n, sizeof(float));
    s->q         = calloc(c->n_head * hd, sizeof(float));
    s->k         = calloc(c->n_kv_head * hd, sizeof(float));
    s->v         = calloc(c->n_kv_head * hd, sizeof(float));
    s->att       = calloc((size_t)c->n_head * c->seq_len, sizeof(float));
    s->y_att     = calloc(c->n_head * hd, sizeof(float));
    s->hb        = calloc(c->mlp_dim, sizeof(float));
    s->logits    = calloc(c->padded_vocab, sizeof(float));
    size_t cs    = (size_t)c->n_layer * c->seq_len * kv;
    s->key_cache   = calloc(cs, sizeof(float));
    s->value_cache = calloc(cs, sizeof(float));
    int half = hd / 2;
    s->cos_cache = calloc((size_t)c->seq_len * half, sizeof(float));
    s->sin_cache = calloc((size_t)c->seq_len * half, sizeof(float));
    if (!s->x || !s->x0 || !s->x0_bigram || !s->xn || !s->q || !s->k || !s->v ||
        !s->att || !s->y_att || !s->hb || !s->logits || !s->key_cache ||
        !s->value_cache || !s->cos_cache || !s->sin_cache) {
        fprintf(stderr, "Failed to allocate state buffers — OOM\n"); exit(1);
    }
}

static void free_state(RunState* s) {
    free(s->x); free(s->x0); free(s->x0_bigram); free(s->xn);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->y_att); free(s->hb); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
}

// ===== LOAD MODEL =========================================================

// Helper: advance pointer for a Q8 matrix [rows, cols]
// Layout: float32 scales[rows] + int8 data[rows * cols]
static void load_q8_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *scales = (float*)(*ptr);
    *ptr += (size_t)rows * sizeof(float);
    *data = (void*)(*ptr);
    *ptr += (size_t)rows * cols * sizeof(int8_t);
}

// Helper: advance pointer for an FP16 matrix [rows, cols]
static void load_f16_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *data = (void*)(*ptr);
    *scales = NULL;
    *ptr += (size_t)rows * cols * sizeof(uint16_t);
}

// Helper: advance pointer for a Q4 matrix [rows, cols]
// Layout: float32 scales[rows] + packed uint8 data[rows * cols/2] (2 values per byte)
static void load_q4_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *scales = (float*)(*ptr);
    *ptr += (size_t)rows * sizeof(float);
    *data = (void*)(*ptr);
    *ptr += (size_t)rows * (cols / 2) * sizeof(uint8_t);
}

static void load_model(Model* m, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open weights"); exit(1); }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    m->mmap_data = data; m->mmap_size = file_size; m->fd = fd;

    // Parse header
    int32_t* ih = (int32_t*)data;
    if ((uint32_t)ih[0] != MAGIC) {
        fprintf(stderr, "Bad magic: 0x%08X\n", (uint32_t)ih[0]); exit(1);
    }
    Config* c = &m->config;
    c->n_layer     = ih[2];
    if (c->n_layer <= 0 || c->n_layer > MAX_LAYERS) {
        fprintf(stderr, "Invalid n_layer: %d (max %d)\n", c->n_layer, MAX_LAYERS);
        exit(1);
    }
    c->n_embd      = ih[3];
    c->n_head      = ih[4];
    c->n_kv_head   = ih[5];
    c->head_dim    = ih[6];
    c->vocab_size  = ih[7];
    c->padded_vocab= ih[8];
    c->seq_len     = ih[9];
    c->bigram_vocab= ih[10];
    c->n_ve_layers = ih[11];
    c->window_pattern_len = ih[12];
    if (c->window_pattern_len < 0 || c->window_pattern_len > 256) {
        fprintf(stderr, "Invalid window_pattern_len: %d\n", c->window_pattern_len);
        exit(1);
    }
    memcpy(c->window_pattern, (uint8_t*)data + 52, c->window_pattern_len);
    c->quant_type  = ih[16]; // offset 64 = index 16

    c->kv_dim = c->n_kv_head * c->head_dim;
    c->mlp_dim = 4 * c->n_embd;
    c->ve_parity = (c->n_layer - 1) % 2;
    // Set per-tensor quant types
    if (c->quant_type == 3) {
        // Hybrid: embeddings use q8, attention/MLP use q4
        c->embed_quant = 1;  // q8
        c->attn_quant = 2;   // q4
    } else {
        c->embed_quant = c->quant_type;
        c->attn_quant = c->quant_type;
    }
    for (int i = 0; i < c->n_layer; i++) {
        int pi = i % c->window_pattern_len;
        int is_long = (i == c->n_layer - 1) ? 1 : (int)c->window_pattern[pi];
        c->window_sizes[i] = is_long ? c->seq_len : c->seq_len / 2;
        c->has_ve[i] = (i % 2 == c->ve_parity) ? 1 : 0;
    }

    const char* qt = c->quant_type == 3 ? "Q4H" : (c->quant_type == 2 ? "INT4" : (c->quant_type == 1 ? "INT8" : "FP16"));
    printf("Config: n_layer=%d n_embd=%d n_head=%d n_kv_head=%d head_dim=%d [%s]\n",
           c->n_layer, c->n_embd, c->n_head, c->n_kv_head, c->head_dim, qt);
    printf("        vocab=%d padded=%d seq_len=%d bigram=%d ve=%d\n",
           c->vocab_size, c->padded_vocab, c->seq_len, c->bigram_vocab, c->n_ve_layers);

    // Walk file to set up weight pointers
    uint8_t* ptr = (uint8_t*)data + HEADER_SIZE;
    Weights* w = &m->weights;
    int eq = c->embed_quant;  // quant for embeddings
    int aq = c->attn_quant;   // quant for attention/MLP
    int n = c->n_embd, kv = c->kv_dim;

    // Macro for loading embedding matrices (uses embed_quant)
    #define LOAD_EMBED(data_ptr, scale_ptr, rows, cols) \
        if (eq == 2) { load_q4_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else if (eq == 1) { load_q8_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else { load_f16_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); }

    // Macro for loading attention/MLP matrices (uses attn_quant)
    #define LOAD_ATTN(data_ptr, scale_ptr, rows, cols) \
        if (aq == 2) { load_q4_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else if (aq == 1) { load_q8_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else { load_f16_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); }

    // 1. Token embedding (uses embed_quant)
    LOAD_EMBED(w->wte, w->wte_scales, c->padded_vocab, n);
    // 2. Bigram embedding (uses embed_quant)
    LOAD_EMBED(w->bigram_embed, w->bigram_scales, c->bigram_vocab, n);

    // 3. Lambdas (always fp16)
    w->resid_lambdas = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);
    w->x0_lambdas    = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);
    w->bigram_lambdas = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);

    // 4. Per-layer (uses attn_quant for attention/MLP)
    for (int i = 0; i < c->n_layer; i++) {
        LOAD_ATTN(w->c_q[i], w->c_q_s[i], c->n_head * c->head_dim, n);
        LOAD_ATTN(w->c_k[i], w->c_k_s[i], c->n_kv_head * c->head_dim, n);
        LOAD_ATTN(w->c_v[i], w->c_v_s[i], c->n_kv_head * c->head_dim, n);
        LOAD_ATTN(w->c_proj[i], w->c_proj_s[i], n, n);
        if (c->has_ve[i]) {
            w->ve_gate[i] = (uint16_t*)ptr;
            ptr += (size_t)c->n_kv_head * 32 * sizeof(uint16_t);
        } else {
            w->ve_gate[i] = NULL;
        }
        LOAD_ATTN(w->mlp_fc[i], w->mlp_fc_s[i], c->mlp_dim, n);
        LOAD_ATTN(w->mlp_proj[i], w->mlp_proj_s[i], n, c->mlp_dim);
    }

    // 5. Value embeddings (uses embed_quant)
    memset(w->value_embeds, 0, sizeof(w->value_embeds));
    memset(w->value_embeds_s, 0, sizeof(w->value_embeds_s));
    for (int i = 0; i < c->n_layer; i++) {
        if (c->has_ve[i]) {
            LOAD_EMBED(w->value_embeds[i], w->value_embeds_s[i], c->vocab_size, kv);
        }
    }

    // 6. LM head (uses embed_quant)
    LOAD_EMBED(w->lm_head, w->lm_head_s, c->padded_vocab, n);

    #undef LOAD_EMBED
    #undef LOAD_ATTN

    size_t consumed = (size_t)(ptr - (uint8_t*)data);
    if (consumed != file_size) {
        fprintf(stderr, "Warning: consumed %zu / %zu bytes (diff %zd)\n",
                consumed, file_size, (ssize_t)(file_size - consumed));
    } else {
        printf("Loaded %.1f MB from %s\n", file_size/1024.0/1024.0, path);
    }

    alloc_state(&m->state, c);
    precompute_rope(&m->state, c);
}

static void close_model(Model* m) {
    free_state(&m->state);
    munmap(m->mmap_data, m->mmap_size);
    close(m->fd);
}

// ===== TOKENIZER ==========================================================

static void load_tokenizer(Tokenizer* tok, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("open tokenizer"); exit(1); }
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fprintf(stderr, "Truncated tokenizer file\n"); exit(1); }
    if (magic != TOK_MAGIC) { fprintf(stderr, "Bad tok magic\n"); exit(1); }
    int32_t vs, ml;
    if (fread(&vs, 4, 1, f) != 1 || fread(&ml, 4, 1, f) != 1) {
        fprintf(stderr, "Truncated tokenizer header\n"); exit(1);
    }
    if (vs <= 0 || vs > 1000000) { fprintf(stderr, "Invalid vocab size: %d\n", vs); exit(1); }
    tok->vocab_size = vs; tok->max_token_len = ml;
    tok->tokens = malloc((size_t)vs * sizeof(char*));
    tok->token_lens = malloc((size_t)vs * sizeof(int));
    if (!tok->tokens || !tok->token_lens) { fprintf(stderr, "Tokenizer alloc failed\n"); exit(1); }
    for (int i = 0; i < vs; i++) {
        int32_t len;
        if (fread(&len, 4, 1, f) != 1) { fprintf(stderr, "Truncated token %d\n", i); exit(1); }
        if (len < 0 || len > 65536) { fprintf(stderr, "Invalid token len %d\n", len); exit(1); }
        tok->token_lens[i] = len;
        tok->tokens[i] = malloc((size_t)len + 1);
        if (!tok->tokens[i]) { fprintf(stderr, "Token alloc failed\n"); exit(1); }
        if (len > 0 && fread(tok->tokens[i], 1, len, f) != (size_t)len) {
            fprintf(stderr, "Truncated token data %d\n", i); exit(1);
        }
        tok->tokens[i][len] = '\0';
    }
    tok->bos_id = tok->user_start_id = tok->user_end_id = -1;
    tok->assistant_start_id = tok->assistant_end_id = -1;
    int32_t ns;
    if (fread(&ns, 4, 1, f) != 1) ns = 0;
    for (int i = 0; i < ns; i++) {
        int32_t tid, nl;
        if (fread(&tid, 4, 1, f) != 1 || fread(&nl, 4, 1, f) != 1) break;
        if (nl < 0 || nl >= 255) { fseek(f, nl, SEEK_CUR); continue; }
        char name[256];
        if (fread(name, 1, nl, f) != (size_t)nl) break;
        name[nl] = '\0';
        if (strcmp(name, "<|bos|>") == 0) tok->bos_id = tid;
        else if (strcmp(name, "<|user_start|>") == 0) tok->user_start_id = tid;
        else if (strcmp(name, "<|user_end|>") == 0) tok->user_end_id = tid;
        else if (strcmp(name, "<|assistant_start|>") == 0) tok->assistant_start_id = tid;
        else if (strcmp(name, "<|assistant_end|>") == 0) tok->assistant_end_id = tid;
    }
    fclose(f);
    printf("Tokenizer: vocab=%d bos=%d end=%d\n", tok->vocab_size, tok->bos_id, tok->assistant_end_id);
}

static void free_tokenizer(Tokenizer* tok) {
    for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
    free(tok->tokens); free(tok->token_lens);
}

static const char* decode_token(const Tokenizer* tok, int id) {
    if (id < 0 || id >= tok->vocab_size) return "<?>";
    return tok->tokens[id];
}

// ===== FORWARD PASS =======================================================

static void forward(Model* m, int token, int prev_token, int pos) {
    Config* c = &m->config;
    Weights* w = &m->weights;
    RunState* s = &m->state;
    int n = c->n_embd;
    int kv = c->kv_dim;
    int hd = c->head_dim;
    int eq = c->embed_quant;  // for embeddings
    int aq = c->attn_quant;   // for attention/MLP

    // === Embedding ===
    embed_lookup(s->x, w->wte, w->wte_scales, token, n, eq);

    // Bigram embedding
    int bg = (pos == 0) ? c->bigram_vocab - 1 : bigram_hash(token, prev_token, c->bigram_vocab);
    embed_lookup(s->x0_bigram, w->bigram_embed, w->bigram_scales, bg, n, eq);

    // RMSNorm + save x0
    rmsnorm(s->x, s->x, n);
    memcpy(s->x0, s->x, n * sizeof(float));

    // === Transformer blocks ===
    for (int layer = 0; layer < c->n_layer; layer++) {
        // Residual mixing
        float rl = half_to_float(w->resid_lambdas[layer]);
        float xl = half_to_float(w->x0_lambdas[layer]);
        float bl = half_to_float(w->bigram_lambdas[layer]);
        for (int i = 0; i < n; i++)
            s->x[i] = rl * s->x[i] + xl * s->x0[i] + bl * s->x0_bigram[i];

        // Pre-norm
        rmsnorm(s->xn, s->x, n);

        // Q, K, V projections
        matmul(s->q, w->c_q[layer], w->c_q_s[layer], s->xn, c->n_head * hd, n, aq);
        matmul(s->k, w->c_k[layer], w->c_k_s[layer], s->xn, c->n_kv_head * hd, n, aq);
        matmul(s->v, w->c_v[layer], w->c_v_s[layer], s->xn, c->n_kv_head * hd, n, aq);

        // Value Embedding gate
        if (c->has_ve[layer] && w->value_embeds[layer]) {
            for (int h = 0; h < c->n_kv_head; h++) {
                float gv = 0.0f;
                uint16_t* gr = w->ve_gate[layer] + h * 32;
                for (int j = 0; j < 32; j++) gv += half_to_float(gr[j]) * s->xn[j];
                gv = 2.0f * sigmoidf(gv);
                // ve lookup for this kv head's slice
                if (eq == 1) {
                    const int8_t* ve = (const int8_t*)w->value_embeds[layer];
                    float sc = w->value_embeds_s[layer][token];
                    const int8_t* vr = ve + (size_t)token * kv + h * hd;
                    for (int d = 0; d < hd; d++)
                        s->v[h * hd + d] += gv * ((float)vr[d] * sc);
                } else {
                    const uint16_t* ve = (const uint16_t*)w->value_embeds[layer];
                    const uint16_t* vr = ve + (size_t)token * kv + h * hd;
                    for (int d = 0; d < hd; d++)
                        s->v[h * hd + d] += gv * half_to_float(vr[d]);
                }
            }
        }

        // RoPE
        for (int h = 0; h < c->n_head; h++) apply_rope(s->q + h * hd, pos, s, hd);
        for (int h = 0; h < c->n_kv_head; h++) apply_rope(s->k + h * hd, pos, s, hd);

        // QK-Norm
        for (int h = 0; h < c->n_head; h++) rmsnorm(s->q + h * hd, s->q + h * hd, hd);
        for (int h = 0; h < c->n_kv_head; h++) rmsnorm(s->k + h * hd, s->k + h * hd, hd);

        // Store K,V in cache
        size_t co = (size_t)layer * c->seq_len * kv + (size_t)pos * kv;
        memcpy(s->key_cache + co, s->k, kv * sizeof(float));
        memcpy(s->value_cache + co, s->v, kv * sizeof(float));

        // Attention
        int window = c->window_sizes[layer];
        int start = pos - window + 1;
        if (start < 0) start = 0;
        float scale = 1.0f / sqrtf((float)hd);
        int hpkv = c->n_head / c->n_kv_head;

        for (int h = 0; h < c->n_head; h++) {
            int kvh = h / hpkv;
            float* qh = s->q + h * hd;
            float* ar = s->att + (size_t)h * c->seq_len;
            for (int t = start; t <= pos; t++) {
                float* kc = s->key_cache + (size_t)layer * c->seq_len * kv + (size_t)t * kv + kvh * hd;
                float sc = 0.0f;
                for (int d = 0; d < hd; d++) sc += qh[d] * kc[d];
                ar[t] = sc * scale;
            }
            softmax(ar + start, pos - start + 1);
            float* yh = s->y_att + h * hd;
            memset(yh, 0, hd * sizeof(float));
            for (int t = start; t <= pos; t++) {
                float a = ar[t];
                float* vc = s->value_cache + (size_t)layer * c->seq_len * kv + (size_t)t * kv + kvh * hd;
                for (int d = 0; d < hd; d++) yh[d] += a * vc[d];
            }
        }

        // Output projection + residual
        matmul(s->xn, w->c_proj[layer], w->c_proj_s[layer], s->y_att, n, n, aq);
        for (int i = 0; i < n; i++) s->x[i] += s->xn[i];

        // MLP: pre-norm
        rmsnorm(s->xn, s->x, n);
        matmul(s->hb, w->mlp_fc[layer], w->mlp_fc_s[layer], s->xn, c->mlp_dim, n, aq);
        // ReLU²
        for (int i = 0; i < c->mlp_dim; i++) {
            float v = s->hb[i] > 0.0f ? s->hb[i] : 0.0f;
            s->hb[i] = v * v;
        }
        matmul(s->xn, w->mlp_proj[layer], w->mlp_proj_s[layer], s->hb, n, c->mlp_dim, aq);
        for (int i = 0; i < n; i++) s->x[i] += s->xn[i];
    }

    // Final norm
    rmsnorm(s->x, s->x, n);

    // LM head (uses embed_quant)
    matmul(s->logits, w->lm_head, w->lm_head_s, s->x, c->padded_vocab, n, eq);

    // Softcap
    float cap = 15.0f;
    for (int i = 0; i < c->vocab_size; i++)
        s->logits[i] = cap * tanhf(s->logits[i] / cap);
}

// ===== SAMPLING ===========================================================

static int sample_argmax(const float* logits, int n) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

static int sample_topk(const float* logits, int vocab, float temp, int top_k, unsigned long long* rng) {
    if (temp <= 0.0f) return sample_argmax(logits, vocab);
    if (vocab > 65536) {
        fprintf(stderr, "sample_topk: vocab %d exceeds buffer size 65536\n", vocab);
        return sample_argmax(logits, vocab);
    }
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    int k = top_k < vocab ? top_k : vocab;
    static int idx[65536]; static float probs[65536];
    for (int i = 0; i < vocab; i++) idx[i] = i;
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i+1; j < vocab; j++) if (logits[idx[j]] > logits[idx[best]]) best = j;
        int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
    }
    float mx = logits[idx[0]]; float sum = 0.0f;
    for (int i = 0; i < k; i++) { probs[i] = expf((logits[idx[i]] - mx) / temp); sum += probs[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < k; i++) probs[i] *= inv;
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    float r = (float)(*rng & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float cdf = 0.0f;
    for (int i = 0; i < k; i++) { cdf += probs[i]; if (r <= cdf) return idx[i]; }
    return idx[k - 1];
}

// ===== MAIN ===============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <weights.bin> <tokenizer.tok> [options]\n", argv[0]);
        fprintf(stderr, "  -t <temp>    Temperature (default: 0.8)\n");
        fprintf(stderr, "  -k <top_k>   Top-k (default: 50)\n");
        fprintf(stderr, "  -n <max>     Max tokens (default: 256)\n");
        fprintf(stderr, "  -s <seed>    RNG seed\n");
        fprintf(stderr, "  -p <tokens>  Prompt token IDs (space-separated)\n");
        return 1;
    }

    float temperature = 0.8f; int top_k = 50, max_tokens = 256;
    unsigned long long rng_seed = 0; const char* ptokens = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i+1<argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i+1<argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i+1<argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1<argc) rng_seed = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i+1<argc) ptokens = argv[++i];
    }
    if (rng_seed == 0) rng_seed = (unsigned long long)time(NULL);

    Model model; Tokenizer tokenizer;
    printf("Loading model...\n");
    load_model(&model, argv[1]);
    printf("Loading tokenizer...\n");
    load_tokenizer(&tokenizer, argv[2]);

    int prompt[4096]; int plen = 0;
    if (ptokens) {
        char buf[65536]; strncpy(buf, ptokens, sizeof(buf)-1); buf[sizeof(buf)-1]='\0';
        char* t = strtok(buf, " ");
        while (t && plen < 4096) { prompt[plen++] = atoi(t); t = strtok(NULL, " "); }
    } else {
        if (tokenizer.bos_id >= 0) prompt[plen++] = tokenizer.bos_id;
        if (tokenizer.assistant_start_id >= 0) prompt[plen++] = tokenizer.assistant_start_id;
    }
    if (plen == 0) { fprintf(stderr, "No prompt\n"); return 1; }

    printf("\nPrompt (%d tokens)\n--- Generation ---\n", plen);

    int pos = 0, prev = 0, gen = 0;
    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    for (int i = 0; i < plen; i++) {
        forward(&model, prompt[i], prev, pos);
        prev = prompt[i]; pos++;
    }

    for (int i = 0; i < max_tokens; i++) {
        int next = sample_topk(model.state.logits, model.config.vocab_size, temperature, top_k, &rng_seed);
        if (next == tokenizer.assistant_end_id || next == tokenizer.bos_id) break;
        printf("%s", decode_token(&tokenizer, next));
        fflush(stdout);
        forward(&model, next, prev, pos);
        prev = next; pos++; gen++;
        if (pos >= model.config.seq_len) { fprintf(stderr, "\n[max seq]\n"); break; }
    }

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;
    printf("\n\n--- %d tokens in %.2fs (%.2f tok/s) ---\n", gen, elapsed, gen / elapsed);

    free_tokenizer(&tokenizer);
    close_model(&model);
    return 0;
}
