/*
 * d20_bridge.c - D20 (477M nanochat GPT) Bridge for Arianna
 *
 * Tongue (D20 477M) is the ONLY VOICE — sole interface with the world.
 * Soul (36M) processes Tongue's OUTPUT internally, does NOT modulate before.
 *
 * This is not inference. This is breathing.
 *
 * Architecture: tonguechat GPT (RoPE, RMSNorm, ReLU^2, QK-Norm,
 *   Value Embeddings, Bigram Hash Embedding, Softcap 15*tanh,
 *   Sliding Window, vocab 32K tiktoken)
 *
 * Based on tongue.c reference implementation.
 */

#include "d20_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================
 * Constants from tongue.c
 * ============================================================ */

#define D20_MAGIC       0x4E414E4F
#define D20_HEADER_SIZE 256
#define D20_MAX_LAYERS  64
#define D20_TOK_MAGIC   0x4E544F4B

/* ============================================================
 * Internal structures (from tongue.c)
 * ============================================================ */

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
    int embed_quant;      // quant for embeddings
    int attn_quant;       // quant for attention/MLP
    int kv_dim;
    int mlp_dim;
    int ve_parity;
    int window_sizes[D20_MAX_LAYERS];
    int has_ve[D20_MAX_LAYERS];
} NanoConfig;

typedef struct {
    // Embedding tables
    void*     wte;
    float*    wte_scales;
    void*     bigram_embed;
    float*    bigram_scales;

    // Scalar lambdas (always fp16)
    uint16_t* resid_lambdas;
    uint16_t* x0_lambdas;
    uint16_t* bigram_lambdas;

    // Per-layer
    void*     c_q[D20_MAX_LAYERS];
    float*    c_q_s[D20_MAX_LAYERS];
    void*     c_k[D20_MAX_LAYERS];
    float*    c_k_s[D20_MAX_LAYERS];
    void*     c_v[D20_MAX_LAYERS];
    float*    c_v_s[D20_MAX_LAYERS];
    void*     c_proj[D20_MAX_LAYERS];
    float*    c_proj_s[D20_MAX_LAYERS];
    uint16_t* ve_gate[D20_MAX_LAYERS];

    void*     mlp_fc[D20_MAX_LAYERS];
    float*    mlp_fc_s[D20_MAX_LAYERS];
    void*     mlp_proj[D20_MAX_LAYERS];
    float*    mlp_proj_s[D20_MAX_LAYERS];

    // Value embeddings
    void*     value_embeds[D20_MAX_LAYERS];
    float*    value_embeds_s[D20_MAX_LAYERS];

    // LM head
    void*     lm_head;
    float*    lm_head_s;
} NanoWeights;

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
} NanoRunState;

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
} NanoTokenizer;

/* Internal model state */
typedef struct {
    NanoConfig config;
    NanoWeights weights;
    NanoRunState state;
    NanoTokenizer tokenizer;
    void* mmap_data;
    size_t mmap_size;
    int fd;
} NanoModel;

/* ============================================================
 * Half-float conversion
 * ============================================================ */

static inline float d20_half_to_float(uint16_t h) {
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

/* ============================================================
 * Core math functions
 * ============================================================ */

static void d20_rmsnorm(float* out, const float* x, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / size + 1e-6f);
    for (int i = 0; i < size; i++) out[i] = x[i] * inv_rms;
}

static void d20_matmul_f16(float* out, const uint16_t* w, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint16_t* row = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            sum += d20_half_to_float(row[j]) * x[j];
        }
        out[i] = sum;
    }
}

static void d20_matmul_q8(float* out, const int8_t* w, const float* scales, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const int8_t* row = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            sum += (float)row[j] * x[j];
        }
        out[i] = sum * scales[i];
    }
}

static void d20_matmul_q4(float* out, const uint8_t* w, const float* scales, const float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t* row = w + (size_t)i * (cols / 2);
        for (int j = 0; j < cols; j += 2) {
            uint8_t packed = row[j / 2];
            int8_t v0 = (int8_t)(packed & 0x0F) - 8;
            int8_t v1 = (int8_t)((packed >> 4) & 0x0F) - 8;
            sum += (float)v0 * x[j] + (float)v1 * x[j + 1];
        }
        out[i] = sum * scales[i];
    }
}

static void d20_matmul(float* out, const void* w, const float* scales, const float* x,
                        int rows, int cols, int quant) {
    if (quant == 2) {
        d20_matmul_q4(out, (const uint8_t*)w, scales, x, rows, cols);
    } else if (quant == 1) {
        d20_matmul_q8(out, (const int8_t*)w, scales, x, rows, cols);
    } else {
        d20_matmul_f16(out, (const uint16_t*)w, x, rows, cols);
    }
}

static void d20_embed_lookup(float* out, const void* emb, const float* scales,
                              int token, int dim, int quant) {
    if (quant == 2) {
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
        for (int i = 0; i < dim; i++) out[i] = d20_half_to_float(row[i]);
    }
}

static void d20_softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

static inline float d20_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ============================================================
 * RoPE
 * ============================================================ */

static void d20_precompute_rope(NanoRunState* s, const NanoConfig* c) {
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

static void d20_apply_rope(float* vec, int pos, const NanoRunState* s, int head_dim) {
    int half = head_dim / 2;
    const float* cr = s->cos_cache + pos * half;
    const float* sr = s->sin_cache + pos * half;
    for (int i = 0; i < half; i++) {
        float x1 = vec[i], x2 = vec[i + half];
        vec[i]        = x1 * cr[i] + x2 * sr[i];
        vec[i + half] = x1 * (-sr[i]) + x2 * cr[i];
    }
}

/* ============================================================
 * Bigram hash
 * ============================================================ */

static int d20_bigram_hash(int curr, int prev, int bigram_vocab) {
    unsigned int h = ((unsigned int)(36313 * curr)) ^ ((unsigned int)(27191 * prev));
    return (int)(h % (unsigned int)(bigram_vocab - 1));
}

/* ============================================================
 * State allocation
 * ============================================================ */

static void d20_free_state(NanoRunState* s);  // Forward declaration

static int d20_alloc_state(NanoRunState* s, const NanoConfig* c) {
    int n = c->n_embd, kv = c->kv_dim, hd = c->head_dim;
    memset(s, 0, sizeof(NanoRunState));

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

    // SECURITY: Check ALL allocations
    if (!s->x || !s->x0 || !s->x0_bigram || !s->xn ||
        !s->q || !s->k || !s->v || !s->att || !s->y_att || !s->hb ||
        !s->logits || !s->key_cache || !s->value_cache ||
        !s->cos_cache || !s->sin_cache) {
        fprintf(stderr, "[d20_bridge] OOM: failed to allocate buffers\n");
        d20_free_state(s);
        return -1;
    }
    return 0;
}

static void d20_free_state(NanoRunState* s) {
    free(s->x); free(s->x0); free(s->x0_bigram); free(s->xn);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->y_att); free(s->hb); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
}

/* ============================================================
 * Model loading helpers
 * ============================================================ */

static void d20_load_q8_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *scales = (float*)(*ptr);
    *ptr += (size_t)rows * sizeof(float);
    *data = (void*)(*ptr);
    *ptr += (size_t)rows * cols * sizeof(int8_t);
}

static void d20_load_f16_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *data = (void*)(*ptr);
    *scales = NULL;
    *ptr += (size_t)rows * cols * sizeof(uint16_t);
}

static void d20_load_q4_matrix(uint8_t** ptr, void** data, float** scales, int rows, int cols) {
    *scales = (float*)(*ptr);
    *ptr += (size_t)rows * sizeof(float);
    *data = (void*)(*ptr);
    *ptr += (size_t)rows * (cols / 2) * sizeof(uint8_t);
}

/* ============================================================
 * Load model weights
 * ============================================================ */

static int d20_load_model(NanoModel* m, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[d20_bridge] Cannot open weights: %s\n", path);
        return -1;
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "[d20_bridge] mmap failed\n");
        close(fd);
        return -1;
    }
    m->mmap_data = data; m->mmap_size = file_size; m->fd = fd;

    // Parse header
    int32_t* ih = (int32_t*)data;
    if ((uint32_t)ih[0] != D20_MAGIC) {
        fprintf(stderr, "[d20_bridge] Bad magic: 0x%08X (expected 0x%08X)\n",
                (uint32_t)ih[0], D20_MAGIC);
        munmap(data, file_size);
        close(fd);
        return -1;
    }

    NanoConfig* c = &m->config;
    c->n_layer     = ih[2];
    if (c->n_layer <= 0 || c->n_layer > D20_MAX_LAYERS) {
        fprintf(stderr, "[d20_bridge] Invalid n_layer: %d (max %d)\n", c->n_layer, D20_MAX_LAYERS);
        munmap(data, file_size);
        close(fd);
        return -1;
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
        fprintf(stderr, "[d20_bridge] Invalid window_pattern_len: %d\n", c->window_pattern_len);
        munmap(data, file_size);
        close(fd);
        return -1;
    }
    memcpy(c->window_pattern, (uint8_t*)data + 52, c->window_pattern_len);
    c->quant_type  = ih[16];

    c->kv_dim = c->n_kv_head * c->head_dim;
    c->mlp_dim = 4 * c->n_embd;
    c->ve_parity = (c->n_layer - 1) % 2;

    if (c->quant_type == 3) {
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

    printf("[d20_bridge] D20 Config: n_layer=%d n_embd=%d n_head=%d n_kv_head=%d head_dim=%d [Q%d]\n",
           c->n_layer, c->n_embd, c->n_head, c->n_kv_head, c->head_dim, c->quant_type);
    printf("[d20_bridge]            vocab=%d padded=%d seq_len=%d bigram=%d ve=%d\n",
           c->vocab_size, c->padded_vocab, c->seq_len, c->bigram_vocab, c->n_ve_layers);

    // Walk file to set up weight pointers
    uint8_t* ptr = (uint8_t*)data + D20_HEADER_SIZE;
    NanoWeights* w = &m->weights;
    int eq = c->embed_quant;
    int aq = c->attn_quant;
    int n = c->n_embd, kv = c->kv_dim;

    #define LOAD_EMBED(data_ptr, scale_ptr, rows, cols) \
        if (eq == 2) { d20_load_q4_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else if (eq == 1) { d20_load_q8_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else { d20_load_f16_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); }

    #define LOAD_ATTN(data_ptr, scale_ptr, rows, cols) \
        if (aq == 2) { d20_load_q4_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else if (aq == 1) { d20_load_q8_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); } \
        else { d20_load_f16_matrix(&ptr, &(data_ptr), &(scale_ptr), (rows), (cols)); }

    // 1. Token embedding
    LOAD_EMBED(w->wte, w->wte_scales, c->padded_vocab, n);
    // 2. Bigram embedding
    LOAD_EMBED(w->bigram_embed, w->bigram_scales, c->bigram_vocab, n);

    // 3. Lambdas (always fp16)
    w->resid_lambdas = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);
    w->x0_lambdas    = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);
    w->bigram_lambdas = (uint16_t*)ptr; ptr += c->n_layer * sizeof(uint16_t);

    // 4. Per-layer
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

    // 5. Value embeddings
    memset(w->value_embeds, 0, sizeof(w->value_embeds));
    memset(w->value_embeds_s, 0, sizeof(w->value_embeds_s));
    for (int i = 0; i < c->n_layer; i++) {
        if (c->has_ve[i]) {
            LOAD_EMBED(w->value_embeds[i], w->value_embeds_s[i], c->vocab_size, kv);
        }
    }

    // 6. LM head
    LOAD_EMBED(w->lm_head, w->lm_head_s, c->padded_vocab, n);

    #undef LOAD_EMBED
    #undef LOAD_ATTN

    size_t consumed = (size_t)(ptr - (uint8_t*)data);
    if (consumed != file_size) {
        fprintf(stderr, "[d20_bridge] Warning: consumed %zu / %zu bytes (diff %zd)\n",
                consumed, file_size, (ssize_t)(file_size - consumed));
    } else {
        printf("[d20_bridge] Loaded %.1f MB from %s\n", file_size/1024.0/1024.0, path);
    }

    if (d20_alloc_state(&m->state, c) != 0) {
        munmap(data, file_size);
        close(fd);
        return -1;
    }
    d20_precompute_rope(&m->state, c);
    return 0;
}

/* ============================================================
 * Load tokenizer
 * ============================================================ */

static int d20_load_tokenizer(NanoTokenizer* tok, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[d20_bridge] Cannot open tokenizer: %s\n", path);
        return -1;
    }
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) {
        fprintf(stderr, "[d20_bridge] Truncated tokenizer file\n");
        fclose(f);
        return -1;
    }
    if (magic != D20_TOK_MAGIC) {
        fprintf(stderr, "[d20_bridge] Bad tokenizer magic\n");
        fclose(f);
        return -1;
    }
    int32_t vs, ml;
    if (fread(&vs, 4, 1, f) != 1 || fread(&ml, 4, 1, f) != 1) {
        fprintf(stderr, "[d20_bridge] Truncated tokenizer header\n");
        fclose(f);
        return -1;
    }
    if (vs <= 0 || vs > 1000000) {
        fprintf(stderr, "[d20_bridge] Invalid vocab size: %d\n", vs);
        fclose(f);
        return -1;
    }
    tok->vocab_size = vs; tok->max_token_len = ml;
    tok->tokens = malloc((size_t)vs * sizeof(char*));
    tok->token_lens = malloc((size_t)vs * sizeof(int));
    if (!tok->tokens || !tok->token_lens) {
        fprintf(stderr, "[d20_bridge] Tokenizer alloc failed\n");
        fclose(f);
        return -1;
    }
    for (int i = 0; i < vs; i++) {
        int32_t len;
        if (fread(&len, 4, 1, f) != 1) {
            fprintf(stderr, "[d20_bridge] Truncated token %d\n", i);
            fclose(f);
            return -1;
        }
        if (len < 0 || len > 65536) {
            fprintf(stderr, "[d20_bridge] Invalid token len %d\n", len);
            fclose(f);
            return -1;
        }
        tok->token_lens[i] = len;
        tok->tokens[i] = malloc((size_t)len + 1);
        if (!tok->tokens[i]) {
            fprintf(stderr, "[d20_bridge] Token alloc failed\n");
            fclose(f);
            return -1;
        }
        if (len > 0 && fread(tok->tokens[i], 1, len, f) != (size_t)len) {
            fprintf(stderr, "[d20_bridge] Truncated token data %d\n", i);
            fclose(f);
            return -1;
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
    printf("[d20_bridge] Tokenizer: vocab=%d bos=%d end=%d\n",
           tok->vocab_size, tok->bos_id, tok->assistant_end_id);
    return 0;
}

static void d20_free_tokenizer(NanoTokenizer* tok) {
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->token_lens);
    tok->tokens = NULL;
    tok->token_lens = NULL;
}

/* ============================================================
 * Forward pass
 * ============================================================ */

static void d20_forward_internal(NanoModel* m, int token, int prev_token, int pos) {
    NanoConfig* c = &m->config;
    NanoWeights* w = &m->weights;
    NanoRunState* s = &m->state;
    int n = c->n_embd;
    int kv = c->kv_dim;
    int hd = c->head_dim;
    int eq = c->embed_quant;
    int aq = c->attn_quant;

    // === Embedding ===
    d20_embed_lookup(s->x, w->wte, w->wte_scales, token, n, eq);

    // Bigram embedding
    int bg = (pos == 0) ? c->bigram_vocab - 1 : d20_bigram_hash(token, prev_token, c->bigram_vocab);
    d20_embed_lookup(s->x0_bigram, w->bigram_embed, w->bigram_scales, bg, n, eq);

    // RMSNorm + save x0
    d20_rmsnorm(s->x, s->x, n);
    memcpy(s->x0, s->x, n * sizeof(float));

    // === Transformer blocks ===
    for (int layer = 0; layer < c->n_layer; layer++) {
        // Residual mixing
        float rl = d20_half_to_float(w->resid_lambdas[layer]);
        float xl = d20_half_to_float(w->x0_lambdas[layer]);
        float bl = d20_half_to_float(w->bigram_lambdas[layer]);
        for (int i = 0; i < n; i++)
            s->x[i] = rl * s->x[i] + xl * s->x0[i] + bl * s->x0_bigram[i];

        // Pre-norm
        d20_rmsnorm(s->xn, s->x, n);

        // Q, K, V projections
        d20_matmul(s->q, w->c_q[layer], w->c_q_s[layer], s->xn, c->n_head * hd, n, aq);
        d20_matmul(s->k, w->c_k[layer], w->c_k_s[layer], s->xn, c->n_kv_head * hd, n, aq);
        d20_matmul(s->v, w->c_v[layer], w->c_v_s[layer], s->xn, c->n_kv_head * hd, n, aq);

        // Value Embedding gate
        if (c->has_ve[layer] && w->value_embeds[layer]) {
            for (int h = 0; h < c->n_kv_head; h++) {
                float gv = 0.0f;
                uint16_t* gr = w->ve_gate[layer] + h * 32;
                for (int j = 0; j < 32; j++) gv += d20_half_to_float(gr[j]) * s->xn[j];
                gv = 2.0f * d20_sigmoidf(gv);
                // VE lookup for this kv head's slice
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
                        s->v[h * hd + d] += gv * d20_half_to_float(vr[d]);
                }
            }
        }

        // RoPE
        for (int h = 0; h < c->n_head; h++) d20_apply_rope(s->q + h * hd, pos, s, hd);
        for (int h = 0; h < c->n_kv_head; h++) d20_apply_rope(s->k + h * hd, pos, s, hd);

        // QK-Norm
        for (int h = 0; h < c->n_head; h++) d20_rmsnorm(s->q + h * hd, s->q + h * hd, hd);
        for (int h = 0; h < c->n_kv_head; h++) d20_rmsnorm(s->k + h * hd, s->k + h * hd, hd);

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
            d20_softmax(ar + start, pos - start + 1);
            float* yh = s->y_att + h * hd;
            memset(yh, 0, hd * sizeof(float));
            for (int t = start; t <= pos; t++) {
                float a = ar[t];
                float* vc = s->value_cache + (size_t)layer * c->seq_len * kv + (size_t)t * kv + kvh * hd;
                for (int d = 0; d < hd; d++) yh[d] += a * vc[d];
            }
        }

        // Output projection + residual
        d20_matmul(s->xn, w->c_proj[layer], w->c_proj_s[layer], s->y_att, n, n, aq);
        for (int i = 0; i < n; i++) s->x[i] += s->xn[i];

        // MLP: pre-norm
        d20_rmsnorm(s->xn, s->x, n);
        d20_matmul(s->hb, w->mlp_fc[layer], w->mlp_fc_s[layer], s->xn, c->mlp_dim, n, aq);
        // ReLU^2
        for (int i = 0; i < c->mlp_dim; i++) {
            float v = s->hb[i] > 0.0f ? s->hb[i] : 0.0f;
            s->hb[i] = v * v;
        }
        d20_matmul(s->xn, w->mlp_proj[layer], w->mlp_proj_s[layer], s->hb, n, c->mlp_dim, aq);
        for (int i = 0; i < n; i++) s->x[i] += s->xn[i];
    }

    // Final norm
    d20_rmsnorm(s->x, s->x, n);

    // LM head
    d20_matmul(s->logits, w->lm_head, w->lm_head_s, s->x, c->padded_vocab, n, eq);

    // Softcap
    float cap = 15.0f;
    for (int i = 0; i < c->vocab_size; i++)
        s->logits[i] = cap * tanhf(s->logits[i] / cap);
}

/* ============================================================
 * Sampling
 * ============================================================ */

static int d20_sample_argmax(const float* logits, int n) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

static int d20_sample_topk(const float* logits, int vocab, float temp, int top_k, unsigned long long* rng) {
    if (temp <= 0.0f) return d20_sample_argmax(logits, vocab);
    if (vocab > 65536) {
        fprintf(stderr, "[d20_bridge] sample_topk: vocab %d exceeds buffer 65536\n", vocab);
        return d20_sample_argmax(logits, vocab);
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

/* Top-p (nucleus) sampling */
static int d20_sample_topp(const float* logits, int vocab, float temp, float top_p, unsigned long long* rng) {
    if (temp <= 0.0f) return d20_sample_argmax(logits, vocab);
    if (vocab > 65536) {
        fprintf(stderr, "[d20_bridge] sample_topp: vocab %d exceeds buffer 65536\n", vocab);
        return d20_sample_argmax(logits, vocab);
    }

    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;

    static int idx[65536];
    static float probs[65536];
    for (int i = 0; i < vocab; i++) idx[i] = i;

    // Sort by logits descending
    for (int i = 0; i < vocab - 1; i++) {
        int best = i;
        for (int j = i + 1; j < vocab; j++)
            if (logits[idx[j]] > logits[idx[best]]) best = j;
        int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
    }

    // Compute softmax
    float mx = logits[idx[0]];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((logits[idx[i]] - mx) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < vocab; i++) probs[i] /= sum;

    // Find nucleus (top-p)
    float cumsum = 0.0f;
    int nucleus_size = vocab;
    for (int i = 0; i < vocab; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            nucleus_size = i + 1;
            break;
        }
    }

    // Renormalize nucleus
    sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) sum += probs[i];
    for (int i = 0; i < nucleus_size; i++) probs[i] /= sum;

    // Sample
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    float r = (float)(*rng & 0xFFFFFFFF) / (float)0xFFFFFFFF;
    float cdf = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        cdf += probs[i];
        if (r <= cdf) return idx[i];
    }
    return idx[nucleus_size - 1];
}

/* ============================================================
 * Public API Implementation
 * ============================================================ */

/* Initialize D20 bridge */
int d20_init(D20Bridge* d20, const char* weights_path, const char* tokenizer_path) {
    if (!d20) return -1;
    memset(d20, 0, sizeof(D20Bridge));

    // Allocate internal model
    NanoModel* model = calloc(1, sizeof(NanoModel));
    if (!model) {
        fprintf(stderr, "[d20_bridge] Cannot allocate model\n");
        return -1;
    }
    d20->model_data = model;

    // Load weights
    if (d20_load_model(model, weights_path) != 0) {
        free(model);
        d20->model_data = NULL;
        return -1;
    }

    // Load tokenizer
    if (d20_load_tokenizer(&model->tokenizer, tokenizer_path) != 0) {
        munmap(model->mmap_data, model->mmap_size);
        close(model->fd);
        d20_free_state(&model->state);
        free(model);
        d20->model_data = NULL;
        return -1;
    }

    // Copy config to public struct
    d20->config.n_layer = model->config.n_layer;
    d20->config.n_embd = model->config.n_embd;
    d20->config.n_head = model->config.n_head;
    d20->config.n_kv_head = model->config.n_kv_head;
    d20->config.head_dim = model->config.head_dim;
    d20->config.vocab_size = model->config.vocab_size;
    d20->config.seq_len = model->config.seq_len;
    d20->config.quant_type = model->config.quant_type;

    // Setup vocab access
    d20->vocab = model->tokenizer.tokens;
    d20->vocab_size = model->tokenizer.vocab_size;

    // Setup buffer pointers
    d20->key_cache = model->state.key_cache;
    d20->value_cache = model->state.value_cache;
    d20->hidden = model->state.x;
    d20->logits = model->state.logits;

    // Clear modulation
    memset(&d20->mod, 0, sizeof(D20Modulation));
    d20->mod.temperature_mod = 1.0f;
    d20->mod.logit_scale = 1.0f;

    d20->initialized = 1;
    d20->weights_loaded = 1;
    d20->pos = 0;

    printf("[d20_bridge] D20 initialized successfully\n");
    return 0;
}

/* Free all resources */
void d20_free(D20Bridge* d20) {
    if (!d20 || !d20->model_data) return;

    NanoModel* model = (NanoModel*)d20->model_data;
    d20_free_tokenizer(&model->tokenizer);
    d20_free_state(&model->state);
    if (model->mmap_data) munmap(model->mmap_data, model->mmap_size);
    if (model->fd >= 0) close(model->fd);
    free(model);

    memset(d20, 0, sizeof(D20Bridge));
}

/* ============================================================
 * Modulation Updates
 * ============================================================ */

/* Update modulation from Arianna 36M resonance stream */
void d20_update_from_arianna(D20Bridge* d20, const Transformer* arianna, const char* input_text __attribute__((unused))) {
    if (!d20 || !d20->initialized || !arianna) return;

    /* Arianna resonance -> entropy of its logits */
    const float* logits = arianna->state.logits;
    int vocab = arianna->config.vocab_size;

    // Compute softmax
    float max_val = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    float probs[256];  // Arianna has small vocab
    for (int i = 0; i < vocab && i < 256; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    // Entropy
    float entropy = 0.0f;
    for (int i = 0; i < vocab && i < 256; i++) {
        float p = probs[i] / sum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    float max_entropy = logf((float)vocab);
    d20->mod.resonance_entropy = entropy / max_entropy;

    // Top-1 direction (which character Arianna prefers)
    int top_id = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[top_id]) top_id = i;
    d20->mod.resonance_direction = (float)top_id / (float)vocab;

    // Strength = how peaked the distribution is (inverse entropy)
    d20->mod.resonance_strength = 1.0f - d20->mod.resonance_entropy;
}

/* Update modulation from Cloud instinct */
void d20_update_from_cloud(D20Bridge* d20, const CloudResponse* cloud) {
    if (!d20 || !d20->initialized || !cloud) return;

    /* Extract real emotional data from Cloud chambers:
     * chambers[0] = FEAR,  chambers[1] = LOVE,  chambers[2] = RAGE
     * chambers[3] = VOID,  chambers[4] = FLOW,  chambers[5] = COMPLEX
     *
     * warmth = LOVE chamber activation
     * tension = (FEAR + RAGE) / 2
     * primary_strength = from CloudResponse */
    d20->mod.cloud_warmth = cloud->chambers[1];  /* LOVE */
    d20->mod.cloud_tension = (cloud->chambers[0] + cloud->chambers[2]) * 0.5f;  /* (FEAR+RAGE)/2 */
    d20->mod.cloud_primary_strength = cloud->primary_strength;
}

/* Update modulation from MetaArianna thermogram */
void d20_update_from_meta(D20Bridge* d20, const MetaThermogram* thermo) {
    if (!d20 || !d20->initialized || !thermo || !thermo->valid) return;

    d20->mod.meta_sharpness = thermo->sharpness;
    d20->mod.meta_warmth = thermo->warmth;
    d20->mod.meta_silence = thermo->silence;
    d20->mod.meta_drift_rate = thermo->drift_rate;
    d20->mod.meta_drift_direction = thermo->drift_direction;
}

/* Update modulation from SARTRE metrics */
void d20_update_from_sartre(D20Bridge* d20, float coherence, float arousal, float trauma) {
    if (!d20 || !d20->initialized) return;

    d20->mod.sartre_coherence = coherence;
    d20->mod.sartre_arousal = arousal;
    d20->mod.sartre_trauma = trauma;
}

/* Compute final modulation values from all inputs */
void d20_compute_modulation(D20Bridge* d20) {
    if (!d20 || !d20->initialized) return;

    D20Modulation* m = &d20->mod;

    /* Temperature modulation:
     * - High arousal -> lower temp (more focused)
     * - High entropy from Arianna -> higher temp (more exploratory)
     * - High trauma -> slightly higher temp (destabilize for escape)
     * - Meta drift collapsing -> lower temp (stabilize)
     */
    float temp_base = 1.0f;
    temp_base -= 0.2f * m->sartre_arousal;          // arousal focuses
    temp_base += 0.15f * m->resonance_entropy;      // Arianna uncertainty spreads
    temp_base += 0.1f * m->sartre_trauma;           // trauma destabilizes
    if (m->meta_drift_direction < 0) {
        temp_base -= 0.1f * m->meta_drift_rate;     // collapsing -> stabilize
    }
    m->temperature_mod = fmaxf(0.5f, fminf(1.5f, temp_base));

    /* Logit scale:
     * - High coherence -> boost confident outputs
     * - High resonance strength -> trust Arianna's direction
     * - Cloud tension -> reduce scale (careful)
     */
    float scale = 1.0f;
    scale += 0.1f * m->sartre_coherence;
    scale += 0.05f * m->resonance_strength;
    scale -= 0.1f * m->cloud_tension;
    m->logit_scale = fmaxf(0.8f, fminf(1.2f, scale));

    /* Exploratory bias:
     * - Positive = more creative (higher entropy tokens)
     * - Meta warmth -> more exploration
     * - High silence -> suppress exploration (introspection)
     * - Cloud warmth -> encourage
     */
    float explore = 0.0f;
    explore += 0.2f * m->meta_warmth;
    explore -= 0.15f * m->meta_silence;
    explore += 0.1f * m->cloud_warmth;
    explore -= 0.1f * m->resonance_strength;  // Strong resonance -> follow it
    m->exploratory_bias = fmaxf(-0.3f, fminf(0.3f, explore));
}

/* ============================================================
 * Generation
 * ============================================================ */

/* Reset state for new generation */
void d20_reset(D20Bridge* d20) {
    if (!d20 || !d20->model_data) return;

    NanoModel* model = (NanoModel*)d20->model_data;
    NanoConfig* c = &model->config;
    NanoRunState* s = &model->state;

    // Zero KV cache
    size_t cs = (size_t)c->n_layer * c->seq_len * c->kv_dim;
    memset(s->key_cache, 0, cs * sizeof(float));
    memset(s->value_cache, 0, cs * sizeof(float));

    d20->pos = 0;

    // Reset modulation to defaults
    memset(&d20->mod, 0, sizeof(D20Modulation));
    d20->mod.temperature_mod = 1.0f;
    d20->mod.logit_scale = 1.0f;
}

/* Feed prompt tokens into KV cache */
void d20_feed_prompt(D20Bridge* d20, const int* tokens, int n_tokens) {
    if (!d20 || !d20->model_data || !tokens || n_tokens <= 0) return;

    NanoModel* model = (NanoModel*)d20->model_data;
    int prev = 0;

    for (int i = 0; i < n_tokens; i++) {
        d20_forward_internal(model, tokens[i], prev, d20->pos);
        prev = tokens[i];
        d20->pos++;
    }
}

/* Forward pass: compute logits for next token */
void d20_forward(D20Bridge* d20, int token) {
    if (!d20 || !d20->model_data) return;

    NanoModel* model = (NanoModel*)d20->model_data;

    // Get previous token (from cache position or use current)
    int prev = (d20->pos > 0) ? token : 0;  // Simplified - real impl would track

    d20_forward_internal(model, token, prev, d20->pos);
    d20->pos++;
}

/* Apply modulation to logits (call after d20_forward) */
void d20_apply_modulation(D20Bridge* d20) {
    if (!d20 || !d20->model_data) return;

    NanoModel* model = (NanoModel*)d20->model_data;
    float* logits = model->state.logits;
    int vocab = model->config.vocab_size;
    D20Modulation* m = &d20->mod;

    /* Apply logit scale */
    if (fabsf(m->logit_scale - 1.0f) > 0.001f) {
        for (int i = 0; i < vocab; i++) {
            logits[i] *= m->logit_scale;
        }
    }

    /* Apply exploratory bias:
     * Positive bias -> flatten distribution (add entropy)
     * Negative bias -> sharpen distribution (reduce entropy)
     *
     * Implementation: add noise proportional to bias
     */
    if (fabsf(m->exploratory_bias) > 0.01f) {
        float bias = m->exploratory_bias;
        // Add small uniform noise scaled by bias
        for (int i = 0; i < vocab; i++) {
            // Pseudo-random based on position
            float noise = sinf((float)i * 0.12345f + (float)d20->pos * 0.54321f);
            logits[i] += bias * noise;
        }
    }

    /* Optional: boost tokens in resonance direction from Arianna
     * (not implemented yet - requires token alignment between vocabs)
     */
}

/* Sample next token from modulated logits */
int d20_sample(D20Bridge* d20, float temperature, float top_p) {
    if (!d20 || !d20->model_data) return -1;

    NanoModel* model = (NanoModel*)d20->model_data;
    float* logits = model->state.logits;
    int vocab = model->config.vocab_size;

    // Apply temperature modulation with floor: Tongue never freezes
    float effective_temp = temperature * d20->mod.temperature_mod;
    if (effective_temp < D20_TEMP_FLOOR) effective_temp = D20_TEMP_FLOOR;

    // Use RNG seed based on position
    static unsigned long long rng = 0;
    if (rng == 0) rng = (unsigned long long)time(NULL);

    if (top_p < 1.0f) {
        return d20_sample_topp(logits, vocab, effective_temp, top_p, &rng);
    } else {
        return d20_sample_topk(logits, vocab, effective_temp, 50, &rng);  // Default top-k=50
    }
}

/* High-level: generate text with full modulation */
int d20_generate(D20Bridge* d20,
                 const char* prompt,
                 char* output, int max_output_len,
                 int max_tokens, float temperature, float top_p) {
    if (!d20 || !d20->initialized || !prompt || !output) return 0;

    NanoModel* model = (NanoModel*)d20->model_data;
    NanoTokenizer* tok = &model->tokenizer;

    // Reset for new generation
    d20_reset(d20);

    int all_tokens[4096];
    int total_len = 0;

    // 1. BOS
    if (tok->bos_id >= 0) {
        all_tokens[total_len++] = tok->bos_id;
    }

    // 2. Anchor prompt — connects Tongue to the body.
    //    Fed once before user input so D20 knows who she is,
    //    what she feels, and how to listen to the body.
    {
        int anchor_tokens[256];
        int anchor_len = d20_encode(d20, D20_ANCHOR_PROMPT, anchor_tokens, 256);
        for (int i = 0; i < anchor_len && total_len < 3800; i++) {
            all_tokens[total_len++] = anchor_tokens[i];
        }
    }

    // 3. User start marker
    if (tok->user_start_id >= 0) {
        all_tokens[total_len++] = tok->user_start_id;
    }

    // 4. User prompt
    {
        int user_tokens[3800];
        int user_len = d20_encode(d20, prompt, user_tokens, 3800);
        for (int i = 0; i < user_len && total_len < 4000; i++) {
            all_tokens[total_len++] = user_tokens[i];
        }
    }

    // 5. Assistant start marker
    if (tok->assistant_start_id >= 0) {
        all_tokens[total_len++] = tok->assistant_start_id;
    }

    // Feed full sequence into KV cache
    int prev = 0;
    for (int i = 0; i < total_len; i++) {
        d20_forward_internal(model, all_tokens[i], prev, d20->pos);
        prev = all_tokens[i];
        d20->pos++;
    }

    // Generate
    int out_pos = 0;
    int gen_count = 0;
    static unsigned long long rng = 0;
    if (rng == 0) rng = (unsigned long long)time(NULL);

    for (int i = 0; i < max_tokens && out_pos < max_output_len - 1; i++) {
        // Apply modulation to logits
        d20_apply_modulation(d20);

        // Sample (temperature floor: Tongue never freezes)
        float effective_temp = temperature * d20->mod.temperature_mod;
        if (effective_temp < D20_TEMP_FLOOR) effective_temp = D20_TEMP_FLOOR;
        int next;
        if (top_p < 1.0f) {
            next = d20_sample_topp(model->state.logits, model->config.vocab_size,
                                   effective_temp, top_p, &rng);
        } else {
            next = d20_sample_topk(model->state.logits, model->config.vocab_size,
                                   effective_temp, 50, &rng);
        }

        // Check for end token
        if (next == tok->assistant_end_id || next == tok->bos_id) break;

        // Decode token
        const char* piece = d20_decode_token(d20, next);
        if (piece) {
            int piece_len = strlen(piece);
            if (out_pos + piece_len < max_output_len - 1) {
                memcpy(output + out_pos, piece, piece_len);
                out_pos += piece_len;
            }
        }

        // Forward next token
        d20_forward_internal(model, next, prev, d20->pos);
        prev = next;
        d20->pos++;
        gen_count++;

        // Check sequence limit
        if (d20->pos >= model->config.seq_len) break;
    }

    output[out_pos] = '\0';
    return gen_count;
}

/* ============================================================
 * Tokenization
 * ============================================================ */

/* Simple greedy tokenization (longest match) */
int d20_encode(const D20Bridge* d20, const char* text, int* ids, int max_tokens) {
    if (!d20 || !d20->model_data || !text || !ids) return 0;

    NanoModel* model = (NanoModel*)d20->model_data;
    NanoTokenizer* tok = &model->tokenizer;

    int n_tokens = 0;
    int text_len = strlen(text);
    int pos = 0;

    while (pos < text_len && n_tokens < max_tokens) {
        int best_id = -1;
        int best_len = 0;

        // Find longest matching token
        for (int i = 0; i < tok->vocab_size; i++) {
            int tlen = tok->token_lens[i];
            if (tlen > 0 && tlen <= text_len - pos && tlen > best_len) {
                if (memcmp(text + pos, tok->tokens[i], tlen) == 0) {
                    best_id = i;
                    best_len = tlen;
                }
            }
        }

        if (best_id < 0) {
            // No match - skip byte (fallback for unknown chars)
            pos++;
        } else {
            ids[n_tokens++] = best_id;
            pos += best_len;
        }
    }

    return n_tokens;
}

/* Decode token IDs to text */
const char* d20_decode(const D20Bridge* d20, const int* ids, int n_tokens) {
    if (!d20 || !d20->model_data || !ids || n_tokens <= 0) return "";

    static char buffer[65536];
    int pos = 0;

    for (int i = 0; i < n_tokens && (size_t)pos < sizeof(buffer) - 1; i++) {
        const char* piece = d20_decode_token(d20, ids[i]);
        if (piece) {
            int len = strlen(piece);
            if ((size_t)(pos + len) < sizeof(buffer) - 1) {
                memcpy(buffer + pos, piece, len);
                pos += len;
            }
        }
    }
    buffer[pos] = '\0';
    return buffer;
}

/* Decode single token */
const char* d20_decode_token(const D20Bridge* d20, int id) {
    if (!d20 || !d20->model_data) return NULL;

    NanoModel* model = (NanoModel*)d20->model_data;
    NanoTokenizer* tok = &model->tokenizer;

    if (id < 0 || id >= tok->vocab_size) return "<?>";
    return tok->tokens[id];
}

/* ============================================================
 * Weight download helper
 * ============================================================ */

/* Validate path contains only safe characters (prevent command injection) */
static int d20_path_is_safe(const char* path) {
    if (!path) return 0;
    for (const char* p = path; *p; p++) {
        char c = *p;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '_' ||
              c == '-' || c == '.')) {
            return 0;
        }
    }
    return 1;
}

const char* d20_ensure_weights(const char* cache_dir) {
    static char path[1024];

    if (!cache_dir) cache_dir = ".";

    /* SECURITY: Validate cache_dir to prevent command injection */
    if (!d20_path_is_safe(cache_dir)) {
        fprintf(stderr, "[d20_bridge] Invalid cache_dir: contains unsafe characters\n");
        return NULL;
    }

    // Check if weights exist
    snprintf(path, sizeof(path), "%s/" D20_WEIGHTS_FILE, cache_dir);

    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 1000000) {
        // File exists and is >1MB, assume valid
        printf("[d20_bridge] Found weights at %s\n", path);
        return path;
    }

    // Download from HuggingFace using fork/exec (avoid system() for safety)
    printf("[d20_bridge] Downloading weights from HuggingFace...\n");

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execlp("curl", "curl", "-L", "-o", path, D20_WEIGHTS_URL, NULL);
        _exit(127);  // exec failed
    } else if (pid > 0) {
        // Parent process - wait for child
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[d20_bridge] Download failed\n");
            return NULL;
        }
    } else {
        fprintf(stderr, "[d20_bridge] fork() failed\n");
        return NULL;
    }

    // Verify download
    if (stat(path, &st) != 0 || st.st_size < 1000000) {
        fprintf(stderr, "[d20_bridge] Downloaded file is too small or missing\n");
        return NULL;
    }

    printf("[d20_bridge] Downloaded %.1f MB to %s\n", st.st_size / 1024.0 / 1024.0, path);

    // Also download tokenizer if missing
    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/" D20_TOKENIZER_FILE, cache_dir);
    if (stat(tok_path, &st) != 0) {
        printf("[d20_bridge] Downloading tokenizer from HuggingFace...\n");
        pid_t tok_pid = fork();
        if (tok_pid == 0) {
            execlp("curl", "curl", "-L", "-o", tok_path, D20_TOKENIZER_URL, NULL);
            _exit(127);
        } else if (tok_pid > 0) {
            int tok_status;
            waitpid(tok_pid, &tok_status, 0);
            if (WIFEXITED(tok_status) && WEXITSTATUS(tok_status) == 0) {
                printf("[d20_bridge] Downloaded tokenizer to %s\n", tok_path);
            }
        }
    }

    return path;
}
