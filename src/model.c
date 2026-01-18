/*
 * model.c - GPT-2 style transformer implementation
 * Pure C, no dependencies beyond libc + math
 */

#include "arianna.h"

// Dynamic vocab - array of UTF-8 strings (one per token)
static char** VOCAB = NULL;      // array of strings
static int* VOCAB_LENS = NULL;   // byte length of each string
static int VOCAB_SIZE = 0;

// Default vocab (dialogue) - ASCII only
static const char* DEFAULT_VOCAB_STR = "\n \"'(),-./05:;?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz";
static const int DEFAULT_VOCAB_SIZE = 67;

// Load binary vocab from file
// Format: vocab_size (4 bytes) + for each: len (1 byte) + utf-8 bytes
int load_vocab(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    // Read vocab size
    int vocab_size;
    if (fread(&vocab_size, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    // Free old vocab
    if (VOCAB) {
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (VOCAB[i]) free(VOCAB[i]);
        }
        free(VOCAB);
        free(VOCAB_LENS);
    }

    // Allocate
    VOCAB = (char**)malloc(vocab_size * sizeof(char*));
    VOCAB_LENS = (int*)malloc(vocab_size * sizeof(int));
    VOCAB_SIZE = vocab_size;

    // Read each char
    for (int i = 0; i < vocab_size; i++) {
        unsigned char len;
        if (fread(&len, 1, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        VOCAB[i] = (char*)malloc(len + 1);
        VOCAB_LENS[i] = len;
        if (fread(VOCAB[i], 1, len, f) != len) {
            fclose(f);
            return -1;
        }
        VOCAB[i][len] = '\0';
    }

    fclose(f);
    fprintf(stderr, "[vocab] loaded %d tokens from %s\n", VOCAB_SIZE, path);
    return 0;
}

// Initialize default vocab (ASCII, single bytes)
static void init_default_vocab(void) {
    if (VOCAB) return;

    VOCAB_SIZE = DEFAULT_VOCAB_SIZE;
    VOCAB = (char**)malloc(VOCAB_SIZE * sizeof(char*));
    VOCAB_LENS = (int*)malloc(VOCAB_SIZE * sizeof(int));

    for (int i = 0; i < VOCAB_SIZE; i++) {
        VOCAB[i] = (char*)malloc(2);
        VOCAB[i][0] = DEFAULT_VOCAB_STR[i];
        VOCAB[i][1] = '\0';
        VOCAB_LENS[i] = 1;
    }
}

// Auto-detect vocab file from weights path (weights/foo_brain.bin -> weights/vocab_foo.bin)
static void auto_load_vocab(const char* weights_path) {
    char vocab_path[512];

    // Extract directory and base name
    const char* basename = strrchr(weights_path, '/');
    if (basename) {
        basename++;
    } else {
        basename = weights_path;
    }

    // Remove _brain suffix if present
    char name[256];
    strncpy(name, basename, sizeof(name) - 1);
    name[sizeof(name) - 1] = '\0';

    char* dot = strrchr(name, '.');
    if (dot) *dot = '\0';

    char* brain_suffix = strstr(name, "_brain");
    if (brain_suffix) *brain_suffix = '\0';

    // Build vocab path (try .bin first)
    const char* dir = weights_path;
    int dir_len = (int)(basename - weights_path);

    // Strategy 1: weights/foo_brain.bin -> weights/vocab_foo.bin
    if (dir_len > 0) {
        snprintf(vocab_path, sizeof(vocab_path), "%.*svocab_%s.bin", dir_len, dir, name);
    } else {
        snprintf(vocab_path, sizeof(vocab_path), "vocab_%s.bin", name);
    }
    if (load_vocab(vocab_path) == 0) return;

    // Strategy 2: Try common names
    const char* common_names[] = {"dialogue", "personality", "sartre", NULL};
    for (int i = 0; common_names[i]; i++) {
        if (dir_len > 0) {
            snprintf(vocab_path, sizeof(vocab_path), "%.*svocab_%s.bin", dir_len, dir, common_names[i]);
        } else {
            snprintf(vocab_path, sizeof(vocab_path), "vocab_%s.bin", common_names[i]);
        }
        if (load_vocab(vocab_path) == 0) return;
    }

    // Fallback to default
    fprintf(stderr, "[vocab] using default vocab (%d tokens)\n", DEFAULT_VOCAB_SIZE);
    init_default_vocab();
}

// ============================================================
// Char Tokenization
// ============================================================

int char_to_token(char c) {
    if (!VOCAB) init_default_vocab();
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (VOCAB[i] && VOCAB[i][0] == c && VOCAB_LENS[i] == 1) return i;
    }
    return 1;  // default to space
}

char token_to_char(int token) {
    if (!VOCAB) init_default_vocab();
    if (token >= 0 && token < VOCAB_SIZE && VOCAB[token] && VOCAB_LENS[token] == 1) {
        return VOCAB[token][0];
    }
    return '?';
}

// ============================================================
// Memory Management
// ============================================================

void malloc_weights(Transformer* t) {
    Config* c = &t->config;
    Weights* w = &t->weights;

    int dim = c->dim;
    int n_layers = c->n_layers;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int max_seq = c->max_seq_len;

    // Embeddings
    w->wte = (float*)calloc(vocab_size * dim, sizeof(float));
    w->wpe = (float*)calloc(max_seq * dim, sizeof(float));

    // Per-layer weights
    w->ln1_weight = (float*)calloc(n_layers * dim, sizeof(float));
    w->ln1_bias = (float*)calloc(n_layers * dim, sizeof(float));
    w->c_attn_weight = (float*)calloc(n_layers * dim * 3 * dim, sizeof(float));
    w->c_attn_bias = (float*)calloc(n_layers * 3 * dim, sizeof(float));
    w->c_proj_weight = (float*)calloc(n_layers * dim * dim, sizeof(float));
    w->c_proj_bias = (float*)calloc(n_layers * dim, sizeof(float));
    w->ln2_weight = (float*)calloc(n_layers * dim, sizeof(float));
    w->ln2_bias = (float*)calloc(n_layers * dim, sizeof(float));
    w->c_fc_weight = (float*)calloc(n_layers * dim * hidden_dim, sizeof(float));
    w->c_fc_bias = (float*)calloc(n_layers * hidden_dim, sizeof(float));
    w->c_proj2_weight = (float*)calloc(n_layers * hidden_dim * dim, sizeof(float));
    w->c_proj2_bias = (float*)calloc(n_layers * dim, sizeof(float));

    // Final layer norm
    w->ln_f_weight = (float*)calloc(dim, sizeof(float));
    w->ln_f_bias = (float*)calloc(dim, sizeof(float));

    // lm_head (NULL = tied with wte)
    w->lm_head = NULL;
}

void malloc_run_state(Transformer* t) {
    Config* c = &t->config;
    RunState* s = &t->state;

    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int max_seq = c->max_seq_len;
    int n_layers = c->n_layers;

    s->x = (float*)calloc(max_seq * dim, sizeof(float));
    s->xb = (float*)calloc(dim, sizeof(float));
    s->qkv = (float*)calloc(3 * dim, sizeof(float));
    s->attn_out = (float*)calloc(dim, sizeof(float));
    s->ffn_buf = (float*)calloc(hidden_dim, sizeof(float));
    s->logits = (float*)calloc(vocab_size, sizeof(float));

    s->key_cache = (float*)calloc(n_layers * max_seq * dim, sizeof(float));
    s->value_cache = (float*)calloc(n_layers * max_seq * dim, sizeof(float));
    s->cache_len = 0;
}

void free_transformer(Transformer* t) {
    Weights* w = &t->weights;
    RunState* s = &t->state;

    free(w->wte);
    free(w->wpe);
    free(w->ln1_weight);
    free(w->ln1_bias);
    free(w->c_attn_weight);
    free(w->c_attn_bias);
    free(w->c_proj_weight);
    free(w->c_proj_bias);
    free(w->ln2_weight);
    free(w->ln2_bias);
    free(w->c_fc_weight);
    free(w->c_fc_bias);
    free(w->c_proj2_weight);
    free(w->c_proj2_bias);
    free(w->ln_f_weight);
    free(w->ln_f_bias);
    if (w->lm_head) free(w->lm_head);

    free(s->x);
    free(s->xb);
    free(s->qkv);
    free(s->attn_out);
    free(s->ffn_buf);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ============================================================
// Core Operations
// ============================================================

void layer_norm(float* out, float* x, float* weight, float* bias, int size) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;

    // Normalize
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void gelu(float* x, int size) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float SQRT_2_PI = 0.7978845608f;
    for (int i = 0; i < size; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = SQRT_2_PI * (xi + 0.044715f * x3);
        x[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// out[d] = x[n] @ w[n, d]
void matmul(float* out, float* x, float* w, int n, int d) {
    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i] * w[i * d + j];
        }
        out[j] = sum;
    }
}

// out[d] = x[n] @ w[n, d] + b[d]
void matmul_add(float* out, float* x, float* w, float* b, int n, int d) {
    for (int j = 0; j < d; j++) {
        float sum = b[j];
        for (int i = 0; i < n; i++) {
            sum += x[i] * w[i * d + j];
        }
        out[j] = sum;
    }
}

// ============================================================
// Forward Pass (Single Token)
// ============================================================

void forward(Transformer* t, int token, int pos) {
    Config* c = &t->config;
    Weights* w = &t->weights;
    RunState* s = &t->state;

    int dim = c->dim;
    int n_heads = c->n_heads;
    int head_dim = c->head_dim;
    int hidden_dim = c->hidden_dim;
    int max_seq = c->max_seq_len;

    // Get embedding: x = wte[token] + wpe[pos]
    float* x = s->x + pos * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = w->wte[token * dim + i] + w->wpe[pos * dim + i];
    }

    // Process through layers
    for (int layer = 0; layer < c->n_layers; layer++) {
        // Layer weight offsets
        float* ln1_w = w->ln1_weight + layer * dim;
        float* ln1_b = w->ln1_bias + layer * dim;
        float* c_attn_w = w->c_attn_weight + layer * dim * 3 * dim;
        float* c_attn_b = w->c_attn_bias + layer * 3 * dim;
        float* c_proj_w = w->c_proj_weight + layer * dim * dim;
        float* c_proj_b = w->c_proj_bias + layer * dim;
        float* ln2_w = w->ln2_weight + layer * dim;
        float* ln2_b = w->ln2_bias + layer * dim;
        float* c_fc_w = w->c_fc_weight + layer * dim * hidden_dim;
        float* c_fc_b = w->c_fc_bias + layer * hidden_dim;
        float* c_proj2_w = w->c_proj2_weight + layer * hidden_dim * dim;
        float* c_proj2_b = w->c_proj2_bias + layer * dim;

        // LayerNorm 1
        layer_norm(s->xb, x, ln1_w, ln1_b, dim);

        // QKV projection
        matmul_add(s->qkv, s->xb, c_attn_w, c_attn_b, dim, 3 * dim);

        float* q = s->qkv;
        float* k = s->qkv + dim;
        float* v = s->qkv + 2 * dim;

        // Store K, V in cache
        int cache_offset = layer * max_seq * dim + pos * dim;
        memcpy(s->key_cache + cache_offset, k, dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, v, dim * sizeof(float));

        // Multi-head attention
        memset(s->attn_out, 0, dim * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < n_heads; h++) {
            float* q_h = q + h * head_dim;
            float* out_h = s->attn_out + h * head_dim;

            // Attention scores for all cached positions
            float scores[1024];  // max seq len
            int n_ctx = pos + 1;

            for (int t = 0; t < n_ctx; t++) {
                float* k_t = s->key_cache + layer * max_seq * dim + t * dim + h * head_dim;
                float dot = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    dot += q_h[i] * k_t[i];
                }
                scores[t] = dot * scale;
            }

            // Softmax
            softmax(scores, n_ctx);

            // Weighted sum of values
            for (int t = 0; t < n_ctx; t++) {
                float* v_t = s->value_cache + layer * max_seq * dim + t * dim + h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    out_h[i] += scores[t] * v_t[i];
                }
            }
        }

        // Output projection + residual
        float proj_out[1024];  // max dim
        matmul_add(proj_out, s->attn_out, c_proj_w, c_proj_b, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += proj_out[i];
        }

        // LayerNorm 2
        layer_norm(s->xb, x, ln2_w, ln2_b, dim);

        // FFN: fc -> gelu -> proj
        matmul_add(s->ffn_buf, s->xb, c_fc_w, c_fc_b, dim, hidden_dim);
        gelu(s->ffn_buf, hidden_dim);

        float ffn_out[1024];  // max dim
        matmul_add(ffn_out, s->ffn_buf, c_proj2_w, c_proj2_b, hidden_dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            x[i] += ffn_out[i];
        }
    }

    // Final LayerNorm
    layer_norm(s->xb, x, w->ln_f_weight, w->ln_f_bias, dim);

    // Logits: x @ wte.T (weight tying) or x @ lm_head
    float* lm = w->lm_head ? w->lm_head : w->wte;
    for (int v = 0; v < c->vocab_size; v++) {
        float dot = 0.0f;
        for (int i = 0; i < dim; i++) {
            dot += s->xb[i] * lm[v * dim + i];
        }
        s->logits[v] = dot;
    }
}

// ============================================================
// Sampling
// ============================================================

int sample(Transformer* t, float temperature) {
    float* logits = t->state.logits;
    int vocab_size = t->config.vocab_size;

    // Work on a copy to not corrupt original logits
    float probs[1024];  // max vocab size

    // Apply temperature and copy
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = logits[i] / temperature;
    }

    // Softmax on copy
    softmax(probs, vocab_size);

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

// ============================================================
// Generation
// ============================================================

void generate(Transformer* t, const char* prompt, int max_tokens, float temperature) {
    // Reset cache
    t->state.cache_len = 0;

    // Encode and process prompt
    int pos = 0;
    const char* p = prompt;
    while (*p && pos < t->config.max_seq_len - 1) {
        int token = char_to_token(*p);
        forward(t, token, pos);
        printf("%c", *p);
        fflush(stdout);
        pos++;
        p++;
    }

    // Generate new tokens
    int last_token = char_to_token(*(p - 1));
    for (int i = 0; i < max_tokens && pos < t->config.max_seq_len; i++) {
        int next_token = sample(t, temperature);

        char c = token_to_char(next_token);
        printf("%c", c);
        fflush(stdout);

        forward(t, next_token, pos);
        pos++;
        last_token = next_token;

        // Stop on double newline
        if (c == '\n' && i > 0 && token_to_char(last_token) == '\n') {
            break;
        }
    }
}

// ============================================================
// Weight Loading
// ============================================================

int load_weights(Transformer* t, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[model] cannot open: %s\n", path);
        return -1;
    }

    // Read header (7 ints): dim, n_layers, n_heads, vocab_size, max_seq, hidden_dim, use_fp16
    int header[7];
    if (fread(header, sizeof(int), 7, f) != 7) {
        fprintf(stderr, "[model] cannot read header\n");
        fclose(f);
        return -1;
    }

    Config* c = &t->config;
    c->dim = header[0];
    c->n_layers = header[1];
    c->n_heads = header[2];
    c->vocab_size = header[3];
    c->max_seq_len = header[4];
    c->hidden_dim = header[5];
    c->head_dim = c->dim / c->n_heads;
    int use_fp16 = header[6];

    fprintf(stderr, "[model] dim=%d layers=%d heads=%d vocab=%d seq=%d hidden=%d\n",
            c->dim, c->n_layers, c->n_heads, c->vocab_size, c->max_seq_len, c->hidden_dim);

    // Auto-load vocab file based on weights path
    auto_load_vocab(path);

    // Verify vocab size matches
    if (VOCAB_SIZE != c->vocab_size) {
        fprintf(stderr, "[model] WARNING: vocab size mismatch! file=%d, loaded=%d\n",
                c->vocab_size, VOCAB_SIZE);
        fprintf(stderr, "[model] Generation will likely produce garbage!\n");
    }

    // Allocate
    malloc_weights(t);
    malloc_run_state(t);

    Weights* w = &t->weights;
    int dim = c->dim;
    int n_layers = c->n_layers;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int max_seq = c->max_seq_len;

    // Helper macro
    #define READ(ptr, count) fread(ptr, sizeof(float), count, f)

    // Read embeddings
    READ(w->wte, vocab_size * dim);
    READ(w->wpe, max_seq * dim);

    // Read layers
    for (int l = 0; l < n_layers; l++) {
        READ(w->ln1_weight + l * dim, dim);
        READ(w->ln1_bias + l * dim, dim);
        READ(w->c_attn_weight + l * dim * 3 * dim, dim * 3 * dim);
        READ(w->c_attn_bias + l * 3 * dim, 3 * dim);
        READ(w->c_proj_weight + l * dim * dim, dim * dim);
        READ(w->c_proj_bias + l * dim, dim);
        READ(w->ln2_weight + l * dim, dim);
        READ(w->ln2_bias + l * dim, dim);
        READ(w->c_fc_weight + l * dim * hidden_dim, dim * hidden_dim);
        READ(w->c_fc_bias + l * hidden_dim, hidden_dim);
        READ(w->c_proj2_weight + l * hidden_dim * dim, hidden_dim * dim);
        READ(w->c_proj2_bias + l * dim, dim);
    }

    // Final layer norm
    READ(w->ln_f_weight, dim);
    READ(w->ln_f_bias, dim);

    #undef READ

    fclose(f);
    fprintf(stderr, "[model] loaded successfully\n");
    return 0;
}
