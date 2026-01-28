/*
 * ariannabody.c - Llama 3.5 Arianna Unified Edition (20M params)
 * Pure C transformer inference — the "body" of Arianna
 *
 * Architecture: Llama 3 (RMSNorm, RoPE, SwiGLU, GQA)
 * Based on Dubrovsky/alexey.c
 *
 * UNIFIED 20M (Jan 2026):
 *   dim=448, layers=8, heads=8, kv_heads=8, hidden=1280, vocab=84
 *   Total: ~20M parameters (unified personality + knowledge)
 *
 * LEGACY 10M (personality only):
 *   dim=384, layers=6, heads=6, kv_heads=2, hidden=1024, vocab=80
 *   Available as arianna_legacy.bin
 */

#include "arianna.h"
#include "bpe_tokenizer.h"

// ============================================================
// Tokenizer (char-level or BPE, auto-detected)
// ============================================================

// Char-level tokenizer state
static char* VOCAB_CHARS = NULL;      // id -> char
static int* CHAR_TO_ID = NULL;        // char -> id (256 entries)
static int TOKENIZER_VOCAB_SIZE = 0;

// BPE tokenizer state
static BPETokenizer g_bpe_tok;
static TokenizerType g_tokenizer_type = TOKENIZER_CHAR;

// Static decode buffer for char-level
static char g_decode_buffer[4096];

int load_tokenizer(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[tokenizer] cannot open: %s\n", path);
        return -1;
    }

    // Read entire file to detect type
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long len = ftell(f);
    if (len < 0 || len > 10*1024*1024) {  // cap at 10MB for safety
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    char* content = malloc((size_t)len + 1);
    if (!content) {
        fclose(f);
        return -1;
    }
    if (fread(content, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(content);
        return -1;
    }
    content[len] = '\0';
    fclose(f);

    // Detect tokenizer type: BPE has "id_to_piece", char-level has "char_to_id"
    if (strstr(content, "\"id_to_piece\"")) {
        // BPE tokenizer
        free(content);  // bpe_load will re-read the file

        if (bpe_load(&g_bpe_tok, path) != 0) {
            fprintf(stderr, "[tokenizer] BPE load failed: %s\n", path);
            return -1;
        }

        g_tokenizer_type = TOKENIZER_BPE;
        TOKENIZER_VOCAB_SIZE = g_bpe_tok.vocab_size;
        fprintf(stderr, "[tokenizer] BPE mode, %d tokens from %s\n", TOKENIZER_VOCAB_SIZE, path);
        return 0;
    }

    // Char-level tokenizer
    g_tokenizer_type = TOKENIZER_CHAR;

    // Find vocab_size
    char* vs = strstr(content, "\"vocab_size\":");
    if (vs) {
        TOKENIZER_VOCAB_SIZE = atoi(vs + 14);
    } else {
        TOKENIZER_VOCAB_SIZE = 84;  // Default for Arianna Unified 20M
    }

    // Allocate
    if (VOCAB_CHARS) free(VOCAB_CHARS);
    if (CHAR_TO_ID) free(CHAR_TO_ID);

    VOCAB_CHARS = calloc(TOKENIZER_VOCAB_SIZE, sizeof(char));
    CHAR_TO_ID = calloc(256, sizeof(int));

    // Initialize all to space token (fallback)
    for (int i = 0; i < 256; i++) CHAR_TO_ID[i] = 1;

    // Parse char_to_id mappings
    char* p = strstr(content, "\"char_to_id\":");
    if (p) {
        p = strchr(p, '{');
        if (p) {
            p++;
            while (*p && *p != '}') {
                // Skip whitespace
                while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;

                if (*p == '}') break;
                if (*p != '"') { p++; continue; }

                // Parse key (character)
                p++;  // Skip opening quote
                int c;
                if (*p == '\\') {
                    p++;
                    if (*p == 'n') c = '\n';
                    else if (*p == 't') c = '\t';
                    else if (*p == 'r') c = '\r';
                    else if (*p == '\\') c = '\\';
                    else if (*p == '"') c = '"';
                    else c = *p;
                    p++;
                } else {
                    // Handle UTF-8 (just take first byte for now)
                    c = (unsigned char)*p;
                    p++;
                    // Skip rest of multi-byte char
                    while ((*p & 0xC0) == 0x80) p++;
                }

                // Skip to colon
                while (*p && *p != ':' && *p != '"') p++;
                if (*p == '"') p++;  // closing quote
                while (*p && *p != ':') p++;
                if (*p == ':') p++;

                // Parse value (id)
                while (*p == ' ') p++;
                int id = atoi(p);

                // Store mapping
                if (c >= 0 && c < 256 && id >= 0 && id < TOKENIZER_VOCAB_SIZE) {
                    CHAR_TO_ID[c] = id;
                    VOCAB_CHARS[id] = (char)c;
                }

                // Skip to next entry
                while (*p && *p != ',' && *p != '}') p++;
            }
        }
    }

    free(content);
    fprintf(stderr, "[tokenizer] char-level mode, %d tokens from %s\n", TOKENIZER_VOCAB_SIZE, path);
    return 0;
}

int char_to_token(char c) {
    if (!CHAR_TO_ID) return 1;  // space
    return CHAR_TO_ID[(unsigned char)c];
}

char token_to_char(int token) {
    if (!VOCAB_CHARS || token < 0 || token >= TOKENIZER_VOCAB_SIZE) return '?';
    return VOCAB_CHARS[token];
}

int get_vocab_size(void) {
    return TOKENIZER_VOCAB_SIZE > 0 ? TOKENIZER_VOCAB_SIZE : VOCAB_SIZE;
}

TokenizerType get_tokenizer_type(void) {
    return g_tokenizer_type;
}

int encode_text(const char* text, int* ids, int max_tokens) {
    if (!text || !ids || max_tokens <= 0) return 0;

    if (g_tokenizer_type == TOKENIZER_BPE) {
        return bpe_encode(&g_bpe_tok, text, ids, max_tokens);
    }

    // Char-level: one token per character
    int len = strlen(text);
    if (len > max_tokens) len = max_tokens;
    for (int i = 0; i < len; i++) {
        ids[i] = char_to_token(text[i]);
    }
    return len;
}

const char* decode_tokens(const int* ids, int n_tokens) {
    if (!ids || n_tokens <= 0) return "";

    if (g_tokenizer_type == TOKENIZER_BPE) {
        return bpe_decode(&g_bpe_tok, ids, n_tokens);
    }

    // Char-level: one char per token
    int pos = 0;
    for (int i = 0; i < n_tokens && pos < (int)sizeof(g_decode_buffer) - 1; i++) {
        g_decode_buffer[pos++] = token_to_char(ids[i]);
    }
    g_decode_buffer[pos] = '\0';
    return g_decode_buffer;
}

// Static buffer for single token decode
static char g_single_token_buffer[BPE_MAX_PIECE_LEN];

const char* decode_token(int id) {
    if (g_tokenizer_type == TOKENIZER_BPE) {
        return bpe_decode_token(&g_bpe_tok, id);
    }

    // Char-level: single character
    g_single_token_buffer[0] = token_to_char(id);
    g_single_token_buffer[1] = '\0';
    return g_single_token_buffer;
}

// Reset streaming decode state (call at generation start)
void reset_decode_state(void) {
    if (g_tokenizer_type == TOKENIZER_BPE) {
        bpe_reset_decode_state();
    }
}

// ============================================================
// Memory Management
// ============================================================

// Returns 0 on success, -1 on allocation failure
int malloc_weights(Transformer* t) {
    Config* c = &t->config;
    Weights* w = &t->weights;

    int dim = c->dim;
    int n_layers = c->n_layers;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int kv_dim = c->n_kv_heads * c->head_dim;

    // Zero out weights struct first
    memset(w, 0, sizeof(Weights));

    // Token embedding (no position embedding - we use RoPE)
    w->tok_emb = calloc(vocab_size * dim, sizeof(float));

    // Per-layer weights
    w->attn_norm = calloc(n_layers * dim, sizeof(float));
    w->wq = calloc(n_layers * dim * dim, sizeof(float));
    w->wk = calloc(n_layers * dim * kv_dim, sizeof(float));
    w->wv = calloc(n_layers * dim * kv_dim, sizeof(float));
    w->wo = calloc(n_layers * dim * dim, sizeof(float));
    w->ffn_norm = calloc(n_layers * dim, sizeof(float));
    w->w_gate = calloc(n_layers * dim * hidden_dim, sizeof(float));
    w->w_up = calloc(n_layers * dim * hidden_dim, sizeof(float));
    w->w_down = calloc(n_layers * hidden_dim * dim, sizeof(float));

    // Final norm and output
    w->final_norm = calloc(dim, sizeof(float));
    w->lm_head = calloc(vocab_size * dim, sizeof(float));

    // SECURITY: Check all allocations succeeded
    if (!w->tok_emb || !w->attn_norm || !w->wq || !w->wk || !w->wv ||
        !w->wo || !w->ffn_norm || !w->w_gate || !w->w_up || !w->w_down ||
        !w->final_norm || !w->lm_head) {
        fprintf(stderr, "[model] OOM: failed to allocate weights\n");
        // Free any successful allocations
        free(w->tok_emb); free(w->attn_norm); free(w->wq); free(w->wk);
        free(w->wv); free(w->wo); free(w->ffn_norm); free(w->w_gate);
        free(w->w_up); free(w->w_down); free(w->final_norm); free(w->lm_head);
        memset(w, 0, sizeof(Weights));
        return -1;
    }
    return 0;
}

// Returns 0 on success, -1 on allocation failure
int malloc_run_state(Transformer* t) {
    Config* c = &t->config;
    RunState* s = &t->state;

    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int max_seq = c->max_seq_len;
    int n_layers = c->n_layers;
    int kv_dim = c->n_kv_heads * c->head_dim;

    // Zero out state struct first
    memset(s, 0, sizeof(RunState));

    // Activation buffers
    s->x = calloc(dim, sizeof(float));
    s->xb = calloc(dim, sizeof(float));
    s->xb2 = calloc(dim, sizeof(float));
    s->hb = calloc(hidden_dim, sizeof(float));
    s->hb2 = calloc(hidden_dim, sizeof(float));

    // Attention buffers
    s->q = calloc(dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(c->n_heads * max_seq, sizeof(float));

    // KV cache
    s->key_cache = calloc(n_layers * max_seq * kv_dim, sizeof(float));
    s->value_cache = calloc(n_layers * max_seq * kv_dim, sizeof(float));

    // RoPE precomputed
    s->rope_cos = calloc(max_seq * (c->head_dim / 2), sizeof(float));
    s->rope_sin = calloc(max_seq * (c->head_dim / 2), sizeof(float));

    // Output
    s->logits = calloc(vocab_size, sizeof(float));

    // SECURITY: Check all allocations succeeded
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att ||
        !s->key_cache || !s->value_cache ||
        !s->rope_cos || !s->rope_sin || !s->logits) {
        fprintf(stderr, "[model] OOM: failed to allocate run state\n");
        // Free any successful allocations
        free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
        free(s->q); free(s->k); free(s->v); free(s->att);
        free(s->key_cache); free(s->value_cache);
        free(s->rope_cos); free(s->rope_sin); free(s->logits);
        memset(s, 0, sizeof(RunState));
        return -1;
    }

    // Precompute RoPE frequencies
    float theta = c->rope_theta;
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < c->head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / c->head_dim);
            float angle = pos * freq;
            s->rope_cos[pos * (c->head_dim / 2) + i] = cosf(angle);
            s->rope_sin[pos * (c->head_dim / 2) + i] = sinf(angle);
        }
    }

    return 0;
}

void free_transformer(Transformer* t) {
    Weights* w = &t->weights;
    RunState* s = &t->state;

    free(w->tok_emb);
    free(w->attn_norm);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->ffn_norm);
    free(w->w_gate);
    free(w->w_up);
    free(w->w_down);
    free(w->final_norm);
    free(w->lm_head);

    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->key_cache);
    free(s->value_cache);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s->logits);
}

// ============================================================
// Core Operations (Llama 3 style)
// ============================================================

void rms_norm(float* out, float* x, float* weight, int size, float eps) {
    // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss = 1.0f / sqrtf(ss + eps);

    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void silu(float* x, int size) {
    // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
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

void matmul(float* out, float* x, float* w, int n, int d) {
    // W (d, n) @ x (n,) = out (d,)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

void apply_rope(float* q, float* k, float* rope_cos, float* rope_sin,
                int n_heads, int n_kv_heads, int head_dim, int pos) {
    int half = head_dim / 2;
    float* cos = rope_cos + pos * half;
    float* sin = rope_sin + pos * half;

    // Apply to Q heads
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[2*i];
            float q1 = qh[2*i + 1];
            qh[2*i] = q0 * cos[i] - q1 * sin[i];
            qh[2*i + 1] = q0 * sin[i] + q1 * cos[i];
        }
    }

    // Apply to K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k0 = kh[2*i];
            float k1 = kh[2*i + 1];
            kh[2*i] = k0 * cos[i] - k1 * sin[i];
            kh[2*i + 1] = k0 * sin[i] + k1 * cos[i];
        }
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
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden_dim = c->hidden_dim;

    // Token embedding
    float* tok_vec = w->tok_emb + token * dim;
    memcpy(s->x, tok_vec, dim * sizeof(float));

    // Transformer layers
    for (int layer = 0; layer < c->n_layers; layer++) {
        // Pre-norm for attention
        rms_norm(s->xb, s->x, w->attn_norm + layer * dim, dim, c->norm_eps);

        // QKV projection
        matmul(s->q, s->xb, w->wq + layer * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + layer * dim * kv_dim, dim, kv_dim);

        // Apply RoPE
        apply_rope(s->q, s->k, s->rope_cos, s->rope_sin,
                   c->n_heads, c->n_kv_heads, c->head_dim, pos);

        // Store in KV cache
        int kv_cache_offset = layer * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + kv_cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + kv_cache_offset, s->v, kv_dim * sizeof(float));

        // Multi-head attention with GQA
        memset(s->xb, 0, dim * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            float* qh = s->q + h * c->head_dim;
            float* atth = s->att + h * c->max_seq_len;
            int kv_h = h / c->n_kv_groups;  // Which KV head this Q head uses

            // Compute attention scores
            float scale = 1.0f / sqrtf((float)c->head_dim);
            for (int t = 0; t <= pos; t++) {
                float* kh = s->key_cache + layer * c->max_seq_len * kv_dim + t * kv_dim + kv_h * c->head_dim;
                float score = 0.0f;
                for (int i = 0; i < c->head_dim; i++) {
                    score += qh[i] * kh[i];
                }
                atth[t] = score * scale;
            }

            // Softmax over positions
            softmax(atth, pos + 1);

            // Weighted sum of values
            float* xbh = s->xb + h * c->head_dim;
            for (int t = 0; t <= pos; t++) {
                float* vh = s->value_cache + layer * c->max_seq_len * kv_dim + t * kv_dim + kv_h * c->head_dim;
                float a = atth[t];
                for (int i = 0; i < c->head_dim; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }

        // Output projection
        matmul(s->xb2, s->xb, w->wo + layer * dim * dim, dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // Pre-norm for FFN
        rms_norm(s->xb, s->x, w->ffn_norm + layer * dim, dim, c->norm_eps);

        // SwiGLU FFN
        matmul(s->hb, s->xb, w->w_gate + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w_up + layer * dim * hidden_dim, dim, hidden_dim);

        // SiLU activation and element-wise multiply
        for (int i = 0; i < hidden_dim; i++) {
            float gate = s->hb[i];
            s->hb[i] = (gate / (1.0f + expf(-gate))) * s->hb2[i];
        }

        // Down projection
        matmul(s->xb, s->hb, w->w_down + layer * hidden_dim * dim, hidden_dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    // Final norm
    rms_norm(s->x, s->x, w->final_norm, dim, c->norm_eps);

    // Output logits
    matmul(s->logits, s->x, w->lm_head, dim, c->vocab_size);
}

// ============================================================
// Sampling
// ============================================================

int sample(Transformer* t, float temperature) {
    float* logits = t->state.logits;
    int vocab_size = t->config.vocab_size;

    // Guard against division by zero
    if (temperature < 1e-6f) temperature = 1e-6f;

    // Apply temperature
    if (temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }

    // Softmax
    softmax(logits, vocab_size);

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

int sample_top_p(Transformer* t, float temperature, float top_p) {
    float* logits = t->state.logits;
    int vocab_size = t->config.vocab_size;

    // Guard against division by zero
    if (temperature < 1e-6f) temperature = 1e-6f;

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    // Softmax
    softmax(logits, vocab_size);

    // Top-p (nucleus) sampling
    if (top_p < 1.0f) {
        // Dynamic allocation for arbitrary vocab sizes (fixes buffer overflow for vocab > 256)
        int* indices = (int*)malloc(vocab_size * sizeof(int));
        if (!indices) {
            // Fallback: return last token if malloc fails
            return vocab_size - 1;
        }
        for (int i = 0; i < vocab_size; i++) indices[i] = i;

        // Bubble sort by probability (descending)
        for (int i = 0; i < vocab_size - 1; i++) {
            for (int j = 0; j < vocab_size - i - 1; j++) {
                if (logits[indices[j]] < logits[indices[j+1]]) {
                    int tmp = indices[j];
                    indices[j] = indices[j+1];
                    indices[j+1] = tmp;
                }
            }
        }

        // Accumulate until top_p
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += logits[indices[i]];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out beyond cutoff
        for (int i = cutoff; i < vocab_size; i++) {
            logits[indices[i]] = 0.0f;
        }

        // Free dynamically allocated indices
        free(indices);

        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += logits[i];
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) logits[i] /= sum;
        }
    }

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

// ============================================================
// Generation
// ============================================================

void generate(Transformer* t, const char* prompt, int max_tokens, float temperature) {
    // Process prompt
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
    for (int i = 0; i < max_tokens && pos < t->config.max_seq_len; i++) {
        int next_token = sample(t, temperature);

        char c = token_to_char(next_token);
        printf("%c", c);
        fflush(stdout);

        forward(t, next_token, pos);
        pos++;

        // Stop on newline (end of answer in Q&A format)
        if (c == '\n') {
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

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    fprintf(stderr, "[model] loading %.2f MB from %s\n", file_size / 1024.0f / 1024.0f, path);

    Config* c = &t->config;

    // Try to read magic number (for files with embedded config)
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) == 1 && magic == 0x616B616E) {
        // File has embedded config - read it
        fprintf(stderr, "[model] reading embedded config...\n");

        if (fread(&c->dim, sizeof(int), 1, f) != 1 ||
            fread(&c->n_layers, sizeof(int), 1, f) != 1 ||
            fread(&c->n_heads, sizeof(int), 1, f) != 1 ||
            fread(&c->n_kv_heads, sizeof(int), 1, f) != 1 ||
            fread(&c->head_dim, sizeof(int), 1, f) != 1 ||
            fread(&c->hidden_dim, sizeof(int), 1, f) != 1 ||
            fread(&c->max_seq_len, sizeof(int), 1, f) != 1 ||
            fread(&c->vocab_size, sizeof(int), 1, f) != 1 ||
            fread(&c->n_kv_groups, sizeof(int), 1, f) != 1 ||
            fread(&c->rope_theta, sizeof(float), 1, f) != 1 ||
            fread(&c->norm_eps, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "[model] error reading config\n");
            fclose(f);
            return -1;
        }

        // SECURITY: Validate embedded config to prevent integer overflow in allocations
        if (c->dim <= 0 || c->dim > 16384 ||
            c->n_layers <= 0 || c->n_layers > 256 ||
            c->n_heads <= 0 || c->n_heads > 256 ||
            c->n_kv_heads <= 0 || c->n_kv_heads > c->n_heads ||
            c->head_dim <= 0 || c->head_dim > 512 ||
            c->hidden_dim <= 0 || c->hidden_dim > 65536 ||
            c->max_seq_len <= 0 || c->max_seq_len > 131072 ||
            c->vocab_size <= 0 || c->vocab_size > 1000000 ||
            c->n_kv_groups <= 0 || c->n_kv_groups > c->n_heads) {
            fprintf(stderr, "[model] embedded config values out of safe range\n");
            fclose(f);
            return -1;
        }
    } else {
        // Legacy file without magic - use defaults and rewind
        fseek(f, 0, SEEK_SET);
        c->dim = DIM;
        c->n_layers = N_LAYERS;
        c->n_heads = N_HEADS;
        c->n_kv_heads = N_KV_HEADS;
        c->head_dim = c->dim / c->n_heads;
        c->hidden_dim = HIDDEN_DIM;
        c->max_seq_len = MAX_SEQ_LEN;
        c->vocab_size = get_vocab_size();
        c->n_kv_groups = c->n_heads / c->n_kv_heads;
        c->rope_theta = 10000.0f;
        c->norm_eps = 1e-5f;
        fprintf(stderr, "[model] using default config (legacy file)\n");
    }

    int kv_dim = c->n_kv_heads * c->head_dim;

    fprintf(stderr, "[model] dim=%d layers=%d heads=%d kv_heads=%d vocab=%d hidden=%d\n",
            c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->hidden_dim);

    // Allocate with OOM checks
    if (malloc_weights(t) != 0) {
        fclose(f);
        return -1;
    }
    if (malloc_run_state(t) != 0) {
        // Free already allocated weights
        free(t->weights.tok_emb); free(t->weights.attn_norm);
        free(t->weights.wq); free(t->weights.wk); free(t->weights.wv);
        free(t->weights.wo); free(t->weights.ffn_norm);
        free(t->weights.w_gate); free(t->weights.w_up); free(t->weights.w_down);
        free(t->weights.final_norm); free(t->weights.lm_head);
        fclose(f);
        return -1;
    }

    Weights* w = &t->weights;
    int dim = c->dim;
    int n_layers = c->n_layers;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;

    // Read weights in dubrovsky/export_weights.py order:
    // tok_emb, then per layer: attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down
    // then final_norm, lm_head

    #define READ(ptr, count) do { \
        if (fread(ptr, sizeof(float), count, f) != (size_t)(count)) { \
            fprintf(stderr, "[model] read error at %s\n", #ptr); \
            fclose(f); \
            free_transformer(t); \
            return -1; \
        } \
    } while(0)

    // Token embeddings
    READ(w->tok_emb, vocab_size * dim);

    // Per-layer weights
    for (int l = 0; l < n_layers; l++) {
        READ(w->attn_norm + l * dim, dim);
        READ(w->wq + l * dim * dim, dim * dim);
        READ(w->wk + l * dim * kv_dim, dim * kv_dim);
        READ(w->wv + l * dim * kv_dim, dim * kv_dim);
        READ(w->wo + l * dim * dim, dim * dim);
        READ(w->ffn_norm + l * dim, dim);
        READ(w->w_gate + l * dim * hidden_dim, dim * hidden_dim);
        READ(w->w_up + l * dim * hidden_dim, dim * hidden_dim);
        READ(w->w_down + l * hidden_dim * dim, hidden_dim * dim);
    }

    // Final norm
    READ(w->final_norm, dim);

    // LM head
    READ(w->lm_head, vocab_size * dim);

    #undef READ

    fclose(f);
    fprintf(stderr, "[model] loaded successfully\n");
    return 0;
}
