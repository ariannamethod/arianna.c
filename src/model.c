/*
 * arianna.c - Model implementation
 * Pure C transformer with RoPE
 */

#include "arianna.h"

// ============================================================
// Memory Management
// ============================================================

void malloc_weights(Transformer* t) {
    Config* c = &t->config;
    Weights* w = &t->weights;

    // Token embedding
    w->token_embedding = (float*)calloc(c->vocab_size * c->dim, sizeof(float));
    if (!w->token_embedding) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // Attention weights for all layers
    int attn_size = c->n_layers * c->dim * c->dim;
    w->wq = (float*)calloc(attn_size, sizeof(float));
    if (!w->wq) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->wk = (float*)calloc(attn_size, sizeof(float));
    if (!w->wk) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->wv = (float*)calloc(attn_size, sizeof(float));
    if (!w->wv) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->wo = (float*)calloc(attn_size, sizeof(float));
    if (!w->wo) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // FFN weights for all layers
    w->w1 = (float*)calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    if (!w->w1) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->w2 = (float*)calloc(c->n_layers * c->hidden_dim * c->dim, sizeof(float));
    if (!w->w2) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // Layer norms
    w->ln1_weight = (float*)calloc(c->n_layers * c->dim, sizeof(float));
    if (!w->ln1_weight) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->ln2_weight = (float*)calloc(c->n_layers * c->dim, sizeof(float));
    if (!w->ln2_weight) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // Final layer norm and output
    w->ln_final_weight = (float*)calloc(c->dim, sizeof(float));
    if (!w->ln_final_weight) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    w->output_weight = (float*)calloc(c->dim * c->vocab_size, sizeof(float));
    if (!w->output_weight) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // Initialize layer norms to 1.0
    for (int i = 0; i < c->n_layers * c->dim; i++) {
        w->ln1_weight[i] = 1.0f;
        w->ln2_weight[i] = 1.0f;
    }
    for (int i = 0; i < c->dim; i++) {
        w->ln_final_weight[i] = 1.0f;
    }
}

void malloc_run_state(Transformer* t) {
    Config* c = &t->config;
    RunState* s = &t->state;

    s->x = (float*)calloc(c->max_seq_len * c->dim, sizeof(float));
    if (!s->x) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->xb = (float*)calloc(c->dim, sizeof(float));
    if (!s->xb) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->q = (float*)calloc(c->dim, sizeof(float));  // Only current position
    if (!s->q) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    // KV cache: [n_layers, max_seq_len, dim]
    s->k = (float*)calloc(c->n_layers * c->max_seq_len * c->dim, sizeof(float));
    if (!s->k) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->v = (float*)calloc(c->n_layers * c->max_seq_len * c->dim, sizeof(float));
    if (!s->v) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->att = (float*)calloc(c->n_heads * c->max_seq_len, sizeof(float));
    if (!s->att) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->ffn_hidden = (float*)calloc(c->hidden_dim, sizeof(float));
    if (!s->ffn_hidden) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    s->logits = (float*)calloc(c->vocab_size, sizeof(float));
    if (!s->logits) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
}

void free_transformer(Transformer* t) {
    Weights* w = &t->weights;
    RunState* s = &t->state;

    free(w->token_embedding);
    free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->w1); free(w->w2);
    free(w->ln1_weight); free(w->ln2_weight);
    free(w->ln_final_weight); free(w->output_weight);

    free(s->x); free(s->xb);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->ffn_hidden); free(s->logits);
}

// ============================================================
// Core Operations
// ============================================================

void rmsnorm(float* out, float* x, float* weight, int size) {
    // RMS normalization
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void softmax(float* x, int size) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* out, float* a, float* b, int m, int k, int n) {
    // out[m,n] = a[m,k] @ b[k,n]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// ============================================================
// RoPE Positional Encoding
// ============================================================

void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            int idx = h * head_dim + i;

            // Rotate q
            float q0 = q[idx];
            float q1 = q[idx + 1];
            q[idx] = q0 * cos_val - q1 * sin_val;
            q[idx + 1] = q0 * sin_val + q1 * cos_val;

            // Rotate k
            float k0 = k[idx];
            float k1 = k[idx + 1];
            k[idx] = k0 * cos_val - k1 * sin_val;
            k[idx + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ============================================================
// Forward Pass (Single Token)
// ============================================================

void forward(Transformer* t, int* tokens, int pos) {
    Config* c = &t->config;
    Weights* w = &t->weights;
    RunState* s = &t->state;

    int dim = c->dim;
    int head_dim = c->head_dim;
    int n_heads = c->n_heads;
    int hidden_dim = c->hidden_dim;

    // Get token embedding
    int token = tokens[pos];
    float* x = s->x + pos * dim;
    memcpy(x, w->token_embedding + token * dim, dim * sizeof(float));

    // Process through layers
    for (int layer = 0; layer < c->n_layers; layer++) {
        // Layer offsets
        float* wq = w->wq + layer * dim * dim;
        float* wk = w->wk + layer * dim * dim;
        float* wv = w->wv + layer * dim * dim;
        float* wo = w->wo + layer * dim * dim;
        float* w1 = w->w1 + layer * dim * hidden_dim;
        float* w2 = w->w2 + layer * hidden_dim * dim;
        float* ln1 = w->ln1_weight + layer * dim;
        float* ln2 = w->ln2_weight + layer * dim;

        // Pre-norm
        rmsnorm(s->xb, x, ln1, dim);

        // QKV projections - K and V go to per-layer cache
        float* q = s->q;  // Current position only
        // KV cache offset: [layer, pos, dim]
        int kv_offset = layer * c->max_seq_len * dim + pos * dim;
        float* k = s->k + kv_offset;
        float* v = s->v + kv_offset;

        matmul(q, s->xb, wq, 1, dim, dim);
        matmul(k, s->xb, wk, 1, dim, dim);
        matmul(v, s->xb, wv, 1, dim, dim);

        // Apply RoPE
        apply_rope(q, k, pos, head_dim, n_heads);

        // Multi-head attention
        for (int h = 0; h < n_heads; h++) {
            float* att = s->att + h * c->max_seq_len;

            // Attention scores for this head
            for (int t = 0; t <= pos; t++) {
                float score = 0.0f;
                // Key from this layer's cache
                float* kt = s->k + layer * c->max_seq_len * dim + t * dim + h * head_dim;
                float* qt = q + h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    score += qt[i] * kt[i];
                }
                att[t] = score / sqrtf((float)head_dim);
            }

            // Softmax
            softmax(att, pos + 1);

            // Weighted sum of values
            float* out = s->xb + h * head_dim;
            memset(out, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // Value from this layer's cache
                float* vt = s->v + layer * c->max_seq_len * dim + t * dim + h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    out[i] += att[t] * vt[i];
                }
            }
        }

        // Output projection and residual
        float* attn_out = (float*)malloc(dim * sizeof(float));
        matmul(attn_out, s->xb, wo, 1, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += attn_out[i];
        }
        free(attn_out);

        // FFN
        rmsnorm(s->xb, x, ln2, dim);

        // w1: [dim] -> [hidden_dim] with SiLU
        float* ffn = s->ffn_hidden;
        matmul(ffn, s->xb, w1, 1, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            // SiLU activation: x * sigmoid(x)
            ffn[i] = ffn[i] * (1.0f / (1.0f + expf(-ffn[i])));
        }

        // w2: [hidden_dim] -> [dim]
        float* ffn_out = (float*)malloc(dim * sizeof(float));
        matmul(ffn_out, ffn, w2, 1, hidden_dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            x[i] += ffn_out[i];
        }
        free(ffn_out);
    }

    // Final layer norm
    rmsnorm(s->xb, x, w->ln_final_weight, dim);

    // Output projection to logits
    matmul(s->logits, s->xb, w->output_weight, 1, dim, c->vocab_size);
}

// ============================================================
// Sampling
// ============================================================

int sample(float* logits, int vocab_size, float temperature) {
    if (temperature == 0.0f) {
        // Greedy
        int max_idx = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[max_idx]) max_idx = i;
        }
        return max_idx;
    }

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    softmax(logits, vocab_size);

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return vocab_size - 1;
}

// ============================================================
// Generation
// ============================================================

void generate(Transformer* t, char* prompt, int max_tokens, float temperature) {
    int tokens[MAX_SEQ_LEN];
    int n_tokens = strlen(prompt);
    
    // Protect against overflow
    if (n_tokens >= MAX_SEQ_LEN) {
        fprintf(stderr, "Warning: Prompt truncated from %d to %d characters\n", n_tokens, MAX_SEQ_LEN - 1);
        n_tokens = MAX_SEQ_LEN - 1;
    }

    // Tokenize prompt (char-level)
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = (unsigned char)prompt[i];
    }

    // Process prompt
    for (int pos = 0; pos < n_tokens; pos++) {
        forward(t, tokens, pos);
    }

    // Generate
    printf("%s", prompt);
    for (int i = 0; i < max_tokens && n_tokens < MAX_SEQ_LEN; i++) {
        int next_token = sample(t->state.logits, t->config.vocab_size, temperature);
        tokens[n_tokens] = next_token;

        printf("%c", (char)next_token);
        fflush(stdout);

        forward(t, tokens, n_tokens);
        n_tokens++;
    }
    printf("\n");
}

// ============================================================
// I/O
// ============================================================

int save_weights(Transformer* t, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    Config* c = &t->config;
    Weights* w = &t->weights;

    // Write config
    fwrite(c, sizeof(Config), 1, f);

    // Write weights
    fwrite(w->token_embedding, sizeof(float), c->vocab_size * c->dim, f);
    fwrite(w->wq, sizeof(float), c->n_layers * c->dim * c->dim, f);
    fwrite(w->wk, sizeof(float), c->n_layers * c->dim * c->dim, f);
    fwrite(w->wv, sizeof(float), c->n_layers * c->dim * c->dim, f);
    fwrite(w->wo, sizeof(float), c->n_layers * c->dim * c->dim, f);
    fwrite(w->w1, sizeof(float), c->n_layers * c->dim * c->hidden_dim, f);
    fwrite(w->w2, sizeof(float), c->n_layers * c->hidden_dim * c->dim, f);
    fwrite(w->ln1_weight, sizeof(float), c->n_layers * c->dim, f);
    fwrite(w->ln2_weight, sizeof(float), c->n_layers * c->dim, f);
    fwrite(w->ln_final_weight, sizeof(float), c->dim, f);
    fwrite(w->output_weight, sizeof(float), c->dim * c->vocab_size, f);

    fclose(f);
    return 0;
}

int load_weights(Transformer* t, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open weights file: %s\n", path);
        return -1;
    }

    Config* c = &t->config;

    // Read config
    if (fread(c, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read config from weights file\n");
        fclose(f);
        return -1;
    }

    // Allocate
    malloc_weights(t);
    malloc_run_state(t);

    Weights* w = &t->weights;

    // Read weights
    if (fread(w->token_embedding, sizeof(float), c->vocab_size * c->dim, f) != (size_t)(c->vocab_size * c->dim)) {
        fprintf(stderr, "Error: Failed to read token embeddings from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->wq, sizeof(float), c->n_layers * c->dim * c->dim, f) != (size_t)(c->n_layers * c->dim * c->dim)) {
        fprintf(stderr, "Error: Failed to read wq weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->wk, sizeof(float), c->n_layers * c->dim * c->dim, f) != (size_t)(c->n_layers * c->dim * c->dim)) {
        fprintf(stderr, "Error: Failed to read wk weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->wv, sizeof(float), c->n_layers * c->dim * c->dim, f) != (size_t)(c->n_layers * c->dim * c->dim)) {
        fprintf(stderr, "Error: Failed to read wv weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->wo, sizeof(float), c->n_layers * c->dim * c->dim, f) != (size_t)(c->n_layers * c->dim * c->dim)) {
        fprintf(stderr, "Error: Failed to read wo weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->w1, sizeof(float), c->n_layers * c->dim * c->hidden_dim, f) != (size_t)(c->n_layers * c->dim * c->hidden_dim)) {
        fprintf(stderr, "Error: Failed to read w1 weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->w2, sizeof(float), c->n_layers * c->hidden_dim * c->dim, f) != (size_t)(c->n_layers * c->hidden_dim * c->dim)) {
        fprintf(stderr, "Error: Failed to read w2 weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->ln1_weight, sizeof(float), c->n_layers * c->dim, f) != (size_t)(c->n_layers * c->dim)) {
        fprintf(stderr, "Error: Failed to read ln1 weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->ln2_weight, sizeof(float), c->n_layers * c->dim, f) != (size_t)(c->n_layers * c->dim)) {
        fprintf(stderr, "Error: Failed to read ln2 weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->ln_final_weight, sizeof(float), c->dim, f) != (size_t)(c->dim)) {
        fprintf(stderr, "Error: Failed to read final layer norm weights from weights file\n");
        fclose(f);
        return -1;
    }
    if (fread(w->output_weight, sizeof(float), c->dim * c->vocab_size, f) != (size_t)(c->dim * c->vocab_size)) {
        fprintf(stderr, "Error: Failed to read output weights from weights file\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}
