/*
 * sartre_bridge.c - SARTRE Transformer Bridge Implementation
 *
 * Refactored from sartre.c: all NN ops are static, types prefixed Sartre*,
 * public API for init/free/forward/sample/generate.
 *
 * SARTRE: 14.3M params, dim=416, layers=7, heads=8, kv_heads=2 (GQA),
 *         vocab=93, hidden=1280, head_dim=52, max_seq=256
 */

#include "sartre_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Static NN Operations (no symbol conflicts with Arianna)
 * ============================================================================ */

static void sartre_rms_norm(float* out, float* x, float* weight, int size) {
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

static void sartre_matmul(float* out, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

static void sartre_softmax(float* x, int size) {
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

static void sartre_apply_rope(float* q, float* k, float* rope_cos, float* rope_sin,
                               int n_heads, int n_kv_heads, int head_dim, int pos) {
    int half = head_dim / 2;
    float* cos_p = rope_cos + pos * half;
    float* sin_p = rope_sin + pos * half;

    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[2*i];
            float q1 = qh[2*i + 1];
            qh[2*i]     = q0 * cos_p[i] - q1 * sin_p[i];
            qh[2*i + 1] = q0 * sin_p[i] + q1 * cos_p[i];
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k0 = kh[2*i];
            float k1 = kh[2*i + 1];
            kh[2*i]     = k0 * cos_p[i] - k1 * sin_p[i];
            kh[2*i + 1] = k0 * sin_p[i] + k1 * cos_p[i];
        }
    }
}

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

static int sartre_malloc_run_state(SartreRunState* s, SartreConfig* c) {
    int kv_dim = c->n_kv_heads * c->head_dim;

    s->x      = calloc(c->dim, sizeof(float));
    s->xb     = calloc(c->dim, sizeof(float));
    s->xb2    = calloc(c->dim, sizeof(float));
    s->hb     = calloc(c->hidden_dim, sizeof(float));
    s->hb2    = calloc(c->hidden_dim, sizeof(float));

    s->q      = calloc(c->n_heads * c->head_dim, sizeof(float));
    s->k      = calloc(kv_dim, sizeof(float));
    s->v      = calloc(kv_dim, sizeof(float));
    s->att    = calloc(c->n_heads * c->max_seq_len, sizeof(float));

    s->key_cache   = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));

    s->rope_cos = calloc(c->max_seq_len * (c->head_dim / 2), sizeof(float));
    s->rope_sin = calloc(c->max_seq_len * (c->head_dim / 2), sizeof(float));

    s->logits = calloc(c->vocab_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att ||
        !s->key_cache || !s->value_cache ||
        !s->rope_cos || !s->rope_sin || !s->logits) {
        return -1;
    }

    /* Precompute RoPE frequencies */
    float theta = 10000.0f;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        for (int i = 0; i < c->head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / c->head_dim);
            float angle = pos * freq;
            s->rope_cos[pos * (c->head_dim / 2) + i] = cosf(angle);
            s->rope_sin[pos * (c->head_dim / 2) + i] = sinf(angle);
        }
    }

    return 0;
}

static void sartre_free_run_state(SartreRunState* s) {
    free(s->x);       free(s->xb);      free(s->xb2);
    free(s->hb);      free(s->hb2);
    free(s->q);       free(s->k);       free(s->v);
    free(s->att);
    free(s->key_cache);  free(s->value_cache);
    free(s->rope_cos);   free(s->rope_sin);
    free(s->logits);
}

/* ============================================================================
 * Weight Loading
 * ============================================================================ */

static void sartre_map_weights(SartreWeights* w, SartreConfig* c, float* ptr) {
    int kv_dim = c->n_kv_heads * c->head_dim;

    /* Token embeddings (direct pointer — first in file) */
    w->tok_emb = ptr;
    ptr += c->vocab_size * c->dim;

    /* Per-layer: calloc + memcpy (sartre.c original approach) */
    w->attn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->wq       = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->wk       = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wv       = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wo       = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->ffn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->w_gate   = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_up     = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_down   = calloc(c->n_layers * c->hidden_dim * c->dim, sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        memcpy(w->attn_norm + l * c->dim, ptr, c->dim * sizeof(float));
        ptr += c->dim;

        memcpy(w->wq + l * c->dim * c->dim, ptr, c->dim * c->dim * sizeof(float));
        ptr += c->dim * c->dim;

        memcpy(w->wk + l * c->dim * kv_dim, ptr, c->dim * kv_dim * sizeof(float));
        ptr += c->dim * kv_dim;

        memcpy(w->wv + l * c->dim * kv_dim, ptr, c->dim * kv_dim * sizeof(float));
        ptr += c->dim * kv_dim;

        memcpy(w->wo + l * c->dim * c->dim, ptr, c->dim * c->dim * sizeof(float));
        ptr += c->dim * c->dim;

        memcpy(w->ffn_norm + l * c->dim, ptr, c->dim * sizeof(float));
        ptr += c->dim;

        memcpy(w->w_gate + l * c->dim * c->hidden_dim, ptr, c->dim * c->hidden_dim * sizeof(float));
        ptr += c->dim * c->hidden_dim;

        memcpy(w->w_up + l * c->dim * c->hidden_dim, ptr, c->dim * c->hidden_dim * sizeof(float));
        ptr += c->dim * c->hidden_dim;

        memcpy(w->w_down + l * c->hidden_dim * c->dim, ptr, c->hidden_dim * c->dim * sizeof(float));
        ptr += c->hidden_dim * c->dim;
    }

    /* Final norm + lm_head (direct pointers into weight blob) */
    w->final_norm = ptr;
    ptr += c->dim;

    w->lm_head = ptr;
}

static void sartre_free_weights(SartreWeights* w) {
    /* Only free the calloc'd per-layer arrays, not direct pointers */
    free(w->attn_norm);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->ffn_norm);
    free(w->w_gate);
    free(w->w_up);
    free(w->w_down);
}

/* ============================================================================
 * Config Loading (from JSON)
 * ============================================================================ */

static void sartre_default_config(SartreConfig* c) {
    c->dim         = 416;
    c->n_layers    = 7;
    c->n_heads     = 8;
    c->n_kv_heads  = 2;
    c->vocab_size  = 93;
    c->max_seq_len = 256;
    c->head_dim    = 52;
    c->hidden_dim  = 1280;
    c->n_kv_groups = 4;
}

static int sartre_parse_int(const char* json, const char* key) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char* p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

static int sartre_load_config(SartreConfig* c, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (len <= 0 || len > 4096) { fclose(f); return -1; }

    char* buf = malloc(len + 1);
    if (!buf) { fclose(f); return -1; }
    if ((long)fread(buf, 1, len, f) != len) { free(buf); fclose(f); return -1; }
    buf[len] = '\0';
    fclose(f);

    /* Start with defaults */
    sartre_default_config(c);

    int v;
    if ((v = sartre_parse_int(buf, "dim"))         > 0) c->dim = v;
    if ((v = sartre_parse_int(buf, "n_layers"))     > 0) c->n_layers = v;
    if ((v = sartre_parse_int(buf, "n_heads"))      > 0) c->n_heads = v;
    if ((v = sartre_parse_int(buf, "n_kv_heads"))   > 0) c->n_kv_heads = v;
    if ((v = sartre_parse_int(buf, "vocab_size"))   > 0) c->vocab_size = v;
    if ((v = sartre_parse_int(buf, "max_seq_len"))  > 0) c->max_seq_len = v;
    if ((v = sartre_parse_int(buf, "head_dim"))     > 0) c->head_dim = v;
    if ((v = sartre_parse_int(buf, "hidden_dim"))   > 0) c->hidden_dim = v;
    if ((v = sartre_parse_int(buf, "n_kv_groups"))  > 0) c->n_kv_groups = v;

    /* Derive if not explicitly set */
    if (c->head_dim <= 0 && c->n_heads > 0) c->head_dim = c->dim / c->n_heads;
    if (c->n_kv_groups <= 0 && c->n_kv_heads > 0) c->n_kv_groups = c->n_heads / c->n_kv_heads;

    free(buf);
    return 0;
}

/* ============================================================================
 * Tokenizer Loading (from JSON, char-level)
 * ============================================================================ */

static int sartre_load_tokenizer(SartreTokenizer* t, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (len <= 0 || len > 65536) { fclose(f); return -1; }

    char* content = malloc(len + 1);
    if (!content) { fclose(f); return -1; }
    if ((long)fread(content, 1, len, f) != len) { free(content); fclose(f); return -1; }
    content[len] = '\0';
    fclose(f);

    /* Parse vocab_size */
    char* vs = strstr(content, "\"vocab_size\":");
    t->vocab_size = vs ? atoi(vs + 13) : 93;

    if (t->vocab_size < 1 || t->vocab_size > 256) {
        free(content);
        return -1;
    }

    /* Initialize */
    memset(t->chars, 0, sizeof(t->chars));
    for (int i = 0; i < 256; i++) t->char_to_id[i] = -1;

    /* Parse char_to_id */
    char* p = strstr(content, "\"char_to_id\":");
    if (p) {
        p = strchr(p, '{');
        if (p) {
            p++;
            while (*p && *p != '}') {
                while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
                if (*p == '}') break;
                if (*p != '"') { p++; continue; }

                p++; /* skip opening quote */
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
                    c = (unsigned char)*p;
                    p++;
                }

                while (*p && *p != ':') p++;
                if (*p == ':') p++;
                while (*p == ' ') p++;
                int id = atoi(p);

                if (c >= 0 && c < 256 && id >= 0 && id < t->vocab_size) {
                    t->char_to_id[c] = id;
                    t->chars[id] = (char)c;
                }

                while (*p && *p != ',' && *p != '}') p++;
            }
        }
    }

    free(content);
    return 0;
}

/* ============================================================================
 * Forward Pass (GQA attention)
 * ============================================================================ */

static void sartre_forward_pass(SartreConfig* c, SartreWeights* w,
                                 SartreRunState* s, int token, int pos) {
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden_dim = c->hidden_dim;

    /* Token embedding */
    memcpy(s->x, w->tok_emb + token * dim, dim * sizeof(float));

    /* Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        /* Pre-norm for attention */
        sartre_rms_norm(s->xb, s->x, w->attn_norm + l * dim, dim);

        /* QKV projection */
        sartre_matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        sartre_matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        sartre_matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        /* RoPE */
        sartre_apply_rope(s->q, s->k, s->rope_cos, s->rope_sin,
                          c->n_heads, c->n_kv_heads, c->head_dim, pos);

        /* Store in KV cache */
        int kv_off = l * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + kv_off, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + kv_off, s->v, kv_dim * sizeof(float));

        /* Multi-head attention with GQA */
        memset(s->xb, 0, dim * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            float* qh = s->q + h * c->head_dim;
            float* atth = s->att + h * c->max_seq_len;
            int kv_h = h / c->n_kv_groups;

            /* Attention scores */
            float scale = 1.0f / sqrtf((float)c->head_dim);
            for (int t = 0; t <= pos; t++) {
                float* kh = s->key_cache + l * c->max_seq_len * kv_dim
                            + t * kv_dim + kv_h * c->head_dim;
                float score = 0.0f;
                for (int i = 0; i < c->head_dim; i++) {
                    score += qh[i] * kh[i];
                }
                atth[t] = score * scale;
            }

            sartre_softmax(atth, pos + 1);

            /* Weighted sum of values */
            float* xbh = s->xb + h * c->head_dim;
            memset(xbh, 0, c->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vh = s->value_cache + l * c->max_seq_len * kv_dim
                            + t * kv_dim + kv_h * c->head_dim;
                float a = atth[t];
                for (int i = 0; i < c->head_dim; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }

        /* Output projection */
        sartre_matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        /* Residual */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* Pre-norm for FFN */
        sartre_rms_norm(s->xb, s->x, w->ffn_norm + l * dim, dim);

        /* SwiGLU FFN */
        sartre_matmul(s->hb,  s->xb, w->w_gate + l * dim * hidden_dim, dim, hidden_dim);
        sartre_matmul(s->hb2, s->xb, w->w_up   + l * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float gate = s->hb[i];
            float silu = gate / (1.0f + expf(-gate));
            s->hb[i] = silu * s->hb2[i];
        }

        sartre_matmul(s->xb, s->hb, w->w_down + l * hidden_dim * dim, hidden_dim, dim);

        /* Residual */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    /* Final norm */
    sartre_rms_norm(s->x, s->x, w->final_norm, dim);

    /* Output logits */
    sartre_matmul(s->logits, s->x, w->lm_head, dim, c->vocab_size);
}

/* ============================================================================
 * Sampling
 * ============================================================================ */

static int sartre_sample_multinomial(float* probs, int n) {
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

int sartre_transformer_init(SartreTransformer* st,
                             const char* weights_path,
                             const char* tokenizer_path,
                             const char* config_path) {
    memset(st, 0, sizeof(SartreTransformer));

    /* Load config */
    if (config_path && sartre_load_config(&st->config, config_path) == 0) {
        fprintf(stderr, "[sartre] config from %s\n", config_path);
    } else {
        sartre_default_config(&st->config);
        fprintf(stderr, "[sartre] using default config\n");
    }

    /* Load tokenizer */
    if (sartre_load_tokenizer(&st->tokenizer, tokenizer_path) != 0) {
        fprintf(stderr, "[sartre] failed to load tokenizer: %s\n", tokenizer_path);
        return -1;
    }

    /* Override vocab_size from tokenizer if different */
    if (st->tokenizer.vocab_size > 0) {
        st->config.vocab_size = st->tokenizer.vocab_size;
    }

    fprintf(stderr, "[sartre] dim=%d layers=%d heads=%d kv_heads=%d vocab=%d hidden=%d\n",
            st->config.dim, st->config.n_layers, st->config.n_heads,
            st->config.n_kv_heads, st->config.vocab_size, st->config.hidden_dim);

    /* Load weights */
    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        fprintf(stderr, "[sartre] failed to open weights: %s\n", weights_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    fprintf(stderr, "[sartre] loading %.2f MB from %s\n",
            file_size / (1024.0f * 1024.0f), weights_path);

    /* Validate file size */
    SartreConfig* c = &st->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    long expected = (long)(
        c->vocab_size * c->dim +
        c->n_layers * (
            c->dim +
            c->dim * c->dim +
            c->dim * kv_dim +
            c->dim * kv_dim +
            c->dim * c->dim +
            c->dim +
            c->dim * c->hidden_dim +
            c->dim * c->hidden_dim +
            c->hidden_dim * c->dim
        ) +
        c->dim +
        c->vocab_size * c->dim
    ) * (long)sizeof(float);

    if (file_size < expected) {
        fprintf(stderr, "[sartre] weights file too small (%ld < %ld bytes)\n",
                file_size, expected);
        fclose(f);
        return -1;
    }

    st->weight_data = malloc(file_size);
    if (!st->weight_data) {
        fprintf(stderr, "[sartre] failed to allocate %ld bytes for weights\n", file_size);
        fclose(f);
        return -1;
    }

    if ((long)fread(st->weight_data, 1, file_size, f) != file_size) {
        fprintf(stderr, "[sartre] failed to read weights\n");
        free(st->weight_data);
        st->weight_data = NULL;
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Map weights */
    sartre_map_weights(&st->weights, &st->config, st->weight_data);

    /* Allocate run state */
    if (sartre_malloc_run_state(&st->state, &st->config) != 0) {
        fprintf(stderr, "[sartre] failed to allocate run state\n");
        sartre_free_weights(&st->weights);
        free(st->weight_data);
        st->weight_data = NULL;
        return -1;
    }

    st->initialized = 1;
    st->current_pos = 0;

    fprintf(stderr, "[sartre] initialized successfully\n");
    return 0;
}

void sartre_transformer_free(SartreTransformer* st) {
    if (!st->initialized) return;

    sartre_free_run_state(&st->state);
    sartre_free_weights(&st->weights);
    free(st->weight_data);

    st->weight_data = NULL;
    st->initialized = 0;
}

void sartre_forward(SartreTransformer* st, int token) {
    if (!st->initialized) return;
    if (st->current_pos >= st->config.max_seq_len) return;

    sartre_forward_pass(&st->config, &st->weights, &st->state,
                        token, st->current_pos);
    st->current_pos++;
}

void sartre_reset_state(SartreTransformer* st) {
    if (!st->initialized) return;

    int kv_dim = st->config.n_kv_heads * st->config.head_dim;
    int kv_size = st->config.n_layers * st->config.max_seq_len * kv_dim;

    memset(st->state.key_cache, 0, kv_size * sizeof(float));
    memset(st->state.value_cache, 0, kv_size * sizeof(float));
    memset(st->state.x, 0, st->config.dim * sizeof(float));

    st->current_pos = 0;
}

int sartre_encode_char(SartreTransformer* st, char c) {
    int id = st->tokenizer.char_to_id[(unsigned char)c];
    return id >= 0 ? id : 1; /* unknown -> space (id=1) */
}

char sartre_decode_char(SartreTransformer* st, int id) {
    if (id >= 0 && id < st->tokenizer.vocab_size) {
        return st->tokenizer.chars[id];
    }
    return '?';
}

int sartre_sample(SartreTransformer* st, float temperature, float top_p) {
    if (!st->initialized) return 0;

    int n = st->config.vocab_size;
    float* logits = st->state.logits;

    /* Apply temperature */
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < n; i++) {
            logits[i] /= temperature;
        }
    }

    /* Softmax */
    sartre_softmax(logits, n);

    /* Top-p (nucleus) sampling */
    if (top_p < 1.0f) {
        /* Sort indices by probability (bubble sort — fine for vocab=93) */
        int indices[256]; /* vocab <= 256 guaranteed */
        for (int i = 0; i < n; i++) indices[i] = i;

        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (logits[indices[j]] < logits[indices[j+1]]) {
                    int tmp = indices[j];
                    indices[j] = indices[j+1];
                    indices[j+1] = tmp;
                }
            }
        }

        /* Accumulate until top_p */
        float cumsum = 0.0f;
        int cutoff = n;
        for (int i = 0; i < n; i++) {
            cumsum += logits[indices[i]];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        /* Zero beyond cutoff */
        for (int i = cutoff; i < n; i++) {
            logits[indices[i]] = 0.0f;
        }

        /* Renormalize */
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += logits[i];
        if (sum > 0.0f) {
            for (int i = 0; i < n; i++) logits[i] /= sum;
        } else {
            for (int i = 0; i < cutoff && i < n; i++)
                logits[indices[i]] = 1.0f / cutoff;
        }
    }

    return sartre_sample_multinomial(logits, n);
}

int sartre_feed_prompt(SartreTransformer* st, const char* text, int len) {
    if (!st->initialized) return 0;

    int processed = 0;
    for (int i = 0; i < len && st->current_pos < st->config.max_seq_len; i++) {
        int token = sartre_encode_char(st, text[i]);
        sartre_forward(st, token);
        processed++;
    }
    return processed;
}

int sartre_generate(SartreTransformer* st, char* output, int max_len,
                    int max_tokens, float temperature, float top_p,
                    int stop_newline) {
    if (!st->initialized) return 0;

    int gen = 0;
    for (int i = 0; i < max_tokens && gen < max_len - 1 &&
         st->current_pos < st->config.max_seq_len; i++) {
        int next = sartre_sample(st, temperature, top_p);
        char c = sartre_decode_char(st, next);

        output[gen++] = c;

        if (stop_newline && c == '\n') break;

        sartre_forward(st, next);
    }

    output[gen] = '\0';
    return gen;
}
