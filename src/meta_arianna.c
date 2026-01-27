/*
 * meta_arianna.c - MetaArianna: Pulsating Meta-Observer
 *
 * "Inhale -> observe -> exhale. Breathing."
 *
 * FluidTransformer: ephemeral 20M observer instances.
 * Born, observe Arianna<->SARTRE dialogue, extract thermogram, die.
 *
 * Uses shared building blocks from ariannabody.c:
 *   rms_norm, matmul, softmax, apply_rope,
 *   malloc_weights, malloc_run_state, free_transformer
 *
 * Custom meta_forward() adds attention biases + layer focus
 * per template — the FluidTransformer's core differentiator.
 *
 * 20M Config: dim=448, layers=8, heads=8, kv_heads=8,
 *             hidden=1280, vocab=84, head_dim=56
 */

#include "meta_arianna.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ============================================================
 * Internal: Tokenizer loading (observer's own, not global)
 *
 * MetaArianna has vocab=84 vs Arianna 34M vocab=86.
 * Separate tokenizer avoids global state conflicts.
 * ============================================================ */

static int meta_load_tokenizer(FluidTransformer* ft, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[meta] cannot open tokenizer: %s\n", path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long len = ftell(f);
    if (len < 0 || len > 10 * 1024 * 1024) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

    char* content = malloc((size_t)len + 1);
    if (!content) { fclose(f); return -1; }
    if (fread(content, 1, (size_t)len, f) != (size_t)len) {
        free(content); fclose(f); return -1;
    }
    content[len] = '\0';
    fclose(f);

    /* Parse vocab_size */
    char* vs = strstr(content, "\"vocab_size\":");
    ft->obs_vocab_size = vs ? atoi(vs + 14) : 84;

    /* Initialize mappings */
    memset(ft->obs_vocab, 0, sizeof(ft->obs_vocab));
    for (int i = 0; i < 256; i++) ft->obs_char_to_id[i] = 1; /* default: space */

    /* Parse char_to_id (same format as ariannabody.c) */
    char* p = strstr(content, "\"char_to_id\":");
    if (p) {
        p = strchr(p, '{');
        if (p) {
            p++;
            while (*p && *p != '}') {
                while (*p == ' ' || *p == '\n' || *p == '\r' ||
                       *p == '\t' || *p == ',') p++;
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
                    while ((*p & 0xC0) == 0x80) p++; /* skip multibyte */
                }

                /* Skip to colon */
                while (*p && *p != ':' && *p != '"') p++;
                if (*p == '"') p++;
                while (*p && *p != ':') p++;
                if (*p == ':') p++;
                while (*p == ' ') p++;

                int id = atoi(p);

                if (c >= 0 && c < 256 && id >= 0 && id < ft->obs_vocab_size) {
                    ft->obs_char_to_id[c] = id;
                    if (id < 128) ft->obs_vocab[id] = (char)c;
                }

                while (*p && *p != ',' && *p != '}') p++;
            }
        }
    }

    free(content);
    fprintf(stderr, "[meta] tokenizer: %d tokens from %s\n",
            ft->obs_vocab_size, path);
    return 0;
}

/* ============================================================
 * meta_init — allocate RunState, load weights + tokenizer
 * ============================================================ */

int meta_init(FluidTransformer* ft,
              const char* weights_path,
              const char* tokenizer_path)
{
    memset(ft, 0, sizeof(FluidTransformer));

    /* Load tokenizer first (sets obs_vocab_size) */
    if (meta_load_tokenizer(ft, tokenizer_path) != 0) return -1;

    /* Open weights file */
    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        fprintf(stderr, "[meta] cannot open weights: %s\n", weights_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    fprintf(stderr, "[meta] loading %.2f MB from %s\n",
            file_size / 1024.0f / 1024.0f, weights_path);

    Config* c = &ft->observer.config;

    /* Try magic number for embedded config (0x616B616E) */
    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) == 1 && magic == 0x616B616E) {
        fprintf(stderr, "[meta] reading embedded config...\n");
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
            fprintf(stderr, "[meta] error reading embedded config\n");
            fclose(f);
            return -1;
        }
    } else {
        /* 20M defaults (no magic — legacy weight file) */
        fseek(f, 0, SEEK_SET);
        c->dim = 448;
        c->n_layers = 8;
        c->n_heads = 8;
        c->n_kv_heads = 8;
        c->head_dim = 56;      /* 448 / 8 */
        c->hidden_dim = 1280;
        c->max_seq_len = 512;
        c->vocab_size = ft->obs_vocab_size;
        c->n_kv_groups = 1;    /* 8 / 8 */
        c->rope_theta = 10000.0f;
        c->norm_eps = 1e-5f;
        fprintf(stderr, "[meta] using 20M default config\n");
    }

    fprintf(stderr, "[meta] dim=%d layers=%d heads=%d kv=%d vocab=%d hidden=%d\n",
            c->dim, c->n_layers, c->n_heads, c->n_kv_heads,
            c->vocab_size, c->hidden_dim);

    /* Allocate weights + RunState (reuse ariannabody.c infrastructure) */
    malloc_weights(&ft->observer);
    malloc_run_state(&ft->observer);

    /* Read weights (same order as ariannabody.c / dubrovsky export) */
    Weights* w = &ft->observer.weights;
    int dim = c->dim;
    int n_layers = c->n_layers;
    int hidden_dim = c->hidden_dim;
    int vocab_size = c->vocab_size;
    int kv_dim = c->n_kv_heads * c->head_dim;

    #define META_READ(ptr, count) do { \
        if (fread(ptr, sizeof(float), count, f) != (size_t)(count)) { \
            fprintf(stderr, "[meta] read error at %s\n", #ptr); \
            fclose(f); \
            free_transformer(&ft->observer); \
            return -1; \
        } \
    } while(0)

    META_READ(w->tok_emb, vocab_size * dim);

    for (int l = 0; l < n_layers; l++) {
        META_READ(w->attn_norm + l * dim, dim);
        META_READ(w->wq + l * dim * dim, dim * dim);
        META_READ(w->wk + l * dim * kv_dim, dim * kv_dim);
        META_READ(w->wv + l * dim * kv_dim, dim * kv_dim);
        META_READ(w->wo + l * dim * dim, dim * dim);
        META_READ(w->ffn_norm + l * dim, dim);
        META_READ(w->w_gate + l * dim * hidden_dim, dim * hidden_dim);
        META_READ(w->w_up + l * dim * hidden_dim, dim * hidden_dim);
        META_READ(w->w_down + l * hidden_dim * dim, hidden_dim * dim);
    }

    META_READ(w->final_norm, dim);
    META_READ(w->lm_head, vocab_size * dim);

    #undef META_READ

    fclose(f);

    ft->initialized = 1;
    ft->weights_loaded = 1;
    fprintf(stderr, "[meta] FluidTransformer ready (observer 20M)\n");
    return 0;
}

/* ============================================================
 * meta_free — release all resources
 * ============================================================ */

void meta_free(FluidTransformer* ft) {
    if (!ft->initialized) return;
    free_transformer(&ft->observer);
    ft->initialized = 0;
    ft->weights_loaded = 0;
}

/* ============================================================
 * meta_forward — forward pass with attention biases + layer focus
 *
 * Identical to forward() from ariannabody.c EXCEPT:
 *   1. attention_biases[h] added to attention scores per head
 *   2. layer_focus[layer] scales residual contributions
 *
 * These two knobs are what make each template "see" differently:
 *   - Silence Observer focuses early layers, biases Q heads
 *   - Drift Detector focuses middle layers, biases K heads
 *   - Field Reader uses all layers with high temperature
 * ============================================================ */

static void meta_forward(FluidTransformer* ft, int token, int pos) {
    Transformer* tr = &ft->observer;
    Config* c = &tr->config;
    Weights* w = &tr->weights;
    RunState* s = &tr->state;
    const MetaTemplateParams* tp = &ft->params;

    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden_dim = c->hidden_dim;

    /* Token embedding */
    float* tok_vec = w->tok_emb + token * dim;
    memcpy(s->x, tok_vec, dim * sizeof(float));

    /* Transformer layers */
    for (int layer = 0; layer < c->n_layers; layer++) {
        float focus = (layer < 8) ? tp->layer_focus[layer] : 1.0f;

        /* Pre-norm for attention */
        rms_norm(s->xb, s->x, w->attn_norm + layer * dim, dim, c->norm_eps);

        /* QKV projection */
        matmul(s->q, s->xb, w->wq + layer * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + layer * dim * kv_dim, dim, kv_dim);

        /* RoPE */
        apply_rope(s->q, s->k, s->rope_cos, s->rope_sin,
                   c->n_heads, c->n_kv_heads, c->head_dim, pos);

        /* KV cache */
        int kv_off = layer * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + kv_off, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + kv_off, s->v, kv_dim * sizeof(float));

        /* Multi-head attention with per-head output scaling.
         * Template biases scale each head's contribution (louder/quieter),
         * NOT added to attention scores (softmax is shift-invariant). */
        memset(s->xb, 0, dim * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            float* qh = s->q + h * c->head_dim;
            float* atth = s->att + h * c->max_seq_len;
            int kv_h = h / c->n_kv_groups;
            float head_scale = (h < 8) ? (1.0f + tp->attention_biases[h]) : 1.0f;

            /* Attention scores (standard scaled dot-product) */
            float scale = 1.0f / sqrtf((float)c->head_dim);
            for (int ts = 0; ts <= pos; ts++) {
                float* kh = s->key_cache
                          + layer * c->max_seq_len * kv_dim
                          + ts * kv_dim + kv_h * c->head_dim;
                float score = 0.0f;
                for (int i = 0; i < c->head_dim; i++) {
                    score += qh[i] * kh[i];
                }
                atth[ts] = score * scale;
            }

            softmax(atth, pos + 1);

            /* Weighted sum of values, scaled by template bias */
            float* xbh = s->xb + h * c->head_dim;
            for (int ts = 0; ts <= pos; ts++) {
                float* vh = s->value_cache
                          + layer * c->max_seq_len * kv_dim
                          + ts * kv_dim + kv_h * c->head_dim;
                float a = atth[ts] * head_scale;
                for (int i = 0; i < c->head_dim; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }

        /* Output projection */
        matmul(s->xb2, s->xb, w->wo + layer * dim * dim, dim, dim);

        /* Residual connection scaled by layer focus */
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i] * focus;
        }

        /* Pre-norm for FFN */
        rms_norm(s->xb, s->x, w->ffn_norm + layer * dim, dim, c->norm_eps);

        /* SwiGLU FFN */
        matmul(s->hb, s->xb,
               w->w_gate + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb,
               w->w_up + layer * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float gate = s->hb[i];
            s->hb[i] = (gate / (1.0f + expf(-gate))) * s->hb2[i];
        }

        matmul(s->xb, s->hb,
               w->w_down + layer * hidden_dim * dim, hidden_dim, dim);

        /* FFN residual scaled by layer focus */
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i] * focus;
        }
    }

    /* Final norm */
    rms_norm(s->x, s->x, w->final_norm, dim, c->norm_eps);

    /* Output logits */
    matmul(s->logits, s->x, w->lm_head, dim, c->vocab_size);
}

/* ============================================================
 * Thermogram extraction helpers
 * ============================================================ */

float meta_compute_entropy(const float* logits, int vocab_size) {
    /* Entropy of softmax(logits), normalized to [0,1].
     * 0 = deterministic (cold), 1 = uniform (warm). */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    float probs[META_MAX_VOCAB]; /* bounded by META_MAX_VOCAB */
    if (vocab_size > META_MAX_VOCAB) vocab_size = META_MAX_VOCAB;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    float entropy = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = probs[i] / sum;
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }

    /* Normalize by max entropy (log(vocab_size)) */
    float log_vs = logf((float)vocab_size);
    return (log_vs > 0.0f) ? entropy / log_vs : 0.0f;
}

float meta_compute_kl_uniform(const float* logits, int vocab_size) {
    /* KL(p || uniform) = log(V) - H(p), normalized to [0,1].
     * = 1 - H(p)/log(V)
     * 0 = uniform (viscous), 1 = deterministic (sharp). */
    float norm_entropy = meta_compute_entropy(logits, vocab_size);
    return 1.0f - norm_entropy;
}

float meta_compute_silence_prob(const float* logits, int vocab_size,
                                const int* char_to_id) {
    /* Probability mass on pause tokens: . , ; : \n space */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    float probs[256];
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    const unsigned char silence_chars[] = {'.', ',', ';', ':', '\n', ' '};
    float silence = 0.0f;
    for (int j = 0; j < 6; j++) {
        int id = char_to_id[silence_chars[j]];
        if (id >= 0 && id < vocab_size) {
            silence += probs[id];
        }
    }
    return silence;
}

/* ============================================================
 * History and drift detection
 * ============================================================ */

void meta_push_history(FluidTransformer* ft,
                       float arousal, float coherence) {
    int pos = ft->history_pos;
    ft->arousal_history[pos] = arousal;
    ft->coherence_history[pos] = coherence;
    ft->history_pos = (pos + 1) % META_HISTORY_SIZE;
    if (ft->history_count < META_HISTORY_SIZE) ft->history_count++;
}

void meta_compute_drift(FluidTransformer* ft,
                        float* drift_rate, int* drift_direction) {
    if (ft->history_count < 4) {
        *drift_rate = 0.0f;
        *drift_direction = 0;
        return;
    }

    /* Compare first half vs second half of recent window */
    int n = ft->history_count < 8 ? ft->history_count : 8;
    int half = n / 2;

    float first_a = 0.0f, second_a = 0.0f;
    float first_c = 0.0f, second_c = 0.0f;

    for (int i = 0; i < half; i++) {
        int idx = (ft->history_pos - n + i + META_HISTORY_SIZE)
                % META_HISTORY_SIZE;
        first_a += ft->arousal_history[idx];
        first_c += ft->coherence_history[idx];
    }
    for (int i = half; i < n; i++) {
        int idx = (ft->history_pos - n + i + META_HISTORY_SIZE)
                % META_HISTORY_SIZE;
        second_a += ft->arousal_history[idx];
        second_c += ft->coherence_history[idx];
    }

    first_a /= half;
    second_a /= (n - half);
    first_c /= half;
    second_c /= (n - half);

    float da = second_a - first_a;
    float dc = second_c - first_c;
    float combined = (da + dc) / 2.0f;

    *drift_rate = fabsf(combined);

    if (combined > 0.05f)       *drift_direction = 1;   /* unfolding */
    else if (combined < -0.05f) *drift_direction = -1;  /* collapsing */
    else                        *drift_direction = 0;    /* stable */
}

/* ============================================================
 * meta_observe — one complete observation cycle
 *
 * Birth -> forward pass -> thermogram extraction -> ready to die
 * ============================================================ */

void meta_observe(FluidTransformer* ft,
                  const MetaTemplateParams* params,
                  const char* dialogue_log, int log_len)
{
    if (!ft->initialized || !ft->weights_loaded) return;
    if (!dialogue_log || log_len <= 0) return;

    /* Set template params for this observation */
    ft->params = *params;

    Config* c = &ft->observer.config;
    RunState* s = &ft->observer.state;

    /* Template prefix — each template "sees" through a different lens */
    static const char* prefixes[META_N_TEMPLATES] = {
        "[THERMAL] ",
        "[SILENCE] ",
        "[DRIFT] ",
        "[FIELD] "
    };
    int tmpl = params->template_type;
    if (tmpl < 0 || tmpl >= META_N_TEMPLATES) tmpl = 0;
    const char* prefix = prefixes[tmpl];

    /* Forward pass: prefix + dialogue_log */
    int pos = 0;
    int max_pos = c->max_seq_len - 1;

    /* Process prefix tokens */
    for (const char* pp = prefix; *pp && pos < max_pos; pp++) {
        int tok = ft->obs_char_to_id[(unsigned char)*pp];
        meta_forward(ft, tok, pos);
        pos++;
    }

    /* Process dialogue log (keep most recent if too long) */
    int start = 0;
    int remaining = max_pos - pos;
    if (log_len > remaining) {
        start = log_len - remaining;
    }

    for (int i = start; i < log_len && pos < max_pos; i++) {
        int tok = ft->obs_char_to_id[(unsigned char)dialogue_log[i]];
        meta_forward(ft, tok, pos);
        pos++;
    }

    /* --- Extract thermogram from final state --- */
    float* logits = s->logits;
    int vs = c->vocab_size;

    /* Apply observation temperature to get meaningful thermogram.
     * Char-level model is too peaked on raw logits — MetaArianna
     * needs to "squint" to see distribution shapes, not raw peaks.
     * Base temp META_OBSERVE_TEMP, modulated by template's temperature param. */
    float obs_temp = META_OBSERVE_TEMP * params->temperature;
    if (obs_temp < 0.1f) obs_temp = 0.1f; /* safety clamp */
    float scaled_logits[META_MAX_VOCAB];
    if (vs > META_MAX_VOCAB) vs = META_MAX_VOCAB;
    for (int i = 0; i < vs; i++) {
        scaled_logits[i] = logits[i] / obs_temp;
    }

    ft->result.warmth      = meta_compute_entropy(scaled_logits, vs);
    ft->result.sharpness   = meta_compute_kl_uniform(scaled_logits, vs);
    ft->result.silence     = meta_compute_silence_prob(scaled_logits, vs,
                                                       ft->obs_char_to_id);
    ft->result.uncertainty = ft->result.warmth; /* entropy IS uncertainty */

    /* Drift from history ring buffer */
    meta_compute_drift(ft, &ft->result.drift_rate,
                       &ft->result.drift_direction);

    /* Field vector: project hidden state (dim=448) to 8D
     * by averaging groups of 56 dimensions */
    int dim = c->dim;
    int group = dim / 8;
    for (int d = 0; d < 8; d++) {
        float sum = 0.0f;
        int si = d * group;
        int ei = (d == 7) ? dim : si + group;
        for (int i = si; i < ei; i++) {
            sum += s->x[i];
        }
        ft->result.field_vector[d] = sum / (ei - si);
    }

    ft->result.valid = 1;
    ft->result.template_used = tmpl;

    static const char* tmpl_names[META_N_TEMPLATES] = {
        "THERMO", "SILENCE", "DRIFT", "FIELD"
    };
    fprintf(stderr, "[meta:%s] warmth=%.3f sharp=%.3f silence=%.3f "
            "drift=%.3f(%+d)\n",
            tmpl_names[tmpl],
            ft->result.warmth, ft->result.sharpness,
            ft->result.silence, ft->result.drift_rate,
            ft->result.drift_direction);
}

/* ============================================================
 * meta_reset — the "death" of the observer
 *
 * Zero KV cache and activations. Memory stays allocated
 * for the next birth cycle. Thermogram invalidated.
 * ============================================================ */

void meta_reset(FluidTransformer* ft) {
    if (!ft->initialized) return;

    Config* c = &ft->observer.config;
    RunState* s = &ft->observer.state;
    int kv_dim = c->n_kv_heads * c->head_dim;

    /* Zero KV cache (the observer's "memory" dies) */
    memset(s->key_cache, 0,
           (size_t)c->n_layers * c->max_seq_len * kv_dim * sizeof(float));
    memset(s->value_cache, 0,
           (size_t)c->n_layers * c->max_seq_len * kv_dim * sizeof(float));

    /* Zero activation buffers */
    memset(s->x, 0, c->dim * sizeof(float));
    memset(s->xb, 0, c->dim * sizeof(float));
    memset(s->xb2, 0, c->dim * sizeof(float));
    memset(s->logits, 0, c->vocab_size * sizeof(float));

    /* Thermogram no longer valid */
    ft->result.valid = 0;
}

/* ============================================================
 * meta_apply_thermogram — additive feedback to main Arianna
 *
 * Modulates Arianna's logit distribution based on observation.
 * Bias magnitude is intentionally small (<=0.3) to preserve
 * Arianna's own voice — MetaArianna whispers, not shouts.
 * ============================================================ */

void meta_apply_thermogram(const MetaThermogram* thermo,
                           float* logits, int vocab_size)
{
    if (!thermo->valid) return;

    /*
     * Sharpness modulation: scale logits to sharpen/soften distribution.
     *   sharp (>0.7) → scale up slightly → more peaked
     *   viscous (<0.3) → scale down slightly → more uniform
     */
    float sharp_scale = 0.95f + thermo->sharpness * 0.1f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= sharp_scale;
    }

    /*
     * Warmth modulation: uniform additive bias.
     *   warm (>0.5) → positive bias → slightly more exploratory
     *   cold (<0.5) → negative bias → slightly more conservative
     * This shifts the distribution center, softmax handles the rest.
     */
    float warmth_bias = (thermo->warmth - 0.5f) * 0.3f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] += warmth_bias;
    }

    /*
     * Note: silence boost and drift modulation are applied
     * at the integration layer (arianna_dynamic.c) where we have
     * access to the main tokenizer's char mapping and temperature.
     */
}

/* ============================================================
 * meta_default_params — default configurations for 4 templates
 *
 * Each template is a different "lens" for the observer:
 *   THERMOGRAPH: steady, sees temperature (V-focused)
 *   SILENCE: cool, finds pauses (Q-focused, early layers)
 *   DRIFT: warm, tracks change (K-focused, middle layers)
 *   FIELD: hot, integrates everything (all, late layers)
 * ============================================================ */

void meta_default_params(MetaTemplateParams* params, int template_type) {
    memset(params, 0, sizeof(MetaTemplateParams));
    params->template_type = template_type;

    /* Default: all heads unbiased, all layers fully focused */
    for (int i = 0; i < 8; i++) {
        params->attention_biases[i] = 0.0f;
        params->layer_focus[i] = 1.0f;
    }

    switch (template_type) {
    case META_TEMPLATE_THERMOGRAPH:
        /* Steady observer. Sees what IS. */
        params->temperature = 0.5f;
        params->delta_target = 2;  /* V — what is observed */
        break;

    case META_TEMPLATE_SILENCE:
        /* Cool and still. Finds the gaps. */
        params->temperature = 0.3f;
        params->delta_target = 0;  /* Q — what is asked */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i < 4) ? 1.0f : 0.3f;
        }
        break;

    case META_TEMPLATE_DRIFT:
        /* Warm and tracking. Follows the current. */
        params->temperature = 0.7f;
        params->delta_target = 1;  /* K — what is matched */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i >= 3 && i <= 5) ? 1.0f : 0.4f;
        }
        break;

    case META_TEMPLATE_FIELD:
        /* Hot and wide. Sees the whole field. */
        params->temperature = 0.9f;
        params->delta_target = 3;  /* all Q/K/V */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i >= 5) ? 1.0f : 0.4f;
        }
        break;

    default:
        params->temperature = 0.5f;
        params->delta_target = 2;
        break;
    }
}
