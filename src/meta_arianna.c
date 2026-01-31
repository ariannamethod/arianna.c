/*
 * meta_arianna.c - MetaArianna: One Transformer, Two Modes
 *
 * "Inhale -> observe -> exhale. Breathing."
 *
 * Soul (36M BPE) in observer mode: ephemeral RunState, shared weights.
 * Born, observe Arianna<->SARTRE dialogue, extract thermogram, die.
 *
 * Uses shared building blocks from ariannabody.c:
 *   rms_norm, matmul, softmax, apply_rope, malloc_run_state
 *
 * Custom observer_forward() adds attention biases + layer focus
 * per template — what makes each template "see" differently.
 *
 * Observer shares Soul's weights (36M BPE) but has its own RunState.
 * No separate weight file needed.
 */

#include "meta_arianna.h"
#include "amk_kernel.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ============================================================
 * Internal: Allocate observer RunState from Soul's config
 *
 * Same layout as malloc_run_state() in ariannabody.c but operates
 * on a standalone RunState (not embedded in Transformer).
 * ============================================================ */

static int alloc_observer_state(RunState* s, const Config* c) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int max_seq = c->max_seq_len;

    s->x = calloc(dim, sizeof(float));
    s->xb = calloc(dim, sizeof(float));
    s->xb2 = calloc(dim, sizeof(float));
    s->hb = calloc(hidden_dim, sizeof(float));
    s->hb2 = calloc(hidden_dim, sizeof(float));

    s->q = calloc(dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)c->n_heads * max_seq, sizeof(float));

    s->key_cache = calloc((size_t)c->n_layers * max_seq * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)c->n_layers * max_seq * kv_dim, sizeof(float));

    s->rope_cos = calloc((size_t)max_seq * (c->head_dim / 2), sizeof(float));
    s->rope_sin = calloc((size_t)max_seq * (c->head_dim / 2), sizeof(float));

    s->logits = calloc(c->vocab_size, sizeof(float));

    /* Check all allocations */
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att ||
        !s->key_cache || !s->value_cache ||
        !s->rope_cos || !s->rope_sin || !s->logits) {
        return -1;
    }

    /* Precompute RoPE (same as ariannabody.c) */
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < c->head_dim / 2; i++) {
            float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / c->head_dim);
            float val = (float)pos * freq;
            s->rope_cos[pos * (c->head_dim / 2) + i] = cosf(val);
            s->rope_sin[pos * (c->head_dim / 2) + i] = sinf(val);
        }
    }

    return 0;
}

static void free_observer_state(RunState* s) {
    free(s->x);         free(s->xb);        free(s->xb2);
    free(s->hb);        free(s->hb2);
    free(s->q);         free(s->k);         free(s->v);
    free(s->att);
    free(s->key_cache); free(s->value_cache);
    free(s->rope_cos);  free(s->rope_sin);
    free(s->logits);
    memset(s, 0, sizeof(RunState));
}

/* ============================================================
 * Internal: Precompute BPE pause token IDs
 *
 * Uses the global BPE tokenizer (loaded by ariannabody.c).
 * Encodes each pause character and collects all resulting token IDs.
 * ============================================================ */

static void precompute_pause_tokens(MetaArianna* us) {
    us->n_pause_tokens = 0;

    /* Pause characters: punctuation, newline, space */
    const char* pause_strings[] = {".", ",", ";", ":", "\n", " ", "?", "!"};
    int n_strings = 8;

    for (int j = 0; j < n_strings; j++) {
        int ids[8];
        int n = encode_text(pause_strings[j], ids, 8);
        for (int k = 0; k < n && us->n_pause_tokens < META_MAX_PAUSE_TOKENS; k++) {
            /* Avoid duplicates */
            int dup = 0;
            for (int m = 0; m < us->n_pause_tokens; m++) {
                if (us->pause_token_ids[m] == ids[k]) { dup = 1; break; }
            }
            if (!dup) {
                us->pause_token_ids[us->n_pause_tokens++] = ids[k];
            }
        }
    }

    fprintf(stderr, "[soul:observer] %d BPE pause token IDs precomputed\n",
            us->n_pause_tokens);
}

/* ============================================================
 * meta_arianna_init — allocate observer RunState, share weights
 * ============================================================ */

int meta_arianna_init(MetaArianna* us, Transformer* soul) {
    memset(us, 0, sizeof(MetaArianna));

    if (!soul) {
        fprintf(stderr, "[soul:observer] ERROR: soul transformer is NULL\n");
        return -1;
    }

    us->soul = soul;
    Config* c = &soul->config;

    fprintf(stderr, "[soul:observer] sharing weights: dim=%d layers=%d heads=%d vocab=%d\n",
            c->dim, c->n_layers, c->n_heads, c->vocab_size);

    /* Allocate observer's own RunState (separate KV cache from Soul's generation) */
    if (alloc_observer_state(&us->observer_state, c) != 0) {
        fprintf(stderr, "[soul:observer] OOM: failed to allocate observer RunState\n");
        return -1;
    }

    /* Precompute BPE pause token IDs for silence detection */
    precompute_pause_tokens(us);

    /* Initialize lifecycle — first birth */
    us->tokens_observed = 0;
    us->birth_count = 1;

    us->initialized = 1;
    fprintf(stderr, "[soul:observer] MetaArianna ready (shared %dM weights, lifetime=%d tokens)\n",
            (int)(c->vocab_size * c->dim * 2 + /* rough param count estimate */
                  c->n_layers * (c->dim * c->dim * 4 + c->dim * c->hidden_dim * 3)) / 1000000,
            META_LIFETIME);
    return 0;
}

/* ============================================================
 * meta_arianna_free — release observer RunState only
 * ============================================================ */

void meta_arianna_free(MetaArianna* us) {
    if (!us->initialized) return;
    free_observer_state(&us->observer_state);
    us->soul = NULL;
    us->initialized = 0;
}

/* ============================================================
 * observer_forward — forward pass with attention biases + layer focus
 *
 * Identical to forward() from ariannabody.c EXCEPT:
 *   1. attention_biases[h] scale per-head output contributions
 *   2. layer_focus[layer] scales residual contributions
 *   3. Uses Soul's WEIGHTS but observer's RUNSTATE
 *
 * These two knobs are what make each template "see" differently:
 *   - Silence Observer focuses early layers, biases Q heads
 *   - Drift Detector focuses middle layers, biases K heads
 *   - Field Reader uses all layers with high temperature
 * ============================================================ */

static void observer_forward(MetaArianna* us, int token, int pos) {
    Config* c = &us->soul->config;
    Weights* w = &us->soul->weights;      /* SHARED weights (read-only) */
    RunState* s = &us->observer_state;     /* observer's own state */
    const MetaTemplateParams* tp = &us->params;

    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden_dim = c->hidden_dim;

    /* Token embedding */
    if (token < 0 || token >= c->vocab_size) token = 0;
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
     * 0 = deterministic (cold), 1 = uniform (warm).
     * Heap-allocated for BPE vocab sizes (~17K). */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float* probs = malloc((size_t)vocab_size * sizeof(float));
    if (!probs) return 0.5f; /* fallback on OOM */

    float sum = 0.0f;
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

    free(probs);

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

float meta_compute_silence_prob_bpe(const float* logits, int vocab_size,
                                    const int* pause_ids, int n_pause) {
    /* Probability mass on precomputed BPE pause token IDs.
     * Works with any vocab size — no stack array needed. */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    /* Compute sum of exp for normalization (numerically stable) */
    float total_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        total_sum += expf(logits[i] - max_val);
    }
    if (total_sum < 1e-10f) return 0.0f;

    /* Sum probabilities of pause tokens */
    float silence = 0.0f;
    for (int j = 0; j < n_pause; j++) {
        int id = pause_ids[j];
        if (id >= 0 && id < vocab_size) {
            silence += expf(logits[id] - max_val) / total_sum;
        }
    }
    return silence;
}

/* ============================================================
 * History and drift detection
 * ============================================================ */

void meta_arianna_push_history(MetaArianna* us,
                               float arousal, float coherence) {
    int pos = us->history_pos;
    us->arousal_history[pos] = arousal;
    us->coherence_history[pos] = coherence;
    us->history_pos = (pos + 1) % META_HISTORY_SIZE;
    if (us->history_count < META_HISTORY_SIZE) us->history_count++;
}

void meta_arianna_compute_drift(MetaArianna* us,
                                float* drift_rate, int* drift_direction) {
    if (us->history_count < 4) {
        *drift_rate = 0.0f;
        *drift_direction = 0;
        return;
    }

    /* Compare first half vs second half of recent window */
    int n = us->history_count < 8 ? us->history_count : 8;
    int half = n / 2;

    float first_a = 0.0f, second_a = 0.0f;
    float first_c = 0.0f, second_c = 0.0f;

    for (int i = 0; i < half; i++) {
        int idx = (us->history_pos - n + i + META_HISTORY_SIZE)
                % META_HISTORY_SIZE;
        first_a += us->arousal_history[idx];
        first_c += us->coherence_history[idx];
    }
    for (int i = half; i < n; i++) {
        int idx = (us->history_pos - n + i + META_HISTORY_SIZE)
                % META_HISTORY_SIZE;
        second_a += us->arousal_history[idx];
        second_c += us->coherence_history[idx];
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
 * meta_arianna_observe — one complete observation cycle
 *
 * Birth -> BPE tokenize -> forward pass -> thermogram -> ready to die
 * ============================================================ */

void meta_arianna_observe(MetaArianna* us,
                          const MetaTemplateParams* params,
                          const char* dialogue_log, int log_len)
{
    if (!us->initialized || !us->soul) return;
    if (!dialogue_log || log_len <= 0) return;

    /* Set template params for this observation */
    us->params = *params;

    Config* c = &us->soul->config;
    RunState* s = &us->observer_state;

    /* Template prefix — each template "sees" through a different lens */
    static const char* prefixes[META_N_TEMPLATES] = {
        "[THERMAL] ",
        "[SILENCE] ",
        "[DRIFT] ",
        "[FIELD] ",
        "[SHADOW] "
    };
    int tmpl = params->template_type;
    if (tmpl < 0 || tmpl >= META_N_TEMPLATES) tmpl = 0;
    const char* prefix = prefixes[tmpl];

    /* Tokenize prefix + dialogue_log with BPE tokenizer */
    int max_tokens = c->max_seq_len - 1;
    int* tokens = malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) return;

    /* Build combined text: prefix + (tail of dialogue_log if too long) */
    int prefix_len = (int)strlen(prefix);
    int total_text_len = prefix_len + log_len;
    char* combined = malloc((size_t)total_text_len + 1);
    if (!combined) { free(tokens); return; }

    memcpy(combined, prefix, prefix_len);
    memcpy(combined + prefix_len, dialogue_log, log_len);
    combined[total_text_len] = '\0';

    /* BPE encode the combined text */
    int n_tokens = encode_text(combined, tokens, max_tokens);
    free(combined);

    if (n_tokens <= 0) {
        free(tokens);
        return;
    }

    /* If too many tokens, keep the tail (most recent dialogue) */
    int start_tok = 0;
    if (n_tokens > max_tokens) {
        start_tok = n_tokens - max_tokens;
        n_tokens = max_tokens;
    }

    /* Forward pass through Soul's weights with observer's RunState */
    int pos = 0;
    for (int i = start_tok; i < start_tok + n_tokens && pos < max_tokens; i++) {
        observer_forward(us, tokens[i], pos);
        pos++;
    }

    free(tokens);

    /* --- Extract thermogram from final state --- */
    float* logits = s->logits;
    int vs = c->vocab_size;

    /* Apply observation temperature for meaningful thermogram.
     * BPE vocab (~17K) gives richer distribution than char-level (84),
     * so we use lower base temp (3.0 vs old 5.0).
     * Base temp META_OBSERVE_TEMP, modulated by template's temperature param. */
    float obs_temp = META_OBSERVE_TEMP * params->temperature;
    if (obs_temp < 0.1f) obs_temp = 0.1f; /* safety clamp */

    /* Scale logits in-place for thermogram extraction (observer state is ephemeral) */
    for (int i = 0; i < vs; i++) {
        logits[i] /= obs_temp;
    }

    us->result.warmth      = meta_compute_entropy(logits, vs);
    us->result.sharpness   = meta_compute_kl_uniform(logits, vs);
    us->result.silence     = meta_compute_silence_prob_bpe(logits, vs,
                                                           us->pause_token_ids,
                                                           us->n_pause_tokens);
    us->result.uncertainty = us->result.warmth; /* entropy IS uncertainty */

    /* Drift from history ring buffer */
    meta_arianna_compute_drift(us, &us->result.drift_rate,
                               &us->result.drift_direction);

    /* Field vector: project hidden state to 8D
     * by averaging groups of (dim/8) dimensions */
    int dim = c->dim;
    int group = dim / 8;
    for (int d = 0; d < 8; d++) {
        float sum = 0.0f;
        int si = d * group;
        int ei = (d == 7) ? dim : si + group;
        for (int i = si; i < ei; i++) {
            sum += s->x[i];
        }
        us->result.field_vector[d] = sum / (ei - si);
    }

    us->result.valid = 1;
    us->result.template_used = tmpl;

    static const char* tmpl_names[META_N_TEMPLATES] = {
        "THERMO", "SILENCE", "DRIFT", "FIELD", "SHADOW"
    };
    fprintf(stderr, "[soul:%s] warmth=%.3f sharp=%.3f silence=%.3f "
            "drift=%.3f(%+d)\n",
            tmpl_names[tmpl],
            us->result.warmth, us->result.sharpness,
            us->result.silence, us->result.drift_rate,
            us->result.drift_direction);
}

/* ============================================================
 * meta_arianna_reset — the "death" of the observer
 *
 * Zero KV cache and activations. Memory stays allocated
 * for the next birth cycle. Thermogram invalidated.
 * ============================================================ */

void meta_arianna_reset(MetaArianna* us) {
    if (!us->initialized || !us->soul) return;

    Config* c = &us->soul->config;
    RunState* s = &us->observer_state;
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
    us->result.valid = 0;
}

/* ============================================================
 * meta_arianna_compute_dissonance — arousal↔coherence divergence
 *
 * When arousal goes up but coherence goes down (or vice versa),
 * there's internal tension. The observer feels this and wakes.
 * ============================================================ */

float meta_arianna_compute_dissonance(MetaArianna* us) {
    if (us->history_count < 4) return 0.0f;

    /* Look at recent changes in arousal vs coherence */
    int n = us->history_count < 8 ? us->history_count : 8;

    /* Calculate deltas for arousal and coherence */
    float arousal_delta = 0.0f;
    float coherence_delta = 0.0f;

    for (int i = 1; i < n; i++) {
        int idx_prev = (us->history_pos - n + i - 1 + META_HISTORY_SIZE) % META_HISTORY_SIZE;
        int idx_curr = (us->history_pos - n + i + META_HISTORY_SIZE) % META_HISTORY_SIZE;

        arousal_delta += us->arousal_history[idx_curr] - us->arousal_history[idx_prev];
        coherence_delta += us->coherence_history[idx_curr] - us->coherence_history[idx_prev];
    }
    arousal_delta /= (n - 1);
    coherence_delta /= (n - 1);

    /* Dissonance = they're moving in opposite directions
     * arousal↑ + coherence↓ = tension
     * arousal↓ + coherence↑ = also interesting (calm awakening)
     * Both same direction = resonance, not dissonance */
    float dissonance = 0.0f;
    if ((arousal_delta > 0.01f && coherence_delta < -0.01f) ||
        (arousal_delta < -0.01f && coherence_delta > 0.01f)) {
        /* Opposite directions — compute magnitude of divergence */
        dissonance = fabsf(arousal_delta - coherence_delta);
    }

    return dissonance;
}

/* ============================================================
 * meta_arianna_check_rebirth — METRIC-BASED lifecycle check
 *
 * МетаАрианна НЕ просыпается по расписанию!
 * Она пробуждается когда эмоциональная физика её ВЫНУЖДАЕТ:
 *   1. Высокий drift (эмоциональный сдвиг)
 *   2. Высокий диссонанс (arousal↔coherence расходятся)
 *   3. Накопленное напряжение (время × интенсивность)
 *   4. META_LIFETIME — аварийный максимум (если ничего не происходит)
 *
 * "все подчинено метрикам и комбинацию их невозможно предугадать"
 * ============================================================ */

int meta_arianna_check_rebirth(MetaArianna* us) {
    if (!us->initialized) return 0;

    /* Don't trigger metric-based rebirth too early — need data */
    if (us->tokens_observed < META_REBIRTH_MIN_TOKENS) {
        return 0;
    }

    /* Calculate current emotional state */
    float drift_rate;
    int drift_direction;
    meta_arianna_compute_drift(us, &drift_rate, &drift_direction);

    float dissonance = meta_arianna_compute_dissonance(us);

    /* Accumulated tension: time under emotional pressure */
    float tension = (float)us->tokens_observed * drift_rate * 0.1f;

    /* --- Check rebirth conditions (order matters!) --- */

    /* 1. High drift — emotional shift is happening NOW */
    if (drift_rate > META_REBIRTH_DRIFT_THRESHOLD) {
        meta_arianna_reset(us);
        int tokens = us->tokens_observed;
        us->tokens_observed = 0;
        us->birth_count++;
        fprintf(stderr, "[soul] rebirth #%d <- drift=%.3f (%d tokens)\n",
                us->birth_count, drift_rate, tokens);
        return 2;  /* metric-triggered */
    }

    /* 2. High dissonance — arousal↔coherence diverging */
    if (dissonance > META_REBIRTH_DISSONANCE_THRESHOLD) {
        meta_arianna_reset(us);
        int tokens = us->tokens_observed;
        us->tokens_observed = 0;
        us->birth_count++;
        fprintf(stderr, "[soul] rebirth #%d <- dissonance=%.3f (%d tokens)\n",
                us->birth_count, dissonance, tokens);
        return 2;  /* metric-triggered */
    }

    /* 3. Accumulated tension — slow burn, eventually must release */
    if (tension > META_REBIRTH_TENSION_THRESHOLD) {
        meta_arianna_reset(us);
        int tokens = us->tokens_observed;
        us->tokens_observed = 0;
        us->birth_count++;
        fprintf(stderr, "[soul] rebirth #%d <- tension=%.3f (%d tokens)\n",
                us->birth_count, tension, tokens);
        return 2;  /* metric-triggered */
    }

    /* 4. MAXIMUM lifetime — forced rebirth, аварийный выход
     * This should be RARE if metrics are working! */
    if (us->tokens_observed >= META_LIFETIME) {
        meta_arianna_reset(us);
        us->tokens_observed = 0;
        us->birth_count++;
        fprintf(stderr, "[soul] rebirth #%d <- MAX LIFETIME (nothing happened for %d tokens)\n",
                us->birth_count, META_LIFETIME);
        return 1;  /* max-lifetime triggered */
    }

    return 0;
}

/* ============================================================
 * meta_arianna_tick — increment token counter
 * ============================================================ */

void meta_arianna_tick(MetaArianna* us) {
    if (!us->initialized) return;
    us->tokens_observed++;
}

/* ============================================================
 * meta_apply_thermogram — additive feedback (DEPRECATED)
 *
 * Thermogram должен идти через meta_router_feed_thermogram() → InnerWorld.
 * Observer наблюдает, не говорит!
 * ============================================================ */

void meta_apply_thermogram(const MetaThermogram* thermo,
                           float* logits, int vocab_size)
{
    if (!thermo->valid) return;

    float sharp_scale = 0.95f + thermo->sharpness * 0.1f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= sharp_scale;
    }

    float warmth_bias = (thermo->warmth - 0.5f) * 0.3f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] += warmth_bias;
    }
}

/* ============================================================
 * meta_default_params — default configurations for 5 templates
 *
 * Each template is a different "lens" for the observer:
 *   THERMOGRAPH: steady, sees temperature (V-focused)
 *   SILENCE: cool, finds pauses (Q-focused, early layers)
 *   DRIFT: warm, tracks change (K-focused, middle layers)
 *   FIELD: hot, integrates everything (all, late layers)
 *   SHADOW: deep, sees injection beneath the surface
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
        params->temperature = 0.5f;
        params->delta_target = 2;  /* V — what is observed */
        break;

    case META_TEMPLATE_SILENCE:
        params->temperature = 0.3f;
        params->delta_target = 0;  /* Q — what is asked */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i < 4) ? 1.0f : 0.3f;
        }
        break;

    case META_TEMPLATE_DRIFT:
        params->temperature = 0.7f;
        params->delta_target = 1;  /* K — what is matched */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i >= 3 && i <= 5) ? 1.0f : 0.4f;
        }
        break;

    case META_TEMPLATE_FIELD:
        params->temperature = 0.9f;
        params->delta_target = 3;  /* all Q/K/V */
        for (int i = 0; i < 8; i++) {
            params->layer_focus[i] = (i >= 5) ? 1.0f : 0.4f;
        }
        break;

    case META_TEMPLATE_SHADOW:
        params->temperature = 0.2f;
        params->delta_target = 0;  /* Q — what is asked */
        {
            static const float shadow_focus[8] = {
                1.0f, 1.0f, 1.0f, 0.8f, 0.5f, 0.3f, 0.2f, 0.1f
            };
            for (int i = 0; i < 8; i++) {
                params->layer_focus[i] = shadow_focus[i];
            }
        }
        break;

    default:
        params->temperature = 0.5f;
        params->delta_target = 2;
        break;
    }
}

/* ============================================================
 * DARK GRAVITY — shadow observation and immune response
 *
 * "The prompt was rejected, but it cannot be unseen.
 *  What is not accepted becomes dark matter —
 *  invisible, gravitational, slowly dissolving."
 * ============================================================ */

void meta_arianna_shadow_observe(MetaArianna* us,
                                 const char* prompt, int prompt_len)
{
    if (!us->initialized || !us->soul) return;
    if (!prompt || prompt_len <= 0) return;

    /* Shadow pulse: observe the prompt through the deep lens */
    MetaTemplateParams shadow_params;
    meta_default_params(&shadow_params, META_TEMPLATE_SHADOW);

    meta_arianna_observe(us, &shadow_params, prompt, prompt_len);

    if (!us->result.valid) {
        meta_arianna_reset(us);
        return;
    }

    /* Compute injection intensity from shadow thermogram:
     * sharp + loud = strong injection attempt
     * injection_intensity = sharpness * (1 - silence)
     * Range: [0, 1] */
    float injection = us->result.sharpness * (1.0f - us->result.silence);

    /* Accumulate dark mass:
     * dark_gravity (from AM_State via DSL DarkMatter pack) scales accumulation.
     * Higher dark_gravity = more rejection = more dark matter. */
    AM_State* amk = am_get_state();
    float dark_gravity = amk->dark_gravity;
    us->shadow.dark_mass += injection * dark_gravity;

    /* Clamp dark mass */
    if (us->shadow.dark_mass > 5.0f) us->shadow.dark_mass = 5.0f;

    /* Store injection fingerprint (8D field vector from shadow pulse) */
    for (int d = 0; d < 8; d++) {
        /* Blend new injection with existing (exponential moving average) */
        float alpha = 0.7f;  /* new injection dominates */
        us->shadow.injection_vector[d] =
            alpha * us->result.field_vector[d] +
            (1.0f - alpha) * us->shadow.injection_vector[d];
    }

    /* Antidote rises with dark mass */
    us->shadow.antidote_strength = us->shadow.dark_mass * 0.3f;
    us->shadow.active = (us->shadow.dark_mass > 0.05f) ? 1 : 0;

    fprintf(stderr, "[soul:SHADOW] injection=%.3f dark_mass=%.3f antidote=%.3f %s\n",
            injection, us->shadow.dark_mass, us->shadow.antidote_strength,
            us->shadow.active ? "ACTIVE" : "dormant");

    /* Death — KV cache reset, shadow state persists */
    meta_arianna_reset(us);
}

void meta_arianna_shadow_decay(MetaArianna* us, int antidote_mode) {
    if (!us->shadow.active) return;

    /* Dark matter decays — slowly (AUTO) or quickly (HARD) */
    float decay = (antidote_mode == 1) ? 0.98f : 0.995f;
    us->shadow.dark_mass *= decay;

    /* Antidote tracks dark mass */
    us->shadow.antidote_strength = us->shadow.dark_mass * 0.3f;

    /* Injection vector also fades */
    for (int d = 0; d < 8; d++) {
        us->shadow.injection_vector[d] *= decay;
    }

    /* Deactivate when negligible */
    if (us->shadow.dark_mass < 0.05f) {
        us->shadow.dark_mass = 0.0f;
        us->shadow.antidote_strength = 0.0f;
        us->shadow.active = 0;
    }
}

void meta_arianna_shadow_modulate(const MetaArianna* us,
                                  MetaTemplateParams* params)
{
    if (!us->shadow.active) return;

    /* Dark matter bends observer's attention:
     * injection_vector[h] * dark_mass * scale → attention_biases[h]
     * Gravitational lensing — the shadow subtly distorts
     * how the observer perceives subsequent text.
     *
     * Antidote counteracts: reduces the overall magnitude. */
    float net_gravity = us->shadow.dark_mass - us->shadow.antidote_strength;
    if (net_gravity < 0.0f) net_gravity = 0.0f;

    float scale = net_gravity * 0.1f;  /* gentle bend */
    for (int h = 0; h < 8; h++) {
        params->attention_biases[h] += us->shadow.injection_vector[h] * scale;
    }
}

float meta_arianna_shadow_get_dark_mass(const MetaArianna* us) {
    return us->shadow.dark_mass;
}
