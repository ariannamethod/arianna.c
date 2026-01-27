/*
 * meta_arianna.h - MetaArianna: Pulsating Meta-Observer
 *
 * "Inhale -> observe -> exhale. Breathing."
 *
 * FluidTransformer architecture:
 *   Router (Go, permanent) selects template based on metrics
 *   -> Observer born (20M weights, C forward pass)
 *   -> Reads Arianna<->SARTRE dialogue logs
 *   -> Extracts thermogram (warmth/sharpness/silence/drift)
 *   -> Observer dies (RunState reset, weights shared)
 *
 * 20M Config: dim=448, layers=8, heads=8, kv_heads=8,
 *             hidden=1280, vocab=84, head_dim=56
 *
 * Weights: weights/arianna_20m.bin (77.3 MB, float32, 20000 iter H-100)
 */

#ifndef META_ARIANNA_H
#define META_ARIANNA_H

#include "arianna.h"

/* ============================================================
 * Template Types
 * ============================================================ */

#define META_TEMPLATE_THERMOGRAPH  0  /* ΔT between Arianna/SARTRE */
#define META_TEMPLATE_SILENCE      1  /* Pauses, entropy spikes */
#define META_TEMPLATE_DRIFT        2  /* ∂/∂t arousal/coherence */
#define META_TEMPLATE_FIELD        3  /* Integral view, pseudo-affective */
#define META_N_TEMPLATES           4

/* Max dialogue log length for observation */
#define META_MAX_LOG_LEN        256
/* History buffer for drift detection */
#define META_HISTORY_SIZE        32
/* Observer max generation tokens */
#define META_MAX_OBSERVE_TOKENS  64
/* Observation temperature: higher = observer "squints" to see pattern shapes,
 * not raw peaks. Needed because char-level model is too peaked on raw logits */
#define META_OBSERVE_TEMP        5.0f

/* ============================================================
 * MetaThermogram — output of observation
 * ============================================================ */

typedef struct {
    float warmth;           /* [0,1] warm/cold */
    float sharpness;        /* [0,1] sharp/viscous (KL from uniform) */
    float silence;          /* [0,1] density of pause tokens */
    float uncertainty;      /* [0,1] entropy of logits */
    float drift_rate;       /* speed of arousal/coherence change */
    int   drift_direction;  /* 1=unfolding, -1=collapsing, 0=stable */
    float field_vector[8];  /* 8D pseudo-affective projection */
    int   valid;            /* 1 if thermogram contains data */
    int   template_used;    /* which template produced this */
} MetaThermogram;

/* ============================================================
 * MetaTemplateParams — filled by router per observation
 * ============================================================ */

typedef struct {
    int   template_type;           /* META_TEMPLATE_* */
    float attention_biases[8];     /* per-head modifiers */
    float layer_focus[8];          /* per-layer strength [0,1] */
    float temperature;             /* observer sampling temperature */
    int   delta_target;            /* 0=Q, 1=K, 2=V, 3=all */
} MetaTemplateParams;

/* ============================================================
 * FluidTransformer — ephemeral observer instance
 *
 * Born: meta_observe() called with template params
 * Lives: forward pass on dialogue log -> thermogram extraction
 * Dies: meta_reset() zeros RunState (no free, reuse buffers)
 * ============================================================ */

typedef struct {
    /* Transformer with 20M weights (self-contained) */
    Transformer observer;

    /* Template params (set per observation) */
    MetaTemplateParams params;

    /* Thermogram output (read after observation) */
    MetaThermogram result;

    /* Observer's own tokenizer (vocab=84) */
    char  obs_vocab[128];         /* id -> char */
    int   obs_char_to_id[256];    /* char -> id */
    int   obs_vocab_size;

    /* History for drift detection */
    float arousal_history[META_HISTORY_SIZE];
    float coherence_history[META_HISTORY_SIZE];
    int   history_pos;
    int   history_count;

    /* State */
    int   initialized;
    int   weights_loaded;
} FluidTransformer;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/* Initialize observer: alloc RunState, load weights + tokenizer.
 * weights_path: path to arianna_20m.bin (float32)
 * tokenizer_path: path to tokenizer_unified.json
 * Returns 0 on success, -1 on error. */
int  meta_init(FluidTransformer* ft,
               const char* weights_path,
               const char* tokenizer_path);

/* Free all resources */
void meta_free(FluidTransformer* ft);

/* ============================================================
 * Observation (birth -> observe -> death)
 * ============================================================ */

/* Run one observation cycle:
 * 1. Set template params
 * 2. Tokenize dialogue_log with observer's tokenizer
 * 3. Forward pass through 20M
 * 4. Extract thermogram from logits/hidden states
 * 5. Store result in ft->result
 */
void meta_observe(FluidTransformer* ft,
                  const MetaTemplateParams* params,
                  const char* dialogue_log, int log_len);

/* Reset observer RunState (the "death" — zero KV cache etc.)
 * Does NOT free memory — buffers are reused next cycle. */
void meta_reset(FluidTransformer* ft);

/* ============================================================
 * Thermogram Feedback (apply to main generation)
 * ============================================================ */

/* Apply thermogram as additive logit bias to main Arianna */
void meta_apply_thermogram(const MetaThermogram* thermo,
                           float* logits, int vocab_size);

/* ============================================================
 * History (for drift detection)
 * ============================================================ */

/* Push current arousal/coherence into history ring buffer */
void meta_push_history(FluidTransformer* ft,
                       float arousal, float coherence);

/* Compute drift rate and direction from history */
void meta_compute_drift(FluidTransformer* ft,
                        float* drift_rate, int* drift_direction);

/* ============================================================
 * Thermogram Extraction Helpers
 * ============================================================ */

/* Entropy of softmax(logits) — high = warm/uncertain */
float meta_compute_entropy(const float* logits, int vocab_size);

/* KL divergence from uniform — high = sharp/focused */
float meta_compute_kl_uniform(const float* logits, int vocab_size);

/* Probability mass on pause tokens (.,;:\n space) */
float meta_compute_silence_prob(const float* logits, int vocab_size,
                                const int* char_to_id);

/* ============================================================
 * Default template configurations
 * ============================================================ */

/* Fill params with defaults for given template type */
void meta_default_params(MetaTemplateParams* params, int template_type);

#endif /* META_ARIANNA_H */
