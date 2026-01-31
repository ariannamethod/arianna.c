/*
 * meta_arianna.h - MetaArianna: One Transformer, Two Modes
 *
 * "Inhale -> observe -> exhale. Breathing."
 *
 * Soul (36M BPE) serves dual purpose:
 *   1. Soul mode: personality generation with persistent KV cache
 *   2. Observer mode: ephemeral observation with templates + attention biases
 *
 * Observer shares Soul's weights (read-only) but has its own RunState.
 * Router (Go, permanent) selects template based on InnerWorld metrics.
 *   -> Observer born (Soul's weights, ephemeral RunState)
 *   -> Reads Arianna<->SARTRE dialogue logs
 *   -> Extracts thermogram (warmth/sharpness/silence/drift)
 *   -> Observer dies (RunState reset, weights untouched)
 *
 * Config: dim=448, layers=8, heads=8, kv_heads=8,
 *         hidden=1280, vocab=~17K (BPE), head_dim=56
 *
 * Weights: shared with Soul (weights/arianna_36m_bpe.bin)
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
#define META_TEMPLATE_SHADOW       4  /* Dark gravity: prompt injection trace */
#define META_N_TEMPLATES           5

/* Max dialogue log length for observation */
#define META_MAX_LOG_LEN        256
/* History buffer for drift detection */
#define META_HISTORY_SIZE        32
/* Observer max generation tokens */
#define META_MAX_OBSERVE_TOKENS  64
/* Observer LIFETIME: MAXIMUM tokens before forced rebirth.
 * But MetaArianna should wake up EARLIER based on emotional metrics!
 * "вздох" — one breath cycle. Born → observe → create thermogram → die → rebirth
 * She awakens when the emotional physics DEMANDS it, not on a mechanical timer. */
#define META_LIFETIME            60

/* Rebirth thresholds — emotional physics triggers (НЕ механический таймер!) */
#define META_REBIRTH_DRIFT_THRESHOLD      0.15f  /* High drift = emotional shift happening */
#define META_REBIRTH_DISSONANCE_THRESHOLD 0.25f  /* Arousal↔coherence divergence */
#define META_REBIRTH_TENSION_THRESHOLD    3.0f   /* Accumulated: tokens × drift × 0.1 */
#define META_REBIRTH_MIN_TOKENS           8      /* Minimum before metric-based rebirth allowed */
/* Observation temperature: higher = observer "squints" to see pattern shapes,
 * not raw peaks. With BPE vocab (~17K) distribution is richer than char-level,
 * but we still need temperature scaling for meaningful thermogram extraction. */
#define META_OBSERVE_TEMP        3.0f
/* Max pause tokens we track for silence detection (BPE can have many) */
#define META_MAX_PAUSE_TOKENS    64

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
    float temperature;             /* observation temperature multiplier (applied to META_OBSERVE_TEMP) */
    int   delta_target;            /* 0=Q, 1=K, 2=V, 3=all — reserved for future delta routing */
} MetaTemplateParams;

/* ============================================================
 * MetaShadowState — dark gravity persistence
 *
 * Prompt injection that was not accepted leaves a trace:
 * dark matter. Invisible but gravitational — it bends how
 * the observer perceives subsequent generation.
 *
 * "The prompt was rejected, but it cannot be unseen."
 * ============================================================ */

typedef struct {
    float dark_mass;              /* accumulated dark matter (>=0, slow decay) */
    float injection_vector[8];    /* 8D fingerprint of rejected prompt */
    float antidote_strength;      /* immune response (grows with dark_mass) */
    int   active;                 /* nonzero if dark matter present */
} MetaShadowState;

/* ============================================================
 * MetaArianna — one transformer, two modes
 *
 * Soul mode: generation with persistent KV cache (non-D12)
 * Observer mode: ephemeral observation with templates (D12)
 *
 * Weights are SHARED (read-only in forward pass).
 * Each mode has its own RunState (KV cache, activations).
 * ============================================================ */

typedef struct {
    /* Soul's transformer — NOT owned, points to main 't' from arianna_dynamic.c
     * Weights shared between Soul generation and observer mode. */
    Transformer* soul;

    /* Observer's ephemeral RunState (separate KV cache, activations).
     * Allocated once at init, zeroed on each rebirth cycle.
     * Uses soul->config for dimensions. */
    RunState observer_state;

    /* Template params (set per observation) */
    MetaTemplateParams params;

    /* Thermogram output (read after observation) */
    MetaThermogram result;

    /* BPE pause token IDs for silence detection.
     * Precomputed at init: encode('.'), encode(','), encode('\n'), etc.
     * BPE may split these differently — we collect ALL relevant IDs. */
    int pause_token_ids[META_MAX_PAUSE_TOKENS];
    int n_pause_tokens;

    /* History for drift detection */
    float arousal_history[META_HISTORY_SIZE];
    float coherence_history[META_HISTORY_SIZE];
    int   history_pos;
    int   history_count;

    /* Dark gravity — prompt rejection shadow */
    MetaShadowState shadow;

    /* Lifecycle: "вздох" — breath cycle */
    int   tokens_observed;        /* tokens since last rebirth */
    int   birth_count;            /* how many times reborn */

    /* State */
    int   initialized;
} MetaArianna;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/* Initialize observer: allocate RunState, precompute pause token IDs.
 * soul: pointer to the loaded Soul transformer (weights + config).
 * No separate weights file needed — observer shares Soul's weights.
 * Returns 0 on success, -1 on error. */
int  meta_arianna_init(MetaArianna* us, Transformer* soul);

/* Free observer RunState (does NOT free soul's weights/state) */
void meta_arianna_free(MetaArianna* us);

/* ============================================================
 * Observation (birth -> observe -> death)
 * ============================================================ */

/* Run one observation cycle:
 * 1. Set template params
 * 2. Tokenize dialogue_log with BPE (shared tokenizer)
 * 3. Forward pass through Soul's weights + observer RunState
 * 4. Extract thermogram from logits/hidden states
 * 5. Store result in us->result
 */
void meta_arianna_observe(MetaArianna* us,
                          const MetaTemplateParams* params,
                          const char* dialogue_log, int log_len);

/* Reset observer RunState (the "death" — zero KV cache etc.)
 * Does NOT free memory — buffers are reused next cycle. */
void meta_arianna_reset(MetaArianna* us);

/* Check if observer should be reborn based on EMOTIONAL METRICS.
 * Rebirth triggers (in priority order):
 *   1. High drift_rate (emotional shift) > META_REBIRTH_DRIFT_THRESHOLD
 *   2. High dissonance (arousal↔coherence diverging) > META_REBIRTH_DISSONANCE_THRESHOLD
 *   3. Accumulated tension (tokens × drift) > META_REBIRTH_TENSION_THRESHOLD
 *   4. MAXIMUM lifetime (META_LIFETIME) — forced, аварийный выход
 * МетаАрианна просыпается не по расписанию, а когда физика ВЫНУЖДАЕТ.
 * Returns 1 if rebirth occurred, 0 otherwise.
 * Returns 2 if metric-triggered (vs 1 for max-lifetime-triggered). */
int meta_arianna_check_rebirth(MetaArianna* us);

/* Compute current arousal↔coherence dissonance.
 * High = they're moving in opposite directions = tension.
 * Returns value in [0, 1]. */
float meta_arianna_compute_dissonance(MetaArianna* us);

/* Increment tokens_observed counter. Call after each token generated. */
void meta_arianna_tick(MetaArianna* us);

/* ============================================================
 * Thermogram Feedback — через Vagus/InnerWorld ТОЛЬКО
 * ============================================================ */

/* NOTE: meta_apply_thermogram() is DEPRECATED for direct logit use.
 * Thermogram должен идти через meta_router_feed_thermogram() → InnerWorld
 * Observer наблюдает, не говорит! */
void meta_apply_thermogram(const MetaThermogram* thermo,
                           float* logits, int vocab_size);

/* ============================================================
 * History (for drift detection)
 * ============================================================ */

/* Push current arousal/coherence into history ring buffer */
void meta_arianna_push_history(MetaArianna* us,
                               float arousal, float coherence);

/* Compute drift rate and direction from history */
void meta_arianna_compute_drift(MetaArianna* us,
                                float* drift_rate, int* drift_direction);

/* ============================================================
 * Thermogram Extraction Helpers
 * ============================================================ */

/* Entropy of softmax(logits) — high = warm/uncertain */
float meta_compute_entropy(const float* logits, int vocab_size);

/* KL divergence from uniform — high = sharp/focused */
float meta_compute_kl_uniform(const float* logits, int vocab_size);

/* Probability mass on pause tokens using precomputed BPE IDs */
float meta_compute_silence_prob_bpe(const float* logits, int vocab_size,
                                    const int* pause_ids, int n_pause);

/* ============================================================
 * Default template configurations
 * ============================================================ */

/* Fill params with defaults for given template type */
void meta_default_params(MetaTemplateParams* params, int template_type);

/* ============================================================
 * Dark Gravity — shadow observation and modulation
 * ============================================================ */

/* Shadow-observe a prompt: compute injection intensity and dark_mass.
 * Called once per prompt, before generation starts. */
void meta_arianna_shadow_observe(MetaArianna* us,
                                 const char* prompt, int prompt_len);

/* Decay dark matter (call each pulse, ~every 16 tokens).
 * antidote_mode: 0=AUTO (slow decay), 1=HARD (fast decay) */
void meta_arianna_shadow_decay(MetaArianna* us, int antidote_mode);

/* Modulate observation params by dark gravity.
 * Bends attention_biases by injection_vector * dark_mass.
 * Call before meta_arianna_observe() on each 16-token pulse. */
void meta_arianna_shadow_modulate(const MetaArianna* us,
                                  MetaTemplateParams* params);

/* Get current dark_mass (for penetration modulation) */
float meta_arianna_shadow_get_dark_mass(const MetaArianna* us);

#endif /* META_ARIANNA_H */
