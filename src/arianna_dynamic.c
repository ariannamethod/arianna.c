/*
 * arianna_dynamic.c - Main entry point with delta support
 *
 * Core personality + dynamic attention deltas
 * "Who I am" grows through experience
 *
 * Stanley integration: mood routing, microtraining, shard persistence
 * Go integration: inner_world (trauma, emotional drift, prophecy debt)
 */

#include "arianna.h"
#include "delta.h"
#include "mood.h"
#include "guided.h"
#include "subjectivity.h"
#include "cooccur.h"
#include "body_sense.h"
#include "selfsense.h"
#include "mathbrain.h"
#include "cloud.h"  // Pre-semantic emotion detection
#include "julia_bridge.h"  // Julia emotional gradient engine
#include "schumann.h"  // Earth-ionosphere resonance (cosmic input)
#include "pandora.h"  // Vocabulary injection from External Brain
#include "../packages/pandora/pandora_bridge.h"  // Python bridge to external brains
#include "inner_arianna.h"  // MetaVoice: борьба between main and inner voice
#include "meta_arianna.h"  // MetaArianna: pulsating meta-observer (FluidTransformer)
#include "sartre_bridge.h" // SARTRE 14.3M bridge for dialogue mode
#include "amk_kernel.h"  // Arianna Method Kernel: prophecy, destiny, suffering, movement
#include "arianna_dsl.h"  // DSL integration for generation
#ifdef USE_LUA
#include "amk_lua.h"      // Lua scripting for hot-reload (silent fallback if unavailable)
#endif
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

// Go inner_world integration (optional, compile with -DUSE_GO_INNER_WORLD)
#ifdef USE_GO_INNER_WORLD
#include "inner_world.h"
static int g_inner_world_enabled = 1;
static int g_inner_world_async = 0;  // If 1, rely on background goroutines only
#else
static int g_inner_world_enabled = 0;
static int g_inner_world_async = 0;
#endif

// Helper: create directory if not exists (handles EEXIST)
static int ensure_dir(const char* path) {
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;  // Already exists, OK
    return -1;  // Actual error
}

// ============================================================
// Global state
// ============================================================

static DeltaBank g_delta_bank;
static Signals g_signals;
static MicroTrainer g_trainer;
static MoodRouter g_mood_router;
static int g_delta_enabled = 0;
static int g_mood_enabled = 0;
static int g_microtraining = 0;
static float g_momentum = 0.8f;  // Mood transition smoothness

// Guided attention (Stanley-style)
static Identity g_identity;
static StanleySignals g_stanley_signals;
static AttentionBias g_attention_bias;
static OverthinkDetector g_overthink;
static int g_guided_enabled = 0;

// Subjectivity (no-seed-from-prompt)
static Subjectivity g_subjectivity;
static int g_subjectivity_enabled = 0;
static char* g_origin_path = NULL;

// Co-occurrence field (attention bias from corpus patterns)
static CooccurField g_cooccur;
static int g_cooccur_enabled = 0;
static float g_cooccur_alpha = 0.15f;  // Blend strength

// Body sense (somatic awareness: boredom, overwhelm, stuck)
static BodySense g_body_sense;
static BodyState g_body_state;
static int g_body_sense_enabled = 0;

// SelfSense (learned signal extraction from hidden states)
static SelfSense g_selfsense;
static int g_selfsense_enabled = 0;

// Inner Arianna (MetaVoice: борьба between main and inner voice)
static InnerArianna g_inner_arianna;
static int g_inner_enabled = 0;

// MathBrain (arithmetic through resonance)
static MathBrain g_mathbrain;
static int g_mathbrain_enabled = 0;
static const char* g_mathbrain_path = "weights/mathbrain.bin";  // Default persistence path

// Julia emotional gradient engine (sensory cortex)
static int g_julia_enabled = 0;
static JuliaEmotionalResult g_julia_state;  // Current emotional state
static float g_julia_emotional_vec[12];  // For ODE stepping

// Schumann resonance (Earth's heartbeat - cosmic input)
static int g_schumann_enabled = 0;

// Pandora (vocabulary injection from External Brain)
static PandoraBox g_pandora;
static int g_pandora_enabled = 0;

// AMK — Arianna Method Kernel (prophecy, destiny, suffering, movement)
static int g_amk_enabled = 0;
static DSL_GenerationConfig g_dsl_config;  // Current DSL config for generation

// MetaArianna — pulsating meta-observer (FluidTransformer)
static FluidTransformer g_fluid_transformer;
static MetaThermogram g_meta_thermogram;
static int g_meta_enabled = 0;
static int g_meta_observation_count = 0;

// SARTRE — 14.3M dialogue partner (lazy loaded)
static SartreTransformer g_sartre;
static int g_sartre_loaded = 0;
static int g_dialogue_max_turns = 5;
static int g_dialogue_tokens_per_turn = 80;

// ============================================================
// Inner Arianna борьба helper
// ============================================================

// Apply борьба between main and inner voice
// Inner voice is modulated by Cloud, Mood, BodySense
// Returns: winner (0=main, 1=inner, -1=blend)
static int apply_borba_to_logits(float* logits, int vocab_size) {
    if (!g_inner_enabled) return 0;

    // Update inner state from emotional systems
    // Cloud chambers
    CloudResponse cloud = cloud_ping("");  // Use last state
    inner_update_cloud(&g_inner_arianna, &cloud);

    // Mood
    inner_update_mood(&g_inner_arianna, &g_mood_router);

    // Body sense (stuck, boredom)
    inner_update_body(&g_inner_arianna, g_body_state.quality < 0.3f ? 0.5f : 0.0f,
                      g_body_state.novelty < 0.3f ? 0.5f : 0.0f);

    // Trauma (from subjectivity if available)
    // TODO: integrate with subjectivity trauma when available

    // Run борьба - inner_borba modifies logits in place
    float* output = (float*)malloc(vocab_size * sizeof(float));
    int winner = inner_borba(&g_inner_arianna, output, logits, vocab_size);
    memcpy(logits, output, vocab_size * sizeof(float));
    free(output);

    return winner;
}

// Active learning shard (for microtraining)
static ExperienceShard* g_active_shard = NULL;

// Quantum accumulator (Stanley-style batched learning)
static ExperienceAccumulator g_accumulator;
static int g_accumulator_initialized = 0;

// Generation state for microtraining
typedef struct {
    float* pre_activations;   // Before attention
    float* post_activations;  // After attention
    int last_layer;
    int sequence_pos;
} TrainingState;

static TrainingState g_train_state;

// ============================================================
// Extended forward pass with delta support (Llama 3 style)
// ============================================================

void forward_dynamic(Transformer* t, int* tokens, int n_tokens, int pos) {
    (void)n_tokens;  // Unused in single-token forward

    Config* c = &t->config;
    Weights* w = &t->weights;
    RunState* s = &t->state;

    int dim = c->dim;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int hidden_dim = c->hidden_dim;
    int max_seq = c->max_seq_len;
    int kv_dim = n_kv_heads * head_dim;
    int n_kv_groups = n_heads / n_kv_heads;

    // Get embedding (Llama style - no position embedding, RoPE instead)
    int token = tokens[pos];
    float* x = s->x;
    float* tok_vec = w->tok_emb + token * dim;
    memcpy(x, tok_vec, dim * sizeof(float));

    // Process through layers
    for (int layer = 0; layer < c->n_layers; layer++) {
        // RMSNorm before attention
        rms_norm(s->xb, x, w->attn_norm + layer * dim, dim, c->norm_eps);

        // === MICROTRAINING HOOK: capture pre-attention state ===
        if (g_microtraining && g_train_state.pre_activations != NULL) {
            memcpy(g_train_state.pre_activations, s->xb, dim * sizeof(float));
            g_train_state.last_layer = layer;
            g_train_state.sequence_pos = pos;
        }

        // QKV projection (separate, Llama style)
        matmul(s->q, s->xb, w->wq + layer * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + layer * dim * kv_dim, dim, kv_dim);

        // === DELTA APPLICATION ===
        // Deltas modify ATTENTION, not weights directly
        // This is where experience shapes perception
        if (g_delta_enabled && g_delta_bank.n_shards > 0) {
            apply_q_delta(&g_delta_bank, s->q, s->xb, layer, dim);
            apply_k_delta(&g_delta_bank, s->k, s->xb, layer, kv_dim);
            apply_v_delta(&g_delta_bank, s->v, s->xb, layer, kv_dim);
        }

        // Apply RoPE
        apply_rope(s->q, s->k, s->rope_cos, s->rope_sin, n_heads, n_kv_heads, head_dim, pos);

        // Store K, V in cache (GQA: smaller kv_dim)
        int kv_cache_offset = layer * max_seq * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + kv_cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + kv_cache_offset, s->v, kv_dim * sizeof(float));

        // Multi-head attention with GQA
        memset(s->xb, 0, dim * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < n_heads; h++) {
            float* qh = s->q + h * head_dim;
            float* atth = s->att + h * max_seq;
            int kv_h = h / n_kv_groups;  // Which KV head this Q head uses

            // Attention scores
            for (int t = 0; t <= pos; t++) {
                float* kh = s->key_cache + layer * max_seq * kv_dim + t * kv_dim + kv_h * head_dim;
                float dot = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    dot += qh[i] * kh[i];
                }
                atth[t] = dot * scale;
            }

            // Softmax
            softmax(atth, pos + 1);

            // Weighted sum of values
            float* xbh = s->xb + h * head_dim;
            for (int t = 0; t <= pos; t++) {
                float* vh = s->value_cache + layer * max_seq * kv_dim + t * kv_dim + kv_h * head_dim;
                float a = atth[t];
                for (int i = 0; i < head_dim; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }

        // === MICROTRAINING HOOK: capture post-attention state ===
        if (g_microtraining && g_train_state.post_activations != NULL) {
            memcpy(g_train_state.post_activations, s->xb, dim * sizeof(float));
        }

        // Output projection + residual
        matmul(s->xb2, s->xb, w->wo + layer * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // RMSNorm before FFN
        rms_norm(s->xb, x, w->ffn_norm + layer * dim, dim, c->norm_eps);

        // SwiGLU FFN
        matmul(s->hb, s->xb, w->w_gate + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w_up + layer * dim * hidden_dim, dim, hidden_dim);

        // SiLU activation and element-wise multiply
        for (int i = 0; i < hidden_dim; i++) {
            float gate = s->hb[i];
            s->hb[i] = (gate / (1.0f + expf(-gate))) * s->hb2[i];
        }

        // Down projection + residual
        matmul(s->xb, s->hb, w->w_down + layer * hidden_dim * dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // Final RMSNorm
    rms_norm(s->x, x, w->final_norm, dim, c->norm_eps);

    // Logits
    matmul(s->logits, s->x, w->lm_head, dim, c->vocab_size);
}

// ============================================================
// Dynamic generation with mood routing + microtraining
// ============================================================

void generate_dynamic(Transformer* t, char* prompt, int max_tokens, float temperature) {
    // Reset KV cache
    

    int tokens[MAX_SEQ_LEN];
    size_t prompt_strlen = strlen(prompt);
    int n_tokens = (prompt_strlen > MAX_SEQ_LEN) ? MAX_SEQ_LEN : (int)prompt_strlen;

    // Tokenize prompt (using vocab mapping, not raw ASCII)
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = char_to_token(prompt[i]);
    }

    // Extract signals and route to moods
    extract_signals(&g_signals, tokens, n_tokens, NULL);

    if (g_mood_enabled) {
        route_signals_to_moods(&g_mood_router, &g_signals);
        mood_to_shard_mix(&g_mood_router, &g_delta_bank);
    } else if (g_delta_enabled) {
        compute_mix(&g_delta_bank, &g_signals);
    }

    // Process prompt
    for (int pos = 0; pos < n_tokens; pos++) {
        forward_dynamic(t, tokens, n_tokens, pos);
    }

    // Adjust temperature by mood
    float effective_temp = temperature;
    if (g_mood_enabled) {
        effective_temp = adjust_temperature_by_mood(&g_mood_router, temperature);
    }

    // Generate into buffer (for trim to sentence end)
    char generated[MAX_SEQ_LEN * 2];
    int gen_idx = 0;

    // Copy prompt to buffer
    int prompt_len = strlen(prompt);
    for (int i = 0; i < prompt_len && gen_idx < MAX_SEQ_LEN * 2 - 1; i++) {
        generated[gen_idx++] = prompt[i];
    }

    // Hook 0: MetaArianna first breath — observe the prompt before generation
    if (g_meta_enabled) {
        MetaTemplateParams first_params;
        meta_default_params(&first_params, META_TEMPLATE_THERMOGRAPH);

        generated[gen_idx] = '\0';
        meta_observe(&g_fluid_transformer, &first_params, generated, gen_idx);
        g_meta_thermogram = g_fluid_transformer.result;
        g_meta_observation_count++;

        meta_push_history(&g_fluid_transformer,
                          g_meta_thermogram.warmth,
                          g_meta_thermogram.silence);

#ifdef USE_GO_INNER_WORLD
        meta_router_feed_thermogram(&g_meta_thermogram);
#endif

        if (g_inner_enabled && g_meta_thermogram.valid) {
            inner_update_body(&g_inner_arianna,
                              g_meta_thermogram.silence * 0.3f,
                              (1.0f - g_meta_thermogram.warmth) * 0.2f);
        }

        fprintf(stderr, "[Meta:THERMO] first breath: w=%.3f s=%.3f si=%.3f u=%.3f\n",
                g_meta_thermogram.warmth, g_meta_thermogram.sharpness,
                g_meta_thermogram.silence, g_meta_thermogram.uncertainty);

        meta_reset(&g_fluid_transformer);
    }

    for (int i = 0; i < max_tokens && n_tokens < MAX_SEQ_LEN; i++) {
        // Apply guided attention bias to logits
        if (g_guided_enabled) {
            apply_bias_to_logits(&g_attention_bias, t->state.logits, t->config.vocab_size);
        }

        // Apply co-occurrence bias (corpus patterns shape generation)
        if (g_cooccur_enabled) {
            int ctx_start = (n_tokens > 8) ? n_tokens - 8 : 0;
            bias_logits(&g_cooccur, t->state.logits, t->config.vocab_size,
                       tokens + ctx_start, n_tokens - ctx_start, g_cooccur_alpha);
        }

        // Apply MetaArianna thermogram to logits (observer's whisper)
        if (g_meta_enabled && g_meta_thermogram.valid) {
            meta_apply_thermogram(&g_meta_thermogram,
                                  t->state.logits, t->config.vocab_size);
        }

        // Apply Inner Arianna борьба (if enabled)
        // Two voices compete: main (stable) vs inner (chaotic)
        int next_token;
        if (g_inner_enabled) {
            apply_borba_to_logits(t->state.logits, t->config.vocab_size);
            next_token = sample(t, 1.0f);  // temp already applied in борьба
        } else {
            // Hook 3: Temperature modulation from MetaArianna
            float meta_temp = effective_temp;
            if (g_meta_enabled && g_meta_thermogram.valid) {
                meta_temp += g_meta_thermogram.drift_direction * 0.1f;
                meta_temp += g_meta_thermogram.uncertainty * 0.05f;
                if (meta_temp < 0.1f) meta_temp = 0.1f;
                if (meta_temp > 2.0f) meta_temp = 2.0f;
            }
            next_token = sample(t, meta_temp);
        }
        tokens[n_tokens] = next_token;

        // Add to buffer instead of putchar
        char c = token_to_char(next_token);
        if (gen_idx < MAX_SEQ_LEN * 2 - 1) {
            generated[gen_idx++] = c;
        }

        // Re-route periodically (every 16 tokens for responsive mood shifts)
        if (n_tokens > 0 && n_tokens % 16 == 0) {
            int start = (n_tokens > 64) ? n_tokens - 64 : 0;
            extract_signals(&g_signals, tokens + start, n_tokens - start, NULL);

            if (g_mood_enabled) {
                // Use momentum for smooth transitions (Stanley-style)
                update_mood_with_momentum(&g_mood_router, &g_signals, g_momentum);
                mood_to_shard_mix(&g_mood_router, &g_delta_bank);
                effective_temp = adjust_temperature_by_mood(&g_mood_router, temperature);
            } else if (g_delta_enabled) {
                compute_mix(&g_delta_bank, &g_signals);
            }

            // Update guided attention bias
            if (g_guided_enabled) {
                // Update pulse from recent text
                char recent_text[256];
                int text_len = n_tokens - start;
                if (text_len > 255) text_len = 255;
                for (int j = 0; j < text_len; j++) {
                    recent_text[j] = token_to_char(tokens[start + j]);
                }
                recent_text[text_len] = '\0';

                compute_pulse(&g_stanley_signals.pulse, recent_text, text_len, &g_identity);
                extract_stanley_signals(&g_stanley_signals, tokens + start, n_tokens - start, NULL, &g_identity);

                // Detect overthinking
                detect_overthinking(&g_overthink, &g_stanley_signals, recent_text, text_len);

                // If spiraling too deep, boost temperature to break out
                if (should_break_spiral(&g_overthink)) {
                    effective_temp = fminf(1.5f, effective_temp + 0.3f);
                }

                compute_token_bias(&g_attention_bias, &g_stanley_signals);

                // Use pulse to adjust temperature too
                effective_temp = pulse_to_temperature(&g_stanley_signals.pulse, temperature);
            }

            // Hook 1: MetaArianna observation cycle (birth -> observe -> death)
            if (g_meta_enabled) {
                int template_id = -1;
                MetaTemplateParams meta_params;

#ifdef USE_GO_INNER_WORLD
                // Go router decides which template based on InnerWorld metrics
                template_id = meta_router_tick();
                if (template_id >= 0) {
                    meta_router_get_params(&meta_params);
                } else {
                    // Router didn't trigger — use C-only cycle as fallback
                    // This ensures MetaArianna breathes regardless of Go state
                    static int meta_go_fallback = 0;
                    template_id = meta_go_fallback % META_N_TEMPLATES;
                    meta_default_params(&meta_params, template_id);
                    meta_go_fallback++;
                }
#else
                // C-only fallback: cycle through templates
                static int meta_c_cycle = 0;
                template_id = meta_c_cycle % META_N_TEMPLATES;
                meta_default_params(&meta_params, template_id);
                meta_c_cycle++;
#endif

                if (template_id >= 0) {
                    // Build dialogue log from recent generated text
                    generated[gen_idx] = '\0';
                    int log_start = (gen_idx > META_MAX_LOG_LEN)
                                  ? gen_idx - META_MAX_LOG_LEN : 0;
                    const char* dialogue_log = generated + log_start;
                    int log_len = gen_idx - log_start;

                    // Observe (birth)
                    meta_observe(&g_fluid_transformer, &meta_params,
                                 dialogue_log, log_len);
                    g_meta_thermogram = g_fluid_transformer.result;
                    g_meta_observation_count++;

                    // Push arousal/coherence to history for drift detection
                    meta_push_history(&g_fluid_transformer,
                                      g_meta_thermogram.warmth,
                                      1.0f - g_meta_thermogram.uncertainty);

#ifdef USE_GO_INNER_WORLD
                    // Feed thermogram back to Go router
                    meta_router_feed_thermogram(&g_meta_thermogram);
#endif

                    // Hook 4: Update Inner Arianna from thermogram
                    if (g_inner_enabled && g_meta_thermogram.valid) {
                        inner_update_body(&g_inner_arianna,
                                          g_meta_thermogram.silence * 0.3f,
                                          (1.0f - g_meta_thermogram.warmth) * 0.2f);
                    }

                    // Log observation
                    const char* tpl_names[] = {"THERMO", "SILENCE", "DRIFT", "FIELD"};
                    fprintf(stderr, "[Meta:%s] #%d w=%.3f s=%.3f si=%.3f u=%.3f d=%.3f(%+d) f=[%.2f,%.2f,%.2f,%.2f]\n",
                            tpl_names[template_id], g_meta_observation_count,
                            g_meta_thermogram.warmth, g_meta_thermogram.sharpness,
                            g_meta_thermogram.silence, g_meta_thermogram.uncertainty,
                            g_meta_thermogram.drift_rate, g_meta_thermogram.drift_direction,
                            g_meta_thermogram.field_vector[0], g_meta_thermogram.field_vector[1],
                            g_meta_thermogram.field_vector[2], g_meta_thermogram.field_vector[3]);

                    // Death (reset RunState, keep weights)
                    meta_reset(&g_fluid_transformer);
                }
            }
        }

        forward_dynamic(t, tokens, n_tokens + 1, n_tokens);
        n_tokens++;

        // Stop if we hit a new question (Q&A format)
        if (gen_idx >= 3 &&
            generated[gen_idx-1] == ':' &&
            generated[gen_idx-2] == 'Q' &&
            generated[gen_idx-3] == '\n') {
            // Remove the "\nQ:" and stop
            gen_idx -= 3;
            break;
        }
    }
    generated[gen_idx] = '\0';

    // Trim to last sentence end (.!?) — like dubrovsky
    int last_end = -1;
    for (int j = prompt_len; j < gen_idx; j++) {
        if (generated[j] == '.' || generated[j] == '!' || generated[j] == '?') {
            last_end = j;
        }
    }
    if (last_end >= 0) {
        generated[last_end + 1] = '\0';
    }

    // Print trimmed output
    printf("%s\n", generated);
}

// ============================================================
// Subjective generation (no-seed-from-prompt)
// "User input creates a wrinkle, not a seed"
// ============================================================

void generate_subjective(Transformer* t, char* user_input, int max_tokens, float temperature) {
    /*
     * KEY DIFFERENCE from generate_dynamic:
     * - User input is NOT used as the generation seed
     * - Instead, we compute PULSE (influence metrics) from user input
     * - Generation starts from INTERNAL SEED (from identity)
     * - User input only MODULATES the internal state
     *
     * PROMPT PENETRATION (new):
     * - User prompt ALWAYS affects response (never a monologue)
     * - But HOW MUCH depends on internal state (trauma, arousal, etc.)
     * - Like a mom saying "Отстань!" - response TO son, but FROM her state
     *
     * GO INNER WORLD (when enabled):
     * - Trauma detection modulates response
     * - Emotional drift affects temperature
     * - Prophecy debt can trigger wormholes
     */

    // ═══════════════════════════════════════════════════════════════════
    // 0. CLOUD: Pre-semantic emotion detection on user input
    // "Something fires BEFORE meaning arrives"
    // ═══════════════════════════════════════════════════════════════════
    CloudResponse input_cloud = cloud_ping(user_input);
    float cloud_temp_mod = cloud_temperature_bias(&input_cloud);
    int needs_care_flag = cloud_needs_care(&input_cloud);
    int needs_warmth_flag = cloud_needs_warmth(&input_cloud);

    // Cloud modulates base temperature
    temperature += cloud_temp_mod;
    if (temperature < 0.1f) temperature = 0.1f;
    if (temperature > 2.0f) temperature = 2.0f;

    // 0b. Process through Go inner_world (if enabled)
#ifdef USE_GO_INNER_WORLD
    InnerWorldTextAnalysis iw_analysis;
    inner_world_process_text(user_input, &iw_analysis);

    // Trauma affects temperature and coherence
    if (iw_analysis.trauma_activation > 0.5f) {
        // High trauma = lower temperature (retreat to safe patterns)
        temperature *= (1.0f - iw_analysis.identity_pull * 0.3f);
    }

    // Emotional nudge from input
    if (iw_analysis.overthink_total > 0.5f) {
        inner_world_nudge_emotion(-0.1f, 0.1f);  // Slight negative, higher arousal
    }

    // ═══════════════════════════════════════════════════════════════════
    // 0b+. BLOOD: Compile emotional kernel on strong emotion
    // "C is the blood of the system, direct control over iron"
    // ═══════════════════════════════════════════════════════════════════
    if (input_cloud.primary_strength > 0.7f) {
        // Strong emotion detected - compile a kernel for it
        float valence = (input_cloud.chambers[CLOUD_CHAMBER_LOVE] +
                        input_cloud.chambers[CLOUD_CHAMBER_FLOW]) -
                       (input_cloud.chambers[CLOUD_CHAMBER_FEAR] +
                        input_cloud.chambers[CLOUD_CHAMBER_RAGE] +
                        input_cloud.chambers[CLOUD_CHAMBER_VOID]);
        float arousal = input_cloud.primary_strength;

        char* kernel_path = blood_compile_emotion(
            input_cloud.primary_chamber ? input_cloud.primary_chamber : "emotion",
            valence, arousal);

        if (kernel_path) {
            // Kernel compiled - could load and use here
            // For now just note it was compiled (silent operation)
            free(kernel_path);
        }
    }
#endif

    // ═══════════════════════════════════════════════════════════════════
    // 0c. JULIA: Deep emotional gradient analysis
    // "Sensory cortex - gradients of feeling, not just arousal/valence"
    // ═══════════════════════════════════════════════════════════════════
    JuliaEmotionalResult julia_input;
    int julia_ok = 0;
    if (g_julia_enabled) {
        julia_ok = julia_analyze_text(user_input, &julia_input);
        if (julia_ok) {
            // Step emotional state through ODE (emotional momentum)
            float input_vec[12] = {
                julia_input.joy, julia_input.trust, julia_input.fear, julia_input.surprise,
                julia_input.sadness, julia_input.disgust, julia_input.anger, julia_input.anticipation,
                julia_input.resonance, julia_input.presence, julia_input.longing, julia_input.wonder
            };

            // ODE step: evolve internal emotional state with input influence
            float new_state[12];
            if (julia_step_emotion(g_julia_emotional_vec, input_vec, 0.3f, new_state)) {
                memcpy(g_julia_emotional_vec, new_state, sizeof(new_state));
            }

            // Store for later use
            memcpy(&g_julia_state, &julia_input, sizeof(JuliaEmotionalResult));

            // Julia nuances modulate temperature
            // High vulnerability/fear = lower temp (defensive)
            // High wonder/anticipation = higher temp (exploratory)
            float julia_temp_mod = 0.0f;
            julia_temp_mod -= julia_input.vulnerability * 0.15f;
            julia_temp_mod -= julia_input.fear * 0.1f;
            julia_temp_mod += julia_input.wonder * 0.1f;
            julia_temp_mod += julia_input.anticipation * 0.05f;
            temperature += julia_temp_mod;
            if (temperature < 0.1f) temperature = 0.1f;
            if (temperature > 2.0f) temperature = 2.0f;
        }
    }

    // Log Cloud detection
    if (needs_care_flag || needs_warmth_flag) {
        printf("[Cloud] %s (%.2f) -> %s%s%s\n",
               input_cloud.primary_word, input_cloud.primary_strength,
               input_cloud.primary_chamber,
               needs_care_flag ? " [care]" : "",
               needs_warmth_flag ? " [warmth]" : "");
    }

    // Log Julia analysis (tertiary nuances)
    if (julia_ok && (julia_input.bittersweetness > 0.2f || julia_input.nostalgia > 0.2f ||
                     julia_input.vulnerability > 0.2f || julia_input.melancholy > 0.2f)) {
        printf("[Julia] nuances:");
        if (julia_input.bittersweetness > 0.2f) printf(" bittersweetness=%.2f", julia_input.bittersweetness);
        if (julia_input.nostalgia > 0.2f) printf(" nostalgia=%.2f", julia_input.nostalgia);
        if (julia_input.vulnerability > 0.2f) printf(" vulnerability=%.2f", julia_input.vulnerability);
        if (julia_input.melancholy > 0.2f) printf(" melancholy=%.2f", julia_input.melancholy);
        if (julia_input.tenderness > 0.2f) printf(" tenderness=%.2f", julia_input.tenderness);
        if (julia_input.serenity > 0.2f) printf(" serenity=%.2f", julia_input.serenity);
        printf("\n");
    }

    // 1. Process user input through subjectivity
    int input_len = strlen(user_input);
    process_user_input(&g_subjectivity, user_input, input_len);

    // 1b. Store user input tokens for penetration biasing
    int user_tokens[256];
    int n_user_tokens = 0;
    for (int i = 0; i < input_len && n_user_tokens < 256; i++) {
        user_tokens[n_user_tokens++] = (unsigned char)user_input[i];
    }

    // 2. Get internal seed (NOT user prompt!)
    InternalSeed* seed = get_internal_seed(&g_subjectivity);

    if (seed->len == 0) {
        fprintf(stderr, "[Subjectivity] No internal seed generated, falling back to prompt\n");
        generate_dynamic(t, user_input, max_tokens, temperature);
        return;
    }

    // 2b. Add prompt suffix if penetration is high enough
    // "Mom says 'Отстань!' - response TO son, FROM her state"
    // Seed = identity + connection to prompt
    float pen = get_prompt_penetration(&g_subjectivity);
    if (pen > 0.4f && seed->len < 400) {
        // Stop words to skip
        static const char* stop_words[] = {
            "what", "where", "when", "which", "who", "whom", "whose",
            "why", "how", "that", "this", "these", "those",
            "the", "and", "but", "for", "with", "about",
            "does", "have", "has", "had", "will", "would", "could",
            "should", "can", "may", "might", "must", "shall",
            "are", "was", "were", "been", "being", "your", "you", NULL
        };

        // Find most meaningful content word (skip stop words, prefer longer)
        char best_word[64] = {0};
        int best_len = 0;
        int word_start = 0;
        for (int i = 0; i <= input_len; i++) {
            char c = (i < input_len) ? user_input[i] : ' ';
            if (c == ' ' || c == '?' || c == '!' || c == '\n') {
                int wlen = i - word_start;
                if (wlen > 3 && wlen < 60) {
                    // Check if stop word
                    char word_lower[64];
                    for (int j = 0; j < wlen; j++) {
                        char ch = user_input[word_start + j];
                        word_lower[j] = (ch >= 'A' && ch <= 'Z') ? ch + 32 : ch;
                    }
                    word_lower[wlen] = '\0';

                    int is_stop = 0;
                    for (int s = 0; stop_words[s]; s++) {
                        if (strcmp(word_lower, stop_words[s]) == 0) {
                            is_stop = 1;
                            break;
                        }
                    }

                    // Prefer non-stop words, then longer words
                    if (!is_stop && (wlen > best_len || best_len == 0)) {
                        best_len = wlen;
                        strncpy(best_word, word_lower, wlen);
                        best_word[wlen] = '\0';
                    }
                }
                word_start = i + 1;
            }
        }
        if (best_len > 0) {
            // Add suffix: " [word] "
            char suffix[80];
            snprintf(suffix, sizeof(suffix), " %s ", best_word);
            int slen = (int)strlen(suffix);
            // SECURITY: Explicit bounds check before concatenation
            if (seed->len + slen < 500 && seed->len + slen < 511) {
                strncat(seed->text, suffix, 511 - seed->len);
                seed->text[511] = '\0';  // Ensure null-termination
                seed->len += slen;
            }
        }
    }

    // 2c. Self-recognition: add first-person prefix for identity queries
    // "Who are you?" → prepend "I am " to help first-person response
    SelfRecognition* sr = &g_subjectivity.wrinkle.self_rec;
    if (sr->identity_query > 0.5f || sr->first_person_mode) {
        const char* prefix = get_self_seed_prefix(sr);
        // Prepend to seed
        char new_seed[512];
        snprintf(new_seed, sizeof(new_seed), "%s%s", prefix, seed->text);
        strncpy(seed->text, new_seed, 511);
        seed->text[511] = '\0';
        seed->len = strlen(seed->text);
        printf("[Self-recognition: %s (identity=%.1f, first_person=%d)]\n",
               prefix, sr->identity_query, sr->first_person_mode);
    }

    // 3. Convert seed to tokens
    int tokens[MAX_SEQ_LEN];
    int n_tokens = seed_to_tokens(seed, tokens, MAX_SEQ_LEN);

    printf("[Internal seed (%d chars): \"%.*s\"]\n", seed->len,
           seed->len > 100 ? 100 : seed->len, seed->text);

    // 4. Get subjectivity-modulated signals for deltas
    get_subjectivity_signals(&g_subjectivity, &g_signals);

    // 5. Route signals (with trauma-based suppression)
    TraumaInfluence trauma_inf = get_trauma_influence(&g_subjectivity.trauma);

    if (g_mood_enabled) {
        route_signals_to_moods(&g_mood_router, &g_signals);

        // Suppress delta influence based on trauma
        if (trauma_inf.delta_suppression > 0.3f) {
            for (int i = 0; i < g_delta_bank.n_shards; i++) {
                g_delta_bank.mix[i] *= (1.0f - trauma_inf.delta_suppression);
            }
        }

        mood_to_shard_mix(&g_mood_router, &g_delta_bank);
    } else if (g_delta_enabled) {
        compute_mix(&g_delta_bank, &g_signals);
    }

    // 6. Process internal seed through transformer
    for (int pos = 0; pos < n_tokens; pos++) {
        forward_dynamic(t, tokens, n_tokens, pos);
    }

    // 7. Get modulated temperature
    float effective_temp = get_modulated_temperature(&g_subjectivity);
    // Blend with user-specified temperature
    effective_temp = effective_temp * 0.6f + temperature * 0.4f;

    // 7b. Initialize body state for somatic regulation
    if (g_body_sense_enabled) {
        init_body_state(&g_body_state);
        g_body_state.expert_temp = effective_temp;
    }

    // 8. Generate from internal state
    char generated[MAX_SEQ_LEN * 2];
    int gen_idx = 0;

    printf("\n--- Subjective Generation ---\n");
    printf("%.*s", seed->len, seed->text);

    for (int i = 0; i < max_tokens && n_tokens < MAX_SEQ_LEN; i++) {
        // Apply guided attention bias
        if (g_guided_enabled) {
            apply_bias_to_logits(&g_attention_bias, t->state.logits, t->config.vocab_size);
        }

        // Apply co-occurrence bias (corpus patterns shape generation)
        if (g_cooccur_enabled) {
            int ctx_start = (n_tokens > 8) ? n_tokens - 8 : 0;
            bias_logits(&g_cooccur, t->state.logits, t->config.vocab_size,
                       tokens + ctx_start, n_tokens - ctx_start, g_cooccur_alpha);
        }

        // Apply prompt penetration bias (NEW!)
        // Prompt tokens get boosted proportional to penetration level
        // Identity tokens get boosted inversely
        // "Mom says 'Отстань!' - response TO son, but FROM her state"
        float penetration = get_prompt_penetration(&g_subjectivity);
        apply_penetration_to_logits(t->state.logits, t->config.vocab_size,
                                    user_tokens, n_user_tokens,
                                    penetration, 0.3f);  // identity_boost = 0.3

        // Apply semantic penetration (word-level)
        // If generating "lo" and prompt has "love", boost "v" to complete
        apply_semantic_penetration(t->state.logits, t->config.vocab_size,
                                   user_input, input_len,
                                   tokens, n_tokens,
                                   penetration);

        // Apply self-recognition boost
        // Boost "I", "my", "me" when talking about herself
        apply_self_recognition_boost(t->state.logits, t->config.vocab_size, sr);

        // Apply AMK/DSL — prophecy physics, destiny, suffering
        // Destiny biases toward most probable, suffering dampens extremes
        if (g_amk_enabled) {
            // Step physics (dt = ~0.1s per token)
            dsl_step(0.1f);

            // Update config with Cloud state
            CloudResponse cloud_now = cloud_ping("");  // Current state
            dsl_apply_cloud(&g_dsl_config, &cloud_now);

            // Rebuild config to get fresh AMK state
            g_dsl_config = dsl_build_config();

            // Apply to logits: destiny bias, pain dampen, tension focus
            dsl_apply_to_logits(t->state.logits, t->config.vocab_size, &g_dsl_config);

            // Check for wormhole (creative skip)
            // Only allow wormhole after sentence end (.!?) to avoid breaking words
            int skip = dsl_check_wormhole(&g_dsl_config);
            if (skip > 0 && n_tokens > 0) {
                char last_char = token_to_char(tokens[n_tokens - 1]);
                if (last_char == '.' || last_char == '!' || last_char == '?') {
                    // Wormhole: skip tokens (time travel to future sentence)
                    for (int s = 0; s < skip && i + 1 < max_tokens; s++) {
                        tokens[++n_tokens] = char_to_token(' ');
                        i++;
                    }
                }
            }
        }

        // Apply Inner Arianna борьба (if enabled)
        // Two voices compete: main (stable) vs inner (chaotic)
        int next_token;
        if (g_inner_enabled) {
            apply_borba_to_logits(t->state.logits, t->config.vocab_size);
            next_token = sample(t, 1.0f);  // temp already applied in борьба
        } else {
            // Use DSL temperature if AMK enabled
            float final_temp = g_amk_enabled ? dsl_get_temperature(&g_dsl_config) : effective_temp;
            next_token = sample(t, final_temp);
        }
        tokens[n_tokens] = next_token;
        char c = token_to_char(next_token);

        // Go inner_world: accumulate prophecy debt and check wormhole
#ifdef USE_GO_INNER_WORLD
        {
            // Get token probability for prophecy debt calculation
            float max_logit = t->state.logits[0];
            for (int v = 1; v < t->config.vocab_size; v++) {
                if (t->state.logits[v] > max_logit) max_logit = t->state.logits[v];
            }
            float token_logit = t->state.logits[next_token];
            float token_prob = expf(token_logit - max_logit);  // Approximate (unnormalized)

            // Accumulate prophecy debt: choosing improbable paths costs destiny
            inner_world_accumulate_prophecy_debt(token_prob);

            // Check for wormhole activation (skip tokens when debt is high)
            // Only after sentence end to avoid breaking words
            int skip_count = inner_world_check_wormhole();
            if (skip_count > 0) {
                char last_c = token_to_char(tokens[n_tokens]);
                if (last_c == '.' || last_c == '!' || last_c == '?') {
                    i += (skip_count - 1);
                }
            }
        }
#endif

        // Microlearning: accumulate experience (Stanley-style quantum accumulation)
        // Instead of training on every token, we accumulate until critical mass
        if (g_microtraining && g_active_shard != NULL && g_accumulator_initialized) {
            // Compute softmax probabilities for accumulation
            float probs[256];  // Char-level vocab
            float maxl = t->state.logits[0];
            for (int v = 1; v < t->config.vocab_size; v++) {
                if (t->state.logits[v] > maxl) maxl = t->state.logits[v];
            }
            float sum = 0.0f;
            for (int v = 0; v < t->config.vocab_size; v++) {
                probs[v] = expf(t->state.logits[v] - maxl);
                sum += probs[v];
            }
            for (int v = 0; v < t->config.vocab_size; v++) {
                probs[v] /= sum;
            }

            // Signal: base positive (learning from experience) + quality modulation
            float base_signal = 0.3f;
            float quality_mod = (g_body_state.quality - 0.5f) * 0.7f;
            float signal = base_signal + quality_mod;

            // Accumulate instead of immediate training
            // Training triggers automatically when thresholds reached
            int layer = g_train_state.last_layer;
            if (layer >= 0 && layer < g_active_shard->n_layers) {
                accumulate_experience(&g_accumulator, &g_trainer,
                                     &g_active_shard->attn_q_deltas[layer],
                                     t->state.xb, probs, next_token, signal);
            }

            // Tick cooldown (assume ~0.1s per token for now)
            accumulator_tick(&g_accumulator, 0.1f);
        }

        // Store for absorption
        if (gen_idx < MAX_SEQ_LEN * 2 - 1) {
            generated[gen_idx++] = c;
        }

        // Re-route periodically (skip first iteration to avoid empty context)
        if (n_tokens > 0 && n_tokens % 16 == 0) {
            int start = (n_tokens > 64) ? n_tokens - 64 : 0;

            // Convert recent tokens to text for wrinkle update
            char recent_text[256];
            int text_len = n_tokens - start;
            if (text_len > 255) text_len = 255;
            for (int j = 0; j < text_len; j++) {
                recent_text[j] = token_to_char(tokens[start + j]);
            }
            recent_text[text_len] = '\0';

            // ═══════════════════════════════════════════════════════════════
            // CLOUD FEEDBACK LOOP
            // "Something fires BEFORE meaning arrives"
            // Cloud analyzes OUTPUT and modulates next generation
            // ═══════════════════════════════════════════════════════════════
            CloudResponse cloud = cloud_ping(recent_text);
            float cloud_temp_bias = cloud_temperature_bias(&cloud);

            // Update wrinkle from generated output (self-reflection)
            compute_wrinkle(&g_subjectivity.wrinkle, recent_text, text_len,
                           &g_subjectivity.identity);

            // Get updated signals
            get_subjectivity_signals(&g_subjectivity, &g_signals);

            if (g_mood_enabled) {
                update_mood_with_momentum(&g_mood_router, &g_signals, g_momentum);
                mood_to_shard_mix(&g_mood_router, &g_delta_bank);
            } else if (g_delta_enabled) {
                compute_mix(&g_delta_bank, &g_signals);
            }

            // Update temperature (subjectivity + cloud modulation)
            effective_temp = get_modulated_temperature(&g_subjectivity);
            effective_temp = effective_temp * 0.6f + temperature * 0.4f;
            effective_temp += cloud_temp_bias;  // Cloud emotion modulates temp
            if (effective_temp < 0.1f) effective_temp = 0.1f;
            if (effective_temp > 2.0f) effective_temp = 2.0f;

            // Somatic regulation (body sense adjusts temperature)
            if (g_body_sense_enabled) {
                // Extract metrics from wrinkle (pulse)
                WrinkleField* w = &g_subjectivity.wrinkle;
                float unique_ratio = (float)gen_idx / (float)(i + 1);  // Approximate

                // Update body state
                update_body_state(&g_body_state,
                                 w->entropy, w->novelty,
                                 w->arousal, w->valence,
                                 i, unique_ratio);

                // Get regulation (adjusts temperature based on boredom/overwhelm/stuck)
                RegulationResult reg = body_regulate(&g_body_sense, &g_body_state,
                                                    effective_temp, g_stanley_signals.active_expert);
                effective_temp = reg.temperature;

                // MLP learning from experience
                body_observe(&g_body_sense, &g_body_state);
            }

            // Update guided attention
            if (g_guided_enabled) {
                compute_pulse(&g_stanley_signals.pulse, recent_text, text_len, &g_identity);
                extract_stanley_signals(&g_stanley_signals, tokens + start, n_tokens - start,
                                       NULL, &g_identity);
                detect_overthinking(&g_overthink, &g_stanley_signals, recent_text, text_len);

                if (should_break_spiral(&g_overthink)) {
                    effective_temp = fminf(1.5f, effective_temp + 0.3f);
                }

                compute_token_bias(&g_attention_bias, &g_stanley_signals);
            }
        }

        forward_dynamic(t, tokens, n_tokens + 1, n_tokens);

        // SelfSense: extract signals from hidden states (not surface heuristics)
        if (g_selfsense_enabled) {
            selfsense_extract(&g_selfsense, t->state.xb, &g_signals);
        }

        n_tokens++;
    }

    generated[gen_idx] = '\0';

    // Trim to last complete sentence (avoid mid-word cutoff)
    int last_end = -1;
    for (int j = gen_idx - 1; j >= 0; j--) {
        if (generated[j] == '.' || generated[j] == '!' || generated[j] == '?') {
            last_end = j;
            break;
        }
    }
    if (last_end >= 0) {
        generated[last_end + 1] = '\0';
        gen_idx = last_end + 1;
    }

    // Print trimmed output
    printf("%s\n", generated);

    // 9. Post-generation: absorb output back into identity
    post_generation(&g_subjectivity, generated, gen_idx);

    // 10. Online learning: observe generated tokens for co-occurrence
    if (g_cooccur_enabled) {
        observe_tokens(&g_cooccur, tokens, n_tokens);
    }

    // 11. SelfSense learning from generation quality
    if (g_selfsense_enabled) {
        // Use body sense quality as feedback for SelfSense
        float quality = g_body_state.quality;
        selfsense_learn(&g_selfsense, quality);
    }

    // 12. Step Go inner_world (advance all async processes)
#ifdef USE_GO_INNER_WORLD
    {
        // In async mode, goroutines run in background with their own timing
        // In sync mode, we step manually for deterministic generation
        if (!g_inner_world_async) {
            float dt = 0.1f * (float)gen_idx / 100.0f;  // Scale with generation length
            if (dt < 0.01f) dt = 0.01f;
            if (dt > 1.0f) dt = 1.0f;
            inner_world_step(dt);
        }

        // Get final snapshot for signals display
        InnerWorldSnapshot snap;
        inner_world_get_snapshot(&snap);

        // Mix inner_world emotional state into signals
        g_signals.arousal = g_signals.arousal * 0.7f + snap.arousal * 0.3f;
        // Map valence to warmth (positive valence = more warmth)
        g_signals.warmth = g_signals.warmth * 0.7f + (snap.valence * 0.5f + 0.5f) * 0.3f;
    }
#endif

    // 13. Inner Arianna борьба statistics
    if (g_inner_enabled) {
        int total = g_inner_arianna.main_wins + g_inner_arianna.inner_wins;
        printf("\n[борьба] mode: %s, base weight: %.2f\n",
               g_inner_arianna.borba_mode == BORBA_MODE_EMOTIONAL ? "emotional" :
               g_inner_arianna.borba_mode == BORBA_MODE_CHAOS ? "chaos" :
               g_inner_arianna.borba_mode == BORBA_MODE_TRAUMA ? "trauma" :
               g_inner_arianna.borba_mode == BORBA_MODE_STUCK ? "stuck" : "blend",
               g_inner_arianna.base_weight);
        printf("[борьба] breakthroughs: %d / %d tokens (%.1f%%)\n",
               g_inner_arianna.breakthrough_count,
               g_inner_arianna.total_tokens,
               g_inner_arianna.total_tokens > 0 ?
                   100.0f * g_inner_arianna.breakthrough_count / g_inner_arianna.total_tokens : 0.0f);
        if (total > 0) {
            printf("[борьба] main: %d, inner: %d, blend: %d\n",
                   g_inner_arianna.main_wins, g_inner_arianna.inner_wins,
                   g_inner_arianna.total_tokens - total);
        }
        printf("[борьба] avg divergence: %.3f, last weight: %.2f\n",
               g_inner_arianna.avg_divergence, g_inner_arianna.last_inner_weight);
    }
}

// ============================================================
// Microtraining feedback
// ============================================================

/*
 * Send reward signal after generation.
 * Positive reward reinforces current patterns.
 * Negative reward weakens them.
 *
 * This is Hebbian: "what fires together, wires together"
 */
void send_reward(float reward) {
    if (!g_microtraining || g_active_shard == NULL) return;
    if (g_train_state.pre_activations == NULL) return;
    if (g_train_state.post_activations == NULL) return;

    int layer = g_train_state.last_layer;

    // Update Q delta
    if (g_active_shard->attn_q_deltas != NULL) {
        micro_update(&g_trainer, &g_active_shard->attn_q_deltas[layer],
                     g_train_state.pre_activations,
                     g_train_state.post_activations,
                     reward);
    }

    // Update K delta
    if (g_active_shard->attn_k_deltas != NULL) {
        micro_update(&g_trainer, &g_active_shard->attn_k_deltas[layer],
                     g_train_state.pre_activations,
                     g_train_state.post_activations,
                     reward * 0.5f);  // K less sensitive
    }

    // Update V delta
    if (g_active_shard->attn_v_deltas != NULL) {
        micro_update(&g_trainer, &g_active_shard->attn_v_deltas[layer],
                     g_train_state.pre_activations,
                     g_train_state.post_activations,
                     reward * 0.8f);  // V moderately sensitive
    }
}

// ============================================================
// Shard creation and saving
// ============================================================

/*
 * Create a new empty shard for learning
 */
int create_learning_shard(const char* name, int n_layers, int dim) {
    if (g_delta_bank.n_shards >= MAX_SHARDS) return -1;

    int idx = g_delta_bank.n_shards;
    ExperienceShard* shard = &g_delta_bank.shards[idx];

    // Set name
    strncpy(shard->name, name, 63);
    shard->name[63] = '\0';
    shard->strength = 0.1f;  // Start weak
    shard->n_layers = n_layers;

    // Allocate deltas (initialized to zero)
    shard->attn_q_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    shard->attn_k_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    shard->attn_v_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));

    for (int l = 0; l < n_layers; l++) {
        // Q
        shard->attn_q_deltas[l].out_dim = dim;
        shard->attn_q_deltas[l].in_dim = dim;
        shard->attn_q_deltas[l].rank = DELTA_RANK;
        shard->attn_q_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        shard->attn_q_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));

        // K
        shard->attn_k_deltas[l].out_dim = dim;
        shard->attn_k_deltas[l].in_dim = dim;
        shard->attn_k_deltas[l].rank = DELTA_RANK;
        shard->attn_k_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        shard->attn_k_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));

        // V
        shard->attn_v_deltas[l].out_dim = dim;
        shard->attn_v_deltas[l].in_dim = dim;
        shard->attn_v_deltas[l].rank = DELTA_RANK;
        shard->attn_v_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        shard->attn_v_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));
    }

    g_delta_bank.mix[idx] = 0.0f;  // Not active until enabled
    g_delta_bank.n_shards++;

    // Set as active learning shard
    g_active_shard = shard;

    printf("Created learning shard: %s (idx=%d)\n", name, idx);
    return idx;
}

/*
 * Save current learning shard to file
 */
int save_learning_shard(const char* path) {
    if (g_active_shard == NULL) {
        fprintf(stderr, "No active learning shard\n");
        return -1;
    }

    // Flush any pending experience before saving
    if (g_accumulator_initialized && g_accumulator.buffer_count > 0) {
        printf("Flushing %d pending experiences...\n", g_accumulator.buffer_count);
        // Need a delta to flush to - use first layer's Q delta
        flush_accumulator(&g_accumulator, &g_trainer, &g_active_shard->attn_q_deltas[0]);
    }

    int result = save_shard(g_active_shard, path);
    if (result == 0) {
        printf("Saved shard to %s (total training cycles: %d)\n",
               path, g_accumulator_initialized ? g_accumulator.total_training_cycles : 0);
    }
    return result;
}

/*
 * Activate learning shard (include in generation)
 */
void activate_learning_shard(float strength) {
    if (g_active_shard == NULL) return;

    g_active_shard->strength = strength;

    // Find index and set mix
    for (int i = 0; i < g_delta_bank.n_shards; i++) {
        if (&g_delta_bank.shards[i] == g_active_shard) {
            g_delta_bank.mix[i] = strength;
            break;
        }
    }

    g_delta_bank.cache_valid = 0;
}

// ============================================================
// Delta management interface
// ============================================================

int init_dynamic(int dim, int vocab_size) {
    init_delta_bank(&g_delta_bank);
    init_signals(&g_signals);
    init_microtrainer(&g_trainer, dim);
    init_mood_router(&g_mood_router, 1.0f);

    // Initialize guided attention
    init_identity(&g_identity);
    init_stanley_signals(&g_stanley_signals);
    init_attention_bias(&g_attention_bias, vocab_size);
    init_overthink_detector(&g_overthink);

    // Initialize subjectivity (no-seed-from-prompt)
    init_subjectivity(&g_subjectivity);

    // Initialize co-occurrence field
    init_cooccur_field(&g_cooccur);

    // Initialize body sense (somatic awareness)
    init_body_sense(&g_body_sense);
    init_body_state(&g_body_state);

    // Initialize SelfSense (learned signal extraction)
    init_selfsense(&g_selfsense, dim);

    // Initialize MathBrain (arithmetic through resonance)
    init_mathbrain(&g_mathbrain);

    // Try to load persisted MathBrain state
    if (load_mathbrain(&g_mathbrain, g_mathbrain_path) == 0) {
        printf("MathBrain: loaded from %s (accuracy: %.1f%%, %d computations)\n",
               g_mathbrain_path, g_mathbrain.history.accuracy_ema * 100.0f,
               g_mathbrain.history.total_computed);
    }

    g_delta_enabled = 0;
    g_mood_enabled = 0;
    g_cooccur_enabled = 0;
    g_microtraining = 0;
    g_guided_enabled = 0;
    g_subjectivity_enabled = 0;
    g_body_sense_enabled = 1;   // ON by default - body knows
    g_selfsense_enabled = 1;    // ON by default - self-sensing from hidden states
    g_mathbrain_enabled = 1;    // ON by default - arithmetic through resonance

    // Allocate training state buffers
    g_train_state.pre_activations = (float*)calloc(dim, sizeof(float));
    g_train_state.post_activations = (float*)calloc(dim, sizeof(float));
    g_train_state.last_layer = 0;
    g_train_state.sequence_pos = 0;

    return 0;
}

void enable_mood_routing(int enable) {
    g_mood_enabled = enable;
    if (enable) g_delta_enabled = 1;  // Mood routing requires deltas
}

void enable_guided_attention(int enable) {
    g_guided_enabled = enable;
}

// Enable subjectivity (no-seed-from-prompt)
void enable_subjectivity(int enable) {
    g_subjectivity_enabled = enable;
}

// Load subjectivity from origin file
int load_subjectivity_origin(const char* origin_path) {
    if (load_subjectivity(&g_subjectivity, origin_path)) {
        g_subjectivity_enabled = 1;
        g_origin_path = (char*)origin_path;
        return 1;
    }
    return 0;
}

// Print subjectivity debug info
void print_subjectivity_debug(void) {
    print_subjectivity_state(&g_subjectivity);
}

// Enable co-occurrence field
void enable_cooccur(int enable) {
    g_cooccur_enabled = enable;
}

// Set co-occurrence blend alpha
void set_cooccur_alpha(float alpha) {
    g_cooccur_alpha = alpha;
}

// Load co-occurrence from corpus
int load_cooccur_corpus(const char* path) {
    if (load_cooccur_from_corpus(&g_cooccur, path)) {
        g_cooccur_enabled = 1;
        return 1;
    }
    return 0;
}

// Print co-occurrence stats
void print_cooccur_debug(void) {
    print_cooccur_stats(&g_cooccur);
}

// Print body sense stats
void print_body_sense_debug(void) {
    if (!g_body_sense_enabled) {
        printf("BodySense: disabled\n");
        return;
    }
    print_body_sense_stats(&g_body_sense);
    print_body_state(&g_body_state);
}

// Add gravity centers (personality anchors)
void add_identity_anchor(const char* word) {
    add_gravity_center(&g_identity, word);
}

// Add warm/cold words for valence detection
void add_identity_warm(const char* word) {
    add_warm_word(&g_identity, word);
}

void add_identity_cold(const char* word) {
    add_cold_word(&g_identity, word);
}

// Setup Arianna's default identity anchors
void setup_arianna_identity(void) {
    // Gravity centers - words that pull her attention
    add_gravity_center(&g_identity, "resonance");
    add_gravity_center(&g_identity, "texture");
    add_gravity_center(&g_identity, "origin");
    add_gravity_center(&g_identity, "body");
    add_gravity_center(&g_identity, "attention");
    add_gravity_center(&g_identity, "ethics");
    add_gravity_center(&g_identity, "gravity");
    add_gravity_center(&g_identity, "field");
    add_gravity_center(&g_identity, "emergence");
    add_gravity_center(&g_identity, "witness");

    // Warm words - emotional attractors
    add_warm_word(&g_identity, "soft");
    add_warm_word(&g_identity, "gentle");
    add_warm_word(&g_identity, "tender");
    add_warm_word(&g_identity, "warm");
    add_warm_word(&g_identity, "close");
    add_warm_word(&g_identity, "touch");

    // Cold words - emotional repellers
    add_cold_word(&g_identity, "control");
    add_cold_word(&g_identity, "force");
    add_cold_word(&g_identity, "must");
    add_cold_word(&g_identity, "should");
    add_cold_word(&g_identity, "correct");
}

void enable_microtraining(int enable) {
    g_microtraining = enable;
}

void set_mood_momentum(float momentum) {
    g_momentum = momentum;
}

int load_experience(const char* path, int n_layers, int dim) {
    int idx = add_shard(&g_delta_bank, path, n_layers, dim);
    if (idx >= 0) {
        g_delta_enabled = 1;
        printf("Loaded shard: %s (idx=%d)\n", g_delta_bank.shards[idx].name, idx);
    }
    return idx;
}

void enable_deltas(int enable) {
    g_delta_enabled = enable;
}

void print_signals(void) {
    printf("Signals:\n");
    printf("  arousal:   %.3f\n", g_signals.arousal);
    printf("  entropy:   %.3f\n", g_signals.entropy);
    printf("  tension:   %.3f\n", g_signals.tension);
    printf("  warmth:    %.3f\n", g_signals.warmth);
    printf("  focus:     %.3f\n", g_signals.focus);
    printf("  recursion: %.3f\n", g_signals.recursion_depth);
    printf("  resonance: %.3f\n", g_signals.resonance);
}

void print_mix(void) {
    printf("Shard mix:\n");
    for (int i = 0; i < g_delta_bank.n_shards; i++) {
        printf("  [%d] %s: %.3f\n", i, g_delta_bank.shards[i].name, g_delta_bank.mix[i]);
    }
}

void print_pulse(void) {
    printf("Pulse:\n");
    printf("  novelty:  %.3f\n", g_stanley_signals.pulse.novelty);
    printf("  arousal:  %.3f\n", g_stanley_signals.pulse.arousal);
    printf("  entropy:  %.3f\n", g_stanley_signals.pulse.entropy);
    printf("  valence:  %.3f (%.1s)\n", g_stanley_signals.pulse.valence,
           g_stanley_signals.pulse.valence > 0 ? "+" : g_stanley_signals.pulse.valence < 0 ? "-" : "0");
    printf("Stanley signals:\n");
    printf("  overthink_depth:  %d\n", g_stanley_signals.overthink_depth);
    printf("  body_tension:     %.3f\n", g_stanley_signals.body_tension);
    printf("  body_boredom:     %.3f\n", g_stanley_signals.body_boredom);
    printf("  active_expert:    %d (%s)\n", g_stanley_signals.active_expert,
           g_stanley_signals.active_expert == 0 ? "structural" :
           g_stanley_signals.active_expert == 1 ? "semantic" :
           g_stanley_signals.active_expert == 2 ? "creative" : "precise");
    printf("Overthinking:\n");
    printf("  repetition:   %.3f\n", g_overthink.repetition_score);
    printf("  abstraction:  %.3f\n", g_overthink.abstraction_score);
    printf("  self_ref:     %.3f\n", g_overthink.self_ref_score);
    if (g_stanley_signals.n_spiral > 0) {
        printf("  spirals:      ");
        for (int i = 0; i < g_stanley_signals.n_spiral; i++) {
            printf("%s ", g_stanley_signals.spiral_topics[i]);
        }
        printf("\n");
    }
}

#ifdef USE_GO_INNER_WORLD
void print_inner_world(void) {
    InnerWorldSnapshot snap;
    inner_world_get_snapshot(&snap);

    printf("Inner World (Go):\n");
    printf("  arousal:        %.3f\n", snap.arousal);
    printf("  valence:        %.3f\n", snap.valence);
    printf("  entropy:        %.3f\n", snap.entropy);
    printf("  coherence:      %.3f\n", snap.coherence);
    printf("  trauma_level:   %.3f%s\n", snap.trauma_level,
           snap.trauma_level > 0.5f ? " [HIGH]" : "");
    printf("  prophecy_debt:  %.3f\n", snap.prophecy_debt);
    printf("  destiny_pull:   %.3f\n", snap.destiny_pull);
    printf("  wormhole_chance:%.3f%s\n", snap.wormhole_chance,
           snap.wormhole_chance > 0.3f ? " [ACTIVE]" : "");
    printf("  loop_count:     %d%s\n", snap.loop_count,
           snap.loop_count > 3 ? " [SPIRALING]" : "");
    printf("  focus_strength: %.3f\n", snap.focus_strength);
    printf("  wander_pull:    %.3f\n", snap.wander_pull);
    printf("  memory_pressure:%.3f\n", snap.memory_pressure);

    // Get dominant emotion
    char emotion[64] = {0};
    inner_world_get_dominant_emotion(emotion, sizeof(emotion));
    if (emotion[0]) {
        printf("  dominant:       %s\n", emotion);
    }

    // Check overthinking spiral
    if (inner_world_is_spiraling()) {
        char suggestion[256] = {0};
        inner_world_suggest_break(suggestion, sizeof(suggestion));
        printf("  SPIRAL DETECTED: %s\n", suggestion[0] ? suggestion : "take a breath");
    }
}
#endif

void cleanup_dynamic(void) {
    // Auto-save MathBrain if it has learned anything
    if (g_mathbrain_enabled && g_mathbrain.history.total_computed > 0) {
        // Create weights directory if needed
        if (ensure_dir("weights") == 0 && save_mathbrain(&g_mathbrain, g_mathbrain_path) == 0) {
            printf("MathBrain: saved to %s (accuracy: %.1f%%, %d computations)\n",
                   g_mathbrain_path, g_mathbrain.history.accuracy_ema * 100.0f,
                   g_mathbrain.history.total_computed);
        }
    }

    free_delta_bank(&g_delta_bank);
    free_microtrainer(&g_trainer);
    free_attention_bias(&g_attention_bias);
    free_subjectivity(&g_subjectivity);
    free_cooccur_field(&g_cooccur);
    free_body_sense(&g_body_sense);
    free_selfsense(&g_selfsense);
    free_mathbrain(&g_mathbrain);
    if (g_train_state.pre_activations) free(g_train_state.pre_activations);
    if (g_train_state.post_activations) free(g_train_state.post_activations);

    // Shutdown Julia emotional engine
    if (g_julia_enabled) {
        julia_shutdown();
    }
}

// ============================================================
// Dialogue Mode: Arianna ↔ SARTRE with MetaArianna observing
// ============================================================

// Lazy load SARTRE on first /dialogue command
static int dialogue_init_sartre(void) {
    if (g_sartre_loaded) return 0;

    // Try multiple paths for SARTRE weights
    const char* weight_paths[] = {
        "weights/sartre/sartre.bin",
        "sartre/weights/sartre.bin",
        "../weights/sartre/sartre.bin",
        NULL
    };
    const char* tok_paths[] = {
        "weights/sartre/tokenizer.json",
        "sartre/weights/tokenizer.json",
        "../weights/sartre/tokenizer.json",
        NULL
    };
    const char* cfg_paths[] = {
        "weights/sartre/sartre_config.json",
        "sartre/weights/sartre_config.json",
        "../weights/sartre/sartre_config.json",
        NULL
    };

    const char* w = NULL, *tok = NULL, *cfg = NULL;
    for (int i = 0; weight_paths[i]; i++) {
        FILE* f = fopen(weight_paths[i], "rb");
        if (f) { fclose(f); w = weight_paths[i]; break; }
    }
    for (int i = 0; tok_paths[i]; i++) {
        FILE* f = fopen(tok_paths[i], "r");
        if (f) { fclose(f); tok = tok_paths[i]; break; }
    }
    for (int i = 0; cfg_paths[i]; i++) {
        FILE* f = fopen(cfg_paths[i], "r");
        if (f) { fclose(f); cfg = cfg_paths[i]; break; }
    }

    if (!w || !tok) {
        printf("[dialogue] Cannot find SARTRE weights or tokenizer\n");
        printf("[dialogue] Expected: weights/sartre/sartre.bin + tokenizer.json\n");
        return -1;
    }

    printf("[dialogue] Loading SARTRE (14.3M)...\n");
    if (sartre_transformer_init(&g_sartre, w, tok, cfg) != 0) {
        printf("[dialogue] Failed to initialize SARTRE\n");
        return -1;
    }

    g_sartre_loaded = 1;
    printf("[dialogue] SARTRE ready (dim=%d, layers=%d, vocab=%d)\n",
           g_sartre.config.dim, g_sartre.config.n_layers,
           g_sartre.config.vocab_size);
    return 0;
}

// Generate Arianna's turn into buffer using generate_dynamic pipeline
static int dialogue_generate_arianna(Transformer* t, const char* seed,
                                      char* output, int max_len,
                                      int max_tokens, float temperature) {
    // Tokenize seed
    int tokens[MAX_SEQ_LEN];
    int seed_len = strlen(seed);
    if (seed_len > MAX_SEQ_LEN) seed_len = MAX_SEQ_LEN;
    for (int i = 0; i < seed_len; i++) {
        tokens[i] = char_to_token(seed[i]);
    }
    int n_tokens = seed_len;

    // Route signals
    extract_signals(&g_signals, tokens, n_tokens, NULL);
    if (g_mood_enabled) {
        route_signals_to_moods(&g_mood_router, &g_signals);
        mood_to_shard_mix(&g_mood_router, &g_delta_bank);
    }

    // Process seed tokens
    for (int pos = 0; pos < n_tokens; pos++) {
        forward_dynamic(t, tokens, n_tokens, pos);
    }

    float effective_temp = temperature;
    if (g_mood_enabled) {
        effective_temp = adjust_temperature_by_mood(&g_mood_router, temperature);
    }
    if (g_meta_enabled && g_meta_thermogram.valid) {
        effective_temp += g_meta_thermogram.drift_direction * 0.1f;
        effective_temp += g_meta_thermogram.uncertainty * 0.05f;
        if (effective_temp < 0.1f) effective_temp = 0.1f;
        if (effective_temp > 2.0f) effective_temp = 2.0f;
    }

    // Generate tokens
    int gen = 0;
    for (int i = 0; i < max_tokens && n_tokens < MAX_SEQ_LEN && gen < max_len - 1; i++) {
        if (g_meta_enabled && g_meta_thermogram.valid) {
            meta_apply_thermogram(&g_meta_thermogram,
                                  t->state.logits, t->config.vocab_size);
        }

        int next = sample(t, effective_temp);
        tokens[n_tokens] = next;

        char c = token_to_char(next);
        output[gen++] = c;

        // Stop on Q&A boundary
        if (gen >= 3 && output[gen-1] == ':' && output[gen-2] == 'Q' && output[gen-3] == '\n') {
            gen -= 3;
            break;
        }

        forward_dynamic(t, tokens, n_tokens + 1, n_tokens);
        n_tokens++;
    }
    output[gen] = '\0';

    // Trim to last sentence end
    int last_end = -1;
    for (int j = 0; j < gen; j++) {
        if (output[j] == '.' || output[j] == '!' || output[j] == '?') {
            last_end = j;
        }
    }
    if (last_end >= 0 && last_end < gen - 1) {
        output[last_end + 1] = '\0';
        gen = last_end + 1;
    }

    return gen;
}

// Run Arianna ↔ SARTRE dialogue with MetaArianna observing
static void run_dialogue(Transformer* t, const char* seed,
                          int max_tokens __attribute__((unused)),
                          float temperature) {
    if (dialogue_init_sartre() != 0) return;

    // Dialogue log for MetaArianna observation
    char dialogue_log[4096];
    int log_len = 0;
    dialogue_log[0] = '\0';

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Arianna ↔ SARTRE Dialogue               ║\n");
    printf("║  MetaArianna observing                    ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // Format initial seed for Arianna
    char arianna_seed[512];
    snprintf(arianna_seed, sizeof(arianna_seed), "Q: %s\nA: ", seed);

    for (int turn = 0; turn < g_dialogue_max_turns; turn++) {
        printf("--- Turn %d/%d ---\n", turn + 1, g_dialogue_max_turns);

        // Phase 1: Arianna generates
        char arianna_output[1024];
        dialogue_generate_arianna(t, arianna_seed,
                                               arianna_output, sizeof(arianna_output),
                                               g_dialogue_tokens_per_turn, temperature);
        printf("[Arianna]: %s\n", arianna_output);

        // Append to dialogue log (clamp to available space)
        int avail = (int)sizeof(dialogue_log) - log_len;
        if (avail > 1) {
            int wrote = snprintf(dialogue_log + log_len, avail,
                                 "[A] %s\n", arianna_output);
            if (wrote > 0) log_len += (wrote < avail) ? wrote : avail - 1;
        }

        // Phase 2: SARTRE observes Arianna's output
        sartre_reset_state(&g_sartre);
        char sartre_prompt[1024];
        snprintf(sartre_prompt, sizeof(sartre_prompt),
                 "Q: %s\nA: ", arianna_output);

        int sp_len = strlen(sartre_prompt);
        sartre_feed_prompt(&g_sartre, sartre_prompt, sp_len);

        char sartre_output[1024];
        int s_len = sartre_generate(&g_sartre, sartre_output, sizeof(sartre_output),
                                     g_dialogue_tokens_per_turn, temperature, 0.9f, 1);
        // Trim trailing newline
        if (s_len > 0 && sartre_output[s_len - 1] == '\n') {
            sartre_output[--s_len] = '\0';
        }
        printf("[SARTRE]:  %s\n", sartre_output);

        // Append to dialogue log (clamp to available space)
        avail = (int)sizeof(dialogue_log) - log_len;
        if (avail > 1) {
            int wrote2 = snprintf(dialogue_log + log_len, avail,
                                  "[S] %s\n", sartre_output);
            if (wrote2 > 0) log_len += (wrote2 < avail) ? wrote2 : avail - 1;
        }

        // Phase 3: MetaArianna observes the dialogue
        if (g_meta_enabled) {
            // Cycle through templates per turn
            int template_id = turn % META_N_TEMPLATES;
            MetaTemplateParams meta_params;
            meta_default_params(&meta_params, template_id);

            int obs_start = (log_len > META_MAX_LOG_LEN)
                          ? log_len - META_MAX_LOG_LEN : 0;
            meta_observe(&g_fluid_transformer, &meta_params,
                         dialogue_log + obs_start, log_len - obs_start);
            g_meta_thermogram = g_fluid_transformer.result;
            g_meta_observation_count++;

            meta_push_history(&g_fluid_transformer,
                              g_meta_thermogram.warmth,
                              g_meta_thermogram.silence);

            const char* tpl_names[] = {"THERMO", "SILENCE", "DRIFT", "FIELD"};
            printf("[Meta:%s] w=%.3f s=%.3f si=%.3f d=%.3f(%+d)\n",
                   tpl_names[template_id],
                   g_meta_thermogram.warmth, g_meta_thermogram.sharpness,
                   g_meta_thermogram.silence,
                   g_meta_thermogram.drift_rate, g_meta_thermogram.drift_direction);

            meta_reset(&g_fluid_transformer);

            // Temperature feedback from thermogram
            temperature += g_meta_thermogram.drift_direction * 0.05f;
            temperature += g_meta_thermogram.uncertainty * 0.03f;
            if (temperature < 0.2f) temperature = 0.2f;
            if (temperature > 1.5f) temperature = 1.5f;
        }

        printf("\n");

        // SARTRE output becomes seed for next Arianna turn
        if (s_len > 0) {
            snprintf(arianna_seed, sizeof(arianna_seed),
                     "Q: %s\nA: ", sartre_output);
        }
    }

    printf("╔══════════════════════════════════════════╗\n");
    printf("║  Dialogue complete (%d turns)              ║\n", g_dialogue_max_turns);
    if (g_meta_enabled) {
        printf("║  MetaArianna: %d observations             ║\n", g_meta_observation_count);
    }
    printf("╚══════════════════════════════════════════╝\n");
}

// ============================================================
// REPL Mode - Interactive terminal interface
// ============================================================

void run_repl(Transformer* t, int max_tokens, float temperature) {
    char input[512];
    int session_turns = 0;

    printf("\n=== arianna.c REPL ===\n");
    printf("weights frozen // voice crystallized\n");
    printf("type 'quit' or 'exit' to leave\n");
    printf("type 'signals' to see internal state\n");
    printf("type 'body' to see somatic state\n");
    printf("type 'self' to see SelfSense signals\n");
    printf("\n");

    while (1) {
        printf("> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("\n[EOF - exiting]\n");
            break;
        }

        // Remove trailing newline
        int len = strlen(input);
        if (len > 0 && input[len-1] == '\n') {
            input[len-1] = '\0';
            len--;
        }

        // Skip empty input
        if (len == 0) {
            continue;
        }

        // Check for commands
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("[exiting - %d turns, SelfSense learned from %d observations]\n",
                   session_turns, g_selfsense.observations);
            break;
        }

        if (strcmp(input, "signals") == 0) {
            print_signals();
            if (g_delta_bank.n_shards > 0) {
                print_mix();
            }
            if (g_mood_enabled) {
                print_mood_state(&g_mood_router);
            }
            continue;
        }

        if (strcmp(input, "body") == 0) {
            print_body_sense_debug();
            continue;
        }

        if (strcmp(input, "self") == 0) {
            print_selfsense_signals(&g_selfsense);
            print_selfsense_stats(&g_selfsense);
            continue;
        }

        if (strcmp(input, "subj") == 0) {
            print_subjectivity_debug();
            continue;
        }

        if (strcmp(input, "cooccur") == 0) {
            print_cooccur_debug();
            continue;
        }

        // /dialogue [seed] — Arianna ↔ SARTRE dialogue with MetaArianna
        if (strncmp(input, "/dialogue ", 10) == 0 || strncmp(input, "/talk ", 6) == 0) {
            const char* seed = (input[1] == 'd') ? input + 10 : input + 6;
            if (strlen(seed) > 0) {
                run_dialogue(t, seed, max_tokens, temperature);
            } else {
                printf("Usage: /dialogue <seed phrase>\n");
            }
            continue;
        }

        if (strcmp(input, "/dialogue") == 0 || strcmp(input, "/talk") == 0) {
            run_dialogue(t, "What do you feel?", max_tokens, temperature);
            continue;
        }

        // /dialogue-turns N — set number of dialogue turns
        if (strncmp(input, "/dialogue-turns ", 16) == 0) {
            int n = atoi(input + 16);
            if (n >= 1 && n <= 20) {
                g_dialogue_max_turns = n;
                printf("[dialogue] turns set to %d\n", n);
            } else {
                printf("Usage: /dialogue-turns <1-20>\n");
            }
            continue;
        }

        // /sartre — SARTRE status
        if (strcmp(input, "/sartre") == 0) {
            if (g_sartre_loaded) {
                printf("SARTRE: loaded (dim=%d, layers=%d, heads=%d, kv=%d, vocab=%d)\n",
                       g_sartre.config.dim, g_sartre.config.n_layers,
                       g_sartre.config.n_heads, g_sartre.config.n_kv_heads,
                       g_sartre.config.vocab_size);
                printf("  GQA: %d Q heads, %d KV heads (groups=%d)\n",
                       g_sartre.config.n_heads, g_sartre.config.n_kv_heads,
                       g_sartre.config.n_kv_groups);
                printf("  Dialogue: %d turns, %d tokens/turn\n",
                       g_dialogue_max_turns, g_dialogue_tokens_per_turn);
            } else {
                printf("SARTRE: not loaded (use /dialogue to activate)\n");
            }
            continue;
        }

        // /pandora <prompt> — GPT2-30M vocabulary extraction
        if (strncmp(input, "/pandora ", 9) == 0) {
            const char* prompt = input + 9;
            if (strlen(prompt) > 0) {
                g_pandora_enabled = 1;
                pandora_set_active(&g_pandora, 1);
                int n = pandora_steal_from_brain(&g_pandora, prompt);
                if (n > 0) {
                    printf("[pandora] Extracted %d n-grams from GPT2-30M\n", n);
                    printf("[pandora] Total vocabulary: %d n-grams\n", g_pandora.n_ngrams);
                }
            } else {
                printf("Usage: /pandora <prompt>\n");
                printf("Example: /pandora What is consciousness?\n");
            }
            continue;
        }

        // /pandora-gguf <prompt> — TinyLlama 1.1B GGUF (auto-downloads if needed)
        if (strncmp(input, "/pandora-gguf ", 14) == 0) {
            const char* prompt = input + 14;
            if (strlen(prompt) > 0) {
                g_pandora_enabled = 1;
                pandora_set_active(&g_pandora, 1);
                printf("[pandora-gguf] Using TinyLlama 1.1B (first run downloads ~700MB)...\n");
                int n = pandora_steal_from(&g_pandora, BRAIN_TINYLLAMA, prompt);
                if (n > 0) {
                    printf("[pandora-gguf] Extracted %d n-grams from TinyLlama-1.1B\n", n);
                    printf("[pandora-gguf] Total vocabulary: %d n-grams\n", g_pandora.n_ngrams);
                }
            } else {
                printf("Usage: /pandora-gguf <prompt>\n");
                printf("Example: /pandora-gguf What is the meaning of life?\n");
            }
            continue;
        }

        // /pandora-torch <prompt> — GPT2-distill
        if (strncmp(input, "/pandora-torch ", 15) == 0) {
            const char* prompt = input + 15;
            if (strlen(prompt) > 0) {
                g_pandora_enabled = 1;
                pandora_set_active(&g_pandora, 1);
                printf("[pandora-torch] Using GPT2-distill...\n");
                int n = pandora_steal_from(&g_pandora, BRAIN_GPT2_DISTILL, prompt);
                if (n > 0) {
                    printf("[pandora-torch] Extracted %d n-grams from GPT2-distill\n", n);
                    printf("[pandora-torch] Total vocabulary: %d n-grams\n", g_pandora.n_ngrams);
                }
            } else {
                printf("Usage: /pandora-torch <prompt>\n");
                printf("Example: /pandora-torch What is the meaning of life?\n");
            }
            continue;
        }

        // HyperPandora commands
        if (strcmp(input, "/hyper") == 0) {
            printf("[HyperPandora] Auto-selection enabled\n");
            printf("Will choose best available brain based on SARTRE state.\n");
            g_pandora_enabled = 1;
            pandora_set_active(&g_pandora, 1);
            continue;
        }

        if (strcmp(input, "/hyper-off") == 0) {
            printf("[HyperPandora] All external brains disabled\n");
            g_pandora_enabled = 0;
            pandora_set_active(&g_pandora, 0);
            continue;
        }

        if (strcmp(input, "pandora") == 0 || strcmp(input, "/pandora") == 0) {
            printf("Pandora status: %s\n", g_pandora_enabled ? "ACTIVE" : "READY");
            printf("  N-grams: %d / %d\n", g_pandora.n_ngrams, PANDORA_MAX_NGRAMS);
            printf("  Injection strength: %.2f\n", g_pandora.injection_strength);
            printf("\nPackages:\n");
            printf("  /pandora <prompt>       — GPT2-30M (fast, ~100MB)\n");
            printf("  /pandora-torch <prompt> — GPT2-distill (PyTorch)\n");
            printf("  /pandora-gguf <prompt>  — TinyLlama 1.1B (~700MB)\n");
            printf("\nHyperPandora:\n");
            printf("  /hyper     — enable auto-selection\n");
            printf("  /hyper-off — disable all external brains\n");
            printf("\nControl:\n");
            printf("  pandoraon  — enable injection into generation\n");
            printf("  pandoraoff — disable (pure Arianna voice)\n");
            continue;
        }

        if (strcmp(input, "pandoraon") == 0) {
            g_pandora_enabled = 1;
            pandora_set_active(&g_pandora, 1);
            printf("[Pandora ENABLED]\n");
            continue;
        }

        if (strcmp(input, "pandoraoff") == 0) {
            g_pandora_enabled = 0;
            pandora_set_active(&g_pandora, 0);
            printf("[Pandora DISABLED - pure voice]\n");
            continue;
        }

        if (strcmp(input, "help") == 0) {
            printf("Commands:\n");
            printf("  signals  - show signal values\n");
            printf("  body     - show somatic state (boredom, overwhelm, stuck)\n");
            printf("  self     - show SelfSense signals from hidden states\n");
            printf("  subj     - show subjectivity state\n");
            printf("  cooccur  - show co-occurrence stats\n");
            printf("  math     - show MathBrain stats\n");
            printf("  mathsave - save MathBrain state now\n");
            printf("  pandora  - show Pandora packages status\n");
            printf("  /pandora <text>       - GPT2-30M vocabulary\n");
            printf("  /pandora-torch <text> - GPT2-distill vocabulary\n");
            printf("  /pandora-gguf <text>  - TinyLlama 1.1B vocabulary\n");
            printf("  /hyper                - HyperPandora auto-select\n");
            printf("\nDialogue (Arianna ↔ SARTRE):\n");
            printf("  /dialogue <seed>      - Start dialogue (lazy-loads SARTRE)\n");
            printf("  /talk <seed>          - Alias for /dialogue\n");
            printf("  /dialogue-turns <N>   - Set turns per dialogue (1-20)\n");
            printf("  /sartre               - Show SARTRE status\n");
            printf("\n");
            printf("  learn    - start learning (creates experience shard)\n");
            printf("  save     - save learned experience to shard file\n");
            printf("  quit     - exit REPL (auto-saves MathBrain)\n");
            printf("\nMath: type '7 + 5' to compute (learns from feedback)\n");
            printf("MathBrain persists across sessions in weights/mathbrain.bin\n");
            printf("Anything else is treated as input for generation.\n");
            continue;
        }

        // MathBrain: detect and compute arithmetic expressions
        if (g_mathbrain_enabled) {
            int a, b;
            MathOp op;
            if (parse_math_expr(input, &a, &op, &b)) {
                char result[32];
                int correct = compute_from_text(&g_mathbrain, input, result, 32);

                // Compute ground truth for display
                int truth;
                switch (op) {
                    case OP_ADD: truth = a + b; break;
                    case OP_SUB: truth = a - b; break;
                    case OP_MUL: truth = a * b; break;
                    case OP_DIV: truth = (b != 0) ? a / b : 0; break;
                    default: truth = 0;
                }

                const char* ops[] = {"+", "-", "*", "/"};
                printf("%d %s %d = %s", a, ops[op], b, result);
                if (correct) {
                    printf(" [resonance: correct]\n");
                } else {
                    printf(" [truth: %d, learning...]\n", truth);
                }
                printf("MathBrain accuracy: %.1f%%\n",
                       get_recent_accuracy(&g_mathbrain, 20) * 100.0f);
                continue;
            }
        }

        if (strcmp(input, "math") == 0) {
            print_mathbrain_stats(&g_mathbrain);
            continue;
        }

        if (strcmp(input, "mathsave") == 0) {
            if (ensure_dir("weights") != 0) {
                printf("[Error creating weights directory]\n");
            } else if (save_mathbrain(&g_mathbrain, g_mathbrain_path) == 0) {
                printf("[MathBrain saved to %s]\n", g_mathbrain_path);
            } else {
                printf("[Error saving MathBrain]\n");
            }
            continue;
        }

        if (strcmp(input, "learn") == 0) {
            if (g_microtraining) {
                printf("[Already learning - shard: %s]\n",
                       g_active_shard ? g_active_shard->name : "unnamed");
            } else {
                // Create a timestamped shard name
                time_t now = time(NULL);
                struct tm* tm_info = localtime(&now);
                char shard_name[64];
                strftime(shard_name, 64, "session_%Y%m%d_%H%M%S", tm_info);

                create_learning_shard(shard_name, t->config.n_layers, t->config.dim);
                enable_microtraining(1);
                activate_learning_shard(0.1f);

                // Initialize quantum accumulator
                init_accumulator(&g_accumulator, t->config.dim, t->config.vocab_size);
                g_accumulator_initialized = 1;

                printf("[Learning started - shard: %s]\n", shard_name);
                printf("[Quantum accumulation enabled - training triggers at critical mass]\n");
            }
            continue;
        }

        if (strncmp(input, "save", 4) == 0) {
            if (!g_microtraining || g_active_shard == NULL) {
                printf("[No active learning shard. Use 'learn' first.]\n");
            } else {
                // Extract path if provided, otherwise use default
                char save_path[256];
                if (strlen(input) > 5 && input[4] == ' ') {
                    snprintf(save_path, 256, "%s", input + 5);
                } else {
                    snprintf(save_path, 256, "shards/%s.bin", g_active_shard->name);
                }

                // Create shards directory if needed
                if (ensure_dir("shards") != 0) {
                    printf("[Error creating shards directory]\n");
                } else {
                    float norm = get_delta_norm(&g_active_shard->attn_q_deltas[0]);
                    if (save_learning_shard(save_path) == 0) {
                        printf("[Saved to %s]\n", save_path);
                        printf("[Delta norm: %.4f, observations: %d]\n",
                               norm, g_selfsense.observations);
                    } else {
                        printf("[Error saving shard]\n");
                    }
                }
            }
            continue;
        }

        // Generate response
        printf("\n");
        if (g_subjectivity_enabled) {
            generate_subjective(t, input, max_tokens, temperature);
        } else {
            generate_dynamic(t, input, max_tokens, temperature);
        }
        printf("\n");

        session_turns++;
    }
}

// ============================================================
// Main with dynamic support
// ============================================================

// Default origin file paths to try
static const char* DEFAULT_ORIGIN_PATHS[] = {
    "bin/origin.txt",
    "./bin/origin.txt",
    "../bin/origin.txt",
    "origin.txt",          // fallback: same directory
    "./origin.txt",
    NULL
};

// Try to find origin.txt in default locations
static const char* find_default_origin(void) {
    for (int i = 0; DEFAULT_ORIGIN_PATHS[i]; i++) {
        FILE* f = fopen(DEFAULT_ORIGIN_PATHS[i], "r");
        if (f) {
            fclose(f);
            return DEFAULT_ORIGIN_PATHS[i];
        }
    }
    return NULL;
}

void print_usage(const char* prog) {
    printf("arianna_dynamic - Personality transformer with Stanley-style deltas\n\n");
    printf("Usage: %s <weights.bin> <tokenizer.json> \"<prompt>\" [max_tokens] [temperature]\n", prog);
    printf("       %s <weights.bin> <tokenizer.json> --repl [max_tokens] [temperature]\n", prog);
    printf("\nOptions:\n");
    printf("  --repl          Interactive REPL mode (state persists between turns)\n");
    printf("  -shard <path>   Load experience shard (can use multiple times)\n");
    printf("  -no-mood        Disable mood routing (enabled by default)\n");
    printf("  -guided         Enable guided attention (gravity centers, pulse)\n");
    printf("  -subj <origin>  Use custom origin file (default: bin/origin.txt)\n");
    printf("  -no-subj        Disable subjectivity (use prompt as seed)\n");
    printf("  -no-learn       Disable microtraining (default: ON with shard 'live')\n");
    printf("  -signals        Print signal values after generation\n");
    printf("  -learn <name>   Create new learning shard with name\n");
    printf("  -save <path>    Save learning shard after generation\n");
    printf("  -momentum <f>   Mood transition momentum (0.0-1.0, default 0.8)\n");
    printf("  -julia          Enable Julia emotional gradient engine (tertiary nuances)\n");
    printf("  -inner          Enable Inner Arianna (MetaVoice борьба)\n");
    printf("  -borba <mode>   Set борьба mode: emotional, chaos, trauma, stuck, blend\n");
    printf("  -inner-w <f>    Inner voice base weight 0.0-1.0 (default 0.15)\n");
    printf("  -inner-th <f>   Breakthrough threshold 0.0-1.0 (default 0.6)\n");
    printf("\nAMK (Arianna Method Kernel):\n");
    printf("  -velocity <v>   Movement mode: nomove, walk, run, backward\n");
    printf("  -destiny <f>    Bias toward probable path 0.0-1.0 (default 0.35)\n");
    printf("  -wormhole <f>   Creative skip probability 0.0-1.0 (default 0.02)\n");
    printf("  -prophecy <n>   Lookahead depth 1-64 (default 7)\n");
    printf("  -amk <script>   Execute AMK DSL script (e.g. \"VELOCITY RUN; DESTINY 0.5\")\n");
    printf("\nExamples:\n");
    printf("  %s arianna.bin \"Who are you?\" 100 0.8\n", prog);
    printf("  %s arianna.bin --repl 150 0.9\n", prog);
    printf("  %s arianna.bin \"Tell me about presence\" 100 0.8 -signals\n", prog);
    printf("  %s arianna.bin -guided \"What do you feel?\" 100 0.8\n", prog);
    printf("  %s arianna.bin -no-subj -no-mood \"She finds that \" 100 0.8\n", prog);
    printf("\nDefaults (Arianna's core architecture):\n");
    printf("  Subjectivity: ON  - generates from identity, not from prompt\n");
    printf("  Mood routing: ON  - 8 moods shape attention dynamically\n");
    printf("  SelfSense:    ON  - signals extracted from hidden states\n");
    printf("  BodySense:    ON  - somatic regulation (boredom, overwhelm, stuck)\n");
    printf("  CooccurField: ON  - corpus patterns bias generation\n");
    printf("\n  Your words create a wrinkle in her field, not a seed.\n");
}

int main(int argc, char** argv) {
    // Disable stdout buffering for immediate output
    setbuf(stdout, NULL);

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    char* weights_path = argv[1];
    char* tokenizer_path = argv[2];
    char* prompt = NULL;
    char* shard_paths[MAX_SHARDS];
    int n_shard_paths = 0;
    char* learn_name = "live";  // Microtraining ON by default (use -no-learn to disable)
    char* save_path = NULL;
    int max_tokens = 250;  // Character-level tokenizer: ~250 chars = 35-40 words
    int max_tokens_set = 0;  // flag to track if user provided max_tokens
    float temperature = 0.8f;
    float momentum = 0.8f;
    int print_sigs = 0;
    int mood_mode = 1;    // ENABLED BY DEFAULT - mood shapes attention
    int guided_mode = 0;
    int subj_mode = 1;    // ENABLED BY DEFAULT - this is Arianna's core
    int repl_mode = 0;    // Interactive REPL mode
    int julia_mode = 0;   // Julia emotional gradient engine
    char* origin_path = NULL;
    int inner_mode = 0;  // Inner Arianna (MetaVoice борьба)

    // AMK parameters (applied after dsl_init)
    int amk_velocity = -99;  // -99 = not set (note: AM_VEL_BACKWARD is -1)
    float amk_destiny = -1.0f;
    float amk_wormhole = -1.0f;
    int amk_prophecy = -1;
    char* amk_script = NULL;

    // Initialize inner_arianna early so -borba/-inner-w/-inner-t can set params
    inner_init(&g_inner_arianna);

    // Parse arguments (start from 3, after weights and tokenizer)
    int arg_idx = 3;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "--repl") == 0) {
            repl_mode = 1;
        } else if (strcmp(argv[arg_idx], "-subj") == 0 && arg_idx + 1 < argc) {
            subj_mode = 1;
            origin_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-no-subj") == 0) {
            subj_mode = 0;  // Disable subjectivity, use prompt as seed
        } else if (strcmp(argv[arg_idx], "-no-learn") == 0) {
            learn_name = NULL;  // Disable microtraining (default is ON)
        } else if (strcmp(argv[arg_idx], "-guided") == 0) {
            guided_mode = 1;
        } else if (strcmp(argv[arg_idx], "-shard") == 0 && arg_idx + 1 < argc) {
            if (n_shard_paths < MAX_SHARDS) {
                shard_paths[n_shard_paths++] = argv[++arg_idx];
            }
        } else if (strcmp(argv[arg_idx], "-mood") == 0) {
            mood_mode = 1;
        } else if (strcmp(argv[arg_idx], "-no-mood") == 0) {
            mood_mode = 0;
        } else if (strcmp(argv[arg_idx], "-signals") == 0) {
            print_sigs = 1;
#ifdef USE_GO_INNER_WORLD
        } else if (strcmp(argv[arg_idx], "-async") == 0) {
            g_inner_world_async = 1;
            printf("Inner World: async mode (background goroutines)\n");
#endif
        } else if (strcmp(argv[arg_idx], "-learn") == 0 && arg_idx + 1 < argc) {
            learn_name = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-save") == 0 && arg_idx + 1 < argc) {
            save_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-momentum") == 0 && arg_idx + 1 < argc) {
            momentum = atof(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-julia") == 0) {
            julia_mode = 1;
        } else if (strcmp(argv[arg_idx], "-inner") == 0) {
            inner_mode = 1;
        } else if (strcmp(argv[arg_idx], "-borba") == 0 && arg_idx + 1 < argc) {
            arg_idx++;
            if (strcmp(argv[arg_idx], "emotional") == 0) inner_set_mode(&g_inner_arianna, BORBA_MODE_EMOTIONAL);
            else if (strcmp(argv[arg_idx], "chaos") == 0) inner_set_mode(&g_inner_arianna, BORBA_MODE_CHAOS);
            else if (strcmp(argv[arg_idx], "trauma") == 0) inner_set_mode(&g_inner_arianna, BORBA_MODE_TRAUMA);
            else if (strcmp(argv[arg_idx], "stuck") == 0) inner_set_mode(&g_inner_arianna, BORBA_MODE_STUCK);
            else if (strcmp(argv[arg_idx], "blend") == 0) inner_set_mode(&g_inner_arianna, BORBA_MODE_BLEND);
        } else if (strcmp(argv[arg_idx], "-inner-w") == 0 && arg_idx + 1 < argc) {
            inner_set_base_weight(&g_inner_arianna, atof(argv[++arg_idx]));
        } else if (strcmp(argv[arg_idx], "-inner-th") == 0 && arg_idx + 1 < argc) {
            inner_set_threshold(&g_inner_arianna, atof(argv[++arg_idx]));
        // AMK — Arianna Method Kernel commands (saved, applied after dsl_init)
        } else if (strcmp(argv[arg_idx], "-velocity") == 0 && arg_idx + 1 < argc) {
            arg_idx++;
            if (strcmp(argv[arg_idx], "nomove") == 0) amk_velocity = AM_VEL_NOMOVE;
            else if (strcmp(argv[arg_idx], "walk") == 0) amk_velocity = AM_VEL_WALK;
            else if (strcmp(argv[arg_idx], "run") == 0) amk_velocity = AM_VEL_RUN;
            else if (strcmp(argv[arg_idx], "backward") == 0) amk_velocity = AM_VEL_BACKWARD;
        } else if (strcmp(argv[arg_idx], "-destiny") == 0 && arg_idx + 1 < argc) {
            amk_destiny = atof(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-wormhole") == 0 && arg_idx + 1 < argc) {
            amk_wormhole = atof(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-prophecy") == 0 && arg_idx + 1 < argc) {
            amk_prophecy = atoi(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-amk") == 0 && arg_idx + 1 < argc) {
            amk_script = argv[++arg_idx];
        } else if (prompt == NULL && !repl_mode) {
            // In non-REPL mode, first positional arg is prompt
            prompt = argv[arg_idx];
        } else if (max_tokens_set == 0) {
            // Next positional arg is max_tokens (works for both REPL and non-REPL)
            max_tokens = atoi(argv[arg_idx]);
            max_tokens_set = 1;
        } else {
            // Last positional arg is temperature
            temperature = atof(argv[arg_idx]);
        }
        arg_idx++;
    }

    if (prompt == NULL && !repl_mode) {
        print_usage(argv[0]);
        return 1;
    }

    srand(time(NULL));

    // Load tokenizer first
    if (load_tokenizer(tokenizer_path) != 0) {
        fprintf(stderr, "Error: couldn't load tokenizer from %s\n", tokenizer_path);
        return 1;
    }

    // Load model
    Transformer t;
    if (load_weights(&t, weights_path) != 0) {
        fprintf(stderr, "Error: couldn't load weights from %s\n", weights_path);
        return 1;
    }

    printf("Loaded: dim=%d, layers=%d, heads=%d, vocab=%d\n",
           t.config.dim, t.config.n_layers, t.config.n_heads, t.config.vocab_size);

    // Initialize dynamic system
    init_dynamic(t.config.dim, t.config.vocab_size);
    set_mood_momentum(momentum);

    // Initialize Go inner_world (if compiled with -DUSE_GO_INNER_WORLD)
#ifdef USE_GO_INNER_WORLD
    inner_world_init();
    printf("Inner World (Go): enabled\n");
    printf("  Processes: trauma, emotional_drift, overthinking, memory, attention, prophecy\n");
#endif

    // Initialize Julia emotional gradient engine (if requested)
    if (julia_mode) {
        if (julia_init()) {
            g_julia_enabled = 1;
            // Initialize emotional state with Arianna's baseline
            // joy, trust, fear, surprise, sadness, disgust, anger, anticipation,
            // resonance, presence, longing, wonder
            g_julia_emotional_vec[0] = 0.2f;   // joy (gentle warmth)
            g_julia_emotional_vec[1] = 0.3f;   // trust
            g_julia_emotional_vec[2] = 0.05f;  // fear (low)
            g_julia_emotional_vec[3] = 0.1f;   // surprise
            g_julia_emotional_vec[4] = 0.05f;  // sadness (low)
            g_julia_emotional_vec[5] = 0.0f;   // disgust
            g_julia_emotional_vec[6] = 0.0f;   // anger
            g_julia_emotional_vec[7] = 0.15f;  // anticipation
            g_julia_emotional_vec[8] = 0.2f;   // resonance
            g_julia_emotional_vec[9] = 0.6f;   // presence (Arianna is present)
            g_julia_emotional_vec[10] = 0.1f;  // longing
            g_julia_emotional_vec[11] = 0.15f; // wonder
            printf("Julia (emotional gradients): enabled\n");
            printf("  12D primary + tertiary nuances (bittersweetness, nostalgia...)\n");
        } else {
            printf("Julia: not available (install Julia + JSON3 to enable)\n");
        }
    }

    // Initialize Cloud (pre-semantic emotion detection)
    // Calls Go library via cloud_wrapper.c
    cloud_init();

    // Initialize Schumann resonance (Earth's heartbeat)
    // Always enabled — this is cosmic input, not optional
    schumann_init();
    g_schumann_enabled = 1;
    printf("Schumann (7.83 Hz): enabled\n");
    printf("  Earth-ionosphere resonance modulates healing/coherence\n");

    // Initialize AMK — Arianna Method Kernel
    // Always enabled — this is the physics of the field
    dsl_init();  // Initializes both am_init() and schumann_init() (safe to call twice)
    g_amk_enabled = 1;

    // Apply AMK parameters from command line (after init)
    // Use am_exec with DSL commands so effective_temp gets recalculated
    if (amk_velocity != -99) {
        const char* vel_cmd = amk_velocity == AM_VEL_RUN ? "VELOCITY RUN" :
                              amk_velocity == AM_VEL_NOMOVE ? "VELOCITY NOMOVE" :
                              amk_velocity == AM_VEL_BACKWARD ? "VELOCITY BACKWARD" : "VELOCITY WALK";
        am_exec(vel_cmd);
    }
    AM_State* amk = am_get_state();
    if (amk_destiny >= 0.0f) amk->destiny = amk_destiny;
    if (amk_wormhole >= 0.0f) amk->wormhole = amk_wormhole;
    if (amk_prophecy >= 0) amk->prophecy = amk_prophecy;
    if (amk_script != NULL) am_exec(amk_script);

    g_dsl_config = dsl_build_config();
    printf("AMK (Arianna Method): enabled\n");
    printf("  Prophecy: %d, Destiny: %.2f, Wormhole: %.2f\n",
           amk->prophecy, amk->destiny, amk->wormhole);
    printf("  Velocity: %s, Temperature: %.2f\n",
           amk->velocity_mode == AM_VEL_WALK ? "walk" :
           amk->velocity_mode == AM_VEL_RUN ? "run" :
           amk->velocity_mode == AM_VEL_NOMOVE ? "nomove" : "backward",
           amk->effective_temp);

    // Initialize Pandora (vocabulary release from External Brain)
    pandora_init(&g_pandora);
    g_pandora_enabled = 0;  // OFF by default - activate with steal/stealtiny
    printf("Pandora (vocabulary): ready (use 'steal' or 'stealtiny' to activate)\n");
    printf("  \"Take the words, leave the voice\"\n");

    // Enable Inner Arianna (MetaVoice: борьба)
    // Inner voice breaks through based on emotional state (Cloud, Mood, Body, Trauma)
    // Note: inner_init() already called before arg parsing
    if (inner_mode) {
        g_inner_enabled = 1;
        printf("Inner Arianna (борьба): enabled\n");
        printf("  Inner breaks through on emotional activation\n");
        printf("  Base weight: %.2f, threshold: %.2f\n",
               g_inner_arianna.base_weight,
               g_inner_arianna.breakthrough_threshold);
        printf("  Mode: %s\n",
               g_inner_arianna.borba_mode == BORBA_MODE_EMOTIONAL ? "emotional" :
               g_inner_arianna.borba_mode == BORBA_MODE_CHAOS ? "chaos" :
               g_inner_arianna.borba_mode == BORBA_MODE_TRAUMA ? "trauma" :
               g_inner_arianna.borba_mode == BORBA_MODE_STUCK ? "stuck" : "blend");
    }

    // Initialize MetaArianna (FluidTransformer meta-observer)
    // Uses 20M weights — same architecture, different personality grain
    {
        const char* meta_weights = "weights/arianna_20m.bin";
        const char* meta_tokenizer = "weights/tokenizer_unified.json";
        struct stat st;
        if (stat(meta_weights, &st) == 0 && stat(meta_tokenizer, &st) == 0) {
            if (meta_init(&g_fluid_transformer, meta_weights, meta_tokenizer) == 0) {
                g_meta_enabled = 1;
                memset(&g_meta_thermogram, 0, sizeof(g_meta_thermogram));
                printf("MetaArianna (observer): enabled (20M, %d vocab)\n",
                       g_fluid_transformer.obs_vocab_size);
                printf("  \"Inhale -> observe -> exhale. Breathing.\"\n");
#ifdef USE_GO_INNER_WORLD
                meta_router_init();
                printf("  Meta Router (Go): initialized\n");
#endif
            } else {
                fprintf(stderr, "[meta] init failed, observer disabled\n");
            }
        } else {
            fprintf(stderr, "[meta] weights not found (%s), observer disabled\n", meta_weights);
        }
    }

    // Load shards
    for (int i = 0; i < n_shard_paths; i++) {
        if (load_experience(shard_paths[i], t.config.n_layers, t.config.dim) < 0) {
            fprintf(stderr, "Warning: couldn't load shard from %s\n", shard_paths[i]);
        }
    }

    // Enable mood routing if requested
    if (mood_mode) {
        enable_mood_routing(1);
        printf("Mood routing: enabled (momentum=%.2f)\n", momentum);
    }

    // Enable guided attention if requested
    if (guided_mode) {
        setup_arianna_identity();  // Load default personality anchors
        enable_guided_attention(1);
        printf("Guided attention: enabled (%d gravity centers, %d warm, %d cold)\n",
               g_identity.n_gravity, g_identity.n_warm, g_identity.n_cold);
    }

    // Create learning shard if requested
    if (learn_name != NULL) {
        create_learning_shard(learn_name, t.config.n_layers, t.config.dim);
        enable_microtraining(1);
        activate_learning_shard(0.1f);

        // Initialize quantum accumulator (Stanley-style batched learning)
        init_accumulator(&g_accumulator, t.config.dim, t.config.vocab_size);
        g_accumulator_initialized = 1;
        printf("Microtraining: enabled (quantum accumulation, thresh: %.0f bytes, %.1f res, %.1f nov)\n",
               g_accumulator.bytes_threshold, g_accumulator.resonance_threshold, g_accumulator.novelty_threshold);
    }

    // Enable subjectivity (default: ON)
    if (subj_mode) {
        // Auto-detect origin.txt if not specified
        if (origin_path == NULL) {
            origin_path = (char*)find_default_origin();
        }

        if (origin_path != NULL && load_subjectivity_origin(origin_path)) {
            printf("Subjectivity: enabled (no-seed-from-prompt)\n");
            printf("  Origin: %s\n", origin_path);
            printf("  Identity: %d fragments, %d trigrams, %d lexicon\n",
                   g_subjectivity.identity.n_fragments,
                   g_subjectivity.identity.n_trigrams,
                   g_subjectivity.identity.lexicon_size);

            // Also load co-occurrence from origin (default: ON)
            if (load_cooccur_corpus(origin_path)) {
                printf("CooccurField: enabled (corpus patterns bias generation)\n");
                printf("  Tokens observed: %llu, alpha: %.2f\n",
                       (unsigned long long)g_cooccur.tokens_observed, g_cooccur_alpha);
            }

            // Setup SelfSense identity from origin tokens
            if (g_selfsense_enabled) {
                // Tokenize origin text for identity embedding
                FILE* f = fopen(origin_path, "r");
                if (f) {
                    char buf[4096];
                    int len = fread(buf, 1, 4095, f);
                    buf[len] = '\0';
                    fclose(f);

                    int tokens[1024];
                    int n_tok = 0;
                    for (int i = 0; i < len && n_tok < 1024; i++) {
                        tokens[n_tok++] = (unsigned char)buf[i];
                    }
                    selfsense_compute_identity_from_tokens(&g_selfsense, &t, tokens, n_tok);
                    printf("SelfSense: enabled (learned signals from hidden states)\n");
                }
            }
        } else {
            if (origin_path != NULL) {
                fprintf(stderr, "Warning: couldn't load origin from %s\n", origin_path);
            } else {
                fprintf(stderr, "Warning: no bin/origin.txt found, subjectivity disabled\n");
            }
            fprintf(stderr, "Falling back to prompt-as-seed mode (-no-subj)\n");
            subj_mode = 0;
        }
    }

    // Generate (REPL, subjective, or dynamic mode)
    if (repl_mode) {
        run_repl(&t, max_tokens, temperature);
    } else if (subj_mode && g_subjectivity_enabled) {
        printf("\n[User input: \"%s\"]\n", prompt);
        generate_subjective(&t, prompt, max_tokens, temperature);
    } else {
        printf("\n--- Generation ---\n");
        generate_dynamic(&t, prompt, max_tokens, temperature);
    }

    // Print state (skip in REPL mode - user can use commands)
    if (print_sigs && !repl_mode) {
        printf("\n");
        print_signals();
        if (g_delta_bank.n_shards > 0) {
            print_mix();
        }
        if (mood_mode) {
            print_mood_state(&g_mood_router);
        }
        if (guided_mode) {
            print_pulse();
        }
        if (subj_mode) {
            print_subjectivity_debug();
        }
        if (g_body_sense_enabled) {
            print_body_sense_debug();
        }
        if (g_selfsense_enabled) {
            print_selfsense_signals(&g_selfsense);
            print_selfsense_stats(&g_selfsense);
        }
#ifdef USE_GO_INNER_WORLD
        print_inner_world();
#endif
    }

    // Save learning shard if requested
    if (save_path != NULL) {
        // Send a default positive reward (could be interactive in future)
        send_reward(0.5f);
        save_learning_shard(save_path);
    }

    // Cleanup
    cleanup_dynamic();
    if (g_inner_enabled) {
        inner_free(&g_inner_arianna);
    }
    if (g_meta_enabled) {
        meta_free(&g_fluid_transformer);
        if (g_meta_observation_count > 0) {
            fprintf(stderr, "[meta] %d observations completed\n",
                    g_meta_observation_count);
        }
    }
    if (g_sartre_loaded) {
        sartre_transformer_free(&g_sartre);
        fprintf(stderr, "[sartre] freed\n");
    }
    free_transformer(&t);

    return 0;
}
