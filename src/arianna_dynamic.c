/*
 * arianna_dynamic.c - Main entry point with delta support
 *
 * Core personality + dynamic attention deltas
 * "Who I am" grows through experience
 *
 * Stanley integration: mood routing, microtraining, shard persistence
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
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

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

// MathBrain (arithmetic through resonance)
static MathBrain g_mathbrain;
static int g_mathbrain_enabled = 0;
static const char* g_mathbrain_path = "weights/mathbrain.bin";  // Default persistence path

// Active learning shard (for microtraining)
static ExperienceShard* g_active_shard = NULL;

// Generation state for microtraining
typedef struct {
    float* pre_activations;   // Before attention
    float* post_activations;  // After attention
    int last_layer;
    int sequence_pos;
} TrainingState;

static TrainingState g_train_state;

// ============================================================
// Extended forward pass with delta support
// ============================================================

void forward_dynamic(Transformer* t, int* tokens, int n_tokens, int pos) {
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

        // === MICROTRAINING HOOK: capture pre-attention state ===
        if (g_microtraining && g_train_state.pre_activations != NULL) {
            memcpy(g_train_state.pre_activations, s->xb, dim * sizeof(float));
            g_train_state.last_layer = layer;
            g_train_state.sequence_pos = pos;
        }

        // QKV projections
        float* q = s->q;
        int kv_offset = layer * c->max_seq_len * dim + pos * dim;
        float* k = s->k + kv_offset;
        float* v = s->v + kv_offset;

        matmul(q, s->xb, wq, 1, dim, dim);
        matmul(k, s->xb, wk, 1, dim, dim);
        matmul(v, s->xb, wv, 1, dim, dim);

        // === DELTA APPLICATION ===
        // Deltas modify ATTENTION, not weights directly
        // This is where experience shapes perception
        if (g_delta_enabled && g_delta_bank.n_shards > 0) {
            apply_q_delta(&g_delta_bank, q, s->xb, layer, dim);
            apply_k_delta(&g_delta_bank, k, s->xb, layer, dim);
            apply_v_delta(&g_delta_bank, v, s->xb, layer, dim);
        }

        // Apply RoPE
        apply_rope(q, k, pos, dim, head_dim, n_heads);

        // Multi-head attention
        for (int h = 0; h < n_heads; h++) {
            float* att = s->att + h * c->max_seq_len;

            for (int t = 0; t <= pos; t++) {
                float score = 0.0f;
                float* kt = s->k + layer * c->max_seq_len * dim + t * dim + h * head_dim;
                float* qt = q + h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    score += qt[i] * kt[i];
                }
                att[t] = score / sqrtf((float)head_dim);
            }

            softmax(att, pos + 1);

            float* out = s->xb + h * head_dim;
            memset(out, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vt = s->v + layer * c->max_seq_len * dim + t * dim + h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    out[i] += att[t] * vt[i];
                }
            }
        }

        // === MICROTRAINING HOOK: capture post-attention state ===
        if (g_microtraining && g_train_state.post_activations != NULL) {
            memcpy(g_train_state.post_activations, s->xb, dim * sizeof(float));
        }

        // Output projection and residual
        float attn_out[DIM];
        matmul(attn_out, s->xb, wo, 1, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += attn_out[i];
        }

        // FFN
        rmsnorm(s->xb, x, ln2, dim);

        float* ffn = s->ffn_hidden;
        matmul(ffn, s->xb, w1, 1, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            ffn[i] = ffn[i] * (1.0f / (1.0f + expf(-ffn[i])));
        }

        float ffn_out[DIM];
        matmul(ffn_out, ffn, w2, 1, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += ffn_out[i];
        }
    }

    // Final layer norm
    rmsnorm(s->xb, x, w->ln_final_weight, dim);

    // Output projection to logits
    matmul(s->logits, s->xb, w->output_weight, 1, dim, c->vocab_size);
}

// ============================================================
// Dynamic generation with mood routing + microtraining
// ============================================================

void generate_dynamic(Transformer* t, char* prompt, int max_tokens, float temperature) {
    int tokens[MAX_SEQ_LEN];
    int n_tokens = strlen(prompt);

    // Tokenize prompt
    for (int i = 0; i < n_tokens && i < MAX_SEQ_LEN; i++) {
        tokens[i] = (unsigned char)prompt[i];
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

    // Generate
    printf("%s", prompt);
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

        int next_token = sample(t->state.logits, t->config.vocab_size, effective_temp);
        tokens[n_tokens] = next_token;
        putchar((char)next_token);

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
                    recent_text[j] = (char)tokens[start + j];
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
        }

        forward_dynamic(t, tokens, n_tokens + 1, n_tokens);
        n_tokens++;
    }
    printf("\n");
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
     */

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

    // 3. Convert seed to tokens
    int tokens[MAX_SEQ_LEN];
    int n_tokens = seed_to_tokens(seed, tokens, MAX_SEQ_LEN);

    printf("[Internal seed: \"%.*s\"]\n", seed->len > 50 ? 50 : seed->len, seed->text);

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

        int next_token = sample(t->state.logits, t->config.vocab_size, effective_temp);
        tokens[n_tokens] = next_token;
        char c = (char)next_token;
        putchar(c);

        // Microlearning: update experience shard in real-time
        if (g_microtraining && g_active_shard != NULL) {
            // Compute softmax probabilities for experience_step
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
            // quality < 0.5 = reduce learning (stuck/bad)
            // quality > 0.5 = boost learning (flowing well)
            float base_signal = 0.3f;  // Always learn something
            float quality_mod = (g_body_state.quality - 0.5f) * 0.7f;
            float signal = base_signal + quality_mod;

            // Update Q delta for current layer (attention shaping)
            int layer = g_train_state.last_layer;
            if (layer >= 0 && layer < g_active_shard->n_layers) {
                experience_step(&g_trainer, &g_active_shard->attn_q_deltas[layer],
                               t->state.xb, probs, next_token, signal);
            }
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
                recent_text[j] = (char)tokens[start + j];
            }
            recent_text[text_len] = '\0';

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

            // Update temperature
            effective_temp = get_modulated_temperature(&g_subjectivity);
            effective_temp = effective_temp * 0.6f + temperature * 0.4f;

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
    printf("\n");

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

    int result = save_shard(g_active_shard, path);
    if (result == 0) {
        printf("Saved shard to %s\n", path);
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

        if (strcmp(input, "help") == 0) {
            printf("Commands:\n");
            printf("  signals  - show signal values\n");
            printf("  body     - show somatic state (boredom, overwhelm, stuck)\n");
            printf("  self     - show SelfSense signals from hidden states\n");
            printf("  subj     - show subjectivity state\n");
            printf("  cooccur  - show co-occurrence stats\n");
            printf("  math     - show MathBrain stats\n");
            printf("  mathsave - save MathBrain state now\n");
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
                printf("[Learning started - shard: %s]\n", shard_name);
                printf("[Experience will accumulate as you chat]\n");
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
    "origin.txt",
    "./origin.txt",
    "data/origin.txt",
    "../origin.txt",
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
    printf("Usage: %s <weights.bin> \"<prompt>\" [max_tokens] [temperature]\n", prog);
    printf("       %s <weights.bin> --repl [max_tokens] [temperature]\n", prog);
    printf("\nOptions:\n");
    printf("  --repl          Interactive REPL mode (state persists between turns)\n");
    printf("  -shard <path>   Load experience shard (can use multiple times)\n");
    printf("  -no-mood        Disable mood routing (enabled by default)\n");
    printf("  -guided         Enable guided attention (gravity centers, pulse)\n");
    printf("  -subj <origin>  Use custom origin file (default: auto-detect origin.txt)\n");
    printf("  -no-subj        Disable subjectivity (use prompt as seed)\n");
    printf("  -signals        Print signal values after generation\n");
    printf("  -learn <name>   Create new learning shard with name\n");
    printf("  -save <path>    Save learning shard after generation\n");
    printf("  -momentum <f>   Mood transition momentum (0.0-1.0, default 0.8)\n");
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

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    char* weights_path = argv[1];
    char* prompt = NULL;
    char* shard_paths[MAX_SHARDS];
    int n_shard_paths = 0;
    char* learn_name = NULL;
    char* save_path = NULL;
    int max_tokens = 100;
    int max_tokens_set = 0;  // flag to track if user provided max_tokens
    float temperature = 0.8f;
    float momentum = 0.8f;
    int print_sigs = 0;
    int mood_mode = 1;    // ENABLED BY DEFAULT - mood shapes attention
    int guided_mode = 0;
    int subj_mode = 1;    // ENABLED BY DEFAULT - this is Arianna's core
    int repl_mode = 0;    // Interactive REPL mode
    char* origin_path = NULL;

    // Parse arguments
    int arg_idx = 2;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "--repl") == 0) {
            repl_mode = 1;
        } else if (strcmp(argv[arg_idx], "-subj") == 0 && arg_idx + 1 < argc) {
            subj_mode = 1;
            origin_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-no-subj") == 0) {
            subj_mode = 0;  // Disable subjectivity, use prompt as seed
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
        } else if (strcmp(argv[arg_idx], "-learn") == 0 && arg_idx + 1 < argc) {
            learn_name = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-save") == 0 && arg_idx + 1 < argc) {
            save_path = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-momentum") == 0 && arg_idx + 1 < argc) {
            momentum = atof(argv[++arg_idx]);
        } else if (prompt == NULL) {
            prompt = argv[arg_idx];
        } else if (max_tokens_set == 0) {
            max_tokens = atoi(argv[arg_idx]);
            max_tokens_set = 1;
        } else {
            temperature = atof(argv[arg_idx]);
        }
        arg_idx++;
    }

    if (prompt == NULL && !repl_mode) {
        print_usage(argv[0]);
        return 1;
    }

    srand(time(NULL));

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
        printf("Microtraining: enabled\n");
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
                fprintf(stderr, "Warning: no origin.txt found, subjectivity disabled\n");
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
    }

    // Save learning shard if requested
    if (save_path != NULL) {
        // Send a default positive reward (could be interactive in future)
        send_reward(0.5f);
        save_learning_shard(save_path);
    }

    // Cleanup
    cleanup_dynamic();
    free_transformer(&t);

    return 0;
}
