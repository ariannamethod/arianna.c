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
#include <time.h>

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

        int next_token = sample(t->state.logits, t->config.vocab_size, effective_temp);
        tokens[n_tokens] = next_token;
        putchar((char)next_token);

        // Re-route periodically (every 16 tokens for responsive mood shifts)
        if ((n_tokens % 16 == 0)) {
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

    g_delta_enabled = 0;
    g_mood_enabled = 0;
    g_microtraining = 0;
    g_guided_enabled = 0;

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
    free_delta_bank(&g_delta_bank);
    free_microtrainer(&g_trainer);
    free_attention_bias(&g_attention_bias);
    if (g_train_state.pre_activations) free(g_train_state.pre_activations);
    if (g_train_state.post_activations) free(g_train_state.post_activations);
}

// ============================================================
// Main with dynamic support
// ============================================================

void print_usage(const char* prog) {
    printf("arianna_dynamic - Personality transformer with Stanley-style deltas\n\n");
    printf("Usage: %s <weights.bin> \"<prompt>\" [max_tokens] [temperature]\n", prog);
    printf("\nOptions:\n");
    printf("  -shard <path>   Load experience shard (can use multiple times)\n");
    printf("  -mood           Enable mood routing (Stanley-style)\n");
    printf("  -guided         Enable guided attention (gravity centers, pulse)\n");
    printf("  -signals        Print signal values after generation\n");
    printf("  -learn <name>   Create new learning shard with name\n");
    printf("  -save <path>    Save learning shard after generation\n");
    printf("  -momentum <f>   Mood transition momentum (0.0-1.0, default 0.8)\n");
    printf("\nExamples:\n");
    printf("  %s arianna.bin \"She finds that \" 100 0.8\n", prog);
    printf("  %s arianna.bin -shard warmth.bin \"She finds that \" 100 0.8\n", prog);
    printf("  %s arianna.bin -mood -shard data/shards/*.bin \"She \" 100 0.8\n", prog);
    printf("  %s arianna.bin -guided \"She \" 100 0.8\n", prog);
    printf("  %s arianna.bin -learn session1 -save session1.bin \"She \" 100\n", prog);
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
    float temperature = 0.8f;
    float momentum = 0.8f;
    int print_sigs = 0;
    int mood_mode = 0;
    int guided_mode = 0;

    // Parse arguments
    int arg_idx = 2;
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "-guided") == 0) {
            guided_mode = 1;
        } else if (strcmp(argv[arg_idx], "-shard") == 0 && arg_idx + 1 < argc) {
            if (n_shard_paths < MAX_SHARDS) {
                shard_paths[n_shard_paths++] = argv[++arg_idx];
            }
        } else if (strcmp(argv[arg_idx], "-mood") == 0) {
            mood_mode = 1;
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
        } else if (max_tokens == 100) {
            max_tokens = atoi(argv[arg_idx]);
        } else {
            temperature = atof(argv[arg_idx]);
        }
        arg_idx++;
    }

    if (prompt == NULL) {
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

    // Generate
    printf("\n--- Generation ---\n");
    generate_dynamic(&t, prompt, max_tokens, temperature);

    // Print state
    if (print_sigs) {
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
