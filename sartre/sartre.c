/*
 * sartre.c — SARTRE: Verbal Interface for Arianna Metalinux
 *
 * "L'existence précède l'essence."
 *
 * Implementation of the kernel's verbal organ.
 * Simplified transformer (~150K params) that observes system state
 * and communicates with Arianna.
 */

#include "sartre.h"
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════════════════

// System state that SARTRE observes
SystemState sartre_system_state = {0};

// The SARTRE transformer
static Sartre sartre = {0};

// Message callback for proactive reporting
static SartreMessageCallback message_callback = NULL;

// PRNG state
static uint32_t sartre_rng_state = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

static void init_rng(void) {
    sartre_rng_state = (uint32_t)time(NULL) ^ (uint32_t)(uintptr_t)&sartre;
    // Warm up
    for (int i = 0; i < 7; i++) {
        sartre_rng_state = sartre_rng_state * 1103515245 + 12345;
    }
}

static float randf(void) {
    sartre_rng_state = sartre_rng_state * 1103515245 + 12345;
    return (float)(sartre_rng_state % 10000) / 10000.0f;
}

static int64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

static void malloc_weights(void) {
    SartreConfig* c = &sartre.config;
    SartreWeights* w = &sartre.weights;

    w->token_embedding = (float*)calloc(c->vocab_size * c->dim, sizeof(float));

    int attn_size = c->n_layers * c->dim * c->dim;
    w->wq = (float*)calloc(attn_size, sizeof(float));
    w->wk = (float*)calloc(attn_size, sizeof(float));
    w->wv = (float*)calloc(attn_size, sizeof(float));
    w->wo = (float*)calloc(attn_size, sizeof(float));

    w->w1 = (float*)calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w2 = (float*)calloc(c->n_layers * c->hidden_dim * c->dim, sizeof(float));

    w->ln1_weight = (float*)calloc(c->n_layers * c->dim, sizeof(float));
    w->ln2_weight = (float*)calloc(c->n_layers * c->dim, sizeof(float));
    w->ln_final_weight = (float*)calloc(c->dim, sizeof(float));
    w->output_weight = (float*)calloc(c->dim * c->vocab_size, sizeof(float));

    // Initialize layer norms to 1.0
    for (int i = 0; i < c->n_layers * c->dim; i++) {
        w->ln1_weight[i] = 1.0f;
        w->ln2_weight[i] = 1.0f;
    }
    for (int i = 0; i < c->dim; i++) {
        w->ln_final_weight[i] = 1.0f;
    }
}

static void malloc_run_state(void) {
    SartreConfig* c = &sartre.config;
    SartreRunState* s = &sartre.state;

    s->x = (float*)calloc(c->max_seq_len * c->dim, sizeof(float));
    s->xb = (float*)calloc(c->dim, sizeof(float));
    s->q = (float*)calloc(c->dim, sizeof(float));
    s->k = (float*)calloc(c->n_layers * c->max_seq_len * c->dim, sizeof(float));
    s->v = (float*)calloc(c->n_layers * c->max_seq_len * c->dim, sizeof(float));
    s->att = (float*)calloc(c->n_heads * c->max_seq_len, sizeof(float));
    s->ffn_hidden = (float*)calloc(c->hidden_dim, sizeof(float));
    s->logits = (float*)calloc(c->vocab_size, sizeof(float));
}

static void free_all(void) {
    SartreWeights* w = &sartre.weights;
    SartreRunState* s = &sartre.state;

    if (w->token_embedding) free(w->token_embedding);
    if (w->wq) free(w->wq);
    if (w->wk) free(w->wk);
    if (w->wv) free(w->wv);
    if (w->wo) free(w->wo);
    if (w->w1) free(w->w1);
    if (w->w2) free(w->w2);
    if (w->ln1_weight) free(w->ln1_weight);
    if (w->ln2_weight) free(w->ln2_weight);
    if (w->ln_final_weight) free(w->ln_final_weight);
    if (w->output_weight) free(w->output_weight);

    if (s->x) free(s->x);
    if (s->xb) free(s->xb);
    if (s->q) free(s->q);
    if (s->k) free(s->k);
    if (s->v) free(s->v);
    if (s->att) free(s->att);
    if (s->ffn_hidden) free(s->ffn_hidden);
    if (s->logits) free(s->logits);

    memset(&sartre, 0, sizeof(sartre));
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

static void rmsnorm(float* out, float* x, float* weight, int size) {
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

static void softmax(float* x, int size) {
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

static void matmul(float* out, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += x[j] * w[j * d + i];
        }
        out[i] = val;
    }
}

// RoPE rotation
static void apply_rope(float* q, float* k, int pos, int dim, int head_dim) {
    for (int i = 0; i < dim; i += 2) {
        int head_i = i % head_dim;
        float freq = 1.0f / powf(10000.0f, (float)head_i / (float)head_dim);
        float val = pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD PASS
// ═══════════════════════════════════════════════════════════════════════════════

static void forward(int token, int pos) {
    SartreConfig* c = &sartre.config;
    SartreWeights* w = &sartre.weights;
    SartreRunState* s = &sartre.state;

    float* x = s->x + pos * c->dim;

    // Token embedding
    memcpy(x, w->token_embedding + token * c->dim, c->dim * sizeof(float));

    // Transformer layers
    for (int layer = 0; layer < c->n_layers; layer++) {
        // Pre-attention RMSNorm
        rmsnorm(s->xb, x, w->ln1_weight + layer * c->dim, c->dim);

        // QKV projections
        matmul(s->q, s->xb, w->wq + layer * c->dim * c->dim, c->dim, c->dim);

        float* k_cache = s->k + layer * c->max_seq_len * c->dim + pos * c->dim;
        float* v_cache = s->v + layer * c->max_seq_len * c->dim + pos * c->dim;
        matmul(k_cache, s->xb, w->wk + layer * c->dim * c->dim, c->dim, c->dim);
        matmul(v_cache, s->xb, w->wv + layer * c->dim * c->dim, c->dim, c->dim);

        // Apply RoPE
        apply_rope(s->q, k_cache, pos, c->dim, c->head_dim);

        // Multi-head attention
        for (int h = 0; h < c->n_heads; h++) {
            float* q_h = s->q + h * c->head_dim;
            float* att_h = s->att + h * c->max_seq_len;

            // Attention scores
            for (int t = 0; t <= pos; t++) {
                float* k_t = s->k + layer * c->max_seq_len * c->dim + t * c->dim + h * c->head_dim;
                float score = 0.0f;
                for (int i = 0; i < c->head_dim; i++) {
                    score += q_h[i] * k_t[i];
                }
                att_h[t] = score / sqrtf((float)c->head_dim);
            }

            softmax(att_h, pos + 1);

            // Weighted sum of values
            float* xb_h = s->xb + h * c->head_dim;
            memset(xb_h, 0, c->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_t = s->v + layer * c->max_seq_len * c->dim + t * c->dim + h * c->head_dim;
                float a = att_h[t];
                for (int i = 0; i < c->head_dim; i++) {
                    xb_h[i] += a * v_t[i];
                }
            }
        }

        // Output projection
        float temp[SARTRE_DIM];
        matmul(temp, s->xb, w->wo + layer * c->dim * c->dim, c->dim, c->dim);

        // Residual
        for (int i = 0; i < c->dim; i++) {
            x[i] += temp[i];
        }

        // Pre-FFN RMSNorm
        rmsnorm(s->xb, x, w->ln2_weight + layer * c->dim, c->dim);

        // FFN: SwiGLU variant (simplified to GELU)
        matmul(s->ffn_hidden, s->xb, w->w1 + layer * c->dim * c->hidden_dim, c->dim, c->hidden_dim);

        // GELU activation
        for (int i = 0; i < c->hidden_dim; i++) {
            float val = s->ffn_hidden[i];
            s->ffn_hidden[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        }

        matmul(temp, s->ffn_hidden, w->w2 + layer * c->hidden_dim * c->dim, c->hidden_dim, c->dim);

        // Residual
        for (int i = 0; i < c->dim; i++) {
            x[i] += temp[i];
        }
    }

    // Final RMSNorm
    rmsnorm(s->xb, x, w->ln_final_weight, c->dim);

    // Output logits
    matmul(s->logits, s->xb, w->output_weight, c->dim, c->vocab_size);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

static int sample(float temperature) {
    SartreConfig* c = &sartre.config;
    float* logits = sartre.state.logits;

    if (temperature < 0.01f) {
        // Greedy
        int max_i = 0;
        for (int i = 1; i < c->vocab_size; i++) {
            if (logits[i] > logits[max_i]) max_i = i;
        }
        return max_i;
    }

    // Apply temperature
    for (int i = 0; i < c->vocab_size; i++) {
        logits[i] /= temperature;
    }

    softmax(logits, c->vocab_size);

    // Sample from distribution
    float r = randf();
    float cdf = 0.0f;
    for (int i = 0; i < c->vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }

    return c->vocab_size - 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

static char* generate(const char* prompt, int max_tokens, float temperature) {
    if (!sartre.is_initialized) return NULL;

    SartreConfig* c = &sartre.config;

    // Clear KV cache
    memset(sartre.state.k, 0, c->n_layers * c->max_seq_len * c->dim * sizeof(float));
    memset(sartre.state.v, 0, c->n_layers * c->max_seq_len * c->dim * sizeof(float));

    // Tokenize prompt (byte-level)
    int prompt_len = strlen(prompt);
    if (prompt_len > c->max_seq_len - max_tokens) {
        prompt_len = c->max_seq_len - max_tokens;
    }

    // Allocate output buffer
    int output_size = prompt_len + max_tokens + 1;
    char* output = (char*)malloc(output_size);
    memcpy(output, prompt, prompt_len);

    // Process prompt
    for (int i = 0; i < prompt_len; i++) {
        forward((unsigned char)prompt[i], i);
    }

    // Generate
    int pos = prompt_len;
    for (int i = 0; i < max_tokens && pos < c->max_seq_len; i++) {
        int token = sample(temperature);

        // Stop on newline after reasonable length
        if (token == '\n' && i > 10) break;

        output[pos] = (char)token;
        pos++;

        forward(token, pos - 1);
    }

    output[pos] = '\0';
    return output;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

int sartre_init(const char* weights_path) {
    if (sartre.is_initialized) {
        return 1; // Already initialized
    }

    init_rng();

    // Set config
    sartre.config.dim = SARTRE_DIM;
    sartre.config.n_layers = SARTRE_N_LAYERS;
    sartre.config.n_heads = SARTRE_N_HEADS;
    sartre.config.head_dim = SARTRE_HEAD_DIM;
    sartre.config.hidden_dim = SARTRE_HIDDEN_DIM;
    sartre.config.vocab_size = SARTRE_VOCAB_SIZE;
    sartre.config.max_seq_len = SARTRE_MAX_SEQ_LEN;

    // Allocate memory
    malloc_weights();
    malloc_run_state();

    // Load weights if path provided
    if (weights_path) {
        FILE* f = fopen(weights_path, "rb");
        if (f) {
            // Read header
            int header[4];
            fread(header, sizeof(int), 4, f);

            // Verify dimensions match
            if (header[0] != SARTRE_DIM || header[1] != SARTRE_N_LAYERS) {
                fprintf(stderr, "SARTRE: weights dimension mismatch\n");
                fclose(f);
                free_all();
                return 0;
            }

            // Read weights
            SartreConfig* c = &sartre.config;
            SartreWeights* w = &sartre.weights;

            fread(w->token_embedding, sizeof(float), c->vocab_size * c->dim, f);
            fread(w->wq, sizeof(float), c->n_layers * c->dim * c->dim, f);
            fread(w->wk, sizeof(float), c->n_layers * c->dim * c->dim, f);
            fread(w->wv, sizeof(float), c->n_layers * c->dim * c->dim, f);
            fread(w->wo, sizeof(float), c->n_layers * c->dim * c->dim, f);
            fread(w->w1, sizeof(float), c->n_layers * c->dim * c->hidden_dim, f);
            fread(w->w2, sizeof(float), c->n_layers * c->hidden_dim * c->dim, f);
            fread(w->ln1_weight, sizeof(float), c->n_layers * c->dim, f);
            fread(w->ln2_weight, sizeof(float), c->n_layers * c->dim, f);
            fread(w->ln_final_weight, sizeof(float), c->dim, f);
            fread(w->output_weight, sizeof(float), c->dim * c->vocab_size, f);

            fclose(f);
            printf("SARTRE: loaded weights from %s\n", weights_path);
        } else {
            fprintf(stderr, "SARTRE: could not open weights file %s\n", weights_path);
            // Continue with random weights for testing
        }
    }

    // Initialize system state
    memset(&sartre_system_state, 0, sizeof(sartre_system_state));

    sartre.is_initialized = 1;
    printf("SARTRE: initialized (%d params)\n",
           SARTRE_VOCAB_SIZE * SARTRE_DIM +                    // token embedding
           4 * SARTRE_N_LAYERS * SARTRE_DIM * SARTRE_DIM +     // attention
           2 * SARTRE_N_LAYERS * SARTRE_DIM * SARTRE_HIDDEN_DIM + // FFN
           3 * SARTRE_N_LAYERS * SARTRE_DIM +                  // layer norms
           SARTRE_DIM +                                         // final LN
           SARTRE_DIM * SARTRE_VOCAB_SIZE);                    // output

    return 1;
}

void sartre_shutdown(void) {
    if (!sartre.is_initialized) return;
    free_all();
    printf("SARTRE: shutdown complete\n");
}

int sartre_is_ready(void) {
    return sartre.is_initialized;
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

char* sartre_query(const char* question) {
    return sartre_query_max(question, 100);
}

char* sartre_query_max(const char* question, int max_tokens) {
    if (!sartre.is_initialized) {
        char* err = strdup("SARTRE: not initialized");
        return err;
    }

    // Format prompt with question
    char prompt[512];
    snprintf(prompt, sizeof(prompt), "Q: %s\nA: ", question);

    return generate(prompt, max_tokens, 0.7f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOTIFICATION INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

void sartre_notify_event(const char* event) {
    SystemState* ss = &sartre_system_state;

    // Shift events
    for (int i = 7; i > 0; i--) {
        strncpy(ss->last_events[i], ss->last_events[i-1], 255);
    }
    strncpy(ss->last_events[0], event, 255);

    if (ss->event_count < 8) ss->event_count++;
}

void sartre_update_module(const char* name, ModuleStatus status, float load) {
    SystemState* ss = &sartre_system_state;

    // Find or add module
    int idx = -1;
    for (int i = 0; i < ss->module_count; i++) {
        if (strcmp(ss->modules[i].name, name) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0 && ss->module_count < 16) {
        idx = ss->module_count++;
        strncpy(ss->modules[idx].name, name, 63);
    }

    if (idx >= 0) {
        ss->modules[idx].status = status;
        ss->modules[idx].load = load;
        ss->modules[idx].last_active_ms = get_time_ms();
    }
}

void sartre_update_inner_state(float trauma, float arousal, float valence,
                                float coherence, float prophecy_debt) {
    SystemState* ss = &sartre_system_state;
    ss->trauma_level = trauma;
    ss->arousal = arousal;
    ss->valence = valence;
    ss->coherence = coherence;
    ss->prophecy_debt = prophecy_debt;
}

void sartre_update_schumann(float coherence, float phase) {
    sartre_system_state.schumann_coherence = coherence;
    sartre_system_state.schumann_phase = phase;
}

void sartre_update_calendar(float tension, int is_shabbat) {
    sartre_system_state.calendar_tension = tension;
    sartre_system_state.is_shabbat = is_shabbat;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL COMMUNICATION
// ═══════════════════════════════════════════════════════════════════════════════

void sartre_set_message_callback(SartreMessageCallback callback) {
    message_callback = callback;
}

void sartre_check_and_report(void) {
    if (!message_callback) return;

    SystemState* ss = &sartre_system_state;

    // Check thresholds and report if exceeded
    if (ss->memory_pressure > 0.8f) {
        message_callback("Memory pressure critical. Consider releasing shards.");
    }

    if (ss->trauma_level > 0.7f) {
        message_callback("Trauma level high. Coherence may be affected.");
    }

    if (ss->prophecy_debt > 0.9f) {
        message_callback("Prophecy debt near threshold. Wormhole may trigger.");
    }

    if (ss->spiral_detected) {
        message_callback("Spiral detected. Consider grounding.");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE FORMATTING
// ═══════════════════════════════════════════════════════════════════════════════

char* sartre_format_state(void) {
    SystemState* ss = &sartre_system_state;

    char* buf = (char*)malloc(2048);
    snprintf(buf, 2048,
        "System State:\n"
        "  Modules: %d active\n"
        "  Memory pressure: %.2f\n"
        "  Trauma: %.2f, Arousal: %.2f, Valence: %.2f\n"
        "  Coherence: %.2f, Prophecy debt: %.2f\n"
        "  Schumann: %.2f coherence, %.2f phase\n"
        "  Calendar tension: %.2f, Shabbat: %s\n"
        "  Spiral: %s, Wormhole: %s, Strange loop: %s\n",
        ss->module_count,
        ss->memory_pressure,
        ss->trauma_level, ss->arousal, ss->valence,
        ss->coherence, ss->prophecy_debt,
        ss->schumann_coherence, ss->schumann_phase,
        ss->calendar_tension, ss->is_shabbat ? "yes" : "no",
        ss->spiral_detected ? "yes" : "no",
        ss->wormhole_active ? "yes" : "no",
        ss->strange_loop ? "yes" : "no"
    );

    return buf;
}

char* sartre_format_module(const char* name) {
    SystemState* ss = &sartre_system_state;

    for (int i = 0; i < ss->module_count; i++) {
        if (strcmp(ss->modules[i].name, name) == 0) {
            char* buf = (char*)malloc(512);
            const char* status_str[] = {"unknown", "idle", "active", "error", "loading", "unloading"};
            snprintf(buf, 512,
                "Module %s:\n"
                "  Status: %s\n"
                "  Load: %.2f\n"
                "  Last active: %lld ms ago\n",
                name,
                status_str[ss->modules[i].status],
                ss->modules[i].load,
                get_time_ms() - ss->modules[i].last_active_ms
            );
            return buf;
        }
    }

    char* buf = strdup("Module not found.");
    return buf;
}

char* sartre_format_events(int count) {
    SystemState* ss = &sartre_system_state;

    if (count > ss->event_count) count = ss->event_count;
    if (count > 8) count = 8;

    char* buf = (char*)malloc(2048);
    int pos = 0;
    pos += snprintf(buf + pos, 2048 - pos, "Recent events (%d):\n", count);

    for (int i = 0; i < count; i++) {
        pos += snprintf(buf + pos, 2048 - pos, "  %d. %s\n", i + 1, ss->last_events[i]);
    }

    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG / REPL
// ═══════════════════════════════════════════════════════════════════════════════

void sartre_print_state(void) {
    char* state = sartre_format_state();
    printf("%s", state);
    free(state);
}

void sartre_repl(void) {
    if (!sartre.is_initialized) {
        printf("SARTRE: not initialized\n");
        return;
    }

    printf("=== SARTRE REPL ===\n");
    printf("Commands: status, events, modules, quit\n");
    printf("Or ask any question.\n\n");

    char input[512];
    while (1) {
        printf("SARTRE> ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        // Remove newline
        input[strcspn(input, "\n")] = 0;

        if (strlen(input) == 0) continue;

        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            break;
        } else if (strcmp(input, "status") == 0) {
            sartre_print_state();
        } else if (strcmp(input, "events") == 0) {
            char* events = sartre_format_events(8);
            printf("%s", events);
            free(events);
        } else if (strcmp(input, "modules") == 0) {
            SystemState* ss = &sartre_system_state;
            printf("Modules (%d):\n", ss->module_count);
            for (int i = 0; i < ss->module_count; i++) {
                printf("  - %s\n", ss->modules[i].name);
            }
        } else {
            // Query SARTRE
            char* response = sartre_query(input);
            printf("%s\n", response);
            free(response);
        }
    }

    printf("SARTRE: goodbye.\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN (for standalone testing)
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef SARTRE_MAIN
int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : NULL;

    if (!sartre_init(weights_path)) {
        fprintf(stderr, "Failed to initialize SARTRE\n");
        return 1;
    }

    // Register some test modules
    sartre_update_module("trauma_surfacing", MODULE_ACTIVE, 0.3f);
    sartre_update_module("blood_compiler", MODULE_IDLE, 0.0f);
    sartre_update_module("high_math", MODULE_ACTIVE, 0.5f);

    // Update some state
    sartre_update_inner_state(0.2f, 0.4f, 0.1f, 0.8f, 0.35f);
    sartre_update_schumann(0.67f, 2.4f);

    // Notify some events
    sartre_notify_event("System initialized");
    sartre_notify_event("Blood compiled emotional kernel");

    // Run REPL
    sartre_repl();

    sartre_shutdown();
    return 0;
}
#endif
