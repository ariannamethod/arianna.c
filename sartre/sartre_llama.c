/*
 * sartre_llama.c — SARTRE: Verbal Interface for Arianna Metalinux
 * Llama 3 architecture (RMSNorm, RoPE, SwiGLU)
 *
 * "L'existence précède l'essence."
 *
 * Tiny transformer (~500K params) that observes system state.
 */

#include "sartre.h"
#include <time.h>
#include <math.h>

// ============================================================
// Types defined in sartre.h
// ============================================================

// ============================================================
// GLOBAL STATE
// ============================================================

static Sartre sartre = {0};
static SystemState system_state = {0};
static int sartre_initialized = 0;

// ============================================================
// CORE OPS (Llama 3 style)
// ============================================================

static void rms_norm(float* out, float* x, float* weight, int size, float eps) {
    float rms = 0.0f;
    for (int i = 0; i < size; i++) rms += x[i] * x[i];
    rms = sqrtf(rms / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] / rms * weight[i];
}

static void silu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void softmax(float* x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul(float* out, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += x[j] * w[i * n + j];
        out[i] = val;
    }
}

// ============================================================
// MEMORY MANAGEMENT
// ============================================================

static void malloc_weights(void) {
    SartreConfig* c = &sartre.config;
    SartreWeights* w = &sartre.weights;

    int kv_dim = c->n_kv_heads * c->head_dim;

    w->tok_emb = calloc(c->vocab_size * c->dim, sizeof(float));
    w->attn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->wq = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->wk = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wv = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wo = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->ffn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->w_gate = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_up = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_down = calloc(c->n_layers * c->hidden_dim * c->dim, sizeof(float));
    w->final_norm = calloc(c->dim, sizeof(float));
    w->lm_head = calloc(c->vocab_size * c->dim, sizeof(float));
}

static void malloc_run_state(void) {
    SartreConfig* c = &sartre.config;
    SartreRunState* s = &sartre.state;

    int kv_dim = c->n_kv_heads * c->head_dim;

    s->x = calloc(c->dim, sizeof(float));
    s->xb = calloc(c->dim, sizeof(float));
    s->xb2 = calloc(c->dim, sizeof(float));
    s->hb = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    s->q = calloc(c->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(c->n_heads * c->max_seq_len, sizeof(float));
    s->key_cache = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    s->logits = calloc(c->vocab_size, sizeof(float));

    // RoPE precompute
    int rope_dim = c->head_dim;
    s->rope_cos = calloc(c->max_seq_len * (rope_dim / 2), sizeof(float));
    s->rope_sin = calloc(c->max_seq_len * (rope_dim / 2), sizeof(float));

    for (int pos = 0; pos < c->max_seq_len; pos++) {
        for (int i = 0; i < rope_dim / 2; i++) {
            float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / rope_dim);
            float angle = pos * freq;
            s->rope_cos[pos * (rope_dim / 2) + i] = cosf(angle);
            s->rope_sin[pos * (rope_dim / 2) + i] = sinf(angle);
        }
    }
}

static void free_sartre(void) {
    SartreWeights* w = &sartre.weights;
    SartreRunState* s = &sartre.state;

    free(w->tok_emb); free(w->attn_norm); free(w->wq); free(w->wk); free(w->wv);
    free(w->wo); free(w->ffn_norm); free(w->w_gate); free(w->w_up); free(w->w_down);
    free(w->final_norm); free(w->lm_head);

    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->att);
    free(s->key_cache); free(s->value_cache);
    free(s->rope_cos); free(s->rope_sin); free(s->logits);
}

// ============================================================
// PUBLIC API
// ============================================================

int sartre_init(const char* weights_path) {
    // Set default config (tiny Llama 3)
    sartre.config.dim = 256;
    sartre.config.n_layers = 3;
    sartre.config.n_heads = 4;
    sartre.config.n_kv_heads = 4;
    sartre.config.head_dim = 64;
    sartre.config.hidden_dim = 512;
    sartre.config.max_seq_len = 256;
    sartre.config.vocab_size = 256;
    sartre.config.n_kv_groups = 1;
    sartre.config.rope_theta = 10000.0f;
    sartre.config.norm_eps = 1e-5f;

    malloc_weights();
    malloc_run_state();

    // TODO: load weights from file
    fprintf(stderr, "[sartre] initialized (weights loading TODO)\n");
    sartre_initialized = 1;
    return 0;
}

void sartre_shutdown(void) {
    if (!sartre_initialized) return;
    free_sartre();
    sartre_initialized = 0;
    fprintf(stderr, "[sartre] shutdown\n");
}

char* sartre_query(const char* question) {
    if (!sartre_initialized) {
        return strdup("ERROR: SARTRE not initialized");
    }

    // TODO: actual generation
    static char response[256];
    snprintf(response, sizeof(response),
             "SARTRE: I observe system state. Question received: '%s'", question);
    return strdup(response);
}

void sartre_notify_event(const char* event) {
    if (!sartre_initialized) return;

    // Store event in system_state
    if (system_state.event_count < 8) {
        strncpy(system_state.last_events[system_state.event_count],
                event, 255);
        system_state.event_count++;
    }

    fprintf(stderr, "[sartre] event: %s\n", event);
}

void sartre_update_state(SystemState* state) {
    if (state) {
        memcpy(&system_state, state, sizeof(SystemState));
    }
}

SystemState* sartre_get_state(void) {
    return &system_state;
}

int sartre_is_ready(void) {
    return sartre_initialized;
}

void sartre_update_inner_state(float trauma, float arousal, float valence,
                                 float coherence, float prophecy_debt) {
    system_state.trauma_level = trauma;
    system_state.arousal = arousal;
    system_state.valence = valence;
    system_state.coherence = coherence;
    system_state.prophecy_debt = prophecy_debt;

    fprintf(stderr, "[sartre] inner_state: trauma=%.2f arousal=%.2f valence=%.2f coherence=%.2f debt=%.2f\n",
            trauma, arousal, valence, coherence, prophecy_debt);
}

void sartre_update_schumann(float coherence, float phase) {
    system_state.schumann_coherence = coherence;
    system_state.schumann_phase = phase;

    fprintf(stderr, "[sartre] schumann: coherence=%.2f phase=%.2f\n", coherence, phase);
}

void sartre_update_calendar(float tension, int is_shabbat) {
    system_state.calendar_tension = tension;
    system_state.is_shabbat = is_shabbat;

    fprintf(stderr, "[sartre] calendar: tension=%.2f shabbat=%d\n", tension, is_shabbat);
}

void sartre_update_module(const char* name, ModuleStatus status, float load) {
    if (!name) return;

    // Find existing module or add new one
    int idx = -1;
    for (int i = 0; i < system_state.module_count; i++) {
        if (strncmp(system_state.modules[i].name, name, 63) == 0) {
            idx = i;
            break;
        }
    }

    if (idx == -1 && system_state.module_count < 16) {
        idx = system_state.module_count++;
        strncpy(system_state.modules[idx].name, name, 63);
        system_state.modules[idx].name[63] = '\0';
    }

    if (idx >= 0) {
        system_state.modules[idx].status = status;
        system_state.modules[idx].load = load;
        system_state.modules[idx].last_active_ms = (int64_t)time(NULL) * 1000;

        fprintf(stderr, "[sartre] module %s: status=%d load=%.2f\n", name, status, load);
    }
}

void sartre_print_state(void) {
    printf("\n=== SARTRE SYSTEM STATE ===\n\n");

    printf("Inner World:\n");
    printf("  trauma: %.2f\n", system_state.trauma_level);
    printf("  arousal: %.2f\n", system_state.arousal);
    printf("  valence: %.2f\n", system_state.valence);
    printf("  coherence: %.2f\n", system_state.coherence);
    printf("  prophecy_debt: %.2f\n", system_state.prophecy_debt);
    printf("  entropy: %.2f\n\n", system_state.entropy);

    printf("Schumann:\n");
    printf("  coherence: %.2f\n", system_state.schumann_coherence);
    printf("  phase: %.2f\n\n", system_state.schumann_phase);

    printf("Calendar:\n");
    printf("  tension: %.2f\n", system_state.calendar_tension);
    printf("  is_shabbat: %d\n\n", system_state.is_shabbat);

    printf("Resources:\n");
    printf("  memory_pressure: %.2f\n", system_state.memory_pressure);
    printf("  cpu_load: %.2f\n\n", system_state.cpu_load);

    printf("Modules (%d):\n", system_state.module_count);
    for (int i = 0; i < system_state.module_count; i++) {
        printf("  [%d] %s: status=%d load=%.2f\n",
               i, system_state.modules[i].name,
               system_state.modules[i].status,
               system_state.modules[i].load);
    }
    printf("\n");

    printf("Recent Events (%d):\n", system_state.event_count);
    for (int i = 0; i < system_state.event_count; i++) {
        printf("  [%d] %s\n", i, system_state.last_events[i]);
    }
    printf("\n");

    printf("Flags:\n");
    printf("  spiral_detected: %d\n", system_state.spiral_detected);
    printf("  wormhole_active: %d\n", system_state.wormhole_active);
    printf("  strange_loop: %d\n", system_state.strange_loop);
    printf("\n");
}
