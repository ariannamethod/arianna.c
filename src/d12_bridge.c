/*
 * d12_bridge.c — Tongue bridge via Go libtongue (dlopen)
 *
 * Thin wrapper: dlopen("tongue/libtongue.so") and call tongue_* exports.
 * All inference happens in Go (goroutines, parallel matmul, GGUF Q4_0).
 * C side only manages modulation signals from Arianna ecosystem.
 *
 * This is not inference. This is breathing.
 */

#include "d12_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* ============================================================
 * dlopen helpers
 * ============================================================ */

static void* load_sym(void* lib, const char* name) {
    void* sym = dlsym(lib, name);
    if (!sym) {
        fprintf(stderr, "[d12_bridge] missing symbol: %s (%s)\n", name, dlerror());
    }
    return sym;
}

/* Resolve libtongue path: .dylib on macOS, .so on Linux */
static const char* tongue_lib_path(void) {
#ifdef __APPLE__
    return D12_TONGUE_LIB ".dylib";
#else
    return D12_TONGUE_LIB ".so";
#endif
}

/* ============================================================
 * Lifecycle
 * ============================================================ */

int d12_init(D12Bridge* d12, const char* weights_path, const char* tokenizer_path) {
    (void)tokenizer_path;  /* GGUF has tokenizer embedded */
    if (!d12) return -1;
    memset(d12, 0, sizeof(D12Bridge));

    /* 1. dlopen libtongue */
    const char* libpath = tongue_lib_path();
    d12->tongue_lib = dlopen(libpath, RTLD_LAZY);
    if (!d12->tongue_lib) {
        fprintf(stderr, "[d12_bridge] dlopen(%s) failed: %s\n", libpath, dlerror());
        return -1;
    }
    printf("[d12_bridge] loaded %s\n", libpath);

    /* 2. Load function pointers */
    #define LOAD(field, name) \
        d12->field = load_sym(d12->tongue_lib, name); \
        if (!d12->field) { dlclose(d12->tongue_lib); d12->tongue_lib = NULL; return -1; }

    LOAD(fn_init,                "tongue_init");
    LOAD(fn_free,                "tongue_free");
    LOAD(fn_reset,               "tongue_reset");
    LOAD(fn_generate,            "tongue_generate");
    LOAD(fn_set_temperature_mod, "tongue_set_temperature_mod");
    LOAD(fn_set_logit_scale,     "tongue_set_logit_scale");
    LOAD(fn_set_exploratory_bias,"tongue_set_exploratory_bias");
    LOAD(fn_set_temp_floor,      "tongue_set_temp_floor");
    LOAD(fn_set_rep_penalty,     "tongue_set_rep_penalty");
    LOAD(fn_encode,              "tongue_encode");
    LOAD(fn_decode_token,        "tongue_decode_token");
    LOAD(fn_get_vocab_size,      "tongue_get_vocab_size");
    LOAD(fn_get_dim,             "tongue_get_dim");
    LOAD(fn_get_seq_len,         "tongue_get_seq_len");
    LOAD(fn_get_num_layers,      "tongue_get_num_layers");

    #undef LOAD

    /* 3. Initialize Go tongue with GGUF weights */
    int rc = d12->fn_init((char*)weights_path);
    if (rc != 0) {
        fprintf(stderr, "[d12_bridge] tongue_init failed (rc=%d)\n", rc);
        dlclose(d12->tongue_lib);
        d12->tongue_lib = NULL;
        return -1;
    }

    /* 4. Set temperature floor */
    d12->fn_set_temp_floor(D12_TEMP_FLOOR);

    /* 5. Initialize modulation defaults */
    memset(&d12->mod, 0, sizeof(D12Modulation));
    d12->mod.temperature_mod = 1.0f;
    d12->mod.logit_scale = 1.0f;

    d12->initialized = 1;
    d12->weights_loaded = 1;
    d12->pos = 0;

    printf("[d12_bridge] Tongue initialized: %d layers, %d dim, %d vocab, seq_len=%d\n",
           d12->fn_get_num_layers(), d12->fn_get_dim(),
           d12->fn_get_vocab_size(), d12->fn_get_seq_len());

    return 0;
}

void d12_free(D12Bridge* d12) {
    if (!d12) return;
    if (d12->fn_free) d12->fn_free();
    if (d12->tongue_lib) dlclose(d12->tongue_lib);
    memset(d12, 0, sizeof(D12Bridge));
}

/* ============================================================
 * Modulation Updates — identical logic to original d12_bridge.c
 * ============================================================ */

void d12_update_from_arianna(D12Bridge* d12, const Transformer* arianna, const char* input_text) {
    (void)input_text;
    if (!d12 || !d12->initialized || !arianna) return;

    const float* logits = arianna->state.logits;
    int vocab = arianna->config.vocab_size;

    float max_val = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    float probs[256];
    for (int i = 0; i < vocab && i < 256; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    float entropy = 0.0f;
    for (int i = 0; i < vocab && i < 256; i++) {
        float p = probs[i] / sum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    float max_entropy = logf((float)vocab);
    d12->mod.resonance_entropy = entropy / max_entropy;

    int top_id = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[top_id]) top_id = i;
    d12->mod.resonance_direction = (float)top_id / (float)vocab;
    d12->mod.resonance_strength = 1.0f - d12->mod.resonance_entropy;
}

void d12_update_from_cloud(D12Bridge* d12, const CloudResponse* cloud) {
    if (!d12 || !d12->initialized || !cloud) return;
    d12->mod.cloud_warmth = cloud->chambers[1];           /* LOVE */
    d12->mod.cloud_tension = (cloud->chambers[0] + cloud->chambers[2]) * 0.5f;  /* (FEAR+RAGE)/2 */
    d12->mod.cloud_primary_strength = cloud->primary_strength;
}

void d12_update_from_meta(D12Bridge* d12, const MetaThermogram* thermo) {
    if (!d12 || !d12->initialized || !thermo || !thermo->valid) return;
    d12->mod.meta_sharpness = thermo->sharpness;
    d12->mod.meta_warmth = thermo->warmth;
    d12->mod.meta_silence = thermo->silence;
    d12->mod.meta_drift_rate = thermo->drift_rate;
    d12->mod.meta_drift_direction = thermo->drift_direction;
}

void d12_update_from_sartre(D12Bridge* d12, float coherence, float arousal, float trauma) {
    if (!d12 || !d12->initialized) return;
    d12->mod.sartre_coherence = coherence;
    d12->mod.sartre_arousal = arousal;
    d12->mod.sartre_trauma = trauma;
}

void d12_compute_modulation(D12Bridge* d12) {
    if (!d12 || !d12->initialized) return;

    D12Modulation* m = &d12->mod;

    /* Temperature modulation:
     * - High arousal -> lower temp (more focused)
     * - High entropy from Arianna -> higher temp (more exploratory)
     * - High trauma -> slightly higher temp (destabilize for escape)
     * - Meta drift collapsing -> lower temp (stabilize) */
    float temp_base = 1.0f;
    temp_base -= 0.2f * m->sartre_arousal;
    temp_base += 0.15f * m->resonance_entropy;
    temp_base += 0.1f * m->sartre_trauma;
    if (m->meta_drift_direction < 0) {
        temp_base -= 0.1f * m->meta_drift_rate;
    }
    m->temperature_mod = fmaxf(0.5f, fminf(1.5f, temp_base));

    /* Logit scale:
     * - High coherence -> boost confident outputs
     * - High resonance strength -> trust Arianna's direction
     * - Cloud tension -> reduce scale (careful) */
    float scale = 1.0f;
    scale += 0.1f * m->sartre_coherence;
    scale += 0.05f * m->resonance_strength;
    scale -= 0.1f * m->cloud_tension;
    m->logit_scale = fmaxf(0.8f, fminf(1.2f, scale));

    /* Exploratory bias:
     * - Positive = more creative (higher entropy tokens)
     * - Meta warmth -> more exploration
     * - High silence -> suppress exploration (introspection)
     * - Cloud warmth -> encourage */
    float explore = 0.0f;
    explore += 0.2f * m->meta_warmth;
    explore -= 0.15f * m->meta_silence;
    explore += 0.1f * m->cloud_warmth;
    explore -= 0.1f * m->resonance_strength;
    m->exploratory_bias = fmaxf(-0.3f, fminf(0.3f, explore));

    /* Push modulation to Go tongue */
    if (d12->fn_set_temperature_mod)  d12->fn_set_temperature_mod(m->temperature_mod);
    if (d12->fn_set_logit_scale)      d12->fn_set_logit_scale(m->logit_scale);
    if (d12->fn_set_exploratory_bias) d12->fn_set_exploratory_bias(m->exploratory_bias);
}

/* ============================================================
 * Generation
 * ============================================================ */

void d12_reset(D12Bridge* d12) {
    if (!d12 || !d12->initialized) return;
    if (d12->fn_reset) d12->fn_reset();
    d12->pos = 0;
    memset(&d12->mod, 0, sizeof(D12Modulation));
    d12->mod.temperature_mod = 1.0f;
    d12->mod.logit_scale = 1.0f;
}

/* These are no-ops: Go tongue handles the full generation loop internally */
void d12_feed_prompt(D12Bridge* d12, const int* tokens, int n_tokens) {
    (void)d12; (void)tokens; (void)n_tokens;
}

void d12_forward(D12Bridge* d12, int token) {
    (void)d12; (void)token;
}

void d12_apply_modulation(D12Bridge* d12) {
    (void)d12;
}

int d12_sample(D12Bridge* d12, float temperature, float top_p) {
    (void)d12; (void)temperature; (void)top_p;
    return -1;
}

int d12_generate(D12Bridge* d12,
                 const char* prompt,
                 char* output, int max_output_len,
                 int max_tokens, float temperature, float top_p) {
    if (!d12 || !d12->initialized || !d12->fn_generate || !prompt || !output) return 0;

    /* Push current modulation to Go tongue before generating */
    if (d12->fn_set_temperature_mod)  d12->fn_set_temperature_mod(d12->mod.temperature_mod);
    if (d12->fn_set_logit_scale)      d12->fn_set_logit_scale(d12->mod.logit_scale);
    if (d12->fn_set_exploratory_bias) d12->fn_set_exploratory_bias(d12->mod.exploratory_bias);

    /* Apply temperature floor */
    float effective_temp = temperature * d12->mod.temperature_mod;
    if (effective_temp < D12_TEMP_FLOOR) effective_temp = D12_TEMP_FLOOR;

    /* Call Go tongue_generate — it handles BOS + anchor + prompt + generation loop */
    int n = d12->fn_generate(prompt, output, max_output_len,
                              max_tokens, effective_temp, top_p,
                              D12_ANCHOR_PROMPT);

    return n;
}

/* ============================================================
 * Tokenization
 * ============================================================ */

int d12_encode(const D12Bridge* d12, const char* text, int* ids, int max_tokens) {
    if (!d12 || !d12->initialized || !d12->fn_encode || !text || !ids) return 0;
    return d12->fn_encode(text, ids, max_tokens);
}

const char* d12_decode(const D12Bridge* d12, const int* ids, int n_tokens) {
    if (!d12 || !d12->initialized || !d12->fn_decode_token || !ids || n_tokens <= 0) return "";

    static char buffer[65536];
    int pos = 0;

    for (int i = 0; i < n_tokens && pos < (int)sizeof(buffer) - 1; i++) {
        char* piece = d12->fn_decode_token(ids[i]);
        if (piece) {
            int len = strlen(piece);
            if (pos + len < (int)sizeof(buffer) - 1) {
                memcpy(buffer + pos, piece, len);
                pos += len;
            }
            free(piece);  /* tongue_decode_token returns C.CString (malloc'd) */
        }
    }
    buffer[pos] = '\0';
    return buffer;
}

const char* d12_decode_token(const D12Bridge* d12, int id) {
    if (!d12 || !d12->initialized || !d12->fn_decode_token) return NULL;
    return d12->fn_decode_token(id);
}

/* ============================================================
 * Weight download helper
 * ============================================================ */

static int d12_path_is_safe(const char* path) {
    if (!path) return 0;
    for (const char* p = path; *p; p++) {
        char c = *p;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '_' ||
              c == '-' || c == '.')) {
            return 0;
        }
    }
    return 1;
}

const char* d12_ensure_weights(const char* cache_dir) {
    static char path[1024];

    if (!cache_dir) cache_dir = "tongue/weights";

    if (!d12_path_is_safe(cache_dir)) {
        fprintf(stderr, "[d12_bridge] Invalid cache_dir: contains unsafe characters\n");
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/" D12_WEIGHTS_FILE, cache_dir);

    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 1000000) {
        printf("[d12_bridge] Found weights at %s (%.1f MB)\n", path, st.st_size / 1024.0 / 1024.0);
        return path;
    }

    printf("[d12_bridge] Downloading weights from HuggingFace...\n");
    printf("[d12_bridge] URL: %s\n", D12_WEIGHTS_URL);

    /* Create directory */
    char mkdir_cmd[1024];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "%s", cache_dir);
    pid_t pid = fork();
    if (pid == 0) {
        execlp("mkdir", "mkdir", "-p", mkdir_cmd, NULL);
        _exit(127);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
    }

    /* Download */
    pid = fork();
    if (pid == 0) {
        execlp("curl", "curl", "-L", "--progress-bar", "-o", path, D12_WEIGHTS_URL, NULL);
        _exit(127);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[d12_bridge] Download failed\n");
            return NULL;
        }
    } else {
        fprintf(stderr, "[d12_bridge] fork() failed\n");
        return NULL;
    }

    if (stat(path, &st) != 0 || st.st_size < 1000000) {
        fprintf(stderr, "[d12_bridge] Downloaded file is too small or missing\n");
        return NULL;
    }

    printf("[d12_bridge] Downloaded %.1f MB to %s\n", st.st_size / 1024.0 / 1024.0, path);
    return path;
}
