// cloud_wrapper.c — C wrapper for Go Cloud library
//
// Bridges the C codebase with the Go implementation of Cloud 200K.
// Uses libarianna.dylib via dlfcn.

#include "cloud.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <limits.h>
#include <libgen.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// GO LIBRARY FUNCTION POINTERS
// ═══════════════════════════════════════════════════════════════════════════════

static void* go_lib = NULL;

typedef int (*cloud_init_fn)(const char*);
typedef int (*cloud_preprocess_fn)(const char*);
typedef float (*cloud_get_temperature_bias_fn)(void);
typedef char* (*cloud_get_primary_fn)(void);
typedef char* (*cloud_get_secondary_fn)(void);
typedef float (*cloud_get_chamber_fn)(const char*);
typedef char* (*cloud_ping_fn)(const char*);
typedef void (*cloud_stop_fn)(void);
typedef void (*cloud_free_fn)(void);

static cloud_init_fn go_cloud_init = NULL;
static cloud_preprocess_fn go_cloud_preprocess = NULL;
static cloud_get_temperature_bias_fn go_cloud_get_temperature_bias = NULL;
static cloud_get_primary_fn go_cloud_get_primary = NULL;
static cloud_get_secondary_fn go_cloud_get_secondary = NULL;
static cloud_get_chamber_fn go_cloud_get_chamber = NULL;
static cloud_ping_fn go_cloud_ping = NULL;
static cloud_stop_fn go_cloud_stop = NULL;
static cloud_free_fn go_cloud_free = NULL;

// ═══════════════════════════════════════════════════════════════════════════════
// STATIC DATA (anchors, chamber names)
// ═══════════════════════════════════════════════════════════════════════════════

static const char* FEAR_ANCHORS[] = {
    "fear", "terror", "panic", "anxious", "dread", "horror",
    "unease", "paranoia", "worry", "nervous", "scared",
    "frighten", "alarm", "tense", "apprehension",
    "threat", "vulnerable", "insecure", "timid", "wary"
};

static const char* LOVE_ANCHORS[] = {
    "love", "warmth", "tenderness", "devotion", "longing",
    "yearning", "affection", "care", "intimacy", "attachment",
    "adoration", "passion", "fondness", "cherish", "desire",
    "compassion", "gentle", "sweet"
};

static const char* RAGE_ANCHORS[] = {
    "anger", "rage", "fury", "hatred", "spite", "disgust",
    "irritation", "frustration", "resentment", "hostility",
    "aggression", "bitterness", "contempt", "loathing",
    "annoyance", "outrage", "wrath"
};

static const char* VOID_ANCHORS[] = {
    "emptiness", "numbness", "hollow", "nothing", "absence",
    "void", "dissociation", "detachment", "apathy",
    "indifference", "drift", "blank", "flat", "dead", "cold"
};

static const char* FLOW_ANCHORS[] = {
    "curiosity", "surprise", "wonder", "confusion",
    "anticipation", "ambivalence", "uncertainty", "restless",
    "searching", "transition", "shift", "change", "flux",
    "between", "liminal"
};

static const char* COMPLEX_ANCHORS[] = {
    "shame", "guilt", "envy", "jealousy", "pride",
    "disappointment", "betrayal", "relief", "nostalgia",
    "bittersweet", "melancholy", "regret", "hope",
    "gratitude", "awe"
};

static const char* CHAMBER_NAMES[] = {
    "FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"
};

// All anchors (100 total)
static const char** ALL_ANCHORS[6] = {
    FEAR_ANCHORS, LOVE_ANCHORS, RAGE_ANCHORS,
    VOID_ANCHORS, FLOW_ANCHORS, COMPLEX_ANCHORS
};
static const int ANCHOR_SIZES[6] = {20, 18, 17, 15, 15, 15};

// Last response cache
static CloudResponse last_response;
static char primary_word_buf[64];
static char secondary_word_buf[64];
static int initialized = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// LIBRARY LOADING
// ═══════════════════════════════════════════════════════════════════════════════

// SECURITY: Get directory of running executable for safe library loading
static int get_executable_dir(char* buf, size_t bufsize) {
#ifdef __APPLE__
    uint32_t size = (uint32_t)bufsize;
    if (_NSGetExecutablePath(buf, &size) == 0) {
        char* dir = dirname(buf);
        if (dir) {
            memmove(buf, dir, strlen(dir) + 1);
            return 0;
        }
    }
#else
    ssize_t len = readlink("/proc/self/exe", buf, bufsize - 1);
    if (len > 0) {
        buf[len] = '\0';
        char* dir = dirname(buf);
        if (dir) {
            memmove(buf, dir, strlen(dir) + 1);
            return 0;
        }
    }
#endif
    return -1;
}

static int load_go_library(void) {
    if (go_lib) return 0;  // Already loaded

    // SECURITY: Use absolute paths relative to executable directory
    char exe_dir[PATH_MAX];
    char lib_path[PATH_MAX];

    if (get_executable_dir(exe_dir, sizeof(exe_dir)) != 0) {
        fprintf(stderr, "[cloud] WARNING: Cannot determine executable directory\n");
        return -1;
    }

    // Try library paths relative to executable
#ifdef __APPLE__
    const char* lib_name = "libarianna.dylib";
#else
    const char* lib_name = "libarianna.so";
#endif

    // Try exe_dir/lib/libarianna.{dylib,so}
    snprintf(lib_path, sizeof(lib_path), "%s/lib/%s", exe_dir, lib_name);
    go_lib = dlopen(lib_path, RTLD_NOW);

    if (!go_lib) {
        // Try exe_dir/../lib/libarianna.{dylib,so}
        snprintf(lib_path, sizeof(lib_path), "%s/../lib/%s", exe_dir, lib_name);
        char resolved[PATH_MAX];
        if (realpath(lib_path, resolved)) {
            go_lib = dlopen(resolved, RTLD_NOW);
            if (go_lib) {
                snprintf(lib_path, sizeof(lib_path), "%s", resolved);
            }
        }
    }

    if (go_lib) {
        fprintf(stderr, "[cloud] loaded Go library from %s\n", lib_path);
    } else {
        fprintf(stderr, "[cloud] WARNING: Go library not found, using fallback\n");
        return -1;
    }

    // Load function pointers
    go_cloud_init = (cloud_init_fn)dlsym(go_lib, "cloud_init");
    go_cloud_preprocess = (cloud_preprocess_fn)dlsym(go_lib, "cloud_preprocess");
    go_cloud_get_temperature_bias = (cloud_get_temperature_bias_fn)dlsym(go_lib, "cloud_get_temperature_bias");
    go_cloud_get_primary = (cloud_get_primary_fn)dlsym(go_lib, "cloud_get_primary");
    go_cloud_get_secondary = (cloud_get_secondary_fn)dlsym(go_lib, "cloud_get_secondary");
    go_cloud_get_chamber = (cloud_get_chamber_fn)dlsym(go_lib, "cloud_get_chamber");
    go_cloud_ping = (cloud_ping_fn)dlsym(go_lib, "cloud_ping");
    go_cloud_stop = (cloud_stop_fn)dlsym(go_lib, "cloud_stop");
    go_cloud_free = (cloud_free_fn)dlsym(go_lib, "cloud_free");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

void cloud_init(void) {
    if (initialized) return;

    if (load_go_library() == 0 && go_cloud_init) {
        go_cloud_init("weights/cloud");
    }

    memset(&last_response, 0, sizeof(last_response));
    initialized = 1;
    fprintf(stderr, "[cloud] initialized (wrapper)\n");
}

CloudResponse cloud_ping(const char* text) {
    CloudResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!initialized) cloud_init();

    if (go_lib && go_cloud_preprocess) {
        // Use Go implementation
        resp.iterations = go_cloud_preprocess(text);

        // Get results
        if (go_cloud_get_primary) {
            char* p = go_cloud_get_primary();
            if (p) {
                strncpy(primary_word_buf, p, sizeof(primary_word_buf) - 1);
                resp.primary_word = primary_word_buf;
                free(p);
            }
        }

        if (go_cloud_get_secondary) {
            char* s = go_cloud_get_secondary();
            if (s) {
                strncpy(secondary_word_buf, s, sizeof(secondary_word_buf) - 1);
                resp.secondary_word = secondary_word_buf;
                free(s);
            }
        }

        if (go_cloud_get_chamber) {
            resp.chambers[0] = go_cloud_get_chamber("FEAR");
            resp.chambers[1] = go_cloud_get_chamber("LOVE");
            resp.chambers[2] = go_cloud_get_chamber("RAGE");
            resp.chambers[3] = go_cloud_get_chamber("VOID");
            resp.chambers[4] = go_cloud_get_chamber("FLOW");
            resp.chambers[5] = go_cloud_get_chamber("COMPLEX");
        }

        // Find primary chamber
        float max_act = 0;
        int max_idx = 0;
        for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
            if (resp.chambers[i] > max_act) {
                max_act = resp.chambers[i];
                max_idx = i;
            }
        }
        resp.primary_strength = max_act;
        resp.primary_chamber = CHAMBER_NAMES[max_idx];

    } else {
        // Fallback: simple keyword matching
        resp.iterations = 1;
        resp.primary_word = "neutral";
        resp.secondary_word = "neutral";
        resp.primary_chamber = "FLOW";
    }

    last_response = resp;
    return resp;
}

const char* cloud_get_anchor(int idx) {
    if (idx < 0 || idx >= CLOUD_N_ANCHORS) return NULL;

    int offset = 0;
    for (int c = 0; c < 6; c++) {
        if (idx < offset + ANCHOR_SIZES[c]) {
            return ALL_ANCHORS[c][idx - offset];
        }
        offset += ANCHOR_SIZES[c];
    }
    return NULL;
}

const char* cloud_get_chamber_name(int idx) {
    if (idx < 0 || idx >= CLOUD_N_CHAMBERS) return NULL;
    return CHAMBER_NAMES[idx];
}

int cloud_get_anchor_chamber(int anchor_idx) {
    if (anchor_idx < 0 || anchor_idx >= CLOUD_N_ANCHORS) return -1;

    int offset = 0;
    for (int c = 0; c < 6; c++) {
        if (anchor_idx < offset + ANCHOR_SIZES[c]) {
            return c;
        }
        offset += ANCHOR_SIZES[c];
    }
    return -1;
}

void cloud_crossfire(float* chambers, int max_iterations) {
    // Coupling matrix
    static const float COUPLING[6][6] = {
        {0.0f, -0.3f, +0.6f, +0.4f, -0.2f, +0.3f},
        {-0.3f, 0.0f, -0.6f, -0.5f, +0.3f, +0.4f},
        {+0.3f, -0.4f, 0.0f, +0.2f, -0.3f, +0.2f},
        {+0.5f, -0.7f, +0.3f, 0.0f, -0.4f, +0.5f},
        {-0.2f, +0.2f, -0.2f, -0.3f, 0.0f, +0.2f},
        {+0.3f, +0.2f, +0.2f, +0.3f, +0.1f, 0.0f},
    };

    static const float DECAY[6] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};
    float momentum = 0.7f;
    float threshold = 0.01f;

    for (int iter = 0; iter < max_iterations; iter++) {
        float new_chambers[6];
        float delta = 0;

        for (int i = 0; i < 6; i++) {
            chambers[i] *= DECAY[i];

            float influence = 0;
            for (int j = 0; j < 6; j++) {
                influence += COUPLING[j][i] * chambers[j];
            }

            new_chambers[i] = momentum * chambers[i] + (1 - momentum) * influence;
            if (new_chambers[i] < 0) new_chambers[i] = 0;
            if (new_chambers[i] > 1) new_chambers[i] = 1;

            float d = new_chambers[i] - chambers[i];
            if (d < 0) d = -d;
            delta += d;
        }

        memcpy(chambers, new_chambers, sizeof(float) * 6);
        if (delta < threshold) break;
    }
}

float cloud_temperature_bias(const CloudResponse* resp) {
    if (!resp) return 0.0f;

    if (go_lib && go_cloud_get_temperature_bias) {
        return go_cloud_get_temperature_bias();
    }

    // Fallback calculation
    float bias = 0.0f;
    bias += resp->chambers[CLOUD_CHAMBER_FEAR] * 0.15f;
    bias += resp->chambers[CLOUD_CHAMBER_RAGE] * 0.10f;
    bias -= resp->chambers[CLOUD_CHAMBER_LOVE] * 0.10f;
    bias -= resp->chambers[CLOUD_CHAMBER_VOID] * 0.05f;

    if (bias > 0.2f) bias = 0.2f;
    if (bias < -0.2f) bias = -0.2f;

    return bias;
}

void cloud_apply_emotion_to_logits(float* logits, int vocab_size,
                                   const CloudResponse* resp) {
    if (!logits || !resp) return;

    // Emotional bias on vocabulary (simplified)
    float fear_boost = resp->chambers[CLOUD_CHAMBER_FEAR] * 0.1f;
    float love_boost = resp->chambers[CLOUD_CHAMBER_LOVE] * 0.1f;

    // Apply subtle modulation (placeholder for real implementation)
    for (int i = 0; i < vocab_size; i++) {
        logits[i] += (fear_boost - love_boost) * 0.05f;
    }
}

int cloud_needs_care(const CloudResponse* resp) {
    if (!resp) return 0;
    return (resp->chambers[CLOUD_CHAMBER_FEAR] > 0.6f ||
            resp->chambers[CLOUD_CHAMBER_VOID] > 0.7f);
}

int cloud_needs_grounding(const CloudResponse* resp) {
    if (!resp) return 0;
    return (resp->chambers[CLOUD_CHAMBER_VOID] > 0.8f);
}

int cloud_needs_warmth(const CloudResponse* resp) {
    if (!resp) return 0;
    return (resp->chambers[CLOUD_CHAMBER_LOVE] > 0.6f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLEANUP
// ═══════════════════════════════════════════════════════════════════════════════

__attribute__((destructor))
static void cloud_cleanup(void) {
    if (go_lib) {
        if (go_cloud_free) go_cloud_free();
        dlclose(go_lib);
        go_lib = NULL;
    }
}
