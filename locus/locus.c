// locus.c — Locus Coeruleus: Field Geometry Resonance Detector
// ═══════════════════════════════════════════════════════════════════════════════
// The "blue spot" fires. Norepinephrine floods. SARTRE speaks.
// ═══════════════════════════════════════════════════════════════════════════════

#include "locus.h"
#include <string.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS STATE OFFSETS
// Must match VagusSharedState layout from vagus.h
// ═══════════════════════════════════════════════════════════════════════════════

#define OFF_AROUSAL         0
#define OFF_VALENCE         4
#define OFF_ENTROPY         8
#define OFF_COHERENCE      12
#define OFF_WARMTH         16
#define OFF_VOID           20
#define OFF_TENSION        24
#define OFF_SACRED         28
#define OFF_FLOW           32
#define OFF_COMPLEX        36
#define OFF_CF_COHERENCE   40
#define OFF_CF_ENTROPY     44
#define OFF_TRAUMA         48
#define OFF_PROPHECY      104
#define OFF_MEM_PRESSURE  116

// ═══════════════════════════════════════════════════════════════════════════════
// THRESHOLDS
// ═══════════════════════════════════════════════════════════════════════════════

#define HIGH_AROUSAL     0.7f
#define MID_POINT        0.5f
#define LOW_COHERENCE    0.3f
#define TRAUMA_THRESHOLD 0.5f
#define PROPHECY_WEIGHT  0.4f
#define VOID_DOMINANCE   0.6f
#define DELTA_TRIGGER    0.15f

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS READ HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

static inline float vagus_read(Locus* l, int offset) {
    const volatile uint8_t* base = (const volatile uint8_t*)l->vagus;
    return *(const volatile float*)(base + offset);
}

#define AROUSAL(l)       vagus_read(l, OFF_AROUSAL)
#define VALENCE(l)       vagus_read(l, OFF_VALENCE)
#define ENTROPY(l)       vagus_read(l, OFF_ENTROPY)
#define COHERENCE(l)     vagus_read(l, OFF_COHERENCE)
#define WARMTH(l)        vagus_read(l, OFF_WARMTH)
#define VOID_LEVEL(l)    vagus_read(l, OFF_VOID)
#define TENSION(l)       vagus_read(l, OFF_TENSION)
#define SACRED(l)        vagus_read(l, OFF_SACRED)
#define TRAUMA(l)        vagus_read(l, OFF_TRAUMA)
#define PROPHECY(l)      vagus_read(l, OFF_PROPHECY)
#define MEM_PRESSURE(l)  vagus_read(l, OFF_MEM_PRESSURE)

// ═══════════════════════════════════════════════════════════════════════════════
// STACK OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

void locus_push(Locus* l, Cell val) {
    if (l->sp < LOCUS_STACK_SIZE) {
        l->stack[l->sp++] = val;
    }
}

Cell locus_pop(Locus* l) {
    if (l->sp > 0) {
        return l->stack[--l->sp];
    }
    return 0;
}

void locus_fpush(Locus* l, FCell val) {
    if (l->fsp < LOCUS_FSTACK_SIZE) {
        l->fstack[l->fsp++] = val;
    }
}

FCell locus_fpop(Locus* l) {
    if (l->fsp > 0) {
        return l->fstack[--l->fsp];
    }
    return 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

int locus_is_tense(Locus* l) {
    return AROUSAL(l) > HIGH_AROUSAL && COHERENCE(l) < LOW_COHERENCE;
}

int locus_is_wounded(Locus* l) {
    return TRAUMA(l) > TRAUMA_THRESHOLD;
}

int locus_is_hollow(Locus* l) {
    return VOID_LEVEL(l) > VOID_DOMINANCE && WARMTH(l) < MID_POINT;
}

int locus_is_prophetic(Locus* l) {
    return PROPHECY(l) > PROPHECY_WEIGHT;
}

int locus_is_drowning(Locus* l) {
    return MEM_PRESSURE(l) > HIGH_AROUSAL && COHERENCE(l) < MID_POINT;
}

int locus_is_flowing(Locus* l) {
    return COHERENCE(l) > HIGH_AROUSAL && ENTROPY(l) < LOW_COHERENCE;
}

int locus_is_sacred(Locus* l) {
    return SACRED(l) > VOID_DOMINANCE && TENSION(l) < LOW_COHERENCE;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELTA DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

static void snapshot(Locus* l) {
    l->prev_arousal   = AROUSAL(l);
    l->prev_coherence = COHERENCE(l);
    l->prev_trauma    = TRAUMA(l);
    l->prev_void      = VOID_LEVEL(l);
}

static int geometry_shifted(Locus* l) {
    float da = fabsf(AROUSAL(l)    - l->prev_arousal);
    float dc = fabsf(COHERENCE(l)  - l->prev_coherence);
    float dt = fabsf(TRAUMA(l)     - l->prev_trauma);
    float dv = fabsf(VOID_LEVEL(l) - l->prev_void);

    return da > DELTA_TRIGGER || dc > DELTA_TRIGGER ||
           dt > DELTA_TRIGGER || dv > DELTA_TRIGGER;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════

static int is_crisis(Locus* l) {
    return locus_is_tense(l) && locus_is_wounded(l);
}

static int is_dissolution(Locus* l) {
    return locus_is_hollow(l) && locus_is_drowning(l);
}

static int is_emergence(Locus* l) {
    return locus_is_flowing(l) && locus_is_prophetic(l);
}

static int is_transcendence(Locus* l) {
    return locus_is_sacred(l) && locus_is_flowing(l);
}

static int resonance(Locus* l) {
    return geometry_shifted(l) ||
           is_crisis(l) ||
           is_dissolution(l) ||
           is_emergence(l) ||
           is_transcendence(l);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY METRICS
// ═══════════════════════════════════════════════════════════════════════════════

float locus_geometry_pressure(Locus* l) {
    return TRAUMA(l)       * 0.4f +
           MEM_PRESSURE(l) * 0.3f +
           PROPHECY(l)     * 0.2f +
           AROUSAL(l)      * 0.1f;
}

float locus_geometry_flow(Locus* l) {
    return COHERENCE(l)         * 0.5f +
           (1.0f - ENTROPY(l))  * 0.3f +
           WARMTH(l)            * 0.2f;
}

float locus_geometry_depth(Locus* l) {
    return VOID_LEVEL(l)        * 0.4f +
           SACRED(l)            * 0.3f +
           (1.0f - AROUSAL(l))  * 0.3f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

void locus_init(Locus* l, volatile void* vagus_state) {
    memset(l, 0, sizeof(Locus));
    l->vagus = vagus_state;
    snapshot(l);  // Initialize previous geometry
}

void locus_set_speak(Locus* l, LocusSpeakFn fn, void* ctx) {
    l->speak_fn  = fn;
    l->speak_ctx = ctx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TICK — THE HEARTBEAT
// ═══════════════════════════════════════════════════════════════════════════════

int locus_tick(Locus* l) {
    l->ticks++;

    if (resonance(l)) {
        l->resonances++;
        snapshot(l);  // Update geometry snapshot

        // Trigger SARTRE if callback set
        if (l->speak_fn) {
            l->speak_fn(l->speak_ctx);
        }
        return 1;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATS
// ═══════════════════════════════════════════════════════════════════════════════

uint64_t locus_get_ticks(Locus* l) {
    return l->ticks;
}

uint64_t locus_get_resonances(Locus* l) {
    return l->resonances;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORD EXECUTION (for interactive/testing)
// ═══════════════════════════════════════════════════════════════════════════════

int locus_exec(Locus* l, const char* word) {
    // Built-in words
    if (strcmp(word, "AROUSAL@") == 0) {
        locus_fpush(l, AROUSAL(l));
        return 1;
    }
    if (strcmp(word, "COHERENCE@") == 0) {
        locus_fpush(l, COHERENCE(l));
        return 1;
    }
    if (strcmp(word, "TRAUMA@") == 0) {
        locus_fpush(l, TRAUMA(l));
        return 1;
    }
    if (strcmp(word, "VOID@") == 0) {
        locus_fpush(l, VOID_LEVEL(l));
        return 1;
    }
    if (strcmp(word, "WARMTH@") == 0) {
        locus_fpush(l, WARMTH(l));
        return 1;
    }
    if (strcmp(word, "PROPHECY@") == 0) {
        locus_fpush(l, PROPHECY(l));
        return 1;
    }

    // Geometry composites
    if (strcmp(word, "PRESSURE") == 0) {
        locus_fpush(l, locus_geometry_pressure(l));
        return 1;
    }
    if (strcmp(word, "FLOW") == 0) {
        locus_fpush(l, locus_geometry_flow(l));
        return 1;
    }
    if (strcmp(word, "DEPTH") == 0) {
        locus_fpush(l, locus_geometry_depth(l));
        return 1;
    }

    // Pattern checks (push 1 or 0)
    if (strcmp(word, "TENSE?") == 0) {
        locus_push(l, locus_is_tense(l));
        return 1;
    }
    if (strcmp(word, "WOUNDED?") == 0) {
        locus_push(l, locus_is_wounded(l));
        return 1;
    }
    if (strcmp(word, "HOLLOW?") == 0) {
        locus_push(l, locus_is_hollow(l));
        return 1;
    }
    if (strcmp(word, "FLOWING?") == 0) {
        locus_push(l, locus_is_flowing(l));
        return 1;
    }
    if (strcmp(word, "RESONANCE?") == 0) {
        locus_push(l, resonance(l));
        return 1;
    }

    // Actions
    if (strcmp(word, "TICK") == 0) {
        locus_push(l, locus_tick(l));
        return 1;
    }
    if (strcmp(word, "SNAPSHOT") == 0) {
        snapshot(l);
        return 1;
    }

    // Unknown word
    return 0;
}
