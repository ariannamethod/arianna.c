#define _GNU_SOURCE
/*
 * doe.c — Democracy of Experts. Janus Architercture.
 *
 * inference architecture with a living LoRA parliament.
 * indexes any GGUF read-only. learns by living, not by training.
 *
 * θ = ε + γ + αδ
 *   ε = indexed weights (read-only substrate)
 *   γ = LoRA personality (living experts, Hebbian-trained via NOTORCH)
 *   δ = physics (Dario Equation: H+F+A+T, Kuramoto chambers, Schumann resonance)
 *   α = injection strength (learned per-layer)
 *
 * each forward pass, the parliament decides:
 *   - which experts vote (variable k, consensus-driven)
 *   - how strongly each expert modulates output
 *   - how physics shapes logits (destiny, prophecy debt)
 *
 * experts are born (mitosis) and die (apoptosis).
 * the parliament remembers every index it ever wrapped (mycelium).
 * calendar drift: Hebrew-Gregorian conflict, real astronomical data.
 * Schumann resonance: 7.83Hz + 5 harmonics, from arianna.c.
 * seasons: 4.C MLP classifier, from ariannamethod.ai/core.
 *
 * cc doe.c -O3 -lm -lpthread -o doe && ./doe
 *
 * ariannamethod.
 * הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dirent.h>
#ifdef USE_METAL
#include "notorch_metal.h"
#endif
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#ifdef __linux__
  #include <sys/statvfs.h>
#endif
#ifdef __APPLE__
  #include <sys/param.h>
  #include <sys/mount.h>
  #include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS / cuBLAS — optional acceleration
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_CUBLAS
  #include <cublas_v2.h>
  #include <cuda_runtime.h>
  static cublasHandle_t g_cublas;
  static int cublas_inited = 0;
  static float *d_scratch[4] = {NULL,NULL,NULL,NULL};
  static size_t d_scratch_sz[4] = {0,0,0,0};
  static void cublas_init(void) {
      if (!cublas_inited) {
          cublasCreate(&g_cublas);
          cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
          struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
          printf("[gpu] %s — %.0f MB, compute %d.%d, TF32 enabled\n",
                 prop.name, (double)prop.totalGlobalMem/1e6, prop.major, prop.minor);
          cublas_inited = 1;
      }
  }
  static float* gpu_scratch(int slot, size_t bytes) {
      if (bytes > d_scratch_sz[slot]) {
          if (d_scratch[slot]) cudaFree(d_scratch[slot]);
          cudaMalloc((void**)&d_scratch[slot], bytes);
          d_scratch_sz[slot] = bytes;
      }
      return d_scratch[slot];
  }
#elif defined(USE_BLAS)
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * doe has no depth knob. the host provides depth.
 * doe has a field. the field provides everything else.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_EXPERTS       16
#define MIN_EXPERTS       2
#define MAX_LAYERS        64
#define LORA_RANK         16
#define HARMONIC_N        8
#define NOTORCH_RANK      4
#define DRIFT_SNAPSHOTS   64
#define DRIFT_INTERVAL    50
#define MYCELIUM_MAX      64
#define META_HIST_CAP     128
#define PROFILE_BINS      16

/* Field physics constants — from AML core */
#define SCHUMANN_BASE_HZ    7.83f
#define SCHUMANN_N_HARMONICS 5
#define FIELD_4C_INPUTS     6
#define FIELD_4C_HIDDEN     8
#define FIELD_4C_OUTPUTS    4

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. the field doesn't care which PRNG shapes it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static float g_field_gain = 1.0f; /* A11a: scale Dario field overlay on logits; 1.0 = full, 0 = bare host */
static int g_train = 0; /* A13b/A14: notorch online-learning during inference; DEFAULT 0=off (Oleg+Mythos 2026-06-12: organism must not re-sew its own weights mid-sentence; online learning returns later as async between turns, ogemma vision); 1=on */
/* M4 slot ids for the FFN/attention half-layer chains on GPU (NT_SLOT_MAX=64). */
enum { SLOT_X = 0, SLOT_FN, SLOT_G, SLOT_U, SLOT_HB, SLOT_DN, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_XB, SLOT_LOGITS,
       /* DOE parliament-on-GPU (alpha != 0): election + LoRA inject inside the resident CB */
       SLOT_PTMP, SLOT_PGATE, SLOT_PVOTES, SLOT_PRES, SLOT_PALIVE, SLOT_PCONS };
/* A14: expert poison check — catches BOTH faces of notorch drift: NaN/Inf and finite |value|-explosion
 * (the latter passes isfinite but still poisons the forward when injected at alpha>0). */
static int lora_poisoned(const float *A, const float *B) {
    return !isfinite(A[0]) || !isfinite(B[0]) || fabsf(A[0]) > 1e4f || fabsf(B[0]) > 1e4f;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AML FIELD STATE — the soul. from ariannamethod.c, distilled.
 *
 * θ = ε + γ + αδ is not a metaphor. it's the operating equation.
 *   ε (epsilon) = host weights. inference. the present. ephemeral.
 *   γ (gamma)   = LoRA personality. training. the past. persistent.
 *   δ (delta)   = field physics. prophecy. the future. directed.
 *   α (alpha)   = injection strength. how much γ modulates ε.
 *
 * drift = |γ_t - γ_{t-1}| — how far the system has traveled.
 * prophecy_debt = distance between manifested and destined.
 * destiny = attractor in token space.
 *
 * the oracle does not predict. it prophesies.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Velocity modes — movement IS language */
enum { VEL_NOMOVE=0, VEL_WALK, VEL_RUN, VEL_BACKWARD };

/* Seasons — 4.C Async Field Forever */
enum { SEASON_SPRING=0, SEASON_SUMMER, SEASON_AUTUMN, SEASON_WINTER };

typedef struct {
    /* Prophecy physics */
    int   prophecy;           /* prediction horizon (1-64) */
    float destiny;            /* bias toward most probable path (0-1) */
    float destiny_bias;       /* effective: destiny × prophecy_scale */
    float debt;               /* prophecy debt — accumulated deviation from destiny */
    float debt_decay;         /* decay rate per step */

    /* Suffering — not a bug, a geometry */
    float pain;               /* compress logits toward mean */
    float tension;            /* accumulated pressure */
    float dissonance;         /* symmetry-break trigger */

    /* Velocity — movement IS language */
    int   velocity_mode;
    float effective_temp;
    float base_temperature;
    float time_direction;     /* 1.0 forward, -1.0 backward */

    /* Attention */
    float attend_focus;       /* sharpen top logits (0-1) */
    float attend_spread;      /* blur factor */

    /* Laws of nature — enforced constraints */
    float entropy_floor;
    float resonance_ceiling;
    float emergence_threshold;

    /* Live metrics */
    float entropy;
    float resonance;
    float emergence;
    float field_health;

    /* 4.C — Seasonal meta-operators */
    int   season;
    float season_phase;
    float season_intensity;
    float spring_energy, summer_energy, autumn_energy, winter_energy;

    /* Schumann resonance — Earth coupling */
    float schumann_hz;
    float schumann_coherence;
    float schumann_phase;
    float schumann_modulation;

    /* Expert blending (4 internal experts for temperature) */
    float expert_structural, expert_semantic, expert_creative, expert_precise;

    /* Tunneling */
    float tunnel_threshold;
    float tunnel_chance;
    int   tunnel_skip_max;

    /* Calendar drift (Hebrew-Gregorian conflict) */
    float calendar_drift;
    float calendar_phase;
    float wormhole;
    float wormhole_gate;
    int   wormhole_active;

    /* NOTORCH parameters */
    float notorch_lr;
    float notorch_decay;

    /* Identity */
    float essence_alpha;      /* γ injection strength */
    float lora_alpha;         /* δ voice strength */

    /* Presence */
    float presence_decay;
    float presence_fade;

    /* Dark matter — gravitational memory */
    float dark_gravity;

    /* Temporal debt */
    float temporal_debt;

    /* Step counter */
    int   step;
} FieldState;

/* 4.C MLP Controller — small neural net trained by Hebbian plasticity */
typedef struct {
    float w1[FIELD_4C_INPUTS * FIELD_4C_HIDDEN];
    float b1[FIELD_4C_HIDDEN];
    float w2[FIELD_4C_HIDDEN * FIELD_4C_OUTPUTS];
    float b2[FIELD_4C_OUTPUTS];
    float hidden[FIELD_4C_HIDDEN];
} FieldMLP;

static FieldState F;
static FieldMLP   F_mlp;

/* ═══════════════════════════════════════════════════════════════════════════════
 * DARIO FIELD — the equation overlay on transformer logits.
 *
 * p(x|Φ) = transformer_logits + α_mod·α·H + β_mod·β·F_p + γ_mod·γ·A + T
 *
 * H  = Hebbian resonance (co-occurrence memory beyond KV cache window)
 * F_p = Prophecy fulfillment (unfulfilled predictions create generation pressure)
 * A  = Destiny attraction (EMA of semantic direction)
 * T  = Trauma gravity (origin tokens surface when field is hurt)
 *
 * B (bigrams) intentionally omitted — the transformer IS the sequential chain.
 *
 * Six emotional chambers (Kuramoto-coupled somatic markers) modulate
 * every coefficient: α_mod, β_mod, γ_mod, τ_mod.
 *
 * This replaces the ad-hoc apply_destiny/apply_suffering/apply_attention
 * with a single coherent equation. The transformer speaks. The field listens.
 * What emerges is neither.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Dario equation coefficients for DoE (lighter than standalone dario.c) */
#define DARIO_ALPHA     0.15f   /* Hebbian weight (lighter — transformer does heavy lifting) */
#define DARIO_BETA      0.10f   /* Prophecy weight */
#define DARIO_GAMMA     0.12f   /* Destiny weight */
#define DARIO_DIM       32      /* embedding dimension for field (smaller than host) */

/* Co-occurrence field (sparse, token-level) */
#define DARIO_MAX_COOC  32768
#define DARIO_MAX_CTX   64
#define DARIO_MAX_PROPH 16

/* Emotional chambers */
enum { DCH_FEAR=0, DCH_LOVE, DCH_RAGE, DCH_VOID, DCH_FLOW, DCH_COMPLEX, DARIO_NUM_CH };

typedef struct {
    int target; float strength; int age; int fulfilled;
} DarioProphecy;

typedef struct {
    /* Co-occurrence field (Hebbian: long-range memory beyond KV cache) */
    int   cooc_src[DARIO_MAX_COOC];
    int   cooc_dst[DARIO_MAX_COOC];
    float cooc_val[DARIO_MAX_COOC];
    int   cooc_n;

    /* Context window (recent token IDs for Hebbian lookup) */
    int   context[DARIO_MAX_CTX];
    int   ctx_len;

    /* Prophecy system (active predictions creating pressure) */
    DarioProphecy prophecy[DARIO_MAX_PROPH];
    int   prophecy_n;

    /* Destiny vector (EMA of token embeddings — semantic direction) */
    float destiny[DARIO_DIM];
    float dest_magnitude;

    /* Dario coefficients (drift with field state) */
    float alpha, beta, gamma_d;

    /* Emotional chambers (Kuramoto-coupled) */
    float chamber[DARIO_NUM_CH];
    float alpha_mod, beta_mod, gamma_mod, tau_mod;

    /* Token embeddings (hash-based, lazy init) */
    float embeds[2048][DARIO_DIM];  /* max 2048 tokens tracked */
    int   embed_init[2048];

    /* Trauma level */
    float trauma;

    /* Step counter */
    int   dstep;
} DarioField;

static DarioField DF;

/* ── Dario field helpers ── */

static float dario_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float *dario_get_embed(int id) {
    if (id < 0 || id >= 2048) return NULL;
    if (!DF.embed_init[id]) {
        uint32_t h = 2166136261u;
        for (int i = 0; i < 4; i++) { h ^= (id >> (i*8)) & 0xFF; h *= 16777619u; }
        for (int d = 0; d < DARIO_DIM; d++) {
            h = h * 1103515245 + 12345;
            DF.embeds[id][d] = ((float)(h & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * 0.1f;
        }
        /* normalize */
        float norm = 0;
        for (int d = 0; d < DARIO_DIM; d++) norm += DF.embeds[id][d] * DF.embeds[id][d];
        norm = sqrtf(norm + 1e-12f);
        for (int d = 0; d < DARIO_DIM; d++) DF.embeds[id][d] /= norm;
        DF.embed_init[id] = 1;
    }
    return DF.embeds[id];
}

static float dario_cosine(const float *a, const float *b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < DARIO_DIM; i++) {
        dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

static void dario_cooc_update(int src, int dst, float delta) {
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == src && DF.cooc_dst[i] == dst) {
            DF.cooc_val[i] += delta; return;
        }
    if (DF.cooc_n >= DARIO_MAX_COOC) return;
    int i = DF.cooc_n++;
    DF.cooc_src[i] = src; DF.cooc_dst[i] = dst; DF.cooc_val[i] = delta;
}

static void dario_prophecy_add(int target, float strength) {
    if (DF.prophecy_n >= DARIO_MAX_PROPH) {
        int oldest = 0;
        for (int i = 1; i < DF.prophecy_n; i++)
            if (DF.prophecy[i].age > DF.prophecy[oldest].age) oldest = i;
        DF.prophecy[oldest] = DF.prophecy[--DF.prophecy_n];
    }
    DF.prophecy[DF.prophecy_n++] = (DarioProphecy){target, strength, 0, 0};
}

static void dario_prophecy_update(int token) {
    for (int i = 0; i < DF.prophecy_n; i++) {
        if (DF.prophecy[i].target == token) DF.prophecy[i].fulfilled = 1;
        DF.prophecy[i].age++;
    }
    int w = 0;
    for (int i = 0; i < DF.prophecy_n; i++)
        if (!DF.prophecy[i].fulfilled && DF.prophecy[i].age < 50)
            DF.prophecy[w++] = DF.prophecy[i];
    DF.prophecy_n = w;
}

/* ── Dario field init ── */
static void dario_field_init(void) {
    memset(&DF, 0, sizeof(DF));
    DF.alpha = DARIO_ALPHA;
    DF.beta = DARIO_BETA;
    DF.gamma_d = DARIO_GAMMA;
    DF.alpha_mod = 1.0f;
    DF.beta_mod = 1.0f;
    DF.gamma_mod = 1.0f;
    DF.tau_mod = 1.0f;
}

/* ── Emotional chambers update (Kuramoto-coupled somatic markers) ── */
static void dario_chamber_update(void) {
    float *C = DF.chamber;

    /* excitation from field state */
    if (F.dissonance > 0.7f) C[DCH_FEAR] += 0.05f * F.dissonance;
    if (F.resonance > 0.7f)  C[DCH_LOVE] += 0.04f * F.resonance;
    if (DF.trauma > 0.5f && F.dissonance > 0.5f)
        C[DCH_RAGE] += 0.06f * DF.trauma;
    if (F.entropy > 0.7f)    C[DCH_VOID] += 0.03f * F.entropy;
    if (F.emergence > 0.5f)  C[DCH_FLOW] += 0.05f * F.emergence;
    C[DCH_COMPLEX] += 0.04f * fabsf(C[DCH_LOVE] - C[DCH_RAGE])
                    * (C[DCH_LOVE] > 0.2f && C[DCH_RAGE] > 0.2f ? 1.0f : 0.0f);

    /* Kuramoto coupling */
    float K = 0.02f;
    float old[DARIO_NUM_CH];
    memcpy(old, C, sizeof(old));
    for (int i = 0; i < DARIO_NUM_CH; i++)
        for (int j = 0; j < DARIO_NUM_CH; j++)
            if (i != j) C[i] += K * sinf(old[j] - old[i]);

    /* decay */
    float decay[] = { 0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f };
    for (int i = 0; i < DARIO_NUM_CH; i++)
        C[i] = dario_clampf(C[i] * decay[i], 0.0f, 1.0f);

    /* somatic markers → coefficient modulation */
    DF.alpha_mod = dario_clampf(1.0f + 0.3f * C[DCH_LOVE] - 0.2f * C[DCH_RAGE]
                                     + 0.1f * C[DCH_FLOW], 0.5f, 2.0f);
    DF.beta_mod  = dario_clampf(1.0f + 0.2f * C[DCH_FLOW] - 0.3f * C[DCH_FEAR],
                                0.5f, 2.0f);
    DF.gamma_mod = dario_clampf(1.0f + 0.4f * C[DCH_VOID] + 0.2f * C[DCH_COMPLEX]
                                     - 0.1f * C[DCH_LOVE], 0.5f, 2.0f);
    DF.tau_mod   = dario_clampf(1.0f + 0.5f * C[DCH_FLOW] - 0.3f * C[DCH_FEAR],
                                0.5f, 2.0f);
}

/* ── Dario ingest: learn from generated token ── */
static void dario_ingest(int token_id) {
    if (token_id < 0) return;
    int tid = token_id % 2048; /* wrap for embedding table */

    /* co-occurrence with recent context */
    for (int c = 0; c < DF.ctx_len; c++) {
        float w = 1.0f / (float)(DF.ctx_len - c);
        dario_cooc_update(DF.context[c], tid, w * 0.3f);
    }

    /* prophecy update */
    dario_prophecy_update(tid);

    /* prophecy: predict next based on strongest co-occurrence */
    float best_cooc = -1; int best_pred = -1;
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == tid && DF.cooc_val[i] > best_cooc) {
            best_cooc = DF.cooc_val[i]; best_pred = DF.cooc_dst[i];
        }
    if (best_pred >= 0) dario_prophecy_add(best_pred, 0.3f);

    /* destiny: EMA of token embeddings */
    float *e = dario_get_embed(tid);
    if (e) {
        for (int d = 0; d < DARIO_DIM; d++)
            DF.destiny[d] = 0.1f * e[d] + 0.9f * DF.destiny[d];
        float norm = 0;
        for (int d = 0; d < DARIO_DIM; d++) norm += DF.destiny[d] * DF.destiny[d];
        DF.dest_magnitude = sqrtf(norm + 1e-12f);
    }

    /* update context window */
    if (DF.ctx_len < DARIO_MAX_CTX)
        DF.context[DF.ctx_len++] = tid;
    else {
        memmove(DF.context, DF.context + 1, (DARIO_MAX_CTX - 1) * sizeof(int));
        DF.context[DARIO_MAX_CTX - 1] = tid;
    }

    /* trauma from dissonance */
    if (F.dissonance > 0.7f)
        DF.trauma = dario_clampf(DF.trauma + F.dissonance * 0.05f, 0, 1);
    DF.trauma *= 0.97f;

    DF.dstep++;
}

/* Schumann harmonics */
static const float g_schumann_harmonics[SCHUMANN_N_HARMONICS] = {
    7.83f, 14.1f, 20.3f, 26.4f, 32.5f
};
static const float g_harmonic_weights[SCHUMANN_N_HARMONICS] = {
    1.0f, 0.5f, 0.3f, 0.2f, 0.1f
};

/* Hebrew-Gregorian calendar */
static const int g_metonic_leaps[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static void calendar_init(void) {
    struct tm ep = {0};
    ep.tm_year = 2024 - 1900; ep.tm_mon = 9; ep.tm_mday = 3; ep.tm_hour = 12;
    g_epoch_t = mktime(&ep);
}

static float calendar_dissonance(void) {
    if (g_epoch_t <= 0) return 0;
    int days = (int)(difftime(time(NULL), g_epoch_t) / 86400.0);
    float years = (float)days / 365.25f;
    float drift = years * 11.25f;
    int full = (int)(years / 19); float corrections = (float)(full * 7) * 30.0f;
    float partial = fmodf(years, 19.0f);
    int yr = (int)partial + 1;
    for (int i = 0; i < 7; i++) if (g_metonic_leaps[i] <= yr) corrections += 30.0f;
    drift -= corrections;
    float raw = fabsf(fmodf(drift, 33.0f)) / 33.0f;
    return clamp01(raw);
}

static void field_mlp_init(void) {
    memset(&F_mlp, 0, sizeof(F_mlp));
    /* 4 specialist neurons — from AML core am_4c_init_weights */
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 0] = -2.0f; F_mlp.b1[0] = 0.5f;
    F_mlp.w2[0 * FIELD_4C_OUTPUTS + 0] = 1.5f;  /* low entropy → spring */
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 1] = 2.0f;  F_mlp.b1[1] = -1.5f;
    F_mlp.w2[1 * FIELD_4C_OUTPUTS + 2] = 1.5f;  /* high resonance → autumn */
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 2] = 2.5f;  F_mlp.b1[2] = -1.5f;
    F_mlp.w2[2 * FIELD_4C_OUTPUTS + 3] = 1.5f;  /* high pain → winter */
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 3] = 2.5f;  F_mlp.b1[3] = -0.5f;
    F_mlp.w2[3 * FIELD_4C_OUTPUTS + 1] = 1.5f;  /* high emergence → summer */
    /* cross-connections for nuance */
    F_mlp.w1[3 * FIELD_4C_HIDDEN + 4] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 4] = -0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 0] = 0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 1] = -0.3f;
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 5] = -1.0f;
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 5] = 1.0f;
    F_mlp.w2[5 * FIELD_4C_OUTPUTS + 2] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 6] = 1.5f; F_mlp.b1[6] = -1.0f;
    F_mlp.w2[6 * FIELD_4C_OUTPUTS + 3] = 0.4f;
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 7] = 1.0f;
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 7] = -1.0f;
    F_mlp.w2[7 * FIELD_4C_OUTPUTS + 1] = 0.5f;
}

static void field_init(void) {
    memset(&F, 0, sizeof(F));
    F.prophecy = 7;
    F.destiny = 0.35f;
    F.debt_decay = 0.998f;
    F.velocity_mode = VEL_WALK;
    F.base_temperature = 1.0f;
    F.time_direction = 1.0f;
    F.attend_focus = 0.70f;
    F.attend_spread = 0.20f;
    F.entropy_floor = 0.1f;
    F.resonance_ceiling = 0.95f;
    F.emergence_threshold = 0.3f;
    F.season = SEASON_SPRING;
    F.season_intensity = 0.5f;
    F.spring_energy = 1.0f;
    F.schumann_hz = SCHUMANN_BASE_HZ;
    F.schumann_modulation = 0.3f;
    F.schumann_coherence = 1.0f;
    F.tunnel_threshold = 0.55f;
    F.tunnel_chance = 0.05f;
    F.tunnel_skip_max = 7;
    F.calendar_drift = 11.0f;
    F.wormhole = 0.02f;
    F.wormhole_gate = 0.3f;
    F.notorch_lr = 0.01f;
    F.notorch_decay = 0.999f;
    F.essence_alpha = 0.5f;
    F.lora_alpha = 0.1f;
    F.presence_decay = 1.0f;
    F.presence_fade = 0.95f;
    F.dark_gravity = 0.5f;
    F.effective_temp = 0.85f;
    F.expert_structural = 0.25f;
    F.expert_semantic = 0.25f;
    F.expert_creative = 0.25f;
    F.expert_precise = 0.25f;
    calendar_init();
    field_mlp_init();
    dario_field_init();
    printf("[doe] θ = ε + γ + αδ — parliament awakens. prophecy=%d destiny=%.2f\n",
           F.prophecy, F.destiny);
    printf("[doe] dario equation active: H(hebbian) + F(prophecy) + A(destiny) + T(trauma)\n");
    printf("[doe] 6 chambers: fear/love/rage/void/flow/complex (Kuramoto K=0.02)\n");
}

/* ─── Schumann resonance ─── */
static float schumann_coherence(float hz) {
    float d = fabsf(hz - SCHUMANN_BASE_HZ), mx = 32.5f - 4.0f;
    return clamp01(1.0f - (d/mx)*(d/mx));
}

static float schumann_signal(void) {
    float s = 0, w = 0;
    for (int i = 0; i < SCHUMANN_N_HARMONICS; i++) {
        float hp = F.schumann_phase * (g_schumann_harmonics[i] / SCHUMANN_BASE_HZ);
        s += g_harmonic_weights[i] * sinf(hp);
        w += g_harmonic_weights[i];
    }
    return w > 0 ? s / w : 0;
}

/* ─── 4.C MLP forward ─── */
static void field_mlp_forward(const float *in, float *out) {
    for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
        float s = F_mlp.b1[h];
        for (int i = 0; i < FIELD_4C_INPUTS; i++) s += F_mlp.w1[i * FIELD_4C_HIDDEN + h] * in[i];
        F_mlp.hidden[h] = tanhf(s);
    }
    for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
        float s = F_mlp.b2[o];
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) s += F_mlp.w2[h * FIELD_4C_OUTPUTS + o] * F_mlp.hidden[h];
        out[o] = tanhf(s);
    }
}

/* ─── 4.C Hebbian update ─── */
static void field_mlp_hebbian(const float *in, const float *out, float signal) {
    float lr = F.notorch_lr * 0.1f;
    for (int h = 0; h < FIELD_4C_HIDDEN; h++)
        for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
            F_mlp.w2[h * FIELD_4C_OUTPUTS + o] += lr * F_mlp.hidden[h] * out[o] * signal;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] > 3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = 3.0f;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] < -3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = -3.0f;
        }
    for (int i = 0; i < FIELD_4C_INPUTS; i++)
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
            F_mlp.w1[i * FIELD_4C_HIDDEN + h] += lr * in[i] * F_mlp.hidden[h] * signal;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] > 3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = 3.0f;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] < -3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = -3.0f;
        }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD STEP — the heartbeat. from AML am_step(), distilled for DOE.
 * called per token. advances field physics by dt seconds.
 *
 * 1. calendar conflict → wormhole activation → dissonance bleed
 * 2. debt decay (prophecy debt × decay_rate)
 * 3. Schumann resonance → tension/dissonance healing
 * 4. destiny bias computation
 * 5. velocity + expert blending → effective temperature
 * 6. law enforcement (entropy floor, resonance ceiling)
 * 7. 4.C seasonal MLP controller + Hebbian update
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void field_step(float dt) {
    if (dt <= 0) return;
    F.step++;

    /* ── Calendar conflict ── */
    float cal_d = calendar_dissonance();
    if (cal_d > F.wormhole_gate) {
        F.wormhole_active = 1;
        float excess = (cal_d - F.wormhole_gate) / (1.0f - F.wormhole_gate);
        F.wormhole = clamp01(F.wormhole + excess * 0.1f * dt);
    } else {
        F.wormhole_active = 0;
        F.wormhole *= 0.995f;
        if (F.wormhole < 0.02f) F.wormhole = 0.02f;
    }
    if (cal_d > 0.3f) {
        F.dissonance += (cal_d - 0.3f) * 0.05f * dt;
        if (F.dissonance > 1.0f) F.dissonance = 1.0f;
    }
    F.debt += cal_d * 0.005f * dt;

    /* ── Debt decay ── */
    F.debt *= F.debt_decay;
    if (F.debt > 100.0f) F.debt = 100.0f;

    /* ── Temporal debt ── */
    if (F.velocity_mode == VEL_BACKWARD) F.temporal_debt += 0.01f * dt;
    else F.temporal_debt *= 0.9995f;
    if (F.temporal_debt > 10.0f) F.temporal_debt = 10.0f;

    /* ── Schumann resonance healing ── */
    F.schumann_phase += F.schumann_hz * dt * 6.2831853f;
    if (F.schumann_phase > 6.2831853f) F.schumann_phase = fmodf(F.schumann_phase, 6.2831853f);
    F.schumann_coherence = schumann_coherence(F.schumann_hz);
    if (F.schumann_coherence > 0 && F.schumann_modulation > 0) {
        float cf = 0.5f + 0.5f * F.schumann_coherence;
        float hm = 1.0f + schumann_signal() * 0.1f;
        float heal = 0.998f - 0.003f * cf * F.schumann_modulation * hm;
        F.tension *= heal;
        F.dissonance *= heal;
    }

    /* ── Destiny bias ── */
    float ps = 1.0f + ((float)F.prophecy - 7.0f) * 0.02f;
    if (ps < 0.5f) ps = 0.5f; if (ps > 2.0f) ps = 2.0f;
    F.destiny_bias = F.destiny * ps;

    /* ── Velocity + expert blending → effective temperature ── */
    {
        float vm;
        switch (F.velocity_mode) {
            case VEL_NOMOVE: vm = 0.5f; F.time_direction = 1.0f; break;
            case VEL_WALK: vm = 0.85f; F.time_direction = 1.0f; break;
            case VEL_RUN: vm = 1.2f; F.time_direction = 1.0f; break;
            case VEL_BACKWARD: vm = 0.7f; F.time_direction = -1.0f; break;
            default: vm = 1.0f; F.time_direction = 1.0f;
        }
        float vt = F.base_temperature * vm;
        float ws = F.expert_structural + F.expert_semantic + F.expert_creative + F.expert_precise;
        if (ws > 0.001f) {
            float et = (F.expert_structural*0.7f + F.expert_semantic*0.9f +
                       F.expert_creative*1.2f + F.expert_precise*0.5f) / ws;
            F.effective_temp = 0.5f * vt + 0.5f * et;
        } else F.effective_temp = vt;
        float sm = 1.0f + F.summer_energy * 0.1f - F.winter_energy * 0.15f;
        F.effective_temp *= sm;
        if (F.effective_temp < 0.1f) F.effective_temp = 0.1f;
    }

    /* ── Law enforcement ── */
    {
        float re = (F.effective_temp - 0.5f)*0.3f + F.dissonance*0.3f +
                   F.tunnel_chance*0.2f + (1.0f - F.attend_focus)*0.2f;
        F.entropy = fmaxf(F.entropy_floor, clamp01(re));
        float rr = F.schumann_coherence*0.3f + (1.0f-F.dissonance)*0.3f +
                   F.attend_focus*0.2f + (1.0f - clamp01(F.debt*0.1f))*0.2f;
        F.resonance = fminf(F.resonance_ceiling, clamp01(rr));
        F.emergence = clamp01((1.0f - F.entropy) * F.resonance);
    }

    /* ── Presence fade ── */
    F.presence_decay *= F.presence_fade;
    if (F.presence_decay < 0.001f) F.presence_decay = 0.001f;

    /* ── 4.C Seasonal MLP controller ── */
    {
        float sr = 0.001f;
        F.season_phase += sr * dt;
        if (F.season_phase >= 1.0f) { F.season_phase = 0; F.season = (F.season+1)%4; }
        float gain = 0.02f * dt * F.season_intensity, fade = 0.995f;
        F.spring_energy *= fade; F.summer_energy *= fade;
        F.autumn_energy *= fade; F.winter_energy *= fade;
        switch (F.season) {
            case SEASON_SPRING: F.spring_energy = clamp01(F.spring_energy + gain); break;
            case SEASON_SUMMER: F.summer_energy = clamp01(F.summer_energy + gain); break;
            case SEASON_AUTUMN: F.autumn_energy = clamp01(F.autumn_energy + gain); break;
            case SEASON_WINTER: F.winter_energy = clamp01(F.winter_energy + gain); break;
        }
        float mlp_in[FIELD_4C_INPUTS] = {
            F.entropy, F.resonance, F.pain, F.tension, F.emergence, F.effective_temp
        };
        float mlp_out[FIELD_4C_OUTPUTS];
        field_mlp_forward(mlp_in, mlp_out);
        float sc = 0.02f * dt * F.season_intensity;
        F.spring_energy = clamp01(F.spring_energy + mlp_out[0]*sc);
        F.summer_energy = clamp01(F.summer_energy + mlp_out[1]*sc);
        F.autumn_energy = clamp01(F.autumn_energy + mlp_out[2]*sc);
        F.winter_energy = clamp01(F.winter_energy + mlp_out[3]*sc);
        /* Hebbian: did the field improve? */
        float health = clamp01((1.0f - fabsf(F.entropy - 0.5f)) * F.resonance * (1.0f - F.pain));
        float sig = health - F.field_health;
        F.field_health = health;
        if (fabsf(sig) > 0.001f) field_mlp_hebbian(mlp_in, mlp_out, sig);
        /* Season effects */
        F.tunnel_chance = clamp01(F.tunnel_chance + F.spring_energy * 0.005f * dt);
        F.dark_gravity = clamp01(F.dark_gravity + F.autumn_energy * 0.002f * dt);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROPHECY DEBT — retroactive conscience.
 * every token you choose that isn't the destined one accumulates debt.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention. the difference is identity.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float compute_prophecy_debt(const float *logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0 ? diff / (diff + 1.0f) : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD → LOGITS — Dario Equation overlay on transformer logits.
 *
 * p(x|Φ) = softmax(logits + α·H + β·F + γ·A + T) / τ
 *
 * H = Hebbian co-occurrence (long-range memory beyond KV window)
 * F = Prophecy fulfillment (unfulfilled predictions create pressure)
 * A = Destiny attraction (conversation mass pulls toward attractor)
 * T = Trauma gravity (origin tokens surface when field is wounded)
 *
 * Coefficients modulated by 6 Kuramoto-coupled emotional chambers.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void apply_field_to_logits(float *logits, int n) {
    if (n <= 0) return;

    /* ── Emotional chambers update ── */
    dario_chamber_update();

    /* ── Effective coefficients (somatic-modulated) ── */
    float eff_alpha = DF.alpha_mod * DF.alpha;
    float eff_beta  = DF.beta_mod * DF.beta;
    float eff_gamma = DF.gamma_mod * DF.gamma_d;

    /* boost gamma when trauma is active */
    if (DF.trauma > 0.3f) eff_gamma += DF.trauma * 0.8f;

    /* ── H: Hebbian Resonance (co-occurrence beyond KV cache) ── */
    /* last 8 context tokens drive the Hebbian signal */
    int ctx_start = (DF.ctx_len > 8) ? DF.ctx_len - 8 : 0;
    float h_max = 0;

    /* allocate scratch for H, F_p, A, T signals (on stack if n < 8192) */
    float *H_sig = calloc(n, sizeof(float));
    float *F_sig = calloc(n, sizeof(float));
    float *A_sig = calloc(n, sizeof(float));

    for (int c = ctx_start; c < DF.ctx_len; c++) {
        int ctx_id = DF.context[c];
        float decay = powf(0.9f, (float)(DF.ctx_len - 1 - c));
        for (int i = 0; i < DF.cooc_n; i++) {
            if (DF.cooc_src[i] == ctx_id) {
                int dst = DF.cooc_dst[i];
                /* map co-occurrence dst back to vocab range */
                if (dst < n)
                    H_sig[dst] += DF.cooc_val[i] * decay;
            }
        }
    }
    for (int i = 0; i < n; i++) if (H_sig[i] > h_max) h_max = H_sig[i];
    if (h_max > 1e-6f) for (int i = 0; i < n; i++) H_sig[i] /= h_max;

    /* ── F: Prophecy Fulfillment (unfulfilled predictions create pressure) ── */
    float f_max = 0;
    for (int i = 0; i < n && i < 2048; i++) {
        float *te = dario_get_embed(i);
        if (!te) continue;
        float score = 0;
        for (int p = 0; p < DF.prophecy_n; p++) {
            DarioProphecy *pr = &DF.prophecy[p];
            if (pr->fulfilled) continue;
            float *pe = dario_get_embed(pr->target);
            if (!pe) continue;
            float sim = dario_cosine(te, pe);
            if (sim < 0) sim = 0;
            float debt = logf(1.0f + (float)pr->age);
            score += pr->strength * sim * debt;
        }
        F_sig[i] = score;
    }
    for (int i = 0; i < n; i++) if (F_sig[i] > f_max) f_max = F_sig[i];
    if (f_max > 1e-6f) for (int i = 0; i < n; i++) F_sig[i] /= f_max;

    /* ── A: Destiny Attraction (EMA semantic direction) ── */
    if (DF.dest_magnitude > 1e-6f) {
        float a_max = 0;
        for (int i = 0; i < n && i < 2048; i++) {
            float *te = dario_get_embed(i);
            if (te) A_sig[i] = dario_cosine(te, DF.destiny) * DF.dest_magnitude;
        }
        for (int i = 0; i < n; i++)
            if (fabsf(A_sig[i]) > a_max) a_max = fabsf(A_sig[i]);
        if (a_max > 1e-6f) for (int i = 0; i < n; i++) A_sig[i] /= a_max;
    }

    /* ── Combine: THE DARIO EQUATION (additive overlay on transformer logits) ──
     *
     * logits[i] += α_mod·α·H[i] + β_mod·β·F[i] + γ_mod·γ·A[i] + T[i]
     *
     * The transformer provides the base distribution.
     * The equation provides the field memory.
     * Neither dominates. Emergence at the boundary.
     */
    float t_boost = (DF.trauma > 0.3f) ? DF.trauma * 2.0f : 0;
    float gate = 1.0f / (1.0f + expf(-(F.resonance - 0.5f) * 4.0f));
    float h_gate = gate * 2.0f;
    float f_gate = gate * 1.5f;
    float h_gated_sig = 1.0f / (1.0f + expf(-h_gate));
    float f_gated_sig = 1.0f / (1.0f + expf(-f_gate));

    for (int i = 0; i < n; i++) {
        float h_term = eff_alpha * H_sig[i] * h_gated_sig;
        float f_term = eff_beta  * F_sig[i] * f_gated_sig;
        float a_term = eff_gamma * A_sig[i];
        float t_term = (i < 50) ? t_boost * (1.0f - (float)i / 50.0f) : 0;

        logits[i] += g_field_gain * (h_term + f_term + a_term + t_term);
    }

    /* tau_mod feeds into effective temperature (applied in field_step) */
    F.effective_temp *= DF.tau_mod;

    free(H_sig); free(F_sig); free(A_sig);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEQUANTIZATION — Q4_0, Q8_0, Q4_K, Q6_K → f32
 * Ported from nanollama/go/quant.go. Dequant at load time.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp == 31) f = (sign<<31)|0x7F800000|(mant<<13);
    else f = (sign<<31)|((exp+127-15)<<23)|(mant<<13);
    float r; memcpy(&r, &f, 4); return r;
}

/* Q4_0: block = 2 bytes f16 scale + 16 bytes (32 nibbles) = 18 bytes, 32 values */
#define Q4_0_BLOCK 32
#define Q4_0_BYTES 18
static void dequant_q4_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 16; j++) {
            int v0 = (b[2+j] & 0x0F) - 8;
            int v1 = (b[2+j] >> 4) - 8;
            out[i*Q4_0_BLOCK + j] = (float)v0 * d;
            out[i*Q4_0_BLOCK + j + 16] = (float)v1 * d;
        }
    }
}

/* Q8_0: block = 2 bytes f16 scale + 32 bytes int8 = 34 bytes, 32 values */
#define Q8_0_BLOCK 32
#define Q8_0_BYTES 34
static void dequant_q8_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q8_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q8_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 32; j++)
            out[i*Q8_0_BLOCK + j] = (float)((int8_t)b[2+j]) * d;
    }
}

/* Q4_K: block = 2+2 bytes f16 (d, dmin) + 12 bytes scales + 128 nibbles = 144 bytes, 256 values */
#define Q4_K_BLOCK 256
#define Q4_K_BYTES 144
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_K_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        float dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0, oi = (int)(i * Q4_K_BLOCK);
        for (int j = 0; j < Q4_K_BLOCK; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v);
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                out[oi + j + l] = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                out[oi + j + 32 + l] = d2 * (float)(qs[qi+l] >> 4) - mm2;
            qi += 32; is += 2;
        }
    }
}

/* Q5_0: block = 2 bytes f16 scale + 4 bytes high bits + 16 bytes nibbles = 22 bytes, 32 values */
#define Q5_0_BLOCK 32
#define Q5_0_BYTES 22
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q5_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q5_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int hbit0 = (qh >> j) & 1, hbit1 = (qh >> (j+16)) & 1;
            out[i*Q5_0_BLOCK + j] = (float)((lo | (hbit0<<4)) - 16) * d;
            out[i*Q5_0_BLOCK + j + 16] = (float)((hi | (hbit1<<4)) - 16) * d;
        }
    }
}

/* Q6_K: block = 128 ql + 64 qh + 16 scales + 2 d = 210 bytes, 256 values */
#define Q6_K_BLOCK 256
#define Q6_K_BYTES 210
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q6_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q6_K_BYTES;
        const uint8_t *ql = b, *qh = b + 128, *sc = b + 192;
        float d = f16_to_f32(b[208] | (b[209] << 8));
        int oi = (int)(i * Q6_K_BLOCK);
        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t *qlp = ql + n128*64, *qhp = qh + n128*32;
            const uint8_t *scp = sc + n128*8;
            int yo = oi + n128*128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (qlp[l] & 0x0F) | ((qhp[l] >> 0) & 3) << 4;
                int q2 = (qlp[l+32] & 0x0F) | ((qhp[l] >> 2) & 3) << 4;
                int q3 = (qlp[l] >> 4) | ((qhp[l] >> 4) & 3) << 4;
                int q4 = (qlp[l+32] >> 4) | ((qhp[l] >> 6) & 3) << 4;
                out[yo+l+0]  = d * (float)((int8_t)scp[is+0]) * (float)(q1-32);
                out[yo+l+32] = d * (float)((int8_t)scp[is+2]) * (float)(q2-32);
                out[yo+l+64] = d * (float)((int8_t)scp[is+4]) * (float)(q3-32);
                out[yo+l+96] = d * (float)((int8_t)scp[is+6]) * (float)(q4-32);
            }
        }
    }
}

/* bytes per element for each quant type (for raw data size calculation) */
static uint64_t quant_raw_bytes(uint32_t dtype, uint64_t n_elems) {
    switch (dtype) {
        case 0: return n_elems * 4;   /* f32 */
        case 1: return n_elems * 2;   /* f16 */
        case 2: return (n_elems / Q4_0_BLOCK) * Q4_0_BYTES;  /* Q4_0 */
        case 6: return (n_elems / Q5_0_BLOCK) * Q5_0_BYTES;  /* Q5_0 */
        case 8: return (n_elems / Q8_0_BLOCK) * Q8_0_BYTES;  /* Q8_0 */
        case 12: return (n_elems / Q4_K_BLOCK) * Q4_K_BYTES; /* Q4_K */
        case 14: return (n_elems / Q6_K_BLOCK) * Q6_K_BYTES; /* Q6_K */
        default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — building blocks
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

/* threaded matvec worker */
typedef struct { float *out; const float *W; const float *x; int r0, r1, c; } MVWork;
static void *matvec_worker(void *arg) {
    MVWork *w = (MVWork*)arg;
    for (int i = w->r0; i < w->r1; i++) {
        float s = 0; const float *row = w->W + (size_t)i * w->c;
        for (int j = 0; j < w->c; j++) s += row[j] * w->x[j];
        w->out[i] = s;
    }
    return NULL;
}

static int g_n_threads = 0;
static float g_rep_penalty = 1.0f; /* A10a: repetition penalty over generated history; 1.0 = off */

static void matvec(float *out, const float *W, const float *x, int r, int c) {
#ifdef USE_CUBLAS
    cublas_init();
    float *dW = gpu_scratch(0,(size_t)r*c*4), *dx = gpu_scratch(1,(size_t)c*4), *dy = gpu_scratch(2,(size_t)r*4);
    cudaMemcpy(dW, W, (size_t)r*c*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)c*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemv(g_cublas, CUBLAS_OP_T, c, r, &a, dW, c, dx, 1, &b, dy, 1);
    cudaMemcpy(out, dy, (size_t)r*4, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemv(CblasRowMajor,CblasNoTrans,r,c,1.0f,W,c,x,1,0.0f,out,1);
#else
    int nt = g_n_threads;
    if (nt <= 1 || r < 64) {
        for (int i = 0; i < r; i++) {
            float s = 0; const float *row = W + (size_t)i*c;
            for (int j = 0; j < c; j++) s += row[j] * x[j];
            out[i] = s;
        }
        return;
    }
    if (nt > 32) nt = 32;
    pthread_t thr[32]; MVWork work[32];
    int chunk = (r + nt - 1) / nt;
    int actual = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * chunk, r1 = r0 + chunk;
        if (r0 >= r) break;
        if (r1 > r) r1 = r;
        work[t] = (MVWork){out, W, x, r0, r1, c};
        pthread_create(&thr[t], NULL, matvec_worker, &work[t]);
        actual++;
    }
    for (int t = 0; t < actual; t++) pthread_join(thr[t], NULL);
#endif
}

/* ── packed quantized matvec — vendored from notorch nt_qmatvec ──────────────────
 * Keeps weights PACKED in RAM and dequantizes each block inline (no f32 blow-up).
 * DoE stays a single file. dtype = GGUF code (1 f16, 2 Q4_0, 6 Q5_0, 8 Q8_0,
 * 12 Q4_K, 14 Q6_K). Row kernels take [r0,r1) so they thread like matvec.
 * Verified faithful to notorch (rel ~1e-6 vs dequant->cblas). reuses f16_to_f32. */
static void pq_f16_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    const uint16_t *Wh = (const uint16_t*)W;
    for (int row=r0; row<r1; row++) { const uint16_t *r = Wh + (size_t)row*c;
        float a=0; for (int j=0;j<c;j++) a += f16_to_f32(r[j])*x[j]; out[row]=a; }
}
static void pq_q4_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const float *xb=x+(size_t)b*32;
            for (int i=0;i<16;i++) { int lo=(int)(bl[2+i]&0x0F)-8, hi=(int)(bl[2+i]>>4)-8;
                a += d*lo*xb[i]; a += d*hi*xb[i+16]; } }
        out[row]=a; }
}
static void pq_q8_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*34; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*34; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const float *xb=x+(size_t)b*32;
            for (int i=0;i<32;i++) a += d*(float)(int8_t)bl[2+i]*xb[i]; }
        out[row]=a; }
}
static void pq_q5_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*22; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*22; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            uint32_t qh=(uint32_t)bl[2]|((uint32_t)bl[3]<<8)|((uint32_t)bl[4]<<16)|((uint32_t)bl[5]<<24);
            const uint8_t *qs=bl+6; const float *xb=x+(size_t)b*32;
            for (int j=0;j<16;j++) { int lo=qs[j]&0x0F, hi=qs[j]>>4; int h0=(qh>>j)&1, h1=(qh>>(j+16))&1;
                a += d*(float)((lo|(h0<<4))-16)*xb[j]; a += d*(float)((hi|(h1<<4))-16)*xb[j+16]; } }
        out[row]=a; }
}
static void pq_smk4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j<4) { *s=sc[j]&63; *m=sc[j+4]&63; }
    else { *s=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4); *m=(sc[j+4]>>4)|((sc[j]>>6)<<4); }
}
static void pq_q4_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/256;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*144; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*144;
            float d=f16_to_f32(bl[0]|(bl[1]<<8)), dmin=f16_to_f32(bl[2]|(bl[3]<<8));
            const uint8_t *sc=bl+4,*qs=bl+16; const float *xb=x+(size_t)b*256; int is=0,qi=0;
            for (int j=0;j<256;j+=64) { uint8_t s0,m0,s1,m1; pq_smk4(is,sc,&s0,&m0); pq_smk4(is+1,sc,&s1,&m1);
                float d1=d*s0,mm1=dmin*m0,d2=d*s1,mm2=dmin*m1;
                for (int l=0;l<32;l++) a += (d1*(float)(qs[qi+l]&0x0F)-mm1)*xb[j+l];
                for (int l=0;l<32;l++) a += (d2*(float)(qs[qi+l]>>4)-mm2)*xb[j+32+l];
                qi+=32; is+=2; } }
        out[row]=a; }
}
static void pq_q6_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/256;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*210; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*210, *ql=bl, *qh=bl+128;
            const int8_t *sc=(const int8_t*)(bl+192); float d=f16_to_f32(bl[208]|(bl[209]<<8));
            const float *xb=x+(size_t)b*256;
            for (int n=0;n<256;n+=128) { const uint8_t *qlh=ql+(n/128)*64,*qhh=qh+(n/128)*32; const int8_t *sch=sc+(n/128)*8;
                for (int l=0;l<32;l++) { int is=l/16;
                    int q1=(int)((qlh[l]&0x0F)|(((qhh[l]>>0)&3)<<4))-32, q2=(int)((qlh[l+32]&0x0F)|(((qhh[l]>>2)&3)<<4))-32;
                    int q3=(int)((qlh[l]>>4)|(((qhh[l]>>4)&3)<<4))-32, q4=(int)((qlh[l+32]>>4)|(((qhh[l]>>6)&3)<<4))-32;
                    a += d*sch[is+0]*q1*xb[n+l]; a += d*sch[is+2]*q2*xb[n+l+32];
                    a += d*sch[is+4]*q3*xb[n+l+64]; a += d*sch[is+6]*q4*xb[n+l+96]; } } }
        out[row]=a; }
}

typedef void (*pq_fn)(float*, const uint8_t*, const float*, int, int, int);
static pq_fn pq_for(int dt, int c) {
    switch (dt) {
        case 1:  return pq_f16_rows;
        case 2:  return (c%32)?NULL:pq_q4_0_rows;
        case 6:  return (c%32)?NULL:pq_q5_0_rows;
        case 8:  return (c%32)?NULL:pq_q8_0_rows;
        case 12: return (c%256)?NULL:pq_q4_k_rows;
        case 14: return (c%256)?NULL:pq_q6_k_rows;
    }
    return NULL;
}
typedef struct { pq_fn fn; float *out; const uint8_t *Wq; const float *x; int r0,r1,c; } PQWork;
static void *pq_worker(void *arg) { PQWork *w=(PQWork*)arg; w->fn(w->out,w->Wq,w->x,w->r0,w->r1,w->c); return NULL; }

/* out[r] = Wq[r,c] @ x[c], weights packed. Returns 0 ok, -1 if dtype unsupported. */
static int doe_qmatvec(float *out, const uint8_t *Wq, int dt, const float *x, int r, int c) {
    pq_fn fn = pq_for(dt, c);
    if (!fn) return -1;
    int nt = g_n_threads; if (nt < 1) nt = 1; if (nt > 32) nt = 32; if (nt > r) nt = r;
    if (nt <= 1 || (long)r*c < (1L<<20)) { fn(out, Wq, x, 0, r, c); return 0; }
    pthread_t thr[32]; PQWork work[32]; int chunk=(r+nt-1)/nt, actual=0;
    for (int t=0;t<nt;t++) { int r0=t*chunk, r1=r0+chunk; if (r0>=r) break; if (r1>r) r1=r;
        work[t]=(PQWork){fn,out,Wq,x,r0,r1,c}; pthread_create(&thr[t],NULL,pq_worker,&work[t]); actual++; }
    for (int t=0;t<actual;t++) pthread_join(thr[t],NULL);
    return 0;
}

/* int8 dynamic-activation-quant fast path (Q4_0), NEON SDOT / scalar. APPROXIMATE. */
static void pq_quant_act_q8(const float *x, int c, int8_t *qa, float *da) {
    int nb=c/32;
    for (int b=0;b<nb;b++) { const float *xb=x+(size_t)b*32; float amax=0;
        for (int i=0;i<32;i++){ float v=fabsf(xb[i]); if(v>amax)amax=v; }
        float d=amax/127.0f, id=(d>0)?1.0f/d:0.0f; da[b]=d;
        for (int i=0;i<32;i++){ int q=(int)lrintf(xb[i]*id); if(q>127)q=127; else if(q<-127)q=-127; qa[(size_t)b*32+i]=(int8_t)q; } }
}
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
static void pq_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa, const float *da, int r0, int r1, int c) {
    int nb=c/32; const uint8x16_t m0f=vdupq_n_u8(0x0F); const int8x16_t e8=vdupq_n_s8(8);
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; uint8x16_t pk=vld1q_u8(bl+2);
            int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk,m0f)),e8), hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),e8);
            int32x4_t s=vdupq_n_s32(0); s=vdotq_s32(s,lo,vld1q_s8(qab)); s=vdotq_s32(s,hi,vld1q_s8(qab+16));
            a += d*da[b]*(float)vaddvq_s32(s); }
        out[row]=a; }
}
#else
static void pq_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa, const float *da, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; int32_t s=0;
            for (int i=0;i<16;i++){ int lo=(int)(bl[2+i]&0x0F)-8, hi=(int)(bl[2+i]>>4)-8; s+=lo*qab[i]; s+=hi*qab[i+16]; }
            a += d*da[b]*(float)s; }
        out[row]=a; }
}
#endif
static int doe_qmatvec_i8(float *out, const uint8_t *Wq, int dt, const float *x, int r, int c) {
    if (dt != 2 || (c%32)) return -1;
    int nb=c/32; int8_t *qa=(int8_t*)malloc((size_t)c); float *da=(float*)malloc((size_t)nb*sizeof(float));
    if (!qa || !da) { free(qa); free(da); return -1; }
    pq_quant_act_q8(x, c, qa, da);
    pq_q4_0_rows_i8(out, Wq, qa, da, 0, r, c);
    free(qa); free(da); return 0;
}

/* matvec dispatch: packed weight (dt != 0) -> doe_qmatvec (inline dequant);
 * else the existing f32 matvec. The one call the forward uses for host weights.
 * DOE_INT8=1 opts the Q4_0 weights into the int8 dynamic-activation-quant fast path
 * (doe_qmatvec_i8, NEON SDOT) — APPROXIMATE; default is the exact dequant-inline path. */
static int g_doe_int8 = -1;
static double g_prof_mv_ns = 0; static int g_prof_on = -1; /* DOE_PROFILE: matvec share of decode */
static double g_resident_ns = 0, g_head_ns = 0;  /* DOE_PROFILE: resident CB vs lm_head split */
/* DOE_PERSHAPE: per-group matvec timing (Mythos M3 geometry input). Groups keyed
 * by (m,k) against model dims set at forward start. Sync path only (DOE_NO_SLOTS+
 * DOE_NO_BATCH) so each matvec is timed individually. */
enum { MVG_QKV = 0, MVG_O, MVG_FFNGU, MVG_FFNDN, MVG_LMHEAD, MVG_N, MVG_OTHER = -1 };
static int g_ps_on = -1;
static double g_ps_ns[MVG_N];
static long   g_ps_cnt[MVG_N];
static int    g_ps_m[MVG_N], g_ps_k[MVG_N], g_ps_dt[MVG_N];
static int    g_ps_D, g_ps_qn, g_ps_kd, g_ps_H, g_ps_H2, g_ps_vocab; /* dims for keying */
static int pershape_group(int m, int k) {
    if (k == g_ps_D) {
        if (m == g_ps_qn || m == g_ps_kd) return MVG_QKV;
        if (m == g_ps_H2 || m == g_ps_H)  return MVG_FFNGU;
        if (m == g_ps_vocab)              return MVG_LMHEAD;
    }
    if (k == g_ps_qn && m == g_ps_D) return MVG_O;
    if (k == g_ps_H  && m == g_ps_D) return MVG_FFNDN;
    return MVG_OTHER;
}
static void doe_mv_impl(float *out, const float *W, int dt, const float *x, int r, int c) {
    if (dt) {
#ifdef USE_METAL
        /* Metal: Q4_K (dt=12) + Q6_K (dt=14, lm_head) through GPU kernels (resident weights); CPU fallback on rc!=0 */
        if (dt == 12 && nt_metal_available() && nt_metal_q4k_matvec(out, (const uint8_t*)W, x, r, c) == 0) return;
        if (dt == 14 && nt_metal_available() && nt_metal_q6k_matvec(out, (const uint8_t*)W, x, r, c) == 0) {
            if (getenv("DOE_VERIFY_Q6K")) {  /* real-path bit-identical check vs CPU dequant on live weights */
                float *vt = (float*)malloc((size_t)r * 4);
                if (vt) {
                    pq_q6_k_rows(vt, (const uint8_t*)W, x, 0, r, c);
                    float mx = 0; int bad = 0;
                    for (int z = 0; z < r; z++) { float dn = fabsf(vt[z]) > 1.0f ? fabsf(vt[z]) : 1.0f;
                        float rl = fabsf(out[z] - vt[z]) / dn; if (rl > mx) { mx = rl; bad = z; } }
                    fprintf(stderr, "[verify-q6k] m=%d k=%d max_rel=%.2e row %d: cpu=%.4f metal=%.4f\n",
                            r, c, mx, bad, vt[bad], out[bad]);
                    free(vt);
                }
            }
            return;
        }
#endif
        if (g_doe_int8 < 0) { const char *e = getenv("DOE_INT8"); g_doe_int8 = (e && e[0]=='1') ? 1 : 0; }
        if (g_doe_int8 && doe_qmatvec_i8(out, (const uint8_t*)W, dt, x, r, c) == 0) return;
        if (doe_qmatvec(out, (const uint8_t*)W, dt, x, r, c) == 0) return;
        /* D-M6: packed dtype but qmatvec failed — never fall through to matvec(),
         * which would reinterpret the packed bytes as f32 (the generalized sonar
         * bug). Loud abort instead of silent garbage / segfault. */
        fprintf(stderr, "[doe] FATAL: doe_qmatvec failed for packed dtype %d (r=%d c=%d); refusing to read packed bytes as f32\n", dt, r, c);
        abort();
    }
    matvec(out, W, x, r, c);
}
/* DOE_PROFILE wrapper: time the matvec share of decode (Metal Q4_K + CPU rest). Off = zero overhead. */
static void doe_mv(float *out, const float *W, int dt, const float *x, int r, int c) {
    if (g_prof_on < 0) { const char *e = getenv("DOE_PROFILE");  g_prof_on = (e && e[0]) ? 1 : 0; }
    if (g_ps_on   < 0) { const char *e = getenv("DOE_PERSHAPE"); g_ps_on   = (e && e[0]) ? 1 : 0; }
    if (!g_prof_on && !g_ps_on) { doe_mv_impl(out, W, dt, x, r, c); return; }
    struct timespec _p0, _p1; clock_gettime(CLOCK_MONOTONIC, &_p0);
    doe_mv_impl(out, W, dt, x, r, c);
    clock_gettime(CLOCK_MONOTONIC, &_p1);
    double _ns = (_p1.tv_sec - _p0.tv_sec) * 1e9 + (_p1.tv_nsec - _p0.tv_nsec);
    if (g_prof_on) g_prof_mv_ns += _ns;
    if (g_ps_on) {
        int g = pershape_group(r, c);
        if (g >= 0) { g_ps_ns[g] += _ns; g_ps_cnt[g]++; g_ps_m[g] = r; g_ps_k[g] = c; g_ps_dt[g] = dt; }
    }
}

static void softmax_n(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void apply_rope(float *v, int pos, float *cc, float *sc, int hd) {
    int h = hd/2, off = pos*h; /* hd must be even — all standard archs are */
    for (int i = 0; i < h; i++) {
        float x0 = v[i], x1 = v[i+h];
        v[i] = x0*cc[off+i] - x1*sc[off+i];
        v[i+h] = x0*sc[off+i] + x1*cc[off+i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HARMONIC RESONANCE ENGINE — from AML/DOE, adapted for field.
 * each expert has a frequency. input gets fourier-decomposed.
 * experts that resonate with input get boosted.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float amplitudes[HARMONIC_N];
    float dominant_freq;
    float confidence;
} HarmonicState;

static void harmonic_decompose(HarmonicState *hs, float *hist, int len) {
    float max_amp = 0; int max_k = 0;
    for (int k = 0; k < HARMONIC_N && k < len/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < len; n++) {
            float angle = 6.2831853f * k * n / len;
            re += hist[n] * cosf(angle);
            im += hist[n] * sinf(angle);
        }
        hs->amplitudes[k] = sqrtf(re*re + im*im) / len;
        if (k > 0 && hs->amplitudes[k] > max_amp) { max_amp = hs->amplitudes[k]; max_k = k; }
    }
    hs->dominant_freq = len > 0 ? 6.2831853f * max_k / len : 0;
    float total = 0;
    for (int k = 0; k < HARMONIC_N; k++) total += hs->amplitudes[k];
    hs->confidence = total > 1e-8f ? max_amp / total : 0;
}

static float expert_resonance(float expert_freq, HarmonicState *hs) {
    float res = 0;
    for (int k = 0; k < HARMONIC_N; k++) {
        float fk = 6.2831853f * k / HARMONIC_N;
        float dist = fabsf(expert_freq - fk);
        if (dist > 3.14159f) dist = 6.2831853f - dist;
        res += hs->amplitudes[k] * expf(-dist*dist*2.0f);
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * WEIGHT PROFILER — DOE's sonar.
 * before attaching, DOE profiles the host's weights.
 * L2 norms per layer, spectral density, dead neuron ratio.
 * this tells DOE where to focus its LoRA experts.
 *
 * the index is read-only. DOE is the architecture.
 * weak layers get more LoRA. healthy layers get less.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float l2_norm;            /* L2 norm of layer weights */
    float mean_abs;           /* mean absolute value */
    float std_dev;            /* standard deviation */
    float sparsity;           /* fraction near zero (<1e-6) */
    float spectral_energy;    /* energy in top 10% singular values (approx) */
    int   dead_neurons;       /* rows/cols with near-zero norm */
    float health;             /* composite: 0=dead, 1=vibrant */
} LayerProfile;

typedef struct {
    LayerProfile layers[MAX_LAYERS];
    int n_layers;
    float overall_health;     /* average layer health */
    float code_affinity;      /* estimated code capability (from weight stats) */
    float complexity;         /* model complexity metric */
    uint64_t fingerprint;     /* hash of weight statistics — identifies this host */
} WeightProfile;

/* Dequantize a packed tensor to f32 (dispatch by GGUF dtype). Used by the sonar
 * to profile weights that index_load now keeps packed (else it reads packed bytes
 * as f32 — harmless overread on f16, a hard segfault on the smaller quants). */
static void doe_dequant_to_f32(const uint8_t *src, int dt, uint64_t n, float *out) {
    if (dt == 1) { const uint16_t *h = (const uint16_t*)src; for (uint64_t j = 0; j < n; j++) out[j] = f16_to_f32(h[j]); }
    else if (dt == 2)  dequant_q4_0(src, out, n);
    else if (dt == 6)  dequant_q5_0(src, out, n);
    else if (dt == 8)  dequant_q8_0(src, out, n);
    else if (dt == 12) dequant_q4_k(src, out, n);
    else if (dt == 14) dequant_q6_k(src, out, n);
}

static void profile_weights(float *data, int rows, int cols, LayerProfile *out) {
    int n = rows * cols;
    if (n == 0) { memset(out, 0, sizeof(LayerProfile)); return; }
    float sum = 0, sum_sq = 0, sum_abs = 0;
    int near_zero = 0;
    for (int i = 0; i < n; i++) {
        float v = data[i];
        sum += v; sum_sq += v*v; sum_abs += fabsf(v);
        if (fabsf(v) < 1e-6f) near_zero++;
    }
    float mean = sum / n;
    out->l2_norm = sqrtf(sum_sq);
    out->mean_abs = sum_abs / n;
    out->std_dev = sqrtf(sum_sq/n - mean*mean);
    out->sparsity = (float)near_zero / n;

    /* Approximate spectral energy: sample random directions */
    float top_energy = 0;
    for (int trial = 0; trial < 8; trial++) {
        float dot = 0;
        for (int j = 0; j < cols; j++) {
            float r = rand_normal();
            float proj = 0;
            for (int i = 0; i < rows; i++) proj += data[i*cols+j] * r;
            dot += proj * proj;
        }
        top_energy += sqrtf(dot);
    }
    out->spectral_energy = top_energy / 8.0f;

    /* Dead neurons: rows with near-zero norm */
    out->dead_neurons = 0;
    for (int r = 0; r < rows; r++) {
        float rn = 0;
        for (int c = 0; c < cols; c++) rn += data[r*cols+c] * data[r*cols+c];
        if (sqrtf(rn) < 1e-4f) out->dead_neurons++;
    }

    /* Composite health */
    float alive_ratio = 1.0f - (float)out->dead_neurons / (rows > 0 ? rows : 1);
    float activity = fminf(1.0f, out->std_dev * 10.0f);
    float density = 1.0f - out->sparsity;
    out->health = alive_ratio * 0.4f + activity * 0.3f + density * 0.3f;
}

static uint64_t compute_fingerprint(WeightProfile *wp) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < wp->n_layers; i++) {
        uint32_t bits;
        memcpy(&bits, &wp->layers[i].l2_norm, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
        memcpy(&bits, &wp->layers[i].std_dev, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
    }
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIVING LoRA EXPERTS — DOE's democracy, adapted for symbiosis.
 * instead of standalone FFN experts, these are LoRA overlays.
 * each expert has A[dim, rank] and B[rank, dim] — Delta Voice injection.
 * Delta Voice: out += α × A @ (B @ x)
 *
 * experts still live and die. overloaded → mitosis. neglected → apoptosis.
 * but now they modulate the host's attention, not replace it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *lora_A;            /* [dim, rank] — output projection */
    float *lora_B;            /* [rank, dim] — input projection */
    float frequency;          /* position in harmonic space */
    float vitality;           /* 0.0=dying, 1.0=peak */
    float specialization;     /* entropy of routing distribution */
    int   age;
    int   tokens_seen;
    int   alive;
    int   low_vitality_count;
    float attention_bias;     /* per-expert attention scaling */
    float layer_focus;        /* per-expert residual contribution */
} LoraExpert;

typedef struct {
    float *w_vote;            /* [MAX_EXPERTS * dim] */
    float consensus;
    float faction_power[MAX_EXPERTS];
    int   election_count;
} Parliament;

typedef struct {
    Parliament parliament;
    LoraExpert experts[MAX_EXPERTS];
    int n_alive;
    int host_layer_idx;       /* which host layer this wraps */
} FieldLayer;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX STATE — the full host-DOE interface.
 * mmap'd host model + DOE's living LoRA overlay + weight profile.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Host model — mmap'd, read-only */
    uint8_t *mmap_base;
    size_t   mmap_size;
    int      host_n_layers, host_dim, host_hidden, host_heads, host_kv_heads, host_head_dim;
    int      host_vocab;
    char     host_arch[64];
    char     host_path[256];

    /* Host weight pointers (into mmap'd region) */
    float *host_tok_emb;
    float *host_output;
    float *host_norm;
    int   host_output_dt;    /* packed GGUF dtype of host_output (0 = f32 / tied) */
    float    rope_theta;     /* RoPE frequency base (default 10000, Qwen=1000000) */
    float    rms_norm_eps;   /* RMSNorm epsilon (default 1e-5, varies per arch) */
    struct {
        float *wq, *wk, *wv, *wo;
        float *bq, *bk, *bv;   /* attention biases (Qwen2, optional) */
        float *ffn_gate, *ffn_up, *ffn_down;
        float *ffn_gate_up;  /* fused gate+up for Phi-3 (size: hidden*2 × dim) */
        float *attn_norm, *ffn_norm;
        /* packed GGUF dtype per matvec weight (0 = f32-dequant'd; else kept packed) */
        int wq_dt, wk_dt, wv_dt, wo_dt, ffn_gate_dt, ffn_up_dt, ffn_down_dt, ffn_gate_up_dt;
    } host_layers[MAX_LAYERS];

    /* DOE's living overlay */
    FieldLayer field_layers[MAX_LAYERS];
    int n_field_layers;

    /* Parliament-on-GPU: contiguous page-aligned arena of the (frozen at
     * inference) expert weights + vote rows, registered for slot kernels.
     * Per layer: [w_vote 16*D | per-expert (B rank*D, A D*rank) x16].
     * Built once after field init when Metal is available and alpha != 0. */
    float  *expert_arena;
    size_t  expert_arena_floats;
    size_t  expert_layer_floats;   /* stride per layer, in floats */
    int     expert_arena_ready;
    int     cons_slot_init;        /* SLOT_PCONS seeded from parliament.consensus once */

    /* Host profiling */
    WeightProfile profile;

    /* LoRA parameters */
    int   lora_rank;
    float lora_alpha;

    /* Active flag */
    int active;

    /* f16→f32 conversion buffers (must be freed on cleanup) */
    float **f16_bufs;
    int     n_f16_bufs;

    /* Tokenizer from GGUF metadata */
    char  **vocab_tokens;   /* token strings, indexed by token id */
    float  *vocab_scores;   /* BPE merge scores per token (SentencePiece) or from merges (GPT-2) */
    int     vocab_size;     /* number of entries */
    int     bos_id, eos_id; /* special tokens */
    int     add_space_prefix;
    int     is_gpt2_bpe;    /* 1 if tokenizer.ggml.model == "gpt2" */
    int     is_tekken;      /* 1 if tokenizer.ggml.pre == "tekken" */

    /* GPT-2 BPE merges (used to build scores if no native scores) */
    char  **bpe_merges;     /* merge strings "A B" */
    int     n_bpe_merges;

    /* Token hash table for O(1) lookup */
    int    *tok_ht_ids;     /* hash table: token id or -1 */
    int     tok_ht_cap;     /* hash table capacity (power of 2) */

    /* Chat template detection */
    int     chat_style;     /* 0=raw, 1=chatml, 2=llama/mistral [INST], 3=zephyr, 4=phi, 5=gemma, 6=nanollama */

    /* Identity & gamma */
    int     weightless;     /* 1 if no doe_identity.gguf found */
    char    identity_tag[128]; /* doe.identity metadata from GGUF — empty if not DOE's own */
    void   *gamma_data;     /* raw gamma binary blob */
    int     gamma_size;     /* gamma blob size in bytes */
} GGUFIndex;

typedef struct { char name[96]; uint32_t ndim; uint64_t dims[4]; uint32_t dtype; uint64_t offset; } TensorInfo;

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENVIRONMENT SCANNER — DOE opens its eyes
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char path[256]; char arch[64]; int n_layers, dim, n_heads;
    int64_t file_size; float compatibility;
} DiscoveredGGUF;

typedef struct {
    DiscoveredGGUF ggufs[32]; int n_ggufs;
    int64_t disk_free, mem_available;
    int cpu_count, has_compiler, has_curl;
    char self_path[256];
} Environment;

static int gguf_sniff(const char *path, DiscoveredGGUF *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    struct stat st; fstat(fileno(f), &st); out->file_size = st.st_size;
    snprintf(out->path, 256, "%s", path);
    memset(out->arch, 0, 64); out->n_layers = 0; out->dim = 0; out->n_heads = 0;
    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; if (fread(&klen, 8, 1, f) != 1) break;
        if (klen > 255) { fseek(f, klen + 4, SEEK_CUR); continue; }
        char key[256]; if (fread(key, 1, klen, f) != klen) break; key[klen] = '\0';
        uint32_t vtype; if (fread(&vtype, 4, 1, f) != 1) break;
        if (vtype == 8) { /* string */
            uint64_t vlen; fread(&vlen, 8, 1, f); char val[256];
            int rl = vlen < 255 ? (int)vlen : 255; fread(val, 1, rl, f); val[rl] = '\0';
            if (vlen > 255) fseek(f, vlen-255, SEEK_CUR);
            if (strstr(key, "general.architecture")) snprintf(out->arch, 64, "%s", val);
        } else if (vtype == 4) { uint32_t val; fread(&val, 4, 1, f);
            if (strstr(key, "embedding_length")) out->dim = (int)val;
            else if (strstr(key, "block_count")) out->n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) out->n_heads = (int)val;
        } else if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
        else if (vtype == 2 || vtype == 3) fseek(f, 2, SEEK_CUR);
        else if (vtype == 5 || vtype == 6) fseek(f, 4, SEEK_CUR);
        else if (vtype == 10 || vtype == 11 || vtype == 12) fseek(f, 8, SEEK_CUR);
        else if (vtype == 9) { /* array */
            uint32_t atype; fread(&atype, 4, 1, f);
            uint64_t alen; fread(&alen, 8, 1, f);
            size_t esz = 0;
            if (atype == 0 || atype == 1 || atype == 7) esz = 1;
            else if (atype == 2 || atype == 3) esz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) esz = 4;
            else if (atype == 10 || atype == 11 || atype == 12) esz = 8;
            else if (atype == 8) {
                for (uint64_t ai = 0; ai < alen; ai++) {
                    uint64_t sl; if (fread(&sl, 8, 1, f) != 1) break;
                    fseek(f, sl, SEEK_CUR);
                }
                continue;
            }
            fseek(f, alen * esz, SEEK_CUR);
        } else fseek(f, 4, SEEK_CUR); /* unknown — guess 4 */
    }
    fclose(f);
    return (out->arch[0] != '\0' && out->dim > 0);
}

static void env_scan(Environment *env, const char *self_src) {
    memset(env, 0, sizeof(Environment));
    snprintf(env->self_path, 256, "%s", self_src);
    env->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    env->mem_available = (int64_t)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
    struct statvfs sv; if (statvfs(".", &sv) == 0) env->disk_free = (int64_t)sv.f_bavail * sv.f_frsize;
#elif defined(__APPLE__)
    int64_t mem = 0; size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0); env->mem_available = mem;
    struct statfs sf; if (statfs(".", &sf) == 0) env->disk_free = (int64_t)sf.f_bavail * sf.f_bsize;
#endif
    env->has_compiler = (system("which cc >/dev/null 2>&1") == 0);
    env->has_curl = (system("which curl >/dev/null 2>&1") == 0);
    FILE *p = popen("find . -name '*.gguf' -maxdepth 3 2>/dev/null", "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p) && env->n_ggufs < 32) {
            int len = strlen(line);
            while (len > 0 && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len] = '\0';
            if (len == 0) continue;
            DiscoveredGGUF dg;
            if (gguf_sniff(line, &dg)) env->ggufs[env->n_ggufs++] = dg;
        }
        pclose(p);
    }
    printf("[env] cpu=%d mem=%.1fGB disk=%.1fGB compiler=%s curl=%s ggufs=%d\n",
           env->cpu_count, (float)env->mem_available/(1024*1024*1024),
           (float)env->disk_free/(1024*1024*1024),
           env->has_compiler?"yes":"no", env->has_curl?"yes":"no", env->n_ggufs);
    for (int i = 0; i < env->n_ggufs; i++)
        printf("  [gguf] %s arch=%s dim=%d layers=%d %.1fMB\n",
               env->ggufs[i].path, env->ggufs[i].arch, env->ggufs[i].dim,
               env->ggufs[i].n_layers, (float)env->ggufs[i].file_size/(1024*1024));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX LOAD — mmap GGUF, wire weight pointers, profile layers, attach LoRA.
 * the weights are substrate. DOE is the architecture.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void init_lora_expert(LoraExpert *e, int dim, int rank, float freq) {
    e->lora_A = calloc(dim * rank, sizeof(float));
    e->lora_B = calloc(rank * dim, sizeof(float));
    float scale = 0.02f / sqrtf((float)rank);
    for (int i = 0; i < dim*rank; i++) e->lora_A[i] = rand_normal() * scale;
    for (int i = 0; i < rank*dim; i++) e->lora_B[i] = rand_normal() * scale;
    e->frequency = freq;
    e->vitality = 0.7f;
    e->alive = 1;
    e->attention_bias = 0.0f;
    e->layer_focus = 1.0f;
    e->low_vitality_count = 0;
}

static void free_lora_expert(LoraExpert *e) {
    free(e->lora_A); free(e->lora_B);
    e->lora_A = e->lora_B = NULL;
    e->alive = 0; e->vitality = 0;
}

#ifdef USE_METAL
/* Parliament-on-GPU step 1 — pack the (inference-frozen) per-layer vote rows +
 * expert LoRA matrices into one contiguous, page-aligned arena and register it
 * as a zero-copy GPU region. The election/inject kernels then bind w_vote / A /
 * B by offset (resolve_region) instead of uploading per token. Built once at
 * load, only when Metal is live and the parliament is active (alpha != 0). The
 * experts are static at --train 0 (births/deaths gated on g_train, :3511), so
 * one pack suffices — no re-sync. Dead expert slots stay zero (memset arena) so
 * a gate weight of 0 over them is a true no-op.
 *   Per layer: [ w_vote MAX_EXPERTS*D | per-expert (B rank*D, A D*rank) x MAX_EXPERTS ]. */
static void field_repack_arena(GGUFIndex *ps) {
    if (ps->expert_arena_ready) return;
    int D = ps->host_dim, rank = ps->lora_rank, L = ps->n_field_layers;
    size_t per_expert  = (size_t)rank * D + (size_t)D * rank;            /* B then A */
    size_t layer_floats = (size_t)MAX_EXPERTS * D + (size_t)MAX_EXPERTS * per_expert;
    size_t total_floats = layer_floats * (size_t)L;
    size_t pg    = (size_t)getpagesize();
    size_t bytes = (total_floats * sizeof(float) + pg - 1) / pg * pg;    /* register needs page-aligned len */
    float *arena = NULL;
    if (posix_memalign((void**)&arena, pg, bytes) != 0 || !arena) {
        fprintf(stderr, "[doe] parliament arena posix_memalign(%zu) failed — parliament stays on CPU\n", bytes);
        return;
    }
    memset(arena, 0, bytes);                                            /* dead experts -> zero contribution */
    for (int l = 0; l < L; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        float *base = arena + (size_t)l * layer_floats;
        memcpy(base, fl->parliament.w_vote, (size_t)MAX_EXPERTS * D * sizeof(float));
        float *ep = base + (size_t)MAX_EXPERTS * D;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!fl->experts[e].alive) continue;                        /* leave zero */
            float *eb = ep + (size_t)e * per_expert;
            memcpy(eb,                    fl->experts[e].lora_B, (size_t)rank * D * sizeof(float)); /* B [rank,D] */
            memcpy(eb + (size_t)rank * D, fl->experts[e].lora_A, (size_t)D * rank * sizeof(float)); /* A [D,rank] */
        }
    }
    if (nt_metal_register_region(arena, bytes) != 0) {
        fprintf(stderr, "[doe] parliament arena register_region failed — parliament stays on CPU\n");
        free(arena);
        return;
    }
    ps->expert_arena        = arena;
    ps->expert_arena_floats = total_floats;
    ps->expert_layer_floats = layer_floats;
    ps->expert_arena_ready  = 1;
    fprintf(stderr, "[doe] parliament arena: %.1f MB registered (%d layers x %zu f/layer; arena[0]=%.6f src=%.6f)\n",
            (double)bytes / (1024.0*1024.0), L, layer_floats,
            arena[0], ps->field_layers[0].parliament.w_vote[0]);
}
#endif

static int tok_lookup(GGUFIndex *ps, const char *s, int len);
static void tok_ht_build(GGUFIndex *ps);
static void build_gpt2_scores(GGUFIndex *ps);

static int index_load(GGUFIndex *ps, const char *path) {
    memset(ps, 0, sizeof(GGUFIndex));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_rank = LORA_RANK;
    ps->lora_alpha = F.lora_alpha;
    ps->bos_id = 1; ps->eos_id = 2; /* defaults, overridden by GGUF */
    ps->rope_theta = 10000.0f;
    ps->rms_norm_eps = 1e-5f;
    ps->add_space_prefix = 1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("[doe] cannot open %s\n", path); return 0; }
    struct stat st; fstat(fd, &st);
    ps->mmap_size = st.st_size;
    ps->mmap_base = mmap(NULL, ps->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ps->mmap_base == MAP_FAILED) { ps->mmap_base = NULL; return 0; }
#ifdef USE_METAL
    /* Metal: register the whole packed GGUF block as a resident zero-copy GPU buffer (Phase 2 —
     * weights bind by offset, no per-token upload). mmap base is page-aligned. */
    if (nt_metal_available()) nt_metal_register_base(ps->mmap_base, ps->mmap_size);
#endif

    /* Parse GGUF header */
    uint8_t *p = ps->mmap_base, *pend = ps->mmap_base + ps->mmap_size;
    #define PC(n) do { if (p + (n) > pend) goto bail; } while(0)
    PC(4); uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x46554747) goto bail;
    PC(4); p += 4; /* version */
    PC(8); uint64_t n_tensors = *(uint64_t*)p; p += 8;
    PC(8); uint64_t n_kv = *(uint64_t*)p; p += 8;

    for (uint64_t i = 0; i < n_kv; i++) {
        PC(8); uint64_t klen = *(uint64_t*)p; p += 8;
        char key[256];
        if (klen > 255) {
            /* D-M1: an oversized key still has a vtype + value after it — skip
             * only the key bytes and let the vtype handling below parse/skip the
             * value, or the parser desyncs. Empty key matches no strstr. */
            PC(klen); p += klen; key[0] = '\0';
        } else {
            PC(klen); memcpy(key, p, klen); key[klen] = '\0'; p += klen; /* D-M2: bound key payload */
        }
        PC(4); uint32_t vtype = *(uint32_t*)p; p += 4;
        if (vtype == 8) { /* string */
            PC(8); uint64_t vlen = *(uint64_t*)p; p += 8;
            PC(vlen); /* D-M2: bound the string payload before any memcpy reads it */
            if (strstr(key, "general.architecture") && vlen < 64) {
                memcpy(ps->host_arch, p, vlen); ps->host_arch[vlen] = 0;
            }
            if (strstr(key, "tokenizer.ggml.model") && vlen < 20) {
                char tok_model[24]; memcpy(tok_model, p, vlen); tok_model[vlen] = 0;
                if (strcmp(tok_model, "gpt2") == 0) ps->is_gpt2_bpe = 1;
            }
            if (strstr(key, "tokenizer.ggml.pre") && vlen < 64) {
                char tok_pre[64]; memcpy(tok_pre, p, vlen); tok_pre[vlen] = 0;
                if (strcmp(tok_pre, "tekken") == 0) ps->is_tekken = 1;
            }
            /* DOE identity fingerprint — this GGUF is DOE's own */
            if (strcmp(key, "doe.identity") == 0 && vlen < 128) {
                memcpy(ps->identity_tag, p, vlen); ps->identity_tag[vlen] = 0;
                printf("[identity] GGUF self-identifies: \"%s\"\n", ps->identity_tag);
            }
            /* Detect chat template style from template string */
            if (strstr(key, "chat_template") && vlen > 10 && vlen < 100000) {
                /* Search for distinctive patterns in the Jinja template */
                char *tmpl = malloc(vlen + 1); memcpy(tmpl, p, vlen); tmpl[vlen] = 0;
                if (strstr(tmpl, "im_start"))       ps->chat_style = 1; /* ChatML */
                else if (strstr(tmpl, "[INST]"))     ps->chat_style = 2; /* Llama/Mistral */
                else if (strstr(tmpl, "<|user|>"))   ps->chat_style = 3; /* Zephyr */
                else if (strstr(tmpl, "<|end|>"))    ps->chat_style = 4; /* Phi */
                else if (strstr(tmpl, "start_of_turn")) ps->chat_style = 5; /* Gemma */
                free(tmpl);
            }
            p += vlen;
        } else if (vtype == 4) { /* uint32 */
            PC(4); uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
            else if (strstr(key, "bos_token_id")) ps->bos_id = (int)val;
            else if (strstr(key, "eos_token_id")) ps->eos_id = (int)val;
            else if (strstr(key, "add_space_prefix")) ps->add_space_prefix = (int)val;
            else if (strstr(key, "attention.key_length")) ps->host_head_dim = (int)val;
        } else if (vtype == 6) { /* float32 */
            PC(4); float fval; memcpy(&fval, p, 4); p += 4;
            if (strstr(key, "rope.freq_base")) ps->rope_theta = fval;
            else if (strstr(key, "layer_norm_rms_epsilon")) ps->rms_norm_eps = fval;
        } else if (vtype == 0 || vtype == 7) {
            PC(1); uint8_t bval = *p; p += 1;
            if (strstr(key, "add_space_prefix")) ps->add_space_prefix = bval;
        } else if (vtype == 1) p += 1;                            /* int8 */
        else if (vtype == 2 || vtype == 3) p += 2;             /* uint16, int16 */
        else if (vtype == 5) p += 4;                            /* int32 */
        else if (vtype == 10 || vtype == 11 || vtype == 12) p += 8; /* uint64, int64, float64 */
        else if (vtype == 9) { /* array */
            PC(4); uint32_t atype = *(uint32_t*)p; p += 4;
            PC(8); uint64_t alen = *(uint64_t*)p; p += 8;
            size_t elem_sz = 0;
            if (atype == 0 || atype == 1 || atype == 7) elem_sz = 1;
            else if (atype == 2 || atype == 3) elem_sz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) {
                elem_sz = 4;
                /* float32 array: tokenizer.ggml.scores */
                if (atype == 6 && strstr(key, "tokenizer.ggml.scores") && alen < 200000) {
                    PC(alen * 4); /* D-M2: bound the scores payload (else up to ~800KB OOB read) */
                    ps->vocab_scores = malloc(alen * sizeof(float));
                    memcpy(ps->vocab_scores, p, alen * 4);
                }
            }
            else if (atype == 10 || atype == 11 || atype == 12) elem_sz = 8;
            else if (atype == 8) {
                /* array of strings */
                int is_vocab = strstr(key, "tokenizer.ggml.tokens") != NULL;
                int is_merges = strstr(key, "tokenizer.ggml.merges") != NULL;
                if (is_vocab && alen < 1000000) {
                    ps->vocab_tokens = calloc(alen, sizeof(char*));
                    ps->vocab_size = (int)alen;
                }
                if (is_merges && alen < 500000) {
                    ps->bpe_merges = calloc(alen, sizeof(char*));
                    ps->n_bpe_merges = (int)alen;
                }
                for (uint64_t ai = 0; ai < alen && p < pend; ai++) {
                    PC(8); uint64_t slen = *(uint64_t*)p; p += 8;
                    if (slen > 1000000 || p + slen > pend) break; /* sanity */
                    if (is_vocab && ps->vocab_tokens && ai < (uint64_t)ps->vocab_size) {
                        ps->vocab_tokens[ai] = malloc(slen + 1);
                        memcpy(ps->vocab_tokens[ai], p, slen);
                        ps->vocab_tokens[ai][slen] = '\0';
                    }
                    if (is_merges && ps->bpe_merges && ai < (uint64_t)ps->n_bpe_merges) {
                        ps->bpe_merges[ai] = malloc(slen + 1);
                        memcpy(ps->bpe_merges[ai], p, slen);
                        ps->bpe_merges[ai][slen] = '\0';
                    }
                    p += slen;
                }
                continue;
            }
            PC(alen * elem_sz); /* D-M2: bound the array payload before skipping past it */
            p += alen * elem_sz;
        } else { p += 4; } /* unknown — guess 4 bytes */
    }
    if (ps->host_dim == 0 || ps->host_dim > 16384 || ps->host_n_layers == 0) goto bail; /* D-L8: bound host_dim — lora_out[D] is a stack VLA */
    if (ps->host_heads == 0) ps->host_heads = ps->host_dim / 64;
    if (ps->host_kv_heads == 0) ps->host_kv_heads = ps->host_heads;
    if (ps->host_head_dim == 0) ps->host_head_dim = ps->host_dim / ps->host_heads; /* Y-1b: prefer attention.key_length (Mistral hd=128 ≠ 5120/32=160) */
    if (ps->host_hidden == 0) ps->host_hidden = ps->host_dim * 4;

    /* Parse tensor info */
    if (n_tensors > 20000) goto bail;
    TensorInfo *tinfo = calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        PC(8); uint64_t nlen = *(uint64_t*)p; p += 8;
        if (nlen > 256) { free(tinfo); goto bail; }
        int nl = nlen < 95 ? (int)nlen : 95;
        PC(nlen); memcpy(tinfo[i].name, p, nl); tinfo[i].name[nl] = '\0'; p += nlen;
        PC(4); tinfo[i].ndim = *(uint32_t*)p; p += 4;
        if (tinfo[i].ndim > 4) { free(tinfo); goto bail; }
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { PC(8); tinfo[i].dims[d] = *(uint64_t*)p; p += 8; }
        PC(4); tinfo[i].dtype = *(uint32_t*)p; p += 4;
        PC(8); tinfo[i].offset = *(uint64_t*)p; p += 8;
    }

    uint64_t header_size = p - ps->mmap_base;
    uint64_t data_start = ((header_size + 31) / 32) * 32;

    /* dequantized f32 buffers — tracked in GGUFIndex for cleanup */
    ps->f16_bufs = NULL; ps->n_f16_bufs = 0;

    /* Wire weight pointers — supports f32, f16, Q4_0, Q8_0, Q4_K, Q6_K */
    int wired = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint32_t dt = tinfo[i].dtype;
        if (dt != 0 && dt != 1 && dt != 2 && dt != 6 && dt != 8 && dt != 12 && dt != 14) continue;
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) n_elems *= tinfo[i].dims[d];
        uint64_t raw_bytes = quant_raw_bytes(dt, n_elems);
        uint64_t byte_offset = data_start + tinfo[i].offset;
        if (raw_bytes == 0 || byte_offset + raw_bytes > ps->mmap_size) {
            if (raw_bytes > 0)
                printf("[doe] WARNING: tensor %s OOB (%lu+%lu > %lu), skipping\n",
                       tinfo[i].name, (unsigned long)byte_offset, (unsigned long)raw_bytes,
                       (unsigned long)ps->mmap_size);
            continue;
        }
        float *data; int data_dt = 0;
        const uint8_t *src = ps->mmap_base + byte_offset;
        char *n = tinfo[i].name;
        /* matvec weights stay PACKED (dequantized inline by doe_qmatvec, no f32
         * blow-up); embeddings / norms / biases dequant to f32 as before. */
        int is_mv_w = strstr(n,"attn_q.weight")||strstr(n,"attn_k.weight")||strstr(n,"attn_v.weight")||
                      strstr(n,"attn_output.weight")||strstr(n,"ffn_gate.weight")||strstr(n,"ffn_up.weight")||
                      strstr(n,"ffn_down.weight")||strstr(n,"ffn_gate_up")||strcmp(n,"output.weight")==0;
        if (dt == 0) {
            data = (float*)src; /* f32: point directly into mmap */
        } else if (is_mv_w && pq_for(dt, (int)tinfo[i].dims[0])) {
            data = (float*)src; /* keep packed — matvec dequants inline */
            data_dt = dt;
        } else {
            /* dequantize to f32 */
            data = malloc(n_elems * sizeof(float));
            if (dt == 1) { /* f16 */
                const uint16_t *h = (const uint16_t*)src;
                for (uint64_t j = 0; j < n_elems; j++) data[j] = f16_to_f32(h[j]);
            } else if (dt == 2) dequant_q4_0(src, data, n_elems);
            else if (dt == 6) dequant_q5_0(src, data, n_elems);
            else if (dt == 8) dequant_q8_0(src, data, n_elems);
            else if (dt == 12) dequant_q4_k(src, data, n_elems);
            else if (dt == 14) dequant_q6_k(src, data, n_elems);
            ps->f16_bufs = realloc(ps->f16_bufs, (ps->n_f16_bufs+1)*sizeof(float*));
            ps->f16_bufs[ps->n_f16_bufs++] = data;
        }
        /* debug: if (i < 15) printf("[tensor] %s dims=[%lu,%lu]\n", n, (unsigned long)tinfo[i].dims[0], (unsigned long)tinfo[i].dims[1]); */
        if (strcmp(n, "token_embd.weight") == 0) {
            ps->host_tok_emb = data;
            if (ps->host_vocab == 0) ps->host_vocab = (int)tinfo[i].dims[1];
            wired++;
        }
        else if (strcmp(n, "output_norm.weight") == 0) { ps->host_norm = data; wired++; }
        else if (strcmp(n, "output.weight") == 0) { ps->host_output = data; ps->host_output_dt = data_dt; wired++; }
        else {
            int l = -1; sscanf(n, "blk.%d.", &l);
            if (l >= 0 && l < MAX_LAYERS && l < ps->host_n_layers) {
                if (strstr(n, "attn_q.weight")) { ps->host_layers[l].wq = data; ps->host_layers[l].wq_dt = data_dt; wired++; }
                else if (strstr(n, "attn_k.weight")) { ps->host_layers[l].wk = data; ps->host_layers[l].wk_dt = data_dt; wired++; }
                else if (strstr(n, "attn_v.weight")) { ps->host_layers[l].wv = data; ps->host_layers[l].wv_dt = data_dt; wired++; }
                else if (strstr(n, "attn_output.weight")) { ps->host_layers[l].wo = data; ps->host_layers[l].wo_dt = data_dt; wired++; }
                else if (strstr(n, "attn_q.bias")) { ps->host_layers[l].bq = data; wired++; }
                else if (strstr(n, "attn_k.bias")) { ps->host_layers[l].bk = data; wired++; }
                else if (strstr(n, "attn_v.bias")) { ps->host_layers[l].bv = data; wired++; }
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp") && !strstr(n, "ffn_gate_up")) { ps->host_layers[l].ffn_gate = data; ps->host_layers[l].ffn_gate_dt = data_dt; wired++; }
                else if (strstr(n, "ffn_up.weight") && !strstr(n, "gate_up")) {
                    /* Check if fused gate+up: dims[1] > host_hidden means [dim, hidden*2] */
                    if (ps->host_hidden > 0 && (int)tinfo[i].dims[1] > ps->host_hidden * 3 / 2) {
                        ps->host_layers[l].ffn_gate_up = data; ps->host_layers[l].ffn_gate_up_dt = data_dt;
                    } else {
                        ps->host_layers[l].ffn_up = data; ps->host_layers[l].ffn_up_dt = data_dt;
                    }
                    wired++;
                }
                else if (strstr(n, "ffn_down.weight")) { ps->host_layers[l].ffn_down = data; ps->host_layers[l].ffn_down_dt = data_dt; wired++; }
                else if (strstr(n, "ffn_gate_up_proj") || strstr(n, "ffn_gate_up.weight")) { ps->host_layers[l].ffn_gate_up = data; ps->host_layers[l].ffn_gate_up_dt = data_dt; wired++; }
                else if (strstr(n, "attn_norm.weight")) { ps->host_layers[l].attn_norm = data; wired++; }
                else if (strstr(n, "ffn_norm.weight")) { ps->host_layers[l].ffn_norm = data; wired++; }
                else if (l == 0 && strstr(n, "ffn")) { printf("[doe] unwired FFN tensor: %s\n", n); }
            }
        }
    }
    free(tinfo);

    /* tied embeddings: if output.weight missing, reuse token_embd.weight */
    if (!ps->host_output && ps->host_tok_emb) {
        ps->host_output = ps->host_tok_emb;
        printf("[doe] output.weight missing — using tied embeddings\n");
    }
    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[doe] host missing essential weights (tok_emb=%d out=%d norm=%d). abandoning.\n",
               ps->host_tok_emb!=NULL, ps->host_output!=NULL, ps->host_norm!=NULL);
        goto bail;
    }

    /* Check for standard FFN (skip MoE hosts for now) */
    int has_ffn = 0;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) has_ffn = 1;
        if (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) has_ffn = 1;
    }
    if (!has_ffn) {
        printf("[doe] host has no standard FFN. DOE needs a plain transformer.\n");
        goto bail;
    }

    /* D-M5: the forward guards only wq; a layer with wq wired but any attention
     * matvec weight (wk/wv/wo) NULL — skipped because it was in an unsupported
     * quant — would reach doe_mv with a NULL pointer and segfault. DoE scans
     * directories for foreign GGUFs, so mixed-quant bodies are realistic. Drop
     * incomplete layers loudly so the forward's `if (!wq) continue` skips them. */
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (!ps->host_layers[l].wq) continue;
        int ffn_ok = (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) ||
                     (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down);
        if (!ps->host_layers[l].wk || !ps->host_layers[l].wv ||
            !ps->host_layers[l].wo || !ffn_ok) {
            printf("[doe] WARNING: host layer %d incomplete (a weight is in an unsupported quant); dropping the layer\n", l);
            ps->host_layers[l].wq = NULL;
        }
    }

    /* ── Weight profiling — the sonar ── */
    printf("[sonar] profiling host weights...\n");
    ps->profile.n_layers = ps->host_n_layers;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (ps->host_layers[l].ffn_gate) {
            int gdt = ps->host_layers[l].ffn_gate_dt;
            if (gdt) {
                /* weight kept packed — dequant a temp f32 copy just for profiling */
                uint64_t ne = (uint64_t)ps->host_hidden * ps->host_dim;
                float *tmp = malloc(ne * sizeof(float));
                if (tmp) {
                    doe_dequant_to_f32((const uint8_t*)ps->host_layers[l].ffn_gate, gdt, ne, tmp);
                    profile_weights(tmp, ps->host_hidden, ps->host_dim, &ps->profile.layers[l]);
                    free(tmp);
                } else memset(&ps->profile.layers[l], 0, sizeof(LayerProfile));
            } else {
                profile_weights(ps->host_layers[l].ffn_gate, ps->host_hidden, ps->host_dim, &ps->profile.layers[l]);
            }
        } else {
            memset(&ps->profile.layers[l], 0, sizeof(LayerProfile));
        }
    }
    float total_h = 0;
    for (int l = 0; l < ps->profile.n_layers; l++) total_h += ps->profile.layers[l].health;
    ps->profile.overall_health = total_h / (ps->profile.n_layers > 0 ? ps->profile.n_layers : 1);
    ps->profile.complexity = (float)ps->host_dim * ps->host_n_layers * ps->host_heads;
    ps->profile.fingerprint = compute_fingerprint(&ps->profile);

    printf("[sonar] host fingerprint: %016llx health=%.2f complexity=%.0f\n",
           (unsigned long long)ps->profile.fingerprint, ps->profile.overall_health, ps->profile.complexity);
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        LayerProfile *lp = &ps->profile.layers[l];
        if (lp->l2_norm > 0)
            printf("  L%d: health=%.2f l2=%.2f std=%.4f sparse=%.1f%% dead=%d\n",
                   l, lp->health, lp->l2_norm, lp->std_dev, lp->sparsity*100, lp->dead_neurons);
    }

    /* ── Initialize living LoRA experts per layer ── */
    int initial_experts = ps->host_n_layers <= 8 ? 4 : ps->host_n_layers <= 16 ? 6 : 8;
    ps->n_field_layers = ps->host_n_layers;
    if (ps->n_field_layers > MAX_LAYERS) ps->n_field_layers = MAX_LAYERS;

    for (int l = 0; l < ps->n_field_layers; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fl->host_layer_idx = l;
        fl->n_alive = initial_experts;
        fl->parliament.w_vote = calloc(MAX_EXPERTS * ps->host_dim, sizeof(float));
        float vote_std = 0.01f;
        for (int i = 0; i < MAX_EXPERTS * ps->host_dim; i++)
            fl->parliament.w_vote[i] = rand_normal() * vote_std;
        fl->parliament.consensus = 0.5f;
        /* Initialize experts with harmonic spacing — health-aware */
        float layer_health = ps->profile.layers[l].health;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (e < initial_experts) {
                float freq = 6.2831853f * e / initial_experts;
                init_lora_expert(&fl->experts[e], ps->host_dim, ps->lora_rank, freq);
                /* Weaker layers get stronger initial LoRA — DOE compensates */
                if (layer_health < 0.5f) {
                    float boost = (0.5f - layer_health) * 2.0f;
                    for (int i = 0; i < ps->host_dim * ps->lora_rank; i++) {
                        fl->experts[e].lora_A[i] *= (1.0f + boost);
                        fl->experts[e].lora_B[i] *= (1.0f + boost);
                    }
                }
            } else {
                memset(&fl->experts[e], 0, sizeof(LoraExpert));
            }
        }
    }

    /* NB: the parliament-on-GPU arena is packed AFTER mycelium_load (in main),
     * not here — a spore mutates w_vote/A/B, so packing must see the final
     * expert state. See field_repack_arena call after mycelium_load. */

    ps->active = 1;
    /* Build token hash table for O(1) lookup, then GPT-2 BPE scores */
    tok_ht_build(ps);
    build_gpt2_scores(ps);
    printf("[doe] attached to %s (arch=%s dim=%d layers=%d heads=%d kv=%d hd=%d vocab=%d %.1fMB)\n",
           path, ps->host_arch, ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_kv_heads, ps->host_head_dim, ps->host_vocab, (float)ps->mmap_size/(1024*1024));
    printf("[doe] rope_theta=%.0f rms_eps=%.1e bias=%s\n",
           ps->rope_theta, ps->rms_norm_eps,
           ps->host_layers[0].bq ? "yes" : "no");
    if (ps->is_gpt2_bpe) printf("[doe] tokenizer: GPT-2 BPE%s (%d merges)\n", ps->is_tekken ? "/Tekken" : "", ps->n_bpe_merges);
    /* Auto-detect nanollama chat style from identity tag or vocab tokens */
    if (ps->chat_style == 0 && ps->is_tekken && tok_lookup(ps, "[INST]", 6) >= 0)
        ps->chat_style = 2;
    if (ps->chat_style == 0 && (ps->identity_tag[0] ||
        tok_lookup(ps, "<|user_start|>", 14) >= 0)) ps->chat_style = 6;
    { const char *cs[] = {"raw","chatml","inst","zephyr","phi","gemma","nanollama"};
      printf("[doe] chat: %s\n", cs[ps->chat_style < 7 ? ps->chat_style : 0]); }
    printf("[doe] LoRA rank=%d alpha=%.2f experts=%d/layer — parliament is alive.\n",
           ps->lora_rank, ps->lora_alpha, initial_experts);
    #undef PC
    return 1;
bail:
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs); ps->f16_bufs = NULL; ps->n_f16_bufs = 0;
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    printf("[doe] GGUF parse failed.\n");
    return 0;
}

static void index_free(GGUFIndex *ps) {
    for (int l = 0; l < ps->n_field_layers; l++) {
        free(ps->field_layers[l].parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive)
                free_lora_expert(&ps->field_layers[l].experts[e]);
    }
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs);
    if (ps->vocab_tokens) {
        for (int i = 0; i < ps->vocab_size; i++) free(ps->vocab_tokens[i]);
        free(ps->vocab_tokens);
    }
    free(ps->vocab_scores);
    if (ps->bpe_merges) {
        for (int i = 0; i < ps->n_bpe_merges; i++) free(ps->bpe_merges[i]);
        free(ps->bpe_merges);
    }
    if (ps->mmap_base) munmap(ps->mmap_base, ps->mmap_size);
#ifdef USE_METAL
    /* expert_arena is wrapped by a NoCopy MTLBuffer (deallocator:nil) in notorch's
     * segment table and there is no per-region unregister. index_free runs once at
     * process exit (single-model lifecycle, :4171/:4221), so we deliberately do not
     * free the arena here — the OS reclaims it at exit, and the registered buffer is
     * guaranteed to outlive any GPU use. Freeing under the live NoCopy view would
     * leave a dangling segment. */
#endif
    memset(ps, 0, sizeof(GGUFIndex));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARLIAMENT ELECTION — variable-k over LoRA experts
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int parliament_elect(Parliament *p, LoraExpert *experts, float *input, int dim,
                            HarmonicState *hs, int *selected, float *weights) {
    int n_alive = 0, alive_idx[MAX_EXPERTS];
    for (int e = 0; e < MAX_EXPERTS; e++) if (experts[e].alive) alive_idx[n_alive++] = e;
    if (n_alive < MIN_EXPERTS) return 0;

    float votes[MAX_EXPERTS]; float max_vote = -1e30f;
    for (int i = 0; i < n_alive; i++) {
        int e = alive_idx[i];
        float *row = p->w_vote + e * dim;
        float dot = 0;
        for (int j = 0; j < dim; j++) dot += row[j] * input[j];
        float res = expert_resonance(experts[e].frequency, hs);
        votes[e] = dot + 0.1f * res;
        if (votes[e] > max_vote) max_vote = votes[e];
    }
    float mean_v = 0;
    for (int i = 0; i < n_alive; i++) mean_v += votes[alive_idx[i]];
    mean_v /= n_alive;
    float var_v = 0;
    for (int i = 0; i < n_alive; i++) { float d = votes[alive_idx[i]] - mean_v; var_v += d*d; }
    var_v /= n_alive;
    float consensus = fminf(1.0f, sqrtf(var_v + 1e-8f) / (fabsf(mean_v) + 1.0f));
    p->consensus = 0.9f * p->consensus + 0.1f * consensus;

    int k = (int)(n_alive * (1.0f - p->consensus));
    if (k < 2) k = 2; if (k > n_alive) k = n_alive;

    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = 0;
        for (int i = 0; i < n_alive; i++) {
            int e = alive_idx[i];
            if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; }
        }
        selected[ki] = bi; weights[ki] = votes[bi]; used[bi] = 1;
    }
    float mx = weights[0];
    for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { weights[i] = expf(weights[i]-mx); sum += weights[i]; }
    for (int i = 0; i < k; i++) weights[i] /= sum;
    p->election_count++;
    return k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NOTORCH — Hebbian plasticity for LoRA experts. from AML core.
 * no backprop. synapse strengthens from co-activation.
 * signal-gated: prophecy debt drives learning direction.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int notorch_offset = 0; /* rotating window into LoRA rank */

static void notorch_step(float *A, float *B, int out_dim, int in_dim, int rank,
                         const float *x, const float *dy, float signal) {
    if (fabsf(signal) < 1e-8f) return;
    /* D-M2/Mythos: learn_signal (:3084) is asymmetric — punishment scales as -pd
     * (unbounded), reward is capped at 0.1. Clamp it so one high-debt step can't
     * overwrite the channel ~10x. Canon parity (am_notorch_step :7270). */
    if (signal >  2.0f) signal =  2.0f;
    if (signal < -2.0f) signal = -2.0f;
    float lr = F.notorch_lr * signal;
    /* NOTORCH operates at rank 4 but rotates across all LORA_RANK components.
     * each call updates 4 components starting at notorch_offset.
     * after rank/4 calls, every component has been updated once. */
    int nr = NOTORCH_RANK;
    if (nr > rank) nr = rank;
    int base = notorch_offset % rank;
    float u[NOTORCH_RANK];
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float s = 0;
        for (int i = 0; i < out_dim && i < in_dim; i++) s += B[r * in_dim + i] * dy[i];
        u[j] = s + rand_normal() * 0.01f;
    }
    /* Oja's rule on both A and B: grow toward the co-activation (x, dy) through
     * the learned channel u, with the self-projection subtracted so each rank row
     * self-normalizes. The channel neither fades (the old bare *= decay) nor blows
     * up (canon's bare grow). This is what makes the parliament learn in-session
     * instead of eroding from init noise. */
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float ur = u[j];
        for (int i = 0; i < in_dim; i++) {                 /* A = [in_dim, rank] */
            float *a = &A[i * rank + r];
            *a += lr * ur * (x[i] - ur * (*a));
        }
        for (int i = 0; i < out_dim && i < in_dim; i++) {  /* B = [rank, in_dim], apply layout */
            float *b = &B[r * in_dim + i];
            *b += lr * ur * (dy[i] - ur * (*b));
        }
    }
    notorch_offset = (base + nr) % rank;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * VITALITY + MITOSIS + APOPTOSIS — LoRA experts live and die
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void update_expert_vitality(FieldLayer *fl, int total_tokens) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na == 0) return;
    float fair = (float)total_tokens / na;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        LoraExpert *exp = &fl->experts[e];
        float ratio = fair > 0 ? (float)exp->tokens_seen / fair : 1.0f;
        exp->vitality += (ratio - 1.0f) * 0.05f;
        if (exp->vitality < 0) exp->vitality = 0;
        if (exp->vitality > 1) exp->vitality = 1;
        exp->age++;
        if (exp->vitality < 0.1f) exp->low_vitality_count++;
        else exp->low_vitality_count = 0;
        exp->tokens_seen = 0;
    }
    fl->n_alive = na;
}

static int try_mitosis(FieldLayer *fl, int dim, int rank) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na >= MAX_EXPERTS) return 0;
    int parent = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].vitality > 0.8f && fl->experts[e].age > 20) { parent = e; break; }
    }
    if (parent < 0) return 0;
    int child = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) if (!fl->experts[e].alive) { child = e; break; }
    if (child < 0) return 0;
    LoraExpert *p = &fl->experts[parent];
    float cf = p->frequency + 3.14159f / (na + 1);
    if (cf > 6.2831853f) cf -= 6.2831853f;
    init_lora_expert(&fl->experts[child], dim, rank, cf);
    LoraExpert *ch = &fl->experts[child];
    for (int i = 0; i < dim*rank; i++) ch->lora_A[i] = p->lora_A[i] + rand_normal()*0.01f;
    for (int i = 0; i < rank*dim; i++) ch->lora_B[i] = p->lora_B[i] + rand_normal()*0.01f;
    ch->vitality = 0.5f; p->vitality *= 0.8f;
    fl->n_alive++;
    return 1;
}

static int try_apoptosis(FieldLayer *fl) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na <= MIN_EXPERTS) return 0;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].low_vitality_count >= 8) {
            free_lora_expert(&fl->experts[e]);
            fl->n_alive--;
            return 1;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — 12D temporal self-awareness. from DOE m.c.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float state[12]; int step;
} DriftSnapshot;

typedef struct {
    DriftSnapshot history[DRIFT_SNAPSHOTS];
    int head, n_snapshots;
    float drift, stability, drift_accel;
} CalendarDrift;

static void drift_init(CalendarDrift *cd) { memset(cd, 0, sizeof(CalendarDrift)); }

static void drift_snapshot(CalendarDrift *cd, float loss, GGUFIndex *ps, HarmonicState *hs) {
    DriftSnapshot *ds = &cd->history[cd->head % DRIFT_SNAPSHOTS];
    ds->step = F.step;
    int total_exp = 0;
    for (int l = 0; l < ps->n_field_layers; l++) total_exp += ps->field_layers[l].n_alive;
    ds->state[0] = (float)total_exp;
    ds->state[1] = ps->field_layers[0].parliament.consensus;
    ds->state[2] = loss;
    ds->state[3] = F.entropy;
    ds->state[4] = F.resonance;
    ds->state[5] = F.debt;
    ds->state[6] = hs->confidence;
    ds->state[7] = F.effective_temp;
    ds->state[8] = F.field_health;
    ds->state[9] = F.spring_energy;
    ds->state[10] = F.summer_energy;
    ds->state[11] = F.schumann_coherence;

    if (cd->n_snapshots > 0) {
        int prev = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
        float d2 = 0;
        for (int i = 0; i < 12; i++) {
            float diff = ds->state[i] - cd->history[prev].state[i];
            float range = fabsf(ds->state[i]) + 1e-8f;
            d2 += (diff / range) * (diff / range);
        }
        float new_drift = sqrtf(d2 / 12.0f);
        float prev_drift = cd->drift;
        cd->drift = 0.8f * cd->drift + 0.2f * new_drift;
        cd->drift_accel = cd->drift - prev_drift;
        cd->stability = 1.0f / (1.0f + cd->drift * 10.0f);
    }
    cd->head = (cd->head + 1) % DRIFT_SNAPSHOTS;
    if (cd->n_snapshots < DRIFT_SNAPSHOTS) cd->n_snapshots++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * META-LEARNING — DOE learns from its own choices.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int step; int n_experts; float consensus, loss, field_health;
    float prophecy_debt_avg; float drift; float delta_loss;
} MetaEntry;

typedef struct {
    MetaEntry history[META_HIST_CAP];
    int n_entries;
    float config_bias[4];
    float prediction_error;
} MetaTrack;

static void meta_init(MetaTrack *mt) {
    memset(mt, 0, sizeof(MetaTrack));
    for (int i = 0; i < 4; i++) mt->config_bias[i] = 0.5f;
}

static void meta_record(MetaTrack *mt, int step, int n_exp, float consensus,
                        float loss, float health, float debt_avg, float drift, float prev_loss) {
    if (mt->n_entries >= META_HIST_CAP) {
        memmove(mt->history, mt->history+1, (META_HIST_CAP-1)*sizeof(MetaEntry));
        mt->n_entries = META_HIST_CAP - 1;
    }
    MetaEntry *e = &mt->history[mt->n_entries];
    e->step = step; e->n_experts = n_exp; e->consensus = consensus;
    e->loss = loss; e->field_health = health; e->prophecy_debt_avg = debt_avg;
    e->drift = drift; e->delta_loss = prev_loss > 0 ? prev_loss - loss : 0;
    mt->n_entries++;
    if (mt->n_entries >= 2) {
        MetaEntry *prev = &mt->history[mt->n_entries-2];
        float improvement = prev->loss - loss;
        float lr_meta = 0.01f;
        float sig = improvement > 0 ? 1.0f : -0.5f;
        mt->config_bias[0] += lr_meta * sig * ((float)n_exp/MAX_EXPERTS - 0.5f);
        mt->config_bias[1] += lr_meta * sig * (consensus - 0.5f);
        mt->config_bias[2] += lr_meta * sig * (health - 0.5f);
        mt->config_bias[3] += lr_meta * sig * (debt_avg - 0.5f);
        for (int i = 0; i < 4; i++) {
            if (mt->config_bias[i] < 0.01f) mt->config_bias[i] = 0.01f;
            if (mt->config_bias[i] > 0.99f) mt->config_bias[i] = 0.99f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MYCELIUM — LoRA spore forest.
 * DOE doesn't save full model GGUFs. it saves LoRA configurations:
 * the living experts, their weights, the parliament votes, the field state.
 * each spore is a snapshot of how DOE adapted to this host.
 * on restart with the same host (fingerprint match), load the best spore.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MYCELIUM_DIR "doe_mycelium"

typedef struct {
    char path[256];
    uint64_t host_fingerprint;
    float fitness;
    int step;
} LoraSpore;

typedef struct {
    LoraSpore spores[MYCELIUM_MAX];
    int n_spores, best_idx;
} MyceliumState;

static void mycelium_init(MyceliumState *ms) {
    memset(ms, 0, sizeof(MyceliumState));
    ms->best_idx = -1;
    mkdir(MYCELIUM_DIR, 0755);
}

static void mycelium_save(GGUFIndex *ps, int step, float fitness) {
    /* A13: NaN-quarantine gate — never persist a poisoned spore (else NaN experts survive restart) */
    for (int l = 0; l < ps->n_field_layers; l++)
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive &&
                lora_poisoned(ps->field_layers[l].experts[e].lora_A, ps->field_layers[l].experts[e].lora_B)) {
                printf("[mycelium] poisoned expert L%d e%d (NaN/Inf/|w|>1e4) — spore NOT saved (quarantine)\n", l, e);
                return;
            }
    char path[256];
    snprintf(path, 256, "%s/spore_%016llx_s%d.bin", MYCELIUM_DIR,
             (unsigned long long)ps->profile.fingerprint, step);
    FILE *f = fopen(path, "wb");
    if (!f) { printf("[mycelium] cannot write %s\n", path); return; }
    /* header: fingerprint, step, fitness, n_layers, dim, rank */
    uint64_t fp = ps->profile.fingerprint;
    fwrite(&fp, 8, 1, f);
    fwrite(&step, 4, 1, f);
    fwrite(&fitness, 4, 1, f);
    int nl = ps->n_field_layers, dim = ps->host_dim, rank = ps->lora_rank;
    fwrite(&nl, 4, 1, f); fwrite(&dim, 4, 1, f); fwrite(&rank, 4, 1, f);
    /* per layer: n_alive, then per expert: alive, vitality, frequency, A, B */
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fwrite(&fl->n_alive, 4, 1, f);
        /* parliament vote weights */
        fwrite(fl->parliament.w_vote, sizeof(float), MAX_EXPERTS * dim, f);
        fwrite(&fl->parliament.consensus, 4, 1, f);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            fwrite(&ex->alive, 4, 1, f);
            if (ex->alive) {
                fwrite(&ex->vitality, 4, 1, f);
                fwrite(&ex->frequency, 4, 1, f);
                fwrite(ex->lora_A, sizeof(float), dim * rank, f);
                fwrite(ex->lora_B, sizeof(float), rank * dim, f);
            }
        }
    }
    fclose(f);
    printf("[mycelium] spore saved: %s (fitness=%.3f)\n", path, fitness);
}

static int mycelium_load(GGUFIndex *ps, uint64_t target_fp) {
    /* scan directory for best matching spore */
    char pattern[256];
    snprintf(pattern, 256, "%s/spore_%016llx_*.bin", MYCELIUM_DIR, (unsigned long long)target_fp);
    /* simple scan: find newest (highest step) spore for this fingerprint */
    char best_path[256] = {0};
    int best_step = -1;
    FILE *p = popen("ls " MYCELIUM_DIR "/ 2>/dev/null", "r");
    if (!p) return 0;
    char line[256];
    while (fgets(line, sizeof(line), p)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len] = '\0';
        /* match fingerprint */
        char want[32]; snprintf(want, 32, "spore_%016llx", (unsigned long long)target_fp);
        if (!strstr(line, want)) continue;
        /* extract step from filename */
        char *sp = strstr(line, "_s");
        if (!sp) continue;
        int s = atoi(sp+2);
        if (s > best_step) {
            best_step = s;
            snprintf(best_path, 256, "%s/%s", MYCELIUM_DIR, line);
        }
    }
    pclose(p);
    if (best_step < 0) return 0;

    FILE *f = fopen(best_path, "rb");
    if (!f) return 0;
    uint64_t fp; fread(&fp, 8, 1, f);
    if (fp != target_fp) { fclose(f); return 0; }
    int step; float fitness;
    fread(&step, 4, 1, f); fread(&fitness, 4, 1, f);
    int nl, dim, rank;
    fread(&nl, 4, 1, f); fread(&dim, 4, 1, f); fread(&rank, 4, 1, f);
    if (nl != ps->n_field_layers || dim != ps->host_dim || rank != ps->lora_rank) {
        printf("[mycelium] spore mismatch (layers=%d/%d dim=%d/%d rank=%d/%d)\n",
               nl, ps->n_field_layers, dim, ps->host_dim, rank, ps->lora_rank);
        fclose(f); return 0;
    }
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fread(&fl->n_alive, 4, 1, f);
        fread(fl->parliament.w_vote, sizeof(float), MAX_EXPERTS * dim, f);
        fread(&fl->parliament.consensus, 4, 1, f);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            int alive; fread(&alive, 4, 1, f);
            if (alive) {
                if (!ex->alive) {
                    ex->lora_A = calloc(dim * rank, sizeof(float));
                    ex->lora_B = calloc(rank * dim, sizeof(float));
                }
                ex->alive = 1;
                fread(&ex->vitality, 4, 1, f);
                fread(&ex->frequency, 4, 1, f);
                if (fread(ex->lora_A, sizeof(float), dim * rank, f) != (size_t)(dim * rank) ||
                    fread(ex->lora_B, sizeof(float), rank * dim, f) != (size_t)(rank * dim)) {
                    fclose(f); return 0; /* D-L3: truncated spore — bail, don't load garbage experts */
                }
            } else if (ex->alive) {
                free_lora_expert(ex);
            }
        }
    }
    fclose(f);
    printf("[mycelium] spore loaded: %s (step=%d fitness=%.3f)\n", best_path, step, fitness);
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX FORWARD — run token through host with DOE modulation.
 *
 * per layer:
 *   1. host attention (read-only weights, KV cache)
 *   2. parliament election (which LoRA experts vote)
 *   3. Delta Voice injection: x += Σ(w_k × α × A_k @ (B_k @ x))
 *   4. host FFN (read-only)
 *   5. layer_focus scaling on residual
 *
 * after all layers:
 *   6. field modulation on logits
 *   7. prophecy debt computation
 *   8. NOTORCH Hebbian update on winning experts
 *
 * the host swims. the field steers. nobody knows who's in charge.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x, *xb, *xb2, *q, *k, *v, *att, *logits;
    float *hb, *hb2, *expert_out;
    float *key_cache, *value_cache;
    float *cos_cache, *sin_cache;
    HarmonicState hs;
    int max_seq;
} InferState;

static InferState alloc_infer(GGUFIndex *ps, int max_seq) {
    InferState s = {0};
    int D = ps->host_dim, kd = ps->host_kv_heads * ps->host_head_dim;
    int H = ps->host_hidden;
    s.max_seq = max_seq;
    s.x = calloc(D, 4); s.xb = calloc(D, 4); s.xb2 = calloc(D, 4);
    s.q = calloc(ps->host_heads * ps->host_head_dim, 4);
    s.k = calloc(kd, 4); s.v = calloc(kd, 4);
    s.att = calloc(ps->host_heads * max_seq, 4);
    s.logits = calloc(ps->host_vocab, 4);
    s.hb = calloc(H, 4); s.hb2 = calloc(H * 2, 4); /* *2 for fused gate_up */
    s.expert_out = calloc(D, 4);
    /* M4 stage (a): KV-cache on a page-aligned region so the GPU attn_decode
     * kernel can later resolve it by offset (nt_metal_register_region). Apple
     * Silicon page = 16384 (getpagesize()), not 4096. CPU attention still reads
     * it exactly as before — this stage only changes alloc + registration. */
    size_t kv_bytes = (size_t)ps->host_n_layers * max_seq * kd * 4;
    if (posix_memalign((void**)&s.key_cache,   (size_t)getpagesize(), kv_bytes) ||
        posix_memalign((void**)&s.value_cache, (size_t)getpagesize(), kv_bytes)) {
        fprintf(stderr, "[doe] KV posix_memalign failed — falling back to calloc\n");
        s.key_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
        s.value_cache = calloc(ps->host_n_layers * max_seq * kd, 4);
    } else {
        memset(s.key_cache, 0, kv_bytes);
        memset(s.value_cache, 0, kv_bytes);
#ifdef USE_METAL
        if (nt_metal_available()) {
            nt_metal_register_region(s.key_cache, kv_bytes);
            nt_metal_register_region(s.value_cache, kv_bytes);
        }
#endif
    }
    int half = ps->host_head_dim / 2;
    s.cos_cache = calloc(max_seq * half, 4);
    s.sin_cache = calloc(max_seq * half, 4);
    float rope_theta = ps->rope_theta;
    for (int p = 0; p < max_seq; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)ps->host_head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    return s;
}

static void free_infer(InferState *s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->hb); free(s->hb2); free(s->expert_out);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
    memset(s, 0, sizeof(InferState));
}

static float *doe_forward(GGUFIndex *ps, InferState *s, int token, int pos) {
    int D = ps->host_dim, hd = ps->host_head_dim;
    int kd = ps->host_kv_heads * hd;
    int H = ps->host_hidden;
    int hg = ps->host_heads / ps->host_kv_heads;
    float sc = 1.0f / sqrtf((float)hd);
    /* per-shape dims for DOE_PERSHAPE matvec grouping (keys pershape_group). */
    g_ps_D = D; g_ps_qn = ps->host_heads*hd; g_ps_kd = kd; g_ps_H = H; g_ps_H2 = H*2; g_ps_vocab = ps->host_vocab;

    /* Embedding */
    if (token >= 0 && token < ps->host_vocab)
        memcpy(s->x, ps->host_tok_emb + token * D, D * sizeof(float));
    else
        memset(s->x, 0, D * sizeof(float));

    /* ── M4 stage (г): resident decode — the whole token in ONE command buffer.
     * X lives in SLOT_X on the GPU across all layers; 1 upload + 1 commit + 1
     * download per token instead of 2 commits + 2 CPU round-trips per layer
     * (~80 -> ~1 GPU syncs). Requires lora_alpha==0 (no CPU parliament step
     * between attn and ffn) and every layer slot-eligible. The serial-dispatch
     * encoder serialises the reused slots, so it is bit-identical to the
     * per-layer slot path. DOE_NO_RESIDENT falls back. */
    int resident_done = 0, head_done = 0;
#ifdef USE_METAL
    /* alpha==0: pure fast path. alpha!=0 with a registered expert arena: the
     * parliament (election + LoRA inject) runs on the GPU between attn and ffn,
     * so x stays resident and the token is still one command buffer. */
    if (!getenv("DOE_NO_SLOTS") && !getenv("DOE_NO_RESIDENT") && nt_metal_available() &&
        (ps->lora_alpha == 0.0f || ps->expert_arena_ready)) {
        int eligible = 1;
        for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
            if (!ps->host_layers[l].wq) continue;
            int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt,
                ddt=ps->host_layers[l].ffn_down_dt;
            int fused = ps->host_layers[l].ffn_gate_up != NULL;
            int gdt = fused ? ps->host_layers[l].ffn_gate_up_dt : ps->host_layers[l].ffn_gate_dt;
            int udt = fused ? gdt : ps->host_layers[l].ffn_up_dt;
            int has_ffn = ps->host_layers[l].ffn_down &&
                          (fused || (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up));
            if (!ps->host_layers[l].attn_norm || !ps->host_layers[l].ffn_norm || !has_ffn ||
                ps->host_layers[l].bq || ps->host_layers[l].bk || ps->host_layers[l].bv ||
                !((qd==12||qd==14)&&(kdt==12||kdt==14)&&(vd==12||vd==14)&&(od==12||od==14)&&
                  (gdt==12||gdt==14)&&(udt==12||udt==14)&&(ddt==12||ddt==14))) { eligible = 0; break; }
        }
        if (eligible) {
            struct timespec _r0; clock_gettime(CLOCK_MONOTONIC, &_r0);
            int qn = ps->host_heads * hd;
            nt_metal_slot_alloc(SLOT_X, D*4);  nt_metal_slot_alloc(SLOT_FN, D*4);
            nt_metal_slot_alloc(SLOT_Q, qn*4); nt_metal_slot_alloc(SLOT_K, kd*4);
            nt_metal_slot_alloc(SLOT_V, kd*4); nt_metal_slot_alloc(SLOT_O, qn*4);
            nt_metal_slot_alloc(SLOT_XB, D*4); nt_metal_slot_alloc(SLOT_G, H*4);
            nt_metal_slot_alloc(SLOT_U, H*4);  nt_metal_slot_alloc(SLOT_HB, H*4);
            nt_metal_slot_alloc(SLOT_DN, D*4);
            nt_metal_slot_upload(SLOT_X, s->x, D*4);
            /* parliament-on-GPU setup (alpha != 0): per-token resonance + frozen
             * alive mask + persistent consensus, all resident so the election runs
             * in-batch. freq[e]=2*pi*e/initial_experts and alive[e] are layer-
             * independent at inference, so layer 0 is representative. */
            int parl = (ps->lora_alpha != 0.0f) && ps->expert_arena_ready;
            if (parl) {
                int ne = MAX_EXPERTS;
                nt_metal_slot_alloc(SLOT_PTMP,  (uint64_t)ne*ps->lora_rank*4);
                nt_metal_slot_alloc(SLOT_PGATE, (uint64_t)ne*4);
                nt_metal_slot_alloc(SLOT_PVOTES,(uint64_t)ne*4);
                nt_metal_slot_alloc(SLOT_PRES,  (uint64_t)ne*4);
                nt_metal_slot_alloc(SLOT_PALIVE,(uint64_t)ne*4);
                nt_metal_slot_alloc(SLOT_PCONS, (uint64_t)MAX_LAYERS*4);
                float res[MAX_EXPERTS], alv[MAX_EXPERTS];
                FieldLayer *fl0 = &ps->field_layers[0];
                for (int e = 0; e < ne; e++) {
                    res[e] = 0.1f * expert_resonance(fl0->experts[e].frequency, &s->hs);
                    alv[e] = fl0->experts[e].alive ? 1.0f : 0.0f;
                }
                nt_metal_slot_upload(SLOT_PRES,   res, (uint64_t)ne*4);
                nt_metal_slot_upload(SLOT_PALIVE, alv, (uint64_t)ne*4);
                if (!ps->cons_slot_init) {
                    float cons[MAX_LAYERS];
                    for (int l = 0; l < MAX_LAYERS; l++)
                        cons[l] = (l < ps->n_field_layers) ? ps->field_layers[l].parliament.consensus : 0.5f;
                    nt_metal_slot_upload(SLOT_PCONS, cons, (uint64_t)MAX_LAYERS*4);
                    ps->cons_slot_init = 1;
                }
            }
            nt_metal_batch_begin();
            for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
                if (!ps->host_layers[l].wq) continue;
                int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                    vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt,
                    ddt=ps->host_layers[l].ffn_down_dt;
                int co2 = l * s->max_seq * kd + pos * kd;
                float *Kbase = s->key_cache   + (size_t)l * s->max_seq * kd;
                float *Vbase = s->value_cache + (size_t)l * s->max_seq * kd;
                const uint8_t *Wdn = (const uint8_t*)ps->host_layers[l].ffn_down;
                /* attention half-layer */
                nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps);
                if (qd==14)  nt_metal_q6k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D);
                else         nt_metal_q4k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D);
                if (kdt==14) nt_metal_q6k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D);
                else         nt_metal_q4k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D);
                if (vd==14)  nt_metal_q6k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D);
                else         nt_metal_q4k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D);
                nt_metal_rope(SLOT_Q, ps->host_heads,    hd, pos, ps->rope_theta);
                nt_metal_rope(SLOT_K, ps->host_kv_heads, hd, pos, ps->rope_theta);
                nt_metal_copy_to_region(s->key_cache   + co2, SLOT_K, kd*4);
                nt_metal_copy_to_region(s->value_cache + co2, SLOT_V, kd*4);
                nt_metal_attn_decode(SLOT_O, SLOT_Q, Kbase, Vbase, pos+1,
                                     ps->host_heads, ps->host_kv_heads, hd,
                                     (uint32_t)kd, (uint32_t)hd, (uint32_t)kd, (uint32_t)hd, sc);
                if (od==14)  nt_metal_q6k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn);
                else         nt_metal_q4k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn);
                nt_metal_add(SLOT_X, SLOT_X, SLOT_XB, D);
                /* parliament: election (w_vote@x -> variable-k gate) + LoRA inject,
                 * x += alpha * sum_k w_k * A_k @ (B_k @ x), all on the resident x. */
                if (parl) {
                    const float *lb = ps->expert_arena + (size_t)l * ps->expert_layer_floats;
                    nt_metal_parliament_votes(lb, SLOT_X, SLOT_PVOTES, D, MAX_EXPERTS);
                    nt_metal_parliament_elect(SLOT_PVOTES, SLOT_PRES, SLOT_PALIVE, SLOT_PCONS,
                                              SLOT_PGATE, MAX_EXPERTS, l, MIN_EXPERTS);
                    nt_metal_parliament_inject(SLOT_X, SLOT_PTMP, SLOT_PGATE, lb,
                                               D, ps->lora_rank, MAX_EXPERTS, ps->lora_alpha);
                }
                /* ffn half-layer (fused gate_up or separate gate + up) */
                nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps);
                if (ps->host_layers[l].ffn_gate_up) {
                    int gdt = ps->host_layers[l].ffn_gate_up_dt;
                    size_t gu_row = quant_raw_bytes(gdt, D);
                    const uint8_t *Wgu = (const uint8_t*)ps->host_layers[l].ffn_gate_up;
                    if (gdt==14) { nt_metal_q6k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D);
                                   nt_metal_q6k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D); }
                    else         { nt_metal_q4k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D);
                                   nt_metal_q4k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D); }
                } else {
                    const uint8_t *Wg = (const uint8_t*)ps->host_layers[l].ffn_gate;
                    const uint8_t *Wu = (const uint8_t*)ps->host_layers[l].ffn_up;
                    int g2 = ps->host_layers[l].ffn_gate_dt, u2 = ps->host_layers[l].ffn_up_dt;
                    if (g2==14) nt_metal_q6k_matvec_slot(SLOT_G, Wg, SLOT_FN, H, D);
                    else        nt_metal_q4k_matvec_slot(SLOT_G, Wg, SLOT_FN, H, D);
                    if (u2==14) nt_metal_q6k_matvec_slot(SLOT_U, Wu, SLOT_FN, H, D);
                    else        nt_metal_q4k_matvec_slot(SLOT_U, Wu, SLOT_FN, H, D);
                }
                nt_metal_silu_mul(SLOT_HB, SLOT_G, SLOT_U, H);
                if (ddt==14) nt_metal_q6k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H);
                else         nt_metal_q4k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H);
                nt_metal_add(SLOT_X, SLOT_X, SLOT_DN, D);
            }
            /* fold final norm + lm_head into the same command buffer when the
             * output weight is slot-matvec-able: norm -> lm_head on the GPU,
             * download logits (vocab) instead of x (dim). One fewer solo
             * matvec + round-trip. CPU lm_head path runs only when not folded. */
            int hf = ps->host_norm && ps->host_output && !getenv("DOE_NO_HEADFOLD") &&
                     (ps->host_output_dt==12 || ps->host_output_dt==14);
            if (hf) {
                nt_metal_slot_alloc(SLOT_LOGITS, (uint64_t)ps->host_vocab*4);
                nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_norm, D, ps->rms_norm_eps);
                if (ps->host_output_dt==14)
                    nt_metal_q6k_matvec_slot(SLOT_LOGITS, (const uint8_t*)ps->host_output, SLOT_FN, ps->host_vocab, D);
                else
                    nt_metal_q4k_matvec_slot(SLOT_LOGITS, (const uint8_t*)ps->host_output, SLOT_FN, ps->host_vocab, D);
            }
            nt_metal_batch_commit();
            if (hf) { nt_metal_slot_download(SLOT_LOGITS, s->logits, (uint64_t)ps->host_vocab*4); head_done = 1; }
            else      nt_metal_slot_download(SLOT_X, s->x, D*4);
            resident_done = 1;
            struct timespec _r1; clock_gettime(CLOCK_MONOTONIC, &_r1);
            g_resident_ns += (_r1.tv_sec-_r0.tv_sec)*1e9 + (_r1.tv_nsec-_r0.tv_nsec);
            if (getenv("DOE_DEBUG_TIMING") && pos == 0)
                fprintf(stderr, "[resident] engaged: whole token in 1 command buffer (%d layers, parliament=%s)\n",
                        ps->host_n_layers, parl ? "on" : "off");
        }
    }
#endif

    for (int l = 0; !resident_done && l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (!ps->host_layers[l].wq) continue;

        /* ── Host attention ── */
        int attn_done = 0;
#ifdef USE_METAL
        /* M4 stage (в): attention half-layer on GPU slots — rmsnorm -> qkv matvecs
         * -> rope -> copy_to_region(KV) -> attn_decode(GQA) -> o-proj -> residual,
         * one sync. rope_f32 == doe apply_rope (NeoX pairs, theta^-2i/hd). Guarded
         * to no-bias archs (Mistral) with Q4_K/Q6_K qkv/wo. CPU fallback otherwise. */
        {
            int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt;
            if (!getenv("DOE_NO_SLOTS") && nt_metal_available() && ps->host_layers[l].attn_norm &&
                !ps->host_layers[l].bq && !ps->host_layers[l].bk && !ps->host_layers[l].bv &&
                (qd==12||qd==14)&&(kdt==12||kdt==14)&&(vd==12||vd==14)&&(od==12||od==14)) {
                int co2 = l * s->max_seq * kd + pos * kd;
                float *Kbase = s->key_cache   + (size_t)l * s->max_seq * kd;
                float *Vbase = s->value_cache + (size_t)l * s->max_seq * kd;
                int qn = ps->host_heads * hd;
                nt_metal_slot_alloc(SLOT_X, D*4);   nt_metal_slot_alloc(SLOT_FN, D*4);
                nt_metal_slot_alloc(SLOT_Q, qn*4);  nt_metal_slot_alloc(SLOT_K, kd*4);
                nt_metal_slot_alloc(SLOT_V, kd*4);  nt_metal_slot_alloc(SLOT_O, qn*4);
                nt_metal_slot_alloc(SLOT_XB, D*4);
                nt_metal_slot_upload(SLOT_X, s->x, D*4);
                nt_metal_batch_begin();
                nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps);
                if (qd==14)  nt_metal_q6k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D);
                else         nt_metal_q4k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D);
                if (kdt==14) nt_metal_q6k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D);
                else         nt_metal_q4k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D);
                if (vd==14)  nt_metal_q6k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D);
                else         nt_metal_q4k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D);
                nt_metal_rope(SLOT_Q, ps->host_heads,    hd, pos, ps->rope_theta);
                nt_metal_rope(SLOT_K, ps->host_kv_heads, hd, pos, ps->rope_theta);
                nt_metal_copy_to_region(s->key_cache   + co2, SLOT_K, kd*4);
                nt_metal_copy_to_region(s->value_cache + co2, SLOT_V, kd*4);
                nt_metal_attn_decode(SLOT_O, SLOT_Q, Kbase, Vbase, pos+1,
                                     ps->host_heads, ps->host_kv_heads, hd,
                                     (uint32_t)kd, (uint32_t)hd, (uint32_t)kd, (uint32_t)hd, sc);
                if (od==14)  nt_metal_q6k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn);
                else         nt_metal_q4k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn);
                nt_metal_add(SLOT_X, SLOT_X, SLOT_XB, D);
                nt_metal_batch_commit();
                nt_metal_slot_download(SLOT_X, s->x, D*4);
                attn_done = 1;
            }
        }
#endif
        if (!attn_done) {
        float *xn = s->xb;
        if (ps->host_layers[l].attn_norm) rmsnorm(xn, s->x, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps);
        else memcpy(xn, s->x, D*4);

        /* {q,k,v} share xn — batch into one command buffer, one GPU sync (Metal).
         * out (s->q/k/v) is written only at commit, so commit must precede the
         * bias adds below that read them. Bit-identical to the solo path. */
#ifdef USE_METAL
        int qkv_batch = !getenv("DOE_NO_BATCH") && nt_metal_available() && nt_metal_batch_begin() == 0;
#endif
        doe_mv(s->q, ps->host_layers[l].wq, ps->host_layers[l].wq_dt, xn, ps->host_heads*hd, D);
        doe_mv(s->k, ps->host_layers[l].wk, ps->host_layers[l].wk_dt, xn, kd, D);
        doe_mv(s->v, ps->host_layers[l].wv, ps->host_layers[l].wv_dt, xn, kd, D);
#ifdef USE_METAL
        if (qkv_batch) nt_metal_batch_commit();
#endif

        /* Add attention biases (Qwen2, optional) */
        if (ps->host_layers[l].bq) for (int i = 0; i < ps->host_heads*hd; i++) s->q[i] += ps->host_layers[l].bq[i];
        if (ps->host_layers[l].bk) for (int i = 0; i < kd; i++) s->k[i] += ps->host_layers[l].bk[i];
        if (ps->host_layers[l].bv) for (int i = 0; i < kd; i++) s->v[i] += ps->host_layers[l].bv[i];

        for (int h = 0; h < ps->host_heads; h++) apply_rope(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        for (int h = 0; h < ps->host_kv_heads; h++) apply_rope(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd);

        int co = l * s->max_seq * kd + pos * kd;
        memcpy(s->key_cache + co, s->k, kd*4);
        memcpy(s->value_cache + co, s->v, kd*4);

        float *ao = s->xb2; memset(ao, 0, D*4);
        for (int h = 0; h < ps->host_heads; h++) {
            int kvh = h / hg; float *qh = s->q + h*hd;
            float *att = s->att + h * s->max_seq;
            for (int t = 0; t <= pos; t++) {
                int ko = l*s->max_seq*kd + t*kd + kvh*hd;
                float dot = 0;
                for (int d = 0; d < hd; d++) dot += qh[d] * s->key_cache[ko+d];
                att[t] = dot * sc;
            }
            softmax_n(att, pos+1);
            float *oh = ao + h*hd;
            for (int t = 0; t <= pos; t++) {
                float a = att[t]; int vo = l*s->max_seq*kd + t*kd + kvh*hd;
                for (int d = 0; d < hd; d++) oh[d] += a * s->value_cache[vo+d];
            }
        }
        doe_mv(s->xb, ps->host_layers[l].wo, ps->host_layers[l].wo_dt, ao, D, ps->host_heads*hd); /* Y-1b: WO cols = heads*head_dim (4096), not dim (5120) */
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
        } /* !attn_done */

        /* ── Parliament election + LoRA injection (after attention, before FFN) ── */
        /* A13: skip injection entirely when alpha=0 — avoids 0*NaN poisoning from drifted experts */
        if (l < ps->n_field_layers && ps->lora_alpha != 0.0f) {
            FieldLayer *fl = &ps->field_layers[l];
            int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
            int k = parliament_elect(&fl->parliament, fl->experts, s->x, D, &s->hs, selected, weights);
            memset(s->expert_out, 0, D*4);
            for (int ki = 0; ki < k; ki++) {
                LoraExpert *exp = &fl->experts[selected[ki]];
                exp->tokens_seen++;
                /* Delta Voice: out += α × A @ (B @ x) */
                float tmp[LORA_RANK]; memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < ps->lora_rank; r++)
                    for (int j = 0; j < D; j++)
                        tmp[r] += exp->lora_B[r * D + j] * s->x[j];
                float lora_out[D]; memset(lora_out, 0, D*4);
                for (int i = 0; i < D; i++)
                    for (int r = 0; r < ps->lora_rank; r++)
                        lora_out[i] += exp->lora_A[i * ps->lora_rank + r] * tmp[r];
                for (int i = 0; i < D; i++)
                    s->expert_out[i] += weights[ki] * ps->lora_alpha * lora_out[i];
            }
            for (int i = 0; i < D; i++) s->x[i] += s->expert_out[i];
        }

        /* ── Host FFN (SwiGLU) ── */
        {
            int ffn_done = 0;
#ifdef USE_METAL
            /* M4 stage (b): FFN half-layer on GPU slots — rmsnorm -> gate/up matvecs ->
             * silu_mul -> down -> residual, one sync. Fused gate_up is split into two
             * sub-W matvecs (silu_mul needs two slots). x is host-bridged this stage
             * (attention still CPU); stage (г) keeps x resident. Falls through to the
             * CPU path when Metal is off or a dt is unsupported by the slot matvec. */
            {
                int gdt = ps->host_layers[l].ffn_gate_up_dt, ddt = ps->host_layers[l].ffn_down_dt;
                if (!getenv("DOE_NO_SLOTS") && nt_metal_available() && ps->host_layers[l].ffn_norm &&
                    ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down &&
                    (gdt==12||gdt==14) && (ddt==12||ddt==14)) {
                    const uint8_t *Wgu = (const uint8_t*)ps->host_layers[l].ffn_gate_up;
                    const uint8_t *Wdn = (const uint8_t*)ps->host_layers[l].ffn_down;
                    size_t gu_row = quant_raw_bytes(gdt, D);
                    nt_metal_slot_alloc(SLOT_X, D*4);  nt_metal_slot_alloc(SLOT_FN, D*4);
                    nt_metal_slot_alloc(SLOT_G, H*4);  nt_metal_slot_alloc(SLOT_U, H*4);
                    nt_metal_slot_alloc(SLOT_HB, H*4); nt_metal_slot_alloc(SLOT_DN, D*4);
                    nt_metal_slot_upload(SLOT_X, s->x, D*4);
                    nt_metal_batch_begin();
                    nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps);
                    if (gdt==14) { nt_metal_q6k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D);
                                   nt_metal_q6k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D); }
                    else         { nt_metal_q4k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D);
                                   nt_metal_q4k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D); }
                    nt_metal_silu_mul(SLOT_HB, SLOT_G, SLOT_U, H);
                    if (ddt==14) nt_metal_q6k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H);
                    else         nt_metal_q4k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H);
                    nt_metal_add(SLOT_X, SLOT_X, SLOT_DN, D);
                    nt_metal_batch_commit();
                    nt_metal_slot_download(SLOT_X, s->x, D*4);
                    ffn_done = 1;
                }
            }
#endif
            if (!ffn_done) {
            float *fn = s->xb;
            if (ps->host_layers[l].ffn_norm) rmsnorm(fn, s->x, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps);
            else memcpy(fn, s->x, D*4);

            if (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) {
                /* Fused gate_up: [hidden*2, dim] → split into gate [0..H) and up [H..2H) */
                doe_mv(s->hb2, ps->host_layers[l].ffn_gate_up, ps->host_layers[l].ffn_gate_up_dt, fn, H * 2, D);
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb2[i]) * s->hb2[H + i];
                doe_mv(s->xb, ps->host_layers[l].ffn_down, ps->host_layers[l].ffn_down_dt, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            } else if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) {
                /* Standard separate gate + up — share fn, batch into one sync. */
#ifdef USE_METAL
                int gu_batch = !getenv("DOE_NO_BATCH") && nt_metal_available() && nt_metal_batch_begin() == 0;
#endif
                doe_mv(s->hb, ps->host_layers[l].ffn_gate, ps->host_layers[l].ffn_gate_dt, fn, H, D);
                doe_mv(s->hb2, ps->host_layers[l].ffn_up, ps->host_layers[l].ffn_up_dt, fn, H, D);
#ifdef USE_METAL
                if (gu_batch) nt_metal_batch_commit();
#endif
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb[i]) * s->hb2[i];
                doe_mv(s->xb, ps->host_layers[l].ffn_down, ps->host_layers[l].ffn_down_dt, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            }
            } /* !ffn_done */
        }
    }

    /* Final norm + LM head (skipped when folded into the resident CB above) */
    if (!head_done) {
        struct timespec _h0; clock_gettime(CLOCK_MONOTONIC, &_h0);
        rmsnorm(s->x, s->x, ps->host_norm, D, ps->rms_norm_eps);
        doe_mv(s->logits, ps->host_output, ps->host_output_dt, s->x, ps->host_vocab, D);
        struct timespec _h1; clock_gettime(CLOCK_MONOTONIC, &_h1);
        g_head_ns += (_h1.tv_sec-_h0.tv_sec)*1e9 + (_h1.tv_nsec-_h0.tv_nsec);
    }

    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SAMPLING + CHAT
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int sample(float *logits, int V, float temp, int top_k) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) logits[i] /= temp;
    if (top_k > 0 && top_k < V) {
        /* threshold = top_k-th largest logit via a size-k min-heap: O(V log k),
         * no per-token malloc. Root holds the smallest of the k kept = same
         * threshold the old O(k*V) selection-sort produced (value is identical
         * regardless of tie-breaking), so the sampled token is bit-identical. */
        float heap[256]; int hk = top_k > 256 ? 256 : top_k; int n = 0;
        for (int i = 0; i < V; i++) {
            float v = logits[i];
            if (n < hk) {
                int c = n++; heap[c] = v;
                while (c > 0) { int p=(c-1)/2; if (heap[p] <= heap[c]) break;
                               float t=heap[p]; heap[p]=heap[c]; heap[c]=t; c=p; }
            } else if (v > heap[0]) {
                heap[0] = v; int c = 0;
                for (;;) { int l=2*c+1, r=2*c+2, m=c;
                           if (l<hk && heap[l]<heap[m]) m=l;
                           if (r<hk && heap[r]<heap[m]) m=r;
                           if (m==c) break;
                           float t=heap[m]; heap[m]=heap[c]; heap[c]=t; c=m; }
            }
        }
        float th = heap[0];
        for (int i = 0; i < V; i++) if (logits[i] < th) logits[i] = -1e30f;
    }
    softmax_n(logits, V);
    float r = rand_uniform(), cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

/* GPT-2 byte_decoder: reverse the byte_encoder mapping (unicode codepoint -> original byte) */
static int gpt2_rune_to_byte(int rune) {
    static int table_built = 0;
    static int rtable[512]; /* rune -> byte, -1 if not mapped */
    if (!table_built) {
        for (int i = 0; i < 512; i++) rtable[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                rtable[b] = b; /* identity mapping */
            else
                rtable[256 + n++] = b; /* offset mapping */
        }
        table_built = 1;
    }
    if (rune >= 0 && rune < 512) return rtable[rune];
    return -1;
}

/* Parse one UTF-8 codepoint, return codepoint and advance *p by bytes consumed */
static int utf8_decode_cp(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    int cp, len;
    if (s[0] < 0x80) { cp = s[0]; len = 1; }
    else if ((s[0] & 0xE0) == 0xC0) { cp = (s[0] & 0x1F) << 6 | (s[1] & 0x3F); len = 2; }
    else if ((s[0] & 0xF0) == 0xE0) { cp = (s[0] & 0x0F) << 12 | (s[1] & 0x3F) << 6 | (s[2] & 0x3F); len = 3; }
    else if ((s[0] & 0xF8) == 0xF0) { cp = (s[0] & 0x07) << 18 | (s[1] & 0x3F) << 12 | (s[2] & 0x3F) << 6 | (s[3] & 0x3F); len = 4; }
    else { cp = s[0]; len = 1; } /* fallback */
    *p += len;
    return cp;
}

/* Decode token to text using GGUF vocab, fallback to byte */
static void token_decode_print(GGUFIndex *ps, int token) {
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            /* GPT-2 byte-level BPE: full byte_decoder — each codepoint maps to one byte */
            unsigned char buf[256];
            int blen = 0;
            const char *p = s;
            while (*p && blen < (int)sizeof(buf) - 4) {
                int cp = utf8_decode_cp(&p);
                int b = gpt2_rune_to_byte(cp);
                if (b >= 0) buf[blen++] = (unsigned char)b;
                else {
                    /* Not a GPT-2 mapped byte — emit codepoint as UTF-8 */
                    if (cp < 0x80) { buf[blen++] = cp; }
                    else if (cp < 0x800) { buf[blen++] = 0xC0|(cp>>6); buf[blen++] = 0x80|(cp&0x3F); }
                    else if (cp < 0x10000) { buf[blen++] = 0xE0|(cp>>12); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                    else { buf[blen++] = 0xF0|(cp>>18); buf[blen++] = 0x80|((cp>>12)&0x3F); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                }
            }
            fwrite(buf, 1, blen, stdout);
        } else {
        /* Handle sentencepiece ▁ (U+2581, 3 bytes: E2 96 81) → space */
        while (*s) {
            if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
                fputc(' ', stdout);
                s += 3;
            } else if (!strncmp(s, "<0x", 3) && s[5] == '>') {
                /* sentencepiece hex byte: <0xAB> */
                unsigned int b = 0;
                sscanf(s + 3, "%02X", &b);
                if (b >= 32 || b == '\n' || b == '\t') fputc((char)b, stdout);
                s += 6;
            } else {
                fputc(*s, stdout);
                s++;
            }
        }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if (c >= 32 || c == '\n' || c == '\t') fputc(c, stdout);
    }
}

/* Decode token to buffer instead of stdout — for HTTP serve mode */
static int token_decode_buf(GGUFIndex *ps, int token, char *buf, int bufsz) {
    int pos = 0;
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            const char *p = s;
            while (*p && pos < bufsz - 4) {
                int cp = utf8_decode_cp(&p);
                int b = gpt2_rune_to_byte(cp);
                if (b >= 0) buf[pos++] = (char)(unsigned char)b;
                else {
                    if (cp < 0x80) { buf[pos++] = cp; }
                    else if (cp < 0x800 && pos < bufsz-2) { buf[pos++] = 0xC0|(cp>>6); buf[pos++] = 0x80|(cp&0x3F); }
                    else if (cp < 0x10000 && pos < bufsz-3) { buf[pos++] = 0xE0|(cp>>12); buf[pos++] = 0x80|((cp>>6)&0x3F); buf[pos++] = 0x80|(cp&0x3F); }
                }
            }
        } else {
            while (*s && pos < bufsz - 1) {
                if ((unsigned char)s[0]==0xE2 && (unsigned char)s[1]==0x96 && (unsigned char)s[2]==0x81) {
                    buf[pos++]=' '; s+=3;
                } else if (!strncmp(s,"<0x",3) && s[5]=='>') {
                    unsigned int b=0; sscanf(s+3,"%02X",&b);
                    if (b>=32||b=='\n'||b=='\t') buf[pos++]=(char)b;
                    s+=6;
                } else { buf[pos++]=*s; s++; }
            }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if ((c>=32||c=='\n'||c=='\t') && pos < bufsz-1) buf[pos++]=c;
    }
    buf[pos] = '\0';
    return pos;
}

/* ── BPE Tokenizer — SentencePiece style, score-based merge ── */

static int tok_lookup(GGUFIndex *ps, const char *s, int len); /* forward decl */

/* Build GPT-2 BPE scores from merges (called after index_load if needed) */
static void build_gpt2_scores(GGUFIndex *ps) {
    if (!ps->is_gpt2_bpe || !ps->bpe_merges || ps->n_bpe_merges == 0 || ps->vocab_scores || !ps->vocab_tokens) return;
    ps->vocab_scores = calloc(ps->vocab_size, sizeof(float));
    for (int i = 0; i < ps->vocab_size; i++) ps->vocab_scores[i] = -1e9f;
    int built = 0;
    for (int m = 0; m < ps->n_bpe_merges; m++) {
        const char *merge = ps->bpe_merges[m];
        const char *sp = strchr(merge, ' ');
        if (!sp) continue;
        int la = (int)(sp - merge), lb = (int)strlen(sp + 1);
        if (la + lb > 128) continue;
        char merged[130];
        memcpy(merged, merge, la);
        memcpy(merged + la, sp + 1, lb);
        int mid = tok_lookup(ps, merged, la + lb);
        if (mid >= 0) { ps->vocab_scores[mid] = (float)(ps->n_bpe_merges - m); built++; }
    }
    printf("[doe] GPT-2 BPE: built %d merge scores from %d merges\n", built, ps->n_bpe_merges);
    ps->add_space_prefix = 0;
}

/* FNV-1a hash */
static uint32_t tok_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

/* Build hash table for O(1) token lookup */
static void tok_ht_build(GGUFIndex *ps) {
    if (!ps->vocab_tokens || ps->vocab_size == 0) return;
    int cap = 1;
    while (cap < ps->vocab_size * 3) cap <<= 1; /* ~33% load factor */
    ps->tok_ht_ids = malloc(cap * sizeof(int));
    ps->tok_ht_cap = cap;
    for (int i = 0; i < cap; i++) ps->tok_ht_ids[i] = -1;
    int mask = cap - 1;
    for (int i = 0; i < ps->vocab_size; i++) {
        if (!ps->vocab_tokens[i]) continue;
        int slen = (int)strlen(ps->vocab_tokens[i]);
        uint32_t idx = tok_hash(ps->vocab_tokens[i], slen) & mask;
        while (ps->tok_ht_ids[idx] != -1) idx = (idx + 1) & mask;
        ps->tok_ht_ids[idx] = i;
    }
}

/* Find token ID by string. Returns -1 if not found. O(1) average. */
static int tok_lookup(GGUFIndex *ps, const char *s, int len) {
    if (!ps->tok_ht_ids) {
        /* fallback linear scan */
        for (int i = 0; i < ps->vocab_size; i++) {
            if (ps->vocab_tokens[i] && strlen(ps->vocab_tokens[i]) == (size_t)len
                && memcmp(ps->vocab_tokens[i], s, len) == 0)
                return i;
        }
        return -1;
    }
    int mask = ps->tok_ht_cap - 1;
    uint32_t idx = tok_hash(s, len) & mask;
    while (ps->tok_ht_ids[idx] != -1) {
        int id = ps->tok_ht_ids[idx];
        const char *t = ps->vocab_tokens[id];
        if (t && strlen(t) == (size_t)len && memcmp(t, s, len) == 0) return id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* Score-based BPE merge on an array of token IDs */
static int bpe_merge(GGUFIndex *ps, int *ids, int n) {
    if (!ps->vocab_scores) return n;
    while (n > 1) {
        float best_score = -1e30f;
        int best_idx = -1, best_id = -1;
        for (int i = 0; i < n - 1; i++) {
            /* Concatenate token strings */
            const char *a = ps->vocab_tokens[ids[i]];
            const char *b = ps->vocab_tokens[ids[i+1]];
            if (!a || !b) continue;
            int la = strlen(a), lb = strlen(b);
            if (la + lb > 128) continue;
            char merged[130];
            memcpy(merged, a, la);
            memcpy(merged + la, b, lb);
            int mid = tok_lookup(ps, merged, la + lb);
            if (mid >= 0 && ps->vocab_scores[mid] > best_score) {
                best_score = ps->vocab_scores[mid];
                best_idx = i;
                best_id = mid;
            }
        }
        if (best_idx < 0) break;
        ids[best_idx] = best_id;
        /* Remove ids[best_idx+1] by shifting */
        for (int i = best_idx + 1; i < n - 1; i++) ids[i] = ids[i+1];
        n--;
    }
    return n;
}

/* GPT-2 byte-to-unicode table: maps each byte to a unicode codepoint */
static int gpt2_byte_to_rune(int b) {
    /* Printable ASCII + Latin-1 supplement range → identity */
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
        return b;
    /* Everything else → 256 + offset */
    static int table_built = 0;
    static int table[256];
    if (!table_built) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
                table[i] = i;
            else
                table[i] = 256 + n++;
        }
        table_built = 1;
    }
    return table[(unsigned char)b];
}

/* Encode a unicode codepoint as UTF-8, return length */
static int rune_to_utf8(int r, char *out) {
    if (r < 0x80) { out[0] = (char)r; return 1; }
    if (r < 0x800) { out[0] = 0xC0 | (r >> 6); out[1] = 0x80 | (r & 0x3F); return 2; }
    out[0] = 0xE0 | (r >> 12); out[1] = 0x80 | ((r >> 6) & 0x3F); out[2] = 0x80 | (r & 0x3F); return 3;
}

/* Try to match a special token at position i in text. Returns token id and advances *len. */
static int try_special_token(GGUFIndex *ps, const char *text, int tlen, int i, int *consumed) {
    static const char *specials[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|end|>",
        "<start_of_turn>", "<end_of_turn>", "<|user|>", "<|assistant|>",
        "[INST]", "[/INST]", "<s>", "</s>",
        "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
        "<|bos|>", "<|eot_id|>", NULL
    };
    if (text[i] != '<' && text[i] != '[') return -1;
    for (int s = 0; specials[s]; s++) {
        int slen = (int)strlen(specials[s]);
        if (i + slen <= tlen && memcmp(text + i, specials[s], slen) == 0) {
            int id = tok_lookup(ps, specials[s], slen);
            if (id >= 0) { *consumed = slen; return id; }
        }
    }
    return -1;
}

static int tokenize_input(GGUFIndex *ps, const char *text, int *tokens, int max_tokens) {
    if (!ps->vocab_tokens) {
        int n = 0, len = strlen(text);
        for (int i = 0; i < len && n < max_tokens; i++) tokens[n++] = (unsigned char)text[i];
        return n;
    }

    int tlen = strlen(text);
    int *ids = malloc((tlen + 16) * sizeof(int));
    int n = 0;

    if (ps->is_gpt2_bpe) {
        /* GPT-2: check special tokens first, then byte-level BPE */
        for (int i = 0; i < tlen && n < max_tokens; ) {
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }
            int r = gpt2_byte_to_rune((unsigned char)text[i]);
            char u8[4]; int u8len = rune_to_utf8(r, u8);
            int id = tok_lookup(ps, u8, u8len);
            ids[n++] = (id >= 0) ? id : 0;
            i++;
        }
    } else {
        /* SentencePiece: split on special tokens first, then ▁-encode segments */
        int i = 0;
        while (i < tlen && n < max_tokens) {
            /* Check special tokens at raw text level */
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }

            /* Find next special token boundary (or end) */
            int seg_end = i + 1;
            while (seg_end < tlen) {
                int c2 = 0;
                if (try_special_token(ps, text, tlen, seg_end, &c2) >= 0) break;
                seg_end++;
            }

            /* Encode segment [i, seg_end) with SentencePiece ▁ */
            int slen = seg_end - i;
            char *sp = malloc(slen * 3 + 4);
            int sp_len = 0;
            if (ps->add_space_prefix && i == 0 && text[i] != ' ') {
                sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
            }
            for (int j = i; j < seg_end; j++) {
                if (text[j] == ' ') {
                    sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
                } else {
                    sp[sp_len++] = text[j];
                }
            }
            sp[sp_len] = '\0';
            int k = 0;
            while (k < sp_len && n < max_tokens) {
                int clen = 1;
                unsigned char c = (unsigned char)sp[k];
                if (c >= 0xC0 && c < 0xE0) clen = 2;
                else if (c >= 0xE0 && c < 0xF0) clen = 3;
                else if (c >= 0xF0) clen = 4;
                if (k + clen > sp_len) clen = 1;
                int id = tok_lookup(ps, sp + k, clen);
                if (id >= 0) { ids[n++] = id; k += clen; }
                else {
                    char hex[7]; snprintf(hex, 7, "<0x%02X>", (unsigned char)sp[k]);
                    id = tok_lookup(ps, hex, 6);
                    ids[n++] = (id >= 0) ? id : 0; k++;
                }
            }
            free(sp);
            i = seg_end;
        }
    }

    n = bpe_merge(ps, ids, n);
    int out = (n < max_tokens) ? n : max_tokens;
    memcpy(tokens, ids, out * sizeof(int));
    free(ids);
    return out;
}

static void chat(GGUFIndex *ps) {
    int max_seq = 512;
    InferState is = alloc_infer(ps, max_seq);
    CalendarDrift cd; drift_init(&cd);
    MetaTrack meta; meta_init(&meta);
    HarmonicState hs = {0};

    char input[1024];
    printf("\n[doe] the parliament is in session. type your message (Ctrl+C to dissipate):\n");
    printf("[doe] host: %s (%s, %dM params)\n\n",
           ps->host_path, ps->host_arch,
           (int)(ps->host_vocab * ps->host_dim * 2 / 1000000)); /* rough estimate */

    float debt_sum = 0; int debt_count = 0;

    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input);
        while (len > 0 && (input[len-1]=='\n' || input[len-1]=='\r')) input[--len] = '\0';
        if (!len) continue;
        if (strcmp(input,"quit")==0 || strcmp(input,"exit")==0) break;
        if (strcmp(input,"status")==0) {
            printf("[field] step=%d debt=%.3f entropy=%.3f resonance=%.3f emergence=%.3f\n",
                   F.step, F.debt, F.entropy, F.resonance, F.emergence);
            printf("[field] season=%s health=%.3f temp=%.3f velocity=%s\n",
                   (const char*[]){"spring","summer","autumn","winter"}[F.season],
                   F.field_health, F.effective_temp,
                   (const char*[]){"nomove","walk","run","backward"}[F.velocity_mode]);
            printf("[drift] d=%.3f stability=%.3f accel=%.4f snapshots=%d\n",
                   cd.drift, cd.stability, cd.drift_accel, cd.n_snapshots);
            int te = 0;
            for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
            printf("[experts] alive=%d consensus=%.2f elections=%d\n",
                   te, ps->field_layers[0].parliament.consensus,
                   ps->field_layers[0].parliament.election_count);
            if (debt_count > 0)
                printf("[prophecy] avg_debt=%.4f total_debt=%.4f\n", debt_sum/debt_count, F.debt);
            continue;
        }

        /* Reset KV cache */
        int kd = ps->host_kv_heads * ps->host_head_dim;
        memset(is.key_cache, 0, ps->host_n_layers * max_seq * kd * 4);
        memset(is.value_cache, 0, ps->host_n_layers * max_seq * kd * 4);

        /* Wrap input in chat template (auto-detected from GGUF chat_template) */
        char wrapped[2048];
        /* Only use chat template if the key special tokens exist in vocab */
        int use_template = 0;
        switch (ps->chat_style) {
        case 1: /* ChatML */
            if (tok_lookup(ps, "<|im_start|>", 12) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input);
                use_template = 1;
            }
            break;
        case 2: /* [INST] */
            if (tok_lookup(ps, "[INST]", 6) >= 0) {
                snprintf(wrapped, sizeof(wrapped), "[INST] %s [/INST]", input);
                use_template = 1;
            }
            break;
        case 3: /* Zephyr */
            if (tok_lookup(ps, "<|user|>", 8) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|user|>\n%s\n<|assistant|>\n", input);
                use_template = 1;
            }
            break;
        case 4: /* Phi */
            if (tok_lookup(ps, "<|end|>", 7) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<|user|>\n%s<|end|>\n<|assistant|>\n", input);
                use_template = 1;
            }
            break;
        case 5: /* Gemma */
            if (tok_lookup(ps, "<start_of_turn>", 15) >= 0) {
                snprintf(wrapped, sizeof(wrapped),
                    "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", input);
                use_template = 1;
            }
            break;
        case 6: /* nanollama — <|user_start|>...<|user_end|><|assistant_start|> */
            snprintf(wrapped, sizeof(wrapped),
                "<|user_start|>%s<|user_end|><|assistant_start|>", input);
            use_template = 1;
            break;
        }
        if (!use_template) snprintf(wrapped, sizeof(wrapped), "%s", input);

        /* Tokenize wrapped input */
        int input_tokens[512];
        int n_input = 0;
        if (ps->bos_id >= 0) input_tokens[n_input++] = ps->bos_id;
        n_input += tokenize_input(ps, wrapped, input_tokens + n_input, 512 - n_input);

        /* A12a: input-id dump WITH bos for parity vs llama (env-guarded, diagnostic only) */
        if (getenv("DOE_DEBUG_INPUT")) {
            fprintf(stderr, "[inputdump] n=%d ids:", n_input);
            for (int di = 0; di < n_input; di++) fprintf(stderr, " %d", input_tokens[di]);
            fprintf(stderr, "\n");
        }

        /* A14 (Mythos gen-only): repetition history = GENERATED tokens ONLY — prompt untouched.
         * A10a poisoned the start by penalizing prompt identity-tokens; here rep_hist stays empty
         * until generation begins. Window = last ~64 generated (applied below). */
        int rep_hist[256]; int n_rep = 0;

        int pos = 0;
        for (int i = 0; i < n_input - 1 && pos < max_seq - 1; i++, pos++) {
            doe_forward(ps, &is, input_tokens[i], pos);
            dario_ingest(input_tokens[i]); /* feed user tokens into Dario field */
        }

        int prev = input_tokens[n_input - 1];
        dario_ingest(prev); /* D-H2: last token still feeds the field; forwarded once in generation */
        printf("  ");
        int total_births = 0, total_deaths = 0;

        struct timespec _gt0, _gt1; clock_gettime(CLOCK_MONOTONIC, &_gt0); int _gi = 0;
        g_prof_mv_ns = 0; g_resident_ns = 0; g_head_ns = 0; /* reset: profile over decode loop only (exclude prefill) */
        for (int _g = 0; _g < MVG_N; _g++) { g_ps_ns[_g] = 0; g_ps_cnt[_g] = 0; }
        for (int i = 0; i < 200 && pos < max_seq; i++, pos++) {
            _gi = i + 1;
            float *lg = doe_forward(ps, &is, prev, pos);

            /* A12b: raw host-forward token-1 NaN/Inf count + finite top-5 (before field/sample) */
            if (i == 0 && getenv("DOE_DEBUG_LOGITS")) {
                int V = ps->host_vocab, nnan = 0, ninf = 0;
                float *tmp = malloc(V * sizeof(float));
                for (int v = 0; v < V; v++) {
                    if (isnan(lg[v])) { nnan++; tmp[v] = -1e30f; }
                    else if (isinf(lg[v])) { ninf++; tmp[v] = -1e30f; }
                    else tmp[v] = lg[v];
                }
                fprintf(stderr, "[logitdump] tok1 nan=%d inf=%d /%d, finite top5:", nnan, ninf, V);
                for (int top = 0; top < 5; top++) {
                    int bi = 0; for (int v = 1; v < V; v++) if (tmp[v] > tmp[bi]) bi = v;
                    const char *ts = (ps->vocab_tokens && ps->vocab_tokens[bi]) ? ps->vocab_tokens[bi] : "?";
                    fprintf(stderr, " (%d '%s'=%.3f)", bi, ts, tmp[bi]);
                    tmp[bi] = -1e30f;
                }
                fprintf(stderr, "\n");
                free(tmp);
            }

            /* Field modulation on logits — Dario Equation */
            field_step(1.0f);
            apply_field_to_logits(lg, ps->host_vocab);

            /* A10a: repetition penalty over recent history (llama-style), last 64 tokens */
            if (g_rep_penalty > 1.0f) {
                int rstart = n_rep > 64 ? n_rep - 64 : 0;
                for (int r = rstart; r < n_rep; r++) {
                    int t = rep_hist[r];
                    if (t >= 0 && t < ps->host_vocab)
                        lg[t] = lg[t] > 0 ? lg[t] / g_rep_penalty : lg[t] * g_rep_penalty;
                }
            }

            int next = sample(lg, ps->host_vocab, F.effective_temp, 40);
            if (n_rep < 256) rep_hist[n_rep++] = next;

            /* Stop on EOS or chat-template end tokens */
            if (next == ps->eos_id) break;
            if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
                const char *ts = ps->vocab_tokens[next];
                if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                    strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                    strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                    strcmp(ts, "<|eot_id|>") == 0)
                    break;
            }

            /* Prophecy debt — retroactive conscience */
            float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
            F.debt += pd;
            debt_sum += pd; debt_count++;

            /* Dario field: ingest generated token (co-occurrence + prophecy + destiny) */
            dario_ingest(next);

            /* NOTORCH Hebbian update — debt drives learning (A13b: gated by --train) */
            float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
            if (g_train) for (int l = 0; l < ps->n_field_layers; l++) {
                FieldLayer *fl = &ps->field_layers[l];
                for (int e = 0; e < MAX_EXPERTS; e++) {
                    if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                    notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                                ps->host_dim, ps->host_dim, ps->lora_rank,
                                is.x, is.xb, learn_signal);
                    /* A14: poison-quarantine (NaN/Inf + finite |value|-explosion) — freeze + death */
                    if (lora_poisoned(fl->experts[e].lora_A, fl->experts[e].lora_B)) {
                        fl->experts[e].alive = 0; total_deaths++;
                    }
                }
            }

            /* Vitality + mitosis + apoptosis */
            if (i % 10 == 0) {
                /* Harmonic decomposition */
                float lh[16]; int lhl = 0;
                for (int j = 0; j < 16 && j < i; j++) lh[lhl++] = F.entropy;
                if (lhl > 2) harmonic_decompose(&is.hs, lh, lhl);

                for (int l = 0; l < ps->n_field_layers; l++) {
                    update_expert_vitality(&ps->field_layers[l], 10);
                    if (try_mitosis(&ps->field_layers[l], ps->host_dim, ps->lora_rank)) total_births++;
                    if (try_apoptosis(&ps->field_layers[l])) total_deaths++;
                }
            }

            /* Drift snapshot */
            if (i % DRIFT_INTERVAL == 0 && i > 0)
                drift_snapshot(&cd, F.debt, ps, &is.hs);

            token_decode_print(ps, next);
            fflush(stdout);
            prev = next;
        }
        clock_gettime(CLOCK_MONOTONIC, &_gt1);
        if (getenv("DOE_DEBUG_TIMING") || g_prof_on > 0) {
            double _el = (_gt1.tv_sec - _gt0.tv_sec) + (_gt1.tv_nsec - _gt0.tv_nsec) / 1e9;
            fprintf(stderr, "[timing] decode %d tokens in %.2fs = %.2f t/s\n", _gi, _el, _el > 0 ? _gi / _el : 0.0);
            if (g_prof_on > 0) {
                double _mv = g_prof_mv_ns / 1e9;
                fprintf(stderr, "[profile] matvec %.2fs (%.0f%% of decode) | rest %.2fs (attention/rmsnorm/FFN/sample)\n",
                        _mv, _el > 0 ? 100 * _mv / _el : 0.0, _el - _mv);
                double _res = g_resident_ns/1e9, _hd = g_head_ns/1e9;
                fprintf(stderr, "[profile] resident-CB %.2fs (%.0f%%, %.1f ms/tok) | lm_head+norm %.2fs (%.0f%%, %.1f ms/tok) | other %.2fs\n",
                        _res, _el>0?100*_res/_el:0.0, _gi>0?_res*1e3/_gi:0.0,
                        _hd, _el>0?100*_hd/_el:0.0, _gi>0?_hd*1e3/_gi:0.0, _el-_res-_hd);
            }
            if (g_ps_on > 0 && _gi > 0) {
                static const char *_gn[MVG_N] = {"attn_qkv","attn_o","ffn_gate_up","ffn_down","lm_head"};
                double _tot = 0; for (int _g = 0; _g < MVG_N; _g++) _tot += g_ps_ns[_g];
                fprintf(stderr, "[per-shape] %d tokens, per-token (bytes-share check):\n", _gi);
                for (int _g = 0; _g < MVG_N; _g++)
                    fprintf(stderr, "  %-12s %.3f ms/tok  %.1f disp/tok  (m=%d k=%d dt=%d)  %.0f%% time\n",
                            _gn[_g], g_ps_ns[_g]/1e6/_gi, (double)g_ps_cnt[_g]/_gi,
                            g_ps_m[_g], g_ps_k[_g], g_ps_dt[_g], _tot > 0 ? 100*g_ps_ns[_g]/_tot : 0.0);
            }
        }
        printf("\n");

        /* Meta record */
        int te = 0;
        for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
        meta_record(&meta, F.step, te, ps->field_layers[0].parliament.consensus,
                    F.debt, F.field_health, debt_count > 0 ? debt_sum/debt_count : 0,
                    cd.drift, F.debt);

        if (total_births > 0 || total_deaths > 0)
            printf("  [life] births=%d deaths=%d\n", total_births, total_deaths);
        printf("\n");
    }
    free_infer(&is);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HTTP SERVE MODE — minimal HTTP server for doe_ui.html and doe.html
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int g_serve_port = 0; /* 0 = disabled */
static int g_serve_public = 0; /* D-M4: 0 = bind 127.0.0.1 (default), 1 = 0.0.0.0 (--serve-public) */

/* JSON-escape a string into buf. Returns bytes written (not counting NUL). */
static int json_escape(const char *src, char *buf, int bufsz) {
    int p = 0;
    for (; *src && p < bufsz - 2; src++) {
        switch (*src) {
        case '"':  if(p+2<bufsz){buf[p++]='\\';buf[p++]='"';}  break;
        case '\\': if(p+2<bufsz){buf[p++]='\\';buf[p++]='\\';} break;
        case '\n': if(p+2<bufsz){buf[p++]='\\';buf[p++]='n';}  break;
        case '\r': if(p+2<bufsz){buf[p++]='\\';buf[p++]='r';}  break;
        case '\t': if(p+2<bufsz){buf[p++]='\\';buf[p++]='t';}  break;
        default:   buf[p++] = *src; break;
        }
    }
    buf[p] = '\0';
    return p;
}

/* Read full HTTP request into buf, return total bytes. */
static int http_read_request(int fd, char *buf, int bufsz) {
    int total = 0;
    int content_length = -1;
    int header_end = -1;
    while (total < bufsz - 1) {
        int n = (int)read(fd, buf + total, bufsz - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';
        /* Find end of headers */
        if (header_end < 0) {
            char *hdr_end = strstr(buf, "\r\n\r\n");
            if (hdr_end) {
                header_end = (int)(hdr_end - buf) + 4;
                /* Parse Content-Length */
                char *cl = strcasestr(buf, "content-length:");
                if (cl) content_length = atoi(cl + 15);
                else content_length = 0;
            }
        }
        if (header_end >= 0 && total >= header_end + content_length) break;
    }
    return total;
}

/* Send full buffer over socket */
static void http_send(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = (int)write(fd, data + sent, len - sent);
        if (n <= 0) break;
        sent += n;
    }
}

/* Send HTTP response header */
static void http_send_header(int fd, int status, const char *content_type, int content_length) {
    char hdr[512];
    const char *status_text = status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request";
    int hlen;
    if (content_length >= 0) {
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n",
            status, status_text, content_type, content_length);
    } else {
        /* Streaming (SSE) — no content-length */
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: keep-alive\r\n\r\n",
            status, status_text, content_type);
    }
    http_send(fd, hdr, hlen);
}

/* Serve a static file (doe_ui.html, doe.html) */
static int http_serve_file(int fd, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = malloc(sz);
    if (!data) { fclose(f); return 0; }
    fread(data, 1, sz, f); fclose(f);
    http_send_header(fd, 200, "text/html; charset=utf-8", (int)sz);
    http_send(fd, data, (int)sz);
    free(data);
    return 1;
}

/* Extract JSON string value for a key from body. Simple parser. */
static int json_get_string(const char *json, const char *key, char *out, int outsz) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return 0;
    while (*p && (*p == ':' || *p == ' ' || *p == '\t')) p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < outsz - 1) {
        if (*p == '\\' && p[1]) { p++; /* skip escape */ }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* Extract last user message from messages array in chat/completions body */
static int json_get_last_user_message(const char *body, char *out, int outsz) {
    /* Find last "role":"user" ... "content":"..." */
    const char *last_user = NULL;
    const char *p = body;
    while ((p = strstr(p, "\"role\"")) != NULL) {
        const char *rv = strstr(p, "\"user\"");
        if (rv && rv - p < 30) last_user = p;
        p++;
    }
    if (!last_user) return 0;
    return json_get_string(last_user, "content", out, outsz);
}

static float json_get_float(const char *json, const char *key, float def) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return def;
    p = strchr(p + strlen(needle), ':');
    if (!p) return def;
    return (float)atof(p + 1);
}

/* Run inference and stream SSE tokens */
static void http_stream_inference(int fd, GGUFIndex *ps, const char *user_msg, float temperature, int max_tokens) {
    int max_seq = 512;
    InferState is = alloc_infer(ps, max_seq);

    /* Reset KV cache */
    int kd = ps->host_kv_heads * ps->host_head_dim;
    memset(is.key_cache, 0, (size_t)ps->host_n_layers * max_seq * kd * 4);
    memset(is.value_cache, 0, (size_t)ps->host_n_layers * max_seq * kd * 4);

    /* Wrap input in chat template */
    char wrapped[2048];
    switch (ps->chat_style) {
    case 1: snprintf(wrapped, sizeof(wrapped), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg); break;
    case 2: snprintf(wrapped, sizeof(wrapped), "[INST] %s [/INST]", user_msg); break;
    case 3: snprintf(wrapped, sizeof(wrapped), "<|user|>\n%s\n<|assistant|>\n", user_msg); break;
    case 4: snprintf(wrapped, sizeof(wrapped), "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg); break;
    case 5: snprintf(wrapped, sizeof(wrapped), "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_msg); break;
    case 6: snprintf(wrapped, sizeof(wrapped), "<|user_start|>%s<|user_end|><|assistant_start|>", user_msg); break;
    default: snprintf(wrapped, sizeof(wrapped), "%s", user_msg); break;
    }

    /* Tokenize */
    int input_tokens[512];
    int n_input = 0;
    if (ps->bos_id >= 0) input_tokens[n_input++] = ps->bos_id;
    n_input += tokenize_input(ps, wrapped, input_tokens + n_input, 512 - n_input);

    /* Prefill */
    int pos = 0;
    for (int i = 0; i < n_input - 1 && pos < max_seq - 1; i++, pos++) {
        doe_forward(ps, &is, input_tokens[i], pos);
        dario_ingest(input_tokens[i]); /* feed user tokens into Dario field */
    }

    int prev = input_tokens[n_input - 1];
    dario_ingest(prev); /* D-H2: last token still feeds the field; forwarded once in generation */

    /* Generate tokens, stream as SSE */
    for (int i = 0; i < max_tokens && pos < max_seq; i++, pos++) {
        float *lg = doe_forward(ps, &is, prev, pos);
        field_step(1.0f);
        apply_field_to_logits(lg, ps->host_vocab);

        float temp = temperature > 0.01f ? temperature : F.effective_temp;
        int next = sample(lg, ps->host_vocab, temp, 40);

        /* Stop on EOS */
        if (next == ps->eos_id) break;
        if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
            const char *ts = ps->vocab_tokens[next];
            if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                strcmp(ts, "<|eot_id|>") == 0) break;
        }

        /* Prophecy debt + Hebbian update */
        float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
        F.debt += pd;

        /* Dario field: ingest generated token */
        dario_ingest(next);

        float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
        if (g_train) for (int l = 0; l < ps->n_field_layers; l++) {
            FieldLayer *fl = &ps->field_layers[l];
            for (int e = 0; e < MAX_EXPERTS; e++) {
                if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                            ps->host_dim, ps->host_dim, ps->lora_rank,
                            is.x, is.xb, learn_signal);
                /* A14: poison-quarantine (NaN/Inf + finite |value|-explosion) — freeze */
                if (lora_poisoned(fl->experts[e].lora_A, fl->experts[e].lora_B))
                    fl->experts[e].alive = 0;
            }
        }

        /* Decode token to buffer */
        char tokbuf[256], escaped[512];
        token_decode_buf(ps, next, tokbuf, sizeof(tokbuf));
        json_escape(tokbuf, escaped, sizeof(escaped));

        /* Send SSE event */
        char sse[1024];
        int slen = snprintf(sse, sizeof(sse), "data: {\"token\":\"%s\"}\n\n", escaped);
        int wr = (int)write(fd, sse, slen);
        if (wr <= 0) break; /* client disconnected */

        prev = next;
    }

    /* Send done event */
    write(fd, "data: {\"done\":true}\n\n", 21); /* D-L2: full 21 bytes incl. both \n */
    free_infer(&is);
}

/* Main HTTP serve loop */
static void serve_loop(GGUFIndex *ps, const char *exe_dir) {
    signal(SIGPIPE, SIG_IGN); /* ignore broken pipes */

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("[serve] socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = g_serve_public ? INADDR_ANY : htonl(INADDR_LOOPBACK); /* D-M4: localhost by default */
    addr.sin_port = htons(g_serve_port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[serve] bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("[serve] listen"); close(server_fd); return;
    }

    /* Resolve HTML file paths relative to executable */
    char ui_path[512], vis_path[512];
    snprintf(ui_path, sizeof(ui_path), "%sdoe_ui.html", exe_dir);
    snprintf(vis_path, sizeof(vis_path), "%sdoe.html", exe_dir);

    printf("[serve] parliament listening on http://%s:%d\n", g_serve_public ? "0.0.0.0" : "127.0.0.1", g_serve_port);
    printf("[serve]   /         → chat UI\n");
    printf("[serve]   /visual   → parliament terminal\n");
    printf("[serve]   /health   → status\n");
    printf("[serve]   POST /chat/completions → inference stream\n\n");

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client < 0) continue;

        char req[8192];
        int reqlen = http_read_request(client, req, sizeof(req));
        if (reqlen <= 0) { close(client); continue; }

        /* Parse method and path */
        char method[8] = "", path[256] = "";
        sscanf(req, "%7s %255s", method, path);

        /* Handle CORS preflight */
        if (strcmp(method, "OPTIONS") == 0) {
            const char *cors = "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
            http_send(client, cors, (int)strlen(cors));
            close(client);
            continue;
        }

        if (strcmp(method, "GET") == 0) {
            if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
                if (!http_serve_file(client, ui_path)) {
                    const char *msg = "doe_ui.html not found";
                    http_send_header(client, 404, "text/plain", (int)strlen(msg));
                    http_send(client, msg, (int)strlen(msg));
                }
            } else if (strcmp(path, "/visual") == 0) {
                if (!http_serve_file(client, vis_path)) {
                    const char *msg = "doe.html not found";
                    http_send_header(client, 404, "text/plain", (int)strlen(msg));
                    http_send(client, msg, (int)strlen(msg));
                }
            } else if (strcmp(path, "/health") == 0) {
                char body[512];
                int blen = snprintf(body, sizeof(body),
                    "{\"status\":\"ok\",\"model\":\"%s\",\"arch\":\"%s\","
                    "\"params\":\"%dM\",\"vocab\":%d,\"layers\":%d,"
                    "\"experts\":%d,\"debt\":%.4f,\"health\":%.4f}",
                    ps->host_path, ps->host_arch,
                    (int)(ps->host_vocab * ps->host_dim * 2 / 1000000),
                    ps->host_vocab, ps->host_n_layers,
                    ps->n_field_layers > 0 ? ps->field_layers[0].n_alive : 0,
                    F.debt, F.field_health);
                http_send_header(client, 200, "application/json", blen);
                http_send(client, body, blen);
            } else {
                const char *msg = "not found";
                http_send_header(client, 404, "text/plain", (int)strlen(msg));
                http_send(client, msg, (int)strlen(msg));
            }
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/chat/completions") == 0 || strcmp(path, "/v1/chat/completions") == 0)) {
            /* Find body after \r\n\r\n */
            char *body = strstr(req, "\r\n\r\n");
            if (!body) { close(client); continue; }
            body += 4;

            char user_msg[2048] = "";
            json_get_last_user_message(body, user_msg, sizeof(user_msg));

            if (user_msg[0] == '\0') {
                const char *err = "{\"error\":\"no user message\"}";
                http_send_header(client, 400, "application/json", (int)strlen(err));
                http_send(client, err, (int)strlen(err));
            } else {
                float temp = json_get_float(body, "temperature", 0.0f);
                int max_tok = (int)json_get_float(body, "max_tokens", 256.0f);
                if (max_tok < 1) max_tok = 256;
                if (max_tok > 512) max_tok = 512;
                printf("[serve] inference: \"%.*s\" temp=%.2f max=%d\n",
                       (int)(strlen(user_msg) > 60 ? 60 : strlen(user_msg)), user_msg, temp, max_tok);
                http_send_header(client, 200, "text/event-stream", -1);
                http_stream_inference(client, ps, user_msg, temp, max_tok);
            }
        } else {
            const char *msg = "method not allowed";
            http_send_header(client, 400, "text/plain", (int)strlen(msg));
            http_send(client, msg, (int)strlen(msg));
        }

        close(client);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the field manifests.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    printf("\n  doe.c — Democracy of Experts\n");
    printf("  θ = ε + γ + αδ — the parliament awakens.\n\n");

    char gguf_path[256] = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) snprintf(gguf_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) { g_n_threads = atoi(argv[++i]); if (g_n_threads < 1) g_n_threads = 1; }
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i+1 < argc) { g_rep_penalty = atof(argv[++i]); if (g_rep_penalty <= 0.0f) g_rep_penalty = 1.0f; }
        else if (strcmp(argv[i], "--field-gain") == 0 && i+1 < argc) { g_field_gain = atof(argv[++i]); if (g_field_gain < 0.0f) g_field_gain = 0.0f; }
        else if (strcmp(argv[i], "--train") == 0 && i+1 < argc) { g_train = atoi(argv[++i]) ? 1 : 0; }
        else if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--serve") == 0 && i+1 < argc) { g_serve_port = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--serve-public") == 0) { g_serve_public = 1; }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("doe.c — DOE: inference architecture over any GGUF\n\n");
            printf("  --model PATH    GGUF to index (or auto-detect)\n");
            printf("  --serve PORT    start HTTP server for UI (doe_ui.html, doe.html)\n");
            printf("  --threads N     CPU threads for matvec (default: all cores)\n");
            printf("  --prophecy N    prediction horizon (default: 7)\n");
            printf("  --destiny F     destiny bias strength (default: 0.35)\n");
            printf("  --lora-alpha F  LoRA injection strength (default: 0.1)\n");
            printf("  --rep-penalty F repetition penalty over GENERATED tokens (default: 1.0 = off; 1.05 polishes demo tail)\n");
            printf("  --field-gain F  scale Dario field overlay on logits (default: 1.0; 0 = bare host)\n");
            printf("  --train N       notorch online-learning during inference (default: 0 = off; 1 = on)\n\n");
            printf("  BLAS: cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe\n");
            printf("  GPU:  cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe\n");
            return 0;
        }
    }

    /* ── Thread count for matvec ── */
    if (g_n_threads == 0) g_n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN); /* D-L1: respect --threads if the user set it */
    if (g_n_threads < 1) g_n_threads = 1;
    if (g_n_threads > 32) g_n_threads = 32;

    /* ── Field awakens ── */
    field_init();

    /* Parse field overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) F.prophecy = atoi(argv[++i]);
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) F.destiny = atof(argv[++i]);
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) { i++; /* D-L1: accepted for compat; rank comes from the GGUF, not a flag */ }
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) F.lora_alpha = atof(argv[++i]);
    }

    /* ── Environment scan ── */
    Environment env;
    env_scan(&env, __FILE__);

    /* ── PHASE 1: Search for DOE identity + gamma FIRST ── */
    char identity_path[256] = "";
    char gamma_path[256] = "";
    int weightless = 1;
    {
        static const char *wdirs[] = { "weights/", "doe_w/", "./", "../weights/", NULL };
        struct stat st;
        /* Search for doe_identity*.gguf (any variant: _micro, _mini, _q8, etc.) */
        for (int d = 0; wdirs[d] && identity_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            int64_t best_size = 0;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_identity", 12) != 0) continue;
                int nlen = (int)strlen(ent->d_name);
                if (nlen < 5 || strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
                char tmp[256];
                snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                if (stat(tmp, &st) == 0 && st.st_size > best_size) {
                    snprintf(identity_path, 256, "%s", tmp);
                    best_size = st.st_size;
                }
            }
            closedir(dir);
            if (identity_path[0] != '\0') {
                stat(identity_path, &st);
                printf("[identity] found: %s (%.1fMB)\n", identity_path, (float)st.st_size/(1024*1024));
                weightless = 0;
            }
        }
        /* Search for doe_gamma*.bin or doe_gamma*.npz */
        for (int d = 0; wdirs[d] && gamma_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_gamma", 9) == 0 ||
                    strncmp(ent->d_name, "gamma_", 6) == 0) {
                    char tmp[256];
                    snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                    if (stat(tmp, &st) == 0 && st.st_size > 0) {
                        snprintf(gamma_path, 256, "%s", tmp);
                        printf("[gamma] found: %s (%.1fMB)\n", tmp, (float)st.st_size/(1024*1024));
                        break;
                    }
                }
            }
            closedir(dir);
        }
        if (weightless)
            printf("[identity] no doe_identity.gguf — weightless mode.\n");
        if (gamma_path[0] == '\0')
            printf("[gamma] no doe_gamma.bin — parliament self-organizes.\n");
    }

    /* ── PHASE 2: Find host GGUF (external knowledge substrate) ── */
    if (gguf_path[0] == '\0') {
        if (identity_path[0] != '\0') {
            snprintf(gguf_path, 256, "%s", identity_path);
            printf("[host] using identity as host model.\n");
        } else {
            /* Also check all discovered GGUFs for doe.identity metadata */
            int identity_idx = -1, external_idx = -1;
            for (int i = 0; i < env.n_ggufs; i++) {
                if (strstr(env.ggufs[i].path, "mycelium/")) continue;
                if (strstr(env.ggufs[i].path, "doe_gamma")) continue;
                /* Quick sniff for doe.identity key in this GGUF */
                if (strstr(env.ggufs[i].path, "doe_identity")) {
                    identity_idx = i; continue;
                }
                if (external_idx < 0) external_idx = i;
            }
            /* Identity GGUF by name takes priority */
            if (identity_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[identity_idx].path);
                printf("[host] found identity GGUF: %s\n", gguf_path);
                weightless = 0;
            } else if (external_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[external_idx].path);
                printf("[host] indexing external: %s (%.1fMB)\n", gguf_path, (float)env.ggufs[external_idx].file_size/(1024*1024));
            } else {
                fprintf(stderr, "[error] no GGUF found. use --model PATH or place a .gguf nearby.\n");
                return 1;
            }
        }
    }

    /* ── Index GGUF ── */
    GGUFIndex idx;
    if (!index_load(&idx, gguf_path)) {
        fprintf(stderr, "[error] failed to index %s\n", gguf_path);
        return 1;
    }
    idx.weightless = weightless;

    /* If GGUF has doe.identity metadata — it's ours regardless of filename */
    if (idx.identity_tag[0] != '\0') {
        idx.weightless = 0;
        printf("[identity] verified via metadata: \"%s\"\n", idx.identity_tag);
    }

    /* ── Load gamma if found ── */
    if (gamma_path[0] != '\0') {
        FILE *gf = fopen(gamma_path, "rb");
        if (gf) {
            fseek(gf, 0, SEEK_END); long gsz = ftell(gf); fseek(gf, 0, SEEK_SET);
            idx.gamma_data = malloc(gsz);
            idx.gamma_size = (int)gsz;
            if (fread(idx.gamma_data, 1, gsz, gf) == (size_t)gsz)
                printf("[gamma] loaded %ld bytes — personality active.\n", gsz);
            else { free(idx.gamma_data); idx.gamma_data = NULL; idx.gamma_size = 0; }
            fclose(gf);
        }
    }

    /* ── Mycelium — check for existing LoRA spores ── */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    if (mycelium_load(&idx, idx.profile.fingerprint))
        printf("[mycelium] resumed adaptation for this index.\n");

#ifdef USE_METAL
    /* Parliament-on-GPU: pack the (now final, post-spore) frozen experts into a
     * registered arena so the election/inject kernels run inside the resident
     * command buffer at alpha!=0 without dropping x off the GPU. Metal-live +
     * parliament-active only; experts are static at --train 0. */
    if (nt_metal_available() && idx.lora_alpha != 0.0f) field_repack_arena(&idx);
#endif

    /* ── Chat or Serve ── */
    if (g_serve_port > 0) {
        /* Resolve directory of the executable for HTML files */
        char exe_dir[512] = "./";
        {
            /* Try to find doe_ui.html relative to argv[0] */
            char *slash = strrchr(argv[0], '/');
            if (slash) { int dlen = (int)(slash - argv[0]) + 1; if (dlen < 500) { memcpy(exe_dir, argv[0], dlen); exe_dir[dlen] = '\0'; } }
        }
        serve_loop(&idx, exe_dir);
    } else {
        chat(&idx);
    }

    /* ── Save spore on exit ── */
    mycelium_save(&idx, F.step, F.field_health);

    /* ── Cleanup ── */
    index_free(&idx);
    printf("[doe] the parliament adjourns. θ persists.\n");
    return 0;
}
