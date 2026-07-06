/*
 * tools/yent_forward.h — Janus v4 176M forward pass for AML.
 *
 * Adapted from dario/infer_v4.c. Same numerics — low-rank RRPRAM,
 * Echo, 3-way gate, RoPE, QK-norm, smear, residual lambdas, backout —
 * but weights load from a notorch GGUF (Q8_0 / Q4_K) instead of raw
 * fp32, and there is no main(): yent.aml owns the entry point.
 *
 * Caller responsibility:
 *   1. yent_load_gguf(&w, "weights/yent_v4_sft_q8_0.gguf")
 *   2. kv_init(T)
 *   3. prefill_batch / forward_token loop
 *   4. yent_free(&w)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "notorch.h"
#include "ariannamethod.h"   /* B2-B.2: am_apply_delta / am_get_state for the δ voice */
#include "gguf.h"

static int V, E, H, D, B, M, T, R;

/* B2-B.2 — low-rank δ voice (Janus side). A=[E,rank], B=[rank,E]; lora_alpha=0 →
 * am_apply_delta no-op (bit-identical). Allocated/loaded + learned in arianna.aml
 * (init am_cooc_load / autumn consolidate); applied here before each head matvec. */
static float *g_delta_A = NULL, *g_delta_B = NULL;
static int    g_delta_rank = AM_DELTA_RANK;

/* ── BLAS-accelerated matmul via notorch ──
 *
 * Two paths:
 *   mm_t(C, A, B, m, k, n) — cblas_sgemm. Used when m > 1 (prefill_batch
 *     processes the whole prompt as a [n_tokens, E] batch).
 *   matvec_t(out, W, x, n, k) — cblas_sgemv. Used in the per-token hot
 *     loop where each Linear is effectively (1×E) @ (E×N). sgemv is
 *     better tuned than sgemm with m=1 on Apple Accelerate.
 *
 * Old infer_v4.c used mm_t for both paths — that predates notorch's
 * nt_blas_matvec API (added in commit 59327ba). yent_forward.h picks
 * the right path per call. */
static void mm_t(float *C, const float *A, const float *BT, int m, int k, int n) {
    /* C[m,n] = A[m,k] @ BT[n,k]^T — BT is stored transposed */
    nt_blas_mmT(C, A, BT, m, k, n);
}

/* out[n] = W[n,k] @ x[k]  (sgemv path). W is stored row-major as [n, k],
 * matching how nn.Linear weights ship in PyTorch state_dict (out, in). */
static void matvec_t(float *out, const float *W, const float *x, int n, int k) {
    nt_blas_matvec(out, W, x, n, k);
}

/* notorch ops — used directly for single-vector operations */
static void rmsnorm(float *o, const float *x, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv;
}

static void softmax_f(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }

/* A .!? at obuf[pos] is a real sentence end unless it follows a single isolated
 * letter ("the inner v." — v is a clipped "voice"). A one-letter word with a
 * space (or start) before it is an abbreviation/clip, not a thought boundary. */
static int sent_end_ok(const char *obuf, int pos) {
    int j = pos - 1, run = 0;
    while (j >= 0 && ((obuf[j] >= 'a' && obuf[j] <= 'z') ||
                      (obuf[j] >= 'A' && obuf[j] <= 'Z'))) { run++; j--; }
    if (run == 1 && (j < 0 || obuf[j] == ' ')) return 0;
    return 1;
}

/* ── DIRECTION injection (Dario A/F field-pressure, NOT logit-boost/token-paste) ──
 * Same mechanism as resonance_forward.h, ported for Janus (embeddings = w->wte,
 * matvec_t over [V,E]). Injected words become a destiny-EMA compass (A) +
 * prophecy targets (F) that tilt the WHOLE distribution by embedding cosine,
 * decaying within the turn. Tuned point: alpha 10, within-decay 0.15. INVARIANT:
 * injected tokens are never re-inserted into the context (anti-fraud). */
#define DIR_PROPH_MAX 16
static float  g_destiny[1024];
static float  g_dest_mag = 0.0f;
static float *g_Acache = NULL, *g_Fcache = NULL, *g_rownorm = NULL;
static int    g_Adirty = 0;
static int    g_proph_tok[DIR_PROPH_MAX], g_proph_age[DIR_PROPH_MAX], g_proph_n = 0;
static float  g_proph_str[DIR_PROPH_MAX];

static float dir_dot(const float *a, const float *b, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static float dir_norm(const float *v, int n) { return sqrtf(dir_dot(v, v, n)); }

/* Fail-loud zeroed alloc: OOM in the forward path is unrecoverable (the caller
 * writes into the buffer immediately), so exit rather than NULL-deref. Uses
 * malloc+memset (not the c-alloc name) so a file-wide sweep to this helper
 * cannot rewrite its own definition. */
static void *yent_xcalloc(size_t n, size_t sz) {
    size_t total = n * sz;
    if (sz != 0 && total / sz != n) { fprintf(stderr, "yent: alloc overflow (%zu x %zu)\n", n, sz); exit(1); }
    void *p = malloc(total ? total : 1);
    if (!p) { fprintf(stderr, "yent: OOM alloc %zu x %zu\n", n, sz); exit(1); }
    memset(p, 0, total);
    return p;
}

static void dir_init_rownorms(const float *emb) {
    if (!g_rownorm) g_rownorm = yent_xcalloc(V, sizeof(float));
    if (!g_Acache)  g_Acache  = yent_xcalloc(V, sizeof(float));
    if (!g_Fcache)  g_Fcache  = yent_xcalloc(V, sizeof(float));
    for (int i = 0; i < V; i++) g_rownorm[i] = dir_norm(emb + (size_t)i * E, E);
    memset(g_destiny, 0, sizeof(g_destiny));
    g_dest_mag = 0.0f; g_proph_n = 0; g_Adirty = 0;
}

static void dir_update(const float *emb, const int *toks, int n) {
    for (int t = 0; t < n; t++) {
        int id = toks[t];
        if (id < 0 || id >= V) continue;
        const float *e = emb + (size_t)id * E;
        for (int d = 0; d < E; d++) g_destiny[d] = 0.1f * e[d] + 0.9f * g_destiny[d];
        if (g_proph_n < DIR_PROPH_MAX) {
            g_proph_tok[g_proph_n] = id; g_proph_age[g_proph_n] = 0; g_proph_str[g_proph_n] = 1.0f; g_proph_n++;
        } else {
            memmove(g_proph_tok, g_proph_tok + 1, (DIR_PROPH_MAX - 1) * sizeof(int));
            memmove(g_proph_age, g_proph_age + 1, (DIR_PROPH_MAX - 1) * sizeof(int));
            memmove(g_proph_str, g_proph_str + 1, (DIR_PROPH_MAX - 1) * sizeof(float));
            g_proph_tok[DIR_PROPH_MAX-1] = id; g_proph_age[DIR_PROPH_MAX-1] = 0; g_proph_str[DIR_PROPH_MAX-1] = 1.0f;
        }
    }
    g_dest_mag = dir_norm(g_destiny, E) * 0.95f;
    if (g_dest_mag > 1.5f) g_dest_mag = 1.5f;
    g_Adirty = 1;
}

static void dir_recompute(const float *emb) {
    if (!g_Adirty || !g_Acache) return;
    float dn = dir_norm(g_destiny, E) + 1e-12f;
    matvec_t(g_Acache, emb, g_destiny, V, E);
    float amax = 1e-12f;
    for (int i = 0; i < V; i++) {
        g_Acache[i] = (g_Acache[i] / (g_rownorm[i] * dn + 1e-12f)) * g_dest_mag;
        float a = fabsf(g_Acache[i]); if (a > amax) amax = a;
    }
    for (int i = 0; i < V; i++) g_Acache[i] /= amax;
    memset(g_Fcache, 0, V * sizeof(float));
    /* Karpathy OPT: the F-term inner products are emb @ te — one cblas gemv per
     * prophecy target instead of a V*E hand loop; relu/norm stay after the dot
     * (algorithm-faithful, the same BLAS path the A-term above already uses). fnum
     * NULL → skip the F tilt (fail-safe, no field prophecy this turn). */
    float *fnum = (float*)malloc((size_t)V * sizeof(float));
    for (int p = 0; fnum && p < g_proph_n; p++) {
        const float *te = emb + (size_t)g_proph_tok[p] * E;
        float tn = g_rownorm[g_proph_tok[p]] + 1e-12f;
        float w = g_proph_str[p] * logf(1.0f + (float)g_proph_age[p]);
        if (w <= 0.0f) continue;
        matvec_t(fnum, emb, te, V, E);
        for (int i = 0; i < V; i++) {
            float c = fnum[i] / (g_rownorm[i] * tn + 1e-12f);
            if (c > 0.0f) g_Fcache[i] += w * c;
        }
    }
    free(fnum);
    float fmax = 1e-12f;
    for (int i = 0; i < V; i++) if (g_Fcache[i] > fmax) fmax = g_Fcache[i];
    for (int i = 0; i < V; i++) g_Fcache[i] /= fmax;
    g_Adirty = 0;
}

static void dir_apply(float *logits, float alpha, float beta) {
    if (g_dest_mag <= 0.0f && g_proph_n == 0) return;
    if (alpha > 0.0f && g_Acache) for (int i = 0; i < V; i++) logits[i] += alpha * g_Acache[i];
    if (beta  > 0.0f && g_Fcache) for (int i = 0; i < V; i++) logits[i] += beta  * g_Fcache[i];
}

static void dir_age(int emitted) {
    int w = 0;
    for (int p = 0; p < g_proph_n; p++) {
        g_proph_age[p]++;
        int fulfilled = (g_proph_tok[p] == emitted);
        if (g_proph_age[p] < 50 && !fulfilled) {
            g_proph_tok[w] = g_proph_tok[p]; g_proph_age[w] = g_proph_age[p]; g_proph_str[w] = g_proph_str[p]; w++;
        } else g_Adirty = 1;
    }
    g_proph_n = w;
}

static void rope_pos(float *q, float *k, int pos, int dim) {
    /* Split-half convention (nanochat/Janus v4): pairs (i, i+D/2), base=100000 */
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(100000.0f, (float)(2*i) / (float)dim);
        float val = pos * freq;
        float cv = cosf(val), sv = sinf(val);
        float q0 = q[i], q1 = q[i + half];
        q[i]        = q0 * cv + q1 * sv;
        q[i + half] = q0 * (-sv) + q1 * cv;
        float k0 = k[i], k1 = k[i + half];
        k[i]        = k0 * cv + k1 * sv;
        k[i + half] = k0 * (-sv) + k1 * cv;
    }
}

/* QK-norm: RMSNorm + scale (from nanochat) */
static void qk_norm(float *q, float *k, int dim) {
    rmsnorm(q, q, dim);
    rmsnorm(k, k, dim);
    for (int i = 0; i < dim; i++) { q[i] *= 1.2f; k[i] *= 1.2f; }
}

/* Weight layout: header(8i) + resid_l(20) + x0_l(20) + smear_l(1) + backout_l(1) + smear_g(24)
 * + wte[V,E] + B * (cq ck cv wr_a wr_b wvr wj gate cproj wg wu wd) + head[V,E] */
#define MBL 24
typedef struct {
    float *resid_l, *x0_l, *smear_l, *backout_l, *smear_g;
    float *wte;                                          /* f32: token-embedding lookup */
    struct {
        float *wr_a, *wr_b, *gate;                       /* f32: read element-wise, not matvec */
        const uint8_t *cq, *ck, *cv, *wvr, *wj, *cproj;  /* packed (wdtype) */
        const uint8_t *wg, *wu, *wd;                     /* packed (wdtype) */
    } b[MBL];
    const uint8_t *head;                                  /* packed (wdtype) */
    int wdtype;            /* GGUF_TYPE_F16 packed (default) | GGUF_TYPE_F32 dense (YENT_DENSE=1) */
    gguf_file *gf;         /* kept open when packed: the const uint8_t* point into gf->data */
} Weights;

/* Holder for dequantized buffers — freed in yent_free. */
typedef struct { float *ptr; } _OwnedTensor;
static _OwnedTensor _owned[2 + MBL * 12 + 2 + 4];   /* loose upper bound */
static int _owned_n = 0;

/* Packed batched matmul: C[m, nout], row p = Wq[nout,k] @ A[p,k], via per-row
 * nt_qmatvec (which dispatches on wdtype: F16 packed or F32 dense). Replaces the
 * dense nt_blas_mmT in prefill so the big weights stay packed in RAM. */
static void qmm(float *C, const float *A, const uint8_t *Wq, int wdtype, int m, int k, int nout) {
    for (int p = 0; p < m; p++)
        nt_qmatvec(C + (size_t)p * nout, Wq, wdtype, A + (size_t)p * k, nout, k);
}

/* Big weight matrix loader. Default: a packed F16 pointer INTO gf->data (no
 * dequant, no copy — caller keeps gf open). YENT_DENSE=1: dequant to an owned
 * f32 buffer (the bit-equivalence reference). Sets *wdtype accordingly. */
static const uint8_t *_load_big(gguf_file *gf, const char *name, size_t expect, int dense, int *wdtype) {
    int idx = gguf_find_tensor(gf, name);
    if (idx < 0) { fprintf(stderr, "yent: tensor '%s' not found in GGUF\n", name); return NULL; }
    if (gf->tensors[idx].n_elements != expect) {   /* J-4: verify GGUF tensor size vs the cfg dimension the forward indexes by */
        fprintf(stderr, "yent: tensor '%s' has %llu elements, cfg expects %zu — GGUF/arch mismatch\n",
                name, (unsigned long long)gf->tensors[idx].n_elements, expect);
        return NULL;
    }
    if (dense) {
        float *buf = gguf_dequant(gf, idx);
        if (!buf) { fprintf(stderr, "yent: dequant failed for '%s'\n", name); return NULL; }
        _owned[_owned_n++].ptr = buf;
        *wdtype = GGUF_TYPE_F32;
        return (const uint8_t *)buf;
    }
    const gguf_tensor_info *t = &gf->tensors[idx];
    if (t->dtype != GGUF_TYPE_F16) {
        fprintf(stderr, "yent: '%s' dtype=%u not F16 (packed path needs F16; use YENT_DENSE=1)\n",
                name, t->dtype);
        return NULL;
    }
    /* bounds the packed F16 span (2 bytes/elem) inside gf->data (cf. M-3) */
    if (t->offset >= gf->data_size || t->n_elements * 2 > gf->data_size - t->offset) {
        fprintf(stderr, "yent: '%s' packed F16 out of bounds (off %llu + %llu*2, data_size %llu)\n",
                name, (unsigned long long)t->offset, (unsigned long long)t->n_elements,
                (unsigned long long)gf->data_size);
        return NULL;
    }
    *wdtype = GGUF_TYPE_F16;
    return gf->data + t->offset;
}

static float *_load_named(gguf_file *gf, const char *name, size_t expect) {
    int idx = gguf_find_tensor(gf, name);
    if (idx < 0) {
        fprintf(stderr, "yent: tensor '%s' not found in GGUF\n", name);
        return NULL;
    }
    if (gf->tensors[idx].n_elements != expect) {   /* J-4: verify GGUF tensor size vs the cfg dimension the forward indexes by */
        fprintf(stderr, "yent: tensor '%s' has %llu elements, cfg expects %zu — GGUF/arch mismatch\n",
                name, (unsigned long long)gf->tensors[idx].n_elements, expect);
        return NULL;
    }
    float *buf = gguf_dequant(gf, idx);
    if (!buf) {
        fprintf(stderr, "yent: dequant failed for '%s'\n", name);
        return NULL;
    }
    _owned[_owned_n++].ptr = buf;
    return buf;
}

static int yent_load_gguf(Weights *w, const char *path) {
    gguf_file *gf = gguf_open(path);
    if (!gf) {
        fprintf(stderr, "yent: gguf_open('%s') failed\n", path);
        return 1;
    }
    int dense = (getenv("YENT_DENSE") != NULL);   /* bit-equivalence reference path (dense f32) */
    w->wdtype = dense ? GGUF_TYPE_F32 : GGUF_TYPE_F16;
    w->gf = NULL;

#define LOAD(field, name, n_elem)                            \
    do {                                                     \
        w->field = _load_named(gf, name, (size_t)(n_elem));  \
        if (!w->field) { gguf_close(gf); return 1; }         \
    } while (0)

    LOAD(resid_l,    "resid_lambdas",          B);
    LOAD(x0_l,       "x0_lambdas",             B);
    LOAD(smear_l,    "smear_lambda",           1);
    LOAD(backout_l,  "backout_lambda",         1);
    LOAD(smear_g,    "smear_gate.weight",      24);
    LOAD(wte,        "transformer.wte.weight", (size_t)V * E);

    char nm[128];
    for (int i = 0; i < B; i++) {
#define LOAD_LAYER(field, suffix, n_elem)                                   \
    do {                                                                    \
        snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, i);            \
        w->b[i].field = _load_named(gf, nm, (size_t)(n_elem));              \
        if (!w->b[i].field) { gguf_close(gf); return 1; }                   \
    } while (0)
#define LOAD_LAYER_BIG(field, suffix, n_elem)                               \
    do {                                                                    \
        snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, i);            \
        w->b[i].field = _load_big(gf, nm, (size_t)(n_elem), dense, &w->wdtype); \
        if (!w->b[i].field) { gguf_close(gf); return 1; }                   \
    } while (0)
        LOAD_LAYER(wr_a,  "attn.wr_a",          (size_t)H * E * R);   /* f32: element-wise */
        LOAD_LAYER(wr_b,  "attn.wr_b",          (size_t)H * R * T);   /* f32: element-wise */
        LOAD_LAYER(gate,  "attn.gate",          (size_t)H * 3);       /* f32: lookup */
        LOAD_LAYER_BIG(cq,    "attn.c_q.weight",    (size_t)E * E);
        LOAD_LAYER_BIG(ck,    "attn.c_k.weight",    (size_t)E * E);
        LOAD_LAYER_BIG(cv,    "attn.c_v.weight",    (size_t)E * E);
        LOAD_LAYER_BIG(wvr,   "attn.wvr.weight",    (size_t)E * E);
        LOAD_LAYER_BIG(wj,    "attn.wj.weight",     (size_t)E * E);
        LOAD_LAYER_BIG(cproj, "attn.c_proj.weight", (size_t)E * E);
        LOAD_LAYER_BIG(wg,    "mlp.w_gate.weight",  (size_t)E * M);
        LOAD_LAYER_BIG(wu,    "mlp.w_up.weight",    (size_t)E * M);
        LOAD_LAYER_BIG(wd,    "mlp.w_down.weight",  (size_t)M * E);
#undef LOAD_LAYER
#undef LOAD_LAYER_BIG
    }
#undef LOAD
    if (!(w->head = _load_big(gf, "lm_head.weight", (size_t)V * E, dense, &w->wdtype))) { gguf_close(gf); return 1; }

    if (dense) gguf_close(gf);   /* dense weights are owned copies — gf no longer needed */
    else       w->gf = gf;       /* packed: keep gf open, the const uint8_t* reference gf->data */
    dir_init_rownorms(w->wte);   /* direction-injection: ‖emb‖ once (Janus uses wte) */
    return 0;
}

static void yent_free(Weights *w) {
    for (int i = 0; i < _owned_n; i++) {
        free(_owned[i].ptr);
        _owned[i].ptr = NULL;
    }
    _owned_n = 0;
    if (w && w->gf) { gguf_close(w->gf); w->gf = NULL; }   /* packed pointers referenced gf->data */
}

/* Initialise V/E/H/D/B/M/T/R from GGUF metadata before yent_load_gguf. */
static int yent_read_cfg(const char *path) {
    gguf_file *gf = gguf_open(path);
    if (!gf) {
        fprintf(stderr, "yent: gguf_open('%s') failed\n", path);
        return 1;
    }
#define READ_U32(key, target)                                     \
    do {                                                          \
        const gguf_kv *kv = gguf_get_kv(gf, key);                 \
        if (!kv) {                                                \
            fprintf(stderr, "yent: missing kv '%s'\n", key);      \
            gguf_close(gf); return 1;                             \
        }                                                         \
        target = (int)kv->val.u32;                                \
    } while (0)
    READ_U32("janus.vocab_size",           V);
    READ_U32("janus.embedding_length",     E);
    READ_U32("janus.attention.head_count", H);
    READ_U32("janus.attention.head_dim",   D);
    READ_U32("janus.block_count",          B);
    READ_U32("janus.feed_forward_length",  M);
    READ_U32("janus.context_length",       T);
    READ_U32("janus.rrpram.rank",          R);
#undef READ_U32
    /* M-1/M-2: validate arch against the fixed forward stack buffers before any
     * allocation/forward. gs[16][3] => H<=16; x/xn/qa/.../cat/ao/mo[1024] => E<=1024;
     * mid/c_out/r_out[128] => D<=128 and R<=128; r_scores/r_attn/attn[2048] => T<=2048;
     * mg/mu[2048] => M<=2048; Weights b[MBL]. H*D must equal E (KV row stride). A wrong
     * or crafted GGUF would otherwise smash these arrays on load/forward. */
    if (V <= 0 || V > (1<<20) || E <= 0 || E > 1024 || H <= 0 || H > 16 || D <= 0 || D > 128 ||
        B <= 0 || B > MBL || M <= 0 || M > 2048 || T <= 0 || T > 2048 ||
        R <= 0 || R > 128 || H * D != E) {
        fprintf(stderr, "yent: GGUF arch out of bounds (H*D must==E): "
                "V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n", V, E, H, D, B, M, T, R);
        gguf_close(gf); return 1;
    }
    gguf_close(gf);
    return 0;
}

/* KV cache for autoregressive generation */
static float *kv_k; /* [B, seqlen, E] */
static float *kv_v; /* [B, seqlen, E] */
static float *kv_vr; /* [B, seqlen, E] */
static float *kv_rrpram_mid; /* [B, H, R] — accumulated RRPRAM intermediate */
static int kv_len;

static void kv_init(int max_seq) {
    kv_k = yent_xcalloc((size_t)B * max_seq * E, sizeof(float));
    kv_v = yent_xcalloc((size_t)B * max_seq * E, sizeof(float));
    kv_vr = yent_xcalloc((size_t)B * max_seq * E, sizeof(float));
    kv_rrpram_mid = yent_xcalloc((size_t)B * H * R, sizeof(float));
    kv_len = 0;
}

/* Parallel prefill: process all prompt tokens through each block together.
 * Matches Python's parallel attention exactly. After prefill, KV cache is populated
 * and generation continues with forward_token (autoregressive). */
static void prefill_batch(Weights *w, int *toks, int n, float *logits, float *hidden) {
    /* Clamp to context window: kv_k/kv_v are sized [B, T, E]; writing K/V at a
     * position p >= T (off = (bl*T+p)*E) overflows the heap allocation. A prompt
     * encoding to more than T tokens is clamped to the first T here. */
    if (n > T) {
        fprintf(stderr, "[yent] WARNING: prefill clamped %d->%d tokens (context window T)\n", n, T);
        n = T;
    }
    float *xs = yent_xcalloc((size_t)n * E, sizeof(float));  /* [n, E] hidden states */
    float *x0s = yent_xcalloc((size_t)n * E, sizeof(float)); /* [n, E] original embeddings */
    float sc = 1.0f / sqrtf((float)D);

    /* Embed all tokens + NORM (nanochat: x = norm(wte(idx))) */
    for (int p = 0; p < n; p++) {
        for (int e = 0; e < E; e++)
            xs[p*E+e] = w->wte[toks[p]*E+e];
        rmsnorm(xs + p*E, xs + p*E, E);  /* norm BEFORE everything */
    }

    /* Smear: mix previous token embedding into current.
     * gate = smear_lambda * sigmoid(smear_gate @ x[:, :24])
     * x[t] += gate[t] * x[t-1]  (for t >= 1) */
    float smear_l = *w->smear_l;
    if (smear_l > 1e-6f) {
        for (int p = 1; p < n; p++) {
            float dot = 0;
            for (int d = 0; d < 24; d++) dot += w->smear_g[d] * xs[p*E+d];
            float gate = smear_l / (1.0f + expf(-dot));
            for (int e = 0; e < E; e++) xs[p*E+e] += gate * xs[(p-1)*E+e];
        }
    }

    /* x0 = x AFTER norm + smear (nanochat line 602: x0 = x) */
    memcpy(x0s, xs, (size_t)n * E * sizeof(float));

    int backout_layer = B / 2;
    float *x_backout = yent_xcalloc((size_t)n * E, sizeof(float));

    for (int bl = 0; bl < B; bl++) {
        /* Residual scaling: x = resid_lambda * x + x0_lambda * x0 */
        float rl = w->resid_l[bl], x0l = w->x0_l[bl];
        for (int i = 0; i < n * E; i++)
            xs[i] = rl * xs[i] + x0l * x0s[i];

        /* Norm all positions */
        float *rns = yent_xcalloc((size_t)n * E, sizeof(float));
        for (int p = 0; p < n; p++)
            rmsnorm(rns + p*E, xs + p*E, E);

        /* QKV projections for all positions: [n, E] @ [E, E]^T = [n, E] */
        float *qa = yent_xcalloc((size_t)n*E, 4), *ka = yent_xcalloc((size_t)n*E, 4);
        float *va = yent_xcalloc((size_t)n*E, 4), *vra = yent_xcalloc((size_t)n*E, 4);
        qmm(qa, rns, w->b[bl].cq, w->wdtype, n, E, E);
        qmm(ka, rns, w->b[bl].ck, w->wdtype, n, E, E);
        qmm(va, rns, w->b[bl].cv, w->wdtype, n, E, E);
        qmm(vra, rns, w->b[bl].wvr, w->wdtype, n, E, E);

        /* RoPE + QK-norm per position per head */
        for (int p = 0; p < n; p++)
            for (int h = 0; h < H; h++) {
                rope_pos(qa + p*E + h*D, ka + p*E + h*D, p, D);
                qk_norm(qa + p*E + h*D, ka + p*E + h*D, D);
            }

        /* Store K, V, Vr in cache for later autoregressive generation */
        for (int p = 0; p < n; p++) {
            size_t off = ((size_t)bl * T + p) * E;
            memcpy(kv_k + off, ka + p*E, E * sizeof(float));
            memcpy(kv_v + off, va + p*E, E * sizeof(float));
            memcpy(kv_vr + off, vra + p*E, E * sizeof(float));
        }

        /* Echo: [n, E] @ [E, E]^T */
        float *echo = yent_xcalloc((size_t)n*E, 4);
        qmm(echo, rns, w->b[bl].wj, w->wdtype, n, E, E);

        /* Gate softmax (same for all positions) */
        float gs[16][3];
        for (int h = 0; h < H; h++) {
            gs[h][0]=w->b[bl].gate[h*3]; gs[h][1]=w->b[bl].gate[h*3+1]; gs[h][2]=w->b[bl].gate[h*3+2];
            softmax_f(gs[h], 3);
        }

        /* Per-head attention (parallel over all positions) */
        float *cat = yent_xcalloc((size_t)n*E, 4);
        for (int h = 0; h < H; h++) {
            /* Content attention: [n, n] scores, causal mask */
            float *scores = yent_xcalloc((size_t)n*n, 4);
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++) {
                    float s = 0;
                    float *qi = qa + i*E + h*D;
                    float *kj_p = ka + j*E + h*D;
                    for (int d = 0; d < D; d++) s += qi[d] * kj_p[d];
                    scores[i*n+j] = s * sc;
                }
            /* Softmax per row (causal) */
            for (int i = 0; i < n; i++) {
                for (int j = i+1; j < n; j++) scores[i*n+j] = -1e30f;
                softmax_f(scores + i*n, n);
            }
            /* Weighted sum of V */
            for (int i = 0; i < n; i++) {
                float c_out[128] = {0};
                for (int j = 0; j < n; j++)
                    for (int d = 0; d < D; d++)
                        c_out[d] += scores[i*n+j] * va[j*E + h*D + d];

                /* RRPRAM (broadcast pattern) */
                float *wr_a_h = w->b[bl].wr_a + h*E*R;
                float *wr_b_h = w->b[bl].wr_b + h*R*T;
                /* intermediate = sum_t sum_e x[t,e] * wr_a[h,e,r] for t=0..n-1 */
                float mid[128] = {0};
                for (int t = 0; t < n; t++)
                    for (int r = 0; r < R; r++)
                        for (int e = 0; e < E; e++)
                            mid[r] += rns[t*E+e] * wr_a_h[e*R+r];
                /* H-1: seed kv_rrpram_mid from the prompt sum so autoregressive
                 * continuation (forward_token does mid_cache[r] += ... per new pos)
                 * starts from the prompt's RRPRAM state instead of zero; the `=`
                 * also resets it per prefill for the daemon path. mid is invariant
                 * in i, so seed once. Port of dario/infer_v4.c:233-238. */
                if (i == 0) {
                    float *mid_cache = kv_rrpram_mid + ((size_t)bl * H + h) * R;
                    for (int r = 0; r < R; r++) mid_cache[r] = mid[r];
                }
                /* scores = mid @ wr_b * sc, broadcast */
                float r_scores[2048];
                for (int j = 0; j < n; j++) {
                    float s = 0;
                    for (int r = 0; r < R; r++) s += mid[r] * wr_b_h[r*T+j];
                    r_scores[j] = s * sc;
                }
                /* RRPRAM attention: attn[i,j] = softmax(r_scores[j] for j<=i) */
                float r_attn[2048];
                for (int j = 0; j <= i; j++) r_attn[j] = r_scores[j];
                for (int j = i+1; j < n; j++) r_attn[j] = -1e30f;
                softmax_f(r_attn, n);
                float r_out[128] = {0};
                for (int j = 0; j < n; j++)
                    for (int d = 0; d < D; d++)
                        r_out[d] += r_attn[j] * vra[j*E + h*D + d];

                /* Echo (simplified - gate is ~0 so minimal impact) */
                float *e_h = echo + i*E + h*D;

                /* Blend */
                for (int d = 0; d < D; d++)
                    cat[i*E + h*D + d] = gs[h][0]*c_out[d] + gs[h][1]*r_out[d] + gs[h][2]*e_h[d];
            }
            free(scores);
        }

        /* Output projection: [n, E] @ [E, E]^T + residual */
        float *ao = yent_xcalloc((size_t)n*E, 4);
        qmm(ao, cat, w->b[bl].cproj, w->wdtype, n, E, E);
        for (int i = 0; i < n*E; i++) xs[i] += ao[i];

        if (bl == backout_layer) memcpy(x_backout, xs, (size_t)n*E*4);

        /* MLP: norm → gate/up → silu*up → down + residual */
        float *rn2s = yent_xcalloc((size_t)n*E, 4);
        for (int p = 0; p < n; p++) rmsnorm(rn2s + p*E, xs + p*E, E);
        float *mg = yent_xcalloc((size_t)n*M, 4), *mu = yent_xcalloc((size_t)n*M, 4), *mo = yent_xcalloc((size_t)n*E, 4);
        qmm(mg, rn2s, w->b[bl].wg, w->wdtype, n, E, M);
        qmm(mu, rn2s, w->b[bl].wu, w->wdtype, n, E, M);
        for (int i = 0; i < n*M; i++) mg[i] = siluf(mg[i]) * mu[i];
        qmm(mo, mg, w->b[bl].wd, w->wdtype, n, M, E);
        for (int i = 0; i < n*E; i++) xs[i] += mo[i];

        free(rns); free(qa); free(ka); free(va); free(vra);
        free(echo); free(cat); free(ao); free(rn2s); free(mg); free(mu); free(mo);
    }

    /* Backout */
    float bl_val = *w->backout_l;
    for (int i = 0; i < n*E; i++) xs[i] -= bl_val * x_backout[i];

    /* Final norm + head for last position */
    float rn_final[1024];
    rmsnorm(rn_final, xs + (n-1)*E, E);
    if (hidden) memcpy(hidden, rn_final, E * sizeof(float));   /* field carry = pre-δ */
    am_apply_delta(rn_final, g_delta_A, g_delta_B, rn_final, E, E, g_delta_rank,
                   am_lora_alpha_effective());   /* B2-B.4: lora_alpha, or lora_alpha*resonance when LORA_DYNAMIC */
    nt_qmatvec(logits, w->head, w->wdtype, rn_final, V, E);
    for (int i = 0; i < V; i++) logits[i] = 15.0f * tanhf(logits[i] / 15.0f);

    if (getenv("YENT_DUMP")) {   /* B2-B.3/.4 probe: first-token logits vs the δ voice strength */
        int am = 0; float mv = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > mv) { mv = logits[i]; am = i; }
        fprintf(stderr, "[yent-dump] wdtype=%d alpha_max=%.3f dyn=%d resonance=%.3f alpha_eff=%.4f argmax=%d max=%.5f | l0=%.5f l100=%.5f l1000=%.5f\n",
                w->wdtype, am_get_state()->lora_alpha, am_get_state()->lora_dynamic, am_get_state()->resonance,
                am_lora_alpha_effective(), am, mv, logits[0], logits[100], logits[1000]);
    }

    free(xs); free(x0s); free(x_backout);
}

/* Forward one token at position pos, using KV cache */
static void forward_token(Weights *w, int tok, int pos, float *logits, float *hidden) {
    float x[1024]; /* E <= 1024 */
    float rn[1024], rn2[1024];
    float sc = 1.0f / sqrtf((float)D);

    /* embed + norm (nanochat: x = norm(wte(idx))) */
    for (int e = 0; e < E; e++) x[e] = w->wte[tok * E + e];
    rmsnorm(x, x, E);

    /* smear: mix previous token (from KV cache position pos-1 block 0 input) */
    /* For autoregressive, smear uses prev_embedding stored externally */
    /* TODO: full smear for autoregressive (minor effect, smear_lambda=0.32) */

    /* x0 = embedding AFTER norm+smear (nanochat line 602: x0 = x) */
    float x0[1024];
    memcpy(x0, x, E * sizeof(float));

    int backout_layer = B / 2;
    static float x_backout[1024]; /* cached mid-layer residual */

    for (int bl = 0; bl < B; bl++) {
        /* nanochat residual scaling: x = resid_lambda * x + x0_lambda * x0 (BEFORE block) */
        float rl = w->resid_l[bl];
        float x0l = w->x0_l[bl];
        for (int e = 0; e < E; e++)
            x[e] = rl * x[e] + x0l * x0[e];

        /* Block: attn(norm(x)) + x, then mlp(norm(x)) + x */
        rmsnorm(rn, x, E);

        /* QKV projections */
        float qa[1024], ka[1024], va[1024], vra[1024];
        nt_qmatvec(qa, w->b[bl].cq, w->wdtype, rn, E, E);
        nt_qmatvec(ka, w->b[bl].ck, w->wdtype, rn, E, E);
        nt_qmatvec(va, w->b[bl].cv, w->wdtype, rn, E, E);
        nt_qmatvec(vra, w->b[bl].wvr, w->wdtype, rn, E, E);

        /* RoPE + QK-norm per head */
        for (int h = 0; h < H; h++) {
            rope_pos(qa + h*D, ka + h*D, pos, D);
            qk_norm(qa + h*D, ka + h*D, D);
        }

        /* store K, V, Vr in cache */
        size_t off = ((size_t)bl * T + pos) * E;
        memcpy(kv_k + off, ka, E * sizeof(float));
        memcpy(kv_v + off, va, E * sizeof(float));
        memcpy(kv_vr + off, vra, E * sizeof(float));

        /* Echo */
        float echo_out[1024];
        nt_qmatvec(echo_out, w->b[bl].wj, w->wdtype, rn, E, E);

        /* Gate softmax */
        float gs[16][3];
        for (int h = 0; h < H; h++) {
            gs[h][0] = w->b[bl].gate[h*3];
            gs[h][1] = w->b[bl].gate[h*3+1];
            gs[h][2] = w->b[bl].gate[h*3+2];
            softmax_f(gs[h], 3);
        }

        float cat[1024];
        memset(cat, 0, E * sizeof(float));

        for (int h = 0; h < H; h++) {
            float *q_h = qa + h*D;

            /* Content attention: Q @ cached_K^T */
            float attn[2048];
            for (int j = 0; j <= pos; j++) {
                float *kj = kv_k + ((size_t)bl * T + j) * E + h*D;
                float s = 0;
                for (int d = 0; d < D; d++) s += q_h[d] * kj[d];
                attn[j] = s * sc;
            }
            softmax_f(attn, pos + 1);

            float c_out[128];
            memset(c_out, 0, D * sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *vj = kv_v + ((size_t)bl * T + j) * E + h*D;
                for (int d = 0; d < D; d++) c_out[d] += attn[j] * vj[d];
            }

            /* RRPRAM low-rank (broadcast pattern):
             * Python: intermediate[h,r] = sum_t sum_e x[t,e] * wr_a[h,e,r]
             *         score[j] = sum_r intermediate[h,r] * wr_b[h,r,j] * sc
             *         attn[i,j] = softmax(score[j] for j<=i)  — SAME score broadcast
             *
             * For autoregressive: accumulate intermediate across positions in cache.
             * rrpram_mid[bl][h][r] += sum_e xn[e] * wr_a[h,e,r] at each new position.
             */
            float *wr_a_h = w->b[bl].wr_a + h*E*R;
            float *wr_b_h = w->b[bl].wr_b + h*R*T;
            /* Accumulate current position's contribution to mid */
            float *mid_cache = kv_rrpram_mid + ((size_t)bl * H + h) * R;
            for (int r = 0; r < R; r++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += rn[e] * wr_a_h[e*R+r];
                mid_cache[r] += s;
            }
            /* Score from accumulated mid */
            float r_attn[2048];
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int r = 0; r < R; r++) s += mid_cache[r] * wr_b_h[r*T+j];
                r_attn[j] = s * sc;
            }
            softmax_f(r_attn, pos + 1);

            float r_out[128];
            memset(r_out, 0, D * sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *vrj = kv_vr + ((size_t)bl * T + j) * E + h*D;
                for (int d = 0; d < D; d++) r_out[d] += r_attn[j] * vrj[d];
            }

            float *e_h = echo_out + h*D;

            for (int d = 0; d < D; d++)
                cat[h*D+d] = gs[h][0]*c_out[d] + gs[h][1]*r_out[d] + gs[h][2]*e_h[d];
        }

        /* Output projection + residual (x = x + attn_out) */
        float ao[1024];
        nt_qmatvec(ao, w->b[bl].cproj, w->wdtype, cat, E, E);
        for (int e = 0; e < E; e++) x[e] += ao[e];

        /* Cache mid-layer for backout */
        if (bl == backout_layer)
            memcpy(x_backout, x, E * sizeof(float));

        /* MLP: x = x + mlp(norm(x)) */
        rmsnorm(rn2, x, E);
        float mg[2048], mu[2048], mo[1024];
        nt_qmatvec(mg, w->b[bl].wg, w->wdtype, rn2, M, E);
        nt_qmatvec(mu, w->b[bl].wu, w->wdtype, rn2, M, E);
        for (int i = 0; i < M; i++) mg[i] = siluf(mg[i]) * mu[i];
        nt_qmatvec(mo, w->b[bl].wd, w->wdtype, mg, E, M);
        for (int e = 0; e < E; e++) x[e] += mo[e];

    }

    /* Backout: subtract cached mid-layer residual */
    float bl_val = *w->backout_l;
    for (int e = 0; e < E; e++) x[e] -= bl_val * x_backout[e];

    rmsnorm(rn, x, E);
    if (hidden) memcpy(hidden, rn, E * sizeof(float));   /* field carry = pre-δ */
    am_apply_delta(rn, g_delta_A, g_delta_B, rn, E, E, g_delta_rank,
                   am_lora_alpha_effective());   /* B2-B.4: lora_alpha, or lora_alpha*resonance when LORA_DYNAMIC */
    nt_qmatvec(logits, w->head, w->wdtype, rn, V, E);

    /* Softcap: logits = 15 * tanh(logits / 15) */
    for (int i = 0; i < V; i++)
        logits[i] = 15.0f * tanhf(logits[i] / 15.0f);

}

/* yent_forward.h ends here — yent.aml owns BPE init, sampling, and main(). */
