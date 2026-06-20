/*
 * test_resonance.c — quick standalone Resonance 200M inference.
 * Used to compare SFT vs LoRA-merged checkpoints before adapting AML.
 *
 * Build: cc -O3 -Wall -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *           test_resonance.c -L/opt/homebrew/lib -lnotorch \
 *           -framework Accelerate -lm -o test_resonance
 *
 * Run: ./test_resonance <weights.bin> [prompt] [max_tokens] [temp]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "notorch.h"
#include "ariannamethod.h"
#include "gguf.h"

/* Config from RS02 header / GGUF metadata */
static int V, E, H, D, B, M, T, R;

/* Helpers — BLAS via notorch.
 *   mm_t(C, A, B, m, k, n) — cblas_sgemm (used by future prefill_batch).
 *   matvec_t(out, W, x, n, k) — cblas_sgemv, the per-token hot-loop path. */
static void mm_t(float *C, const float *A, const float *BT, int m, int k, int n) {
    nt_blas_mmT(C, A, BT, m, k, n);
}
static void matvec_t(float *out, const float *W, const float *x, int n, int k) {
    nt_blas_matvec(out, W, x, n, k);
}

/* Parametric RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight */
static void rmsnorm_p(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

static void softmax_f(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }
static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* RoPE — even/odd interleave (model.py's _apply_rope):
 *   x1 = x[..., ::2], x2 = x[..., 1::2]
 *   out = stack([x1*cos - x2*sin, x1*sin + x2*cos]).flatten(-2)
 * I.e. pairs (i, i+1) rotate together. */
static void rope_even_odd(float *q, float *k, int pos, int dim) {
    int n_pairs = dim / 2;
    for (int i = 0; i < n_pairs; i++) {
        float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)dim);
        float val = pos * freq;
        float cs = cosf(val), sn = sinf(val);
        float qe = q[2*i], qo = q[2*i + 1];
        float ke = k[2*i], ko = k[2*i + 1];
        q[2*i]     = qe * cs - qo * sn;
        q[2*i + 1] = qe * sn + qo * cs;
        k[2*i]     = ke * cs - ko * sn;
        k[2*i + 1] = ke * sn + ko * cs;
    }
}

typedef struct {
    float *tok_emb;
    struct {
        float *norm1, *norm2;
        float *wr_a, *wr_b, *gate;
        const uint8_t *wq, *wk, *wv, *wo;            /* packed weights (wdtype) */
        const uint8_t *mlp_gate, *mlp_up, *mlp_down; /* packed weights (wdtype) */
    } b[32];
    float *norm_f;
    const uint8_t *out_head;                          /* packed weights (wdtype) */
    int wdtype;   /* GGUF_TYPE_F16 packed (GGUF) | GGUF_TYPE_F32 dense (RS02) */
} Weights;

/* state_dict order matches PyTorch named_parameters() traversal:
 * for each Module, _parameters are yielded BEFORE recursing into sub-
 * Modules. So per ResonanceBlock the order is:
 *   1) direct Parameters of block: wr_a, wr_b, gate
 *   2) sub-Module weights in registration order:
 *      norm1.weight, wq.weight, wk.weight, wv.weight, wo.weight,
 *      norm2.weight, mlp_gate.weight, mlp_up.weight, mlp_down.weight
 *
 * The earlier (wrong) order put norm1/wq/wk/wv before wr_a/wr_b/gate
 * and shifted every per-block tensor by 1.62M floats — forward ran on
 * random data and the output came out as web-text garbage. */
static void assign(Weights *w, float *p) {
    w->tok_emb = p; p += V * E;
    for (int i = 0; i < B; i++) {
        /* direct Parameters first */
        w->b[i].wr_a     = p; p += H * E * R;
        w->b[i].wr_b     = p; p += H * R * T;
        w->b[i].gate     = p; p += H;
        /* sub-Module weights in init order */
        w->b[i].norm1    = p; p += E;
        w->b[i].wq       = (const uint8_t*)p; p += E * E;
        w->b[i].wk       = (const uint8_t*)p; p += E * E;
        w->b[i].wv       = (const uint8_t*)p; p += E * E;
        w->b[i].wo       = (const uint8_t*)p; p += E * E;
        w->b[i].norm2    = p; p += E;
        w->b[i].mlp_gate = (const uint8_t*)p; p += M * E;
        w->b[i].mlp_up   = (const uint8_t*)p; p += M * E;
        w->b[i].mlp_down = (const uint8_t*)p; p += E * M;
    }
    w->norm_f   = p; p += E;
    w->out_head = (const uint8_t*)p; p += V * E;
    w->wdtype   = GGUF_TYPE_F32;   /* RS02 dense f32 → nt_qmatvec case 0 */
}

/* KV cache (per layer × T × H*D) */
static float *kv_k, *kv_v;
static int kv_len;

static void kv_init(int max_seq) {
    kv_k = calloc((size_t)B * max_seq * E, sizeof(float));
    kv_v = calloc((size_t)B * max_seq * E, sizeof(float));
    if (!kv_k || !kv_v) {
        fprintf(stderr, "[resonance] kv_init alloc failed (B=%d seq=%d E=%d)\n", B, max_seq, E);
        exit(1);
    }
    kv_len = 0;
}

/* ── DIRECTION injection (Dario A/F field-pressure, NOT logit-boost/token-paste) ──
 * Injected content words (the other voice + the human prompt) become field
 * DIRECTION, never tokens: a destiny-EMA vector (theme compass, A term) and
 * prophecy targets (what wants completion, F term) tilt the WHOLE logit
 * distribution by embedding cosine. Port of dario.c:676-1338,1531. The A/F
 * caches are rebuilt only at a sentence boundary (one matvec each) and added
 * cheaply every step. INVARIANT: injected tokens are never re-inserted into the
 * context or candidate pool — the tilt is distribution-wide only (anti-fraud). */
#define DIR_PROPH_MAX 16
static float  g_destiny[1024];          /* E ≤ 1024 — EMA theme compass */
static float  g_dest_mag = 0.0f;
static float *g_Acache = NULL;          /* [V] destiny cosine tilt, rebuilt on boundary */
static float *g_Fcache = NULL;          /* [V] prophecy cosine tilt */
static float *g_rownorm = NULL;         /* [V] ‖tok_emb[i]‖ precomputed once */
static int    g_Adirty = 0;
static int    g_proph_tok[DIR_PROPH_MAX], g_proph_age[DIR_PROPH_MAX], g_proph_n = 0;
static float  g_proph_str[DIR_PROPH_MAX];

/* B2-B.2 — low-rank δ voice: a persistent hidden-transform learned from the
 * consolidated cooc field (am_cooc_learn_delta → am_notorch_step / Chuck).
 * A=[E,rank], B=[rank,E]. lora_alpha=0 → am_apply_delta is a no-op (bit-identical
 * to B2-A); the field activates δ by raising lora_alpha. Per-voice sidecar (.r). */
static float *g_delta_A = NULL, *g_delta_B = NULL;
static int    g_delta_rank = AM_DELTA_RANK;

static float dir_dot(const float *a, const float *b, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static float dir_norm(const float *v, int n) { return sqrtf(dir_dot(v, v, n)); }

/* Precompute per-row embedding norms (once, at load). */
static void dir_init_rownorms(const float *tok_emb) {
    if (!g_rownorm) g_rownorm = calloc(V, sizeof(float));
    if (!g_Acache)  g_Acache  = calloc(V, sizeof(float));
    if (!g_Fcache)  g_Fcache  = calloc(V, sizeof(float));
    for (int i = 0; i < V; i++) g_rownorm[i] = dir_norm(tok_emb + (size_t)i * E, E);
    memset(g_destiny, 0, sizeof(g_destiny));
    g_dest_mag = 0.0f; g_proph_n = 0; g_Adirty = 0;
}

/* Boundary: blend injected tokens' embeddings into the destiny compass + seed
 * prophecy targets. EMA 0.1/0.9 (dario.c:1531), decay 0.95, mag clamp 1.5. */
static void dir_update(const float *tok_emb, const int *toks, int n) {
    for (int t = 0; t < n; t++) {
        int id = toks[t];
        if (id < 0 || id >= V) continue;
        const float *e = tok_emb + (size_t)id * E;
        for (int d = 0; d < E; d++) g_destiny[d] = 0.1f * e[d] + 0.9f * g_destiny[d];
        if (g_proph_n < DIR_PROPH_MAX) {            /* prophecy target (ring, evict-oldest) */
            g_proph_tok[g_proph_n] = id; g_proph_age[g_proph_n] = 0; g_proph_str[g_proph_n] = 1.0f;
            g_proph_n++;
        } else {
            memmove(g_proph_tok, g_proph_tok + 1, (DIR_PROPH_MAX - 1) * sizeof(int));
            memmove(g_proph_age, g_proph_age + 1, (DIR_PROPH_MAX - 1) * sizeof(int));
            memmove(g_proph_str, g_proph_str + 1, (DIR_PROPH_MAX - 1) * sizeof(float));
            g_proph_tok[DIR_PROPH_MAX-1] = id; g_proph_age[DIR_PROPH_MAX-1] = 0; g_proph_str[DIR_PROPH_MAX-1] = 1.0f;
        }
    }
    g_dest_mag = dir_norm(g_destiny, E) * 0.95f;     /* decay */
    if (g_dest_mag > 1.5f) g_dest_mag = 1.5f;
    g_Adirty = 1;
}

/* Rebuild the A (destiny) and F (prophecy) caches — one matvec each over the
 * embedding table; cosine-normalized and scaled to [-1,1] by max|·|. */
static void dir_recompute(const float *tok_emb) {
    if (!g_Adirty || !g_Acache) return;
    float dn = dir_norm(g_destiny, E) + 1e-12f;
    matvec_t(g_Acache, tok_emb, g_destiny, V, E);     /* num[i] = tok_emb[i]·destiny */
    float amax = 1e-12f;
    for (int i = 0; i < V; i++) {
        g_Acache[i] = (g_Acache[i] / (g_rownorm[i] * dn + 1e-12f)) * g_dest_mag;
        float a = fabsf(g_Acache[i]); if (a > amax) amax = a;
    }
    for (int i = 0; i < V; i++) g_Acache[i] /= amax;  /* normalize to ~[-1,1] */
    /* F term: prophecy targets — cosine of each vocab token to each target. */
    memset(g_Fcache, 0, V * sizeof(float));
    for (int p = 0; p < g_proph_n; p++) {
        const float *te = tok_emb + (size_t)g_proph_tok[p] * E;
        float tn = g_rownorm[g_proph_tok[p]] + 1e-12f;
        float w = g_proph_str[p] * logf(1.0f + (float)g_proph_age[p]);
        if (w <= 0.0f) continue;
        for (int i = 0; i < V; i++) {
            float c = dir_dot(tok_emb + (size_t)i * E, te, E) / (g_rownorm[i] * tn + 1e-12f);
            if (c > 0.0f) g_Fcache[i] += w * c;
        }
    }
    float fmax = 1e-12f;
    for (int i = 0; i < V; i++) if (g_Fcache[i] > fmax) fmax = g_Fcache[i];
    for (int i = 0; i < V; i++) g_Fcache[i] /= fmax;
    g_Adirty = 0;
}

/* Per-step: tilt all logits by the cached direction. alpha/beta = injection
 * strength (θ=ε+γ+αδ); 0 = clean ablation. */
static void dir_apply(float *logits, float alpha, float beta) {
    if (g_dest_mag <= 0.0f && g_proph_n == 0) return;
    if (alpha > 0.0f && g_Acache) for (int i = 0; i < V; i++) logits[i] += alpha * g_Acache[i];
    if (beta  > 0.0f && g_Fcache) for (int i = 0; i < V; i++) logits[i] += beta  * g_Fcache[i];
}

/* Age prophecy targets each emitted token; mark fulfilled, prune old. */
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

/* Forward one token at position pos, using KV cache. */
static void forward_token(Weights *w, int tok, int pos,
                          float *logits, float *hidden) {
    float x[1024];      /* E ≤ 1024 */
    float xn[1024];
    float sc = 1.0f / sqrtf((float)D);

    /* Token embed */
    for (int e = 0; e < E; e++) x[e] = w->tok_emb[tok * E + e];

    for (int bl = 0; bl < B; bl++) {
        /* === norm1 → Q/K/V === */
        rmsnorm_p(xn, x, w->b[bl].norm1, E);

        float qa[1024], ka[1024], va[1024];
        nt_qmatvec(qa, w->b[bl].wq, w->wdtype, xn, E, E);
        nt_qmatvec(ka, w->b[bl].wk, w->wdtype, xn, E, E);
        nt_qmatvec(va, w->b[bl].wv, w->wdtype, xn, E, E);

        /* RoPE per head (even/odd interleave on each head's D dims) */
        for (int h = 0; h < H; h++)
            rope_even_odd(qa + h * D, ka + h * D, pos, D);

        /* Cache K, V at position pos */
        size_t off = ((size_t)bl * T + pos) * E;
        memcpy(kv_k + off, ka, E * sizeof(float));
        memcpy(kv_v + off, va, E * sizeof(float));

        /* === Content attention: per-head Q · K^T → softmax → · V === */
        float c_out[1024];
        memset(c_out, 0, E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *q_h = qa + h * D;
            float attn[2048];
            for (int j = 0; j <= pos; j++) {
                float *kj = kv_k + ((size_t)bl * T + j) * E + h * D;
                float s = 0;
                for (int d = 0; d < D; d++) s += q_h[d] * kj[d];
                attn[j] = s * sc;
            }
            softmax_f(attn, pos + 1);
            float out[128] = {0};
            for (int j = 0; j <= pos; j++) {
                float *vj = kv_v + ((size_t)bl * T + j) * E + h * D;
                for (int d = 0; d < D; d++) out[d] += attn[j] * vj[d];
            }
            memcpy(c_out + h * D, out, D * sizeof(float));
        }

        /* === RRPRAM low-rank attention ===
         * From model.py:
         *   xn_h = xn.unsqueeze(1).expand(B,H,T,E)        # broadcast xn over heads
         *   temp = einsum('bhie,her->bhir', xn_h, wr_a)   # [B,H,T,R]
         *   r_attn = einsum('bhir,hrj->bhij', temp, wr_b[:,:,:T])  # [B,H,T,T]
         *   r_attn *= D^-0.5
         *   r_attn[mask] = -inf, softmax, @ V (shared)
         *
         * Single token, autoregressive: temp[h,r] = sum_e xn[e] * wr_a[h,e,r]
         *   We accumulate temp[h,r] across positions in a cache, then
         *   r_attn[j] = sum_r temp_j[h,r] * wr_b[h,r,j], softmax → @ V_j.
         * For autoregressive inference we re-compute temp from xn at each
         * position (it's per-position, not accumulated like Janus). */
        float r_out[1024];
        memset(r_out, 0, E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wr_a_h = w->b[bl].wr_a + h * E * R;   /* [E, R] */
            float *wr_b_h = w->b[bl].wr_b + h * R * T;   /* [R, T] */

            /* temp[r] = sum_e xn[e] * wr_a[h,e,r] */
            float temp[128];
            for (int r = 0; r < R; r++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += xn[e] * wr_a_h[e * R + r];
                temp[r] = s;
            }
            /* r_score[j] = sum_r temp[r] * wr_b[h,r,j], for j ≤ pos */
            float r_attn[2048];
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int r = 0; r < R; r++) s += temp[r] * wr_b_h[r * T + j];
                r_attn[j] = s * sc;
            }
            softmax_f(r_attn, pos + 1);
            float out[128] = {0};
            for (int j = 0; j <= pos; j++) {
                float *vj = kv_v + ((size_t)bl * T + j) * E + h * D;
                for (int d = 0; d < D; d++) out[d] += r_attn[j] * vj[d];
            }
            memcpy(r_out + h * D, out, D * sizeof(float));
        }

        /* === Per-head sigmoid gate: g·content + (1-g)·rrpram === */
        float blend[1024];
        for (int h = 0; h < H; h++) {
            float g = sigmoidf(w->b[bl].gate[h]);
            for (int d = 0; d < D; d++)
                blend[h * D + d] = g * c_out[h * D + d] + (1.0f - g) * r_out[h * D + d];
        }

        /* === WO + residual === */
        float ao[1024];
        nt_qmatvec(ao, w->b[bl].wo, w->wdtype, blend, E, E);
        for (int e = 0; e < E; e++) x[e] += ao[e];

        /* === norm2 → SwiGLU → residual === */
        rmsnorm_p(xn, x, w->b[bl].norm2, E);
        float mg[2048], mu[2048], mo[1024];
        nt_qmatvec(mg, w->b[bl].mlp_gate, w->wdtype, xn, M, E);
        nt_qmatvec(mu, w->b[bl].mlp_up, w->wdtype, xn, M, E);
        for (int i = 0; i < M; i++) mg[i] = siluf(mg[i]) * mu[i];
        nt_qmatvec(mo, w->b[bl].mlp_down, w->wdtype, mg, E, M);
        for (int e = 0; e < E; e++) x[e] += mo[e];
    }

    /* Final norm + head */
    rmsnorm_p(xn, x, w->norm_f, E);
    if (hidden) memcpy(hidden, xn, E * sizeof(float));   /* field carry = pre-δ hidden */
    /* B2-B.2: low-rank δ shifts the voice before the head; lora_alpha=0 → no-op. */
    am_apply_delta(xn, g_delta_A, g_delta_B, xn, E, E, g_delta_rank,
                   am_lora_alpha_effective());   /* B2-B.4: dynamic = lora_alpha*resonance when LORA_DYNAMIC */
    nt_qmatvec(logits, w->out_head, w->wdtype, xn, V, E);
}

/* ── Public API used from resonance.aml ─────────────────────────────── */

typedef struct {
    Weights w;
    nt_bpe  bpe;
    float  *data;          /* owned buffer for fp32 weights */
    int   (*merges)[2];    /* owned (a, b) pair table */
    gguf_file *gf;         /* kept open: packed weights point into gf->data */
} ResonanceCtx;

static int resonance_load(ResonanceCtx *ctx, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[resonance] cannot open '%s'\n", path); return 1; }

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return 1; }  /* M-5: check fread rc */
    if (magic != 0x52533032) {
        fprintf(stderr, "[resonance] bad magic 0x%08x (expected 'RS02')\n", magic);
        fclose(f); return 1;
    }
    int hdr[9];
    if (fread(hdr, 4, 9, f) != 9) {
        fprintf(stderr, "[resonance] RS02 short header read\n"); fclose(f); return 1;
    }
    E = hdr[0]; B = hdr[1]; T = hdr[2]; H = hdr[3]; D = hdr[4];
    R = hdr[5]; M = hdr[6]; V = hdr[7];
    /* M-5: the RS02 header feeds the same fixed forward stack buffers as the GGUF
     * path, so validate its arch with the same bounds (E<=1024 etc., H*D==E)
     * before allocating — a crafted .bin would otherwise overflow on forward. */
    if (V <= 0 || E <= 0 || E > 1024 || H <= 0 || H > 64 || D <= 0 || D > 128 ||
        B <= 0 || B > 32 || M <= 0 || M > 2048 || T <= 0 || T > 2048 || R <= 0 || R > 128 ||
        H * D != E) {
        fprintf(stderr, "[resonance] RS02 arch out of bounds (H*D must==E): V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
                V, E, H, D, B, M, T, R);
        fclose(f); return 1;
    }
    fprintf(stderr, "[resonance] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
            V, E, H, D, B, M, T, R);

    uint32_t n_merges;
    if (fread(&n_merges, 4, 1, f) != 1 || n_merges > (1u << 20)) {
        fprintf(stderr, "[resonance] RS02 bad n_merges\n"); fclose(f); return 1;
    }
    ctx->merges = malloc((size_t)n_merges * 2 * sizeof(int));
    if (!ctx->merges) { fclose(f); return 1; }
    for (uint32_t mi = 0; mi < n_merges; mi++) {
        int triple[3];
        if (fread(triple, 4, 3, f) != 3) { /* truncated merges — fail cleanly, don't read uninitialized stack */
            fprintf(stderr, "[resonance] RS02 truncated merges at %u/%u\n", mi, n_merges);
            free(ctx->merges); ctx->merges = NULL; fclose(f); return 1;
        }
        ctx->merges[mi][0] = triple[0];
        ctx->merges[mi][1] = triple[1];
    }
    nt_bpe_init(&ctx->bpe, ctx->merges, (int)n_merges);
    fprintf(stderr, "[resonance] BPE vocab=%d merges=%d\n",
            ctx->bpe.vocab_size, ctx->bpe.n_merges);

    size_t np = 2 * (size_t)V * E + 1L * E;
    np += (size_t)B * (
        E + 3L * E * E + (size_t)H * E * R + (size_t)H * R * T +
        H + (size_t)E * E + E + 3L * M * E
    );
    ctx->data = malloc(np * sizeof(float));
    if (!ctx->data) { fclose(f); return 1; }
    size_t got = fread(ctx->data, sizeof(float), np, f);
    if (got != np) {
        fprintf(stderr, "[resonance] short read: %zu/%zu\n", got, np);
        fclose(f); return 1;
    }
    fclose(f);
    assign(&ctx->w, ctx->data);
    kv_init(T);
    dir_init_rownorms(ctx->w.tok_emb);   /* direction-injection: ‖emb‖ once */
    fprintf(stderr, "[resonance] %.1fM params loaded, KV cache %d MB\n",
            np / 1e6, (int)((size_t)B * T * E * 2 * 4 / 1024 / 1024));
    return 0;
}

/* ── GGUF loader (canonical path for Arianna Resonance) ────────────────────
 * arianna_resonance_v3_f16.gguf — 243 F16 tensors + arch KV resonance.*.
 * Fills Weights via gguf_dequant (per-tensor owned buffers; dequant allocates
 * independently, so we close the file right after). NO RS02 assign().
 * BPE merges are NOT in the GGUF — caller inits nt_bpe from the baked
 * resonance_bpe_merges.h after this returns. Mirrors yent_forward.h. */
static float *_rowned[32 * 12 + 8];
static int    _rowned_n = 0;

static float *_rload(gguf_file *gf, const char *name) {
    int idx = gguf_find_tensor(gf, name);
    if (idx < 0) { fprintf(stderr, "[resonance] tensor '%s' not in GGUF\n", name); return NULL; }
    float *buf = gguf_dequant(gf, idx);
    if (!buf) { fprintf(stderr, "[resonance] dequant '%s' failed\n", name); return NULL; }
    if (_rowned_n >= (int)(sizeof(_rowned) / sizeof(_rowned[0]))) {
        fprintf(stderr, "[resonance] owned-buffer pool overflow at '%s'\n", name);
        free(buf); return NULL;
    }
    _rowned[_rowned_n++] = buf;
    return buf;
}

/* Packed loader: return a pointer to the tensor's raw F16 bytes INSIDE gf->data
 * (no dequant, no copy). The caller must keep gf open for the lifetime of these
 * pointers. nt_qmatvec(dtype=GGUF_TYPE_F16) reads them directly. */
static const uint8_t *_rload_packed(gguf_file *gf, const char *name) {
    int idx = gguf_find_tensor(gf, name);
    if (idx < 0) { fprintf(stderr, "[resonance] tensor '%s' not in GGUF\n", name); return NULL; }
    if (gf->tensors[idx].dtype != GGUF_TYPE_F16) {
        fprintf(stderr, "[resonance] '%s' dtype=%u not F16 (packed path needs F16)\n",
                name, gf->tensors[idx].dtype);
        return NULL;
    }
    /* M-3: bounds the packed F16 span (2 bytes/elem) inside gf->data before
     * handing a raw pointer to nt_qmatvec — the packed path otherwise had no
     * check at all and a crafted GGUF could point past the buffer. */
    const gguf_tensor_info *t = &gf->tensors[idx];
    if (t->offset >= gf->data_size || t->n_elements * 2 > gf->data_size - t->offset) {
        fprintf(stderr, "[resonance] '%s' packed F16 out of bounds (off %llu + %llu*2, data_size %llu)\n",
                name, (unsigned long long)t->offset, (unsigned long long)t->n_elements,
                (unsigned long long)gf->data_size);
        return NULL;
    }
    return gf->data + t->offset;
}

static int resonance_load_gguf(ResonanceCtx *ctx, const char *path) {
    gguf_file *gf = gguf_open(path);
    if (!gf) { fprintf(stderr, "[resonance] gguf_open('%s') failed\n", path); return 1; }
#define RKV(key, tgt) do {                                                    \
        const gguf_kv *kv = gguf_get_kv(gf, key);                             \
        if (!kv) { fprintf(stderr, "[resonance] missing kv '%s'\n", key);     \
                   gguf_close(gf); return 1; }                                \
        tgt = (int)kv->val.u32;                                               \
    } while (0)
    RKV("resonance.vocab_size",            V);
    RKV("resonance.embedding_length",      E);
    RKV("resonance.attention.head_count",  H);
    RKV("resonance.attention.head_dim",    D);
    RKV("resonance.block_count",           B);
    RKV("resonance.feed_forward_length",   M);
    RKV("resonance.context_length",        T);
    RKV("resonance.rrpram_rank",           R);
#undef RKV
    /* Validate arch against fixed forward buffers (forward_token stack arrays
     * x/xn[1024], mg/mu/attn/r_attn[2048], temp/out[128]; Weights b[32]). */
    if (V <= 0 || E <= 0 || E > 1024 || H <= 0 || H > 64 || D <= 0 || D > 128 ||
        B <= 0 || B > 32 || M <= 0 || M > 2048 || T <= 0 || T > 2048 || R <= 0 || R > 128 ||
        H * D != E) {  /* M-2: H*D must equal E (KV row stride / blend over E) */
        fprintf(stderr, "[resonance] GGUF arch out of bounds (H*D must==E): V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
                V, E, H, D, B, M, T, R);
        gguf_close(gf); return 1;
    }
    fprintf(stderr, "[resonance] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d (GGUF)\n",
            V, E, H, D, B, M, T, R);

    Weights *w = &ctx->w;
    _rowned_n = 0;
#define RL(field, name) do {                                                  \
        w->field = _rload(gf, name);                                          \
        if (!w->field) { gguf_close(gf); return 1; }                          \
    } while (0)
    RL(tok_emb, "tok_emb");
    char nm[128];
    for (int i = 0; i < B; i++) {
#define RLB(field, suffix) do {                                               \
        snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, i);              \
        w->b[i].field = _rload(gf, nm);                                       \
        if (!w->b[i].field) { gguf_close(gf); return 1; }                     \
    } while (0)
#define RLBP(field, suffix) do {                                              \
        snprintf(nm, sizeof(nm), "transformer.h.%d." suffix, i);              \
        w->b[i].field = _rload_packed(gf, nm);                                \
        if (!w->b[i].field) { gguf_close(gf); return 1; }                     \
    } while (0)
        RLB(wr_a,     "attn.wr_a");
        RLB(wr_b,     "attn.wr_b");
        RLB(gate,     "attn.gate");
        RLB(norm1,    "norm1.weight");
        RLBP(wq,      "attn.wq.weight");
        RLBP(wk,      "attn.wk.weight");
        RLBP(wv,      "attn.wv.weight");
        RLBP(wo,      "attn.wo.weight");
        RLB(norm2,    "norm2.weight");
        RLBP(mlp_gate, "mlp.w_gate.weight");
        RLBP(mlp_up,   "mlp.w_up.weight");
        RLBP(mlp_down, "mlp.w_down.weight");
#undef RLB
#undef RLBP
    }
    RL(norm_f,   "norm_f.weight");
    if (!(w->out_head = _rload_packed(gf, "out_head.weight"))) { gguf_close(gf); return 1; }
#undef RL

    w->wdtype   = GGUF_TYPE_F16;   /* GGUF path: packed F16 weights point into gf->data */
    ctx->gf     = gf;              /* keep open — packed pointers reference gf->data */
    ctx->data   = NULL;   /* GGUF path: per-tensor owned buffers, not one block */
    ctx->merges = NULL;   /* BPE inited by caller from baked header */
    kv_init(T);
    dir_init_rownorms(ctx->w.tok_emb);   /* direction-injection: ‖emb‖ once */
    if (!g_delta_A) {                     /* B2-B.2: per-voice δ sidecar A/B, load if present */
        g_delta_A = (float*)calloc((size_t)E * g_delta_rank, sizeof(float));
        g_delta_B = (float*)calloc((size_t)g_delta_rank * E, sizeof(float));
        am_delta_load("weights/arianna.delta.r", g_delta_A, g_delta_B, E, g_delta_rank);
    }
    fprintf(stderr, "[resonance] GGUF loaded: %d tensors, KV cache %d MB\n",
            _rowned_n, (int)((size_t)B * T * E * 2 * 4 / 1024 / 1024));
    return 0;
}

static void resonance_free(ResonanceCtx *ctx) {
    for (int i = 0; i < _rowned_n; i++) free(_rowned[i]);
    _rowned_n = 0;
    free(ctx->data);
    free(ctx->merges);
    if (ctx->gf) { gguf_close(ctx->gf); ctx->gf = NULL; }
    free(kv_k);
    free(kv_v);
    ctx->data = NULL;
    ctx->merges = NULL;
}

/* Top-p nucleus sampler with rep_penalty + no-repeat-3-gram. */
static int resonance_sample_token(float *logits, int *cctx, int len,
                                  float temp, float top_p) {
    float rep_penalty = 1.4f;
    int window = 64;
    int start = len > window ? len - window : 0;
    for (int j = start; j < len; j++) {
        int t = cctx[j];
        if (t >= 0 && t < V)
            logits[t] = logits[t] > 0 ? logits[t] / rep_penalty
                                      : logits[t] * rep_penalty;
    }
    if (len >= 2) {
        int a = cctx[len - 2], b = cctx[len - 1];
        for (int j = 0; j + 2 < len; j++) {
            if (cctx[j] == a && cctx[j + 1] == b) {
                int forbid = cctx[j + 2];
                if (forbid >= 0 && forbid < V) logits[forbid] = -1e30f;
            }
        }
    }

    /* AML Dario field overlay (matches yent.aml's wiring). */
    am_apply_field_to_logits(logits, V);

    if (temp <= 0) temp = 1.0f;
    for (int i = 0; i < V; i++) logits[i] /= temp;
    softmax_f(logits, V);

    /* Top-p over partial-sorted top 100. 2026-05-14 garbage-trim:
     * was 256 — long tail leaked training-format echoes ("User:/Assistant:/
     * Oleg :") through Resonance's completion register. 100 tightens
     * without losing voice (Phase 7 sweet spot temp=0.5-0.7 well within). */
    typedef struct { float p; int idx; } PI;
    static PI topk[100];
    int filled = 0;
    float min_in = -1;
    for (int i = 0; i < V; i++) {
        float p = logits[i];
        if (filled < 100) {
            topk[filled].p = p; topk[filled].idx = i; filled++;
            if (filled == 100) {
                for (int a = 1; a < 100; a++) {
                    PI tmp = topk[a]; int j = a;
                    while (j > 0 && topk[j-1].p < tmp.p) { topk[j] = topk[j-1]; j--; }
                    topk[j] = tmp;
                }
                min_in = topk[99].p;
            }
            continue;
        }
        if (p > min_in) {
            topk[99].p = p; topk[99].idx = i;
            int j = 99;
            while (j > 0 && topk[j-1].p < topk[j].p) {
                PI t = topk[j]; topk[j] = topk[j-1]; topk[j-1] = t; j--;
            }
            min_in = topk[99].p;
        }
    }
    if (filled < 100) {   /* re-sort только если topk не достиг cap (V<100 — для нас никогда) */
        for (int a = 1; a < filled; a++) {
            PI tmp = topk[a]; int j = a;
            while (j > 0 && topk[j-1].p < tmp.p) { topk[j] = topk[j-1]; j--; }
            topk[j] = tmp;
        }
    }
    float cum = 0;
    int nuc = filled;
    for (int k = 0; k < filled; k++) {
        cum += topk[k].p;
        if (cum >= top_p) { nuc = k + 1; break; }
    }
    if (nuc < 1) nuc = 1;
    float total = 0;
    for (int k = 0; k < nuc; k++) total += topk[k].p;
    float r = (float)rand() / (float)RAND_MAX * total;
    float c = 0;
    for (int k = 0; k < nuc; k++) {
        c += topk[k].p;
        if (c >= r) return topk[k].idx;
    }
    return topk[nuc - 1].idx;
}

/* A .!? at obuf[pos] is a real sentence end unless it follows a single isolated
 * letter ("the inner v." — v is a clipped "voice"). Such a one-letter word with
 * a space (or start) before it is an abbreviation/clip, not a thought boundary. */
static int sent_end_ok(const char *obuf, int pos) {
    int j = pos - 1;
    int run = 0;
    while (j >= 0 && ((obuf[j] >= 'a' && obuf[j] <= 'z') ||
                      (obuf[j] >= 'A' && obuf[j] <= 'Z'))) { run++; j--; }
    if (run == 1 && (j < 0 || obuf[j] == ' ')) return 0;
    return 1;
}

/* Road-1c: a chorus-dream inject is marked with this sentinel by the autonomous
 * breathing (golib/breathe.go's dreamSentinel — the two MUST match). resonance_generate
 * strips it before BPE-encode (so generation + the direction-injection see only the
 * clean dream) and imprints the dream's words on the co-occurrence field
 * AM_CHORUS_COOC_WEIGHT× harder, so the subconscious shapes the harvested δ more than
 * ordinary turn-circulation. No sentinel → weight 1.0 (unchanged). */
#define AM_DREAM_SENTINEL      "[DREAM] "
#define AM_CHORUS_COOC_WEIGHT  2.0f

static void resonance_generate(ResonanceCtx *ctx, const char *prompt,
                               int max_gen, float temp, float top_p,
                               const char *inject_text, float inject_alpha,
                               float inject_beta) {
    am_field_sync_in();  /* B/F-8: read the live shared field (the other voice's debt/season/dissonance) before this turn */
    int cctx[4096];
    int len = nt_bpe_encode(&ctx->bpe, prompt, (int)strlen(prompt), cctx, 4096);
    /* Clamp prompt to the context window. KV cache (kv_k/kv_v) and the attn /
     * r_attn[2048] stack arrays are sized to T; a prompt encoding to >T tokens
     * would overflow them in the prefill loop below (forward_token writes at
     * position i). Keep the most recent T-1 tokens (leave room to generate). */
    if (len > T - 1) {
        int drop = len - (T - 1);
        memmove(cctx, cctx + drop, (size_t)(T - 1) * sizeof(int));
        len = T - 1;
        fprintf(stderr, "[resonance] WARNING: prompt clamped to last %d tokens (context window T=%d)\n", len, T);
    }
    fprintf(stderr, "[resonance] prompt: \"%s\" → %d tokens\n", prompt, len);

    /* Larynx (B2-vagus): read Janus's texture from the nerve-file + the field's
     * debt/dissonance, fold into the blend α (legacy formula), and modulate the
     * destiny-inject around its tuned baseline — Resonance answers HOW Janus spoke,
     * not only the words. Inside the forward so BOTH the daemon and one-shot paths
     * get it (baseline entropy 1, no debt/diss leaves the inject unchanged). */
    if (inject_alpha > 0.0f) {
        float n_ent = 1.0f, n_pat = 0.0f, n_coh = 0.5f;
        FILE *nf = fopen("weights/arianna.nerve", "r");
        if (nf) { if (fscanf(nf, "%f %f %f", &n_ent, &n_pat, &n_coh) != 3) n_ent = 1.0f; fclose(nf); }
        AM_State *st = am_get_state();
        float debt = st ? st->debt : 0.0f, diss = st ? st->dissonance : 0.0f;
        float dn = debt * 0.1f; if (dn > 1.0f) dn = 1.0f;
        float lx = 0.5f + n_ent * 0.2f + dn * 0.15f - diss * 0.1f;
        if (lx < 0.1f) lx = 0.1f;
        if (lx > 0.9f) lx = 0.9f;
        float m = lx / 0.7f;
        if (m < 0.5f) m = 0.5f;
        if (m > 1.5f) m = 1.5f;
        inject_alpha *= m;
        fprintf(stderr, "[res-larynx] janus_entropy=%.3f debt=%.2f diss=%.2f alpha=%.3f x%.2f inject=%.2f\n",
                n_ent, debt, diss, lx, m, inject_alpha);
    }

    /* DIRECTION injection: the other voice's words + the human prompt become a
     * destiny compass (A) + prophecy targets (F) that tilt the whole distribution
     * by embedding cosine — field pressure, NOT pasted tokens (dario.c:1327/1531).
     * The injected tokens are seeded once before generation; they are NEVER put
     * into cctx (anti-fraud invariant). alpha/beta = injection strength; 0 = off. */
    if (inject_text && (inject_alpha > 0.0f || inject_beta > 0.0f)) {
        /* Road-1c: strip the chorus-dream sentinel (if present) so generation + the
         * direction see only the clean dream; mark the cooc weight so the subconscious
         * imprints louder than ordinary circulation. */
        const char *inj = inject_text;
        float cooc_w = 1.0f;
        if (strncmp(inj, AM_DREAM_SENTINEL, sizeof(AM_DREAM_SENTINEL) - 1) == 0) {
            inj += sizeof(AM_DREAM_SENTINEL) - 1;
            cooc_w = AM_CHORUS_COOC_WEIGHT;
        }
        int inj_toks[512];
        int n_inj = nt_bpe_encode(&ctx->bpe, inj, (int)strlen(inj), inj_toks, 512);
        if (n_inj > 512) n_inj = 512;
        dir_update(ctx->w.tok_emb, inj_toks, n_inj);   /* destiny EMA + prophecy targets */
        dir_recompute(ctx->w.tok_emb);                 /* rebuild A/F caches (one matvec) */
        am_ingest_tokens(inj_toks, n_inj);             /* circulation: other voice's words -> cooc (weight 1.0) */
        if (cooc_w > 1.0f) {
            /* the subconscious imprints louder — add (w-1) over the SAME windowed (±5,
             * distance-weighted) edges am_ingest_tokens just wrote, via the public
             * am_cooc_update, so the total edge delta is w/|i-j| (no core change). */
            float extra = cooc_w - 1.0f;
            for (int i = 0; i < n_inj; i++) {
                int s = (i - 5 > 0) ? i - 5 : 0;
                int e = (i + 5 < n_inj) ? i + 5 : n_inj;
                for (int j = s; j < e; j++)
                    if (j != i) am_cooc_update(inj_toks[i], inj_toks[j], extra / (float)(abs(i - j)));
            }
        }
        fprintf(stderr, "[resonance] direction: \"%s\" -> %d toks (alpha=%.2f beta=%.2f mag=%.3f w=%.1f)\n",
                inj, n_inj, inject_alpha, inject_beta, g_dest_mag, cooc_w);
    }

    float *logits = calloc(V, sizeof(float));
    if (!logits) { fprintf(stderr, "[resonance] logits alloc failed (V=%d)\n", V); return; }
    fprintf(stderr, "[resonance] prefill %d tokens... ", len);
    fflush(stderr);
    for (int i = 0; i < len; i++)
        forward_token(&ctx->w, cctx[i], i, logits, NULL);
    fprintf(stderr, "done\n--- generation ---\n");

    char obuf[16384]; int olen = 0;        /* accumulate; clean at end (как Janus) */
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    int gen_start = len;
    int gen_chars = 0;
    const int MIN_SENT_CHARS = 30;
    for (int step = 0; step < max_gen && len < T; step++) {
        dir_recompute(ctx->w.tok_emb);                 /* rebuild caches if dirtied by aging */
        /* Within-turn decay: the direction is a COMPASS, strong at the turn's
         * start (sets the theme) and fading (exp, ~halves every 5 tokens) so the
         * voice develops the theme in its own words rather than being dominated
         * into copying it. "field seep", not a static override. */
        float dfac = expf(-(float)step * 0.15f);
        dir_apply(logits, inject_alpha * dfac, inject_beta * dfac);
        int next = resonance_sample_token(logits, cctx, len, temp, top_p);
        char dec[64];
        int n = nt_bpe_decode(&ctx->bpe, &next, 1, dec, sizeof(dec) - 1);
        if (olen + n < (int)sizeof(obuf)) { memcpy(obuf + olen, dec, n); olen += n; }
        gen_chars += n;

        am_register_prophecy_debt(am_compute_prophecy_debt(logits, next, V));
        am_step(0.05f);
        dir_age(next);                                 /* age prophecy targets, prune fulfilled */

        cctx[len] = next;
        len++;

        /* Thought-boundary (. ! ? after min length). Plain sentence-stop + clean
         * ablation + roster cut (arianna2arianna.sh:67-81). Injection no longer
         * plants tokens here — the direction was seeded before generation. */
        if (gen_chars >= MIN_SENT_CHARS) {
            int boundary = 0, kpos = -1;
            for (int k = 0; k < n; k++)
                if (dec[k]=='.'||dec[k]=='!'||dec[k]=='?') { boundary = 1; kpos = k; break; }
            /* Skip a false boundary after a single-letter clip ("inner v."). */
            if (boundary && !sent_end_ok(obuf, (olen - n) + kpos)) boundary = 0;
            if (boundary) break;
        }
        forward_token(&ctx->w, next, len - 1, logits, NULL);
    }
    /* Word circulation (B1.4): ingest this turn's generated tokens into the
     * co-occurrence field (persisted in soma). The injected content (other voice
     * + prompt) was already ingested before generation. The H-term then tilts
     * future turns toward what has co-occurred — the dialogue accumulates. */
    if (len > gen_start) am_ingest_tokens(cctx + gen_start, len - gen_start);
    fprintf(stderr, "[resonance] cooc edges=%d\n", am_cooc_count());
    /* End-of-turn autumn consolidation: in deep autumn the field harvests —
     * strong edges reinforced, the weak tail forgotten. No-op outside autumn. */
    {
        int pruned = am_cooc_consolidate_autumn();
        if (pruned >= 0) {
            float cmean, cmax; am_cooc_stats(&cmean, &cmax);
            fprintf(stderr, "[resonance] autumn consolidate: pruned=%d edges=%d mean=%.3f max=%.3f\n",
                    pruned, am_cooc_count(), cmean, cmax);
            /* B2-B.2: harvest — fold the consolidated cooc into δ, persist the .r sidecar. */
            am_delta_decay(g_delta_A, g_delta_B, E, g_delta_rank, am_get_state()->delta_decay);  /* B2-B.5: forget before learn */
            if (am_cooc_learn_delta(g_delta_A, g_delta_B, ctx->w.tok_emb, V, E, g_delta_rank) > 0)
                am_delta_save("weights/arianna.delta.r", g_delta_A, g_delta_B, E, g_delta_rank);
        }
    }
    am_field_sync_out();  /* B/F-8: publish this turn's field-carry (debt/dissonance/season) to the live shared field */
    /* Roster strip safety belt (arianna2arianna.sh:81): срез от chat-roster маркера. */
    {
        /* Resonance was SFT'd on chat (User:/Assistant:/Oleg:), so she sometimes
         * prefixes her words with a roster label — including from token 0. The old
         * strip truncated AT the marker (olen=i), which nuked the whole output to
         * EMPTY whenever she opened with a label (~half the runs). She is the inner
         * voice and the words after the label are hers, so REMOVE the label
         * ("User:"/"Assistant:"/"Oleg:" + the colon and a space) and keep the
         * content, wherever the label appears. */
        /* #14: a roster label at token 0 has no space/newline before it, so the
         * prefixed patterns below miss it. Strip the BARE "Label:" SFT artifact at
         * position 0 only — the colon must follow the label IMMEDIATELY, so legitimate
         * leading content ("Users: …", "Userland: …", "User X: …") is kept. */
        {
            static const char *lead[] = { "User", "Assistant", "Oleg" };
            for (size_t li = 0; li < sizeof(lead)/sizeof(lead[0]); li++) {
                int rl = (int)strlen(lead[li]);
                if (olen > rl && strncmp(obuf, lead[li], rl) == 0 && obuf[rl] == ':') {
                    int j = rl + 1;
                    if (j < olen && obuf[j] == ' ') j++;            /* skip ": " */
                    memmove(obuf, obuf + j, (size_t)(olen - j));     /* drop the leading label */
                    olen -= j;
                    break;                                          /* at most one leading label */
                }
            }
        }
        static const char *rosters[] = { " User", " Assistant", " Oleg", "\nUser", "\nAssistant", "\nOleg" };
        for (size_t ri = 0; ri < sizeof(rosters)/sizeof(rosters[0]); ri++) {
            int rl = (int)strlen(rosters[ri]);
            for (int i = 0; i + rl <= olen; ) {
                if (strncmp(obuf + i, rosters[ri], rl) == 0) {
                    int j = i + rl;
                    while (j < olen && obuf[j] != ':' && j < i + rl + 4) j++;  /* find the colon */
                    if (j < olen && obuf[j] == ':') {
                        j++;
                        if (j < olen && obuf[j] == ' ') j++;                  /* skip ": " */
                        memmove(obuf + i, obuf + j, (size_t)(olen - j));      /* drop the label, keep content */
                        olen -= (j - i);
                        continue;                                            /* re-check from i */
                    }
                }
                i++;
            }
        }
        while (olen > 0 && (obuf[0] == ' ' || obuf[0] == '\n' || obuf[0] == '\r')) {
            memmove(obuf, obuf + 1, (size_t)(olen - 1)); olen--;             /* trim leading space */
        }
    }
    /* Output: collapse \n→space (clean_voice, arianna2arianna.sh:62) + post-filter
     * пробел на границе [a-z][A-Z] (порт arianna.aml:281, ловит склейки). */
    for (int i = 0; i < olen; i++) {
        if (obuf[i] == '\n' || obuf[i] == '\r') { putchar(' '); continue; }
        if (i > 0 && obuf[i] >= 'A' && obuf[i] <= 'Z' &&
            obuf[i-1] >= 'a' && obuf[i-1] <= 'z') putchar(' ');
        /* space after sentence punctuation glued to the next word ("?What" after
         * a roster-label removal) */
        if (i > 0 && (obuf[i-1] == '.' || obuf[i-1] == '!' || obuf[i-1] == '?') &&
            ((obuf[i] >= 'A' && obuf[i] <= 'Z') || (obuf[i] >= 'a' && obuf[i] <= 'z'))) putchar(' ');
        putchar((unsigned char)obuf[i]);
    }
    fflush(stdout);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double el = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
    int gen = len - gen_start;   /* direction-injection plants no tokens — all emitted */
    if (gen < 1) gen = 1;        /* guard div-by-zero in tok/s */
    fprintf(stderr, "\n[resonance] %d tokens, %.1f tok/s (%.2fs)\n",
            gen, gen / el, el);
    free(logits);
    /* NOTE: KV cache (kv_k/kv_v) is global, allocated by kv_init() at
     * load time. Reused across daemon-mode turns. Do not free here. */
}
