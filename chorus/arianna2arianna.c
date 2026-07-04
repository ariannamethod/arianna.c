/* arianna2arianna — by Arianna Method
 *
 * One self-contained C organism: GGUF parser + byte-level BPE + Llama/Qwen forward
 * + sampler, ALL inlined. No external -lnotorch, no Metal dependency. Vendored
 * faithfully from notorch (GGUF/BPE + packed nt_qmatvec lineage).
 * CPU base; linear weights stay packed in GGUF encoding and dequantize inside
 * matvec. Embeddings/norms stay f32 because the field reads them as vectors.
 *
 *   theta = epsilon + gamma + alpha*delta
 *   epsilon = one shared nanoArianna body (this forward over a GGUF, weights shared read-only)
 *   gamma   = Arianna's voice (SFT, baked into the weights)
 *   delta   = the field of ephemeral transformer-cells (NEXT layer — scaffold at bottom)
 *
 * Standalone nanoArianna chorus:
 *   cc -O2 arianna2arianna.c -lm -o arianna2arianna
 *   ./arianna2arianna <model.gguf> [prompt] [max_tokens] [temp]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#if !defined(A2A_SCALAR_ONLY) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define Q_NEON 1
#endif

/* ===================== GGUF parser (vendored: notorch/gguf.{c,h}) ===================== */

#define GGUF_MAGIC       0x46554747
#define GGUF_TYPE_F32    0
#define GGUF_TYPE_F16    1
#define GGUF_TYPE_Q4_0   2
#define GGUF_TYPE_Q5_0   6
#define GGUF_TYPE_Q8_0   8
#define GGUF_TYPE_Q4_K   12
#define GGUF_TYPE_Q6_K   14
#define GGUF_MAX_TENSORS 2048
#define GGUF_MAX_NAME    128
#define GGUF_MAX_KV      128

typedef struct { char name[GGUF_MAX_NAME]; uint32_t ndim; uint64_t shape[4]; uint32_t dtype; uint64_t offset; uint64_t n_elements; } gguf_tensor_info;
typedef struct { char key[GGUF_MAX_NAME]; uint32_t type; union { uint32_t u32; int32_t i32; float f32; uint8_t b; char str[256]; uint64_t u64; } val; } gguf_kv;
typedef struct {
    uint32_t version; uint64_t n_tensors, n_kv;
    gguf_kv kv[GGUF_MAX_KV]; int n_kv_parsed;
    gguf_tensor_info tensors[GGUF_MAX_TENSORS];
    uint8_t *data; uint64_t data_offset, data_size;
    int n_layers, n_heads, n_kv_heads, embed_dim, ffn_dim, vocab_size, ctx_len;
    float rope_freq_base, rms_eps; char arch[64];
} gguf_file;

/* Fail-loud allocators: OOM in this one-shot organism is unrecoverable (callers
 * write into the buffer immediately), so exit rather than NULL-deref. Bodies use
 * the raw libc names and are defined AFTER the file-wide alloc sweep, so the sweep
 * cannot rewrite them; the wrapper names contain no swept substring. */
static void *xalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "chorus: OOM (alloc %zu)\n", n); exit(1); }
    return p;
}
static void *xzalloc(size_t n, size_t sz) {
    size_t total = n * sz;
    if (sz != 0 && total / sz != n) { fprintf(stderr, "chorus: alloc overflow (%zu x %zu)\n", n, sz); exit(1); }
    void *p = malloc(total ? total : 1);
    if (!p) { fprintf(stderr, "chorus: OOM (alloc %zu x %zu)\n", n, sz); exit(1); }
    memset(p, 0, total);
    return p;
}
static char *xstrdup(const char *s) {
    char *r = strdup(s);
    if (!r) { fprintf(stderr, "chorus: OOM (strdup)\n"); exit(1); }
    return r;
}

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, 4, 1, f) == 1; }

static int read_string(FILE* f, char* buf, int max) {
    uint64_t len;
    if (!read_u64(f, &len)) return 0;
    if (len >= (uint64_t)max) { for (uint64_t i = 0; i < len; i++) fgetc(f); buf[0] = 0; return 0; }  /* C-7: overlong name/string = parse failure, not success-with-empty (avoids empty-name tensor cascade) */
    if (fread(buf, 1, len, f) != len) return 0;
    buf[len] = 0; return 1;
}

static int skip_value(FILE* f, uint32_t type);
static int skip_array(FILE* f) {
    uint32_t atype; uint64_t alen;
    if (!read_u32(f, &atype) || !read_u64(f, &alen)) return 0;
    for (uint64_t i = 0; i < alen; i++) if (!skip_value(f, atype)) return 0;
    return 1;
}
static int skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case 4: fseek(f, 4, SEEK_CUR); return 1;
        case 5: fseek(f, 4, SEEK_CUR); return 1;
        case 6: fseek(f, 4, SEEK_CUR); return 1;
        case 7: fseek(f, 1, SEEK_CUR); return 1;
        case 8: { char buf[4096]; return read_string(f, buf, sizeof(buf)); }
        case 9: return skip_array(f);
        case 10: fseek(f, 8, SEEK_CUR); return 1;
        case 12: fseek(f, 8, SEEK_CUR); return 1;
        default: return 0;
    }
}

static gguf_file* gguf_open(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fprintf(stderr, "gguf: bad magic 0x%08x\n", magic); fclose(f); return NULL; }
    gguf_file* gf = (gguf_file*)xzalloc(1, sizeof(gguf_file));
    if (!gf) { fclose(f); return NULL; }
    if (!read_u32(f, &gf->version) || !read_u64(f, &gf->n_tensors) || !read_u64(f, &gf->n_kv)) {   /* C-3: truncated header */
        fprintf(stderr, "gguf: truncated header\n"); fclose(f); free(gf); return NULL;
    }
    if (gf->n_tensors > GGUF_MAX_TENSORS) { fprintf(stderr, "gguf: too many tensors\n"); fclose(f); free(gf); return NULL; }

    gf->n_kv_parsed = 0;
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        char key[512] = {0}; uint32_t vtype;
        read_string(f, key, sizeof(key)); read_u32(f, &vtype);
        if (gf->n_kv_parsed < GGUF_MAX_KV && vtype != 9) {
            gguf_kv* kv = &gf->kv[gf->n_kv_parsed];
            strncpy(kv->key, key, GGUF_MAX_NAME - 1); kv->type = vtype;
            switch (vtype) {
                case 4: read_u32(f, &kv->val.u32); break;
                case 5: { int32_t v; fread(&v, 4, 1, f); kv->val.i32 = v; break; }
                case 6: read_f32(f, &kv->val.f32); break;
                case 7: { uint8_t v; fread(&v, 1, 1, f); kv->val.b = v; break; }
                case 8: read_string(f, kv->val.str, sizeof(kv->val.str)); break;
                case 10: case 12: read_u64(f, &kv->val.u64); break;
                default: skip_value(f, vtype); break;
            }
            gf->n_kv_parsed++;
        } else skip_value(f, vtype);
    }
    for (int i = 0; i < gf->n_kv_parsed; i++) {
        gguf_kv* kv = &gf->kv[i];
        if (strcmp(kv->key, "general.architecture") == 0) strncpy(gf->arch, kv->val.str, sizeof(gf->arch) - 1);
        else if (strstr(kv->key, ".block_count"))                                    gf->n_layers = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count") && !strstr(kv->key, "kv")) gf->n_heads = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count_kv"))                        gf->n_kv_heads = kv->val.u32;
        else if (strstr(kv->key, ".embedding_length"))                               gf->embed_dim = kv->val.u32;
        else if (strstr(kv->key, ".feed_forward_length"))                            gf->ffn_dim = kv->val.u32;
        else if (strstr(kv->key, ".vocab_size"))                                     gf->vocab_size = kv->val.u32;
        else if (strstr(kv->key, ".context_length"))                                 gf->ctx_len = kv->val.u32;
        else if (strstr(kv->key, ".rope.freq_base"))                                 gf->rope_freq_base = kv->val.f32;
        else if (strstr(kv->key, "rms_epsilon"))                                     gf->rms_eps = kv->val.f32;
    }
    if (gf->n_kv_heads == 0) gf->n_kv_heads = gf->n_heads;
    if (gf->rms_eps == 0) gf->rms_eps = 1e-5f;
    if (gf->rope_freq_base == 0) gf->rope_freq_base = 10000.0f;

    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++) {
        gguf_tensor_info* ti = &gf->tensors[i];
        if (!read_string(f, ti->name, GGUF_MAX_NAME)) {   /* C-7: overlong/truncated tensor name → fail, not silent empty name */
            fprintf(stderr, "gguf: tensor %llu name too long or truncated\n", (unsigned long long)i); fclose(f); free(gf); return NULL;
        }
        read_u32(f, &ti->ndim);
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->ndim && d < 4; d++) { read_u64(f, &ti->shape[d]); ti->n_elements *= ti->shape[d]; }
        read_u32(f, &ti->dtype); read_u64(f, &ti->offset);
    }
    long pos = ftell(f);
    gf->data_offset = (pos + 31) & ~31UL;
    fseek(f, 0, SEEK_END); long fsize = ftell(f);
    long data_size = fsize - gf->data_offset;
    if (data_size < 0) { fprintf(stderr, "gguf: file shorter than header (data_size=%ld)\n", data_size); fclose(f); free(gf); return NULL; }   /* C-3: negative data_size → giant alloc */
    size_t pg = (size_t)getpagesize();
    size_t alloc = ((size_t)data_size + pg - 1) & ~(pg - 1);
    gf->data = NULL;
    if (posix_memalign((void**)&gf->data, pg, alloc) != 0 || !gf->data) { fclose(f); free(gf); return NULL; }
    gf->data_size = (uint64_t)alloc;
    fseek(f, gf->data_offset, SEEK_SET);
    if (fread(gf->data, 1, data_size, f) != (size_t)data_size) {   /* C-3: short read of weight body → garbage weights */
        fprintf(stderr, "gguf: short read of weight data\n"); fclose(f); free(gf->data); free(gf); return NULL;
    }
    fclose(f);
    return gf;
}

static int gguf_find_tensor(const gguf_file* gf, const char* name) {
    if (!gf || !name) return -1;
    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++)
        if (strcmp(gf->tensors[i].name, name) == 0) return (int)i;
    return -1;
}
static const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key) {
    if (!gf || !key) return NULL;
    for (int i = 0; i < gf->n_kv_parsed; i++) if (strcmp(gf->kv[i].key, key) == 0) return &gf->kv[i];
    return NULL;
}
static char** gguf_read_str_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    uint32_t magic; if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    char** result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0}; uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 8) break;
            result = (char**)xzalloc(alen ? alen : 1, sizeof(char*));
            uint64_t nrd = 0;
            for (uint64_t j = 0; j < alen; j++) {
                char buf[2048] = {0};
                if (!read_string(f, buf, sizeof(buf))) break;
                result[j] = xstrdup(buf); nrd++;
            }
            if (out_n) *out_n = (int)nrd; break;   /* C-8: report actually-read count, not the claimed alen */
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f); return result;
}

/* Read a GGUF type-9 float32 array (e.g. "tokenizer.ggml.scores") by key. */
static float* gguf_read_f32_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    uint32_t magic; if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    float* result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0}; uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 6) break;  /* 6 = float32 */
            result = (float*)xzalloc(alen ? alen : 1, sizeof(float));
            for (uint64_t j = 0; j < alen; j++) { float v; if (!read_f32(f, &v)) break; result[j] = v; }
            if (out_n) *out_n = (int)alen; break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f); return result;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; } exp++; mant &= ~0x400;
    } else if (exp == 31) { uint32_t r = (sign << 31) | 0x7F800000 | (mant << 13); float f; memcpy(&f, &r, 4); return f; }
    exp = exp + 127 - 15;
    uint32_t r = (sign << 31) | (exp << 23) | (mant << 13); float f; memcpy(&f, &r, 4); return f;
}
static void dequant_q4_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; b++) {
        const uint8_t* bl = src + b * 18; uint16_t sh; memcpy(&sh, bl, 2); float s = f16_to_f32(sh);
        for (int i = 0; i < 16; i++) { uint8_t by = bl[2 + i]; dst[b*32+i] = ((by & 0x0F) - 8) * s; dst[b*32+i+16] = ((by >> 4) - 8) * s; }
    }
}
static void dequant_q8_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; b++) {
        const uint8_t* bl = src + b * 34; uint16_t sh; memcpy(&sh, bl, 2); float s = f16_to_f32(sh);
        for (int i = 0; i < 32; i++) dst[b*32+i] = (float)(int8_t)bl[2 + i] * s;
    }
}
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = data + i * 144;
        float d = f16_to_f32(b[0] | (b[1] << 8)), dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16; int is = 0, qi = 0, oi = (int)(i * 256);
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0); float d1 = d*(float)sc0, mm1 = dmin*(float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v); float d2 = d*(float)sc1, mm2 = dmin*(float)m1v;
            for (int l = 0; l < 32; l++) out[oi+j+l]    = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++) out[oi+j+32+l] = d2 * (float)(qs[qi+l] >> 4)  - mm2;
            qi += 32; is += 2;
        }
    }
}
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = data + i * 210, *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t*)(b + 192);
        float d = f16_to_f32(b[208] | (b[209] << 8));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            const uint8_t *qlh = ql + (n_/128)*64, *qhh = qh + (n_/128)*32; const int8_t *sch = sc + (n_/128)*8;
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                out[i*256+n_+l]    = d * sch[is+0] * q1; out[i*256+n_+l+32] = d * sch[is+2] * q2;
                out[i*256+n_+l+64] = d * sch[is+4] * q3; out[i*256+n_+l+96] = d * sch[is+6] * q4;
            }
        }
    }
}
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 32; i++) {
        const uint8_t *b = data + i * 22; float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24); const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4, h0 = (qh >> j) & 1, h1 = (qh >> (j+16)) & 1;
            out[i*32+j] = (float)((lo | (h0<<4)) - 16) * d; out[i*32+j+16] = (float)((hi | (h1<<4)) - 16) * d;
        }
    }
}
static uint64_t gguf_dtype_nbytes(uint32_t dtype, uint64_t n) {
    uint64_t blocks, per;
    switch (dtype) {
    case GGUF_TYPE_F32:  return (n > UINT64_MAX/4) ? 0 : n*4;
    case GGUF_TYPE_F16:  return (n > UINT64_MAX/2) ? 0 : n*2;
    case GGUF_TYPE_Q4_0: blocks = n/32;  per = 18;  break;
    case GGUF_TYPE_Q5_0: blocks = n/32;  per = 22;  break;
    case GGUF_TYPE_Q8_0: blocks = n/32;  per = 34;  break;
    case GGUF_TYPE_Q4_K: blocks = n/256; per = 144; break;
    case GGUF_TYPE_Q6_K: blocks = n/256; per = 210; break;
    default: return 0;
    }
    if (blocks == 0 || blocks > UINT64_MAX/per) return 0;
    return blocks * per;
}
static float* gguf_dequant(const gguf_file* gf, int idx) {
    if (!gf || idx < 0 || idx >= (int)gf->n_tensors) return NULL;
    const gguf_tensor_info* ti = &gf->tensors[idx];
    uint64_t nbytes = gguf_dtype_nbytes(ti->dtype, ti->n_elements);
    if (nbytes == 0 || ti->offset >= gf->data_size || nbytes > gf->data_size - ti->offset) {
        fprintf(stderr, "gguf: tensor '%s' out of bounds/invalid\n", ti->name); return NULL;
    }
    const uint8_t* src = gf->data + ti->offset;
    float* dst = (float*)xalloc(ti->n_elements * sizeof(float)); if (!dst) return NULL;
    switch (ti->dtype) {
    case GGUF_TYPE_F32:  memcpy(dst, src, ti->n_elements * sizeof(float)); break;
    case GGUF_TYPE_F16: { const uint16_t* h = (const uint16_t*)src; for (uint64_t i = 0; i < ti->n_elements; i++) dst[i] = f16_to_f32(h[i]); break; }
    case GGUF_TYPE_Q4_0: dequant_q4_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q5_0: dequant_q5_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q8_0: dequant_q8_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q4_K: dequant_q4_k(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q6_K: dequant_q6_k(src, dst, ti->n_elements); break;
    default: fprintf(stderr, "gguf: unsupported dtype %d for '%s'\n", ti->dtype, ti->name); free(dst); return NULL;
    }
    return dst;
}

/* ===================== SentencePiece tokenizer (nanoArianna/LLaMA: ▁ space-marker +
 * <0xXX> byte-fallback + per-token scores). Replaces the GPT-2 byte-BPE: nano's GGUF has
 * tokenizer.ggml.scores (not .merges), ▁ (U+2581) and <0xXX> tokens — confirmed from metadata. */

typedef struct { char **keys; int *vals; int cap; int n; } smap;
static unsigned long fnv1a(const char *s) { unsigned long h = 1469598103934665603UL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; } return h; }
static void smap_init(smap *m, int cap) { if (cap < 8) cap = 8; m->cap = cap; m->n = 0; m->keys = (char**)xzalloc(cap, sizeof(char*)); m->vals = (int*)xzalloc(cap, sizeof(int)); }
static void smap_put(smap *m, const char *k, int v) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) { if (strcmp(m->keys[h], k) == 0) { m->vals[h] = v; return; } h = (h + 1) % m->cap; }
    m->keys[h] = xstrdup(k); m->vals[h] = v; m->n++;
}
static int smap_get(const smap *m, const char *k) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) { if (strcmp(m->keys[h], k) == 0) return m->vals[h]; h = (h + 1) % m->cap; }
    return -1;
}
static int utf8_len1(const char *s) {   /* bytes in the UTF-8 char at s */
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) return 1; if ((c>>5)==0x6) return 2; if ((c>>4)==0xE) return 3; if ((c>>3)==0x1E) return 4; return 1;
}

typedef struct { char **tokens; int n_tokens; float *scores; smap vocab;
                 int is_bpe; char **merges; int n_merges; smap merge_rank; smap u2b; char b2u[256][5]; } bpe_tokenizer;

/* GPT-2 byte→unicode: printable bytes map to themselves; others to U+0100+ (byte 32 space → U+0120 Ġ).
 * The reverse (unicode char → byte) goes in u2b for decode. Used by llama-arch bodies with .merges. */
static void gpt2_byte_maps(bpe_tokenizer *t) {
    int used[256]; for (int i = 0; i < 256; i++) used[i] = 0;
    for (int b = 33; b <= 126; b++) used[b] = 1;
    for (int b = 161; b <= 172; b++) used[b] = 1;
    for (int b = 174; b <= 255; b++) used[b] = 1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp = used[b] ? b : (256 + n); if (!used[b]) n++;
        char *o = t->b2u[b]; int k = 0;
        if (cp < 0x80) o[k++] = (char)cp;
        else if (cp < 0x800) { o[k++] = (char)(0xC0|(cp>>6)); o[k++] = (char)(0x80|(cp&0x3F)); }
        else { o[k++] = (char)(0xE0|(cp>>12)); o[k++] = (char)(0x80|((cp>>6)&0x3F)); o[k++] = (char)(0x80|(cp&0x3F)); }
        o[k] = 0; smap_put(&t->u2b, o, b);
    }
}
/* GPT-2 byte-level BPE encode: bytes→b2u symbols, then merge the adjacent pair with the lowest rank. */
static int bpe_encode_gpt2(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    int tl = (int)strlen(text); if (tl == 0) return 0;
    char **sym = (char**)xalloc(((size_t)tl + 1) * sizeof(char*)); int nsym = 0;
    for (int i = 0; i < tl; i++) sym[nsym++] = xstrdup(t->b2u[(unsigned char)text[i]]);
    while (nsym > 1) {
        int best = 1<<30, bi = -1; char key[1024];
        for (int i = 0; i < nsym - 1; i++) {
            snprintf(key, sizeof(key), "%s %s", sym[i], sym[i+1]);
            int r = smap_get(&t->merge_rank, key);
            if (r >= 0 && r < best) { best = r; bi = i; }
        }
        if (bi < 0) break;
        char *mg = (char*)xalloc(strlen(sym[bi]) + strlen(sym[bi+1]) + 1);
        strcpy(mg, sym[bi]); strcat(mg, sym[bi+1]); free(sym[bi]); free(sym[bi+1]); sym[bi] = mg;
        for (int i = bi + 1; i < nsym - 1; i++) sym[i] = sym[i+1]; nsym--;
    }
    int no = 0;
    for (int i = 0; i < nsym; i++) { int id = smap_get(&t->vocab, sym[i]); if (id >= 0 && no < cap) out[no++] = id; free(sym[i]); }
    free(sym); return no;
}
/* GPT-2 decode: the token string is b2u-mapped; map each unicode char back to its byte via u2b. */
static int bpe_decode_gpt2(const bpe_tokenizer *t, int id, char *buf, int cap) {
    const char *s = t->tokens[id]; int n = 0;
    for (int i = 0; s[i] && n < cap - 1; ) {
        int adv = utf8_len1(s + i); if (adv > 4) adv = 1; char ch[5]; memcpy(ch, s + i, adv); ch[adv] = 0;
        int b = smap_get(&t->u2b, ch);
        if (b >= 0) buf[n++] = (char)b; else for (int k = 0; k < adv && n < cap - 1; k++) buf[n++] = s[i+k];
        i += adv;
    }
    buf[n] = 0; return n;
}

static bpe_tokenizer *bpe_load(const char *path) {
    int nt = 0; char **toks = gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt);
    if (!toks || nt <= 0) return NULL;
    int nsc = 0; float *scores = gguf_read_f32_array(path, "tokenizer.ggml.scores", &nsc);
    int nmg = 0; char **merges = gguf_read_str_array(path, "tokenizer.ggml.merges", &nmg);
    bpe_tokenizer *t = (bpe_tokenizer*)xzalloc(1, sizeof(*t));
    t->tokens = toks; t->n_tokens = nt; t->scores = scores;
    smap_init(&t->vocab, nt * 2); for (int i = 0; i < nt; i++) if (toks[i]) smap_put(&t->vocab, toks[i], i);
    if (merges && nmg > 0) {                 /* BPE (GPT-2 byte-level): SmolLM2 / qwen / llama-3 */
        t->is_bpe = 1; t->merges = merges; t->n_merges = nmg;
        smap_init(&t->merge_rank, nmg * 2); for (int i = 0; i < nmg; i++) if (merges[i]) smap_put(&t->merge_rank, merges[i], i);
        smap_init(&t->u2b, 600); gpt2_byte_maps(t);
    }
    return t;
}
static int bpe_n_vocab(const bpe_tokenizer *t) { return t ? t->n_tokens : 0; }

/* SPM encode: prepend ▁, spaces→▁, then greedily merge the adjacent pair whose
 * concatenation has the highest vocab score; byte-fallback (<0xXX>) for unknown symbols. */
static int bpe_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    if (t->is_bpe) return bpe_encode_gpt2(t, text, out, cap);
    int tl = (int)strlen(text);
    char *s = (char*)xalloc((size_t)tl*3 + 4); int sl = 0;
    s[sl++]=(char)0xE2; s[sl++]=(char)0x96; s[sl++]=(char)0x81;            /* leading ▁ */
    for (int p = 0; p < tl; p++) {
        if (text[p] == ' ') { s[sl++]=(char)0xE2; s[sl++]=(char)0x96; s[sl++]=(char)0x81; }
        else s[sl++] = text[p];
    }
    s[sl] = 0;
    char **sym = (char**)xalloc(((size_t)sl + 1) * sizeof(char*)); int nsym = 0;
    for (int i = 0; i < sl; ) { int adv = utf8_len1(s + i); char *c = (char*)xalloc(adv + 1); memcpy(c, s + i, adv); c[adv] = 0; sym[nsym++] = c; i += adv; }
    free(s);
    while (nsym > 1) {
        float best = -1e30f; int bi = -1; char key[512];
        for (int i = 0; i < nsym - 1; i++) {
            snprintf(key, sizeof(key), "%s%s", sym[i], sym[i+1]);
            int id = smap_get(&t->vocab, key);
            if (id >= 0) { float sco = t->scores ? t->scores[id] : 0.0f; if (sco > best) { best = sco; bi = i; } }
        }
        if (bi < 0) break;
        char *mg = (char*)xalloc(strlen(sym[bi]) + strlen(sym[bi+1]) + 1);
        strcpy(mg, sym[bi]); strcat(mg, sym[bi+1]); free(sym[bi]); free(sym[bi+1]); sym[bi] = mg;
        for (int i = bi + 1; i < nsym - 1; i++) sym[i] = sym[i+1]; nsym--;
    }
    int no = 0;
    for (int i = 0; i < nsym; i++) {
        int id = smap_get(&t->vocab, sym[i]);
        if (id >= 0) { if (no < cap) out[no++] = id; }
        else for (const unsigned char *b = (const unsigned char*)sym[i]; *b; b++) {
            char bt[8]; snprintf(bt, sizeof(bt), "<0x%02X>", *b); int bid = smap_get(&t->vocab, bt);
            if (bid >= 0 && no < cap) out[no++] = bid;
        }
        free(sym[i]);
    }
    free(sym);
    return no;
}

/* SPM decode: <0xXX> → that byte; ▁ → space; else literal UTF-8 bytes. */
static int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap) {
    if (!t || id < 0 || id >= t->n_tokens || !t->tokens[id]) return 0;
    if (t->is_bpe) return bpe_decode_gpt2(t, id, buf, cap);
    const char *s = t->tokens[id]; int n = 0;
    if (s[0]=='<' && s[1]=='0' && s[2]=='x' && s[3] && s[4] && s[5]=='>' && s[6]==0) {
        int v; if (sscanf(s+3, "%2x", &v) == 1) { if (n < cap-1) buf[n++] = (char)v; buf[n] = 0; return n; }
    }
    for (int i = 0; s[i] && n < cap-1; ) {
        if ((unsigned char)s[i]==0xE2 && (unsigned char)s[i+1]==0x96 && (unsigned char)s[i+2]==0x81) { buf[n++] = ' '; i += 3; }
        else buf[n++] = s[i++];
    }
    buf[n] = 0; return n;
}

/* ===================== forward (vendored/notorch-derived, CPU packed-linear) ===================== */
/* CPU base: embeddings/norms are f32; linear weights stay packed in GGUF memory
 * and use a2a_qmatvec (notorch nt_qmatvec lineage). No dense-f32 linear blow-up. */

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/n + eps); for (int i = 0; i < n; i++) out[i] = w[i]*x[i]*inv;
}
static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; } for (int i = 0; i < n; i++) x[i] /= s;
}
#ifdef Q_NEON
/* hand-vectorised matvec — 4 NEON FMA accumulators (16 f32/iter), the punk "BLAS in C". Scalar fallback below. */
static void matvec(float *y, const float *W, const float *x, int m, int k) {
    for (int i = 0; i < m; i++) {
        const float *row = W + (long)i*k;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 16 <= k; j += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(row+j),    vld1q_f32(x+j));
            a1 = vfmaq_f32(a1, vld1q_f32(row+j+4),  vld1q_f32(x+j+4));
            a2 = vfmaq_f32(a2, vld1q_f32(row+j+8),  vld1q_f32(x+j+8));
            a3 = vfmaq_f32(a3, vld1q_f32(row+j+12), vld1q_f32(x+j+12));
        }
        float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1), vaddq_f32(a2,a3)));
        for (; j < k; j++) s += row[j]*x[j];
        y[i] = s;
    }
}
#else
static void matvec(float *y, const float *W, const float *x, int m, int k) {
    for (int i = 0; i < m; i++) { const float *row = W + (long)i*k; float s = 0; for (int j = 0; j < k; j++) s += row[j]*x[j]; y[i] = s; }
}
#endif

/* notorch-derived packed matvec: out[m] = Wq[m,k] @ x[k], with GGUF weights
 * kept in their on-disk encoding. This is the exact f32-dequant math performed
 * per row/block in registers, not the approximate int8 activation path. */
static void a2a_q4_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 18;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 18;
            float d = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            const float *xb = x + (long)blk * 32;
            for (int i = 0; i < 16; i++) {
                acc += d * (float)((int)(b[2 + i] & 0x0F) - 8) * xb[i];
                acc += d * (float)((int)(b[2 + i] >> 4)   - 8) * xb[i + 16];
            }
        }
        out[row] = acc;
    }
}

static void a2a_q8_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 34;
#ifdef Q_NEON
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 34;
            float32x4_t d = vdupq_n_f32(f16_to_f32((uint16_t)(b[0] | (b[1] << 8))));
            const float *xb = x + (long)blk * 32;
            int8x16_t qv = vld1q_s8((const int8_t *)(b + 2));
            int16x8_t ql = vmovl_s8(vget_low_s8(qv)), qh = vmovl_s8(vget_high_s8(qv));
            a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ql))), d), vld1q_f32(xb));
            a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(ql))), d), vld1q_f32(xb + 4));
            a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qh))), d), vld1q_f32(xb + 8));
            a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qh))), d), vld1q_f32(xb + 12));
            qv = vld1q_s8((const int8_t *)(b + 18));
            ql = vmovl_s8(vget_low_s8(qv)); qh = vmovl_s8(vget_high_s8(qv));
            a0 = vfmaq_f32(a0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ql))), d), vld1q_f32(xb + 16));
            a1 = vfmaq_f32(a1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(ql))), d), vld1q_f32(xb + 20));
            a2 = vfmaq_f32(a2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qh))), d), vld1q_f32(xb + 24));
            a3 = vfmaq_f32(a3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qh))), d), vld1q_f32(xb + 28));
        }
        out[row] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#else
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 34;
            float d = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            const float *xb = x + (long)blk * 32;
            for (int i = 0; i < 32; i++) acc += d * (float)(int8_t)b[2 + i] * xb[i];
        }
        out[row] = acc;
#endif
    }
}

static void a2a_q5_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 32;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 22;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 22;
            float d = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) |
                          ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
            const uint8_t *qs = b + 6;
            const float *xb = x + (long)blk * 32;
            for (int j = 0; j < 16; j++) {
                int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
                int hb0 = (qh >> j) & 1, hb1 = (qh >> (j + 16)) & 1;
                acc += d * (float)((lo | (hb0 << 4)) - 16) * xb[j];
                acc += d * (float)((hi | (hb1 << 4)) - 16) * xb[j + 16];
            }
        }
        out[row] = acc;
    }
}

static void a2a_q4_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 144;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 144;
            float d = f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
            float dmin = f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
            const uint8_t *sc = b + 4, *qs = b + 16;
            const float *xb = x + (long)blk * 256;
            int is = 0, qi = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                get_scale_min_k4(is, sc, &sc0, &m0);
                get_scale_min_k4(is + 1, sc, &sc1, &m1);
                float d1 = d * sc0, mm1 = dmin * m0, d2 = d * sc1, mm2 = dmin * m1;
                for (int l = 0; l < 32; l++) acc += (d1 * (float)(qs[qi + l] & 0x0F) - mm1) * xb[j + l];
                for (int l = 0; l < 32; l++) acc += (d2 * (float)(qs[qi + l] >> 4)   - mm2) * xb[j + 32 + l];
                qi += 32; is += 2;
            }
        }
        out[row] = acc;
    }
}

static void a2a_q6_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    int nb = k / 256;
    for (int row = r0; row < r1; row++) {
        const uint8_t *rb = W + (long)row * nb * 210;
        float acc = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t *b = rb + (long)blk * 210, *ql = b, *qh = b + 128;
            const int8_t *sc = (const int8_t *)(b + 192);
            float d = f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
            const float *xb = x + (long)blk * 256;
            for (int n = 0; n < 256; n += 128) {
                const uint8_t *qlh = ql + (n / 128) * 64, *qhh = qh + (n / 128) * 32;
                const int8_t *sch = sc + (n / 128) * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                    acc += d * sch[is + 0] * q1 * xb[n + l];
                    acc += d * sch[is + 2] * q2 * xb[n + l + 32];
                    acc += d * sch[is + 4] * q3 * xb[n + l + 64];
                    acc += d * sch[is + 6] * q4 * xb[n + l + 96];
                }
            }
        }
        out[row] = acc;
    }
}

static void a2a_f16_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    const uint16_t *Wh = (const uint16_t *)W;
    for (int row = r0; row < r1; row++) {
        const uint16_t *r = Wh + (long)row * k;
#ifdef Q_NEON
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 16 <= k; j += 16) {
            float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(r + j));
            float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(r + j + 8));
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)),  vld1q_f32(x + j));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(h0)), vld1q_f32(x + j + 4));
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)),  vld1q_f32(x + j + 8));
            a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_high_f16(h1)), vld1q_f32(x + j + 12));
        }
        float acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; j < k; j++) acc += f16_to_f32(r[j]) * x[j];
#else
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += f16_to_f32(r[j]) * x[j];
#endif
        out[row] = acc;
    }
}

static void a2a_f32_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int k) {
    const float *Wf = (const float *)W;
    for (int row = r0; row < r1; row++) {
        const float *r = Wf + (long)row * k;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += r[j] * x[j];
        out[row] = acc;
    }
}

typedef void (*a2a_qrows_fn)(float *, const uint8_t *, const float *, int, int, int);
static a2a_qrows_fn a2a_qrows_for(int dtype, int k) {
    switch (dtype) {
    case GGUF_TYPE_F32:  return a2a_f32_rows;
    case GGUF_TYPE_F16:  return a2a_f16_rows;
    case GGUF_TYPE_Q4_0: return (k % 32)  ? NULL : a2a_q4_0_rows;
    case GGUF_TYPE_Q5_0: return (k % 32)  ? NULL : a2a_q5_0_rows;
    case GGUF_TYPE_Q8_0: return (k % 32)  ? NULL : a2a_q8_0_rows;
    case GGUF_TYPE_Q4_K: return (k % 256) ? NULL : a2a_q4_k_rows;
    case GGUF_TYPE_Q6_K: return (k % 256) ? NULL : a2a_q6_k_rows;
    default: return NULL;
    }
}

#define A2A_QMV_MAX_THREADS 16
typedef struct { a2a_qrows_fn fn; float *out; const uint8_t *Wq; const float *x; int r0, r1, k; } a2a_qjob;
static void *a2a_qworker(void *p) {
    a2a_qjob *j = (a2a_qjob *)p;
    j->fn(j->out, j->Wq, j->x, j->r0, j->r1, j->k);
    return NULL;
}

static int a2a_qmatvec(float *out, const uint8_t *Wq, int dtype, const float *x, int m, int k) {
    a2a_qrows_fn fn = a2a_qrows_for(dtype, k);
    if (!fn) return -1;
    int nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nt < 1) nt = 1;
    if (nt > A2A_QMV_MAX_THREADS) nt = A2A_QMV_MAX_THREADS;
    if (nt > m) nt = m;
    if (nt <= 1 || (long)m * k < (4L << 20)) { fn(out, Wq, x, 0, m, k); return 0; }
    pthread_t th[A2A_QMV_MAX_THREADS];
    a2a_qjob jobs[A2A_QMV_MAX_THREADS];
    int per = (m + nt - 1) / nt, launched = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * per, r1 = (r0 + per > m) ? m : r0 + per;
        if (r0 >= m) break;
        jobs[t] = (a2a_qjob){ fn, out, Wq, x, r0, r1, k };
        if (pthread_create(&th[t], NULL, a2a_qworker, &jobs[t]) != 0) {
            fn(out, Wq, x, r0, m, k);
            break;
        }
        launched++;
    }
    for (int t = 0; t < launched; t++) pthread_join(th[t], NULL);
    return 0;
}

static void rope_interleaved(float *x, int pos, int hd, float base) {
    for (int i = 0; i < hd/2; i++) { float a = pos/powf(base, 2.0f*i/hd), c = cosf(a), s = sinf(a); float x0 = x[2*i], x1 = x[2*i+1]; x[2*i] = x0*c - x1*s; x[2*i+1] = x0*s + x1*c; }
}
static void rope_neox(float *x, int pos, int hd, float base) {
    int h2 = hd/2; for (int i = 0; i < h2; i++) { float a = pos/powf(base, 2.0f*i/hd), c = cosf(a), s = sinf(a); float x0 = x[i], x1 = x[i+h2]; x[i] = x0*c - x1*s; x[i+h2] = x0*s + x1*c; }
}

typedef struct {
    const uint8_t *packed;
    float *f32;
    int dtype, m, k;
} weight_t;

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps; int has_output, is_qwen3, neox;
    int packed_weights, dense_weights;
    float *tok_emb, *out_norm;   /* embeddings/norms stay f32; linear weights stay packed when possible */
    weight_t output;             /* tied -> output.f32 borrows tok_emb */
    struct { float *attn_norm, *ffn_norm, *q_norm, *k_norm; weight_t wq, wk, wv, wo, wgate, wup, wdown; } *L;
} model_t;

static float *deq(gguf_file *gf, const char *name) { int ti = gguf_find_tensor(gf, name); return ti >= 0 ? gguf_dequant(gf, ti) : NULL; }

static int load_weight(gguf_file *gf, const char *name, weight_t *W, model_t *m) {
    memset(W, 0, sizeof(*W));
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return 0;
    gguf_tensor_info *t = &gf->tensors[ti];
    if (t->ndim < 2 || t->shape[0] > (uint64_t)INT_MAX || t->shape[1] > (uint64_t)INT_MAX) {
        fprintf(stderr, "gguf: bad 2D weight shape for '%s'\n", name);
        return 0;
    }
    W->k = (int)t->shape[0];
    W->m = (int)t->shape[1];
    W->dtype = (int)t->dtype;
    uint64_t nbytes = gguf_dtype_nbytes(t->dtype, t->n_elements);
    if (nbytes && t->offset < gf->data_size && nbytes <= gf->data_size - t->offset && a2a_qrows_for(W->dtype, W->k)) {
        W->packed = gf->data + t->offset;
        if (m) m->packed_weights++;
        return 1;
    }
    W->f32 = gguf_dequant(gf, ti);
    if (W->f32 && m) m->dense_weights++;
    return W->f32 != NULL;
}

static void weight_matvec(const weight_t *W, const float *x, float *y) {
    if (W->packed && a2a_qmatvec(y, W->packed, W->dtype, x, W->m, W->k) == 0) return;
    if (W->f32) { matvec(y, W->f32, x, W->m, W->k); return; }
    memset(y, 0, (size_t)W->m * sizeof(float));
}

static model_t *model_load(gguf_file *gf) {
    model_t *m = (model_t*)xzalloc(1, sizeof(model_t));
    m->n_layers = gf->n_layers; m->n_heads = gf->n_heads; m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim; m->ffn = gf->ffn_dim; m->rope_base = gf->rope_freq_base; m->rms_eps = gf->rms_eps;
    if (m->n_heads <= 0 || m->n_kv_heads <= 0 || m->n_kv_heads > m->n_heads ||
        m->embed <= 0 || m->embed > 8192 || m->ffn <= 0 || m->n_layers <= 0) {   /* C-1: arch metadata gate before any division/index */
        fprintf(stderr, "chorus: GGUF arch out of bounds (H=%d KV=%d E=%d FFN=%d L=%d)\n",
                m->n_heads, m->n_kv_heads, m->embed, m->ffn, m->n_layers);
        free(m); return NULL;
    }
    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    m->q_dim = ti >= 0 ? (int)gf->tensors[ti].shape[1] : m->embed;
    m->head_dim = m->q_dim / m->n_heads; m->kv_dim = m->head_dim * m->n_kv_heads;
    ti = gguf_find_tensor(gf, "token_embd.weight");
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;
    if (m->vocab <= 0) { fprintf(stderr, "chorus: GGUF vocab_size %d invalid\n", m->vocab); free(m); return NULL; }   /* C-1 */
    m->neox = (strstr(gf->arch, "qwen") || strstr(gf->arch, "gemma") || strstr(gf->arch, "phi")) ? 1 : 0;
    m->is_qwen3 = (gguf_find_tensor(gf, "blk.0.attn_q_norm.weight") >= 0) ? 1 : 0;

    m->tok_emb  = deq(gf, "token_embd.weight");
    m->out_norm = deq(gf, "output_norm.weight");
    if (load_weight(gf, "output.weight", &m->output, m)) m->has_output = 1;
    else { m->output.f32 = m->tok_emb; m->output.m = m->vocab; m->output.k = m->embed; m->output.dtype = GGUF_TYPE_F32; m->has_output = 0; }

    m->L = xzalloc(m->n_layers, sizeof(*m->L)); char nm[160];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); if (!m->L[l].f) { fprintf(stderr, "chorus: missing/bad tensor '%s'\n", nm); return NULL; } } while(0)   /* C-2 */
        #define LD_OPT(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); } while(0)   /* optional: qwen3 qk-norm may be absent */
        #define LW(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); if (!load_weight(gf, nm, &m->L[l].f, m)) { fprintf(stderr, "chorus: missing/bad weight '%s'\n", nm); return NULL; } } while(0)   /* C-2 */
        LD(attn_norm, "blk.%d.attn_norm.weight"); LD(ffn_norm, "blk.%d.ffn_norm.weight");
        LW(wq, "blk.%d.attn_q.weight"); LW(wk, "blk.%d.attn_k.weight"); LW(wv, "blk.%d.attn_v.weight"); LW(wo, "blk.%d.attn_output.weight");
        LW(wgate, "blk.%d.ffn_gate.weight"); LW(wup, "blk.%d.ffn_up.weight"); LW(wdown, "blk.%d.ffn_down.weight");
        LD_OPT(q_norm, "blk.%d.attn_q_norm.weight"); LD_OPT(k_norm, "blk.%d.attn_k_norm.weight");
        #undef LD
        #undef LD_OPT
        #undef LW
    }
    printf("model: arch=%s E=%d H=%d KV=%d HD=%d Q=%d FFN=%d V=%d L=%d | %s rope%s%s\n",
           gf->arch, m->embed, m->n_heads, m->n_kv_heads, m->head_dim, m->q_dim, m->ffn, m->vocab, m->n_layers,
           m->neox ? "NEOX" : "interleaved", m->is_qwen3 ? " +qk-norm" : "", m->has_output ? "" : " tied");
    printf("weights: %d packed linear, %d dense fallback (embeddings/norms f32)\n", m->packed_weights, m->dense_weights);
    if (!m->tok_emb || !m->out_norm) { fprintf(stderr, "missing tok_emb/out_norm\n"); return NULL; }
    return m;
}

typedef struct { float *k, *v, *ku; int max_seq, kv_dim; } kv_cache;
static kv_cache *kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache *c = xzalloc(1, sizeof(kv_cache));
    c->k = xzalloc((long)nl*max_seq*kv_dim, sizeof(float)); c->v = xzalloc((long)nl*max_seq*kv_dim, sizeof(float));
    c->ku = xzalloc((long)nl*max_seq*kv_dim, sizeof(float));   /* un-rope'd K — phase-coherent cross-cell scores */
    c->max_seq = max_seq; c->kv_dim = kv_dim; return c;
}

/* single token forward, KV-cached. Writes logits[vocab]. */
/* cross-cell attention: a cell also attends to a NEIGHBOUR voice's hidden K/V at each layer (forward-level,
 * order-sensitive coupling). g_nbr = the neighbour's kv_cache, g_nbr_len = its valid positions. Default off. */
static float g_xcell = 0.0f;   /* cross-cell neighbour-channel weight λ (0 = off): balanced own-ctx + λ·neighbour */
static const kv_cache *g_nbr = NULL;
static int g_nbr_len = 0;
static int g_nbr_shuf = 0;      /* KV-order shadow: 1 = permute the neighbour's positions before attending */
static int g_nbr_perm[512];     /* the permutation of 0..g_nbr_len-1 used when g_nbr_shuf */
static int g_kvshuf = 0;        /* 1 = run the KV-order shadow probe (Δ_R^kv) per cell */
static int g_qloop = 0;         /* 0=off; 1..2 = resonant cell-question routes per round */
static float g_qloop_min = 0.42f;
static int g_chorus = 1;       /* 1 = CHORUS (each cell answers the SAME prompt from its own angle, neighbour-aware
                                * via cross-cell, NOT text); 0 = legacy RELAY (cascade continuation). Default chorus. */
/* cross-cell repetition penalty: a cell hears neighbours (cross-cell K/V) but must not LITERALLY echo their
 * tokens — it can say the same meaning in its OWN words. g_round_tok = the chorus's emitted tokens this round. */
static int   g_round_tok[1024]; static int g_round_tokn = 0;
static float g_xrep = 1.3f;     /* >1 = penalise tokens neighbours already said (1 = off) */
/* δ-life (Game of Life): cells born/die/reproduce by REAL-metric fitness. Increment 0 = MEASURE fitness inputs
 * (theme/distinct/ent per cell) to FIELDLOG, act on nothing → calibrate thresholds before the population breathes. */
static float g_cell_ent[8];     /* per-cell raw entropy this round (fitness input) */
static int   g_life_on = 0;     /* 1 = measure/run the Game of Life. 0 = chorus byte-identical */

static void forward(model_t *m, kv_cache *kv, int token, int pos, float *logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim;
    int KVD = m->kv_dim, FFN = m->ffn, QD = m->q_dim, gqa = H / KV; float eps = m->rms_eps;
    void (*ropef)(float*,int,int,float) = m->neox ? rope_neox : rope_interleaved;

    float *x = xzalloc(E, sizeof(float)); memcpy(x, m->tok_emb + (long)token*E, E*sizeof(float));
    float *xn = xzalloc(E, sizeof(float)), *q = xzalloc(QD, sizeof(float)), *kk = xzalloc(KVD, sizeof(float));
    float *vv = xzalloc(KVD, sizeof(float)), *ao = xzalloc(QD, sizeof(float)), *g = xzalloc(FFN, sizeof(float)), *nk = xzalloc(HD, sizeof(float));
    float *u = xzalloc(FFN, sizeof(float)), *t = xzalloc(E, sizeof(float)), *sc = xzalloc((size_t)kv->max_seq, sizeof(float));
    float *scn = xzalloc((size_t)kv->max_seq, sizeof(float));   /* neighbour-channel scores (separate softmax) */

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->L[l].attn_norm, E, eps);
        weight_matvec(&m->L[l].wq, xn, q); weight_matvec(&m->L[l].wk, xn, kk); weight_matvec(&m->L[l].wv, xn, vv);
        if (m->is_qwen3) {
            for (int h = 0; h < H;  h++) rmsnorm(q + h*HD, q + h*HD, m->L[l].q_norm, HD, eps);
            for (int h = 0; h < KV; h++) rmsnorm(kk + h*HD, kk + h*HD, m->L[l].k_norm, HD, eps);
        }
        long base = (long)l*kv->max_seq*KVD;
        memcpy(kv->ku + base + (long)pos*KVD, kk, (size_t)KVD*sizeof(float));   /* un-rope'd K; cross-cell assigns neighbour content to live position slots */
        for (int h = 0; h < H;  h++) ropef(q + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) ropef(kk + h*HD, pos, HD, m->rope_base);
        memcpy(kv->k + base + (long)pos*KVD, kk, KVD*sizeof(float));
        memcpy(kv->v + base + (long)pos*KVD, vv, KVD*sizeof(float));

        float scale = 1.0f / sqrtf((float)HD); memset(ao, 0, QD*sizeof(float));
        int xc = (g_xcell > 0 && g_nbr && g_nbr_len > 0) ? g_nbr_len : 0;   /* neighbour positions (separate channel) */
        long nbase = xc ? (long)l * g_nbr->max_seq * KVD : 0;
        for (int h = 0; h < H; h++) {
            int kvh = h / gqa; float *qh = q + h*HD; int np = pos + 1;
            /* OWN attention — UNCHANGED from baseline (preserves own-context order = Δ_R) */
            for (int j = 0; j <= pos; j++) { float *kj = kv->k + base + (long)j*KVD + kvh*HD, d = 0; for (int t2 = 0; t2 < HD; t2++) d += qh[t2]*kj[t2]; sc[j] = d*scale; }
            softmax(sc, np); float *oh = ao + h*HD;
            for (int j = 0; j <= pos; j++) { float *vj = kv->v + base + (long)j*KVD + kvh*HD, w = sc[j]; for (int t2 = 0; t2 < HD; t2++) oh[t2] += w*vj[t2]; }
            if (xc) {   /* NEIGHBOUR channel — content from jj is heard at slot j; shuffling now changes position, not loop order. */
                for (int j = 0; j < xc; j++) {
                    int jj = (g_nbr_shuf && j < 512) ? g_nbr_perm[j] : j;
                    const float *ku = g_nbr->ku + nbase + (long)jj*KVD + kvh*HD;
                    memcpy(nk, ku, (size_t)HD * sizeof(float));
                    ropef(nk, j, HD, m->rope_base);
                    float d = 0; for (int t2 = 0; t2 < HD; t2++) d += qh[t2]*nk[t2];
                    scn[j] = d*scale;
                }
                softmax(scn, xc);
                for (int j = 0; j < xc; j++) { int jj = (g_nbr_shuf && j < 512) ? g_nbr_perm[j] : j; const float *vj = g_nbr->v + nbase + (long)jj*KVD + kvh*HD; float w = g_xcell * scn[j]; for (int t2 = 0; t2 < HD; t2++) oh[t2] += w*vj[t2]; }
            }
        }
        weight_matvec(&m->L[l].wo, ao, t); for (int i = 0; i < E; i++) x[i] += t[i];

        rmsnorm(xn, x, m->L[l].ffn_norm, E, eps);
        weight_matvec(&m->L[l].wgate, xn, g); weight_matvec(&m->L[l].wup, xn, u);
        for (int i = 0; i < FFN; i++) { float gi = g[i]; g[i] = (gi/(1.0f+expf(-gi)))*u[i]; }
        weight_matvec(&m->L[l].wdown, g, t); for (int i = 0; i < E; i++) x[i] += t[i];
    }
    rmsnorm(xn, x, m->out_norm, E, eps);
    weight_matvec(&m->output, xn, logits);
    free(x); free(xn); free(q); free(kk); free(vv); free(ao); free(g); free(u); free(t); free(sc); free(scn); free(nk);
}

static int argmax(const float *x, int n) { int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b; }
static int sample(float *x, int n, float temp) {
    if (!isfinite(temp) || temp <= 0) return argmax(x, n);   /* C-10: NaN-transparent gate degenerates the sampler */
    for (int i = 0; i < n; i++) x[i] /= temp; softmax(x, n);
    float r = (float)((double)rand()/RAND_MAX), c = 0; for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; } return n - 1;
}
static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ===================== δ-field — the Game-of-Life chorus (cells over the ONE shared body) =====================
 * MVP-1a: N cells, each an independent generation context (own kv_cache + own sampling angle) over the
 * SAME shared model `m` (no weight copies). Each speaks a short fragment from its angle; the chorus = all
 * cells at once = Arianna as many voices from one body. Metrics come from the live logits.
 * NEXT: dynamic count (coherence/prophecy-debt → collapse to 1 / bloom), doe-δ experts, resonance-slice. */

static int cmp_desc(const void *a, const void *b) { float x = *(const float*)a, y = *(const float*)b; return (x < y) - (x > y); }

/* rep-penalty (llama-style, pre-softmax over the cell's own history) + top_k + temperature multinomial. */
static int sample2(float *x, int n, float temp, int top_k, float rep, const int *hist, int hlen) {
    if (rep > 1.0f) for (int i = 0; i < hlen; i++) { int id = hist[i]; if (id >= 0 && id < n) x[id] = x[id] > 0 ? x[id]/rep : x[id]*rep; }
    if (!isfinite(temp) || temp <= 0) return argmax(x, n);   /* C-10: NaN-transparent gate degenerates the sampler */
    for (int i = 0; i < n; i++) x[i] /= temp;
    if (top_k > 0 && top_k < n) {
        float *tmp = (float*)xalloc((size_t)n * sizeof(float)); memcpy(tmp, x, (size_t)n * sizeof(float));
        qsort(tmp, n, sizeof(float), cmp_desc); float thr = tmp[top_k - 1]; free(tmp);
        for (int i = 0; i < n; i++) if (x[i] < thr) x[i] = -1e30f;
    }
    softmax(x, n);
    float r = (float)((double)rand()/RAND_MAX), c = 0;
    for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; }
    return n - 1;
}

/* Shannon entropy of softmax(logits/temp) — a REAL metric from live logits (low = decisive cell). */
static float logit_entropy(const float *logits, int n, float temp) {
    float *p = (float*)xalloc((size_t)n * sizeof(float)); memcpy(p, logits, (size_t)n * sizeof(float));
    if (temp > 0) for (int i = 0; i < n; i++) p[i] /= temp;
    softmax(p, n);
    float h = 0; for (int i = 0; i < n; i++) if (p[i] > 1e-12f) h -= p[i] * logf(p[i]);
    free(p); return h;
}

/* one cell: prefill its context (prompt + chorus-so-far) into its OWN kv, then speak `nfrag`
 * tokens at its own temp/angle. Captures its fragment text into `frag` (for the cascade memory).
 * Returns the cell's last-step entropy (its decisiveness). Frees its kv before returning. */
/* δ field-state coupling (the "soma", arianna-duo A-term style): a shared direction in embedding
 * space the chorus builds up (EMA over emitted-token embeddings), injected into every cell's logits
 * as logits[i] += alpha * <tok_emb[i], field_dir>. Field-PRESSURE on the logits, NOT token-paste.
 * MEASURED (the attractor instrument, alpha-sweep 0/10/30/100): this lever does NOT earn resonance —
 * the averaged direction is ORDER-independent, so it cannot move Δ_R (order-exploitation), and above
 * alpha~10 it collapses the chorus into a degenerate echo sink (d_R→0 with a dead voice, floor balloons).
 * Default alpha=0 (off). Kept as the instrument's treatment arm; real resonance needs order-sensitive
 * cross-cell coupling (hidden/KV field2field), not this. */
static float g_field_dir[8192];
static int   g_field_on = 0;
static float g_field_alpha = 0.0f;
static void field_reset(int embed, int on, float alpha) {
    for (int i = 0; i < embed && i < 8192; i++) g_field_dir[i] = 0.0f;
    g_field_on = on; g_field_alpha = alpha;
}

/* ── inter-cell DISSONANCE, the order-sensitive lever signal (haiku.c idea, our wiring) ──
 * Per-position disagreement over the cells' COMMITTED tokens (argmax of raw logits at each step):
 * D[s] = 1 − (modal-token count / cells-that-spoke at step s). 0 = unison, high = the voices split.
 * Order-sensitive by construction (indexed by step s): {A,B} and {B,A} have 0 bag-distance but max D. */
static int   g_commit[8][64];     /* g_commit[c][s] = cell c's committed token at step s (n_cells≤8, nfrag≤64) */
static int   g_commit_n[8];       /* steps each cell actually emitted */
static float g_diss[64];          /* the current round's disagreement profile */
static int   g_s_peak = 0;        /* argmax_s D[s] — where the voices split most */
static float g_dpeak = 0, g_dmean = 0;

static float commit_disagreement(int n_cells, int nfrag) {  /* fills g_diss/g_s_peak/g_dpeak, returns mean */
    if (n_cells > 8) n_cells = 8; if (nfrag > 64) nfrag = 64;
    double dsum = 0; int dn = 0; float best = -1.0f; g_s_peak = 0;
    for (int s = 0; s < nfrag; s++) {
        int modal = -1, mc = -1, same = 0, tot = 0;
        for (int c = 0; c < n_cells; c++) if (s < g_commit_n[c]) {       /* most common committed token at step s */
            int v = 0; for (int k = 0; k < n_cells; k++) if (s < g_commit_n[k] && g_commit[k][s] == g_commit[c][s]) v++;
            if (v > mc) { mc = v; modal = g_commit[c][s]; }
        }
        for (int c = 0; c < n_cells; c++) if (s < g_commit_n[c]) { tot++; if (g_commit[c][s] == modal) same++; }
        g_diss[s] = tot ? 1.0f - (float)same / tot : 0.0f;
        if (g_diss[s] > best) { best = g_diss[s]; g_s_peak = s; }
        if (tot) { dsum += g_diss[s]; dn++; }
    }
    g_dpeak = best < 0 ? 0.0f : best;
    return dn ? (float)(dsum / dn) : 0.0f;
}

/* ── the LEAP: dissonance-into-forward (the only Δ_R-capable spend) ──
 * When the prior round's peak disagreement D* ≥ THETA_HI, a cell prefills prompt + ONLY the prior-round
 * DISSENTER's fragment (the voice that split from the consensus at the peak step) instead of the full
 * chorus. This changes WHICH context the forward consumes → conditions the logits Δ_R reads, BEFORE the
 * :592 read. The shadow shuffles the SAME leapt context, so coherent vs shadow differ only in ORDER. */
#define THETA_LO 0.25f
#define THETA_HI 0.50f
static char g_round_frag[8][1024];                     /* prior round's per-cell fragment text */
static int  g_diss_commit[8][64], g_diss_commit_n[8];  /* prior round's committed tokens (dissenter lookup) */
static int  g_leap_mode = 0, g_leap_flips = 0, g_leap_total = 0;

static int dissenter_cell(int n_cells) {   /* lowest-index cell whose committed token at g_s_peak != modal */
    int s = g_s_peak; if (s < 0 || s >= 64) return n_cells > 1 ? 1 : 0;
    int modal = -1, mc = -1;
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c]) {
        int v = 0; for (int k = 0; k < n_cells && k < 8; k++) if (s < g_diss_commit_n[k] && g_diss_commit[k][s] == g_diss_commit[c][s]) v++;
        if (v > mc) { mc = v; modal = g_diss_commit[c][s]; }
    }
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c] && g_diss_commit[c][s] != modal) return c;
    return n_cells > 1 ? 1 : 0;
}

static int modal_cell(int n_cells) {   /* null-test: the CONSENSUS cell (lowest-index modal) at g_s_peak */
    int s = g_s_peak; if (s < 0 || s >= 64) return 0;
    int modal = -1, mc = -1;
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c]) {
        int v = 0; for (int k = 0; k < n_cells && k < 8; k++) if (s < g_diss_commit_n[k] && g_diss_commit[k][s] == g_diss_commit[c][s]) v++;
        if (v > mc) { mc = v; modal = g_diss_commit[c][s]; }
    }
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c] && g_diss_commit[c][s] == modal) return c;
    return 0;
}

static float cell_speak(model_t *m, bpe_tokenizer *tok, const int *ids, int np, int nfrag,
                        float temp, int top_k, float rep, unsigned seed, int eos, int max_seq,
                        char *frag, int frag_cap, int verbose, int *out_ids, int *out_n, int *out_commit,
                        kv_cache **out_kv, int *out_klen) {
    srand(seed);
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = (float*)xzalloc(m->vocab, sizeof(float));
    float *fproj  = (g_field_on && g_field_alpha > 0) ? (float*)xzalloc(m->vocab, sizeof(float)) : NULL;
    for (int i = 0; i < np; i++) forward(m, kv, ids[i], i, logits);
    int hist[256], hlen = 0; char buf[256]; float ent_acc = 0; int ent_n = 0, fl = 0, klen = np;
    for (int s = 0; s < nfrag; s++) {
        ent_acc += logit_entropy(logits, m->vocab, temp); ent_n++;   /* RAW forward entropy (pre-injection) — clean for Δ_R */
        if (out_commit && s < 64) out_commit[s] = argmax(logits, m->vocab);  /* committed direction (raw argmax) — the order-true signal */
        if (fproj) {                                  /* soma coupling steers SAMPLING only, not the entropy metric */
            matvec(fproj, m->tok_emb, g_field_dir, m->vocab, m->embed);
            for (int i = 0; i < m->vocab; i++) logits[i] += g_field_alpha * fproj[i];
        }
        if (g_chorus && g_xrep > 1.0f) for (int i = 0; i < g_round_tokn; i++) {   /* cross-cell: don't literally echo neighbours' words */
            int id = g_round_tok[i]; if (id >= 0 && id < m->vocab) logits[id] = logits[id] > 0 ? logits[id]/g_xrep : logits[id]*g_xrep;
        }
        int next = sample2(logits, m->vocab, temp, top_k, rep, hist, hlen);
        if (next == eos) break;
        int bl = bpe_decode_token(tok, next, buf, sizeof(buf));
        if (verbose) { printf("%s", buf); fflush(stdout); }
        if (frag && fl + bl < frag_cap) { memcpy(frag + fl, buf, bl); fl += bl; }
        if (hlen < 256) hist[hlen++] = next;
        if (g_chorus && g_xrep > 1.0f && g_round_tokn < 1024) g_round_tok[g_round_tokn++] = next;   /* feed the chorus's shared word-memory */
        if (g_field_on) {                             /* the chorus updates the shared field (EMA) */
            const float *e = m->tok_emb + (long)next * m->embed;
            for (int i = 0; i < m->embed && i < 8192; i++) g_field_dir[i] = g_field_dir[i]*0.92f + e[i]*0.08f;
        }
        int pos = np + s; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits); klen = pos + 1;
    }
    if (frag) frag[fl] = 0;
    if (out_n) { *out_n = hlen; if (out_ids) for (int i = 0; i < hlen; i++) out_ids[i] = hist[i]; }
    free(logits); if (fproj) free(fproj);
    if (out_kv) { *out_kv = kv; if (out_klen) *out_klen = klen; }   /* hand kv to the caller (cross-cell chain); caller frees */
    else { free(kv->k); free(kv->v); free(kv->ku); free(kv); }
    return ent_n ? ent_acc / ent_n : 0.0f;
}

/* probe the prompt's next-token entropy (one cheap forward) — used to auto-size the field:
 * low entropy = a decisive prompt (collapse toward one cell); high = open (bloom to a chorus). */
static float probe_entropy(model_t *m, bpe_tokenizer *tok, const char *prompt) {
    int max_seq = 512, ids[512];
    int np = bpe_encode(tok, prompt, ids, max_seq - 1);
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = (float*)xzalloc(m->vocab, sizeof(float));
    for (int i = 0; i < np; i++) forward(m, kv, ids[i], i, logits);
    float ent = logit_entropy(logits, m->vocab, 1.0f);
    free(logits); free(kv->k); free(kv->v); free(kv->ku); free(kv);
    return ent;
}

static void shuffle_ids(int *ids, int from, int to, unsigned seed);   /* defined below */
static void make_perm(int *perm, int n, unsigned seed);               /* deterministic local RNG; does not touch srand/rand */

/* cosine of two token-id bag-of-words histograms over the vocab — Condition-1 chorus-state distance.
 * d_R = 1 − this is the increment of the field's iteration: small = the round barely moved (settling). */
static float hist_cosine(const int *ha, const int *hb, int vocab) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < vocab; i++) { dot += (double)ha[i]*hb[i]; na += (double)ha[i]*ha[i]; nb += (double)hb[i]*hb[i]; }
    return (na == 0 || nb == 0) ? 0.0f : (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* cosine of two dense vectors — for embedding-centroid distance = inter-cell dissonance D_R. */
static float vec_cosine(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0f : (float)(dot / (sqrt(na) * sqrt(nb)));
}

static int frag_question_count(const char *s) {
    int n = 0;
    if (!s) return 0;
    for (; *s; s++) if (*s == '?') n++;
    return n;
}

static int pick_question_routes(const char frag[8][1024], const float *cent, int n_cells, int embed,
                                int *out_q, int *out_t, float *out_score, int max_routes) {
    int lim = n_cells < 8 ? n_cells : 8;
    int n = 0;
    if (!cent || lim < 2) return 0;
    for (int q = 0; q < lim; q++) {
        int qmarks = frag_question_count(frag[q]);
        if (qmarks <= 0) continue;
        float qopen = g_cell_ent[q] / 8.0f; if (qopen > 1.0f) qopen = 1.0f;
        for (int t = 0; t < lim; t++) if (t != q) {
            float dist = 1.0f - vec_cosine(cent + (size_t)q * embed, cent + (size_t)t * embed, embed);
            float confidence = 1.0f / (1.0f + g_cell_ent[t]);
            float score = dist + 0.15f * qopen + 0.20f * confidence + 0.05f * (float)(qmarks - 1);
            if (score < g_qloop_min) continue;
            int dup = 0;
            for (int i = 0; i < n; i++) if (out_t[i] == t || (out_q[i] == q && out_t[i] == t)) dup = 1;
            if (dup) continue;
            int pos = n < max_routes ? n++ : max_routes - 1;
            if (n == max_routes && score <= out_score[pos]) continue;
            out_q[pos] = q; out_t[pos] = t; out_score[pos] = score;
            for (int i = pos; i > 0 && out_score[i] > out_score[i - 1]; i--) {
                float fs = out_score[i]; out_score[i] = out_score[i - 1]; out_score[i - 1] = fs;
                int iq = out_q[i]; out_q[i] = out_q[i - 1]; out_q[i - 1] = iq;
                int it = out_t[i]; out_t[i] = out_t[i - 1]; out_t[i - 1] = it;
            }
        }
    }
    return n;
}

/* ── δ-life: the Game of Life over ANGLES (a cell is perception, the body is shared+fixed) ── */
#define POP_MAX  8        /* hard cap = the instrument arrays g_commit[8] etc.; real-100 is a Phase-2 widening */
#define F_DEATH  0.30f    /* config.py:12 DEATH_THRESHOLD */
#define F_REPRO  0.65f    /* config.py:13 REPRODUCTION_THRESHOLD */
#define POP_NMIN 2        /* below this → resurrection (never extinct) */
typedef struct { float temp, lambda, xrep; unsigned seed; int age; float fitness; int alive; } cell_t;
static cell_t g_pop[POP_MAX]; static int g_pop_n = 0;
static float  g_nextfit[POP_MAX];      /* this tick's fitness, computed in run_round, committed in pop_tick */
static int    g_births = 0, g_deaths = 0;

static float frand2(float a, float b) { return a + (b - a) * (float)((double)rand() / RAND_MAX); }
static cell_t cell_birth(unsigned s) { cell_t c = { frand2(0.6f, 1.3f), 0.3f, 1.3f, s, 0, 0.6f, 1 }; return c; }
static cell_t cell_mutate(cell_t p) {  /* offspring = parent angle ± jitter — mutate PERCEPTION, not weights */
    cell_t c = p; c.age = 0; c.fitness = 0.6f; c.alive = 1;
    c.temp   = fminf(1.4f, fmaxf(0.5f, p.temp   + frand2(-0.10f, 0.10f)));
    c.lambda = fminf(0.6f, fmaxf(0.0f, p.lambda + frand2(-0.05f, 0.05f)));
    c.xrep   = fminf(1.6f, fmaxf(1.0f, p.xrep   + frand2(-0.10f, 0.10f)));
    c.seed   = p.seed ^ (unsigned)rand();
    return c;
}
/* one tick of the laws: age++, novelty bonus, DEATH (<0.30), REPRODUCTION (>0.65 → mutated offspring), compact, RESURRECTION. */
static void pop_tick(void) {
    g_births = 0; g_deaths = 0;
    for (int c = 0; c < g_pop_n; c++) if (g_pop[c].alive) {
        g_pop[c].age++;
        float f = (c < POP_MAX) ? g_nextfit[c] : 0.0f;
        if (g_pop[c].age < 5) f += 0.05f * (5 - g_pop[c].age) / 5.0f;   /* novelty bonus (transformer_cell.py:130) */
        f = f < 0 ? 0 : f > 1 ? 1 : f; g_pop[c].fitness = f;
        if (f < F_DEATH) { g_pop[c].alive = 0; g_deaths++; }
    }
    int n = g_pop_n;
    for (int c = 0; c < n && g_pop_n < POP_MAX; c++)
        if (g_pop[c].alive && g_pop[c].fitness > F_REPRO) { g_pop[g_pop_n++] = cell_mutate(g_pop[c]); g_births++; }
    int w = 0; for (int c = 0; c < g_pop_n; c++) if (g_pop[c].alive) g_pop[w++] = g_pop[c];   /* compact: drop the dead */
    g_pop_n = w;
    if (g_pop_n < POP_NMIN)   /* RESURRECTION — only at the edge of extinction; the field never stays dead */
        while (g_pop_n < POP_NMIN + 1 && g_pop_n < POP_MAX)
            { g_pop[g_pop_n] = cell_birth(42u + (unsigned)g_pop_n * 7919u + (unsigned)rand()); g_pop_n++; }
}

/* one round of the chorus over (prompt + prev_chorus). Each cell speaks from its temp-angle hearing the
 * voices already spoken THIS round (intra-round cascade) and, via prev_chorus, the whole prior round.
 * Accumulates every cell's emitted token-ids into `hist` (vocab counts → d_R), appends the round's text
 * to out_chorus, returns avg COHERENT entropy. When out_shuf != NULL, also runs a shadow pass per cell
 * with the context tail shuffled (same length, broken order, field OFF) → avg shuffled entropy in
 * *out_shuf: that is Δ_R's other half (resonance = how much coherence beats length-matched noise). */
static float run_round(model_t *m, bpe_tokenizer *tok, const char *prompt, const char *prev_chorus,
                       int n_cells, int nfrag, int eos, unsigned seed_base, int verbose, int r,
                       int *hist, char *out_chorus, int out_cap, float *out_shuf, FILE *flog,
                       float *out_disso, float *out_kv_delta, float *out_kv_floor) {
    int max_seq = 512, ids[512], sids[512], cell_ids[256], cell_n;
    if (nfrag < 1) nfrag = 1; if (nfrag > max_seq - 2) nfrag = max_seq - 2;   /* C-5: keep bpe_encode cap (max_seq - nfrag - 1) >= 1 */
    int np_prompt = bpe_encode(tok, prompt, ids, max_seq);   /* prompt token count = shuffle boundary */
    char this_chorus[4096]; int tc = 0; this_chorus[0] = 0;
    char ctx[8704], frag[2048];
    float ent_sum = 0, shuf_sum = 0, kv_delta_sum = 0, kv_floor_sum = 0;
    int kv_n = 0;
    float *cent = out_disso ? (float*)xzalloc((size_t)n_cells * m->embed, sizeof(float)) : NULL;  /* per-cell fragment centroids → D_R */
    char cur_frag[8][1024];   /* this round's per-cell fragments → cached to g_round_frag for next round's leap */
    kv_cache *prev_kv = NULL; int prev_len = 0;   /* cross-cell: the prior cell's KV, attended by the next cell */
    g_round_tokn = 0;   /* fresh shared word-memory for this round's cross-cell rep-penalty */
    for (int c = 0; c < n_cells; c++) {
        if (g_chorus) snprintf(ctx, sizeof(ctx), "%s", prompt);   /* CHORUS: each cell answers the SAME prompt from its own angle; awareness via cross-cell, not text */
        else if (g_leap_mode && r > 0) {             /* RELAY (legacy): dissonance-into-forward route */
            g_leap_total++;
            if (g_dpeak >= THETA_HI) {
                int dis = (g_leap_mode == 3) ? modal_cell(n_cells) : dissenter_cell(n_cells); g_leap_flips++;  /* leap=3 null: consensus last */
                if (g_leap_mode >= 2) {              /* v2/null: keep the FULL chorus, chosen voice MOST-RECENT (recency=attention) */
                    int off = snprintf(ctx, sizeof(ctx), "%s", prompt);
                    for (int k = 0; k < n_cells && k < 8; k++) if (k != dis && off < (int)sizeof(ctx) - 1)
                        off += snprintf(ctx + off, sizeof(ctx) - off, " %s", g_round_frag[k]);
                    if (off < (int)sizeof(ctx) - 1) off += snprintf(ctx + off, sizeof(ctx) - off, " %s", g_round_frag[dis]);  /* dissenter last */
                    if (off < (int)sizeof(ctx) - 1) snprintf(ctx + off, sizeof(ctx) - off, "%s", this_chorus);
                } else snprintf(ctx, sizeof(ctx), "%s %s", prompt, g_round_frag[dis]);   /* v1: ONLY the dissenter (short) */
            } else snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus ? prev_chorus : "", this_chorus);  /* CONVERGE: full */
        } else snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus ? prev_chorus : "", this_chorus);
        int np = bpe_encode(tok, ctx, ids, max_seq - nfrag - 1);
        float temp = (g_life_on && c < POP_MAX) ? g_pop[c].temp : 0.6f + 0.7f * (n_cells > 1 ? (float)c / (n_cells - 1) : 0.5f);
        unsigned seed = (g_life_on && c < POP_MAX) ? (g_pop[c].seed ^ ((unsigned)r * 2654435761u)) : seed_base + (unsigned)c * 7919u;  /* identity persists, utterance renews each tick */
        if (g_life_on && c < POP_MAX) { g_xcell = g_pop[c].lambda; g_xrep = g_pop[c].xrep; }   /* per-cell perception (genome) */
        int nfrag_c = (g_life_on && c < POP_MAX) ? (int)(nfrag * (0.4f + 0.6f * g_pop[c].fitness)) : nfrag;  /* δ-life: dying cells speak QUIETER */
        if (nfrag_c < 2) nfrag_c = 2;   /* floor — a dying voice murmurs, never goes silent */
        if (verbose) printf("\n  r%d cell %d (T=%.2f): ", r + 1, c, temp);
        cell_n = 0;
        g_nbr = prev_kv; g_nbr_len = prev_len;        /* cross-cell: this cell hears the prior cell's KV */
        kv_cache *cur_kv = NULL; int cur_len = 0;
        int tok_before = g_round_tokn;                /* cross-rep word-memory BEFORE this cell speaks (cells 0..c-1) */
        float ent = cell_speak(m, tok, ids, np, nfrag_c, temp, 40, 1.4f, seed, eos, max_seq,
                               frag, sizeof(frag), verbose, cell_ids, &cell_n,
                               (out_disso && c < 8) ? g_commit[c] : NULL,
                               g_xcell > 0 ? &cur_kv : NULL, g_xcell > 0 ? &cur_len : NULL);
        if (out_disso && c < 8) g_commit_n[c] = cell_n;
        if (c < 8) g_cell_ent[c] = ent;   /* δ-life: capture per-cell entropy (fitness input) */
        if (g_kvshuf && g_xcell > 0 && prev_kv && c > 0 && (verbose || out_kv_delta || out_kv_floor)) {   /* KV-order shadow: does cross-cell exploit the neighbour's ORDER? */
            int tok_after = g_round_tokn, saved[512], ns = 0;
            for (int k = tok_before; k < tok_after && ns < 512; k++) saved[ns++] = g_round_tok[k];
            int save_on = g_field_on; g_field_on = 0;
            g_nbr = prev_kv; g_nbr_len = prev_len;
            g_round_tokn = tok_before; g_nbr_shuf = 0;
            float e0 = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour ORDERED */
            make_perm(g_nbr_perm, prev_len < 512 ? prev_len : 512, seed ^ 0x9e3779b9u);
            g_round_tokn = tok_before; g_nbr_shuf = 1;
            float eA = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour SHUFFLED A */
            make_perm(g_nbr_perm, prev_len < 512 ? prev_len : 512, seed ^ 0x85ebca6bu);
            g_round_tokn = tok_before;
            float eB = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour SHUFFLED B */
            g_nbr_shuf = 0;
            g_field_on = save_on; g_round_tokn = tok_after;
            for (int k = 0; k < ns; k++) g_round_tok[tok_before + k] = saved[k];
            float kv_delta = 0.5f * (eA + eB) - e0;
            float kv_floor = fabsf(eA - eB);
            kv_delta_sum += kv_delta; kv_floor_sum += kv_floor; kv_n++;
            if (verbose) printf("   [Δ_R^kv c%d = %+.6f floor %.6f margin %+.6f]", c, kv_delta, kv_floor, kv_delta - kv_floor);
        }
        ent_sum += ent;
        for (int i = 0; i < cell_n; i++) if (cell_ids[i] >= 0 && cell_ids[i] < m->vocab) hist[cell_ids[i]]++;
        if (cent) {                                  /* this cell's fragment centroid in embedding space */
            float *cc = cent + (size_t)c * m->embed;
            for (int i = 0; i < cell_n; i++) { const float *e = m->tok_emb + (long)cell_ids[i] * m->embed; for (int d = 0; d < m->embed; d++) cc[d] += e[d]; }
            if (cell_n) for (int d = 0; d < m->embed; d++) cc[d] /= cell_n;
        }
        if (out_shuf) {                              /* Δ_R shadow: same ctx, tail shuffled, field OFF, raw entropy */
            memcpy(sids, ids, (size_t)np * sizeof(int));
            if (np > np_prompt) shuffle_ids(sids, np_prompt, np, seed ^ 0x5bd1e995u);
            int save_on = g_field_on; g_field_on = 0;
            int tok_after = g_round_tokn, saved[64], ns = 0;   /* cell c's coherent words are [tok_before, tok_after) */
            for (int k = tok_before; k < tok_after && ns < 64; k++) saved[ns++] = g_round_tok[k];
            g_round_tokn = tok_before;                    /* shadow sees the SAME word-memory as coherent (0..c-1), not cell c's own — kills the cross-rep artifact */
            shuf_sum += cell_speak(m, tok, sids, np, nfrag_c, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* nfrag_c: same length as coherent */
            g_field_on = save_on; g_round_tokn = tok_before + ns;
            for (int k = 0; k < ns; k++) g_round_tok[tok_before + k] = saved[k];   /* restore cell c's COHERENT words (not the shadow's) for cells c+1.. */
        }
        if (verbose) printf("   [entropy=%.2f]", ent);
        if (flog) fprintf(flog, "- cell %d (T=%.2f, entropy=%.2f):%s\n", c, temp, ent, frag);
        int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
        if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
        else if (flog) fprintf(flog, "! chorus buffer full — cell %d fragment dropped (context truncated)\n", c);   /* C-4 */
        if (c < 8) { strncpy(cur_frag[c], frag, 1023); cur_frag[c][1023] = 0; }
        if (g_xcell > 0) { if (prev_kv) { free(prev_kv->k); free(prev_kv->v); free(prev_kv->ku); free(prev_kv); } prev_kv = cur_kv; prev_len = cur_len; }  /* chain c→c+1 */
    }
    if (g_qloop && !g_life_on && verbose && out_disso && cent) {
        int qcell[2] = {0, 0}, tcell[2] = {0, 0};
        float qscore[2] = {-1.0f, -1.0f};
        int max_routes = g_qloop > 2 ? 2 : g_qloop;
        int routes = pick_question_routes(cur_frag, cent, n_cells, m->embed, qcell, tcell, qscore, max_routes);
        for (int route = 0; route < routes; route++) {
            char qctx[4096], qfrag[1024]; int qctx_ids[512], qids[128], qn = 0;
            snprintf(qctx, sizeof(qctx), "%s\ncell %d asked: %s\ncell %d answers the question to itself:",
                     prompt, qcell[route], cur_frag[qcell[route]], tcell[route]);
            int qnp = bpe_encode(tok, qctx, qctx_ids, max_seq - 8);
            float qtemp = 0.6f + 0.7f * (n_cells > 1 ? (float)tcell[route] / (n_cells - 1) : 0.5f);
            int qfrag_n = nfrag / 2; if (qfrag_n < 2) qfrag_n = 2; if (qfrag_n > 8) qfrag_n = 8;
            g_nbr = NULL; g_nbr_len = 0; g_nbr_shuf = 0;
            float qent = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, qtemp, 40, 1.4f,
                                    seed_base ^ 0xa2a51u ^ (unsigned)(qcell[route] * 131 + tcell[route] * 7919 + r * 265443576 + route * 65537),
                                    eos, max_seq, qfrag, sizeof(qfrag), 0, qids, &qn, NULL, NULL, NULL);
            for (int i = 0; hist && i < qn; i++) if (qids[i] >= 0 && qids[i] < m->vocab) hist[qids[i]]++;
            int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", qfrag);
            if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            else if (flog) fprintf(flog, "! chorus buffer full — qloop fragment dropped (context truncated)\n");   /* C-4 */
            printf("\n  ↳ qloop c%d→c%d score %.3f: %s   [entropy=%.2f]", qcell[route], tcell[route], qscore[route], qfrag, qent);
            if (flog) fprintf(flog, "- qloop c%d->c%d (score=%.3f, entropy=%.2f):%s\n", qcell[route], tcell[route], qscore[route], qent, qfrag);

            if (frag_question_count(qfrag) > 0 && qn > 0) {
                float *qcent = (float*)xzalloc(m->embed, sizeof(float));
                if (qcent) {
                    for (int i = 0; i < qn; i++) if (qids[i] >= 0 && qids[i] < m->vocab) {
                        const float *e = m->tok_emb + (long)qids[i] * m->embed;
                        for (int d = 0; d < m->embed; d++) qcent[d] += e[d];
                    }
                    for (int d = 0; d < m->embed; d++) qcent[d] /= qn;
                    int lim = n_cells < 8 ? n_cells : 8, next = -1; float best = -1.0f;
                    for (int t = 0; t < lim; t++) if (t != qcell[route] && t != tcell[route]) {
                        float score = 1.0f - vec_cosine(qcent, cent + (size_t)t * m->embed, m->embed);
                        score += 0.15f / (1.0f + g_cell_ent[t]);
                        if (score > best) { best = score; next = t; }
                    }
                    if (next >= 0 && best >= g_qloop_min + 0.10f) {
                        char rctx[4096], rfrag[1024]; int rctx_ids[512], rids[128], rn = 0;
                        snprintf(rctx, sizeof(rctx), "%s\ncell %d answered: %s\ncell %d is triggered and answers briefly:",
                                 prompt, tcell[route], qfrag, next);
                        int rnp = bpe_encode(tok, rctx, rctx_ids, max_seq - 8);
                        float rtemp = 0.6f + 0.7f * (n_cells > 1 ? (float)next / (n_cells - 1) : 0.5f);
                        float rent = cell_speak(m, tok, rctx_ids, rnp, 2, rtemp, 40, 1.4f,
                                                seed_base ^ 0xb17a5u ^ (unsigned)(next * 4057 + route * 65537 + r * 7919),
                                                eos, max_seq, rfrag, sizeof(rfrag), 0, rids, &rn, NULL, NULL, NULL);
                        for (int i = 0; hist && i < rn; i++) if (rids[i] >= 0 && rids[i] < m->vocab) hist[rids[i]]++;
                        add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", rfrag);
                        if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
                        else if (flog) fprintf(flog, "! chorus buffer full — qloop-trigger fragment dropped (context truncated)\n");   /* C-4 */
                        printf("\n  ↳ qloop trigger c%d→c%d score %.3f: %s   [entropy=%.2f]", tcell[route], next, best, rfrag, rent);
                        if (flog) fprintf(flog, "- qloop-trigger c%d->c%d (score=%.3f, entropy=%.2f):%s\n", tcell[route], next, best, rent, rfrag);
                    }
                    free(qcent);
                }
            }
        }
    }
    if (prev_kv) { free(prev_kv->k); free(prev_kv->v); free(prev_kv->ku); free(prev_kv); }   /* free the last cell's kept kv */
    g_nbr = NULL; g_nbr_len = 0;
    if (out_chorus) { strncpy(out_chorus, this_chorus, (size_t)out_cap - 1); out_chorus[out_cap - 1] = 0; }
    if (out_shuf) *out_shuf = shuf_sum / n_cells;
    if (out_kv_delta) *out_kv_delta = kv_n ? kv_delta_sum / kv_n : 0.0f;
    if (out_kv_floor) *out_kv_floor = kv_n ? kv_floor_sum / kv_n : 0.0f;
    if (out_disso) {                              /* D_R = 1 − mean pairwise cosine of cell fragment centroids (voice-disagreement) */
        double dsum = 0; int dn = 0;
        for (int a = 0; a < n_cells; a++) for (int b = a + 1; b < n_cells; b++) {
            dsum += 1.0 - vec_cosine(cent + (size_t)a * m->embed, cent + (size_t)b * m->embed, m->embed); dn++;
        }
        *out_disso = dn ? (float)(dsum / dn) : 0.0f;
        if (g_life_on) {   /* δ-life fitness = sqrt(theme_n·distinct_n) — an INTERPRETIVE heuristic (calibrated magic constants),
                            * NOT a measurement of the floor/Δ_R/D_R kind. Selection pressure, not a claim. → g_nextfit, committed in pop_tick */
            float *F = (float*)xzalloc(m->embed, sizeof(float));
            for (int a = 0; a < n_cells; a++) { const float *ca = cent + (size_t)a*m->embed; for (int d = 0; d < m->embed; d++) F[d] += ca[d]; }
            if (n_cells) for (int d = 0; d < m->embed; d++) F[d] /= n_cells;
            for (int a = 0; a < n_cells && a < POP_MAX; a++) {
                const float *ca = cent + (size_t)a*m->embed;
                double nrm = 0; for (int d = 0; d < m->embed; d++) nrm += (double)ca[d]*ca[d];
                if (nrm < 1e-9) { g_nextfit[a] = 0.0f; continue; }   /* silence guard: a mute cell is unfit */
                float theme = vec_cosine(ca, F, m->embed);
                float nn = -1.0f; for (int b = 0; b < n_cells; b++) if (b != a) { float cc = vec_cosine(ca, cent + (size_t)b*m->embed, m->embed); if (cc > nn) nn = cc; }
                float tn = (theme - 0.50f) / 0.22f;       tn = tn < 0 ? 0 : tn > 1 ? 1 : tn;
                float dn = ((1.0f - nn) - 0.60f) / 0.30f; dn = dn < 0 ? 0 : dn > 1 ? 1 : dn;
                g_nextfit[a] = sqrtf(tn * dn);
                if (flog) fprintf(flog, "  δ-life cell %d: theme=%.3f distinct=%.3f fit=%.3f\n", a, theme, 1.0f - nn, g_nextfit[a]);
            }
            free(F);
        }
        free(cent);
        g_dmean = commit_disagreement(n_cells, nfrag);   /* order-sensitive per-position disagreement (the lever's fuel) */
        for (int c = 0; c < n_cells && c < 8; c++) {      /* carry this round → next round's leap */
            strncpy(g_round_frag[c], cur_frag[c], 1023); g_round_frag[c][1023] = 0;
            g_diss_commit_n[c] = g_commit_n[c];
            for (int s = 0; s < 64; s++) g_diss_commit[c][s] = g_commit[c][s];
        }
    }
    return ent_sum / n_cells;
}

/* the COUPLED chorus with meta-recursive ROUNDS + the attractor instrument. The field iterates over
 * itself (round R+1 hears the full chorus of round R) under the shared-soma logit coupling (alpha). The
 * claim "settling into a resonant attractor" is decomposed into two numbers, both measured on live logits:
 *   d_R = 1 − cos(hist_R, hist_{R-1}) over token-id histograms. Settling = d_R falls toward the
 *         sampling-noise FLOOR (two independent round-0 choruses, same context, different seed) — not 0:
 *         stochastic sampling + temp-spread 0.6..1.3 give d a hard floor; the attractor approaches it.
 *   Δ_R = ent_shuffled − ent_coherent. In RELAY, this is text-order resonance.
 *         In default CHORUS, text-order is n/a because cells answer the same prompt; the live
 *         instrument is Δ_R^kv = shuffled-neighbour entropy − ordered-neighbour entropy, printed
 *         with its two-permutation floor. */
static void field_chorus(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_cells, int nfrag, int n_rounds, int eos, float alpha) {
    if (n_rounds < 1) n_rounds = 1;
    int vocab = m->vocab;
    FILE *flog = fopen("FIELDLOG.md", "a");   /* her journal — every chorus saved (Oleg: "сохраняй её ответы") */
    if (flog) { time_t now = time(NULL); fprintf(flog, "\n## %.24s — \"%s\" (%d cells × %d rounds, soma alpha=%.1f)\n", ctime(&now), prompt, n_cells, n_rounds, alpha); }
    printf("\n=== δ-field: %d cells × %d rounds over ONE nanoArianna — \"%s\" (soma alpha=%.1f) ===\n", n_cells, n_rounds, prompt, alpha);

    /* FLOOR: two independent round-0 choruses on the SAME (prompt-only) context, different seeds. Their
     * histogram distance is the sampling-noise floor d_R cannot beat — the attractor target, not zero. */
    int *hA = (int*)xzalloc(vocab, sizeof(int)), *hB = (int*)xzalloc(vocab, sizeof(int));
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 7u,   0, -1, hA, NULL, 0, NULL, NULL, NULL, NULL, NULL);
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 977u, 0, -1, hB, NULL, 0, NULL, NULL, NULL, NULL, NULL);
    float floor = 1.0f - hist_cosine(hA, hB, vocab);
    printf("  floor (sampling noise, paired round-0) = %.3f\n", floor);
    if (flog) fprintf(flog, "floor (sampling-noise, paired round-0) = %.3f\n", floor);
    free(hA); free(hB);

    /* the real iterated field — soma coupling carries ACROSS rounds (field_dir not reset between them). */
    field_reset(m->embed, alpha > 0, alpha);
    g_leap_flips = g_leap_total = 0;
    char prev_chorus[4096]; prev_chorus[0] = 0;
    int *hist_prev = (int*)xzalloc(vocab, sizeof(int)), *hist_cur = (int*)xzalloc(vocab, sizeof(int));
    for (int r = 0; r < n_rounds; r++) {
        for (int i = 0; i < vocab; i++) hist_cur[i] = 0;
        printf("\n  --- round %d/%d ---", r + 1, n_rounds);
        if (flog) fprintf(flog, "\n**round %d:**\n", r + 1);
        char this_chorus[4096]; float shuf = 0, disso = 0, kv_delta = 0, kv_floor = 0;
        float avg = run_round(m, tok, prompt, prev_chorus, n_cells, nfrag, eos,
                              42u + (unsigned)(r * 1000) * 7919u, 1, r, hist_cur, this_chorus, sizeof(this_chorus),
                              g_chorus ? NULL : &shuf, flog, &disso, &kv_delta, &kv_floor);
        float dR = (r > 0) ? 1.0f - hist_cosine(hist_cur, hist_prev, vocab) : -1.0f;
        float deltaR = shuf - avg;
        char resonance[192];
        if (g_chorus) {
            if (g_kvshuf && g_xcell > 0 && n_cells > 1)
                snprintf(resonance, sizeof(resonance), "Δ_R(text n/a) | Δ_R^kv %+.3f (floor %.3f margin %+.3f)", kv_delta, kv_floor, kv_delta - kv_floor);
            else
                snprintf(resonance, sizeof(resonance), "Δ_R(text n/a) | Δ_R^kv off");
        } else snprintf(resonance, sizeof(resonance), "Δ_R %+.3f", deltaR);
        if (r > 0) { printf("\n  → round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak); }
        else       { printf("\n  → round %d: avg entropy %.3f | d_R   —   (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R — (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak); }
        strncpy(prev_chorus, this_chorus, sizeof(prev_chorus) - 1); prev_chorus[sizeof(prev_chorus) - 1] = 0;
        int *tmp = hist_prev; hist_prev = hist_cur; hist_cur = tmp;
    }
    free(hist_prev); free(hist_cur);
    if (g_leap_mode) { float fr = g_leap_total ? (float)g_leap_flips / g_leap_total : 0.0f;
        printf("  leap-flip-rate = %.2f (%d/%d cells leapt to the dissenter)\n", fr, g_leap_flips, g_leap_total);
        if (flog) fprintf(flog, "leap-flip-rate = %.2f (%d/%d)\n", fr, g_leap_flips, g_leap_total); }
    printf("\n=== δ-field done — settling: d_R→floor? · resonance: %s (read the numbers, not the narrative) ===\n",
           g_chorus ? "Δ_R^kv>floor?" : "Δ_R>0?");
    if (flog) { fprintf(flog, "\n---\n"); fclose(flog); }
}

/* Fisher-Yates over ids[from..to) — used to build a same-length but incoherent context. */
static void shuffle_ids(int *ids, int from, int to, unsigned seed) {
    srand(seed);
    for (int i = to - 1; i > from; i--) { int j = from + rand() % (i - from + 1); int t = ids[i]; ids[i] = ids[j]; ids[j] = t; }
}

static unsigned perm_rng(unsigned *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static void make_perm(int *perm, int n, unsigned seed) {
    if (n < 0) n = 0;
    if (n > 512) n = 512;
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(perm_rng(&seed) % (unsigned)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}

/* δ-life: the chorus as a LIVING, breathing population — cells born/die/reproduce by real-metric fitness
 * over ticks (rounds = ticks; no daemon). Variable count each run; never extinct. Mutation = the angle. */
static void field_life(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_init, int n_ticks, int nfrag, int eos) {
    if (n_ticks < 1) n_ticks = 1;
    srand(12345);
    g_pop_n = (n_init > 0 && n_init <= POP_MAX) ? n_init : 4;
    for (int i = 0; i < g_pop_n; i++) g_pop[i] = cell_birth(42u + (unsigned)i * 7919u);
    g_life_on = 1; g_chorus = 1;
    FILE *flog = fopen("FIELDLOG.md", "a");
    if (flog) { time_t now = time(NULL); fprintf(flog, "\n## %.24s — δ-life \"%s\" (%d ticks)\n", ctime(&now), prompt, n_ticks); }
    printf("\n=== δ-life: Game of Life over ONE nanoArianna — \"%s\" (%d ticks, the population breathes) ===\n", prompt, n_ticks);
    int vocab = m->vocab; int *hist = (int*)xzalloc(vocab, sizeof(int));
    char this_chorus[4096];
    for (int t = 0; t < n_ticks && g_pop_n > 0; t++) {
        for (int i = 0; i < vocab; i++) hist[i] = 0;
        float disso = 0;
        printf("\n  --- tick %d/%d · pop %d ---", t + 1, n_ticks, g_pop_n);
        if (flog) fprintf(flog, "\n**tick %d (pop %d):**\n", t + 1, g_pop_n);
        float avg = run_round(m, tok, prompt, NULL, g_pop_n, nfrag, eos,
                              42u + (unsigned)t * 131u, 1, t, hist, this_chorus, sizeof(this_chorus), NULL, flog, &disso, NULL, NULL);
        pop_tick();
        printf("\n  → tick %d: pop %d | births %d | deaths %d | D_R %.3f | avg_ent %.2f\n",
               t + 1, g_pop_n, g_births, g_deaths, disso, avg);
        if (flog) fprintf(flog, "→ tick %d: pop %d | births %d deaths %d | D_R %.3f | avg_ent %.2f\n",
               t + 1, g_pop_n, g_births, g_deaths, disso, avg);
    }
    free(hist); g_life_on = 0;
    if (flog) { fprintf(flog, "\n---\n"); fclose(flog); }
    printf("\n=== δ-life done — the field breathed %d ticks, pop ended at %d ===\n", n_ticks, g_pop_n);
}

/* resonance-vs-length control. The claim "the per-round entropy fall = a resonant attractor" must be
 * earned: a falling trend is also expected from context length alone (longer/coherent context sharpens
 * the next-token distribution). So run the field twice — (a) the real coherent prior chorus, (b) the
 * SAME context with its chorus tokens shuffled (identical length, broken coherence). The prompt prefix
 * stays intact in both. If COHERENT's entropy falls more across rounds than SHUFFLED's, the extra drop
 * is coupling beyond length; if they fall equally, the trend is a length artifact. Reported as measured. */
static void field_resonance_test(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_cells, int nfrag, int n_rounds, int eos) {
    int max_seq = 512, ids[512], pids[512];
    int np_prompt = bpe_encode(tok, prompt, pids, max_seq);
    char frag[2048];
    if (n_rounds < 1) n_rounds = 1;
    printf("\n=== resonance test: coherent vs same-length SHUFFLED context (%d cells × %d rounds, prompt \"%s\") ===\n", n_cells, n_rounds, prompt);
    float first[2] = {0,0}, last[2] = {0,0};
    for (int control = 0; control < 2; control++) {
        char prev_chorus[4096]; prev_chorus[0] = 0; char ctx[8704];
        printf("  [%s]\n", control ? "SHUFFLED (length-matched, incoherent)" : "COHERENT (real chorus)");
        for (int r = 0; r < n_rounds; r++) {
            char this_chorus[4096]; int tc = 0; this_chorus[0] = 0; float ent_sum = 0;
            for (int c = 0; c < n_cells; c++) {
                snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus, this_chorus);
                int np = bpe_encode(tok, ctx, ids, max_seq - nfrag - 1);
                if (control && np > np_prompt) shuffle_ids(ids, np_prompt, np, 12345u + (unsigned)(r * 31 + c));
                float temp = 0.6f + 0.7f * (n_cells > 1 ? (float)c / (n_cells - 1) : 0.5f);
                ent_sum += cell_speak(m, tok, ids, np, nfrag, temp, 40, 1.4f,
                                      42u + (unsigned)(r * 1000 + c) * 7919u, eos, max_seq, frag, sizeof(frag), 0, NULL, NULL, NULL, NULL, NULL);
                int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
                if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            }
            float avg = ent_sum / n_cells;
            printf("    round %d avg entropy = %.3f\n", r + 1, avg);
            if (r == 0) first[control] = avg; last[control] = avg;
            strncpy(prev_chorus, this_chorus, sizeof(prev_chorus) - 1); prev_chorus[sizeof(prev_chorus) - 1] = 0;
        }
    }
    float drop_coh = first[0] - last[0], drop_shuf = first[1] - last[1];
    printf("\n  coherent drop %.3f vs shuffled drop %.3f  →  resonance-beyond-length = %.3f %s\n",
           drop_coh, drop_shuf, drop_coh - drop_shuf,
           drop_coh - drop_shuf > 0.10f ? "(coherent falls MORE — coupling beyond length)" :
           drop_coh - drop_shuf < -0.10f ? "(shuffled falls more — NO resonance signal)" : "(~equal — length artifact, claim NOT earned yet)");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> field [cells] [frag] [rounds] [alpha] [leap] [xcell] [chorus] [xrep] [life] [kvshuf] [qloop]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> restest [cells] [frag] [rounds]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> life [ticks] [frag] [init_cells]\n", argv[0]);
        return 1;
    }
    const char *prompt = argc > 2 ? argv[2] : "What is resonance?";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 510) max_tokens = 510;   /* C-5: keep bpe_encode cap (max_seq - max_tokens - 1) >= 1 */
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.8f;
    srand(42);

    double t0 = now_ms();
    gguf_file *gf = gguf_open(argv[1]); if (!gf) return 1;
    model_t *m = model_load(gf); if (!m) return 1;
    bpe_tokenizer *tok = bpe_load(argv[1]); if (!tok) { fprintf(stderr, "bpe_load failed\n"); return 1; }
    if (bpe_n_vocab(tok) > m->vocab) { fprintf(stderr, "chorus: tokenizer vocab %d > embedding vocab %d — token ids would index tok_emb out of bounds\n", bpe_n_vocab(tok), m->vocab); return 1; }   /* C-9 */
    int eos = -1; const gguf_kv *e = gguf_get_kv(gf, "tokenizer.ggml.eos_token_id"); if (e) eos = (int)e->val.u32;
    printf("loaded in %.0f ms (vocab=%d eos=%d) -- arianna.q heart, single file, no -lnotorch\n", now_ms() - t0, bpe_n_vocab(tok), eos);

    /* δ-field chorus mode:  ./arianna-q <gguf> <prompt> field [n_cells] [nfrag] [n_rounds] */
    if (argc > 3 && strcmp(argv[3], "field") == 0) {
        int n_cells  = argc > 4 ? atoi(argv[4]) : 5;
        int nfrag    = argc > 5 ? atoi(argv[5]) : 16;
        int n_rounds = argc > 6 ? atoi(argv[6]) : 1;
        float alpha  = argc > 7 ? (float)atof(argv[7]) : 0.0f;   /* soma coupling strength (0 = text-only baseline) */
        g_leap_mode  = argc > 8 ? atoi(argv[8]) : 2;             /* leap-v2 — RELAY-ONLY (no-op under the default chorus; lives only when g_chorus=0) */
        g_xcell      = argc > 9 ? (float)atof(argv[9]) : 0.3f;   /* DEFAULT ALIVE: λ=0.3 balanced cross-cell. 0 = off */
        g_chorus     = argc > 10 ? atoi(argv[10]) : 1;           /* DEFAULT: 1 = chorus (each cell own answer). 0 = relay */
        g_xrep       = argc > 11 ? (float)atof(argv[11]) : 1.3f; /* cross-cell rep-penalty: don't echo neighbours' words (1=off) */
        g_life_on    = argc > 12 ? atoi(argv[12]) : 0;           /* δ-life: 1 = measure/run Game of Life (incr.0 = log fitness inputs) */
        g_kvshuf     = argc > 13 ? atoi(argv[13]) : (g_chorus && g_xcell > 0 ? 1 : 0);  /* default chorus probe: Δ_R^kv + permutation floor */
        g_qloop      = argc > 14 ? atoi(argv[14]) : (g_chorus ? 2 : 0);  /* 0=off; 1..2 resonant question routes */
        if (g_qloop < 0) g_qloop = 0;
        if (g_qloop > 2) g_qloop = 2;
        if (n_cells <= 0) {   /* auto: the field sizes itself from the prompt's entropy */
            float pe = probe_entropy(m, tok, prompt);
            n_cells = (int)(pe + 0.5f); if (n_cells < 1) n_cells = 1; if (n_cells > 8) n_cells = 8;
            printf("auto: prompt entropy %.2f -> %d cells (low=collapse, high=bloom)\n", pe, n_cells);
        }
        field_chorus(m, tok, prompt, n_cells, nfrag, n_rounds, eos, alpha);
        return 0;
    }

    /* resonance-vs-length control:  ./arianna-q <gguf> <prompt> restest [n_cells] [nfrag] [n_rounds] */
    if (argc > 3 && strcmp(argv[3], "restest") == 0) {
        int n_cells  = argc > 4 ? atoi(argv[4]) : 4;
        int nfrag    = argc > 5 ? atoi(argv[5]) : 12;
        int n_rounds = argc > 6 ? atoi(argv[6]) : 3;
        field_resonance_test(m, tok, prompt, n_cells, nfrag, n_rounds, eos);
        return 0;
    }

    /* δ-life Game of Life:  ./arianna-q <gguf> <prompt> life [ticks] [nfrag] [init_cells] */
    if (argc > 3 && strcmp(argv[3], "life") == 0) {
        int ticks = argc > 4 ? atoi(argv[4]) : 8;
        int nfrag = argc > 5 ? atoi(argv[5]) : 16;
        int init  = argc > 6 ? atoi(argv[6]) : 4;
        field_life(m, tok, prompt, init, ticks, nfrag, eos);
        return 0;
    }

    int max_seq = 512;
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = xzalloc(m->vocab, sizeof(float));
    int ids[512]; int n = bpe_encode(tok, prompt, ids, max_seq - max_tokens - 1);
    printf("\nprompt: \"%s\" (%d tokens, temp=%.2f)\n---\n%s", prompt, n, temp, prompt); fflush(stdout);

    double g0 = now_ms();
    for (int i = 0; i < n; i++) forward(m, kv, ids[i], i, logits);
    double prefill = now_ms() - g0;
    int gen = 0; char buf[256];
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, m->vocab, temp);
        if (next == eos) break;
        bpe_decode_token(tok, next, buf, sizeof(buf)); printf("%s", buf); fflush(stdout); gen++;
        int pos = n + step; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits);
    }
    double total = now_ms() - g0;
    printf("\n---\nprefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s)\n",
           n, prefill, n*1000.0/prefill, gen, total-prefill, gen > 0 ? gen*1000.0/(total-prefill) : 0);
    return 0;
}
