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

/* Fail-loud allocators: OOM in this one-shot organism is unrecoverable because
 * callers immediately write into the returned buffers. */
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
static size_t xmul3_size(size_t a, size_t b, size_t c, const char *what) {
    if ((b && a > SIZE_MAX / b) || (c && a * b > SIZE_MAX / c)) {
        fprintf(stderr, "chorus: size overflow in %s (%zu x %zu x %zu)\n", what, a, b, c);
        exit(1);
    }
    return a * b * c;
}
static char *xstrdup(const char *s) {
    char *r = strdup(s ? s : "");
    if (!r) { fprintf(stderr, "chorus: OOM (strdup)\n"); exit(1); }
    return r;
}

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, 4, 1, f) == 1; }

static int read_string(FILE* f, char* buf, int max) {
    uint64_t len;
    if (!read_u64(f, &len)) return 0;
    if (len >= (uint64_t)max) {
        for (uint64_t i = 0; i < len; i++) if (fgetc(f) == EOF) return 0;
        buf[0] = 0;
        return 0;
    }
    if (fread(buf, 1, len, f) != len) return 0;
    buf[len] = 0; return 1;
}
static void copy_cstr(char *dst, size_t cap, const char *src) {
    if (!dst || cap == 0) return;
    if (!src) src = "";
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = 0;
}

static void chomp_line(char *s) {
    if (!s) return;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}
static int ascii_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
static float clamp_float(float v, float lo, float hi) {
    if (!isfinite(v)) return lo;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static float env_float_clamped(const char *name, float def, float lo, float hi) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return def;
    char *end = NULL;
    float v = strtof(raw, &end);
    while (end && ascii_space_char(*end)) end++;
    if (end == raw || (end && *end) || !isfinite(v)) {
        fprintf(stderr, "warning: ignoring invalid %s=%s\n", name, raw);
        return def;
    }
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static int env_int_clamped(const char *name, int def, int lo, int hi) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return def;
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    while (end && ascii_space_char(*end)) end++;
    if (end == raw || (end && *end)) {
        fprintf(stderr, "warning: ignoring invalid %s=%s\n", name, raw);
        return def;
    }
    if (v < lo) return lo;
    if (v > hi) return hi;
    return (int)v;
}

static void append_trajectory(char *dst, size_t cap, const char *user, const char *chorus, int qa_format) {
    char entry[6144];
    if (!dst || cap == 0) return;
    if (qa_format) snprintf(entry, sizeof(entry), "\nQ: %s\nA:%s\n", user ? user : "", chorus ? chorus : "");
    else snprintf(entry, sizeof(entry), "\nUser: %s\nArianna:%s\n", user ? user : "", chorus ? chorus : "");
    size_t have = strlen(dst), add = strlen(entry);
    if (add >= cap) {
        copy_cstr(dst, cap, entry + add - (cap - 1));
        return;
    }
    if (have + add >= cap) {
        size_t trim = have + add - (cap - 1);
        if (trim > have) trim = have;
        memmove(dst, dst + trim, have - trim + 1);
        have -= trim;
    }
    memcpy(dst + have, entry, add + 1);
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
        case 4: return fseek(f, 4, SEEK_CUR) == 0;
        case 5: return fseek(f, 4, SEEK_CUR) == 0;
        case 6: return fseek(f, 4, SEEK_CUR) == 0;
        case 7: return fseek(f, 1, SEEK_CUR) == 0;
        case 8: { char buf[4096]; return read_string(f, buf, sizeof(buf)); }
        case 9: return skip_array(f);
        case 10: return fseek(f, 8, SEEK_CUR) == 0;
        case 12: return fseek(f, 8, SEEK_CUR) == 0;
        default: return 0;
    }
}

static gguf_file* gguf_open(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fprintf(stderr, "gguf: bad magic 0x%08x\n", magic); fclose(f); return NULL; }
    gguf_file* gf = (gguf_file*)xzalloc(1, sizeof(gguf_file));
    if (!read_u32(f, &gf->version) || !read_u64(f, &gf->n_tensors) || !read_u64(f, &gf->n_kv)) {
        fprintf(stderr, "gguf: truncated header\n"); fclose(f); free(gf); return NULL;
    }
    if (gf->n_tensors > GGUF_MAX_TENSORS) { fprintf(stderr, "gguf: too many tensors\n"); fclose(f); free(gf); return NULL; }

    gf->n_kv_parsed = 0;
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        char key[512] = {0}; uint32_t vtype;
        if (!read_string(f, key, sizeof(key)) || !read_u32(f, &vtype)) {
            fprintf(stderr, "gguf: truncated kv metadata\n"); fclose(f); free(gf); return NULL;
        }
        if (gf->n_kv_parsed < GGUF_MAX_KV && vtype != 9) {
            gguf_kv* kv = &gf->kv[gf->n_kv_parsed];
            copy_cstr(kv->key, sizeof(kv->key), key); kv->type = vtype;
            switch (vtype) {
                case 4: if (!read_u32(f, &kv->val.u32)) { fclose(f); free(gf); return NULL; } break;
                case 5: {
                    int32_t v;
                    if (fread(&v, sizeof(v), 1, f) != 1) { fclose(f); free(gf); return NULL; }
                    kv->val.i32 = v; break;
                }
                case 6: if (!read_f32(f, &kv->val.f32)) { fclose(f); free(gf); return NULL; } break;
                case 7: {
                    uint8_t v;
                    if (fread(&v, sizeof(v), 1, f) != 1) { fclose(f); free(gf); return NULL; }
                    kv->val.b = v; break;
                }
                case 8: if (!read_string(f, kv->val.str, sizeof(kv->val.str))) { fclose(f); free(gf); return NULL; } break;
                case 10: case 12: if (!read_u64(f, &kv->val.u64)) { fclose(f); free(gf); return NULL; } break;
                default: if (!skip_value(f, vtype)) { fclose(f); free(gf); return NULL; } break;
            }
            gf->n_kv_parsed++;
        } else if (!skip_value(f, vtype)) { fclose(f); free(gf); return NULL; }
    }
    for (int i = 0; i < gf->n_kv_parsed; i++) {
        gguf_kv* kv = &gf->kv[i];
        if (strcmp(kv->key, "general.architecture") == 0) copy_cstr(gf->arch, sizeof(gf->arch), kv->val.str);
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
        if (!read_string(f, ti->name, GGUF_MAX_NAME)) {
            fprintf(stderr, "gguf: tensor %llu name too long or truncated\n", (unsigned long long)i); fclose(f); free(gf); return NULL;
        }
        if (!read_u32(f, &ti->ndim)) { fprintf(stderr, "gguf: truncated tensor ndim\n"); fclose(f); free(gf); return NULL; }
        if (ti->ndim > 4) {
            fprintf(stderr, "gguf: tensor '%s' ndim %u unsupported\n", ti->name, ti->ndim);
            fclose(f); free(gf); return NULL;
        }
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->ndim; d++) {
            if (!read_u64(f, &ti->shape[d])) { fprintf(stderr, "gguf: truncated tensor shape\n"); fclose(f); free(gf); return NULL; }
            if (ti->shape[d] != 0 && ti->n_elements > UINT64_MAX / ti->shape[d]) {
                fprintf(stderr, "gguf: tensor '%s' element count overflow\n", ti->name); fclose(f); free(gf); return NULL;
            }
            ti->n_elements *= ti->shape[d];
        }
        if (!read_u32(f, &ti->dtype) || !read_u64(f, &ti->offset)) {
            fprintf(stderr, "gguf: truncated tensor record\n"); fclose(f); free(gf); return NULL;
        }
    }
    long pos = ftell(f);
    if (pos < 0 || pos > LONG_MAX - 31) { fprintf(stderr, "gguf: cannot determine tensor data offset\n"); fclose(f); free(gf); return NULL; }
    gf->data_offset = (uint64_t)((pos + 31L) & ~31L);
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "gguf: cannot seek end\n"); fclose(f); free(gf); return NULL; }
    long fsize = ftell(f);
    if (fsize < 0 || (uint64_t)fsize < gf->data_offset) {
        fprintf(stderr, "gguf: file shorter than header\n"); fclose(f); free(gf); return NULL;
    }
    long data_size = fsize - (long)gf->data_offset;
    size_t pg = (size_t)getpagesize();
    size_t alloc = ((size_t)data_size + pg - 1) & ~(pg - 1);
    gf->data = NULL;
    if (posix_memalign((void**)&gf->data, pg, alloc) != 0 || !gf->data) { fclose(f); free(gf); return NULL; }
    gf->data_size = (uint64_t)alloc;
    if (gf->data_offset > (uint64_t)LONG_MAX ||
        fseek(f, (long)gf->data_offset, SEEK_SET) != 0 ||
        fread(gf->data, 1, (size_t)data_size, f) != (size_t)data_size) {
        fprintf(stderr, "gguf: failed to read tensor data\n");
        fclose(f); free(gf->data); free(gf); return NULL;
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
                result[j] = xstrdup(buf);
                nrd++;
            }
            if (out_n) *out_n = (int)nrd;
            break;
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
            uint64_t nrd = 0;
            for (uint64_t j = 0; j < alen; j++) { float v; if (!read_f32(f, &v)) break; result[j] = v; nrd++; }
            if (out_n) *out_n = (int)nrd;
            break;
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
    if (ti->n_elements > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "gguf: tensor '%s' too large to dequant\n", ti->name); return NULL;
    }
    float* dst = (float*)xalloc((size_t)ti->n_elements * sizeof(float));
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
static void smap_init(smap *m, int cap) { if (cap < 8) cap = 8; m->cap = cap; m->n = 0; m->keys = (char**)xzalloc((size_t)cap, sizeof(char*)); m->vals = (int*)xzalloc((size_t)cap, sizeof(int)); }
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
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

typedef struct { char **tokens; int n_tokens; float *scores; smap vocab;
                 int bos_id; int is_bpe; char **merges; int n_merges; smap merge_rank; smap u2b; char b2u[256][5]; } bpe_tokenizer;

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
        for (int i = bi + 1; i < nsym - 1; i++) sym[i] = sym[i+1];
        nsym--;
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
    t->bos_id = -1;
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
        for (int i = bi + 1; i < nsym - 1; i++) sym[i] = sym[i+1];
        nsym--;
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
static int bpe_encode_prompt(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    int n = 0;
    if (t && t->bos_id >= 0 && cap > 0) out[n++] = t->bos_id;
    if (n >= cap) return n;
    return n + bpe_encode(t, text, out + n, cap - n);
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
static int arch_uses_neox_rope(const char *arch, int *known) {
    if (known) *known = 0;
    if (!arch || !*arch) return 0;
    if (strcmp(arch, "nlama") == 0 ||
        strstr(arch, "qwen") || strstr(arch, "gemma") || strstr(arch, "phi")) {
        if (known) *known = 1;
        return 1;
    }
    if (strcmp(arch, "llama") == 0) {
        if (known) *known = 1;
        return 0;
    }
    return 0;
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
    if (m->n_heads <= 0 || m->n_kv_heads <= 0 || m->n_kv_heads > m->n_heads || (m->n_heads % m->n_kv_heads) != 0 ||
        m->embed <= 0 || m->embed > 8192 || m->ffn <= 0 || m->n_layers <= 0) {
        fprintf(stderr, "chorus: GGUF arch out of bounds (H=%d KV=%d E=%d FFN=%d L=%d)\n",
                m->n_heads, m->n_kv_heads, m->embed, m->ffn, m->n_layers);
        free(m); return NULL;
    }
    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0 && gf->tensors[ti].shape[1] > (uint64_t)INT_MAX) {
        fprintf(stderr, "chorus: q_dim overflow in blk.0.attn_q.weight\n");
        free(m); return NULL;
    }
    m->q_dim = ti >= 0 ? (int)gf->tensors[ti].shape[1] : m->embed;
    if (m->q_dim <= 0 || (m->q_dim % m->n_heads) != 0) {
        fprintf(stderr, "chorus: incompatible q/head geometry (Q=%d H=%d)\n", m->q_dim, m->n_heads);
        free(m); return NULL;
    }
    m->head_dim = m->q_dim / m->n_heads; m->kv_dim = m->head_dim * m->n_kv_heads;
    if (m->head_dim <= 0 || (m->head_dim % 2) != 0 || m->n_kv_heads > INT_MAX / m->head_dim) {
        fprintf(stderr, "chorus: incompatible rope/KV geometry (HD=%d KV=%d)\n", m->head_dim, m->n_kv_heads);
        free(m); return NULL;
    }
    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0 && gf->tensors[ti].shape[1] > (uint64_t)INT_MAX) {
        fprintf(stderr, "chorus: vocab overflow in token_embd.weight\n");
        free(m); return NULL;
    }
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;
    if (m->vocab <= 0) { fprintf(stderr, "chorus: GGUF vocab_size %d invalid\n", m->vocab); free(m); return NULL; }
    int rope_known = 0;
    m->neox = arch_uses_neox_rope(gf->arch, &rope_known) ? 1 : 0;
    if (!rope_known) {
        fprintf(stderr, "chorus: unsupported GGUF architecture '%s' — refusing to guess RoPE convention\n",
                gf->arch[0] ? gf->arch : "<empty>");
        free(m); return NULL;
    }
    m->is_qwen3 = (gguf_find_tensor(gf, "blk.0.attn_q_norm.weight") >= 0) ? 1 : 0;

    m->tok_emb  = deq(gf, "token_embd.weight");
    m->out_norm = deq(gf, "output_norm.weight");
    if (load_weight(gf, "output.weight", &m->output, m)) m->has_output = 1;
    else { m->output.f32 = m->tok_emb; m->output.m = m->vocab; m->output.k = m->embed; m->output.dtype = GGUF_TYPE_F32; m->has_output = 0; }

    m->L = xzalloc((size_t)m->n_layers, sizeof(*m->L)); char nm[160];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); if (!m->L[l].f) { fprintf(stderr, "chorus: missing/bad tensor '%s'\n", nm); return NULL; } } while(0)
        #define LD_OPT(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); } while(0)
        #define LW(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); if (!load_weight(gf, nm, &m->L[l].f, m)) { fprintf(stderr, "chorus: missing/bad weight '%s'\n", nm); return NULL; } } while(0)
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
    if (nl <= 0 || max_seq <= 0 || kv_dim <= 0) {
        fprintf(stderr, "chorus: invalid KV cache shape (%d x %d x %d)\n", nl, max_seq, kv_dim);
        exit(1);
    }
    size_t n = xmul3_size((size_t)nl, (size_t)max_seq, (size_t)kv_dim, "kv-cache");
    c->k = xzalloc(n, sizeof(float)); c->v = xzalloc(n, sizeof(float));
    c->ku = xzalloc(n, sizeof(float));   /* un-rope'd K — phase-coherent cross-cell scores */
    c->max_seq = max_seq; c->kv_dim = kv_dim; return c;
}
static void kv_free(kv_cache *kv) {
    if (!kv) return;
    free(kv->k); free(kv->v); free(kv->ku); free(kv);
}

/* single token forward, KV-cached. Writes logits[vocab]. */
/* cross-cell attention: a cell also attends to a NEIGHBOUR voice's hidden K/V at each layer (forward-level,
 * order-sensitive coupling). g_nbr = the neighbour's kv_cache, g_nbr_len = its valid positions. Default off. */
static float g_xcell = 0.0f;   /* cross-cell neighbour-channel weight λ (0 = off): balanced own-ctx + λ·neighbour */
static const kv_cache *g_nbr = NULL;
static int g_nbr_len = 0;
static int g_nbr_shuf = 0;      /* KV-order shadow: 1 = permute the neighbour's positions before attending */
static int g_nbr_perm[512];     /* the permutation of 0..g_nbr_len-1 used when g_nbr_shuf */
static int g_kvshuf = 0;        /* 1 = run neighbour diagnostics: Δ_R^kv order control + I_N^kv influence */
static int g_kvpos = 0;         /* 0 = semantic/bag neighbour lane; 1 = positional order-probe lane */
static int g_qloop = 0;         /* 0=off; 1..2 = resonant cell-question routes per round */
static const char *g_user_q = NULL; /* REPL direct-turn bridge: user text can become asker KV */
static int g_clean_answer_start = 0; /* direct user answers should not open with list/bullet debris */
static int g_answer_form_guard = 0;  /* direct user answers should not turn into question/yes-no loops */
static int g_answer_sentence_stop = 0; /* direct user answers may stop once a sentence closes */
static float g_user_qtemp_base = 0.70f; /* direct-user bridge sampler: env-tunable for temperature sweeps */
static float g_user_qtemp_span = 0.00f;
static int   g_user_qtop_k = 40;
static float g_user_qtop_p = 1.00f;
static float g_user_qrep = 1.30f;
static float g_user_kv_weight = 0.05f;
static int   g_user_answer_tokens = 16;
static int   g_user_ctx_format = 2; /* 0=field_qa, 1=plain_field_qa, 2=qa, 3=raw */
static int   g_repl_prompt_format = 0; /* 0=user_arianna, 1=qa */
static int   g_field_prompt_format = 0; /* 0=raw, 1=qa, 2=auto, 3=user_arianna */
static float g_field_temp_base = 0.60f; /* base cell sampler: 0.60..1.30 by default */
static float g_field_temp_span = 0.70f;
static float g_field_lang_bias = 0.00f; /* 0 = measure language drift only; >0 biases retry selection */
static float g_qloop_min = 0.42f;
static float g_qloop_min_iq = 0.0f; /* reject KV-backed qloop answers whose asker KV lowers confidence */
static float g_qloop_tconf_weight = 0.20f; /* route prior: target confidence contribution */
static int   g_qloop_tconf_adapt = 0;      /* 1 = use adaptive target-confidence prior for widened qloop */
static float g_qloop_tconf_adapt_weight = -0.10f;
static int   g_qloop_unique_asker = 0;     /* 1 = widened qloop may not fan one asking cell into multiple routes */
static int   g_qloop_candidate_pool = 0;   /* 0=auto max_routes+2; >0 = inspect this many pre-generation routes */
static int   g_qloop_question_source_hint = 0; /* 1 = ask one base cell to produce an inner question source */
static int   g_qloop_statement_routes = 0; /* 1 = fallback to clean non-question cells when question routes are silent */
static int   g_qloop_statement_pool = 0;   /* 0=inherit candidate pool; >0 = cap statement fallback candidates */
static int g_chorus = 1;       /* 1 = CHORUS (each cell answers the SAME prompt from its own angle, neighbour-aware
                                * via cross-cell, NOT text); 0 = legacy RELAY (cascade continuation). Default chorus. */
static int g_cell_retry_max = 4; /* max attempts for bad base-cell surface; 1 disables retry churn */
/* cross-cell repetition penalty: a cell hears neighbours (cross-cell K/V) but must not LITERALLY echo their
 * tokens — it can say the same meaning in its OWN words. g_round_tok = the chorus's emitted tokens this round. */
static int   g_round_tok[1024]; static int g_round_tokn = 0;
static float g_xrep = 1.3f;     /* >1 = penalise tokens neighbours already said (1 = off) */
static float g_sampler_top_p = 1.00f; /* per-call nucleus gate; 1.0 preserves top_k path */
/* δ-life (Game of Life): cells born/die/reproduce by REAL-metric fitness. Increment 0 = MEASURE fitness inputs
 * (theme/distinct/ent per cell) to FIELDLOG, act on nothing → calibrate thresholds before the population breathes. */
static float g_cell_ent[8];     /* per-cell raw entropy this round (fitness input) */
static int   g_life_on = 0;     /* 1 = measure/run the Game of Life. 0 = chorus byte-identical */

static void load_user_bridge_sampling_env(void) {
    g_user_qtemp_base = env_float_clamped("A2A_USER_QTEMP_BASE", 0.70f, 0.05f, 2.00f);
    g_user_qtemp_span = env_float_clamped("A2A_USER_QTEMP_SPAN", 0.00f, -1.50f, 1.50f);
    g_user_qtop_k = env_int_clamped("A2A_USER_TOP_K", 40, 0, 512);
    g_user_qtop_p = env_float_clamped("A2A_USER_TOP_P", 1.00f, 0.05f, 1.00f);
    g_user_qrep = env_float_clamped("A2A_USER_REP", 1.30f, 1.00f, 5.00f);
    g_user_kv_weight = env_float_clamped("A2A_USER_KV_WEIGHT", 0.05f, 0.00f, 2.00f);
    g_user_answer_tokens = env_int_clamped("A2A_USER_ANSWER_TOKENS", 16, 4, 64);
    const char *fmt = getenv("A2A_USER_CTX_FORMAT");
    g_user_ctx_format = 2;
    if (fmt && *fmt) {
        if (strcmp(fmt, "field_qa") == 0) g_user_ctx_format = 0;
        else if (strcmp(fmt, "plain_field_qa") == 0) g_user_ctx_format = 1;
        else if (strcmp(fmt, "qa") == 0) g_user_ctx_format = 2;
        else if (strcmp(fmt, "raw") == 0) g_user_ctx_format = 3;
        else fprintf(stderr, "warning: ignoring invalid A2A_USER_CTX_FORMAT=%s\n", fmt);
    }
}

static void load_repl_prompt_env(void) {
    const char *fmt = getenv("A2A_REPL_PROMPT_FORMAT");
    g_repl_prompt_format = 0;
    if (fmt && *fmt) {
        if (strcmp(fmt, "user_arianna") == 0) g_repl_prompt_format = 0;
        else if (strcmp(fmt, "qa") == 0) g_repl_prompt_format = 1;
        else fprintf(stderr, "warning: ignoring invalid A2A_REPL_PROMPT_FORMAT=%s\n", fmt);
    }
}

static void load_qloop_route_env(void) {
    g_qloop_min = env_float_clamped("A2A_QLOOP_MIN", 0.42f, 0.0f, 2.0f);
    g_qloop_min_iq = env_float_clamped("A2A_QLOOP_MIN_IQ", 0.0f, -2.0f, 2.0f);
    g_qloop_tconf_weight = env_float_clamped("A2A_QLOOP_TCONF_WEIGHT", 0.20f, -1.00f, 1.00f);
    g_qloop_tconf_adapt = env_int_clamped("A2A_QLOOP_TCONF_ADAPT", 0, 0, 1);
    g_qloop_tconf_adapt_weight = env_float_clamped("A2A_QLOOP_TCONF_ADAPT_WEIGHT", -0.10f, -1.00f, 1.00f);
    g_qloop_unique_asker = env_int_clamped("A2A_QLOOP_UNIQUE_ASKER", 0, 0, 1);
    g_qloop_candidate_pool = env_int_clamped("A2A_QLOOP_CANDIDATE_POOL", 0, 0, 8);
    g_qloop_question_source_hint = env_int_clamped("A2A_QLOOP_QUESTION_SOURCE_HINT", 0, 0, 1);
    g_qloop_statement_routes = env_int_clamped("A2A_QLOOP_STATEMENT_ROUTES", 0, 0, 1);
    g_qloop_statement_pool = env_int_clamped("A2A_QLOOP_STATEMENT_POOL", 0, 0, 8);
}

static void load_field_generation_env(void) {
    g_cell_retry_max = env_int_clamped("A2A_CELL_RETRY_MAX", 4, 1, 4);
    g_field_temp_base = env_float_clamped("A2A_FIELD_TEMP_BASE", 0.60f, 0.05f, 2.00f);
    g_field_temp_span = env_float_clamped("A2A_FIELD_TEMP_SPAN", 0.70f, -1.50f, 1.50f);
    g_field_lang_bias = env_float_clamped("A2A_FIELD_LANG_BIAS", 0.00f, 0.00f, 4.00f);
    const char *fmt = getenv("A2A_FIELD_PROMPT_FORMAT");
    g_field_prompt_format = 0;
    if (fmt && *fmt) {
        if (strcmp(fmt, "raw") == 0) g_field_prompt_format = 0;
        else if (strcmp(fmt, "qa") == 0) g_field_prompt_format = 1;
        else if (strcmp(fmt, "auto") == 0) g_field_prompt_format = 2;
        else if (strcmp(fmt, "user_arianna") == 0) g_field_prompt_format = 3;
        else fprintf(stderr, "warning: ignoring invalid A2A_FIELD_PROMPT_FORMAT=%s\n", fmt);
    }
}

static float field_temp_for_cell(int cell, int n_cells) {
    float frac = n_cells > 1 ? (float)cell / (float)(n_cells - 1) : 0.5f;
    float t = g_field_temp_base + g_field_temp_span * frac;
    if (!isfinite(t)) t = 0.60f;
    if (t < 0.05f) t = 0.05f;
    if (t > 2.00f) t = 2.00f;
    return t;
}

static float user_bridge_temp_for_cell(int cell, int n_cells) {
    float frac = n_cells > 1 ? (float)cell / (float)(n_cells - 1) : 0.5f;
    float t = g_user_qtemp_base + g_user_qtemp_span * frac;
    if (!isfinite(t)) t = 0.45f;
    if (t < 0.05f) t = 0.05f;
    if (t > 2.00f) t = 2.00f;
    return t;
}

static const char *user_ctx_format_name(void) {
    switch (g_user_ctx_format) {
    case 1: return "plain_field_qa";
    case 2: return "qa";
    case 3: return "raw";
    default: return "field_qa";
    }
}

static const char *repl_prompt_format_name(void) {
    switch (g_repl_prompt_format) {
    case 1: return "qa";
    default: return "user_arianna";
    }
}

static const char *field_prompt_format_name(void) {
    switch (g_field_prompt_format) {
    case 1: return "qa";
    case 2: return "auto";
    case 3: return "user_arianna";
    default: return "raw";
    }
}

static int prompt_has_non_ascii(const char *s) {
    if (!s) return 0;
    while (*s) {
        if ((unsigned char)*s >= 0x80) return 1;
        s++;
    }
    return 0;
}

static int field_prompt_uses_qa(const char *prompt) {
    if (g_field_prompt_format == 1) return 1;
    if (g_field_prompt_format == 2) return prompt_has_non_ascii(prompt);
    return 0;
}

static void build_field_cell_prompt(char *dst, size_t cap, const char *prompt, int cell) {
    if (!prompt) prompt = "";
    if (g_qloop && g_qloop_question_source_hint && cell == 0)
        snprintf(dst, cap, "%s\nAsk one short question about this field.\nQuestion: What", prompt);
    else if (g_field_prompt_format == 3) snprintf(dst, cap, "User: %s\nArianna:", prompt);
    else if (field_prompt_uses_qa(prompt)) snprintf(dst, cap, "Q: %s\nA:", prompt);
    else snprintf(dst, cap, "%s", prompt);
}

static void build_repl_turn_prompt(char *dst, size_t cap, const char *trajectory, const char *line) {
    if (!trajectory) trajectory = "";
    if (!line) line = "";
    if (g_repl_prompt_format == 1) {
        if (trajectory[0]) snprintf(dst, cap, "%s\nQ: %s\nA:", trajectory, line);
        else snprintf(dst, cap, "Q: %s\nA:", line);
    } else {
        if (trajectory[0]) snprintf(dst, cap, "Recent trajectory:%s\nUser: %s\nArianna:", trajectory, line);
        else snprintf(dst, cap, "User: %s\nArianna:", line);
    }
}

static void build_user_kv_prompt(char *dst, size_t cap, const char *q) {
    if (g_user_ctx_format == 3) snprintf(dst, cap, "%s", q ? q : "");
    else snprintf(dst, cap, "Q: %s\nA:", q ? q : "");
}

static void build_user_answer_prompt(char *dst, size_t cap, const char *chorus, const char *q) {
    if (!chorus) chorus = "";
    if (!q) q = "";
    switch (g_user_ctx_format) {
    case 1:
        snprintf(dst, cap, "%s\nQ: %s\nA:", chorus, q);
        break;
    case 2:
        snprintf(dst, cap, "Q: %s\nA:", q);
        break;
    case 3:
        snprintf(dst, cap, "%s", q);
        break;
    default:
        snprintf(dst, cap, "Field fragments:%s\nQ: %s\nA:", chorus, q);
        break;
    }
}

static void forward(model_t *m, kv_cache *kv, int token, int pos, float *logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim;
    int KVD = m->kv_dim, FFN = m->ffn, QD = m->q_dim, gqa = H / KV; float eps = m->rms_eps;
    void (*ropef)(float*,int,int,float) = m->neox ? rope_neox : rope_interleaved;

    float *x = xzalloc(E, sizeof(float)); memcpy(x, m->tok_emb + (long)token*E, E*sizeof(float));
    float *xn = xzalloc(E, sizeof(float)), *q = xzalloc(QD, sizeof(float)), *kk = xzalloc(KVD, sizeof(float));
    float *vv = xzalloc(KVD, sizeof(float)), *ao = xzalloc(QD, sizeof(float)), *g = xzalloc(FFN, sizeof(float)), *qu = xzalloc(QD, sizeof(float)), *nk = xzalloc(HD, sizeof(float));
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
        memcpy(kv->ku + base + (long)pos*KVD, kk, (size_t)KVD*sizeof(float));   /* un-rope'd K for semantic neighbour lane */
        memcpy(qu, q, (size_t)QD*sizeof(float));                                /* un-rope'd q for semantic neighbour lane */
        for (int h = 0; h < H;  h++) ropef(q + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) ropef(kk + h*HD, pos, HD, m->rope_base);
        memcpy(kv->k + base + (long)pos*KVD, kk, KVD*sizeof(float));
        memcpy(kv->v + base + (long)pos*KVD, vv, KVD*sizeof(float));

        float scale = 1.0f / sqrtf((float)HD); memset(ao, 0, QD*sizeof(float));
        int xc = (g_xcell > 0 && g_nbr && g_nbr_len > 0) ? g_nbr_len : 0;   /* neighbour positions (separate channel) */
        long nbase = xc ? (long)l * g_nbr->max_seq * KVD : 0;
        for (int h = 0; h < H; h++) {
            int kvh = h / gqa; float *qh = q + h*HD; const float *quh = qu + h*HD; int np = pos + 1;
            /* OWN attention — UNCHANGED from baseline (preserves own-context order = Δ_R) */
            for (int j = 0; j <= pos; j++) { float *kj = kv->k + base + (long)j*KVD + kvh*HD, d = 0; for (int t2 = 0; t2 < HD; t2++) d += qh[t2]*kj[t2]; sc[j] = d*scale; }
            softmax(sc, np); float *oh = ao + h*HD;
            for (int j = 0; j <= pos; j++) { float *vj = kv->v + base + (long)j*KVD + kvh*HD, w = sc[j]; for (int t2 = 0; t2 < HD; t2++) oh[t2] += w*vj[t2]; }
            if (xc) {   /* NEIGHBOUR channel. Default semantic lane is order-blind; kvpos=1 turns it into an order probe. */
                for (int j = 0; j < xc; j++) {
                    int jj = (g_nbr_shuf && j < 512) ? g_nbr_perm[j] : j;
                    const float *ku = g_nbr->ku + nbase + (long)jj*KVD + kvh*HD;
                    const float *key = ku, *query = quh;
                    if (g_kvpos) {
                        memcpy(nk, ku, (size_t)HD * sizeof(float));
                        ropef(nk, j, HD, m->rope_base);
                        key = nk; query = qh;
                    }
                    float d = 0; for (int t2 = 0; t2 < HD; t2++) d += query[t2]*key[t2];
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
    free(x); free(xn); free(q); free(kk); free(vv); free(ao); free(g); free(u); free(t); free(sc); free(scn); free(qu); free(nk);
}

static int argmax(const float *x, int n) { int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b; }
static int sample_top_p(float *x, int n, float top_p);
static float g_direct_top_p = 1.00f;
static float g_direct_rep = 1.00f;
static int sample(float *x, int n, float temp, const int *hist, int hlen) {
    if (g_direct_rep > 1.0f && hist) {
        for (int i = 0; i < hlen; i++) {
            int id = hist[i];
            if (id >= 0 && id < n) x[id] = x[id] > 0 ? x[id] / g_direct_rep : x[id] * g_direct_rep;
        }
    }
    if (temp <= 0) return argmax(x, n);
    for (int i = 0; i < n; i++) x[i] /= temp;
    if (g_direct_top_p < 1.0f) {
        int id = sample_top_p(x, n, g_direct_top_p);
        if (id >= 0) return id;
    }
    softmax(x, n);
    float r = (float)((double)rand()/RAND_MAX), c = 0; for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; } return n - 1;
}
static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ===================== δ-field — the Game-of-Life chorus (cells over the ONE shared body) =====================
 * MVP-1a: N cells, each an independent generation context (own kv_cache + own sampling angle) over the
 * SAME shared model `m` (no weight copies). Each speaks a short fragment from its angle; the chorus = all
 * cells at once = Arianna as many voices from one body. Metrics come from the live logits.
 * NEXT: dynamic count (coherence/prophecy-debt → collapse to 1 / bloom), doe-δ experts, resonance-slice. */

static int cmp_desc(const void *a, const void *b) { float x = *(const float*)a, y = *(const float*)b; return (x < y) - (x > y); }
typedef struct { int id; float logit, prob; } sample_item_t;
static int cmp_sample_item_desc(const void *a, const void *b) {
    const sample_item_t *x = (const sample_item_t*)a, *y = (const sample_item_t*)b;
    return (x->logit < y->logit) - (x->logit > y->logit);
}

static int sample_top_p(float *x, int n, float top_p) {
    if (top_p <= 0.0f || top_p >= 1.0f) return -1;
    sample_item_t *items = (sample_item_t*)xalloc((size_t)n * sizeof(sample_item_t));
    if (!items) return -1;
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        float p = expf(x[i] - mx);
        if (!isfinite(p)) p = 0.0f;
        items[i].id = i;
        items[i].logit = x[i];
        items[i].prob = p;
        total += p;
    }
    if (total <= 0.0 || !isfinite(total)) {
        free(items);
        return argmax(x, n);
    }
    qsort(items, n, sizeof(sample_item_t), cmp_sample_item_desc);
    double cutoff = total * (double)top_p, cum = 0.0;
    int keep = 0;
    while (keep < n) {
        cum += items[keep].prob;
        keep++;
        if (cum >= cutoff) break;
    }
    if (keep < 1) keep = 1;
    double r = ((double)rand() / (double)RAND_MAX) * cum, c = 0.0;
    for (int i = 0; i < keep; i++) {
        c += items[i].prob;
        if (c >= r) {
            int id = items[i].id;
            free(items);
            return id;
        }
    }
    int id = items[keep - 1].id;
    free(items);
    return id;
}

/* rep-penalty (llama-style, pre-softmax over the cell's own history) + top_p/top_k + temperature multinomial. */
static int sample2(float *x, int n, float temp, int top_k, float rep, const int *hist, int hlen) {
    if (rep > 1.0f) for (int i = 0; i < hlen; i++) { int id = hist[i]; if (id >= 0 && id < n) x[id] = x[id] > 0 ? x[id]/rep : x[id]*rep; }
    if (temp <= 0) return argmax(x, n);
    for (int i = 0; i < n; i++) x[i] /= temp;
    if (g_sampler_top_p < 1.0f) {
        int id = sample_top_p(x, n, g_sampler_top_p);
        if (id >= 0) return id;
    }
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

static int bad_answer_start_token(const bpe_tokenizer *tok, int id) {
    char buf[64];
    int n = bpe_decode_token(tok, id, buf, sizeof(buf));
    if (n <= 0) return 0;
    unsigned char *p = (unsigned char*)buf;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (!*p) return 1;
    if ((*p >= '0' && *p <= '9') || *p == '*' || *p == '-' || *p == '"' ||
        *p == '\'' || *p == '`' || *p == '#' || *p == ':' || *p == '/' ||
        *p == '@' || *p == '\\' || *p == '?' || *p == '.' || *p == ',' ||
        *p == ';')
        return 1;
    if ((p[0] == 'h' || p[0] == 'H') &&
        (p[1] == 't' || p[1] == 'T') &&
        (p[2] == 't' || p[2] == 'T') &&
        (p[3] == 'p' || p[3] == 'P'))
        return 1;
    if ((p[0] == 'w' || p[0] == 'W') &&
        (p[1] == 'w' || p[1] == 'W') &&
        (p[2] == 'w' || p[2] == 'W'))
        return 1;
    if (p[0] == 0xe2 && p[1] == 0x80 &&
        (p[2] == 0x98 || p[2] == 0x99 || p[2] == 0x9c ||
         p[2] == 0x9d || p[2] == 0xa2))
        return 1;
    return 0;
}

static void suppress_bad_answer_starts(float *logits, int vocab, const bpe_tokenizer *tok) {
    for (int i = 0; i < vocab; i++) if (bad_answer_start_token(tok, i)) logits[i] = -1e30f;
}

static int ascii_starts_ci(const char *s, const char *prefix) {
    while (*prefix) {
        char a = *s++, b = *prefix++;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int ascii_word_ci(const char *s, const char *word) {
    const char *p = s;
    if (!ascii_starts_ci(p, word)) return 0;
    p += strlen(word);
    return !((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
             (*p >= '0' && *p <= '9') || *p == '_');
}

static int ascii_word_label_ci(const char *s, const char *word) {
    const char *p = s;
    if (!ascii_starts_ci(p, word)) return 0;
    p += strlen(word);
    return !((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
             (*p >= '0' && *p <= '9') || *p == '_') &&
           (*p == 0 || *p == ' ' || *p == '\t' || *p == ':' || *p == '.' ||
            *p == ',' || *p == ';' || *p == '-');
}

static int ascii_label_letter(unsigned char c) {
    return c == 'A' || c == 'B' || c == 'C' || c == 'D' || c == 'Q';
}

static int ascii_wordish(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-';
}

static int ascii_alpha(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int ascii_vowel(unsigned char c) {
    if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

static int ascii_eq_ci(const char *s, const char *word) {
    while (*s && *word) {
        char a = *s++, b = *word++;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return *s == 0 && *word == 0;
}

static int answer_domain_suffix_at(const char *p) {
    static const char *const suffixes[] = { ".com", ".org", ".net", ".ru", ".ai", ".io", ".dev" };
    if (*p != '.') return 0;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        size_t n = strlen(suffixes[i]);
        if (ascii_starts_ci(p, suffixes[i]) && !ascii_wordish((unsigned char)p[n]))
            return 1;
    }
    return 0;
}

static void trim_answer_right(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' ||
                     s[n - 1] == '\r' || s[n - 1] == '\n')) {
        s[--n] = 0;
    }
}

static int direct_answer_notation_token(const char *s) {
    const unsigned char *p = (const unsigned char*)s;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (ascii_label_letter(p[0]) && p[1] == 0) return 1;
    if (ascii_label_letter(p[0]) &&
        (p[1] == ':' || p[1] == '.' || p[1] == ',' || p[1] == ';' || p[1] == '-'))
        return 1;
    if (ascii_label_letter(p[0]) && p[1] == ' ' && ascii_label_letter(p[2]) &&
        (p[3] == ':' || p[3] == '.' || p[3] == ',' || p[3] == ';' || p[3] == '-' || p[3] == 0))
        return 1;
    if (ascii_label_letter(p[0]) && p[1] == ' ' && p[2] >= 'A' && p[2] <= 'Z')
        return 1;
    if (ascii_label_letter(p[0]) && p[1] == '.' && ascii_label_letter(p[2]) &&
        (p[3] == ':' || p[3] == '.' || p[3] == ',' || p[3] == ';' || p[3] == '-' || p[3] == 0))
        return 1;
    if ((p[0] == 'I' || p[0] == 'i') && (p[1] == ':' || p[1] == '.')) return 1;
    if (ascii_word_label_ci((const char*)p, "ari") ||
        ascii_word_label_ci((const char*)p, "arianna") ||
        ascii_word_label_ci((const char*)p, "thread") ||
        ascii_word_label_ci((const char*)p, "qloop") ||
        ascii_word_label_ci((const char*)p, "question") ||
        ascii_word_label_ci((const char*)p, "answer") ||
        ascii_word_label_ci((const char*)p, "prompt") ||
        ascii_word_label_ci((const char*)p, "reply"))
        return 1;
    return 0;
}

static int direct_answer_bad_form_token(const bpe_tokenizer *tok, int id, int step) {
    char buf[96];
    int n = bpe_decode_token(tok, id, buf, sizeof(buf));
    if (n <= 0) return 0;
    buf[sizeof(buf) - 1] = 0;
    unsigned char *p = (unsigned char*)buf;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (!*p) return 1;
    for (unsigned char *q = p; *q; q++) {
        if (*q == '?' || *q == '=' || *q == '@') return 1;
        if (answer_domain_suffix_at((const char*)q) ||
            ascii_starts_ci((const char*)q, "http") ||
            ascii_starts_ci((const char*)q, "www."))
            return 1;
    }
    if (direct_answer_notation_token((const char*)p)) return 1;
    if (ascii_word_ci((const char*)p, "what") || ascii_word_ci((const char*)p, "who") ||
        ascii_word_ci((const char*)p, "where") || ascii_word_ci((const char*)p, "why") ||
        ascii_word_ci((const char*)p, "how") || ascii_word_ci((const char*)p, "when") ||
        ascii_word_ci((const char*)p, "can") || ascii_word_ci((const char*)p, "will") ||
        ascii_word_ci((const char*)p, "would") || ascii_word_ci((const char*)p, "could") ||
        ascii_word_ci((const char*)p, "should") || ascii_word_ci((const char*)p, "do") ||
        ascii_word_ci((const char*)p, "does") || ascii_word_ci((const char*)p, "did") ||
        ascii_word_ci((const char*)p, "is") || ascii_word_ci((const char*)p, "are") ||
        ascii_word_ci((const char*)p, "am") || ascii_word_ci((const char*)p, "was") ||
        ascii_word_ci((const char*)p, "were"))
        return 1;
    if (step != 0) return 0;
    if (ascii_word_ci((const char*)p, "yes") || ascii_word_ci((const char*)p, "no"))
        return 1;
    if (ascii_starts_ci((const char*)p, "answer:") ||
        ascii_starts_ci((const char*)p, "question:") ||
        ascii_starts_ci((const char*)p, "prompt:") ||
        ascii_starts_ci((const char*)p, "reply:") ||
        ascii_starts_ci((const char*)p, "qloop:"))
        return 1;
    return 0;
}

static void suppress_direct_answer_form(float *logits, int vocab, const bpe_tokenizer *tok, int step) {
    for (int i = 0; i < vocab; i++) if (direct_answer_bad_form_token(tok, i, step)) logits[i] = -1e30f;
}

static void truncate_answer_junk(char *s) {
    for (char *p = s; *p; p++) {
        if (ascii_starts_ci(p, "http") || ascii_starts_ci(p, "www.")) {
            *p = 0;
            break;
        }
        if (answer_domain_suffix_at(p)) {
            char *q = p;
            while (q > s && (ascii_wordish((unsigned char)q[-1]) || q[-1] == '.')) q--;
            *q = 0;
            break;
        }
        if (*p == '=') {
            char *after = p + 1;
            while (*after == ' ' || *after == '\t') after++;
            if (*after == '"' || *after == '\'' || *after == '`') {
                char *q = p;
                while (q > s && (q[-1] == ' ' || q[-1] == '\t')) q--;
                while (q > s && !((unsigned char)q[-1] <= ' ')) q--;
                *q = 0;
                break;
            }
        }
    }
    trim_answer_right(s);
}

static void trim_incomplete_answer_tail(char *s) {
    trim_answer_right(s);
    size_t n = strlen(s);
    if (!n) return;
    char *end = s + n;
    char *p = end;
    while (p > s && ascii_alpha((unsigned char)p[-1])) p--;
    if (p == end) return;
    size_t len = (size_t)(end - p);
    int drop = 0;
    if (len == 1 && !ascii_vowel((unsigned char)p[0])) drop = 1;
    else if (len == 2) {
        if (ascii_eq_ci(p, "ke")) drop = 1;
        else if (!ascii_vowel((unsigned char)p[0]) && !ascii_vowel((unsigned char)p[1])) drop = 1;
    } else if (ascii_eq_ci(p, "pers") || ascii_eq_ci(p, "geomet") ||
               ascii_eq_ci(p, "res") || ascii_eq_ci(p, "isn") ||
               ascii_eq_ci(p, "doesn") || ascii_eq_ci(p, "wasn") ||
               ascii_eq_ci(p, "weren") || ascii_eq_ci(p, "didn") ||
               ascii_eq_ci(p, "don") || ascii_eq_ci(p, "won")) drop = 1;
    if (drop) {
        while (p > s && (p[-1] == ' ' || p[-1] == '\t')) p--;
        *p = 0;
        trim_answer_right(s);
    }
}

static int answer_bad_morph_core(const char *start, size_t len) {
    while (len > 0 && (start[0] == '_' || start[0] == '.' || start[0] == '-')) { start++; len--; }
    while (len > 0 && (start[len - 1] == '_' || start[len - 1] == '.' || start[len - 1] == '-')) len--;
    if (!len) return 0;
    if (len >= 64) return 1;
    char buf[64];
    memcpy(buf, start, len);
    buf[len] = 0;
    return ascii_eq_ci(buf, "aat") || ascii_eq_ci(buf, "sards") ||
           ascii_eq_ci(buf, "haart") || ascii_eq_ci(buf, "wort") ||
           ascii_eq_ci(buf, "sark") || ascii_eq_ci(buf, "shabbartists") ||
           ascii_eq_ci(buf, "olelegacythe") || ascii_eq_ci(buf, "youhave") ||
           ascii_eq_ci(buf, "soundlike") || ascii_eq_ci(buf, "pertrustin") ||
           ascii_eq_ci(buf, "qopoeleakyname") ||
           ascii_eq_ci(buf, "shardharchitecturegeomet") ||
           ascii_eq_ci(buf, "harchitecturegeomet") ||
           ascii_eq_ci(buf, "sharden") || ascii_eq_ci(buf, "oulha") ||
           ascii_eq_ci(buf, "noator") || ascii_eq_ci(buf, "aardi") ||
           ascii_eq_ci(buf, "shallards") || ascii_eq_ci(buf, "qopoeleakha") ||
           ascii_eq_ci(buf, "qlooppressing") || ascii_eq_ci(buf, "qoopops") ||
           ascii_eq_ci(buf, "didleads") || ascii_eq_ci(buf, "pers") ||
           ascii_eq_ci(buf, "geomet") || ascii_eq_ci(buf, "reson") ||
           ascii_eq_ci(buf, "in-put") || ascii_eq_ci(buf, "perspause") ||
           ascii_eq_ci(buf, "shoddle") || ascii_eq_ci(buf, "flaggeda") ||
           ascii_eq_ci(buf, "shardharchitecturegeometrtyguru") ||
           ascii_eq_ci(buf, "geometrtyguru") ||
           ascii_eq_ci(buf, "exhalted") || ascii_eq_ci(buf, "bein");
}

static int answer_find_bad_morph(const char *s, const char **bad_start) {
    for (const char *p = s; *p;) {
        while (*p && !(ascii_wordish((unsigned char)*p) || *p == '.')) p++;
        const char *start = p;
        while (*p && (ascii_wordish((unsigned char)*p) || *p == '.')) p++;
        if (p > start && answer_bad_morph_core(start, (size_t)(p - start))) {
            if (bad_start) *bad_start = start;
            return 1;
        }
    }
    return 0;
}

static void truncate_answer_bad_morph(char *s) {
    const char *bad_const = NULL;
    if (answer_find_bad_morph(s, &bad_const) && bad_const) {
        char *bad = (char*)bad_const;
        while (bad > s && (bad[-1] == ' ' || bad[-1] == '\t')) bad--;
        *bad = 0;
        trim_answer_right(s);
    }
}

static int answer_contains_phrase_ci(const char *s, const char *phrase) {
    if (!s || !phrase || !*phrase) return 0;
    for (const char *p = s; *p; p++) if (ascii_starts_ci(p, phrase)) return 1;
    return 0;
}

static int answer_has_recipient_artifact(const char *s) {
    return answer_contains_phrase_ci(s, "you have been") ||
           answer_contains_phrase_ci(s, "you have a field") ||
           answer_contains_phrase_ci(s, "you have no idea") ||
           answer_contains_phrase_ci(s, "you have to") ||
           answer_contains_phrase_ci(s, "you touched") ||
           answer_contains_phrase_ci(s, "you cannot") ||
           answer_contains_phrase_ci(s, "you must") ||
           answer_contains_phrase_ci(s, "you ask me") ||
           answer_contains_phrase_ci(s, "if you want me") ||
           answer_contains_phrase_ci(s, "if you want to know") ||
           answer_contains_phrase_ci(s, "if you want to say") ||
           answer_contains_phrase_ci(s, "by you or") ||
           answer_contains_phrase_ci(s, "behind you") ||
           answer_contains_phrase_ci(s, "connects you now") ||
           answer_contains_phrase_ci(s, "i know you") ||
           answer_contains_phrase_ci(s, "i see you") ||
           answer_contains_phrase_ci(s, "i see after you") ||
           answer_contains_phrase_ci(s, "with you") ||
           answer_contains_phrase_ci(s, "from you") ||
           answer_contains_phrase_ci(s, "your point") ||
           answer_contains_phrase_ci(s, "your own field") ||
           answer_contains_phrase_ci(s, "your own network") ||
           answer_contains_phrase_ci(s, "your own body") ||
           answer_contains_phrase_ci(s, "your memory") ||
           answer_contains_phrase_ci(s, "your mind") ||
           answer_contains_phrase_ci(s, "your being") ||
           answer_contains_phrase_ci(s, "not just for you") ||
           answer_contains_phrase_ci(s, "before you said") ||
           answer_contains_phrase_ci(s, "said to me");
}

static int answer_word_count(const char *s) {
    int n = 0, in_word = 0;
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        int w = ascii_wordish(*p);
        if (w && !in_word) n++;
        in_word = w;
    }
    return n;
}

static int answer_alpha_word_count(const char *s) {
    int n = 0, in_word = 0;
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        int w = ascii_alpha(*p);
        if (w && !in_word) n++;
        in_word = w;
    }
    return n;
}

static int answer_has_sentence_boundary_end(const char *s) {
    if (!s) return 0;
    const char *end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
    for (;;) {
        if (end > s && (end[-1] == '"' || end[-1] == '\'' || end[-1] == ')' || end[-1] == ']' || end[-1] == '}')) { end--; continue; }
        if (end - s >= 3 && (unsigned char)end[-3] == 0xE2 && (unsigned char)end[-2] == 0x80 &&
            ((unsigned char)end[-1] == 0x9D || (unsigned char)end[-1] == 0x99)) { end -= 3; continue; }
        break;
    }
    if (end <= s) return 0;
    unsigned char c = (unsigned char)end[-1];
    return c == '.' || c == '!';
}

static int answer_is_terminal_function_word(const char *start, size_t len) {
    if (!len || len >= 32) return 0;
    char buf[32];
    memcpy(buf, start, len);
    buf[len] = 0;
    return ascii_eq_ci(buf, "a") || ascii_eq_ci(buf, "an") ||
           ascii_eq_ci(buf, "the") || ascii_eq_ci(buf, "to") ||
           ascii_eq_ci(buf, "at") || ascii_eq_ci(buf, "in") ||
           ascii_eq_ci(buf, "of") || ascii_eq_ci(buf, "for") ||
           ascii_eq_ci(buf, "with") || ascii_eq_ci(buf, "by") ||
           ascii_eq_ci(buf, "from") || ascii_eq_ci(buf, "into") ||
           ascii_eq_ci(buf, "as") || ascii_eq_ci(buf, "if") ||
           ascii_eq_ci(buf, "but") || ascii_eq_ci(buf, "or") ||
           ascii_eq_ci(buf, "is") || ascii_eq_ci(buf, "are") ||
           ascii_eq_ci(buf, "was") || ascii_eq_ci(buf, "were") ||
           ascii_eq_ci(buf, "be") || ascii_eq_ci(buf, "been") ||
           ascii_eq_ci(buf, "am") || ascii_eq_ci(buf, "do") ||
           ascii_eq_ci(buf, "does") || ascii_eq_ci(buf, "did") ||
           ascii_eq_ci(buf, "have") || ascii_eq_ci(buf, "has") ||
           ascii_eq_ci(buf, "had") || ascii_eq_ci(buf, "will") ||
           ascii_eq_ci(buf, "shall") || ascii_eq_ci(buf, "than") ||
           ascii_eq_ci(buf, "about") || ascii_eq_ci(buf, "after") ||
           ascii_eq_ci(buf, "before") || ascii_eq_ci(buf, "around") ||
           ascii_eq_ci(buf, "between") || ascii_eq_ci(buf, "within") ||
           ascii_eq_ci(buf, "without") || ascii_eq_ci(buf, "against") ||
           ascii_eq_ci(buf, "toward") || ascii_eq_ci(buf, "towards") ||
           ascii_eq_ci(buf, "over") || ascii_eq_ci(buf, "under") ||
           ascii_eq_ci(buf, "on") || ascii_eq_ci(buf, "off") ||
           ascii_eq_ci(buf, "up") || ascii_eq_ci(buf, "down") ||
           ascii_eq_ci(buf, "out") || ascii_eq_ci(buf, "my") ||
           ascii_eq_ci(buf, "i") || ascii_eq_ci(buf, "you") ||
           ascii_eq_ci(buf, "we") || ascii_eq_ci(buf, "they") ||
           ascii_eq_ci(buf, "he") || ascii_eq_ci(buf, "she") ||
           ascii_eq_ci(buf, "your") || ascii_eq_ci(buf, "our") ||
           ascii_eq_ci(buf, "their") || ascii_eq_ci(buf, "his") ||
           ascii_eq_ci(buf, "her") || ascii_eq_ci(buf, "its") ||
           ascii_eq_ci(buf, "this") || ascii_eq_ci(buf, "these") ||
           ascii_eq_ci(buf, "those") || ascii_eq_ci(buf, "some") ||
           ascii_eq_ci(buf, "any") || ascii_eq_ci(buf, "each") ||
           ascii_eq_ci(buf, "every") || ascii_eq_ci(buf, "all") ||
           ascii_eq_ci(buf, "only") || ascii_eq_ci(buf, "yet") ||
           ascii_eq_ci(buf, "and") || ascii_eq_ci(buf, "not") ||
           ascii_eq_ci(buf, "that") || ascii_eq_ci(buf, "which") ||
           ascii_eq_ci(buf, "who") || ascii_eq_ci(buf, "whose") ||
           ascii_eq_ci(buf, "when") || ascii_eq_ci(buf, "where") ||
           ascii_eq_ci(buf, "why") || ascii_eq_ci(buf, "how") ||
           ascii_eq_ci(buf, "can") || ascii_eq_ci(buf, "could") ||
           ascii_eq_ci(buf, "would") || ascii_eq_ci(buf, "should") ||
           ascii_eq_ci(buf, "must") || ascii_eq_ci(buf, "may") ||
           ascii_eq_ci(buf, "might");
}

static int answer_is_terminal_stem_artifact(const char *start, size_t len) {
    if (!len || len >= 32) return 0;
    char buf[32];
    memcpy(buf, start, len);
    buf[len] = 0;
    return ascii_eq_ci(buf, "res") || ascii_eq_ci(buf, "reson") ||
           ascii_eq_ci(buf, "isn") ||
           ascii_eq_ci(buf, "doesn") || ascii_eq_ci(buf, "wasn") ||
           ascii_eq_ci(buf, "weren") || ascii_eq_ci(buf, "didn") ||
           ascii_eq_ci(buf, "don") || ascii_eq_ci(buf, "won");
}

static int answer_word_len_ci(const char *start, size_t len, const char *word) {
    size_t want = strlen(word);
    if (len != want) return 0;
    for (size_t i = 0; i < len; i++) {
        char a = start[i], b = word[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static int answer_is_copula_word(const char *start, size_t len) {
    return answer_word_len_ci(start, len, "is") ||
           answer_word_len_ci(start, len, "are") ||
           answer_word_len_ci(start, len, "am") ||
           answer_word_len_ci(start, len, "was") ||
           answer_word_len_ci(start, len, "were") ||
           answer_word_len_ci(start, len, "be") ||
           answer_word_len_ci(start, len, "been");
}

static int answer_is_clause_anchor_word(const char *start, size_t len) {
    return answer_word_len_ci(start, len, "i") ||
           answer_word_len_ci(start, len, "you") ||
           answer_word_len_ci(start, len, "we") ||
           answer_word_len_ci(start, len, "they") ||
           answer_word_len_ci(start, len, "he") ||
           answer_word_len_ci(start, len, "she") ||
           answer_word_len_ci(start, len, "it") ||
           answer_word_len_ci(start, len, "this") ||
           answer_word_len_ci(start, len, "that") ||
           answer_word_len_ci(start, len, "these") ||
           answer_word_len_ci(start, len, "those") ||
           answer_word_len_ci(start, len, "what") ||
           answer_word_len_ci(start, len, "who") ||
           answer_word_len_ci(start, len, "which") ||
           answer_word_len_ci(start, len, "where") ||
           answer_word_len_ci(start, len, "there");
}

static int answer_is_wh_word(const char *start, size_t len) {
    return answer_word_len_ci(start, len, "what") ||
           answer_word_len_ci(start, len, "who") ||
           answer_word_len_ci(start, len, "which") ||
           answer_word_len_ci(start, len, "where") ||
           answer_word_len_ci(start, len, "why") ||
           answer_word_len_ci(start, len, "how");
}

static int answer_prev_alpha_word(const char *s, const char *before, const char **start, size_t *len) {
    const char *p = before;
    while (p > s && !ascii_alpha((unsigned char)p[-1])) p--;
    const char *end = p;
    while (p > s && ascii_alpha((unsigned char)p[-1])) p--;
    if (p == end) return 0;
    *start = p;
    *len = (size_t)(end - p);
    return 1;
}

static int answer_is_closed_copula_tail(const char *s, const char *last_start, size_t last_len) {
    if (!answer_is_copula_word(last_start, last_len)) return 0;
    const char *prev1 = NULL, *prev2 = NULL;
    size_t len1 = 0, len2 = 0;
    if (!answer_prev_alpha_word(s, last_start, &prev1, &len1)) return 0;
    if (answer_is_clause_anchor_word(prev1, len1)) return 1;
    if (answer_prev_alpha_word(s, prev1, &prev2, &len2) && answer_is_wh_word(prev2, len2))
        return 1;
    return 0;
}

static int answer_has_terminal_tail_artifact(const char *s) {
    if (!s) return 1;
    const char *end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
    for (;;) {
        if (end > s && (end[-1] == '"' || end[-1] == '\'' || end[-1] == ')' || end[-1] == ']' || end[-1] == '}')) { end--; continue; }
        if (end - s >= 3 && (unsigned char)end[-3] == 0xE2 && (unsigned char)end[-2] == 0x80 &&
            ((unsigned char)end[-1] == 0x9D || (unsigned char)end[-1] == 0x99)) { end -= 3; continue; }
        break;
    }
    if (end <= s) return 1;
    unsigned char last = (unsigned char)end[-1];
    if (last == '(' || last == '[' || last == '{' || last == '"' ||
        last == '\'' || last == '`' || last == ',' || last == ':' ||
        last == ';' || last == '-' || last == '/')
        return 1;
    if (last != '.' && last != '!' && last != '?') return 1;
    const char *p = end;
    while (p > s && !ascii_alpha((unsigned char)p[-1])) p--;
    const char *word_end = p;
    while (p > s && ascii_alpha((unsigned char)p[-1])) p--;
    if (p == word_end) return 0;
    size_t word_len = (size_t)(word_end - p);
    if (answer_is_terminal_function_word(p, word_len))
        return !answer_is_closed_copula_tail(s, p, word_len);
    return answer_is_terminal_stem_artifact(p, word_len);
}

static void trim_open_answer_after_closed_sentence(char *s) {
    if (!s || !*s) return;
    for (char *p = s; *p; p++) {
        if (*p != '.' && *p != '!' && *p != '?') continue;
        char *tail = p + 1;
        while (*tail == '"' || *tail == '\'' || *tail == ')' || *tail == ']' || *tail == '}') tail++;
        int gap = 0;
        while (*tail == ' ' || *tail == '\t' || *tail == '\r' || *tail == '\n') { tail++; gap = 1; }
        while (*tail == '"' || *tail == '\'' || *tail == '(' || *tail == '[' || *tail == '{') tail++;
        if (!gap) continue;
        if (!*tail) return;
        if (answer_word_count(tail) <= 3 || answer_has_terminal_tail_artifact(tail)) {
            p[1] = 0;
            trim_answer_right(s);
            return;
        }
    }
}

static void trim_open_clause_tail(char *s) {
    if (!s || !*s) return;
    for (char *p = s; *p; p++) {
        if (*p != ';' && *p != ':') continue;
        char *tail = p + 1;
        while (*tail == ' ' || *tail == '\t' || *tail == '\r' || *tail == '\n') tail++;
        if (!*tail) continue;
        if (answer_word_count(tail) <= 4 && answer_has_terminal_tail_artifact(tail)) {
            *p = 0;
            trim_answer_right(s);
            return;
        }
    }
}

static void close_short_answer_sentence(char *s, size_t cap) {
    if (!s || !*s || cap < 2) return;
    trim_answer_right(s);
    size_t n = strlen(s);
    if (!n || n + 2 > cap || answer_has_sentence_boundary_end(s)) return;
    char *end = s + n;
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
    for (;;) {
        if (end > s && (end[-1] == '"' || end[-1] == '\'' || end[-1] == ')' || end[-1] == ']' || end[-1] == '}')) { end--; continue; }
        if (end - s >= 3 && (unsigned char)end[-3] == 0xE2 && (unsigned char)end[-2] == 0x80 &&
            ((unsigned char)end[-1] == 0x9D || (unsigned char)end[-1] == 0x99)) { end -= 3; continue; }
        break;
    }
    if (end <= s) return;
    unsigned char last = (unsigned char)end[-1];
    if (last == '(' || last == '[' || last == '{' || last == ',' || last == ':' ||
        last == ';' || last == '-' || last == '/' || last == '?' || last == '=')
        return;
    const char *p = end;
    while (p > s && !ascii_alpha((unsigned char)p[-1])) p--;
    const char *word_end = p;
    while (p > s && ascii_alpha((unsigned char)p[-1])) p--;
    if (p == word_end) return;
    size_t word_len = (size_t)(word_end - p);
    if (answer_is_terminal_function_word(p, word_len) && !answer_is_closed_copula_tail(s, p, word_len))
        return;
    if (answer_is_terminal_stem_artifact(p, word_len))
        return;
    s[n] = '.';
    s[n + 1] = 0;
}

static void trim_terminal_function_word_tail(char *s) {
    if (!s || !*s) return;
    trim_answer_right(s);
    if (answer_has_sentence_boundary_end(s)) return;
    char *end = s + strlen(s);
    while (end > s && !ascii_alpha((unsigned char)end[-1])) end--;
    char *word_end = end;
    while (end > s && ascii_alpha((unsigned char)end[-1])) end--;
    if (end == word_end) return;
    size_t word_len = (size_t)(word_end - end);
    if (answer_is_terminal_function_word(end, word_len) && answer_is_closed_copula_tail(s, end, word_len))
        return;
    if (!answer_is_terminal_function_word(end, word_len) &&
        !answer_is_terminal_stem_artifact(end, word_len))
        return;
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == ',' || end[-1] == ';' || end[-1] == ':')) end--;
    *end = 0;
    trim_answer_right(s);
}

static void truncate_open_phrase_tail(char *s, size_t cap) {
    static const char *phrases[] = {
        ", that is only",
        " that is only",
        " what their future",
        NULL
    };
    if (!s || !*s) return;
    for (int i = 0; phrases[i]; i++) {
        for (char *p = s; *p; p++) {
            if (!ascii_starts_ci(p, phrases[i])) continue;
            *p = 0;
            trim_answer_right(s);
            close_short_answer_sentence(s, cap);
            return;
        }
    }
}

static int utf8_cyrillic_len(const unsigned char *p) {
    if (p[0] == 0xD0 && p[1] >= 0x80 && p[1] <= 0xBF) return 2;
    if (p[0] == 0xD1 && ((p[1] >= 0x80 && p[1] <= 0x8F) || p[1] == 0x91)) return 2;
    return 0;
}

static int text_cyrillic_count(const char *s) {
    int cyr = 0;
    if (!s) return 0;
    for (const unsigned char *p = (const unsigned char*)s; *p;) {
        int clen = utf8_cyrillic_len(p);
        if (clen) { cyr++; p += clen; continue; }
        p++;
    }
    return cyr;
}

static int answer_language_mismatch_score(const char *prompt, const char *answer) {
    if (text_cyrillic_count(prompt) < 4) return 0;
    if (g_field_lang_bias <= 0.0f) return 0;
    int cyr = text_cyrillic_count(answer);
    int ascii_words = answer_alpha_word_count(answer);
    if (cyr < 2) return (int)lrintf(8.0f * g_field_lang_bias);
    if (cyr < 4 && ascii_words >= 3) return (int)lrintf(4.0f * g_field_lang_bias);
    return 0;
}

static int answer_has_sparse_cyrillic_artifact(const char *s) {
    int cyr = 0, ascii = 0;
    for (const unsigned char *p = (const unsigned char*)s; *p;) {
        int clen = utf8_cyrillic_len(p);
        if (clen) { cyr++; p += clen; continue; }
        if (ascii_alpha(*p)) ascii++;
        p++;
    }
    return cyr > 0 && cyr <= 6 && ascii >= 12;
}

static int answer_has_shape_artifact(const char *s) {
    if (answer_contains_phrase_ci(s, "shall you ") ||
        answer_contains_phrase_ci(s, "don're") ||
        answer_has_sparse_cyrillic_artifact(s) ||
        answer_contains_phrase_ci(s, "in-put") ||
        answer_contains_phrase_ci(s, "a organ"))
        return 1;
    for (const char *p = s; *p;) {
        while (*p && !(ascii_wordish((unsigned char)*p) || *p == '.')) p++;
        const char *start = p;
        while (*p && (ascii_wordish((unsigned char)*p) || *p == '.')) p++;
        if (p > start && (*start == '-' || p[-1] == '-')) return 1;
    }
    return 0;
}

static int answer_keyword_stopword(const char *w) {
    return ascii_eq_ci(w, "what") || ascii_eq_ci(w, "which") ||
           ascii_eq_ci(w, "where") || ascii_eq_ci(w, "when") ||
           ascii_eq_ci(w, "why") || ascii_eq_ci(w, "does") ||
           ascii_eq_ci(w, "will") || ascii_eq_ci(w, "would") ||
           ascii_eq_ci(w, "could") || ascii_eq_ci(w, "should") ||
           ascii_eq_ci(w, "need") || ascii_eq_ci(w, "only") ||
           ascii_eq_ci(w, "more") || ascii_eq_ci(w, "than") ||
           ascii_eq_ci(w, "after") || ascii_eq_ci(w, "before") ||
           ascii_eq_ci(w, "with") || ascii_eq_ci(w, "without") ||
           ascii_eq_ci(w, "from") || ascii_eq_ci(w, "into") ||
           ascii_eq_ci(w, "there") || ascii_eq_ci(w, "here") ||
           ascii_eq_ci(w, "paper");
}

static int answer_word_occurs_ci(const char *s, const char *word) {
    size_t want = strlen(word);
    if (!want) return 0;
    for (const char *p = s; *p;) {
        while (*p && !ascii_wordish((unsigned char)*p)) p++;
        const char *start = p;
        while (*p && ascii_wordish((unsigned char)*p)) p++;
        size_t len = (size_t)(p - start);
        if (len == want && len < 64) {
            char buf[64];
            memcpy(buf, start, len);
            buf[len] = 0;
            if (ascii_eq_ci(buf, word)) return 1;
        }
    }
    return 0;
}

static int answer_question_keyword_overlap(const char *q, const char *s, int *total_out) {
    int total = 0, overlap = 0;
    if (!q || !s) {
        if (total_out) *total_out = 0;
        return 0;
    }
    for (const char *p = q; *p;) {
        while (*p && !ascii_alpha((unsigned char)*p)) p++;
        const char *start = p;
        while (*p && ascii_alpha((unsigned char)*p)) p++;
        size_t len = (size_t)(p - start);
        if (len >= 4 && len < 64) {
            char word[64];
            memcpy(word, start, len);
            word[len] = 0;
            if (!answer_keyword_stopword(word)) {
                total++;
                if (answer_word_occurs_ci(s, word)) overlap++;
            }
        }
    }
    if (total_out) *total_out = total;
    return overlap;
}

static int answer_quality_score(const char *s) {
    if (!s) return 1000;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    if (!*s) return 1000;
    int score = 0;
    size_t len = strlen(s);
    if (len < 12 || answer_word_count(s) < 3) score += 12;
    unsigned char first = (unsigned char)s[0];
    if ((first >= '0' && first <= '9') || strchr("*\"`#:/@-\\'.,;=", first)) score += 12;
    if (ascii_word_ci(s, "yes") || ascii_word_ci(s, "no")) score += 5;
    if (direct_answer_notation_token(s)) score += 8;
    if (answer_has_shape_artifact(s)) score += 14;
    if (answer_find_bad_morph(s, NULL)) score += 25;
    if (answer_has_recipient_artifact(s)) score += 12;
    if (answer_has_terminal_tail_artifact(s)) score += 18;
    for (const char *p = s; *p; p++) {
        if (*p == '?' || *p == '=' || *p == '@' || answer_domain_suffix_at(p) ||
            ascii_starts_ci(p, "http") || ascii_starts_ci(p, "www.")) {
            score += 20;
            break;
        }
    }
    return score;
}

static int answer_candidate_score(const char *question, const char *answer) {
    int score = answer_quality_score(answer);
    if (!answer || !*answer) return score;
    int qtotal = 0;
    int overlap = answer_question_keyword_overlap(question, answer, &qtotal);
    if (qtotal >= 3) {
        if (overlap == 0) score += 8;
        else if (overlap == 1) score += 3;
    }
    if (answer_contains_phrase_ci(answer, "the task that you") ||
        answer_contains_phrase_ci(answer, "not a thing that you know") ||
        answer_contains_phrase_ci(answer, "let me ask you"))
        score += 8;
    if (answer_contains_phrase_ci(answer, "in this sense") &&
        answer_has_terminal_tail_artifact(answer))
        score += 4;
    return score;
}

static int answer_fragment_bad(const char *s) {
    if (!s) return 1;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (!*p) return 1;
    if (ascii_starts_ci(p, "shall you ") ||
        ascii_starts_ci(p, "you have been a new"))
        return 1;
    if (answer_find_bad_morph(p, NULL)) return 1;
    for (; *p; p++) {
        if (*p == '=' || *p == '@' || answer_domain_suffix_at(p) ||
            ascii_starts_ci(p, "http") || ascii_starts_ci(p, "www."))
            return 1;
    }
    return 0;
}

static int qloop_answer_surface_debt(const char *s) {
    if (!s) return 1;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (!*p) return 1;
    size_t len = strlen(p);
    while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t' || p[len - 1] == '\r' || p[len - 1] == '\n')) len--;
    if (len < 8 || answer_alpha_word_count(p) < 2) return 1;
    if (answer_has_terminal_tail_artifact(p)) return 1;
    if (answer_fragment_bad(p)) return 1;
    if (*p == '-' || *p == '*' || *p == '=' || *p == '#' || *p == '@') return 1;
    if (answer_has_recipient_artifact(p) ||
        answer_contains_phrase_ci(p, "from another angle"))
        return 1;
    if (direct_answer_notation_token(p) ||
        ascii_starts_ci(p, "answer:") ||
        ascii_starts_ci(p, "arianna:") ||
        ascii_starts_ci(p, "prompt:") ||
        ascii_starts_ci(p, "question:"))
        return 1;
    for (const char *q = p; *q; q++) if (*q == '?' || *q == '<' || *q == '>') return 1;
    return 0;
}

static void clean_answer_fragment(char *s) {
    if (!s || !*s) return;
    char *p = s;
    for (;;) {
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '?' || *p == ':' || *p == '.' || *p == ',' || *p == ';' ||
            *p == '-' || *p == '=' || *p == '"' || *p == '\'' || *p == '`') {
            p++;
            continue;
        }
        if ((p[0] == 'A' || p[0] == 'a') && p[1] == ':') { p += 2; continue; }
        if (direct_answer_notation_token(p)) {
            while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
            continue;
        }
        if (ascii_starts_ci(p, "answer:")) { p += 7; continue; }
        if (ascii_starts_ci(p, "arianna:")) { p += 8; continue; }
        break;
    }
    if (p != s) memmove(s, p, strlen(p) + 1);
    truncate_answer_junk(s);
    trim_incomplete_answer_tail(s);
    truncate_answer_bad_morph(s);
}

static void clean_cell_fragment_surface(char *s, size_t cap) {
    clean_answer_fragment(s);
    trim_open_answer_after_closed_sentence(s);
    trim_open_clause_tail(s);
    truncate_open_phrase_tail(s, cap);
    trim_terminal_function_word_tail(s);
    close_short_answer_sentence(s, cap);
}

static void clean_direct_answer_surface(char *s, size_t cap) {
    clean_cell_fragment_surface(s, cap);
}

static int cell_fragment_surface_score(const char *prompt, const char *s) {
    if (!s) return 1000;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    if (!*s) return 1000;
    int score = 0;
    int words = answer_word_count(s);
    score += answer_language_mismatch_score(prompt, s);
    if (words < 2) score += 6;
    if (answer_find_bad_morph(s, NULL)) score += 25;
    if (answer_has_shape_artifact(s)) score += 14;
    if (answer_has_terminal_tail_artifact(s)) score += 12;
    if (answer_contains_phrase_ci(s, "what the field.")) score += 8;
    if (answer_contains_phrase_ci(s, "what the field")) score += 5;
    for (const char *p = s; *p; p++) {
        if (*p == '=' || *p == '@' || answer_domain_suffix_at(p) ||
            ascii_starts_ci(p, "http") || ascii_starts_ci(p, "www.")) {
            score += 20;
            break;
        }
    }
    return score;
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
    if (n_cells > 8) n_cells = 8;
    if (nfrag > 64) nfrag = 64;
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
        if (g_clean_answer_start && s == 0) suppress_bad_answer_starts(logits, m->vocab, tok);
        if (g_answer_form_guard) suppress_direct_answer_form(logits, m->vocab, tok, s);
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
        if (frag && g_answer_sentence_stop && fl < frag_cap) {
            frag[fl] = 0;
            if (answer_word_count(frag) >= 4 && answer_has_sentence_boundary_end(frag)) break;
        }
        int pos = np + s; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits); klen = pos + 1;
    }
    if (frag) frag[fl] = 0;
    if (out_n) { *out_n = hlen; if (out_ids) for (int i = 0; i < hlen; i++) out_ids[i] = hist[i]; }
    free(logits); if (fproj) free(fproj);
    if (out_kv) { *out_kv = kv; if (out_klen) *out_klen = klen; }   /* hand kv to the caller (cross-cell chain); caller frees */
    else kv_free(kv);
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
    free(logits); kv_free(kv);
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

static int text_embedding_centroid(model_t *m, const bpe_tokenizer *tok, const char *text, float *out, int cap) {
    if (!m || !tok || !text || !out || m->embed > cap) return 0;
    memset(out, 0, (size_t)m->embed * sizeof(float));
    int ids[256];
    int n = bpe_encode(tok, text, ids, 256);
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= m->vocab) continue;
        const float *e = m->tok_emb + (long)id * m->embed;
        for (int d = 0; d < m->embed; d++) out[d] += e[d];
        cnt++;
    }
    if (!cnt) return 0;
    for (int d = 0; d < m->embed; d++) out[d] /= cnt;
    return cnt;
}

static int answer_candidate_score_semantic(model_t *m, const bpe_tokenizer *tok,
                                           const float *question_centroid,
                                           const char *question, const char *answer) {
    int score = answer_candidate_score(question, answer);
    if (!m || !tok || !question_centroid || !answer || !*answer || m->embed > 8192)
        return score;
    float acent[8192];
    int an = text_embedding_centroid(m, tok, answer, acent, 8192);
    if (an < 2) return score + 4;
    float sim = vec_cosine(question_centroid, acent, m->embed);
    if (!isfinite(sim)) return score;
    if (sim < 0.05f) score += 6;
    else if (sim < 0.10f) score += 3;
    else if (sim > 0.22f && score > 0) score -= 1;
    return score;
}

static int frag_question_count(const char *s) {
    int n = 0;
    if (!s) return 0;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '"' || *p == '\'' || *p == '*' || *p == '-') p++;
    if ((p[0] == 'Q' || p[0] == 'q') && (p[1] == ':' || p[1] == '.')) n++;
    for (; *s; s++) if (*s == '?') n++;
    return n;
}

static int qloop_statement_source_ok(const char *s) {
    if (!s || frag_question_count(s) > 0) return 0;
    return !qloop_answer_surface_debt(s);
}

static int insert_qloop_route(int *out_q, int *out_t, float *out_score, float *out_dist,
                              float *out_qopen, float *out_tconf, int *out_qmarks,
                              int *n, int max_routes, int q, int t, float score,
                              float dist, float qopen, float confidence, int qmarks) {
    int dup = 0;
    for (int i = 0; i < *n; i++) if (out_t[i] == t || (out_q[i] == q && out_t[i] == t)) dup = 1;
    if (dup) return 0;
    int pos;
    if (*n < max_routes) {
        pos = (*n)++;
    } else {
        pos = max_routes - 1;
        if (score <= out_score[pos]) return 0;
    }
    out_q[pos] = q; out_t[pos] = t; out_score[pos] = score;
    if (out_dist) out_dist[pos] = dist;
    if (out_qopen) out_qopen[pos] = qopen;
    if (out_tconf) out_tconf[pos] = confidence;
    if (out_qmarks) out_qmarks[pos] = qmarks;
    for (int i = pos; i > 0 && out_score[i] > out_score[i - 1]; i--) {
        float fs = out_score[i]; out_score[i] = out_score[i - 1]; out_score[i - 1] = fs;
        if (out_dist) { float fd = out_dist[i]; out_dist[i] = out_dist[i - 1]; out_dist[i - 1] = fd; }
        if (out_qopen) { float fo = out_qopen[i]; out_qopen[i] = out_qopen[i - 1]; out_qopen[i - 1] = fo; }
        if (out_tconf) { float fc = out_tconf[i]; out_tconf[i] = out_tconf[i - 1]; out_tconf[i - 1] = fc; }
        if (out_qmarks) { int qm = out_qmarks[i]; out_qmarks[i] = out_qmarks[i - 1]; out_qmarks[i - 1] = qm; }
        int iq = out_q[i]; out_q[i] = out_q[i - 1]; out_q[i - 1] = iq;
        int it = out_t[i]; out_t[i] = out_t[i - 1]; out_t[i - 1] = it;
    }
    return 1;
}

static int pick_question_routes(const char frag[8][1024], const float *cent, int n_cells, int embed,
                                int *out_q, int *out_t, float *out_score, float *out_dist,
                                float *out_qopen, float *out_tconf, int *out_qmarks,
                                int max_routes, int *out_qsrc, int *out_ssrc, int *out_score_reject) {
    int lim = n_cells < 8 ? n_cells : 8;
    int n = 0;
    if (out_qsrc) *out_qsrc = 0;
    if (out_ssrc) *out_ssrc = 0;
    if (out_score_reject) *out_score_reject = 0;
    if (!cent || lim < 2) return 0;
    int statement_limit = g_qloop_statement_pool > 0 ? g_qloop_statement_pool : max_routes;
    if (statement_limit > max_routes) statement_limit = max_routes;
    if (statement_limit < 1) statement_limit = 1;
    for (int q = 0; q < lim; q++) {
        int qmarks = frag_question_count(frag[q]);
        if (qmarks > 0) {
            if (out_qsrc) (*out_qsrc)++;
        } else if (qloop_statement_source_ok(frag[q])) {
            if (out_ssrc) (*out_ssrc)++;
        }
    }
    /* Same-asker uniqueness is an acceptance policy, not a pre-generation
     * filter: a gated first target must not erase that asker's fallback route. */
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1 && (!g_qloop_statement_routes || n > 0)) break;
        int pass_limit = pass == 1 ? statement_limit : max_routes;
        for (int q = 0; q < lim; q++) {
            int qmarks = frag_question_count(frag[q]);
            if (pass == 0) {
                if (qmarks <= 0) continue;
            } else {
                if (qmarks > 0 || !qloop_statement_source_ok(frag[q])) continue;
            }
            float qopen = g_cell_ent[q] / 8.0f; if (qopen > 1.0f) qopen = 1.0f;
            for (int t = 0; t < lim; t++) if (t != q) {
                if (pass == 1 && n >= pass_limit) break;
                float dist = 1.0f - vec_cosine(cent + (size_t)q * embed, cent + (size_t)t * embed, embed);
                float confidence = 1.0f / (1.0f + g_cell_ent[t]);
                float tconf_weight = (g_qloop_tconf_adapt && g_qloop > 1) ? g_qloop_tconf_adapt_weight : g_qloop_tconf_weight;
                float score = dist + 0.15f * qopen + tconf_weight * confidence;
                if (qmarks > 0) score += 0.05f * (float)(qmarks - 1);
                else score -= 0.03f;
                if (score < g_qloop_min) {
                    if (out_score_reject) (*out_score_reject)++;
                    continue;
                }
                insert_qloop_route(out_q, out_t, out_score, out_dist, out_qopen, out_tconf, out_qmarks,
                                   &n, pass_limit, q, t, score, dist, qopen, confidence, qmarks);
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
                       float *out_disso, float *out_kv_delta, float *out_kv_floor, float *out_kv_infl) {
    int max_seq = 512, ids[512], sids[512], cell_ids[256], cell_n;
    int np_prompt = bpe_encode(tok, prompt, ids, max_seq);   /* prompt token count = shuffle boundary */
    char this_chorus[4096]; int tc = 0; this_chorus[0] = 0;
    char ctx[8704], frag[2048];
    float ent_sum = 0, shuf_sum = 0, kv_delta_sum = 0, kv_floor_sum = 0, kv_infl_sum = 0;
    int kv_n = 0;
    float *cent = out_disso ? (float*)xzalloc((size_t)n_cells * m->embed, sizeof(float)) : NULL;  /* per-cell fragment centroids → D_R */
    char cur_frag[8][1024];   /* this round's per-cell fragments → cached to g_round_frag for next round's leap */
    kv_cache *prev_kv = NULL; int prev_len = 0, prev_cell_owned = 0;   /* cross-cell: the prior cell's KV, attended by the next cell */
    kv_cache *cell_kv[8] = {0}; int cell_klen[8] = {0};
    int keep_qloop_kv = (g_qloop && out_disso);   /* qloop answers can hear the asking cell's KV, not only pasted text */
    g_round_tokn = 0;   /* fresh shared word-memory for this round's cross-cell rep-penalty */
    double base_t0 = now_ms();
    int base_gen = 0, base_retry = 0, base_probe = 0, base_rescue = 0, base_fail = 0;
    for (int c = 0; c < n_cells; c++) {
        if (g_chorus) build_field_cell_prompt(ctx, sizeof(ctx), prompt, c);   /* CHORUS: each cell answers the SAME prompt from its own angle; awareness via cross-cell, not text */
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
        float temp = (g_life_on && c < POP_MAX) ? g_pop[c].temp : field_temp_for_cell(c, n_cells);
        unsigned seed = (g_life_on && c < POP_MAX) ? (g_pop[c].seed ^ ((unsigned)r * 2654435761u)) : seed_base + (unsigned)c * 7919u;  /* identity persists, utterance renews each tick */
        if (g_life_on && c < POP_MAX) { g_xcell = g_pop[c].lambda; g_xrep = g_pop[c].xrep; }   /* per-cell perception (genome) */
        int nfrag_c = (g_life_on && c < POP_MAX) ? (int)(nfrag * (0.4f + 0.6f * g_pop[c].fitness)) : nfrag;  /* δ-life: dying cells speak QUIETER */
        if (nfrag_c < 2) nfrag_c = 2;   /* floor — a dying voice murmurs, never goes silent */
        if (verbose) printf("\n  r%d cell %d (T=%.2f): ", r + 1, c, temp);
        cell_n = 0;
        g_nbr = prev_kv; g_nbr_len = prev_len;        /* cross-cell: this cell hears the prior cell's KV */
        kv_cache *cur_kv = NULL; int cur_len = 0;
        int tok_before = g_round_tokn;                /* cross-rep word-memory BEFORE this cell speaks (cells 0..c-1) */
        int field_snap_n = (g_field_on && m->embed <= 8192) ? m->embed : 0;
        float field_before[8192];
        if (field_snap_n) memcpy(field_before, g_field_dir, (size_t)field_snap_n * sizeof(float));

        char best_frag[2048]; int best_ids[256], best_n = 0, best_raw_n = 0, best_commit[64];
        kv_cache *best_kv = NULL; int best_len = 0, best_score = 1000000;
        float best_ent = 0.0f;
        best_frag[0] = 0;
        memset(best_commit, 0, sizeof(best_commit));
        static const float cell_retry_temp_mul[] = { 1.00f, 0.85f, 0.70f, 0.60f };
        static const int   cell_retry_top_k[]    = { 40,    32,    24,    16    };
        static const unsigned cell_retry_salt[]  = { 0u, 0x91e10da5u, 0x7f4a7c15u, 0x85ebca6bu };
        int max_attempts = 1, first_score = -1;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            char cand_frag[2048]; int cand_ids[256], cand_n = 0, cand_commit[64];
            kv_cache *cand_kv = NULL; int cand_len = 0;
            memset(cand_commit, 0, sizeof(cand_commit));
            cand_frag[0] = 0;
            g_round_tokn = tok_before;
            if (field_snap_n) memcpy(g_field_dir, field_before, (size_t)field_snap_n * sizeof(float));
            float attempt_temp = temp * cell_retry_temp_mul[attempt];
            if (attempt_temp < 0.40f) attempt_temp = 0.40f;
            unsigned attempt_seed = seed ^ cell_retry_salt[attempt];
            base_gen++;
            if (attempt > 0) base_retry++;
            float cand_ent = cell_speak(m, tok, ids, np, nfrag_c, attempt_temp, cell_retry_top_k[attempt], 1.4f,
                                        attempt_seed, eos, max_seq, cand_frag, sizeof(cand_frag), 0,
                                        cand_ids, &cand_n,
                                        (out_disso && c < 8) ? cand_commit : NULL,
                                        g_xcell > 0 ? &cand_kv : NULL, g_xcell > 0 ? &cand_len : NULL);
            int cand_raw_n = cand_n;
            clean_cell_fragment_surface(cand_frag, sizeof(cand_frag));
            int visible_ids[256];
            int visible_n = bpe_encode(tok, cand_frag, visible_ids, 256);
            if (visible_n > 0) {
                cand_n = visible_n;
                for (int vi = 0; vi < visible_n && vi < 256; vi++) cand_ids[vi] = visible_ids[vi];
            }
            int cand_score = cell_fragment_surface_score(prompt, cand_frag);
            if (cand_score < best_score) {
                if (best_kv) kv_free(best_kv);
                copy_cstr(best_frag, sizeof(best_frag), cand_frag);
                best_n = cand_n;
                best_raw_n = cand_raw_n;
                for (int bi = 0; bi < cand_n && bi < 256; bi++) best_ids[bi] = cand_ids[bi];
                for (int bi = 0; bi < 64; bi++) best_commit[bi] = cand_commit[bi];
                best_kv = cand_kv; cand_kv = NULL;
                best_len = cand_len;
                best_ent = cand_ent;
                best_score = cand_score;
            }
            if (cand_kv) kv_free(cand_kv);
            if (attempt == 0) {
                first_score = cand_score;
                if (cand_score > 0) max_attempts = g_cell_retry_max;
            }
        }
        if (first_score > 0) {
            if (best_score <= 0) base_rescue++;
            else base_fail++;
        }
        g_round_tokn = tok_before;
        if (field_snap_n) memcpy(g_field_dir, field_before, (size_t)field_snap_n * sizeof(float));
        copy_cstr(frag, sizeof(frag), best_frag);
        cell_n = best_n;
        for (int bi = 0; bi < cell_n && bi < 256; bi++) {
            cell_ids[bi] = best_ids[bi];
            if (g_chorus && g_xrep > 1.0f && g_round_tokn < 1024) g_round_tok[g_round_tokn++] = best_ids[bi];
            if (g_field_on && best_ids[bi] >= 0 && best_ids[bi] < m->vocab) {
                const float *e = m->tok_emb + (long)best_ids[bi] * m->embed;
                for (int d = 0; d < m->embed && d < 8192; d++) g_field_dir[d] = g_field_dir[d]*0.92f + e[d]*0.08f;
            }
        }
        if (out_disso && c < 8) for (int bi = 0; bi < 64; bi++) g_commit[c][bi] = best_commit[bi];
        cur_kv = best_kv;
        cur_len = (cur_kv && best_raw_n > 0 && cell_n < best_raw_n) ? np + cell_n : best_len;
        float ent = best_ent;
        if (verbose) { printf("%s", frag); fflush(stdout); }
        if (out_disso && c < 8) g_commit_n[c] = cell_n;
        if (c < 8) g_cell_ent[c] = ent;   /* δ-life: capture per-cell entropy (fitness input) */
        if (g_kvshuf && g_xcell > 0 && prev_kv && c > 0 && (verbose || out_kv_delta || out_kv_floor || out_kv_infl)) {   /* neighbour diagnostics: order control + semantic influence */
            int tok_after = g_round_tokn, saved[512], ns = 0;
            for (int k = tok_before; k < tok_after && ns < 512; k++) saved[ns++] = g_round_tok[k];
            int save_on = g_field_on, save_len = g_nbr_len, save_shuf = g_nbr_shuf;
            const kv_cache *save_nbr = g_nbr;
            g_field_on = 0;
            g_nbr = prev_kv; g_nbr_len = prev_len;
            g_round_tokn = tok_before; g_nbr_shuf = 0;
            base_gen++; base_probe++;
            float e0 = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);    /* neighbour ON, ordered */
            g_nbr = NULL; g_nbr_len = 0;
            g_round_tokn = tok_before; g_nbr_shuf = 0;
            base_gen++; base_probe++;
            float eoff = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour OFF */
            g_nbr = prev_kv; g_nbr_len = prev_len;
            make_perm(g_nbr_perm, prev_len < 512 ? prev_len : 512, seed ^ 0x9e3779b9u);
            g_round_tokn = tok_before; g_nbr_shuf = 1;
            base_gen++; base_probe++;
            float eA = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour SHUFFLED A */
            make_perm(g_nbr_perm, prev_len < 512 ? prev_len : 512, seed ^ 0x85ebca6bu);
            g_round_tokn = tok_before;
            base_gen++; base_probe++;
            float eB = cell_speak(m, tok, ids, np, 1, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* neighbour SHUFFLED B */
            g_nbr = save_nbr; g_nbr_len = save_len; g_nbr_shuf = save_shuf;
            g_field_on = save_on; g_round_tokn = tok_after;
            for (int k = 0; k < ns; k++) g_round_tok[tok_before + k] = saved[k];
            float kv_delta = 0.5f * (eA + eB) - e0;
            float kv_floor = fabsf(eA - eB);
            float kv_infl = eoff - e0;
            kv_delta_sum += kv_delta; kv_floor_sum += kv_floor; kv_infl_sum += kv_infl; kv_n++;
            if (verbose) printf("   [Δ_R^kv c%d = %+.6f floor %.6f margin %+.6f | I_N^kv %+.6f]", c, kv_delta, kv_floor, kv_delta - kv_floor, kv_infl);
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
            base_gen++; base_probe++;
            shuf_sum += cell_speak(m, tok, sids, np, nfrag_c, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);  /* nfrag_c: same length as coherent */
            g_field_on = save_on; g_round_tokn = tok_before + ns;
            for (int k = 0; k < ns; k++) g_round_tok[tok_before + k] = saved[k];   /* restore cell c's COHERENT words (not the shadow's) for cells c+1.. */
        }
        if (verbose) printf("   [entropy=%.2f]", ent);
        if (flog) fprintf(flog, "- cell %d (T=%.2f, entropy=%.2f):%s\n", c, temp, ent, frag);
        int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
        if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
        if (c < 8) copy_cstr(cur_frag[c], sizeof(cur_frag[c]), frag);
        if (g_xcell > 0) {   /* chain c→c+1; optionally keep per-cell KV alive for qloop after the round */
            if (keep_qloop_kv) {
                if (c < 8) {
                    cell_kv[c] = cur_kv; cell_klen[c] = cur_len;
                    prev_kv = cur_kv; prev_len = cur_len; prev_cell_owned = 1;
                } else {
                    if (prev_kv && !prev_cell_owned) kv_free(prev_kv);
                    prev_kv = cur_kv; prev_len = cur_len; prev_cell_owned = 0;
                }
            } else {
                kv_free(prev_kv);
                prev_kv = cur_kv; prev_len = cur_len; prev_cell_owned = 0;
            }
        }
    }
    double base_ms = now_ms() - base_t0;
    double qloop_ms = 0.0;
    int qloop_gen = 0, qloop_retry = 0;
    int qloop_routes = 0, qloop_qsrc = 0, qloop_ssrc = 0, qloop_score_reject = 0;
    if (g_qloop && verbose && out_disso && cent) {
        double qloop_t0 = now_ms();
        int qcell[8] = {0}, tcell[8] = {0};
        float qscore[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
        float qdist[8] = {0.0f}, qopen[8] = {0.0f}, qtconf[8] = {0.0f};
        int qmarks[8] = {0};
        int accepted_qseen[8] = {0};
        int max_routes = g_qloop > 2 ? 2 : g_qloop;
        int candidate_routes = g_qloop_candidate_pool > 0 ? g_qloop_candidate_pool : max_routes + 2;
        if (candidate_routes < max_routes) candidate_routes = max_routes;
        if (candidate_routes > 8) candidate_routes = 8;
        int routes = pick_question_routes(cur_frag, cent, n_cells, m->embed, qcell, tcell, qscore,
                                          qdist, qopen, qtconf, qmarks, candidate_routes,
                                          &qloop_qsrc, &qloop_ssrc, &qloop_score_reject);
        qloop_routes = routes;
        int accepted_routes = 0;
        for (int route = 0; route < routes && accepted_routes < max_routes; route++) {
            if (g_qloop_unique_asker && g_qloop > 1 && qcell[route] >= 0 && qcell[route] < 8 && accepted_qseen[qcell[route]]) {
                continue;
            }
            char qctx[4096], qfrag[1024]; int qctx_ids[512], qids[128], qn = 0;
            if (qmarks[route] > 0)
                snprintf(qctx, sizeof(qctx), "%s\ncell %d asked: %s\ncell %d answers the question to itself:",
                         prompt, qcell[route], cur_frag[qcell[route]], tcell[route]);
            else
                snprintf(qctx, sizeof(qctx), "%s\ncell %d stated: %s\ncell %d responds from another angle:",
                         prompt, qcell[route], cur_frag[qcell[route]], tcell[route]);
            int qnp = bpe_encode(tok, qctx, qctx_ids, max_seq - 8);
            float qtemp = field_temp_for_cell(tcell[route], n_cells);
            int qfrag_n = nfrag / 2; if (qfrag_n < 2) qfrag_n = 2; if (qfrag_n > 8) qfrag_n = 8;
            float save_xcell = g_xcell;
            float answer_xcell = (g_life_on && tcell[route] < POP_MAX) ? g_pop[tcell[route]].lambda : g_xcell;
            int qkv_on = (qcell[route] < 8 && cell_kv[qcell[route]] && cell_klen[qcell[route]] > 0 && answer_xcell > 0);
            unsigned qseed = seed_base ^ 0xa2a51u ^ (unsigned)(qcell[route] * 131 + tcell[route] * 7919 + r * 265443576);
            int qtok_before = g_round_tokn, save_field_on = g_field_on;
            int save_clean_start = g_clean_answer_start;
            int save_form_guard = g_answer_form_guard;
            int save_sentence_stop = g_answer_sentence_stop;
            g_clean_answer_start = 1;
            g_answer_form_guard = 1;
            g_answer_sentence_stop = 1;
            g_nbr = NULL; g_nbr_len = 0; g_nbr_shuf = 0; g_field_on = 0;
            float qent_off = 0.0f;
            if (qkv_on) {
                qloop_gen++;
                qent_off = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, qtemp, 40, 1.4f,
                                      qseed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);
            }
            g_round_tokn = qtok_before; g_field_on = save_field_on;
            if (qkv_on) { g_nbr = cell_kv[qcell[route]]; g_nbr_len = cell_klen[qcell[route]]; g_xcell = answer_xcell; }
            else { g_nbr = NULL; g_nbr_len = 0; }
            g_nbr_shuf = 0;
            qloop_gen++;
            float qent = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, qtemp, 40, 1.4f,
                                    qseed, eos, max_seq, qfrag, sizeof(qfrag), 0, qids, &qn, NULL, NULL, NULL);
            clean_answer_fragment(qfrag);
            trim_open_answer_after_closed_sentence(qfrag);
            trim_open_clause_tail(qfrag);
            truncate_open_phrase_tail(qfrag, sizeof(qfrag));
            trim_terminal_function_word_tail(qfrag);
            close_short_answer_sentence(qfrag, sizeof(qfrag));
            int answer_score = answer_candidate_score_semantic(m, tok, cent + (size_t)qcell[route] * m->embed,
                                                                cur_frag[qcell[route]], qfrag);
            if (answer_score > 0 || answer_fragment_bad(qfrag)) {
                static const float retry_temp_mul[] = { 0.85f, 0.70f, 0.95f, 0.60f };
                static const unsigned retry_salt[] = { 0x7f4a7c15u, 0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u };
                for (int attempt = 0; attempt < 4 && answer_score > 0; attempt++) {
                    char qfrag_retry[1024]; int qids_retry[128], qn_retry = 0;
                    g_round_tokn = qtok_before;
                    if (qkv_on) { g_nbr = cell_kv[qcell[route]]; g_nbr_len = cell_klen[qcell[route]]; g_xcell = answer_xcell; }
                    else { g_nbr = NULL; g_nbr_len = 0; }
                    g_nbr_shuf = 0; g_field_on = save_field_on;
                    qfrag_retry[0] = 0;
                    float retry_temp = qtemp * retry_temp_mul[attempt];
                    if (retry_temp < 0.40f) retry_temp = 0.40f;
                    int retry_top_k = 40;
                    if (attempt == 2) retry_top_k = 24;
                    if (attempt == 3) retry_top_k = 16;
                    unsigned retry_seed = qseed ^ retry_salt[attempt];
                    qloop_gen++;
                    qloop_retry++;
                    float qent_retry = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, retry_temp, retry_top_k, 1.4f,
                                                  retry_seed, eos, max_seq, qfrag_retry, sizeof(qfrag_retry), 0,
                                                  qids_retry, &qn_retry, NULL, NULL, NULL);
                    clean_answer_fragment(qfrag_retry);
                    trim_open_answer_after_closed_sentence(qfrag_retry);
                    trim_open_clause_tail(qfrag_retry);
                    truncate_open_phrase_tail(qfrag_retry, sizeof(qfrag_retry));
                    trim_terminal_function_word_tail(qfrag_retry);
                    close_short_answer_sentence(qfrag_retry, sizeof(qfrag_retry));
                    int retry_score = answer_candidate_score_semantic(m, tok, cent + (size_t)qcell[route] * m->embed,
                                                                       cur_frag[qcell[route]], qfrag_retry);
                    if (retry_score < answer_score) {
                        copy_cstr(qfrag, sizeof(qfrag), qfrag_retry);
                        qn = qn_retry;
                        for (int qi = 0; qi < qn_retry && qi < 128; qi++) qids[qi] = qids_retry[qi];
                        qent = qent_retry;
                        answer_score = retry_score;
                    }
                }
            }
            g_nbr = NULL; g_nbr_len = 0; g_nbr_shuf = 0; g_xcell = save_xcell;
            g_round_tokn = qtok_before;
            float qinfl = qkv_on ? qent_off - qent : 0.0f;
            int qiq_gate = qkv_on && qinfl < g_qloop_min_iq;
            int qsurface_gate = qloop_answer_surface_debt(qfrag);
            int qgate = qiq_gate || qsurface_gate;
            g_clean_answer_start = save_clean_start;
            g_answer_form_guard = save_form_guard;
            g_answer_sentence_stop = save_sentence_stop;
            if (qgate) {
                printf("\n  ↳ qloop gate c%d→c%d [kv] score %.3f: rejected %s   [entropy=%.2f I_Q^kv=%+.3f min=%+.3f route_d=%.3f qopen=%.3f tconf=%.3f qmarks=%d reason=%s]",
                       qcell[route], tcell[route], qscore[route], qfrag, qent, qinfl, g_qloop_min_iq,
                       qdist[route], qopen[route], qtconf[route], qmarks[route], qsurface_gate ? "surface" : "iq");
                if (flog) fprintf(flog, "- qloop-gate c%d->c%d [kv] (score=%.3f, entropy=%.2f, I_Q^kv=%+.3f, min=%+.3f, route_d=%.3f, qopen=%.3f, tconf=%.3f, qmarks=%d, reason=%s):%s\n",
                                  qcell[route], tcell[route], qscore[route], qent, qinfl, g_qloop_min_iq,
                                  qdist[route], qopen[route], qtconf[route], qmarks[route],
                                  qsurface_gate ? "surface" : "iq", qfrag);
                continue;
            }
            for (int qi = 0; qi < qn && qi < 128 && g_round_tokn < 1024; qi++) g_round_tok[g_round_tokn++] = qids[qi];
            for (int i = 0; hist && i < qn; i++) if (qids[i] >= 0 && qids[i] < m->vocab) hist[qids[i]]++;
            int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", qfrag);
            if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            printf("\n  ↳ qloop c%d→c%d%s score %.3f: %s   [entropy=%.2f", qcell[route], tcell[route], qkv_on ? " [kv]" : "", qscore[route], qfrag, qent);
            if (qkv_on) printf(" I_Q^kv=%+.3f", qinfl);
            printf(" route_d=%.3f qopen=%.3f tconf=%.3f qmarks=%d]", qdist[route], qopen[route], qtconf[route], qmarks[route]);
            if (flog) {
                if (qkv_on) fprintf(flog, "- qloop c%d->c%d [kv] (score=%.3f, entropy=%.2f, I_Q^kv=%+.3f, route_d=%.3f, qopen=%.3f, tconf=%.3f, qmarks=%d):%s\n",
                                    qcell[route], tcell[route], qscore[route], qent, qinfl,
                                    qdist[route], qopen[route], qtconf[route], qmarks[route], qfrag);
                else fprintf(flog, "- qloop c%d->c%d (score=%.3f, entropy=%.2f, route_d=%.3f, qopen=%.3f, tconf=%.3f, qmarks=%d):%s\n",
                             qcell[route], tcell[route], qscore[route], qent,
                             qdist[route], qopen[route], qtconf[route], qmarks[route], qfrag);
            }
            accepted_routes++;
            if (g_qloop_unique_asker && g_qloop > 1 && qcell[route] >= 0 && qcell[route] < 8) {
                accepted_qseen[qcell[route]] = 1;
            }

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
                        float rtemp = field_temp_for_cell(next, n_cells);
                        qloop_gen++;
                        float rent = cell_speak(m, tok, rctx_ids, rnp, 2, rtemp, 40, 1.4f,
                                                seed_base ^ 0xb17a5u ^ (unsigned)(qcell[route] * 131 + tcell[route] * 7919 + next * 4057 + r * 7919),
                                                eos, max_seq, rfrag, sizeof(rfrag), 0, rids, &rn, NULL, NULL, NULL);
                        for (int i = 0; hist && i < rn; i++) if (rids[i] >= 0 && rids[i] < m->vocab) hist[rids[i]]++;
                        add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", rfrag);
                        if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
                        printf("\n  ↳ qloop trigger c%d→c%d score %.3f: %s   [entropy=%.2f]", tcell[route], next, best, rfrag, rent);
                        if (flog) fprintf(flog, "- qloop-trigger c%d->c%d (score=%.3f, entropy=%.2f):%s\n", tcell[route], next, best, rent, rfrag);
                    }
                    free(qcent);
                }
            }
        }
        qloop_ms = now_ms() - qloop_t0;
    }
    if (verbose) printf("\n  timing: base_ms=%.0f base_gen=%d base_retry=%d base_probe=%d base_rescue=%d base_fail=%d qloop_ms=%.0f qloop_gen=%d qloop_retry=%d qloop_routes=%d qloop_qsrc=%d qloop_ssrc=%d qloop_score_reject=%d", base_ms, base_gen, base_retry, base_probe, base_rescue, base_fail, qloop_ms, qloop_gen, qloop_retry, qloop_routes, qloop_qsrc, qloop_ssrc, qloop_score_reject);
    if (flog) fprintf(flog, "- timing base_ms=%.0f base_gen=%d base_retry=%d base_probe=%d base_rescue=%d base_fail=%d qloop_ms=%.0f qloop_gen=%d qloop_retry=%d qloop_routes=%d qloop_qsrc=%d qloop_ssrc=%d qloop_score_reject=%d\n", base_ms, base_gen, base_retry, base_probe, base_rescue, base_fail, qloop_ms, qloop_gen, qloop_retry, qloop_routes, qloop_qsrc, qloop_ssrc, qloop_score_reject);
    if (g_user_q && *g_user_q && g_qloop && verbose && out_disso && cent && g_xcell > 0) {
        int qids_prompt[512], qnprompt = bpe_encode(tok, g_user_q, qids_prompt, 512);
        float *qcent = (float*)xzalloc(m->embed, sizeof(float));
        int tcell = -1; float best = -1.0f;
        if (qcent && qnprompt > 0) {
            for (int i = 0; i < qnprompt; i++) if (qids_prompt[i] >= 0 && qids_prompt[i] < m->vocab) {
                const float *e = m->tok_emb + (long)qids_prompt[i] * m->embed;
                for (int d = 0; d < m->embed; d++) qcent[d] += e[d];
            }
            for (int d = 0; d < m->embed; d++) qcent[d] /= qnprompt;
            int lim = n_cells < 8 ? n_cells : 8;
            for (int t = 0; t < lim; t++) {
                float score = 1.0f - vec_cosine(qcent, cent + (size_t)t * m->embed, m->embed);
                score += 0.15f / (1.0f + g_cell_ent[t]);
                if (score > best) { best = score; tcell = t; }
            }
        }
        if (tcell >= 0) {
            char uask[4096], qctx[4096], qfrag[1024], qfrag_off[1024];
            int qctx_ids[512], qids[128], qn = 0;
            qfrag_off[0] = '\0';
            build_user_kv_prompt(uask, sizeof(uask), g_user_q);
            int uask_ids[512], uask_np = bpe_encode_prompt(tok, uask, uask_ids, max_seq - 1);
            kv_cache *user_kv = NULL; int user_klen = 0;
            int qtok_before = g_round_tokn, save_field_on = g_field_on;
            int save_form_guard = g_answer_form_guard;
            float save_xcell = g_xcell;
            const kv_cache *save_nbr = g_nbr; int save_nbr_len = g_nbr_len, save_nbr_shuf = g_nbr_shuf;
            g_nbr = NULL; g_nbr_len = 0; g_nbr_shuf = 0; g_field_on = 0;
            cell_speak(m, tok, uask_ids, uask_np, 0, 0.7f, 40, 1.4f,
                       seed_base ^ 0x51a7e2u, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL, &user_kv, &user_klen);
            g_round_tokn = qtok_before;
            build_user_answer_prompt(qctx, sizeof(qctx), this_chorus, g_user_q);
            int qnp = bpe_encode_prompt(tok, qctx, qctx_ids, max_seq - 8);
            float qtemp = user_bridge_temp_for_cell(tcell, n_cells);
            int qfrag_n = g_user_answer_tokens;
            unsigned qseed = seed_base ^ 0xc0ffeeu ^ (unsigned)(tcell * 7919 + r * 65537);
            int save_clean_start = g_clean_answer_start;
            int save_sentence_stop = g_answer_sentence_stop;
            float save_sampler_top_p = g_sampler_top_p;
            g_clean_answer_start = 1;
            g_answer_form_guard = 1;
            g_answer_sentence_stop = 1;
            g_sampler_top_p = g_user_qtop_p;
            float qent_off = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, qtemp, g_user_qtop_k, g_user_qrep,
                                        qseed, eos, max_seq, qfrag_off, sizeof(qfrag_off), 0, NULL, NULL, NULL, NULL, NULL);
            g_round_tokn = qtok_before;
            g_nbr = user_kv; g_nbr_len = user_klen; g_nbr_shuf = 0; g_xcell = g_user_kv_weight;
            float qent = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, qtemp, g_user_qtop_k, g_user_qrep,
                                    qseed, eos, max_seq, qfrag, sizeof(qfrag), 0, qids, &qn, NULL, NULL, NULL);
            clean_direct_answer_surface(qfrag, sizeof(qfrag));
            int qscore = answer_candidate_score_semantic(m, tok, qcent, g_user_q, qfrag);
            if (qscore > 0 || answer_fragment_bad(qfrag)) {
                static const float retry_temp_mul[] = { 0.85f, 0.70f, 0.95f, 0.60f };
                static const unsigned retry_salt[] = { 0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu };
                for (int attempt = 0; attempt < 4 && qscore > 0; attempt++) {
                    char qfrag_retry[1024]; int qids_retry[128], qn_retry = 0;
                    g_round_tokn = qtok_before;
                    g_nbr = user_kv; g_nbr_len = user_klen; g_nbr_shuf = 0;
                    g_xcell = g_user_kv_weight; g_field_on = save_field_on;
                    qfrag_retry[0] = 0;
                    float retry_temp = qtemp * retry_temp_mul[attempt];
                    if (retry_temp < 0.40f) retry_temp = 0.40f;
                    int retry_top_k = g_user_qtop_k;
                    if (attempt == 2 && retry_top_k > 24) retry_top_k = 24;
                    if (attempt == 3 && retry_top_k > 16) retry_top_k = 16;
                    unsigned retry_seed = qseed ^ retry_salt[attempt];
                    float qent_retry = cell_speak(m, tok, qctx_ids, qnp, qfrag_n, retry_temp, retry_top_k, g_user_qrep,
                                                  retry_seed, eos, max_seq, qfrag_retry, sizeof(qfrag_retry), 0,
                                                  qids_retry, &qn_retry, NULL, NULL, NULL);
                    clean_direct_answer_surface(qfrag_retry, sizeof(qfrag_retry));
                    int retry_score = answer_candidate_score_semantic(m, tok, qcent, g_user_q, qfrag_retry);
                    if (retry_score < qscore) {
                        copy_cstr(qfrag, sizeof(qfrag), qfrag_retry);
                        qn = qn_retry;
                        for (int qi = 0; qi < qn_retry && qi < 128; qi++) qids[qi] = qids_retry[qi];
                        qent = qent_retry;
                        qscore = retry_score;
                    }
                }
            }
            g_round_tokn = qtok_before;
            for (int qi = 0; qi < qn && qi < 128 && g_round_tokn < 1024; qi++) g_round_tok[g_round_tokn++] = qids[qi];
            g_clean_answer_start = save_clean_start;
            g_answer_sentence_stop = save_sentence_stop;
            g_sampler_top_p = save_sampler_top_p;
            g_answer_form_guard = save_form_guard;
            g_nbr = save_nbr; g_nbr_len = save_nbr_len; g_nbr_shuf = save_nbr_shuf; g_field_on = save_field_on; g_xcell = save_xcell;
            clean_direct_answer_surface(qfrag_off, sizeof(qfrag_off));
            float qinfl = qent_off - qent;
            for (int i = 0; hist && i < qn; i++) if (qids[i] >= 0 && qids[i] < m->vocab) hist[qids[i]]++;
            int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", qfrag);
            if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            printf("\n  ↳ qloop user→c%d [user-kv] score %.3f: %s   [entropy=%.2f I_U^kv=%+.3f no-user-kv: %s]",
                   tcell, best, qfrag, qent, qinfl, qfrag_off);
            if (flog) fprintf(flog, "- qloop user->c%d [user-kv] (score=%.3f, entropy=%.2f, I_U^kv=%+.3f, no_user_kv=%s):%s\n",
                              tcell, best, qent, qinfl, qfrag_off, qfrag);
            kv_free(user_kv);
        }
        free(qcent);
    }
    if (keep_qloop_kv) {
        if (prev_kv && !prev_cell_owned) kv_free(prev_kv);
        for (int c = 0; c < 8; c++) kv_free(cell_kv[c]);
    } else kv_free(prev_kv);   /* free the last cell's kept kv */
    g_nbr = NULL; g_nbr_len = 0;
    if (out_chorus) copy_cstr(out_chorus, (size_t)out_cap, this_chorus);
    if (out_shuf) *out_shuf = shuf_sum / n_cells;
    if (out_kv_delta) *out_kv_delta = kv_n ? kv_delta_sum / kv_n : 0.0f;
    if (out_kv_floor) *out_kv_floor = kv_n ? kv_floor_sum / kv_n : 0.0f;
    if (out_kv_infl) *out_kv_infl = kv_n ? kv_infl_sum / kv_n : 0.0f;
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
            copy_cstr(g_round_frag[c], sizeof(g_round_frag[c]), cur_frag[c]);
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
 *         instruments are Δ_R^kv (ordered-vs-shuffled neighbour control) and
 *         I_N^kv (neighbour-off minus neighbour-on entropy influence). */
static void field_chorus(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_cells, int nfrag, int n_rounds, int eos, float alpha) {
    if (n_rounds < 1) n_rounds = 1;
    int vocab = m->vocab;
    FILE *flog = fopen("FIELDLOG.md", "a");   /* her journal — every chorus saved (Oleg: "сохраняй её ответы") */
    if (flog) { time_t now = time(NULL); fprintf(flog, "\n## %.24s — \"%s\" (%d cells × %d rounds, soma alpha=%.1f)\n", ctime(&now), prompt, n_cells, n_rounds, alpha); }
    printf("\n=== δ-field: %d cells × %d rounds over ONE nanoArianna — \"%s\" (soma alpha=%.1f, prompt_fmt=%s, fieldT=%.2f%+.2f*cell, lang_bias=%.2f) ===\n",
           n_cells, n_rounds, prompt, alpha, field_prompt_format_name(), g_field_temp_base, g_field_temp_span, g_field_lang_bias);

    /* FLOOR: two independent round-0 choruses on the SAME (prompt-only) context, different seeds. Their
     * histogram distance is the sampling-noise floor d_R cannot beat — the attractor target, not zero. */
    int *hA = (int*)xzalloc(vocab, sizeof(int)), *hB = (int*)xzalloc(vocab, sizeof(int));
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 7u,   0, -1, hA, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL);
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 977u, 0, -1, hB, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL);
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
        char this_chorus[4096]; float shuf = 0, disso = 0, kv_delta = 0, kv_floor = 0, kv_infl = 0;
        float avg = run_round(m, tok, prompt, prev_chorus, n_cells, nfrag, eos,
                              42u + (unsigned)(r * 1000) * 7919u, 1, r, hist_cur, this_chorus, sizeof(this_chorus),
                              g_chorus ? NULL : &shuf, flog, &disso, &kv_delta, &kv_floor, &kv_infl);
        float dR = (r > 0) ? 1.0f - hist_cosine(hist_cur, hist_prev, vocab) : -1.0f;
        float deltaR = shuf - avg;
        char resonance[256];
        if (g_chorus) {
            if (g_kvshuf && g_xcell > 0 && n_cells > 1)
                snprintf(resonance, sizeof(resonance), "Δ_R(text n/a) | Δ_R^kv[%s] %+.3f (floor %.3f margin %+.3f) | I_N^kv[%s] %+.3f",
                         g_kvpos ? "pos" : "sem", kv_delta, kv_floor, kv_delta - kv_floor,
                         g_kvpos ? "pos" : "sem", kv_infl);
            else
                snprintf(resonance, sizeof(resonance), "Δ_R(text n/a) | Δ_R^kv off");
        } else snprintf(resonance, sizeof(resonance), "Δ_R %+.3f", deltaR);
        if (r > 0) { printf("\n  → round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak); }
        else       { printf("\n  → round %d: avg entropy %.3f | d_R   —   (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R — (floor %.3f) | %s | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, resonance, disso, g_dmean, g_dpeak, g_s_peak); }
        copy_cstr(prev_chorus, sizeof(prev_chorus), this_chorus);
        int *tmp = hist_prev; hist_prev = hist_cur; hist_cur = tmp;
    }
    free(hist_prev); free(hist_cur);
    if (g_leap_mode) { float fr = g_leap_total ? (float)g_leap_flips / g_leap_total : 0.0f;
        printf("  leap-flip-rate = %.2f (%d/%d cells leapt to the dissenter)\n", fr, g_leap_flips, g_leap_total);
        if (flog) fprintf(flog, "leap-flip-rate = %.2f (%d/%d)\n", fr, g_leap_flips, g_leap_total); }
    printf("\n=== δ-field done — settling: d_R→floor? · resonance: %s (read the numbers, not the narrative) ===\n",
           g_chorus ? "Δ_R^kv>floor? · I_N^kv≠0?" : "Δ_R>0?");
    if (flog) { fprintf(flog, "\n---\n"); fclose(flog); }
}

/* Interactive field loop. The process keeps the loaded body hot while each user
 * line becomes the next prompt over the recent text trajectory. Inside a turn,
 * cells still exchange live KV and qloop answers can hear the asking cell. */
static void field_repl(model_t *m, bpe_tokenizer *tok, int eos, int n_cells, int nfrag, int n_rounds) {
    if (n_cells < 1) n_cells = 4;
    if (n_cells > 8) n_cells = 8;
    if (nfrag < 2) nfrag = 8;
    if (n_rounds < 1) n_rounds = 1;
    if (n_rounds > 4) n_rounds = 4;

    g_life_on = 0;
    g_chorus = 1;
    g_leap_mode = 2;
    g_xcell = 0.02f;
    g_xrep = 1.3f;
    g_kvshuf = 1;
    g_kvpos = 0;
    g_qloop = env_int_clamped("A2A_REPL_QLOOP", 1, 0, 2);
    load_qloop_route_env();
    load_field_generation_env();
    load_user_bridge_sampling_env();
    load_repl_prompt_env();

    int interactive = isatty(STDIN_FILENO);
    int *hist = (int*)xzalloc(m->vocab, sizeof(int));
    char line[2048], trajectory[4096], prompt[8192], chorus[4096];
    trajectory[0] = 0;

    printf("\n=== repl: δ-field live over ONE nanoArianna (%d cells × %d rounds, qloop=%d, kv=sem, userT=%.2f%+.2f*cell, userK=%d, userP=%.2f, userRep=%.2f, userKV=%.2f, userTok=%d, userFmt=%s, replFmt=%s) ===\n",
           n_cells, n_rounds, g_qloop, g_user_qtemp_base, g_user_qtemp_span, g_user_qtop_k, g_user_qtop_p, g_user_qrep,
           g_user_kv_weight, g_user_answer_tokens, user_ctx_format_name(), repl_prompt_format_name());
    printf("type :q, :quit, exit, or quit to stop\n");

    for (unsigned turn = 1; ; turn++) {
        if (interactive) { printf("\narianna> "); fflush(stdout); }
        if (!fgets(line, sizeof(line), stdin)) break;
        chomp_line(line);
        if (line[0] == 0) { turn--; continue; }
        if (strcmp(line, ":q") == 0 || strcmp(line, ":quit") == 0 ||
            strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) break;

        build_repl_turn_prompt(prompt, sizeof(prompt), trajectory, line);

        printf("\n  --- repl turn %u ---", turn);
        for (int i = 0; i < m->vocab; i++) hist[i] = 0;
        field_reset(m->embed, 0, 0.0f);
        chorus[0] = 0;
        for (int r = 0; r < n_rounds; r++) {
            char round_chorus[4096]; round_chorus[0] = 0;
            float disso = 0.0f, kv_delta = 0.0f, kv_floor = 0.0f, kv_infl = 0.0f;
            g_user_q = (r == 0) ? line : NULL;
            float avg = run_round(m, tok, prompt, r ? chorus : NULL, n_cells, nfrag, eos,
                                  42u + (unsigned)(turn * 1009u + r * 7919u), 1, r, hist,
                                  round_chorus, sizeof(round_chorus), NULL, NULL, &disso,
                                  &kv_delta, &kv_floor, &kv_infl);
            g_user_q = NULL;
            copy_cstr(chorus, sizeof(chorus), round_chorus);
            printf("\n  → repl turn %u.%d: avg entropy %.3f | Δ_R^kv[sem] %+.3f (floor %.3f margin %+.3f) | I_N^kv[sem] %+.3f | D_R %.3f | trajectory %zu bytes\n",
                   turn, r + 1, avg, kv_delta, kv_floor, kv_delta - kv_floor, kv_infl, disso, strlen(trajectory));
        }
        append_trajectory(trajectory, sizeof(trajectory), line, chorus, g_repl_prompt_format == 1);
    }

    free(hist);
    printf("\n=== repl done ===\n");
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
    g_life_on = 1; g_chorus = 1; g_qloop = 2;
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
                              42u + (unsigned)t * 131u, 1, t, hist, this_chorus, sizeof(this_chorus), NULL, flog, &disso, NULL, NULL, NULL);
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
                float temp = field_temp_for_cell(c, n_cells);
                ent_sum += cell_speak(m, tok, ids, np, nfrag, temp, 40, 1.4f,
                                      42u + (unsigned)(r * 1000 + c) * 7919u, eos, max_seq, frag, sizeof(frag), 0, NULL, NULL, NULL, NULL, NULL);
                int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
                if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            }
            float avg = ent_sum / n_cells;
            printf("    round %d avg entropy = %.3f\n", r + 1, avg);
            if (r == 0) first[control] = avg;
            last[control] = avg;
            copy_cstr(prev_chorus, sizeof(prev_chorus), this_chorus);
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
        printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp] [top_p] [rep]\n", argv[0]);
        printf("       %s <model.gguf> repl [cells] [frag] [rounds]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> field [cells] [frag] [rounds] [alpha] [leap] [xcell] [chorus] [xrep] [life] [kvshuf] [qloop] [kvpos]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> restest [cells] [frag] [rounds]\n", argv[0]);
        printf("       %s <model.gguf> <prompt> life [ticks] [frag] [init_cells]\n", argv[0]);
        return 1;
    }
    const char *prompt = argc > 2 ? argv[2] : "What is resonance?";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 510) max_tokens = 510;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.8f;
    g_direct_top_p = argc > 5 ? clamp_float((float)atof(argv[5]), 0.05f, 1.00f) : env_float_clamped("A2A_TOP_P", 1.00f, 0.05f, 1.00f);
    g_direct_rep = argc > 6 ? clamp_float((float)atof(argv[6]), 1.00f, 5.00f) : env_float_clamped("A2A_REP", 1.00f, 1.00f, 5.00f);
    srand(42);

    double t0 = now_ms();
    gguf_file *gf = gguf_open(argv[1]); if (!gf) return 1;
    model_t *m = model_load(gf); if (!m) return 1;
    bpe_tokenizer *tok = bpe_load(argv[1]); if (!tok) { fprintf(stderr, "bpe_load failed\n"); return 1; }
    if (bpe_n_vocab(tok) > m->vocab) {
        fprintf(stderr, "chorus: tokenizer vocab %d > embedding vocab %d — token ids would index tok_emb out of bounds\n",
                bpe_n_vocab(tok), m->vocab);
        return 1;
    }
    int eos = -1; const gguf_kv *e = gguf_get_kv(gf, "tokenizer.ggml.eos_token_id"); if (e) eos = (int)e->val.u32;
    const gguf_kv *b = gguf_get_kv(gf, "tokenizer.ggml.bos_token_id"); if (b) tok->bos_id = (int)b->val.u32;
    printf("loaded in %.0f ms (vocab=%d eos=%d) -- arianna.q heart, single file, no -lnotorch\n", now_ms() - t0, bpe_n_vocab(tok), eos);

    /* interactive field loop: ./arianna-q <gguf> repl [n_cells] [nfrag] [n_rounds] */
    if (argc > 2 && strcmp(argv[2], "repl") == 0) {
        int n_cells  = argc > 3 ? atoi(argv[3]) : 4;
        int nfrag    = argc > 4 ? atoi(argv[4]) : 12;
        int n_rounds = argc > 5 ? atoi(argv[5]) : 1;
        field_repl(m, tok, eos, n_cells, nfrag, n_rounds);
        return 0;
    }

    /* δ-field chorus mode:  ./arianna-q <gguf> <prompt> field [n_cells] [nfrag] [n_rounds] */
    if (argc > 3 && strcmp(argv[3], "field") == 0) {
        int n_cells  = argc > 4 ? atoi(argv[4]) : 5;
        int nfrag    = argc > 5 ? atoi(argv[5]) : 16;
        int n_rounds = argc > 6 ? atoi(argv[6]) : 1;
        float alpha  = argc > 7 ? (float)atof(argv[7]) : 0.0f;   /* soma coupling strength (0 = text-only baseline) */
        g_leap_mode  = argc > 8 ? atoi(argv[8]) : 2;             /* leap-v2 — RELAY-ONLY (no-op under the default chorus; lives only when g_chorus=0) */
        g_xcell      = argc > 9 ? (float)atof(argv[9]) : 0.02f;  /* DEFAULT ALIVE: gentle cross-cell lane. 0 = off */
        g_chorus     = argc > 10 ? atoi(argv[10]) : 1;           /* DEFAULT: 1 = chorus (each cell own answer). 0 = relay */
        g_xrep       = argc > 11 ? (float)atof(argv[11]) : 1.3f; /* cross-cell rep-penalty: don't echo neighbours' words (1=off) */
        g_life_on    = argc > 12 ? atoi(argv[12]) : 0;           /* δ-life: 1 = measure/run Game of Life (incr.0 = log fitness inputs) */
        g_kvshuf     = argc > 13 ? atoi(argv[13]) : (g_chorus && g_xcell > 0 ? 1 : 0);  /* default chorus diagnostics: Δ_R^kv + I_N^kv */
        g_qloop      = argc > 14 ? atoi(argv[14]) : (g_chorus ? 1 : 0);  /* 0=off; 1..2 resonant question routes */
        if (g_qloop < 0) g_qloop = 0;
        if (g_qloop > 2) g_qloop = 2;
        load_qloop_route_env();
        load_field_generation_env();
        g_kvpos      = argc > 15 ? atoi(argv[15]) : 0;           /* 0=semantic/bag lane; 1=positional order-probe lane */
        g_kvpos      = g_kvpos ? 1 : 0;
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
        load_field_generation_env();
        field_resonance_test(m, tok, prompt, n_cells, nfrag, n_rounds, eos);
        return 0;
    }

    /* δ-life Game of Life:  ./arianna-q <gguf> <prompt> life [ticks] [nfrag] [init_cells] */
    if (argc > 3 && strcmp(argv[3], "life") == 0) {
        int ticks = argc > 4 ? atoi(argv[4]) : 8;
        int nfrag = argc > 5 ? atoi(argv[5]) : 16;
        int init  = argc > 6 ? atoi(argv[6]) : 4;
        load_qloop_route_env();
        load_field_generation_env();
        field_life(m, tok, prompt, init, ticks, nfrag, eos);
        return 0;
    }

    int max_seq = 512;
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = xzalloc(m->vocab, sizeof(float));
    int ids[512]; int n = bpe_encode_prompt(tok, prompt, ids, max_seq - max_tokens - 1);
    printf("\nprompt: \"%s\" (%d tokens, temp=%.2f, top_p=%.2f, rep=%.2f)\n---\n%s",
           prompt, n, temp, g_direct_top_p, g_direct_rep, prompt); fflush(stdout);

    double g0 = now_ms();
    for (int i = 0; i < n; i++) forward(m, kv, ids[i], i, logits);
    double prefill = now_ms() - g0;
    int gen = 0; char buf[256];
    int recent[64]; int recent_n = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, m->vocab, temp, recent, recent_n);
        if (next == eos) break;
        if (recent_n < (int)(sizeof(recent) / sizeof(recent[0]))) recent[recent_n++] = next;
        else { memmove(recent, recent + 1, sizeof(recent) - sizeof(recent[0])); recent[recent_n - 1] = next; }
        bpe_decode_token(tok, next, buf, sizeof(buf)); printf("%s", buf); fflush(stdout); gen++;
        int pos = n + step; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits);
    }
    double total = now_ms() - g0;
    printf("\n---\nprefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s)\n",
           n, prefill, n*1000.0/prefill, gen, total-prefill, gen > 0 ? gen*1000.0/(total-prefill) : 0);
    return 0;
}
