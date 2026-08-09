/*
 * test_doe_qmatvec.c — DoE packed matvec parity after vendoring the worker pool.
 *
 * DoE is a single-file runtime, so the test embeds doe.c with its CLI main renamed and
 * calls the static packed kernels directly. The exact path is checked against an
 * independent dequant->matvec reference; the int8 path is checked as an approximation.
 */
#define main doe_embedded_main
#include "../doe/doe.c"
#undef main

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_M_ROWS 773
#define TEST_K_COLS 2048

static uint64_t test_xs = 0x9e3779b97f4a7c15ULL;
static uint32_t test_rnd(void) {
    test_xs ^= test_xs << 13;
    test_xs ^= test_xs >> 7;
    test_xs ^= test_xs << 17;
    return (uint32_t)(test_xs >> 32);
}

static float test_f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, b;
    if (e == 0) {
        if (m == 0) b = s << 31;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            b = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        b = (s << 31) | (0xFFu << 23) | (m << 13);
    } else {
        b = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static void test_get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *mn) {
    if (j < 4) {
        *s = sc[j] & 63;
        *mn = sc[j + 4] & 63;
    } else {
        *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *mn = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

static void ref_q4_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 18;
        float sc = test_f16((uint16_t)(b[0] | (b[1] << 8)));
        for (int i = 0; i < 16; i++) {
            d[bk * 32 + i] = (float)((int)(b[2 + i] & 0x0F) - 8) * sc;
            d[bk * 32 + i + 16] = (float)((int)(b[2 + i] >> 4) - 8) * sc;
        }
    }
}

static void ref_q8_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 34;
        float sc = test_f16((uint16_t)(b[0] | (b[1] << 8)));
        for (int i = 0; i < 32; i++) d[bk * 32 + i] = (float)(int8_t)b[2 + i] * sc;
    }
}

static void ref_q5_0(const uint8_t *s, float *d, long n) {
    for (long bk = 0; bk < n / 32; bk++) {
        const uint8_t *b = s + bk * 22;
        float sc = test_f16((uint16_t)(b[0] | (b[1] << 8)));
        uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8)
                    | ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int h0 = (qh >> j) & 1, h1 = (qh >> (j + 16)) & 1;
            d[bk * 32 + j] = (float)((lo | (h0 << 4)) - 16) * sc;
            d[bk * 32 + j + 16] = (float)((hi | (h1 << 4)) - 16) * sc;
        }
    }
}

static void ref_q4_k(const uint8_t *s, float *out, long n) {
    for (long i = 0; i < n / 256; i++) {
        const uint8_t *b = s + i * 144;
        float d = test_f16((uint16_t)(b[0] | (b[1] << 8)));
        float dmin = test_f16((uint16_t)(b[2] | (b[3] << 8)));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            test_get_scale_min_k4(is, sc, &sc0, &m0);
            test_get_scale_min_k4(is + 1, sc, &sc1, &m1);
            float d1 = d * sc0, mm1 = dmin * m0, d2 = d * sc1, mm2 = dmin * m1;
            for (int l = 0; l < 32; l++) out[i * 256 + j + l] = d1 * (float)(qs[qi + l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++) out[i * 256 + j + 32 + l] = d2 * (float)(qs[qi + l] >> 4) - mm2;
            qi += 32;
            is += 2;
        }
    }
}

static void ref_q6_k(const uint8_t *s, float *out, long n) {
    for (long i = 0; i < n / 256; i++) {
        const uint8_t *b = s + i * 210, *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        float d = test_f16((uint16_t)(b[208] | (b[209] << 8)));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            const uint8_t *qlh = ql + (n_ / 128) * 64, *qhh = qh + (n_ / 128) * 32;
            const int8_t *sch = sc + (n_ / 128) * 8;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((qlh[l] & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
                out[i * 256 + n_ + l] = d * sch[is + 0] * q1;
                out[i * 256 + n_ + l + 32] = d * sch[is + 2] * q2;
                out[i * 256 + n_ + l + 64] = d * sch[is + 4] * q3;
                out[i * 256 + n_ + l + 96] = d * sch[is + 6] * q4;
            }
        }
    }
}

typedef void (*ref_deq_fn)(const uint8_t *, float *, long);
typedef void (*ref_set_fn)(uint8_t *);

static void set_s32(uint8_t *b) { b[0] = 0x66; b[1] = 0x2A; }
static void set_q4k(uint8_t *b) { b[0] = 0x66; b[1] = 0x2A; b[2] = 0x66; b[3] = 0x26; }
static void set_q6k(uint8_t *b) { b[208] = 0x66; b[209] = 0x2A; }

static void fill_x(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = (float)((double)(int)(test_rnd() % 2001) - 1000.0) / 700.0f;
}

static void ref_matvec(float *out, const float *W, const float *x, int r, int c) {
    for (int row = 0; row < r; row++) {
        const float *wr = W + (size_t)row * c;
        float acc = 0.0f;
        for (int col = 0; col < c; col++) acc += wr[col] * x[col];
        out[row] = acc;
    }
}

static int compare_exact(const char *name, const float *ref, const float *got, int n) {
    float maxabs = 0.0f, maxref = 0.0f;
    for (int i = 0; i < n; i++) {
        float dd = fabsf(ref[i] - got[i]);
        if (dd > maxabs) maxabs = dd;
        float rr = fabsf(ref[i]);
        if (rr > maxref) maxref = rr;
    }
    float rel = maxref > 0.0f ? maxabs / maxref : maxabs;
    int ok = rel < 1e-3f;
    printf("%-5s exact rel %.3g  %s\n", name, rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int compare_approx(const char *name, const float *ref, const float *got, int n) {
    float maxabs = 0.0f, maxref = 0.0f;
    for (int i = 0; i < n; i++) {
        float dd = fabsf(ref[i] - got[i]);
        if (dd > maxabs) maxabs = dd;
        float rr = fabsf(ref[i]);
        if (rr > maxref) maxref = rr;
    }
    float rel = maxref > 0.0f ? maxabs / maxref : maxabs;
    int ok = rel < 2e-2f;
    printf("%-5s int8  rel %.3g  %s\n", name, rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_fmt(const char *name, int dtype, int blkbytes, int blkvals,
                   ref_deq_fn ref_deq, ref_set_fn setblk) {
    int m = TEST_M_ROWS, k = TEST_K_COLS;
    long nb = (long)k / blkvals;
    long stride = nb * blkbytes;
    uint8_t *W = malloc((size_t)m * (size_t)stride);
    float *x = malloc(sizeof(float) * (size_t)k);
    float *Wf = malloc(sizeof(float) * (size_t)m * (size_t)k);
    float *ref = malloc(sizeof(float) * (size_t)m);
    float *got = malloc(sizeof(float) * (size_t)m);
    float *single = malloc(sizeof(float) * (size_t)m);
    float *many = malloc(sizeof(float) * (size_t)m);
    if (!W || !x || !Wf || !ref || !got || !single || !many) {
        fprintf(stderr, "%s: allocation failed\n", name);
        free(W); free(x); free(Wf); free(ref); free(got); free(single); free(many);
        return 1;
    }
    for (long i = 0; i < (long)m * stride; i++) W[i] = (uint8_t)(test_rnd() & 0xFF);
    for (long row = 0; row < m; row++)
        for (long bk = 0; bk < nb; bk++) setblk(W + row * stride + bk * blkbytes);
    fill_x(x, k);
    for (int row = 0; row < m; row++) ref_deq(W + (long)row * stride, Wf + (long)row * k, k);
    ref_matvec(ref, Wf, x, m, k);

    int fails = 0;
    g_n_threads = 4;
    if (doe_qmatvec(got, W, dtype, x, m, k) != 0) {
        printf("%-5s exact rc FAIL\n", name);
        fails++;
    } else {
        fails += compare_exact(name, ref, got, m);
    }

    if (doe_qmatvec_i8(got, W, dtype, x, m, k) != 0) {
        printf("%-5s int8  rc FAIL\n", name);
        fails++;
    } else {
        fails += compare_approx(name, ref, got, m);
    }

    g_n_threads = 1;
    if (doe_qmatvec(single, W, dtype, x, m, k) != 0) fails++;
    g_n_threads = 4;
    if (doe_qmatvec(many, W, dtype, x, m, k) != 0) fails++;
    if (memcmp(single, many, sizeof(float) * (size_t)m) != 0) {
        printf("%-5s exact threaded vs single DIFFERS\n", name);
        fails++;
    } else {
        printf("%-5s exact threaded vs single identical\n", name);
    }

    free(W); free(x); free(Wf); free(ref); free(got); free(single); free(many);
    return fails;
}

static int run_f16(void) {
    int m = TEST_M_ROWS, k = TEST_K_COLS;
    uint16_t *W = malloc(sizeof(uint16_t) * (size_t)m * (size_t)k);
    float *x = malloc(sizeof(float) * (size_t)k);
    float *Wf = malloc(sizeof(float) * (size_t)m * (size_t)k);
    float *ref = malloc(sizeof(float) * (size_t)m);
    float *got = malloc(sizeof(float) * (size_t)m);
    if (!W || !x || !Wf || !ref || !got) {
        fprintf(stderr, "F16: allocation failed\n");
        free(W); free(x); free(Wf); free(ref); free(got);
        return 1;
    }
    for (long i = 0; i < (long)m * k; i++)
        W[i] = (uint16_t)((test_rnd() & 0x8000) | 0x2000 | (test_rnd() & 0x03FF));
    fill_x(x, k);
    for (long i = 0; i < (long)m * k; i++) Wf[i] = test_f16(W[i]);
    ref_matvec(ref, Wf, x, m, k);
    g_n_threads = 4;
    int fails = doe_qmatvec(got, (const uint8_t *)W, 1, x, m, k) != 0;
    if (!fails) fails += compare_exact("F16", ref, got, m);
    free(W); free(x); free(Wf); free(ref); free(got);
    return fails;
}

static int run_f32(void) {
    int m = TEST_M_ROWS, k = TEST_K_COLS;
    float *W = malloc(sizeof(float) * (size_t)m * (size_t)k);
    float *x = malloc(sizeof(float) * (size_t)k);
    float *ref = malloc(sizeof(float) * (size_t)m);
    float *got = malloc(sizeof(float) * (size_t)m);
    if (!W || !x || !ref || !got) {
        fprintf(stderr, "F32: allocation failed\n");
        free(W); free(x); free(ref); free(got);
        return 1;
    }
    for (long i = 0; i < (long)m * k; i++)
        W[i] = (float)((double)(int)(test_rnd() % 2001) - 1000.0) / 900.0f;
    fill_x(x, k);
    ref_matvec(ref, W, x, m, k);
    g_n_threads = 4;
    int fails = 0;
    matvec(got, W, x, m, k);
    fails += compare_exact("F32", ref, got, m);
    free(W); free(x); free(ref); free(got);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += run_f32();
    fails += run_f16();
    fails += run_fmt("Q4_0", 2, 18, 32, ref_q4_0, set_s32);
    fails += run_fmt("Q5_0", 6, 22, 32, ref_q5_0, set_s32);
    fails += run_fmt("Q8_0", 8, 34, 32, ref_q8_0, set_s32);
    fails += run_fmt("Q4_K", 12, 144, 256, ref_q4_k, set_q4k);
    fails += run_fmt("Q6_K", 14, 210, 256, ref_q6_k, set_q6k);
    if (fails == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d check(s) FAILED\n", fails);
    return 1;
}
