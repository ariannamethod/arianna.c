/* Read-only numerical comparison of a GGUF forward with the SFT forward.
 * Build twice, with -DPROBE_RESONANCE for the inner voice. No live field opens. */
#define main organism_main
#ifdef PROBE_RESONANCE
#include "../arianna_resonance.c"
#else
#include "../arianna.c"
#endif
#undef main

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--sampling-check")) {
        /* Same recent token set and final pair, different occurrence counts.
         * No repeated trigram: repetition pressure must remain one 1.4 penalty. */
        V = 8;
        am_init();
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float b[8];
        memcpy(b, a, sizeof(a));
        int short_ctx[] = {0, 1, 2, 3, 4};
        int repeated_ctx[] = {0, 1, 2, 0, 4, 3, 4};
#ifdef PROBE_RESONANCE
        resonance_sample_token(a, short_ctx, 5, 1, 1);
        resonance_sample_token(b, repeated_ctx, 7, 1, 1);
#else
        arianna_sample_token(a, short_ctx, 5, 1, 1);
        arianna_sample_token(b, repeated_ctx, 7, 1, 1);
#endif
        float error = 0;
        for (int i = 0; i < 8; i++) error = fmaxf(error, fabsf(a[i] - b[i]));
        printf("repeated-occurrence probability difference: %.9g\n", error);
        return error > 1e-7f;
    }
    if (argc != 5) {
        fprintf(stderr, "usage: forward_probe model.gguf tokens.i32 prompt_length logits.f32\n");
        return 2;
    }
    int ids[512], n = 0;
    FILE *in = fopen(argv[2], "rb");
    if (!in) return 2;
    n = (int)fread(ids, sizeof(int), 512, in);
    fclose(in);
    int prompt_len = atoi(argv[3]);
    if (prompt_len <= 0 || prompt_len > n) return 2;
    am_init();
#ifdef PROBE_RESONANCE
    ResonanceCtx ctx = {0};
    if (resonance_load_gguf(&ctx, argv[1])) return 2;
    float *logits = calloc(V, sizeof(float));
    for (int i = 0; i < n; i++) forward_token(&ctx.w, ids[i], i, logits, NULL);
#else
    Weights w = {0};
    if (yent_read_cfg(argv[1]) || yent_load_gguf(&w, argv[1])) return 2;
    kv_init(T);
    float *logits = calloc(V, sizeof(float));
    prefill_batch(&w, ids, prompt_len, logits, NULL);
    for (int i = prompt_len; i < n; i++) forward_token(&w, ids[i], i, logits, NULL);
#endif
    FILE *out = fopen(argv[4], "wb");
    if (!out) return 2;
    int ok = fwrite(logits, sizeof(float), V, out) == (size_t)V;
    fclose(out);
    free(logits);
    return ok ? 0 : 2;
}
