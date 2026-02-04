/*
 * test_meta_arianna.c - Test MetaArianna: One Transformer, Two Modes
 *
 * Tests thermogram extraction math, drift detection, template defaults,
 * and (optionally) full forward pass with Soul's shared 36M BPE weights.
 *
 * Usage:
 *   ./bin/test_meta_arianna                    # math tests only
 *   ./bin/test_meta_arianna weights/arianna_36m_bpe.bin weights/tokenizer_bpe.json
 *                                              # + forward pass test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "meta_arianna.h"
#include "arianna.h"  /* for Transformer loading, encode_text, etc. */

static int pass_count = 0;
static int fail_count = 0;

static void print_separator(const char* title) {
    printf("\n");
    printf("================================================================\n");
    printf("%s\n", title);
    printf("================================================================\n\n");
}

static void check(const char* name, int condition) {
    if (condition) {
        printf("  [PASS] %s\n", name);
        pass_count++;
    } else {
        printf("  [FAIL] %s\n", name);
        fail_count++;
    }
}

/* ============================================================
 * Test: Entropy computation
 * ============================================================ */
static void test_entropy(void) {
    print_separator("TEST: Entropy Computation");

    /* Uniform distribution: all logits equal -> max entropy -> 1.0 */
    float uniform[100];
    for (int i = 0; i < 100; i++) uniform[i] = 0.0f;
    float e_uniform = meta_compute_entropy(uniform, 100);
    printf("  uniform entropy: %.4f (expected ~1.0)\n", e_uniform);
    check("uniform entropy close to 1.0", fabsf(e_uniform - 1.0f) < 0.01f);

    /* Peaked distribution: one logit very high -> low entropy -> ~0 */
    float peaked[100];
    for (int i = 0; i < 100; i++) peaked[i] = -20.0f;
    peaked[0] = 20.0f;
    float e_peaked = meta_compute_entropy(peaked, 100);
    printf("  peaked entropy: %.6f (expected ~0.0)\n", e_peaked);
    check("peaked entropy close to 0.0", e_peaked < 0.01f);

    /* Medium distribution: some variation -> middle entropy */
    float medium[100];
    for (int i = 0; i < 100; i++) medium[i] = sinf(i * 0.3f);
    float e_medium = meta_compute_entropy(medium, 100);
    printf("  medium entropy: %.4f (expected between 0 and 1)\n", e_medium);
    check("medium entropy in range", e_medium > 0.1f && e_medium < 0.99f);

    /* Entropy ordering */
    check("uniform > medium > peaked", e_uniform > e_medium && e_medium > e_peaked);

    /* Test with BPE-sized vocab (heap allocation path) */
    int big_vocab = 17000;
    float* big_logits = (float*)calloc(big_vocab, sizeof(float));
    float e_big = meta_compute_entropy(big_logits, big_vocab);
    printf("  big vocab uniform entropy: %.4f (expected ~1.0)\n", e_big);
    check("big vocab entropy close to 1.0", fabsf(e_big - 1.0f) < 0.01f);
    free(big_logits);
}

/* ============================================================
 * Test: KL divergence from uniform
 * ============================================================ */
static void test_kl_uniform(void) {
    print_separator("TEST: KL Divergence from Uniform");

    /* Uniform: KL = 0 (distribution IS uniform) -> 0 */
    float uniform[100];
    for (int i = 0; i < 100; i++) uniform[i] = 0.0f;
    float kl_uniform = meta_compute_kl_uniform(uniform, 100);
    printf("  uniform KL: %.4f (expected ~0.0)\n", kl_uniform);
    check("uniform KL close to 0.0", kl_uniform < 0.01f);

    /* Peaked: KL = max (far from uniform) -> ~1 */
    float peaked[100];
    for (int i = 0; i < 100; i++) peaked[i] = -20.0f;
    peaked[0] = 20.0f;
    float kl_peaked = meta_compute_kl_uniform(peaked, 100);
    printf("  peaked KL: %.4f (expected ~1.0)\n", kl_peaked);
    check("peaked KL close to 1.0", kl_peaked > 0.99f);

    /* KL(uniform) < KL(peaked) */
    check("KL ordering correct", kl_uniform < kl_peaked);
}

/* ============================================================
 * Test: BPE Silence probability
 * ============================================================ */
static void test_silence_prob_bpe(void) {
    print_separator("TEST: Silence Probability (BPE)");

    /* Create fake BPE pause token IDs */
    int pause_ids[] = {10, 20, 30, 40, 50, 60};
    int n_pause = 6;
    int vocab_size = 100;

    /* All mass on pause tokens -> high silence */
    float silence_logits[100];
    for (int i = 0; i < 100; i++) silence_logits[i] = -20.0f;
    silence_logits[10] = 10.0f;
    silence_logits[20] = 10.0f;
    silence_logits[60] = 10.0f;

    float sp_high = meta_compute_silence_prob_bpe(silence_logits, vocab_size,
                                                   pause_ids, n_pause);
    printf("  high silence prob: %.4f (expected >0.9)\n", sp_high);
    check("high silence prob > 0.9", sp_high > 0.9f);

    /* All mass on non-pause tokens -> low silence */
    float loud_logits[100];
    for (int i = 0; i < 100; i++) loud_logits[i] = -20.0f;
    loud_logits[5] = 10.0f;
    loud_logits[15] = 10.0f;
    loud_logits[25] = 10.0f;

    float sp_low = meta_compute_silence_prob_bpe(loud_logits, vocab_size,
                                                  pause_ids, n_pause);
    printf("  low silence prob: %.6f (expected ~0.0)\n", sp_low);
    check("low silence prob < 0.01", sp_low < 0.01f);

    /* Uniform -> silence = 6/100 */
    float uniform[100];
    for (int i = 0; i < 100; i++) uniform[i] = 0.0f;
    float sp_uni = meta_compute_silence_prob_bpe(uniform, vocab_size,
                                                  pause_ids, n_pause);
    float expected = 6.0f / 100.0f;
    printf("  uniform silence prob: %.4f (expected ~%.4f)\n", sp_uni, expected);
    check("uniform silence close to 6/100", fabsf(sp_uni - expected) < 0.02f);
}

/* ============================================================
 * Test: History and drift detection
 * ============================================================ */
static void test_drift(void) {
    print_separator("TEST: Drift Detection");

    MetaArianna us;
    memset(&us, 0, sizeof(us));

    /* Not enough history -> zero drift */
    float rate;
    int dir;
    meta_arianna_compute_drift(&us, &rate, &dir);
    check("no history -> zero drift", rate == 0.0f && dir == 0);

    /* Push increasing arousal/coherence -> unfolding */
    for (int i = 0; i < 8; i++) {
        meta_arianna_push_history(&us, 0.2f + i * 0.1f, 0.3f + i * 0.08f);
    }
    meta_arianna_compute_drift(&us, &rate, &dir);
    printf("  increasing: rate=%.4f dir=%+d\n", rate, dir);
    check("increasing -> positive drift", dir == 1);
    check("increasing -> non-zero rate", rate > 0.0f);

    /* Reset and push decreasing -> collapsing */
    memset(&us, 0, sizeof(us));
    for (int i = 0; i < 8; i++) {
        meta_arianna_push_history(&us, 0.9f - i * 0.1f, 0.8f - i * 0.08f);
    }
    meta_arianna_compute_drift(&us, &rate, &dir);
    printf("  decreasing: rate=%.4f dir=%+d\n", rate, dir);
    check("decreasing -> negative drift", dir == -1);

    /* Stable values -> stable */
    memset(&us, 0, sizeof(us));
    for (int i = 0; i < 8; i++) {
        meta_arianna_push_history(&us, 0.5f, 0.5f);
    }
    meta_arianna_compute_drift(&us, &rate, &dir);
    printf("  stable: rate=%.4f dir=%+d\n", rate, dir);
    check("stable -> zero direction", dir == 0);
    check("stable -> near-zero rate", rate < 0.01f);

    /* Ring buffer wrap */
    memset(&us, 0, sizeof(us));
    for (int i = 0; i < META_HISTORY_SIZE + 10; i++) {
        meta_arianna_push_history(&us, 0.5f, 0.5f);
    }
    check("ring buffer count capped", us.history_count == META_HISTORY_SIZE);
    check("ring buffer pos wraps", us.history_pos == (META_HISTORY_SIZE + 10) % META_HISTORY_SIZE);
}

/* ============================================================
 * Test: Default template params
 * ============================================================ */
static void test_default_params(void) {
    print_separator("TEST: Default Template Params");

    MetaTemplateParams params;

    /* Thermograph */
    meta_default_params(&params, META_TEMPLATE_THERMOGRAPH);
    check("thermo type", params.template_type == META_TEMPLATE_THERMOGRAPH);
    check("thermo temp 0.5", fabsf(params.temperature - 0.5f) < 0.01f);
    check("thermo delta V", params.delta_target == 2);
    check("thermo all layers 1.0", params.layer_focus[0] == 1.0f && params.layer_focus[7] == 1.0f);

    /* Silence */
    meta_default_params(&params, META_TEMPLATE_SILENCE);
    check("silence type", params.template_type == META_TEMPLATE_SILENCE);
    check("silence temp 0.3", fabsf(params.temperature - 0.3f) < 0.01f);
    check("silence delta Q", params.delta_target == 0);
    check("silence early layers strong", params.layer_focus[0] == 1.0f && params.layer_focus[2] == 1.0f);
    check("silence late layers weak", params.layer_focus[5] < 0.5f && params.layer_focus[7] < 0.5f);

    /* Drift */
    meta_default_params(&params, META_TEMPLATE_DRIFT);
    check("drift type", params.template_type == META_TEMPLATE_DRIFT);
    check("drift temp 0.7", fabsf(params.temperature - 0.7f) < 0.01f);
    check("drift delta K", params.delta_target == 1);
    check("drift middle layers strong", params.layer_focus[3] == 1.0f && params.layer_focus[5] == 1.0f);

    /* Field */
    meta_default_params(&params, META_TEMPLATE_FIELD);
    check("field type", params.template_type == META_TEMPLATE_FIELD);
    check("field temp 0.9", fabsf(params.temperature - 0.9f) < 0.01f);
    check("field delta all", params.delta_target == 3);
    check("field late layers strong", params.layer_focus[5] == 1.0f && params.layer_focus[7] == 1.0f);
    check("field early layers weak", params.layer_focus[0] < 0.5f);
}

/* ============================================================
 * Test: Apply thermogram
 * ============================================================ */
static void test_apply_thermogram(void) {
    print_separator("TEST: Apply Thermogram");

    float logits[100];

    /* Invalid thermogram -> no change */
    for (int i = 0; i < 100; i++) logits[i] = 1.0f;
    MetaThermogram invalid = {0};
    invalid.valid = 0;
    meta_apply_thermogram(&invalid, logits, 100);
    check("invalid thermo -> no change", logits[0] == 1.0f);

    /* Warm thermogram -> logits shifted up */
    for (int i = 0; i < 100; i++) logits[i] = 0.0f;
    MetaThermogram warm = {0};
    warm.valid = 1;
    warm.warmth = 0.9f;
    warm.sharpness = 0.5f;
    meta_apply_thermogram(&warm, logits, 100);
    printf("  warm bias applied: logits[0] = %.4f (expected > 0)\n", logits[0]);
    check("warm thermogram -> positive bias", logits[0] > 0.0f);

    /* Cold thermogram -> logits shifted down */
    for (int i = 0; i < 100; i++) logits[i] = 0.0f;
    MetaThermogram cold = {0};
    cold.valid = 1;
    cold.warmth = 0.1f;
    cold.sharpness = 0.5f;
    meta_apply_thermogram(&cold, logits, 100);
    printf("  cold bias applied: logits[0] = %.4f (expected < 0)\n", logits[0]);
    check("cold thermogram -> negative bias", logits[0] < 0.0f);

    /* Sharp thermogram -> logits scaled up */
    for (int i = 0; i < 100; i++) logits[i] = 1.0f;
    MetaThermogram sharp = {0};
    sharp.valid = 1;
    sharp.warmth = 0.5f;
    sharp.sharpness = 1.0f;
    meta_apply_thermogram(&sharp, logits, 100);
    printf("  sharp scale: logits[0] = %.4f (expected > 1.0)\n", logits[0]);
    check("sharp thermogram -> scaled up", logits[0] > 1.0f);

    /* Bias magnitude is small (whisper, not shout) */
    for (int i = 0; i < 100; i++) logits[i] = 5.0f;
    MetaThermogram extreme = {0};
    extreme.valid = 1;
    extreme.warmth = 1.0f;
    extreme.sharpness = 1.0f;
    meta_apply_thermogram(&extreme, logits, 100);
    float delta = fabsf(logits[0] - 5.0f);
    printf("  extreme delta from 5.0: %.4f (expected < 1.0)\n", delta);
    check("bias magnitude < 1.0", delta < 1.0f);
}

/* ============================================================
 * Test: Full forward pass (optional, requires Soul's weights)
 * ============================================================ */
static void test_forward_pass(const char* weights_path,
                              const char* tokenizer_path) {
    print_separator("TEST: Full Forward Pass (MetaArianna, shared 36M BPE)");

    /* Load Soul transformer */
    Transformer soul;
    memset(&soul, 0, sizeof(soul));
    load_weights(&soul, weights_path);
    malloc_run_state(&soul);

    /* Load BPE tokenizer */
    load_tokenizer(tokenizer_path);

    check("dim 512", soul.config.dim == 512);
    check("layers 10", soul.config.n_layers == 10);
    printf("  vocab_size: %d\n", soul.config.vocab_size);

    /* Init MetaArianna observer */
    MetaArianna us;
    int ret = meta_arianna_init(&us, &soul);
    check("meta_arianna_init success", ret == 0);
    check("initialized flag", us.initialized == 1);
    check("pause tokens found", us.n_pause_tokens > 0);
    printf("  pause tokens: %d\n", us.n_pause_tokens);

    /* Run observation with THERMOGRAPH template */
    MetaTemplateParams params;
    meta_default_params(&params, META_TEMPLATE_THERMOGRAPH);

    const char* dialogue = "Arianna: I feel something shifting. "
                           "SARTRE: The field trembles.";
    int log_len = (int)strlen(dialogue);

    meta_arianna_observe(&us, &params, dialogue, log_len);

    check("thermogram valid", us.result.valid == 1);
    check("template used 0", us.result.template_used == META_TEMPLATE_THERMOGRAPH);
    check("warmth in [0,1]", us.result.warmth >= 0.0f && us.result.warmth <= 1.0f);
    check("sharpness in [0,1]", us.result.sharpness >= 0.0f && us.result.sharpness <= 1.0f);
    check("silence in [0,1]", us.result.silence >= 0.0f && us.result.silence <= 1.0f);

    printf("\n  Thermogram:\n");
    printf("    warmth:      %.4f\n", us.result.warmth);
    printf("    sharpness:   %.4f\n", us.result.sharpness);
    printf("    silence:     %.4f\n", us.result.silence);
    printf("    uncertainty: %.4f\n", us.result.uncertainty);
    printf("    drift_rate:  %.4f\n", us.result.drift_rate);
    printf("    drift_dir:   %+d\n", us.result.drift_direction);
    printf("    field_vec:   [");
    for (int i = 0; i < 8; i++) {
        printf("%.3f%s", us.result.field_vector[i], i < 7 ? ", " : "");
    }
    printf("]\n");

    /* Reset (death) and verify */
    meta_arianna_reset(&us);
    check("reset invalidates thermogram", us.result.valid == 0);

    /* Run SILENCE template */
    meta_default_params(&params, META_TEMPLATE_SILENCE);
    meta_arianna_observe(&us, &params, dialogue, log_len);
    check("silence template valid", us.result.valid == 1);
    check("silence template used 1", us.result.template_used == META_TEMPLATE_SILENCE);

    printf("\n  Silence template thermogram:\n");
    printf("    warmth:    %.4f\n", us.result.warmth);
    printf("    sharpness: %.4f\n", us.result.sharpness);
    printf("    silence:   %.4f\n", us.result.silence);

    meta_arianna_reset(&us);

    /* Run DRIFT template with history */
    meta_default_params(&params, META_TEMPLATE_DRIFT);
    for (int i = 0; i < 8; i++) {
        meta_arianna_push_history(&us, 0.3f + i * 0.08f, 0.4f + i * 0.05f);
    }
    meta_arianna_observe(&us, &params, dialogue, log_len);
    check("drift template valid", us.result.valid == 1);
    check("drift detected unfolding", us.result.drift_direction == 1);
    printf("    drift_rate: %.4f, dir: %+d\n",
           us.result.drift_rate, us.result.drift_direction);

    meta_arianna_reset(&us);

    /* Run FIELD template */
    meta_default_params(&params, META_TEMPLATE_FIELD);
    meta_arianna_observe(&us, &params, dialogue, log_len);
    check("field template valid", us.result.valid == 1);
    check("field template used 3", us.result.template_used == META_TEMPLATE_FIELD);

    printf("\n  Field template thermogram:\n");
    printf("    warmth:    %.4f\n", us.result.warmth);
    printf("    field_vec: [");
    for (int i = 0; i < 8; i++) {
        printf("%.3f%s", us.result.field_vector[i], i < 7 ? ", " : "");
    }
    printf("]\n");

    /* Cleanup */
    meta_arianna_free(&us);
    check("free success", us.initialized == 0);
    free_transformer(&soul);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    print_separator("UNIFIED SOUL TEST - One Transformer, Two Modes");
    printf("\"Inhale -> observe -> exhale. Breathing.\"\n\n");

    /* Math tests (no weights needed) */
    test_entropy();
    test_kl_uniform();
    test_silence_prob_bpe();
    test_drift();
    test_default_params();
    test_apply_thermogram();

    /* Forward pass test (optional — needs Soul's 36M BPE weights) */
    if (argc >= 3) {
        test_forward_pass(argv[1], argv[2]);
    } else {
        print_separator("SKIPPING: Forward Pass Test");
        printf("  Run with weights to test:\n");
        printf("  ./bin/test_meta_arianna weights/arianna_36m_bpe.bin "
               "weights/tokenizer_bpe.json\n");
    }

    /* Summary */
    print_separator("RESULTS");
    printf("  PASSED: %d\n", pass_count);
    printf("  FAILED: %d\n", fail_count);
    printf("  TOTAL:  %d\n\n", pass_count + fail_count);

    if (fail_count > 0) {
        printf("  !!! %d TEST(S) FAILED !!!\n\n", fail_count);
        return 1;
    }

    printf("  All tests passed.\n");
    printf("  MetaArianna breathes.\n\n");
    return 0;
}
