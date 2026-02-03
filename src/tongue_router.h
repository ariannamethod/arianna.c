/*
 * tongue_router.h — Multi-Model Tongue Router
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Hardware-aware model selection for Tongue.
 * Queries SARTRE for RAM → picks best Qwen2.5 tier → resolves weight path.
 *
 * Three tiers, all deeply finetuned on Arianna identity corpus:
 *   0.5B (336MB)  — canonical default, runs anywhere
 *   1.5B (935MB)  — mid-tier, 4GB+ RAM
 *   3B   (1.82GB) — full power, 8GB+ RAM
 *
 * Separate module. Does NOT touch canonical d12_bridge flow.
 * Opt-in via USE_TONGUE_ROUTER compile flag.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef TONGUE_ROUTER_H
#define TONGUE_ROUTER_H

#include "../sartre/sartre.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Weight definitions per tier */
#define TONGUE_05B_URL  "https://huggingface.co/ataeff/arianna/resolve/main/qw0-5b/qwen05_900_q4_0.gguf"
#define TONGUE_05B_FILE "qwen05_900_q4_0.gguf"

#define TONGUE_15B_URL  "https://huggingface.co/ataeff/arianna/resolve/main/qw1-5b/arianna_qwen15_2500_q4_0.gguf"
#define TONGUE_15B_FILE "arianna_qwen15_2500_q4_0.gguf"

#define TONGUE_3B_URL   "https://huggingface.co/ataeff/arianna/resolve/main/qw3b/qwen3b_2000_q4_0.gguf"
#define TONGUE_3B_FILE  "qwen3b_2000_q4_0.gguf"

/* Initialize router. Queries SARTRE for hardware tier.
 * cache_dir: where to look for / download weights (default: "tongue/weights")
 * Returns resolved TongueTier. */
TongueTier tongue_router_init(const char* cache_dir);

/* Get path to best available weights on disk.
 * Checks preferred tier first, falls back to smaller.
 * Returns static path string, or NULL if nothing available. */
const char* tongue_router_get_weights_path(void);

/* Ensure weights exist for current tier. Downloads if needed.
 * Falls back: 3B → 1.5B → 0.5B → NULL.
 * Returns path to weights, or NULL on total failure. */
const char* tongue_router_ensure_weights(void);

/* Current tier */
TongueTier tongue_router_tier(void);

/* Override tier (triggers re-resolve on next ensure_weights) */
void tongue_router_set_override(TongueTier tier);

/* Back to auto-detect */
void tongue_router_set_auto(void);

/* Human-readable info string */
const char* tongue_router_info(void);

#ifdef __cplusplus
}
#endif

#endif /* TONGUE_ROUTER_H */
