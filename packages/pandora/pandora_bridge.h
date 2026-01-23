// pandora_bridge.h — Bridge to external brain (Python GPT2-30M)
// Part of SARTRE kernel package system
//
// "Take the words, leave the voice"
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PANDORA_BRIDGE_H
#define PANDORA_BRIDGE_H

#include "../../src/pandora.h"

// Maximum tokens from external brain
#define EXTERNAL_BRAIN_MAX_TOKENS 512

// Path to external brain Python scripts
#define EXTERNAL_BRAIN_GPT2_SCRIPT "packages/pandora/external_brain.py"
#define EXTERNAL_BRAIN_GGUF_SCRIPT "packages/pandora/external_brain_gguf.py"

// External brain types
typedef enum {
    BRAIN_GPT2_30M = 0,      // GPT2-30M (fast, local)
    BRAIN_TINYLLAMA = 1      // TinyLlama 1.1B GGUF (larger, auto-download)
} ExternalBrainType;

// ═══════════════════════════════════════════════════════════════════════════════
// EXTERNAL BRAIN API
// ═══════════════════════════════════════════════════════════════════════════════

// Call external brain and extract vocabulary
// Returns number of Arianna tokens extracted, or -1 on error
// Tokens are written to `tokens` array (must be at least EXTERNAL_BRAIN_MAX_TOKENS)
int external_brain_extract(const char* prompt, int* tokens, int max_tokens);
int external_brain_extract_from(ExternalBrainType brain, const char* prompt, int* tokens, int max_tokens);

// Higher-level: extract and feed directly to Pandora
// Returns number of n-grams added to pandora
int pandora_steal_from_brain(PandoraBox* pandora, const char* prompt);
int pandora_steal_from(PandoraBox* pandora, ExternalBrainType brain, const char* prompt);

// Get brain name for display
const char* external_brain_name(ExternalBrainType brain);

#endif // PANDORA_BRIDGE_H
