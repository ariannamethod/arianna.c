// cloud.c — Pre-semantic Emotion Detection (CLOUD)
// "Something fires BEFORE meaning arrives"
//
// From haze/cloud — lightweight C port
// 100 emotion anchors, 6 chambers, cross-fire coupling
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "cloud.h"
#include <string.h>
#include <ctype.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTION ANCHORS — 100 words in 6 chambers
// ═══════════════════════════════════════════════════════════════════════════════

// FEAR (20) — with stems to catch variations
static const char* FEAR_ANCHORS[] = {
    "fear", "terrif", "panic", "anxious", "dread", "horror",
    "unease", "paranoi", "worr", "nervous", "scare",
    "frighten", "alarm", "tense", "apprehens",
    "threaten", "vulnerab", "insecur", "timid", "wary"
};

// LOVE (18)
static const char* LOVE_ANCHORS[] = {
    "love", "warmth", "tenderness", "devotion", "longing",
    "yearning", "affection", "care", "intimacy", "attachment",
    "adoration", "passion", "fondness", "cherish", "desire",
    "compassion", "gentle", "sweet"
};

// RAGE (17)
static const char* RAGE_ANCHORS[] = {
    "anger", "rage", "fury", "hatred", "spite", "disgust",
    "irritation", "frustration", "resentment", "hostility",
    "aggression", "bitterness", "contempt", "loathing",
    "annoyance", "outrage", "wrath"
};

// VOID (15)
static const char* VOID_ANCHORS[] = {
    "emptiness", "numbness", "hollow", "nothing", "absence",
    "void", "dissociation", "detachment", "apathy",
    "indifference", "drift", "blank", "flat", "dead", "cold"
};

// FLOW (15)
static const char* FLOW_ANCHORS[] = {
    "curiosity", "surprise", "wonder", "confusion",
    "anticipation", "ambivalence", "uncertainty", "restless",
    "searching", "transition", "shift", "change", "flux",
    "between", "liminal"
};

// COMPLEX (15)
static const char* COMPLEX_ANCHORS[] = {
    "shame", "guilt", "envy", "jealousy", "pride",
    "disappointment", "betrayal", "relief", "nostalgia",
    "bittersweet", "melancholy", "regret", "hope",
    "gratitude", "awe"
};

static const char* CHAMBER_NAMES[] = {
    "FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"
};

// ═══════════════════════════════════════════════════════════════════════════════
// COUPLING MATRIX — cross-fire influence between chambers
// ═══════════════════════════════════════════════════════════════════════════════

static const float COUPLING_MATRIX[6][6] = {
    //     FEAR   LOVE   RAGE   VOID   FLOW   CMPLX
    {     0.0f, -0.3f, +0.6f, +0.4f, -0.2f, +0.3f },  // FEAR → feeds rage/void/complex
    {    -0.3f,  0.0f, -0.6f, -0.5f, +0.3f, +0.4f },  // LOVE → suppresses fear/rage/void
    {    +0.3f, -0.4f,  0.0f, +0.2f, -0.3f, +0.2f },  // RAGE → feeds fear/void, kills flow
    {    +0.5f, -0.7f, +0.3f,  0.0f, -0.4f, +0.5f },  // VOID → feeds fear/rage/complex
    {    -0.2f, +0.2f, -0.2f, -0.3f,  0.0f, +0.2f },  // FLOW → dampens extremes
    {    +0.3f, +0.2f, +0.2f, +0.3f, +0.1f,  0.0f },  // COMPLEX → ripple effect
};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: case-insensitive substring search
// ═══════════════════════════════════════════════════════════════════════════════

static int count_substring(const char* text, const char* sub) {
    int count = 0;
    int sub_len = strlen(sub);
    if (sub_len == 0) return 0;

    const char* p = text;
    while (*p) {
        // Case-insensitive compare
        int match = 1;
        for (int i = 0; i < sub_len && p[i]; i++) {
            if (tolower((unsigned char)p[i]) != tolower((unsigned char)sub[i])) {
                match = 0;
                break;
            }
        }
        if (match) {
            // Check word START boundary (not in middle of word)
            // But allow word END to have suffix (terror→terrified)
            int before_ok = (p == text) || !isalpha((unsigned char)p[-1]);
            if (before_ok) {
                count++;
                // Skip past this match to avoid double counting
                p += sub_len;
                continue;
            }
        }
        p++;
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

void cloud_init(void) {
    // Nothing to initialize for now (stateless)
}

// ═══════════════════════════════════════════════════════════════════════════════
// GET ANCHOR/CHAMBER
// ═══════════════════════════════════════════════════════════════════════════════

const char* cloud_get_anchor(int idx) {
    if (idx < 0 || idx >= CLOUD_N_ANCHORS) return "unknown";

    if (idx < 20) return FEAR_ANCHORS[idx];
    idx -= 20;
    if (idx < 18) return LOVE_ANCHORS[idx];
    idx -= 18;
    if (idx < 17) return RAGE_ANCHORS[idx];
    idx -= 17;
    if (idx < 15) return VOID_ANCHORS[idx];
    idx -= 15;
    if (idx < 15) return FLOW_ANCHORS[idx];
    idx -= 15;
    if (idx < 15) return COMPLEX_ANCHORS[idx];

    return "unknown";
}

const char* cloud_get_chamber_name(int idx) {
    if (idx < 0 || idx >= CLOUD_N_CHAMBERS) return "UNKNOWN";
    return CHAMBER_NAMES[idx];
}

int cloud_get_anchor_chamber(int anchor_idx) {
    if (anchor_idx < 20) return CLOUD_CHAMBER_FEAR;
    if (anchor_idx < 38) return CLOUD_CHAMBER_LOVE;
    if (anchor_idx < 55) return CLOUD_CHAMBER_RAGE;
    if (anchor_idx < 70) return CLOUD_CHAMBER_VOID;
    if (anchor_idx < 85) return CLOUD_CHAMBER_FLOW;
    return CLOUD_CHAMBER_COMPLEX;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-FIRE COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

void cloud_crossfire(float* chambers, int max_iterations) {
    float delta[CLOUD_N_CHAMBERS];
    float learning_rate = 0.1f;
    float threshold = 0.001f;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute deltas from coupling
        float max_delta = 0.0f;
        for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
            delta[i] = 0.0f;
            for (int j = 0; j < CLOUD_N_CHAMBERS; j++) {
                delta[i] += COUPLING_MATRIX[j][i] * chambers[j];
            }
            delta[i] *= learning_rate;
            if (fabsf(delta[i]) > max_delta) max_delta = fabsf(delta[i]);
        }

        // Apply deltas
        for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
            chambers[i] += delta[i];
            // Clamp to [0, 1]
            if (chambers[i] < 0.0f) chambers[i] = 0.0f;
            if (chambers[i] > 1.0f) chambers[i] = 1.0f;
        }

        // Check convergence
        if (max_delta < threshold) break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PING — main emotion detection
// ═══════════════════════════════════════════════════════════════════════════════

CloudResponse cloud_ping(const char* text) {
    CloudResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!text || !*text) {
        resp.primary_word = "neutral";
        resp.secondary_word = "neutral";
        resp.primary_chamber = "FLOW";
        return resp;
    }

    // 1. Compute resonances (substring matching)
    float sum = 0.0f;
    int idx = 0;

    // FEAR
    for (int i = 0; i < CLOUD_FEAR_SIZE; i++) {
        int count = count_substring(text, FEAR_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(FEAR_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // LOVE
    for (int i = 0; i < CLOUD_LOVE_SIZE; i++) {
        int count = count_substring(text, LOVE_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(LOVE_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // RAGE
    for (int i = 0; i < CLOUD_RAGE_SIZE; i++) {
        int count = count_substring(text, RAGE_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(RAGE_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // VOID
    for (int i = 0; i < CLOUD_VOID_SIZE; i++) {
        int count = count_substring(text, VOID_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(VOID_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // FLOW
    for (int i = 0; i < CLOUD_FLOW_SIZE; i++) {
        int count = count_substring(text, FLOW_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(FLOW_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // COMPLEX
    for (int i = 0; i < CLOUD_COMPLEX_SIZE; i++) {
        int count = count_substring(text, COMPLEX_ANCHORS[i]);
        resp.resonances[idx] = (float)count * strlen(COMPLEX_ANCHORS[i]);
        sum += resp.resonances[idx];
        idx++;
    }

    // Normalize resonances
    if (sum > 0.0f) {
        for (int i = 0; i < CLOUD_N_ANCHORS; i++) {
            resp.resonances[i] /= sum;
        }
    }

    // 2. Aggregate into chambers
    idx = 0;
    for (int i = 0; i < CLOUD_FEAR_SIZE; i++) resp.chambers[0] += resp.resonances[idx++];
    for (int i = 0; i < CLOUD_LOVE_SIZE; i++) resp.chambers[1] += resp.resonances[idx++];
    for (int i = 0; i < CLOUD_RAGE_SIZE; i++) resp.chambers[2] += resp.resonances[idx++];
    for (int i = 0; i < CLOUD_VOID_SIZE; i++) resp.chambers[3] += resp.resonances[idx++];
    for (int i = 0; i < CLOUD_FLOW_SIZE; i++) resp.chambers[4] += resp.resonances[idx++];
    for (int i = 0; i < CLOUD_COMPLEX_SIZE; i++) resp.chambers[5] += resp.resonances[idx++];

    // Save pre-crossfire chambers for primary detection
    float chambers_pre[CLOUD_N_CHAMBERS];
    memcpy(chambers_pre, resp.chambers, sizeof(chambers_pre));

    // 3. Apply cross-fire coupling
    resp.iterations = 10;  // max iterations
    cloud_crossfire(resp.chambers, resp.iterations);

    // 4. Find primary emotion (from pre-crossfire, more raw)
    float max_res = 0.0f;
    for (int i = 0; i < CLOUD_N_ANCHORS; i++) {
        if (resp.resonances[i] > max_res) {
            max_res = resp.resonances[i];
            resp.primary_idx = i;
        }
    }
    resp.primary_strength = max_res;
    resp.primary_word = cloud_get_anchor(resp.primary_idx);
    resp.primary_chamber = cloud_get_chamber_name(cloud_get_anchor_chamber(resp.primary_idx));

    // 5. Find secondary emotion (from post-crossfire, more processed)
    float max_chamber = 0.0f;
    int max_chamber_idx = 0;
    for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
        if (resp.chambers[i] > max_chamber) {
            max_chamber = resp.chambers[i];
            max_chamber_idx = i;
        }
    }

    // Pick strongest anchor from secondary chamber (different from primary)
    int secondary_start, secondary_end;
    switch (max_chamber_idx) {
        case 0: secondary_start = 0; secondary_end = 20; break;
        case 1: secondary_start = 20; secondary_end = 38; break;
        case 2: secondary_start = 38; secondary_end = 55; break;
        case 3: secondary_start = 55; secondary_end = 70; break;
        case 4: secondary_start = 70; secondary_end = 85; break;
        default: secondary_start = 85; secondary_end = 100; break;
    }

    float max_sec = 0.0f;
    for (int i = secondary_start; i < secondary_end; i++) {
        if (i != resp.primary_idx && resp.resonances[i] > max_sec) {
            max_sec = resp.resonances[i];
            resp.secondary_idx = i;
        }
    }
    resp.secondary_strength = max_sec;
    resp.secondary_word = cloud_get_anchor(resp.secondary_idx);

    return resp;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION WITH ARIANNA
// ═══════════════════════════════════════════════════════════════════════════════

float cloud_temperature_bias(const CloudResponse* resp) {
    // High fear/void → lower temperature (more focused)
    // High rage → higher temperature (more chaotic)
    // High love → balanced
    // High flow → slightly higher (curious)

    float bias = 0.0f;
    bias -= resp->chambers[CLOUD_CHAMBER_FEAR] * 0.1f;
    bias -= resp->chambers[CLOUD_CHAMBER_VOID] * 0.15f;
    bias += resp->chambers[CLOUD_CHAMBER_RAGE] * 0.2f;
    bias += resp->chambers[CLOUD_CHAMBER_FLOW] * 0.1f;

    return bias;
}

void cloud_apply_emotion_to_logits(float* logits, int vocab_size,
                                   const CloudResponse* resp) {
    // Simple version: slight boost to emotional words
    // Full implementation would need word lists
    (void)logits;
    (void)vocab_size;
    (void)resp;
    // TODO: Implement vocabulary bias based on emotion
}

int cloud_needs_care(const CloudResponse* resp) {
    // High fear or void indicates need for gentle handling
    return resp->chambers[CLOUD_CHAMBER_FEAR] > 0.3f ||
           resp->chambers[CLOUD_CHAMBER_VOID] > 0.3f;
}

int cloud_needs_grounding(const CloudResponse* resp) {
    // High void (dissociation) indicates need for grounding
    return resp->chambers[CLOUD_CHAMBER_VOID] > 0.4f;
}

int cloud_needs_warmth(const CloudResponse* resp) {
    // High love response indicates warmth in response
    return resp->chambers[CLOUD_CHAMBER_LOVE] > 0.3f;
}
