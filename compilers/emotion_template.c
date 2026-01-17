/*
 * emotion_template.c — Template for Blood-generated emotional kernels
 *
 * Blood replaces:
 *   {{NAME}} — emotion name
 *   {{VALENCE}} — emotional valence (-1 to 1)
 *   {{AROUSAL}} — emotional arousal (0 to 1)
 *   {{KEYWORDS}} — keyword matching code
 *   {{TIMESTAMP}} — generation timestamp
 */

#include <string.h>
#include <math.h>

static const float BASE_VALENCE = {{VALENCE}};
static const float BASE_AROUSAL = {{AROUSAL}};

// Check if text triggers this emotion
int {{NAME}}_check(const char* text) {
    return ({{KEYWORDS}}) ? 1 : 0;
}

// Get emotional response (modulates valence, arousal)
void {{NAME}}_respond(const char* text, float* valence, float* arousal) {
    if ({{NAME}}_check(text)) {
        *valence = (*valence + BASE_VALENCE) / 2.0f;
        *arousal = (*arousal + BASE_AROUSAL) / 2.0f;
    }
}

// Apply emotional modulation to logits
void {{NAME}}_modulate_logits(float* logits, int vocab_size, float strength) {
    float mod = BASE_VALENCE * strength;

    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= (1.0f + mod * 0.1f);
    }
}

// Get base emotional values
void {{NAME}}_get_base(float* valence, float* arousal) {
    *valence = BASE_VALENCE;
    *arousal = BASE_AROUSAL;
}
