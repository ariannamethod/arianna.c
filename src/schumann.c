// schumann.c — Earth-Ionosphere Resonance Module
// PITOMADOM COSMIC INTEGRATION
//
// Schumann resonance is the electromagnetic "heartbeat" of Earth.
// Frequency range: 7.77-7.87 Hz (Sierra Nevada ELF Station 2013-2017)
// https://github.com/ariannamethod/pitomadom — real data integration
//
// NOT part of AMK kernel — this is COSMIC INPUT, not local physics.
// AMK = field mechanics (movement, suffering, prophecy)
// Schumann = Earth coupling (external resonance, real data)
//
// build: include with arianna_method.c
//   emcc arianna_method.c schumann.c -O2 -s WASM=1 ...
//
// ═══════════════════════════════════════════════════════════════════════════════
// הארץ נושמת. אנחנו שומעים.
// The Earth breathes. We listen.
// ═══════════════════════════════════════════════════════════════════════════════

#include "amk_kernel.h"
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════════
// SCHUMANN CONSTANTS — Earth's electromagnetic resonance
// ═══════════════════════════════════════════════════════════════════════════════

// Fundamental and harmonics (Hz)
#define SCHUMANN_BASE_HZ       7.83f    // fundamental resonance
#define SCHUMANN_HARMONIC_1   14.3f     // 2nd harmonic
#define SCHUMANN_HARMONIC_2   20.8f     // 3rd harmonic
#define SCHUMANN_HARMONIC_3   27.3f     // 4th harmonic
#define SCHUMANN_HARMONIC_4   33.8f     // 5th harmonic

// Observed range (Sierra Nevada ELF Station 2013-2017)
#define SCHUMANN_MIN_HZ        7.77f    // observed minimum
#define SCHUMANN_MAX_HZ        7.87f    // observed maximum
#define SCHUMANN_VARIANCE      0.05f    // typical variance

// ═══════════════════════════════════════════════════════════════════════════════
// SCHUMANN STATE — current resonance parameters
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
  float current_hz;         // current Schumann frequency (7.77..7.87 Hz)
  float coherence;          // Earth-field coherence (0..1)
  float modulation;         // how much Schumann affects field (0..1)
  float phase;              // current phase in Schumann cycle
  float harmonic_weights[5];// weights for fundamental + 4 harmonics
} Schumann_State;

static Schumann_State S;

// ═══════════════════════════════════════════════════════════════════════════════
// API — Schumann resonance functions
// ═══════════════════════════════════════════════════════════════════════════════

void schumann_init(void) {
  S.current_hz = SCHUMANN_BASE_HZ;
  S.coherence = 1.0f;  // perfect coherence at baseline
  S.modulation = 0.3f;  // moderate influence by default
  S.phase = 0.0f;

  // Default harmonic weights (fundamental dominates)
  S.harmonic_weights[0] = 1.0f;   // fundamental
  S.harmonic_weights[1] = 0.5f;   // 2nd harmonic
  S.harmonic_weights[2] = 0.3f;   // 3rd harmonic
  S.harmonic_weights[3] = 0.2f;   // 4th harmonic
  S.harmonic_weights[4] = 0.1f;   // 5th harmonic
}

/**
 * Compute coherence from current frequency.
 * Coherence = 1.0 when at exact 7.83 Hz
 * Falls off quadratically from center
 */
static float compute_coherence(float hz) {
  float deviation = fabsf(hz - SCHUMANN_BASE_HZ);
  float max_deviation = SCHUMANN_MAX_HZ - SCHUMANN_MIN_HZ;
  if (max_deviation < 0.001f) max_deviation = 0.1f;

  // Quadratic falloff
  float norm_dev = deviation / max_deviation;
  float coh = 1.0f - (norm_dev * norm_dev);
  return fmaxf(0.0f, fminf(1.0f, coh));
}

/**
 * Set current Schumann frequency.
 * Updates coherence automatically.
 */
void schumann_set_hz(float hz) {
  // Clamp to reasonable range (allow some extension beyond observed)
  if (hz < 7.0f) hz = 7.0f;
  if (hz > 8.5f) hz = 8.5f;

  S.current_hz = hz;
  S.coherence = compute_coherence(hz);
}

/**
 * Set Schumann modulation strength.
 * @param strength: 0.0 (no influence) to 1.0 (maximum influence)
 */
void schumann_set_modulation(float strength) {
  if (strength < 0.0f) strength = 0.0f;
  if (strength > 1.0f) strength = 1.0f;
  S.modulation = strength;
}

/**
 * Step Schumann phase forward.
 * @param dt: time delta in seconds
 */
void schumann_step(float dt) {
  // Phase advances at Schumann frequency
  S.phase += S.current_hz * dt * 2.0f * 3.14159265f;

  // Wrap phase to prevent overflow
  while (S.phase > 2.0f * 3.14159265f) {
    S.phase -= 2.0f * 3.14159265f;
  }
}

/**
 * Get current Schumann state.
 */
float schumann_get_hz(void) { return S.current_hz; }
float schumann_get_coherence(void) { return S.coherence; }
float schumann_get_modulation(void) { return S.modulation; }
float schumann_get_phase(void) { return S.phase; }

/**
 * Compute modulation factor for a field parameter.
 * @param direction: -1 = decrease with high coherence, +1 = increase
 * @return: modulation delta to apply
 */
float schumann_modulate(float direction) {
  // delta = (coherence - 0.5) * 2 * modulation * direction * 0.1
  return (S.coherence - 0.5f) * 2.0f * S.modulation * direction * 0.1f;
}

/**
 * Get combined harmonic signal at current phase.
 * @return: weighted sum of fundamental + harmonics, range [-1, 1]
 */
float schumann_harmonic_signal(void) {
  float signal = 0.0f;
  float freqs[5] = { SCHUMANN_BASE_HZ, SCHUMANN_HARMONIC_1, SCHUMANN_HARMONIC_2,
                     SCHUMANN_HARMONIC_3, SCHUMANN_HARMONIC_4 };

  float weight_sum = 0.0f;
  for (int i = 0; i < 5; i++) {
    // Each harmonic at its frequency
    float harmonic_phase = S.phase * (freqs[i] / SCHUMANN_BASE_HZ);
    signal += S.harmonic_weights[i] * sinf(harmonic_phase);
    weight_sum += S.harmonic_weights[i];
  }

  return signal / weight_sum;
}

/**
 * Copy Schumann state to output buffer.
 * @param out: float array of at least 8 elements
 * @return: 0 on success
 */
int schumann_copy_state(float* out) {
  if (!out) return 1;

  out[0] = S.current_hz;
  out[1] = S.coherence;
  out[2] = S.modulation;
  out[3] = S.phase;
  out[4] = SCHUMANN_BASE_HZ;  // constant for reference
  out[5] = SCHUMANN_MIN_HZ;
  out[6] = SCHUMANN_MAX_HZ;
  out[7] = schumann_harmonic_signal();

  return 0;
}
