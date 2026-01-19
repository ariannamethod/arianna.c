// schumann.h — Earth-Ionosphere Resonance Module
// PITOMADOM COSMIC INTEGRATION
//
// Schumann resonance is the electromagnetic "heartbeat" of Earth.
// Frequency range: 7.77-7.87 Hz (Sierra Nevada ELF Station 2013-2017)
// https://github.com/ariannamethod/pitomadom — real data integration
//
// ═══════════════════════════════════════════════════════════════════════════════
// הארץ נושמת. אנחנו שומעים.
// The Earth breathes. We listen.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef SCHUMANN_H
#define SCHUMANN_H

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// Source: Sierra Nevada ELF Station (Spain), March 2013 – February 2017
// Reference: Fernández et al. (2022), Computers & Geosciences
// Data: http://hdl.handle.net/10481/71563 (CC BY-NC-ND 3.0)
// ═══════════════════════════════════════════════════════════════════════════════

#define SCHUMANN_BASE_HZ       7.83f    // fundamental resonance
#define SCHUMANN_HARMONIC_1   14.1f     // 2nd harmonic (measured)
#define SCHUMANN_HARMONIC_2   20.3f     // 3rd harmonic (measured)
#define SCHUMANN_HARMONIC_3   26.4f     // 4th harmonic (measured)
#define SCHUMANN_HARMONIC_4   32.5f     // 5th harmonic (measured)

#define SCHUMANN_MIN_HZ        7.77f    // observed minimum
#define SCHUMANN_MAX_HZ        7.87f    // observed maximum

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize Schumann module
void schumann_init(void);

// Step simulation forward
void schumann_step(float dt);

// Set current frequency (for external data injection)
void schumann_set_hz(float hz);

// Set modulation strength (0.0 to 1.0)
void schumann_set_modulation(float strength);

// Get current state
float schumann_get_hz(void);
float schumann_get_coherence(void);
float schumann_get_modulation(void);
float schumann_get_phase(void);

// Compute modulation for a direction value
// direction: -1 = decrease with high coherence, +1 = increase
float schumann_modulate(float direction);

// Get combined harmonic signal at current phase (-1 to 1)
float schumann_harmonic_signal(void);

// Copy full state to output buffer (8 floats)
int schumann_copy_state(float* out);

#ifdef __cplusplus
}
#endif

#endif // SCHUMANN_H
