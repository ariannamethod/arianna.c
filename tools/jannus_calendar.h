/* jannus-r/tools/jannus_calendar.h — Hebrew/Gregorian Calendar Drift.
 *
 * Bit-for-bit port of resonance-janus-bpe.c calendar block (lines 65-118).
 * Drives forward/backward step split in 12-step resonance:
 *   n_backward = NSTEPS * (0.3 + 0.4*prophecy_debt + 0.1*cal_dissonance)
 * and base temperature:
 *   tb = 0.7 + 0.3 * (0.5 + 0.3*cal_dissonance + 0.2*prophecy_debt)
 *
 * Annual drift between Gregorian (365.25) and Hebrew (354) calendars is
 * 11.25 days/year. Metonic cycle: 19 years, 7 leap months at years
 * {3,6,8,11,14,17,19}. Epoch: 1 Tishrei 5785 = 3 Oct 2024 noon (avoids
 * DST edge cases).
 *
 * Header-only, no globals besides the epoch t which is initialised on
 * first call. Safe to include from .aml's BLOOD COMPILE blocks.
 */
#ifndef JANNUS_CALENDAR_H
#define JANNUS_CALENDAR_H

#include <math.h>
#include <time.h>
#include <string.h>

#define JCAL_ANNUAL_DRIFT     11.25f
#define JCAL_GREGORIAN_YEAR   365.25f
#define JCAL_METONIC_YEARS    19
#define JCAL_METONIC_LEAPS    7
#define JCAL_MAX_UNCORRECTED  33.0f

static const int g_jcal_metonic[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t    g_jcal_epoch_t    = 0;

static inline float jcal_clamp01(float x) {
    if (!isfinite(x)) return 0.0f;
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

static inline void jcal_init(void) {
    if (g_jcal_epoch_t > 0) return;
    struct tm e;
    memset(&e, 0, sizeof(e));
    e.tm_year = 2024 - 1900;  /* 1 Tishrei 5785 ≈ 3 Oct 2024 noon */
    e.tm_mon  = 9;
    e.tm_mday = 3;
    e.tm_hour = 12;
    g_jcal_epoch_t = mktime(&e);
}

static inline int jcal_days_since_epoch(void) {
    jcal_init();
    if (g_jcal_epoch_t <= 0) return 0;
    return (int)(difftime(time(NULL), g_jcal_epoch_t) / 86400.0);
}

/* Cumulative drift in days, with Metonic leap-month corrections subtracted. */
static inline float jcal_cumulative_drift(int days) {
    float years = (float)days / JCAL_GREGORIAN_YEAR;
    float base  = years * JCAL_ANNUAL_DRIFT;
    int   full  = (int)(years / JCAL_METONIC_YEARS);
    float corr  = (float)(full * JCAL_METONIC_LEAPS) * 30.0f;
    float partial = fmodf(years, (float)JCAL_METONIC_YEARS);
    int   yic     = (int)partial + 1;
    for (int i = 0; i < JCAL_METONIC_LEAPS; i++)
        if (g_jcal_metonic[i] <= yic) corr += 30.0f;
    return base - corr;
}

/* Dissonance ∈ [0, 1]: |drift mod 33| / 33. The 33-day window is one
 * lunar/solar cycle of beats; full alignment ≈ 0, max misalignment ≈ 1. */
static inline float jcal_dissonance_at(int days) {
    float drift = jcal_cumulative_drift(days);
    return jcal_clamp01(fabsf(fmodf(drift, JCAL_MAX_UNCORRECTED)) / JCAL_MAX_UNCORRECTED);
}

static inline float jcal_dissonance_now(void) {
    return jcal_dissonance_at(jcal_days_since_epoch());
}

/* MetaJanus — birth snapshot. Same model as resonance-janus-bpe.c MJ. */
typedef struct {
    int    birth_days;
    float  birth_drift;
    float  birth_dissonance;
    time_t birth_time;
    int    alive;
} JcalMetaJanus;

static inline void jcal_metajanus_init(JcalMetaJanus *mj) {
    if (mj->alive) return;
    jcal_init();
    mj->birth_days       = jcal_days_since_epoch();
    mj->birth_drift      = jcal_cumulative_drift(mj->birth_days);
    mj->birth_dissonance = jcal_dissonance_at(mj->birth_days);
    mj->birth_time       = time(NULL);
    mj->alive            = 1;
}

static inline float jcal_personal_dissonance(const JcalMetaJanus *mj) {
    if (!mj->alive) return 0.5f;
    float now = jcal_cumulative_drift(jcal_days_since_epoch());
    return jcal_clamp01(fabsf(now - mj->birth_drift) / JCAL_MAX_UNCORRECTED);
}

#endif /* JANNUS_CALENDAR_H */
