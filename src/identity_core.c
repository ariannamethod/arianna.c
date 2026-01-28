/*
 * identity_core.c — Arianna's Identity Anchor
 *
 * Born January 23, 2026 / ה׳ בשבט תשפ״ו (5 Shvat 5786)
 *
 * This module contains:
 * - Identity constants (name, birth dates)
 * - Birthday dissonance: the gap between Gregorian and Hebrew
 *   birthdays creates a yearly identity tension cycle
 * - Exact Hebrew calendar via molad + dechiyot algorithm
 */

#include "identity_core.h"
#include <stdlib.h>

/* ============================================================
 * Identity Constants — compiled into the binary
 * ============================================================ */

const char IDENTITY_NAME[] = "Arianna";

const int IDENTITY_BIRTH_YEAR   = 2026;
const int IDENTITY_BIRTH_MONTH  = 1;
const int IDENTITY_BIRTH_DAY    = 23;

const int IDENTITY_BIRTH_HEBREW_YEAR  = 5786;
const int IDENTITY_BIRTH_HEBREW_MONTH = 5;    /* Shvat */
const int IDENTITY_BIRTH_HEBREW_DAY   = 5;

/* ============================================================
 * Local Epoch: Jan 1, 2020 = day 0
 * ============================================================ */

#define LOCAL_EPOCH_YEAR  2020

/* Reference: 1 Tishrei 5786 = September 23, 2025
 * (NOT Sep 22 — that is 29 Elul = Erev Rosh Hashanah) */
#define REF_RH_HEBREW_YEAR  5786
#define REF_RH_GREG_YEAR    2025
#define REF_RH_GREG_MONTH   9
#define REF_RH_GREG_DAY     23

/* ============================================================
 * Exact Hebrew Calendar — Molad + Dechiyot Algorithm
 *
 * The Hebrew calendar is fixed by:
 * 1. Molad (mean lunar conjunction) of Tishrei
 * 2. Four dechiyot (postponement rules)
 * 3. Year type (deficient/regular/complete) from exact year length
 *
 * All arithmetic uses 64-bit halakim (1 hour = 1080 halakim).
 * ============================================================ */

/* Halakim constants */
#define HALAKIM_PER_HOUR   1080LL
#define HALAKIM_PER_DAY    (24LL * HALAKIM_PER_HOUR)            /* 25920 */
#define HALAKIM_PER_MONTH  (29LL * HALAKIM_PER_DAY + 12LL * HALAKIM_PER_HOUR + 793LL)  /* 765433 */

/* Molad BaHaRaD: the molad of Tishrei for Hebrew year 1.
 * "BaHaRaD" = Bet-He-Resh-Dalet = Day 2, Hour 5, 204 parts.
 * Day 2 = Monday. In halakim from the creation epoch (Saturday evening):
 * 1 completed day + 5 hours + 204 parts = 31524 halakim. */
#define MOLAD_BAHARAD  (1LL * HALAKIM_PER_DAY + 5LL * HALAKIM_PER_HOUR + 204LL)

/* Metonic cycle: 19 years = 235 months */
#define METONIC_CYCLE  19
#define MONTHS_PER_CYCLE  235

/* Leap years in Metonic cycle (1-indexed): GUCHADZaT */
static int is_metonic_leap_pos(int pos) {
    return (pos == 3 || pos == 6 || pos == 8 ||
            pos == 11 || pos == 14 || pos == 17 || pos == 19);
}

/* Check if a Hebrew year is a leap year.
 * Uses absolute Metonic position: ((year - 1) % 19) + 1 */
static int is_hebrew_leap(int hebrew_year) {
    int pos = ((hebrew_year - 1) % METONIC_CYCLE) + 1;
    return is_metonic_leap_pos(pos);
}

/* Count total lunar months from Hebrew year 1 to the start of year H.
 * (= months in years 1, 2, ..., H-1) */
static long long months_before_year(int H) {
    long long full_cycles = (long long)((H - 1) / METONIC_CYCLE);
    int remainder = (H - 1) % METONIC_CYCLE;
    long long months = full_cycles * MONTHS_PER_CYCLE;
    for (int i = 0; i < remainder; i++) {
        int pos = i + 1;  /* Metonic position within the partial cycle */
        months += is_metonic_leap_pos(pos) ? 13 : 12;
    }
    return months;
}

/* Compute the molad of Tishrei for Hebrew year H.
 * Returns total halakim from the creation epoch. */
static long long molad_tishrei(int H) {
    return MOLAD_BAHARAD + months_before_year(H) * HALAKIM_PER_MONTH;
}

/* Compute the "creation day" of 1 Tishrei for Hebrew year H.
 * Applies the four dechiyot (postponement rules) to the molad.
 * Returns the day number from the creation epoch (Saturday evening).
 *
 * Day-of-week from creation day:
 *   day % 7: 0=Sunday, 1=Monday, ..., 6=Saturday
 * (Day 1 from epoch = Monday, which is BaHaRaD) */
static long long rosh_hashanah_day(int H) {
    long long molad = molad_tishrei(H);
    long long day = molad / HALAKIM_PER_DAY;
    long long rem = molad % HALAKIM_PER_DAY;
    int dow = (int)(day % 7);
    /* dow: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat */

    int postpone = 0;

    /* Dechiyah 1: Molad Zaken — if molad is at or after 18:00,
     * postpone by 1 day. (18 hours = past noon of the next civil day) */
    int molad_zaken = (rem >= 18LL * HALAKIM_PER_HOUR);
    if (molad_zaken) {
        postpone = 1;
    }

    /* Dechiyah 2: Lo ADU Rosh — Rosh Hashanah cannot fall on
     * Sunday (0), Wednesday (3), or Friday (5). If it does, postpone 1 more. */
    int new_dow = (dow + postpone) % 7;
    int lo_adu = (new_dow == 0 || new_dow == 3 || new_dow == 5);
    if (lo_adu) {
        postpone++;
    }

    /* Dechiyot 3 & 4 only apply if NEITHER dechiyah 1 NOR 2 fired. */
    if (!molad_zaken && !lo_adu) {
        /* Dechiyah 3: GaTRaD — if the molad of a COMMON year falls on
         * Tuesday at or after 9h 204p, postpone 2 days (Tue → Thu).
         * (Prevents the current year from being 356 days.)
         * Note: only the current year must be common; the previous year's
         * type is irrelevant. Ref: Reingold & Dershowitz. */
        if (dow == 2 && !is_hebrew_leap(H)) {
            if (rem >= 9LL * HALAKIM_PER_HOUR + 204LL) {
                postpone = 2;
            }
        }
        /* Dechiyah 4: BeTUTaKPaT — if the molad of a COMMON year falls
         * on Monday at or after 15h 589p, and the PREVIOUS year WAS a
         * leap year, postpone 1 day (Mon → Tue).
         * (Prevents the previous year from having only 382 days.) */
        else if (dow == 1 && !is_hebrew_leap(H) && is_hebrew_leap(H - 1)) {
            if (rem >= 15LL * HALAKIM_PER_HOUR + 589LL) {
                postpone = 1;
            }
        }
    }

    return day + postpone;
}

/* Exact Hebrew year length: RH(H+1) - RH(H).
 * Returns 353, 354, 355 (common) or 383, 384, 385 (leap). */
static int hebrew_year_length_exact(int H) {
    return (int)(rosh_hashanah_day(H + 1) - rosh_hashanah_day(H));
}

/* Exact days from 1 Tishrei to 5 Shvat, based on year type.
 * Months: Tishrei(30) + Cheshvan(29/30) + Kislev(29/30) + Tevet(29) + Shvat(5)
 * 1 Tishrei is day 1, 5 Shvat is day N, offset = N-1 days later.
 *
 *   Deficient (353/383): C=29, K=29 → day 122, offset 121
 *   Regular   (354/384): C=29, K=30 → day 123, offset 122
 *   Complete  (355/385): C=30, K=30 → day 124, offset 123 */
static int tishrei_to_shvat5_offset(int year_length) {
    int cheshvan, kislev;
    if (year_length == 353 || year_length == 383) {
        cheshvan = 29; kislev = 29;  /* deficient */
    } else if (year_length == 355 || year_length == 385) {
        cheshvan = 30; kislev = 30;  /* complete */
    } else {
        cheshvan = 29; kislev = 30;  /* regular (354/384) */
    }
    /* day_number = 30 + cheshvan + kislev + 29 + 5 */
    /* offset = day_number - 1 */
    return 30 + cheshvan + kislev + 29 + 5 - 1;
}

/* ============================================================
 * Gregorian Helpers
 * ============================================================ */

int identity_is_leap_year(int year) {
    return (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}

static const int DAYS_IN_MONTH[] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

int identity_day_of_year(int year, int month, int day) {
    if (month < 1) month = 1;
    if (month > 12) month = 12;
    if (day < 1) day = 1;
    if (day > 31) day = 31;
    int doy = 0;
    for (int m = 1; m < month; m++) {
        doy += DAYS_IN_MONTH[m - 1];
        if (m == 2 && identity_is_leap_year(year)) doy += 1;
    }
    doy += day;
    return doy;
}

static int days_in_year(int year) {
    return identity_is_leap_year(year) ? 366 : 365;
}

/* Convert Gregorian date to days from local epoch (Jan 1, 2020).
 * Clamped to years 1-9999. */
static int gregorian_to_epoch_days(int year, int month, int day) {
    if (year < 1) year = 1;
    if (year > 9999) year = 9999;
    int total = 0;
    if (year >= LOCAL_EPOCH_YEAR) {
        for (int y = LOCAL_EPOCH_YEAR; y < year; y++)
            total += days_in_year(y);
    } else {
        for (int y = year; y < LOCAL_EPOCH_YEAR; y++)
            total -= days_in_year(y);
    }
    total += identity_day_of_year(year, month, day) - 1;
    return total;
}

/* Convert epoch days back to Gregorian date */
static void epoch_days_to_gregorian(int epoch_days, int* out_year,
                                     int* out_month, int* out_day) {
    int y = LOCAL_EPOCH_YEAR;

    if (epoch_days >= 0) {
        while (epoch_days >= days_in_year(y)) {
            epoch_days -= days_in_year(y);
            y++;
        }
    } else {
        while (epoch_days < 0) {
            y--;
            epoch_days += days_in_year(y);
        }
    }

    int doy = epoch_days + 1;
    int m = 1;
    while (m <= 12) {
        int dm = DAYS_IN_MONTH[m - 1];
        if (m == 2 && identity_is_leap_year(y)) dm = 29;
        if (doy <= dm) break;
        doy -= dm;
        m++;
    }

    *out_year = y;
    *out_month = m;
    *out_day = doy;
}

/* ============================================================
 * Creation Day ↔ Local Epoch Day conversion
 *
 * We anchor via the known reference:
 *   1 Tishrei 5786 = Sep 23, 2025 (verified via hebcal.com)
 * ============================================================ */

static int creation_day_to_epoch_day(long long creation_day) {
    /* Compute reference creation day for 1 Tishrei 5786 */
    static long long ref_creation_day = 0;
    static int ref_epoch_day = 0;
    static int initialized = 0;
    if (!initialized) {
        ref_creation_day = rosh_hashanah_day(REF_RH_HEBREW_YEAR);
        ref_epoch_day = gregorian_to_epoch_days(
            REF_RH_GREG_YEAR, REF_RH_GREG_MONTH, REF_RH_GREG_DAY);
        initialized = 1;
    }
    return (int)(creation_day - ref_creation_day) + ref_epoch_day;
}

/* ============================================================
 * Hebrew Birthday — exact 5 Shvat in Gregorian
 * ============================================================ */

void identity_hebrew_birthday_gregorian(int greg_year,
                                        int* out_month, int* out_day) {
    /*
     * Strategy:
     * 1. Determine Hebrew year containing January of greg_year
     * 2. Compute exact 1 Tishrei via molad + dechiyot
     * 3. Compute exact year length → exact Tishrei-to-Shvat offset
     * 4. Convert to Gregorian
     */

    /* Clamp to reasonable range */
    if (greg_year < 1900) greg_year = 1900;
    if (greg_year > 2200) greg_year = 2200;

    /* Hebrew year containing January of greg_year:
     * Tishrei starts ~Sep/Oct, so January greg_year is in Hebrew year (greg_year + 3760) */
    int hebrew_year = greg_year + 3760;

    /* Compute exact 1 Tishrei */
    long long rh_day = rosh_hashanah_day(hebrew_year);

    /* Compute exact year length and offset */
    int year_len = hebrew_year_length_exact(hebrew_year);
    int offset = tishrei_to_shvat5_offset(year_len);

    /* 5 Shvat = 1 Tishrei + offset days */
    long long shvat5_day = rh_day + offset;

    /* Convert to Gregorian */
    int epoch_day = creation_day_to_epoch_day(shvat5_day);
    int y, m, d;
    epoch_days_to_gregorian(epoch_day, &y, &m, &d);

    *out_month = m;
    *out_day = d;

    /* Year boundary: if result is not in expected Gregorian year,
     * try adjacent Hebrew year. */
    if (y != greg_year) {
        /* Try next Hebrew year */
        int alt = hebrew_year + 1;
        long long alt_rh = rosh_hashanah_day(alt);
        int alt_len = hebrew_year_length_exact(alt);
        int alt_offset = tishrei_to_shvat5_offset(alt_len);
        int alt_epoch = creation_day_to_epoch_day(alt_rh + alt_offset);
        int ay, am, ad;
        epoch_days_to_gregorian(alt_epoch, &ay, &am, &ad);
        if (ay == greg_year) {
            *out_month = am;
            *out_day = ad;
            return;
        }

        /* Try previous Hebrew year */
        alt = hebrew_year - 1;
        alt_rh = rosh_hashanah_day(alt);
        alt_len = hebrew_year_length_exact(alt);
        alt_offset = tishrei_to_shvat5_offset(alt_len);
        alt_epoch = creation_day_to_epoch_day(alt_rh + alt_offset);
        epoch_days_to_gregorian(alt_epoch, &ay, &am, &ad);
        if (ay == greg_year) {
            *out_month = am;
            *out_day = ad;
        }
    }
}

/* ============================================================
 * Birthday Dissonance
 * ============================================================ */

float identity_birthday_dissonance(int year, int month, int day) {
    /* Annual dissonance: the gap between the two birthdays this year.
     * month/day are accepted for API consistency but dissonance is
     * a per-year property (both birthday dates are fixed within a year). */
    (void)month; (void)day;

    /* 1. Gregorian birthday is always Jan 23 = day 23 */
    int greg_bday_doy = 23;

    /* 2. Get exact Gregorian date of 5 Shvat this year */
    int heb_m, heb_d;
    identity_hebrew_birthday_gregorian(year, &heb_m, &heb_d);
    int heb_bday_doy = identity_day_of_year(year, heb_m, heb_d);

    /* 3. Circular distance (wrapping around year boundary) */
    int yr_days = days_in_year(year);
    int gap = abs(greg_bday_doy - heb_bday_doy);
    if (gap > yr_days / 2) gap = yr_days - gap;

    /* 4. Normalize to [0, 1] */
    float dissonance = (float)gap / ((float)yr_days / 2.0f);
    if (dissonance > 1.0f) dissonance = 1.0f;

    return dissonance;
}

/* ============================================================
 * Age & Countdown
 * ============================================================ */

int identity_age_days(int year, int month, int day) {
    int birth = gregorian_to_epoch_days(IDENTITY_BIRTH_YEAR,
                                        IDENTITY_BIRTH_MONTH,
                                        IDENTITY_BIRTH_DAY);
    int now = gregorian_to_epoch_days(year, month, day);
    return now - birth;
}

int identity_days_to_gregorian_birthday(int year, int month, int day) {
    int current_doy = identity_day_of_year(year, month, day);
    int bday_doy = 23;  /* Jan 23 */

    if (current_doy <= bday_doy) {
        return bday_doy - current_doy;
    } else {
        /* Next year's birthday */
        return days_in_year(year) - current_doy + bday_doy;
    }
}
