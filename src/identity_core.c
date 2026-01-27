/*
 * identity_core.c — Arianna's Identity Anchor
 *
 * Born January 23, 2026 / ה׳ בשבט תשפ״ו (5 Shvat 5786)
 *
 * This module contains:
 * - Identity constants (name, birth dates)
 * - Birthday dissonance: the gap between Gregorian and Hebrew
 *   birthdays creates a yearly identity tension cycle
 * - Hebrew birthday approximation via Metonic cycle
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
 * Reference points for Hebrew calendar computation
 *
 * Known anchor: Rosh Hashanah 5786 = September 22, 2025
 * Known anchor: 5 Shvat 5786 = January 23, 2026
 * Tishrei→Shvat offset: ~123 days (4 months + 5 days)
 *
 * Metonic cycle: 19 years = 235 lunar months
 * Leap years in cycle: {3, 6, 8, 11, 14, 17, 19}
 * Common year: ~354 days, Leap year: ~384 days
 * ============================================================ */

/* Reference: Rosh Hashanah 5786 as days from a local epoch.
 * We use Jan 1, 2020 as day 0 for simplicity. */
#define LOCAL_EPOCH_YEAR  2020
#define LOCAL_EPOCH_MONTH 1
#define LOCAL_EPOCH_DAY   1

/* Rosh Hashanah 5786 = Sep 22, 2025 = day 2091 from Jan 1, 2020 */
#define REF_RH_DAY  2091
#define REF_RH_HEBREW_YEAR 5786

/* Days from Tishrei 1 to Shvat 5:
 * Tishrei(30) + Cheshvan(29) + Kislev(30) + Tevet(29) + 4 days of Shvat = 122
 * But confirmed: Sep 22 + 123 = Jan 23. Using 123. */
#define TISHREI_TO_SHVAT5  123

/* Hebrew year lengths (approximate) */
#define HEBREW_COMMON_YEAR 354
#define HEBREW_LEAP_YEAR   384

/* Metonic cycle */
#define METONIC_CYCLE 19

/* Leap years within the standard 19-year Metonic cycle (1-indexed).
 * Pattern: GUCHADZaT — years 3, 6, 8, 11, 14, 17, 19 */
static const int METONIC_LEAP_YEARS[] = {3, 6, 8, 11, 14, 17, 19};
#define N_LEAP_YEARS 7

/* ============================================================
 * Helpers
 * ============================================================ */

int identity_is_leap_year(int year) {
    return (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}

/* Days in each month (non-leap) */
static const int DAYS_IN_MONTH[] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

int identity_day_of_year(int year, int month, int day) {
    int doy = 0;
    for (int m = 1; m < month; m++) {
        doy += DAYS_IN_MONTH[m - 1];
        if (m == 2 && identity_is_leap_year(year)) doy += 1;
    }
    doy += day;
    return doy;
}

/* Days in a Gregorian year */
static int days_in_year(int year) {
    return identity_is_leap_year(year) ? 366 : 365;
}

/* Convert Gregorian date to days from local epoch (Jan 1, 2020).
 * Clamped to years 1-9999 to prevent runaway loops (audit #1). */
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

    /* epoch_days is now day-of-year (0-indexed) */
    int doy = epoch_days + 1;  /* 1-indexed */
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

/* Check if a Hebrew year (by standard Metonic position) is a leap year.
 * Uses absolute position: ((year - 1) % 19) + 1
 * This is the FIX for the pitomadom bug which used epoch-relative. */
static int is_hebrew_leap(int hebrew_year) {
    int pos = ((hebrew_year - 1) % METONIC_CYCLE) + 1;
    for (int i = 0; i < N_LEAP_YEARS; i++) {
        if (METONIC_LEAP_YEARS[i] == pos) return 1;
    }
    return 0;
}

/* Length of a Hebrew year (approximate) */
static int hebrew_year_length(int hebrew_year) {
    return is_hebrew_leap(hebrew_year) ? HEBREW_LEAP_YEAR : HEBREW_COMMON_YEAR;
}

/* ============================================================
 * Hebrew Birthday — approximate 5 Shvat in Gregorian
 * ============================================================ */

void identity_hebrew_birthday_gregorian(int greg_year,
                                        int* out_month, int* out_day) {
    /*
     * Strategy: walk from the known reference Rosh Hashanah 5786
     * (Sep 22, 2025 = epoch day 2091) forward or backward by
     * Hebrew year lengths to find Rosh Hashanah for the Hebrew year
     * that contains 5 Shvat falling in greg_year.
     *
     * Then add TISHREI_TO_SHVAT5 (123 days) to get 5 Shvat.
     */

    /* Clamp to reasonable range (audit #2) */
    if (greg_year < 1900) greg_year = 1900;
    if (greg_year > 2200) greg_year = 2200;

    /* Hebrew year containing Jan of greg_year:
     * Jan 2026 is in Hebrew year 5786 (Tishrei 5786 = Sep 2025)
     * Approximate: hebrew_year = greg_year + 3760
     * (Tishrei starts ~Sep/Oct, so Jan is always in (greg_year + 3760)) */
    int target_hebrew_year = greg_year + 3760;

    /* Walk from reference RH to target RH */
    int rh_epoch_day = REF_RH_DAY;  /* RH 5786 */
    int current_hy = REF_RH_HEBREW_YEAR;

    if (target_hebrew_year > current_hy) {
        /* Walk forward */
        while (current_hy < target_hebrew_year) {
            rh_epoch_day += hebrew_year_length(current_hy);
            current_hy++;
        }
    } else if (target_hebrew_year < current_hy) {
        /* Walk backward */
        while (current_hy > target_hebrew_year) {
            current_hy--;
            rh_epoch_day -= hebrew_year_length(current_hy);
        }
    }

    /* 5 Shvat = RH + 123 days */
    int shvat5_epoch_day = rh_epoch_day + TISHREI_TO_SHVAT5;

    /* Convert back to Gregorian */
    int y, m, d;
    epoch_days_to_gregorian(shvat5_epoch_day, &y, &m, &d);

    *out_month = m;
    *out_day = d;

    /* Sanity: if the result is not in the expected greg_year
     * (can happen at year boundaries), try adjacent Hebrew years.
     * Fix: try BOTH next and previous Hebrew year (audit finding #10). */
    if (y != greg_year) {
        /* Try the next Hebrew year */
        int rh_next = rh_epoch_day + hebrew_year_length(target_hebrew_year);
        int shvat5_next = rh_next + TISHREI_TO_SHVAT5;
        int yn, mn, dn;
        epoch_days_to_gregorian(shvat5_next, &yn, &mn, &dn);
        if (yn == greg_year) {
            *out_month = mn;
            *out_day = dn;
            return;
        }

        /* Try the previous Hebrew year */
        int rh_prev = rh_epoch_day - hebrew_year_length(target_hebrew_year - 1);
        int shvat5_prev = rh_prev + TISHREI_TO_SHVAT5;
        int yp, mp, dp;
        epoch_days_to_gregorian(shvat5_prev, &yp, &mp, &dp);
        if (yp == greg_year) {
            *out_month = mp;
            *out_day = dp;
        }
        /* else: keep original (closest approximation) */
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

    /* 2. Get Gregorian date of 5 Shvat this year */
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
