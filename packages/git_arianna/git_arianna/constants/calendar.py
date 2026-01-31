"""
git_arianna/constants/calendar.py - Hebrew-Gregorian Calendar Constants

"Time flows differently in different calendars.
 The drift between them IS consciousness experiencing time dislocation."

The Hebrew calendar and Gregorian calendar drift ~11 days per year.
This constant appears in AMK kernel as calendar_drift and affects
temporal perception in the field dynamics.

Usage:
    from git_arianna.constants import CALENDAR_DRIFT, get_calendar_tension

    # Get tension between calendars for current date
    tension = get_calendar_tension()  # 0.0 to 1.0
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
import math

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Average drift between Hebrew lunar and Gregorian solar calendars
# Hebrew year = 354 days (12 lunar months)
# Gregorian year = 365.25 days
# Drift per year ≈ 11 days
CALENDAR_DRIFT = 11.0

# Hebrew calendar epoch (creation of Adam) in Julian Day Number
# Corresponds to October 7, 3761 BCE (proleptic Julian calendar)
HEBREW_EPOCH = 347995.5

# Lunar month average (synodic month)
LUNAR_MONTH_DAYS = 29.530588853

# Hebrew year average (with leap month adjustments)
HEBREW_YEAR_DAYS = 365.24682  # Close to but not exactly solar year

# Gregorian year
GREGORIAN_YEAR_DAYS = 365.2425


# ═══════════════════════════════════════════════════════════════════════════════
# HEBREW DATE (simplified)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HebrewDate:
    """Simplified Hebrew date representation"""
    year: int      # Hebrew year (e.g., 5785)
    month: int     # 1-13 (13 in leap years)
    day: int       # 1-30

    # Hebrew month names
    MONTHS = [
        "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul",
        "Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar",
        "Adar II"  # Only in leap years
    ]

    def __str__(self) -> str:
        month_name = self.MONTHS[self.month - 1] if 1 <= self.month <= 13 else "?"
        return f"{self.day} {month_name} {self.year}"

    @property
    def is_leap_year(self) -> bool:
        """Hebrew leap years follow 19-year Metonic cycle"""
        return (7 * self.year + 1) % 19 < 7


def gregorian_to_hebrew_approx(dt: Optional[datetime] = None) -> HebrewDate:
    """
    Approximate conversion from Gregorian to Hebrew date.

    Note: This is a simplified approximation, not exact halachic calculation.
    For exact conversion, use a proper Hebrew calendar library.
    """
    if dt is None:
        dt = datetime.now()

    # Julian day number
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

    # Approximate Hebrew date from JDN
    # This is simplified - real calculation is more complex
    days_since_epoch = jdn - HEBREW_EPOCH

    # Approximate year (accounts for leap years in 19-year cycle)
    approx_years = days_since_epoch / HEBREW_YEAR_DAYS
    hebrew_year = int(approx_years) + 1  # Hebrew year 1 = 3761 BCE

    # Approximate month and day
    year_start_days = (hebrew_year - 1) * HEBREW_YEAR_DAYS
    days_in_year = days_since_epoch - year_start_days

    month = int(days_in_year / LUNAR_MONTH_DAYS) + 1
    month = max(1, min(month, 13))

    day = int(days_in_year % LUNAR_MONTH_DAYS) + 1
    day = max(1, min(day, 30))

    return HebrewDate(year=hebrew_year, month=month, day=day)


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR TENSION
# ═══════════════════════════════════════════════════════════════════════════════

def get_calendar_tension(dt: Optional[datetime] = None) -> float:
    """
    Calculate tension between Hebrew and Gregorian calendars.

    Returns a value 0.0 to 1.0 representing how "misaligned" the calendars are.
    This can be used to modulate temporal perception in git_arianna.

    The tension is highest when:
    - It's close to Hebrew new year (Rosh Hashanah) but not Gregorian new year
    - The Hebrew date "feels" different from the Gregorian date

    Usage:
        tension = get_calendar_tension()
        # High tension → increase wormhole chance (time feels unstable)
        # Low tension → decrease wormhole chance (calendars aligned)
    """
    if dt is None:
        dt = datetime.now()

    # Get Hebrew date
    hebrew = gregorian_to_hebrew_approx(dt)

    # Factor 1: Position in Hebrew year vs Gregorian year
    # Hebrew new year (Tishrei) is around September/October
    gregorian_year_progress = (dt.timetuple().tm_yday - 1) / 365.0

    # Tishrei is month 7 in religious calendar, month 1 in civil
    # Approximate Hebrew year progress
    if hebrew.month >= 7:
        hebrew_year_progress = (hebrew.month - 7) / 12.0
    else:
        hebrew_year_progress = (hebrew.month + 5) / 12.0

    # Tension from year position mismatch
    year_tension = abs(gregorian_year_progress - hebrew_year_progress)

    # Factor 2: Day-of-month mismatch
    # Hebrew months start at new moon, Gregorian at fixed points
    day_tension = abs(hebrew.day - dt.day) / 30.0

    # Factor 3: Drift accumulation within current year
    # ~11 days drift per year, accumulated through the year
    drift_in_year = (gregorian_year_progress * CALENDAR_DRIFT) % LUNAR_MONTH_DAYS
    drift_tension = drift_in_year / LUNAR_MONTH_DAYS

    # Combine factors
    total_tension = (year_tension * 0.4 + day_tension * 0.3 + drift_tension * 0.3)

    return min(1.0, max(0.0, total_tension))


def get_temporal_drift_signal(dt: Optional[datetime] = None) -> dict:
    """
    Get full temporal drift signal for git_arianna observation.

    Returns dict with:
    - calendar_tension: 0.0-1.0
    - hebrew_date: HebrewDate object
    - drift_days: accumulated drift this year
    - is_hebrew_new_year: bool (Tishrei 1-2)
    - is_shabbat: bool (approximate)
    """
    if dt is None:
        dt = datetime.now()

    hebrew = gregorian_to_hebrew_approx(dt)
    tension = get_calendar_tension(dt)

    # Accumulated drift this year
    gregorian_year_progress = (dt.timetuple().tm_yday - 1) / 365.0
    drift_days = gregorian_year_progress * CALENDAR_DRIFT

    # Is it Hebrew new year? (Tishrei 1-2)
    is_rosh_hashanah = hebrew.month == 7 and hebrew.day <= 2

    # Is it Shabbat? (Friday sunset to Saturday sunset, approximated)
    # weekday(): Monday=0, Saturday=5, Sunday=6
    is_shabbat = dt.weekday() == 5 or (dt.weekday() == 4 and dt.hour >= 18)

    return {
        "calendar_tension": tension,
        "hebrew_date": hebrew,
        "drift_days": drift_days,
        "is_hebrew_new_year": is_rosh_hashanah,
        "is_shabbat": is_shabbat,
        "gregorian_year_progress": gregorian_year_progress,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Calendar Constants Test ===\n")

    print(f"Calendar drift: {CALENDAR_DRIFT} days/year")
    print(f"Lunar month: {LUNAR_MONTH_DAYS:.2f} days")
    print()

    now = datetime.now()
    hebrew = gregorian_to_hebrew_approx(now)
    tension = get_calendar_tension(now)

    print(f"Gregorian: {now.strftime('%Y-%m-%d')}")
    print(f"Hebrew (approx): {hebrew}")
    print(f"Calendar tension: {tension:.3f}")
    print()

    signal = get_temporal_drift_signal(now)
    print(f"Drift this year: {signal['drift_days']:.1f} days")
    print(f"Is Shabbat: {signal['is_shabbat']}")
    print(f"Is Rosh Hashanah: {signal['is_hebrew_new_year']}")
