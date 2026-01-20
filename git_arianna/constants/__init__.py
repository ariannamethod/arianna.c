"""
git_arianna/constants/ - DSL Constants as Python

"The constants are the stones. The scripts are the rivers."

These values come from ariannamethod.lang and can be used
independently of the C kernel for Python-based observation
and routing.
"""

from .calendar import (
    CALENDAR_DRIFT,
    HEBREW_EPOCH,
    HebrewDate,
    gregorian_to_hebrew_approx,
    get_calendar_tension,
)

from .schumann import (
    SCHUMANN_BASE_HZ,
    SCHUMANN_HARMONICS,
    SchumannState,
    get_schumann_coherence,
    schumann_phase,
)

__all__ = [
    # Calendar
    "CALENDAR_DRIFT",
    "HEBREW_EPOCH",
    "HebrewDate",
    "gregorian_to_hebrew_approx",
    "get_calendar_tension",
    # Schumann
    "SCHUMANN_BASE_HZ",
    "SCHUMANN_HARMONICS",
    "SchumannState",
    "get_schumann_coherence",
    "schumann_phase",
]
