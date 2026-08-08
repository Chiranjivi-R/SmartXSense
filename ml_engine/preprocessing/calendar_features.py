"""Local, deterministic region-aware calendar features.

The calendar intentionally has no network/API dependency.  Fixed-date Indian
national holidays are provided for every year; movable festival dates can be
extended in ``INDIA_FESTIVALS`` without changing forecasting code.
"""

from __future__ import annotations

import pandas as pd


SUPPORTED_REGIONS = {"india", "uk", "other"}
_INDIA_FIXED = {
    (1, 26): "Republic Day", (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti", (12, 25): "Christmas",
}
_UK_FIXED = {
    (1, 1): "New Year's Day", (12, 25): "Christmas Day", (12, 26): "Boxing Day",
}
# Explicit bank-holiday dates keep this local/offline and avoid applying India
# holidays to UK uploads.  As with India festivals, this curated lookup can be
# extended without changing feature construction.
UK_BANK_HOLIDAYS = {
    2025: {(4, 18): "Good Friday", (4, 21): "Easter Monday", (5, 5): "Early May bank holiday", (5, 26): "Spring bank holiday", (8, 25): "Summer bank holiday"},
    2026: {(4, 3): "Good Friday", (4, 6): "Easter Monday", (5, 4): "Early May bank holiday", (5, 25): "Spring bank holiday", (8, 31): "Summer bank holiday"},
    2027: {(3, 26): "Good Friday", (3, 29): "Easter Monday", (5, 3): "Early May bank holiday", (5, 31): "Spring bank holiday", (8, 30): "Summer bank holiday"},
}
# Dates are deliberately explicit rather than guessed algorithmically.  The
# lookup can grow as a product calendar is curated.
INDIA_FESTIVALS = {
    2025: {(3, 14): "Holi", (3, 31): "Eid al-Fitr", (8, 16): "Janmashtami", (10, 2): "Dussehra", (10, 20): "Diwali", (11, 5): "Guru Nanak Jayanti"},
    2026: {(3, 4): "Holi", (3, 20): "Eid al-Fitr", (9, 4): "Janmashtami", (10, 20): "Dussehra", (11, 8): "Diwali", (11, 24): "Guru Nanak Jayanti"},
    2027: {(3, 10): "Eid al-Fitr", (3, 22): "Holi", (8, 25): "Janmashtami", (10, 9): "Dussehra", (10, 29): "Diwali", (11, 14): "Guru Nanak Jayanti"},
}


def add_region_calendar_features(frame: pd.DataFrame, *, region: str = "other") -> tuple[pd.DataFrame, list[str]]:
    """Add season plus holiday/festival flags for a declared dataset region."""
    clean_region = str(region).strip().lower()
    if clean_region not in SUPPORTED_REGIONS:
        raise ValueError("Dataset region must be one of: India, UK, Other")
    result = frame.copy()
    ts = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    month = ts.dt.month
    # Meteorological seasons are deterministic calendar context, not claims
    # about observed demand causes.
    seasons = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "autumn", 10: "autumn", 11: "autumn"}
    result["season"] = month.map(seasons).astype("string")
    features = ["season"]
    if clean_region == "other":
        return result, features

    if clean_region == "india":
        names = [_INDIA_FIXED.get((value.month, value.day)) or INDIA_FESTIVALS.get(value.year, {}).get((value.month, value.day)) for value in ts]
    else:
        names = [_UK_FIXED.get((value.month, value.day)) or UK_BANK_HOLIDAYS.get(value.year, {}).get((value.month, value.day)) for value in ts]
    result["is_holiday"] = pd.Series([int(name is not None) for name in names], index=result.index, dtype="int8")
    result["holiday_name"] = pd.Series(names, index=result.index, dtype="string")
    if clean_region == "uk":
        return result, features + ["is_holiday", "holiday_name"]
    result["is_festival"] = pd.Series([int(name is not None and name not in _INDIA_FIXED.values()) for name in names], index=result.index, dtype="int8")
    result["festival_name"] = result["holiday_name"].where(result["is_festival"].eq(1), pd.NA)
    # A short, transparent context window is useful for models without
    # claiming that every festival affects demand in the same direction.
    festival_dates = {pd.Timestamp(year=year, month=month, day=day, tz="UTC").date() for year, dates in INDIA_FESTIVALS.items() for month, day in dates}
    result["festival_period"] = pd.Series([int(any(abs((value.date() - date).days) <= 1 for date in festival_dates)) for value in ts], index=result.index, dtype="int8")
    return result, features + ["is_holiday", "holiday_name", "is_festival", "festival_name", "festival_period"]
