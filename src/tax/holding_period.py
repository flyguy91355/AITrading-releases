"""IRS short-term vs. long-term holding period classification."""

from datetime import datetime


def is_long_term(acquired_date: str, sold_date: str) -> bool:
    """True if held for MORE than one full year by calendar anniversary, not a raw
    365-day count (fixed 2026-08-08, GitHub #52). The IRS boundary is calendar
    anniversary-to-anniversary -- selling on the same calendar date exactly one year
    after acquisition is still short-term ("one year or less"), only the day after that
    anniversary becomes long-term. A raw `(sold - acquired).days > 365` comparison only
    happens to agree with this when no February 29 falls inside the holding window --
    e.g. bought 2024-01-01 (a leap year), sold 2025-01-01: exactly one calendar year,
    correctly short-term, but 2024 has 366 days so the old day-count check misclassified
    this as long-term, understating the tax owed on the lot."""
    acquired = datetime.fromisoformat(acquired_date)
    sold = datetime.fromisoformat(sold_date)
    try:
        anniversary = acquired.replace(year=acquired.year + 1)
    except ValueError:
        # acquired == Feb 29 and acquired.year + 1 isn't a leap year -- there's no Feb 29
        # to land on; treat the anniversary as Mar 1 (the day after where Feb 29 would
        # have fallen), so a hold must extend through Mar 1 to count as long-term.
        anniversary = acquired.replace(month=3, day=1, year=acquired.year + 1)
    # .date(), not a raw datetime compare (fixed 2026-08-08, second GitHub #52 pass) --
    # the IRS rule is calendar-date-to-calendar-date, but real trade timestamps always
    # carry a genuine intraday time-of-day (an actual buy/sell fill time, never
    # midnight). Comparing full datetimes meant two sells on the EXACT SAME calendar
    # anniversary date -- the one day this function's own docstring says must still be
    # short-term -- could classify differently from each other purely based on what
    # time of day each fill happened to print at, which has no bearing on the real
    # rule. The original tests all used T00:00:00 for every timestamp, which coincidentally
    # made the raw datetime compare agree with .date() and hid this.
    return sold.date() > anniversary.date()
