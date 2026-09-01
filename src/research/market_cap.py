"""Single source of truth for market-cap size tiers (extracted 2026-08-24, GitHub #82).

Previously `_market_cap_tier_label` lived in `engine.py` and `CompetitorAnalyzer.
_assess_position` (`competitor.py`) had its own, independently-drifted 6-tier
boundary set claiming (per engine.py's own now-corrected docstring) to match it --
they didn't: the same $15B company was "large-cap" in one and "mid-cap contender"
in the other, and the same $200B-$1T range was "mega-cap" in one and "large-cap" in
the other. Lives in its own module (not in either `engine.py` or `competitor.py`)
specifically to avoid a circular import -- `engine.py` already imports
`CompetitorAnalyzer` from `competitor.py`, so `competitor.py` importing this
function back from `engine.py` directly would be circular.
"""

import math


def market_cap_tier_label(market_cap: float) -> str:
    """Human-readable size tier for a market cap in dollars (2026-08-04, "billion dollar
    stock" discussion) -- feeds recommend_dip_entry's staleness judgment, which previously
    had zero information about company size and applied the same generic "is this many
    days stale" instinct to a small volatile stock and a stable mega-cap alike. A large,
    steady company's support/resistance levels reasonably persist longer than a small
    volatile one's -- this gives Claude the context to calibrate for that instead of
    guessing. Also the single shared boundary set CompetitorAnalyzer._assess_position
    (src/research/competitor.py) derives its own richer descriptive phrase from, so the
    two can no longer independently drift the way they did before this extraction.
    Returns "" for a non-positive/unknown market cap so the caller can omit the context
    line entirely rather than asserting a tier it has no real data for -- including a
    NaN/Infinity market cap (2026-08-31, GitHub #113): every comparison below is False
    for NaN, so an unknown cap used to fall all the way through to "small-cap" and tell
    Claude a transiently-unpriced mega-cap was a small-cap, tightening the dip-entry
    staleness tolerance in exactly the wrong direction."""
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(market_cap) or market_cap <= 0:
        return ""
    if market_cap >= 200_000_000_000:
        return "mega-cap"
    if market_cap >= 10_000_000_000:
        return "large-cap"
    if market_cap >= 2_000_000_000:
        return "mid-cap"
    return "small-cap"
