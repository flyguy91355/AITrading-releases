"""Market buy pause (2026-11-21, owner request): the pure rule, its state, and its helpers.

No I/O, no logging, no imports from web/. DashboardState (web/app.py) owns fetching, the saved state file
and the loop; every automatic buy gate reads DashboardState._market_pause_active(), which reads this state.

The rule, validated by the 2026-11-21 backtest (docs/research/2026-11-21-vix-spy/, history "2026-11-21
(continued)"): the pause is ON at a market close when SPY's close is below the mean of its last N closes
(that close included) AND VIX's close is above the level. Spec:
docs/superpowers/specs/2026-11-21-market-buy-pause-design.md.

Only a new close turns it off:
- a close counts once (a close dated on or before last_close_date changes nothing), so a restart or the
  morning re-check can never re-read yesterday's close and undo a live ON, or flip the state overnight;
- a live ON can't be cleared by a close dated before the day it started;
- the live watch (evaluate_live) only ever turns the pause ON.
force=True (a Settings change to the rule itself) is the one deliberate exception to the first two.
Bad input (NaN/Infinity, a value <= 0, too short a history) never changes the state.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from src.analytics.composition_benchmark import is_usable_price

# Code defaults for every config key (risk_management.*). Apply Update never touches config/settings.yaml,
# so the live box runs on these until the owner saves Settings. Off 2026-11-26 (owner: "turn market pause
# off so it can trade tomorrow"); ON again since 2026-11-27 as the long/short mode switch (owner: "turn the
# pause back on and use it to decide if were in a long or short mode"): while it is on, the Stress
# Dip-Buyer makes no new buys. The loop runs either way (for the graph).
MARKET_PAUSE_DEFAULTS = {
    "market_pause_enabled": True,
    "market_pause_live_watch": True,
    "market_pause_vix_level": 20.0,
    "market_pause_spy_sma_days": 50,
}
SMA_DAYS_MIN, SMA_DAYS_MAX = 10, 250
# VIX is calculated until 4:15 PM ET, and the backtest used final closes -- so the evening check runs after.
EVENING_CHECK_TIME = time(16, 20)
MORNING_CHECK_LEAD_MINUTES = 10
# How much daily history a close-check fetches: enough for SMA_DAYS_MAX sessions plus room for the first-run
# walk-back. (A 6mo fetch, ~126 sessions, silently froze any average over ~128 days -- final review 2026-11-21.)
HISTORY_PERIOD = "2y"


@dataclass(frozen=True)
class MarketPauseConfig:
    enabled: bool
    live_watch: bool
    vix_level: float
    sma_days: int


def market_pause_config(risk_cfg: dict | None) -> MarketPauseConfig:
    """The four settings from config['risk_management'], each falling back to MARKET_PAUSE_DEFAULTS."""
    cfg = {**MARKET_PAUSE_DEFAULTS, **{k: v for k, v in (risk_cfg or {}).items() if k in MARKET_PAUSE_DEFAULTS}}
    return MarketPauseConfig(
        enabled=bool(cfg["market_pause_enabled"]),
        live_watch=bool(cfg["market_pause_live_watch"]),
        vix_level=float(cfg["market_pause_vix_level"]),
        sma_days=int(cfg["market_pause_spy_sma_days"]),
    )


def coerce_vix_level(value) -> float:
    """Settings-save coercion: a finite number above 0, else ValueError (which aborts the whole save)."""
    level = float(value)
    if not math.isfinite(level) or level <= 0:
        raise ValueError(f"VIX level must be a number above 0, got {value!r}")
    return level


def coerce_sma_days(value) -> int:
    """Settings-save coercion: a whole number from SMA_DAYS_MIN to SMA_DAYS_MAX, else ValueError."""
    days = int(str(value).strip())
    if not SMA_DAYS_MIN <= days <= SMA_DAYS_MAX:
        raise ValueError(f"SPY average length must be {SMA_DAYS_MIN}-{SMA_DAYS_MAX} days, got {value!r}")
    return days


def finished_closes(bars: list[dict], today: str, after_evening_check: bool) -> list[tuple[str, float]]:
    """(date, close) for finished sessions only. yfinance's daily history includes today's unfinished bar
    during market hours, so today's bar counts only once the evening check time has passed."""
    out = []
    for bar in bars:
        day, close = bar.get("date"), bar.get("close")
        if not isinstance(day, str) or not _ok(close):
            continue
        if day < today or (after_evening_check and day == today):
            out.append((day, float(close)))
    return out


def close_check_slot(now_et: datetime, pre_open: time, done: set[str]) -> str | None:
    """Which daily close-check is due now: "evening" (from EVENING_CHECK_TIME), else "morning" (from
    MORNING_CHECK_LEAD_MINUTES before the pre-open batch), else None. Every day of the week, so weekend
    checks keep the staleness clock honest; a repeated check is harmless because a close counts once."""
    day = now_et.strftime("%Y-%m-%d")
    now_t = now_et.time()
    if now_t >= EVENING_CHECK_TIME:
        return "evening" if f"{day}:evening" not in done else None
    morning_from = (datetime.combine(now_et.date(), pre_open) - timedelta(minutes=MORNING_CHECK_LEAD_MINUTES)).time()
    if now_t >= morning_from and f"{day}:morning" not in done:
        return "morning"
    return None


def is_stale(state: MarketPauseState, now_et: datetime, max_age_hours: float = 26.0) -> bool:
    """True when the last successful close-check is older than max_age_hours. A state that was never
    checked is not "stale" -- that case is shown as "not checked yet"."""
    if not state.last_close_check_at:
        return False
    try:
        checked = datetime.fromisoformat(state.last_close_check_at)
        return (now_et - checked) > timedelta(hours=max_age_hours)
    except (ValueError, TypeError):
        return True


@dataclass(frozen=True)
class MarketPauseState:
    active: bool
    source: str                 # "close" | "live" | "none" (never evaluated)
    since: str | None           # "YYYY-MM-DD" (close-started) or ISO ET timestamp (live-started); None when off
    reason: str                 # plain sentence for the banner/log
    spy: float | None           # the values behind the current state
    spy_sma: float | None
    vix: float | None
    last_close_date: str | None       # newest close date applied
    last_close_check_at: str | None   # last successful close-check (the staleness clock)
    checked_at: str | None            # last successful check of either kind


NEVER_EVALUATED = MarketPauseState(
    active=False, source="none", since=None, reason="", spy=None, spy_sma=None, vix=None,
    last_close_date=None, last_close_check_at=None, checked_at=None)

_SOURCES = ("close", "live", "none")
_STR_FIELDS = ("since", "last_close_date", "last_close_check_at", "checked_at")
_NUM_FIELDS = ("spy", "spy_sma", "vix")


def _ok(value) -> bool:
    return is_usable_price(value) and value > 0


def _rule(spy: float, sma: float, vix: float, vix_level: float) -> bool:
    return spy < sma and vix > vix_level


def _reason(on: bool, spy: float, sma: float, vix: float, sma_days: int, vix_level: float) -> str:
    if on:
        return (f"SPY {spy:.2f} is below its {sma_days}-day average {sma:.2f} "
                f"and VIX is {vix:.2f} (above {vix_level:g})")
    parts = []
    if not spy < sma:
        parts.append(f"SPY {spy:.2f} is above its {sma_days}-day average {sma:.2f}")
    if not vix > vix_level:
        parts.append(f"VIX is {vix:.2f} (at or under {vix_level:g})")
    return " and ".join(parts)


def _as_map(series) -> dict | None:
    try:
        return {str(d): c for d, c in series}
    except (TypeError, ValueError):
        return None


def _window(spy_dates: list[str], spy_map: dict, index: int, sma_days: int) -> list[float] | None:
    """The sma_days closes ending at spy_dates[index], or None if too short or any is unusable."""
    if index + 1 < sma_days:
        return None
    window = [spy_map[d] for d in spy_dates[index + 1 - sma_days:index + 1]]
    return window if all(_ok(c) for c in window) else None


def evaluate_close(spy: list[tuple[str, float]], vix: list[tuple[str, float]], prev: MarketPauseState,
                   sma_days: int, vix_level: float, now_iso: str, force: bool = False) -> MarketPauseState:
    """Apply the newest close both series share. See the module docstring for the rules."""
    spy_map, vix_map = _as_map(spy), _as_map(vix)
    if spy_map is None or vix_map is None or not isinstance(sma_days, int) or sma_days < 1 or not _ok(vix_level):
        return prev
    spy_dates = sorted(spy_map)
    index_of = {d: i for i, d in enumerate(spy_dates)}
    shared = [d for d in spy_dates if d in vix_map]
    if not shared:
        return prev
    day = shared[-1]
    window = _window(spy_dates, spy_map, index_of[day], sma_days)
    vix_close = vix_map[day]
    if window is None or not _ok(vix_close):
        return prev

    if not force and prev.last_close_date is not None and day <= prev.last_close_date:
        return dataclasses.replace(prev, last_close_check_at=now_iso, checked_at=now_iso)

    spy_close, sma = window[-1], sum(window) / sma_days
    on = _rule(spy_close, sma, vix_close, vix_level)

    if (not force and not on and prev.active and prev.source == "live" and prev.since
            and day < prev.since[:10]):
        return dataclasses.replace(prev, last_close_date=day, last_close_check_at=now_iso, checked_at=now_iso)

    since = None
    if on:
        if prev.active:
            since = prev.since
        elif prev.source == "none":
            since = day
            for d in reversed(shared[:-1]):
                w = _window(spy_dates, spy_map, index_of[d], sma_days)
                if w is None or not _ok(vix_map[d]) or not _rule(w[-1], sum(w) / sma_days, vix_map[d], vix_level):
                    break
                since = d
        else:
            since = day

    return MarketPauseState(
        active=on, source="close", since=since,
        reason=_reason(on, spy_close, sma, vix_close, sma_days, vix_level),
        spy=spy_close, spy_sma=sma, vix=vix_close,
        last_close_date=day, last_close_check_at=now_iso, checked_at=now_iso)


def evaluate_live(spy_closes: list[float], spy_live: float, vix_live: float, prev: MarketPauseState,
                  sma_days: int, vix_level: float, now_iso: str) -> MarketPauseState:
    """Intraday check: the mean of the last sma_days-1 finished closes plus the live price. Only ever turns
    the pause ON; otherwise returns prev with checked_at updated."""
    if not isinstance(sma_days, int) or sma_days < 2 or not _ok(vix_level):
        return prev
    window = list(spy_closes)[-(sma_days - 1):]
    if len(window) < sma_days - 1 or not all(_ok(c) for c in window) or not _ok(spy_live) or not _ok(vix_live):
        return prev
    if prev.active:
        return dataclasses.replace(prev, checked_at=now_iso)
    sma = (sum(window) + spy_live) / sma_days
    if not _rule(spy_live, sma, vix_live, vix_level):
        return dataclasses.replace(prev, checked_at=now_iso)
    return MarketPauseState(
        active=True, source="live", since=now_iso,
        reason=_reason(True, spy_live, sma, vix_live, sma_days, vix_level),
        spy=spy_live, spy_sma=sma, vix=vix_live,
        last_close_date=prev.last_close_date, last_close_check_at=prev.last_close_check_at, checked_at=now_iso)


def is_usable_quote(value) -> bool:
    """A live price the rule can use: a real, finite number above 0. A quote source that answers with 0 or
    NaN (e.g. a fallback feed that doesn't carry ^VIX) is junk, and the caller treats it as a failed fetch."""
    return _ok(value)


def rejudge(spy: list[tuple[str, float]], vix: list[tuple[str, float]], prev: MarketPauseState,
            sma_days: int, vix_level: float, now_iso: str,
            live: tuple[list[float], float, float] | None = None) -> MarketPauseState:
    """A Settings change to the rule: re-judge the newest close under the new values (force=True), then --
    during market hours with the live watch on -- fold in a fresh live check, live=(finished closes before
    today, live SPY, live VIX). Without that, a change that still flags today's live market would briefly
    lift a same-day live pause (one OFF/ON notice pair and up to a minute in which a buy could go through)."""
    state = evaluate_close(spy, vix, prev, sma_days, vix_level, now_iso, force=True)
    if live is not None:
        state = evaluate_live(live[0], live[1], live[2], state, sma_days, vix_level, now_iso)
    return state


@dataclass(frozen=True)
class MarketReadings:
    """The numbers in the dashboard market graph's header and live point (2026-11-21, owner request: "can the right vix be put
    on the ticker?" -- TradingView's free strip refuses the real VIX index). The real VIX and SPY against its
    N-day average, from the same Yahoo fetches the pause decides on. Display only: the rule never reads it.
    Spec: docs/superpowers/specs/2026-11-21-market-readings-box-design.md."""
    kind: str                      # "close" | "live"
    as_of: str                     # close: "YYYY-MM-DD"; live: ISO ET timestamp
    vix: float
    vix_prev_close: float | None   # the VIX close before as_of's session (for the change), None if unknown
    spy: float
    spy_sma: float
    sma_days: int


def close_readings(spy: list[tuple[str, float]], vix: list[tuple[str, float]],
                   sma_days: int) -> MarketReadings | None:
    """The newest close both series share (the same date evaluate_close judges), or None on too little or bad
    data. The SPY average is the mean of the sma_days closes ending at that date."""
    spy_map, vix_map = _as_map(spy), _as_map(vix)
    if spy_map is None or vix_map is None or not isinstance(sma_days, int) or sma_days < 1:
        return None
    spy_dates = sorted(spy_map)
    shared = [d for d in spy_dates if d in vix_map]
    if not shared:
        return None
    day = shared[-1]
    window = _window(spy_dates, spy_map, spy_dates.index(day), sma_days)
    vix_close = vix_map[day]
    if window is None or not _ok(vix_close):
        return None
    vix_dates = sorted(vix_map)
    at = vix_dates.index(day)
    prev = vix_map[vix_dates[at - 1]] if at > 0 else None
    return MarketReadings(kind="close", as_of=day, vix=float(vix_close),
                          vix_prev_close=float(prev) if _ok(prev) else None,
                          spy=float(window[-1]), spy_sma=sum(window) / sma_days, sma_days=sma_days)


def live_readings(spy_closes: list[float], spy_live: float, vix_live: float, vix_prev_close: float | None,
                  sma_days: int, now_iso: str) -> MarketReadings | None:
    """Live SPY/VIX with the live average evaluate_live judges: the last sma_days-1 finished closes plus the
    live price. None on too little or bad data."""
    if not isinstance(sma_days, int) or sma_days < 2:
        return None
    window = list(spy_closes)[-(sma_days - 1):]
    if len(window) < sma_days - 1 or not all(_ok(c) for c in window) or not _ok(spy_live) or not _ok(vix_live):
        return None
    return MarketReadings(kind="live", as_of=now_iso, vix=float(vix_live),
                          vix_prev_close=float(vix_prev_close) if _ok(vix_prev_close) else None,
                          spy=float(spy_live), spy_sma=(sum(window) + spy_live) / sma_days, sma_days=sma_days)


def current_readings(close: MarketReadings | None, live: MarketReadings | None) -> MarketReadings | None:
    """What the box shows: a live reading from a newer session than the newest close, else the close. So the
    16:20 close replaces the day's live numbers, and a Settings re-judge (which recomputes yesterday's close
    during market hours) never knocks today's live numbers back to yesterday."""
    if close is None or live is None:
        return close or live
    return live if live.as_of[:10] > close.as_of[:10] else close


def readings_to_dict(r: MarketReadings | None, vix_level: float) -> dict | None:
    return None if r is None else {**dataclasses.asdict(r), "vix_level": vix_level}


CHART_DAYS = 252   # about a year of sessions: the dashboard graph's longest range (1Y)


def chart_series(spy: list[tuple[str, float]], vix: list[tuple[str, float]], sma_days: int, vix_level: float,
                 days: int = CHART_DAYS) -> list[dict]:
    """The dashboard market graph's daily points (2026-11-21, owner request: "a graph instead with both on it"),
    oldest first: {date, spy, sma, vix, paused} for the last `days` dates both series share -- the same date
    matching and N-close window evaluate_close uses, and `paused` is the rule at that close (strict, like
    _rule). Dates without a full window or a usable VIX close are skipped; a junk vix_level shades nothing.
    Spec: docs/superpowers/specs/2026-11-21-market-chart-panel-design.md."""
    spy_map, vix_map = _as_map(spy), _as_map(vix)
    if spy_map is None or vix_map is None or not isinstance(sma_days, int) or sma_days < 1:
        return []
    level_ok = _ok(vix_level)
    spy_dates = sorted(spy_map)
    points = []
    for index, day in enumerate(spy_dates):
        if day not in vix_map:
            continue
        window = _window(spy_dates, spy_map, index, sma_days)
        vix_close = vix_map[day]
        if window is None or not _ok(vix_close):
            continue
        spy_close, sma = float(window[-1]), sum(window) / sma_days
        points.append({"date": day, "spy": spy_close, "sma": sma, "vix": float(vix_close),
                       "paused": level_ok and _rule(spy_close, sma, vix_close, vix_level)})
    return points[-days:] if days > 0 else []


def to_dict(state: MarketPauseState) -> dict:
    return dataclasses.asdict(state)


def from_dict(d) -> MarketPauseState:
    """The saved state, or NEVER_EVALUATED for anything missing or malformed."""
    if not isinstance(d, dict):
        return NEVER_EVALUATED
    try:
        if (not isinstance(d["active"], bool) or d["source"] not in _SOURCES
                or not isinstance(d["reason"], str)):
            return NEVER_EVALUATED
        values = {"active": d["active"], "source": d["source"], "reason": d["reason"]}
        for key in _STR_FIELDS:
            if d[key] is not None and not isinstance(d[key], str):
                return NEVER_EVALUATED
            values[key] = d[key]
        for key in _NUM_FIELDS:
            if d[key] is not None and not is_usable_price(d[key]):
                return NEVER_EVALUATED
            values[key] = None if d[key] is None else float(d[key])
    except (KeyError, TypeError):
        return NEVER_EVALUATED
    return MarketPauseState(**values)


def _clock(iso: str | None) -> str:
    """"Nov 20 4:20 PM" from an ISO ET timestamp (portable: no %-I, which Windows lacks)."""
    try:
        dt = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return "unknown"
    return f"{dt.strftime('%b %d')} {dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def status_text(state: MarketPauseState, stale: bool) -> str:
    """The read-only status line on Settings > Market Pause."""
    if state.source == "none":
        return "Not checked yet"
    if state.active:
        since = state.since or ""
        if "T" in since:
            try:
                dt = datetime.fromisoformat(since)
                when = f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'} on {dt.strftime('%m-%d')} (live)"
            except ValueError:
                when = since
        else:
            when = f"the {since[5:]} close"
        head = f"Short mode since {when}: {state.reason}"
    else:
        head = f"Long mode: {state.reason}"
    text = f"{head} · last checked {_clock(state.checked_at)}"
    if stale:
        text += f" · market data is stale (last close check {_clock(state.last_close_check_at)})"
    return text
