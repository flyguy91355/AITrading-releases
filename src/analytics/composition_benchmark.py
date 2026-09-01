"""Pure computation for the composition-weighted portfolio benchmark (2026-07-29).

Replaces the flat equal-weighted SPY/QQQ/DIA blend with a benchmark weighted
to match this portfolio's own sector + market-cap-tier composition. Every
function in this module is pure (no I/O, no DB, no network) so it's fully
unit-testable without the DashboardState import-time constraint that blocks
testing anything in web/app.py directly. See
docs/superpowers/specs/2026-07-29-composition-weighted-benchmark-design.md.
"""

import json
import logging
import math
from pathlib import Path
from typing import NamedTuple

from src.analytics.trade_log_signal import classify_trade_signal

logger = logging.getLogger(__name__)

SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


def sector_to_etf(sector: str | None) -> str | None:
    """Maps yfinance's own sector string to its SPDR Select Sector ETF.
    Returns None for an unknown/empty/missing sector -- the caller falls
    back to SPY-only weighting for that position rather than guessing."""
    if not sector:
        return None
    return SECTOR_ETF_MAP.get(sector)


def cap_tier_to_etf(ticker: str, sp500: set[str], sp400: set[str], sp600: set[str]) -> str:
    """Reverse-looks-up which cap tier a ticker belongs to and returns the
    matching broad index ETF. Defaults to SPY when the ticker isn't found in
    any of the three sets (rare, but not guaranteed impossible) or somehow
    appears in more than one (S&P 500 takes priority as the safest default)."""
    if ticker in sp500:
        return "SPY"
    if ticker in sp400:
        return "MDY"
    if ticker in sp600:
        return "IJR"
    return "SPY"


def compound_index(seed: float, daily_returns: list[float]) -> list[float]:
    """Chain-compounds a sequence of daily fractional returns onto a seed
    value. Returns a list starting with the seed itself, one entry longer
    than daily_returns (index 0 = seed, index i = value after applying
    daily_returns[i-1])."""
    values = [seed]
    for r in daily_returns:
        values.append(values[-1] * (1 + r))
    return values


def weighted_daily_return(
    holdings_value: dict[str, float],
    classifications: dict[str, tuple[str | None, str]],
    etf_daily_returns: dict[str, float],
) -> float:
    """Dollar-weighted average, across every held ticker, of that position's
    OWN blended benchmark return for the day. Each position's blended return
    is 50% its sector ETF's daily return + 50% its cap-tier ETF's daily
    return (or 100% cap-tier alone if sector_etf is None -- unclassifiable
    sector, see cap_tier_to_etf's SPY-default). classifications maps ticker
    -> (sector_etf_or_None, cap_tier_etf). etf_daily_returns maps ETF ticker
    -> that day's fractional return; only the ETFs actually needed for the
    held tickers must be present.

    A position whose own dollar value, or whose required ETF return(s), is
    not a real finite number (NaN/Infinity -- see is_usable_price) is dropped
    from BOTH the numerator and the denominator, so the remaining positions
    still produce a real weighted average instead of the whole day's figure
    silently collapsing to NaN (GitHub #106; the 2026-07-29 production
    incident documented on carry_forward_price is exactly this failure mode
    one layer up -- a single NaN close from yfinance poisoning every sum it
    touches). A genuinely MISSING ETF key still raises KeyError, unchanged --
    callers (web/app.py's get_performance_today, weighted_intraday_series
    below) deliberately depend on that to detect an incomplete bar and carry
    the previous value forward rather than silently reweighting."""
    weighted_sum = 0.0
    total_value = 0.0
    for ticker, value in holdings_value.items():
        sector_etf, cap_tier_etf = classifications[ticker]
        cap_tier_return = etf_daily_returns[cap_tier_etf]
        if sector_etf is None:
            inputs = (cap_tier_return,)
            position_return = cap_tier_return
        else:
            sector_return = etf_daily_returns[sector_etf]
            inputs = (sector_return, cap_tier_return)
            position_return = (sector_return + cap_tier_return) / 2
        if not is_usable_price(value) or not all(is_usable_price(r) for r in inputs):
            logger.warning(
                "weighted_daily_return: dropping %s from today's benchmark weighting -- "
                "unusable value (%r) or ETF return(s) (%r)", ticker, value, inputs,
            )
            continue
        weighted_sum += value * position_return
        total_value += value
    if total_value == 0:
        return 0.0
    return weighted_sum / total_value


class TradeEvent(NamedTuple):
    ticker: str
    is_buy: bool
    shares: float
    timestamp: str  # ISO 8601 string, sortable lexicographically as-is


def parse_trade_events(jsonl_dir: Path, since: str | None = None) -> list[TradeEvent]:
    """Reads every *.jsonl file in jsonl_dir (data/trade_history/'s real
    format: one file per day, one JSON object per line with ticker/signal/
    shares/timestamp) and returns all buy+sell events across every file,
    sorted chronologically. A signal is a buy if "BUY" appears in it
    (covers "BUY" and "STRONG BUY"), a sell if "SELL" appears in it (covers
    "SELL" and "STRONG SELL") -- anything else (e.g. "HOLD", which is never
    actually supposed to appear in an executed-trade log but is tolerated
    defensively) is skipped, not raised on.

    since (optional ISO date/datetime string, inclusive) drops any event
    with an earlier timestamp -- confirmed live (2026-07-29) that
    data/trade_history/*.jsonl on both local and Hetzner includes ~3 weeks
    of leftover pre-migration dev/test pollution (2026-06-25 through
    2026-07-10, unrealistic full-capital test trades), already filtered
    elsewhere in web/app.py via _LIVE_ACCOUNT_START = "2026-07-12". Callers
    reconstructing real trading history should always pass that same
    cutoff, or an unsold test trade from that period would incorrectly
    count as a real held position forever."""
    events: list[TradeEvent] = []
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Shared with src/tax/trade_log_reader.py (GitHub #143) -- only the
            # BUY/SELL convention is shared; field extraction stays separate.
            is_buy = classify_trade_signal(
                row.get("signal", ""), source="parse_trade_events",
                ticker=row.get("ticker"), file_name=path.name,
            )
            if is_buy is None:
                continue
            if since is not None and row["timestamp"] < since:
                continue
            events.append(TradeEvent(row["ticker"], is_buy, row["shares"], row["timestamp"]))
    events.sort(key=lambda e: e.timestamp)
    return events


def reconstruct_daily_holdings(
    events: list[TradeEvent], dates: list[str],
) -> dict[str, dict[str, float]]:
    """Replays events chronologically and snapshots cumulative shares held,
    per ticker, at the END of each requested date (dates must already be
    sorted ascending, e.g. straight from performance_history). A ticker
    that nets to ~0 shares (full sell) is dropped from that day's dict
    entirely rather than kept at a negligible float value."""
    result: dict[str, dict[str, float]] = {}
    holdings: dict[str, float] = {}
    event_idx = 0
    for d in dates:
        # ".999999", not ".999999"-less "T23:59:59" (GitHub #142): real
        # timestamps carry microseconds (datetime.now().isoformat()), and a
        # longer string sharing the same prefix sorts GREATER
        # lexicographically -- so "...T23:59:59.500000" <= "...T23:59:59" is
        # False, and a trade logged in the final second of a calendar day was
        # silently pushed into the next day's holdings snapshot.
        day_end = d + "T23:59:59.999999"
        while event_idx < len(events) and events[event_idx].timestamp <= day_end:
            e = events[event_idx]
            holdings[e.ticker] = holdings.get(e.ticker, 0.0) + (e.shares if e.is_buy else -e.shares)
            if abs(holdings[e.ticker]) < 1e-6:
                del holdings[e.ticker]
            event_idx += 1
        result[d] = dict(holdings)
    return result


def weighted_intraday_series(
    holdings_value: dict[str, float],
    classifications: dict[str, tuple[str | None, str]],
    etf_bars: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Given today's live holdings (dollar value at market open) and each
    ETF's intraday bars for today, returns [(timestamp, cumulative_pct_change)]
    normalized to the first bar. Reuses weighted_daily_return per-bar by
    treating each bar-over-first-bar move as that "day's" return. If
    holdings_value is empty, returns 0.0 at every timestamp (nothing held
    today -- benchmark line is flat, matching reconstruct_daily_holdings'
    convention for a no-position day)."""
    any_etf_bars = next(iter(etf_bars.values()), [])
    timestamps = [ts for ts, _ in any_etf_bars]
    if not holdings_value:
        return [(ts, 0.0) for ts in timestamps]

    # Every ETF weighted_daily_return will actually look up for these
    # holdings -- if any is missing at a given bar (one ETF's intraday fetch
    # can have fewer bars than another's), that bar carries forward the
    # last successfully-computed value instead of crashing (2026-07-29 live
    # incident: a real KeyError here took down get_performance_today with a
    # 500; this function has the identical shape and was only saved from
    # the same crash by its caller's broader try/except silently dropping
    # the whole live point instead of just the one affected bar).
    #
    # A bar whose own close is NaN/Infinity, or an ETF whose opening close is
    # unusable (or zero -- it's the divisor), is treated exactly like a
    # missing bar: that ETF is simply absent from this bar's returns dict, so
    # the required-ETFs subset check below fails and the previous value is
    # carried forward for that one timestamp (GitHub #106). Without this, a
    # single bad intraday close produced a NaN return that propagated into
    # last_pct and then poisoned every SUBSEQUENT timestamp too, since
    # last_pct is what the carry-forward path re-emits -- corrupting the live
    # "today" benchmark line for the rest of the trading day.
    required_etfs = {c[0] for c in classifications.values() if c[0]} | {
        c[1] for c in classifications.values()}
    first_close = {
        etf: bars[0][1] for etf, bars in etf_bars.items()
        if bars and is_usable_price(bars[0][1]) and bars[0][1] != 0
    }
    series = []
    last_pct = 0.0
    for i, ts in enumerate(timestamps):
        etf_returns_since_open = {
            etf: (bars[i][1] / first_close[etf] - 1)
            for etf, bars in etf_bars.items()
            if etf in first_close and i < len(bars) and is_usable_price(bars[i][1])
        }
        if required_etfs <= etf_returns_since_open.keys():
            blended = weighted_daily_return(
                holdings_value, classifications, etf_returns_since_open) * 100
            # Belt-and-braces: weighted_daily_return already drops unusable
            # inputs, so this can only fire on a genuinely unforeseen shape --
            # never let a non-finite value become the carried-forward value.
            if is_usable_price(blended):
                last_pct = blended
        series.append((ts, last_pct))
    return series


def is_usable_price(value) -> bool:
    """True only for a real, finite number -- rejects None, NaN, +/-Infinity, and
    any non-numeric value alike (extracted 2026-08-24, GitHub #85 -- single shared
    guard replacing 3 independently-drifted "is this price/share value usable"
    checks: this module's own carry_forward_price/seed_last_known_prices/
    has_real_close, all isnan-only; web/app.py's _live_holdings_value, also
    isnan-only with no Infinity guard; and web/app.py's _fin(), which already used
    isfinite -- the stricter, correct condition this shared helper standardizes
    on). math.isnan alone lets a float('inf')/float('-inf') straight through,
    which this codebase has no defense against without this guard -- confirmed
    live (2026-07-29) that a NaN price from yfinance can silently poison an
    entire day's composition-benchmark computation; an Infinity value would be
    just as capable of corrupting a downstream sum, undetected by 2 of the 3
    old guards."""
    if isinstance(value, bool):
        return False  # bool is a subclass of int -- never a real price/share value
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def carry_forward_price(
    ticker: str, date: str, closes_by_ticker: dict[str, dict[str, float]],
    last_known: dict[str, float],
) -> float | None:
    """Returns the real close price for ticker on date if present and valid,
    else falls back to the last known good price seen for that ticker (or
    None if there isn't one yet). Treats NaN exactly like a missing value --
    confirmed live (2026-07-29 production incident) that yfinance can return
    an actual float('nan') close (not just a missing date key) for the most
    recent day when the data provider hasn't finished publishing that day's
    official close yet, which a plain `is not None` check does not catch.
    Letting a NaN through silently poisons every downstream sum it touches
    (a single NaN ticker corrupts that whole day's total portfolio value and
    therefore every position's composition weight, not just its own)."""
    price = closes_by_ticker.get(ticker, {}).get(date)
    if is_usable_price(price):
        last_known[ticker] = price
        return price
    return last_known.get(ticker)


def seed_last_known_prices(
    closes_by_symbol: dict[str, dict[str, float]], before_date: str,
) -> dict[str, float]:
    """Primes a carry_forward_price `last_known` cache with the most recent
    valid (non-NaN) close strictly before before_date, for every symbol that
    has one. Needed for incremental backfill re-runs (processing only newly-
    added dates): carry_forward_price's cache is normally built up by
    processing prior dates within the SAME run, so a run that only
    reprocesses one new date has nothing to carry forward from even when
    the prior day's real close is sitting right there in already-fetched
    history -- confirmed live (2026-07-29) to silently drop every held
    ticker when that new date's own close also came back NaN."""
    seeded: dict[str, float] = {}
    for symbol, by_date in closes_by_symbol.items():
        candidates = sorted(d for d in by_date if d < before_date)
        for d in reversed(candidates):
            price = by_date[d]
            if is_usable_price(price):
                seeded[symbol] = price
                break
    return seeded


def has_real_close(symbol: str, date: str, closes_by_symbol: dict[str, dict[str, float]]) -> bool:
    """True only if closes_by_symbol has a real (present, non-NaN) close for
    symbol on this exact date -- distinct from carry_forward_price, which
    deliberately returns an OLDER price as a usable fallback. A day where any
    held ticker or ETF needed that fallback isn't real settled history yet
    (2026-07-29 finding) -- it must be marked provisional so a later backfill
    run knows to recompute it once the real data is published, rather than
    permanently freezing in a placeholder value."""
    return is_usable_price(closes_by_symbol.get(symbol, {}).get(date))
