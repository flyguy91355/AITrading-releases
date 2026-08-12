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
    held tickers must be present."""
    total_value = sum(holdings_value.values())
    if total_value == 0:
        return 0.0
    weighted_sum = 0.0
    for ticker, value in holdings_value.items():
        sector_etf, cap_tier_etf = classifications[ticker]
        cap_tier_return = etf_daily_returns[cap_tier_etf]
        if sector_etf is None:
            position_return = cap_tier_return
        else:
            sector_return = etf_daily_returns[sector_etf]
            position_return = (sector_return + cap_tier_return) / 2
        weighted_sum += value * position_return
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
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            signal = row.get("signal", "")
            if "BUY" in signal:
                is_buy = True
            elif "SELL" in signal:
                is_buy = False
            else:
                logger.warning(
                    "parse_trade_events: unrecognized signal %r for %s in %s -- skipping",
                    signal, row.get("ticker"), path.name,
                )
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
        day_end = d + "T23:59:59"
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
    required_etfs = {c[0] for c in classifications.values() if c[0]} | {
        c[1] for c in classifications.values()}
    first_close = {etf: bars[0][1] for etf, bars in etf_bars.items() if bars}
    series = []
    last_pct = 0.0
    for i, ts in enumerate(timestamps):
        etf_returns_since_open = {
            etf: (bars[i][1] / first_close[etf] - 1)
            for etf, bars in etf_bars.items()
            if i < len(bars)
        }
        if required_etfs <= etf_returns_since_open.keys():
            last_pct = weighted_daily_return(holdings_value, classifications, etf_returns_since_open) * 100
        series.append((ts, last_pct))
    return series


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
    if price is not None and not math.isnan(price):
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
            if price is not None and not math.isnan(price):
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
    price = closes_by_symbol.get(symbol, {}).get(date)
    return price is not None and not math.isnan(price)
