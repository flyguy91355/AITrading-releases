"""Stock price and financial data fetching via yfinance with Finnhub fallback."""

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
import pandas as pd
import yfinance as yf

from src.analytics.composition_benchmark import is_usable_price

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    ticker: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    timestamp: datetime
    change_pct: float = 0.0


@dataclass
class Financials:
    ticker: str
    revenue: float = 0.0
    revenue_growth: float = 0.0
    eps: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    peg_ratio: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    free_cash_flow: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    roe: float = 0.0
    roic: float = 0.0
    dividend_yield: float = 0.0
    market_cap: float = 0.0
    sector: str = ""


@dataclass
class TechnicalIndicators:
    ticker: str
    sma_50: float = 0.0
    sma_200: float = 0.0
    rsi: float = 0.0
    avg_volume_30d: int = 0
    support_level: float = 0.0
    resistance_level: float = 0.0
    atr_pct: float = 0.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    obv_trend_pct: float = 0.0


@dataclass
class LongTermTrend:
    """Multi-year price context (2026-07-29, BEN discussion) -- get_technicals() above
    only ever looks at 1 year of history, so a stock that fell for years and has since
    rallied hard (e.g. BEN: ~$25 in 2021 -> ~$17.57 low in April 2025 -> ~$32 today) gets
    analyzed with no visibility into whether that's a genuine turnaround or a bounce back
    toward a prior structural decline -- the whole multi-year decline falls outside a
    1-year window. See format_long_term_trend_summary for how this becomes prompt text."""
    ticker: str
    years: int
    high: float = 0.0
    high_date: str = ""
    low: float = 0.0
    low_date: str = ""
    current_price: float = 0.0


def _compute_long_term_trend(
    ticker: str, years: int, closes: list[tuple[str, float]], current_price: float,
) -> LongTermTrend:
    """Pure: given (date_str, close) pairs spanning `years`, find the high/low and their
    dates. Separated from get_long_term_trend's yfinance fetch so this logic is directly
    unit-testable without a network call.

    NaN/Infinity closes are dropped before the max/min (GitHub #122): every comparison
    against NaN is False in Python, so max()/min() -- which work by pairwise comparison
    against a running candidate -- get permanently STUCK on a NaN if the NaN happens to
    be the first element scanned (here, the oldest close, since this list is
    chronological oldest-first). The result was a NaN high/low reaching
    format_long_term_trend_summary, whose own `high <= 0` guard is also False for NaN,
    embedding a literal "nan%" straight into the live ANALYSIS_PROMPT. Filtering the
    input fixes it regardless of where in the list the bad close sits."""
    closes = [p for p in closes if is_usable_price(p[1])]
    if not closes:
        return LongTermTrend(ticker=ticker, years=years, current_price=current_price)
    high_date, high = max(closes, key=lambda p: p[1])
    low_date, low = min(closes, key=lambda p: p[1])
    return LongTermTrend(
        ticker=ticker, years=years,
        high=round(high, 2), high_date=high_date,
        low=round(low, 2), low_date=low_date,
        current_price=current_price,
    )


@dataclass
class SmaCrossoverInfo:
    """Result of Stage 2 (crossing-date detection) of the SMA Trend-Confirmation Track
    (2026-08-24 design, ported from AIShortTrading). sma_50/sma_200 are the CURRENT
    point values (same numbers TechnicalIndicators already carries);
    crossover_date/days_since_crossover describe WHEN the relationship most recently
    flipped into its current state, which TechnicalIndicators has no way to answer
    since it only ever stores a single current point, not a series."""
    ticker: str
    sma_50: float = 0.0
    sma_200: float = 0.0
    crossover_date: str | None = None
    days_since_crossover: int | None = None


def find_sma_crossover_date(
    closes: list[tuple[str, float]], direction: str = "below",
) -> str | None:
    """Pure: given oldest-first (date_str, close) pairs, computes the SMA50/SMA200
    series across the full window and walks backward from the most recent point to
    find the most recent date the relationship flipped INTO the qualifying state.
    direction="above": SMA50 crossed above SMA200 (a golden cross -- this program's
    long-side trend-confirmation signal); direction="below" is the death-cross
    mirror (AIShortTrading's own short-side equivalent).

    Needs at least 201 closes to compute even a single SMA200 point -- returns None
    (not enough data) if fewer are supplied. Also returns None if the relationship
    isn't currently in the qualifying state at all, or if it's been in the qualifying
    state for the ENTIRE computable window (no flip found -- a sustained, longer-
    standing relationship rather than a fresh cross). The caller
    (_build_sma_trend_section, via its own "no crossover found" branch) already
    treats None as "held this relationship the whole time," never as an error.

    NaN/Infinity closes are dropped first (GitHub #104): the SMA series below is built
    with raw Python sum()/N, so a single bad close poisons EVERY 50- or 200-bar window
    that contains it, and since NaN comparisons are always False the _qualifies() check
    then silently returns False for those points -- which can hide a real crossover, or
    misreport None ("held this relationship the whole time") when the window merely had
    one bad data point. The sibling get_technicals() never had this problem because it
    uses pandas' .mean(), which is skipna=True by default."""
    closes = [p for p in closes if is_usable_price(p[1])]
    if len(closes) < 201:
        return None

    n = len(closes)
    sma50_series: list[float | None] = [None] * n
    sma200_series: list[float | None] = [None] * n
    for i in range(n):
        if i >= 49:
            sma50_series[i] = sum(c for _, c in closes[i - 49:i + 1]) / 50
        if i >= 199:
            sma200_series[i] = sum(c for _, c in closes[i - 199:i + 1]) / 200

    def _qualifies(s50, s200) -> bool:
        return s50 < s200 if direction == "below" else s50 > s200

    last_s50, last_s200 = sma50_series[-1], sma200_series[-1]
    if last_s50 is None or last_s200 is None or not _qualifies(last_s50, last_s200):
        return None

    for i in range(n - 1, 199, -1):
        s50, s200 = sma50_series[i], sma200_series[i]
        prev_s50, prev_s200 = sma50_series[i - 1], sma200_series[i - 1]
        if prev_s50 is None or prev_s200 is None:
            break
        if _qualifies(s50, s200) and not _qualifies(prev_s50, prev_s200):
            return closes[i][0]

    return None


def format_long_term_trend_summary(trend: LongTermTrend) -> str:
    """Pure formatter -- builds the ANALYSIS_PROMPT section text from a LongTermTrend.
    Returns "" when there's no usable data (an empty/failed fetch upstream), so the
    prompt section can be omitted cleanly rather than showing garbage.

    The `<= 0` checks alone can never catch a NaN (every NaN comparison is False), so
    they're paired with the shared is_usable_price guard (GitHub #122) -- defense in
    depth behind _compute_long_term_trend's own input filter, since a LongTermTrend
    can also be constructed directly by a caller."""
    if not is_usable_price(trend.high) or not is_usable_price(trend.low):
        return ""
    if not is_usable_price(trend.current_price):
        return ""
    if trend.high <= 0 or trend.low <= 0:
        return ""
    pct_off_high = (trend.current_price - trend.high) / trend.high * 100
    pct_off_low = (trend.current_price - trend.low) / trend.low * 100
    return (
        f"{trend.years}-Year Range: Low ${trend.low:.2f} ({trend.low_date}) | "
        f"High ${trend.high:.2f} ({trend.high_date}) | Current ${trend.current_price:.2f} "
        f"is {pct_off_high:+.1f}% vs the {trend.years}yr high and "
        f"{pct_off_low:+.1f}% vs the {trend.years}yr low."
    )


def format_technical_summary(price: float, technicals: "TechnicalIndicators") -> str:
    """Pure formatter -- the single shared source for the ANALYSIS_PROMPT/batch-scan
    "TECHNICAL CONTEXT" section (2026-08-29), extracted from what used to be 4x
    independently-duplicated inline string-building in engine.py so MACD/ADX/OBV
    framing can't drift between call sites.

    The added MACD/ADX/OBV line and the confluence instruction are deliberately
    NEUTRAL and METHODOLOGY-ONLY -- they teach Claude HOW to weigh multiple
    technical signals together (per explicit owner direction: "they must be set
    up to complment each other"), but never state or imply WHAT the current
    numbers already mean. This is a direct lesson from this project's own
    market-context-section incident (2026-08-04 to 2026-08-27): a single
    injected interpretive phrase ("a rising tide can squeeze a short position")
    silently biased 162 real above-gate verdicts to a 0% accept rate before it
    was found and removed. See tests/test_technical_summary_format.py's own
    regression-lock test, which asserts no bullish/bearish/favorable-flavored
    word can ever appear in this function's output, for any input."""
    obv_sign = "+" if technicals.obv_trend_pct >= 0 else ""
    return (
        f"Price: ${price:.2f} | SMA50: ${technicals.sma_50:.2f} | "
        f"SMA200: ${technicals.sma_200:.2f} | RSI: {technicals.rsi:.1f} | "
        f"Support: ${technicals.support_level:.2f} | "
        f"Resistance: ${technicals.resistance_level:.2f} | "
        f"Avg Volume 30d: {technicals.avg_volume_30d:,}\n"
        f"MACD: line {technicals.macd_line:.2f}, signal {technicals.macd_signal:.2f}, "
        f"histogram {technicals.macd_histogram:.2f} (histogram = line minus signal) | "
        f"ADX: {technicals.adx:.1f} (trend strength, 0-100 scale; higher = stronger "
        f"trend, regardless of direction) | "
        f"20-Day Volume Trend: {obv_sign}{technicals.obv_trend_pct:.1f}% "
        f"(positive = net buying volume, negative = net selling volume)\n"
        f"Read MACD, ADX, and the volume trend together, not independently: weigh a "
        f"reading more heavily when multiple of these point the same direction, and "
        f"treat a single indicator moving alone -- or any signal during a weak trend "
        f"(low ADX) -- with more caution. If they conflict, treat the technical "
        f"picture as genuinely mixed rather than picking whichever one supports a "
        f"preferred conclusion."
    )


def _usable_closes(hist) -> list[tuple[str, float]]:
    """Builds the oldest-first (date_str, close) pair list both get_long_term_trend and
    get_sma_crossover_info feed into their own pure computation, dropping any bar whose
    close isn't a real finite number (GitHub #104/#122).

    Single shared builder rather than a filter re-written at each call site, matching
    the consolidation this codebase already applied to its other NaN guard (GitHub #85,
    is_usable_price) -- the recurring failure mode here has been the SAME guard being
    re-implemented inconsistently, not the guard itself being wrong."""
    closes: list[tuple[str, float]] = []
    for idx, row in hist.iterrows():
        close = float(row["Close"])
        if is_usable_price(close):
            closes.append((idx.strftime("%Y-%m-%d"), close))
    return closes


class MarketDataFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        self._market_change_cache: tuple[float, float] | None = None  # (value, fetched_at_ts)

    async def get_market_change_pct(
        self, index_ticker: str = "SPY", cache_seconds: float = 300.0,
    ) -> float | None:
        """Today's % change for a broad-market proxy (2026-08-04, "does the AI know if
        the market's up or down" discussion) -- SPY (an S&P 500 tracking ETF, not the raw
        ^GSPC index symbol) for the same reliability reasons this codebase already
        prefers real, liquid tickers elsewhere (see the composition-benchmark package's
        own IJR-over-SLY note). Reuses this same NaN-guarded get_quote() rather than a
        separate fetch path, so this benefits from that fix automatically.

        Cached for cache_seconds (default 5 min) on this instance -- callers building
        many prompts in one batch run (submit_analysis_batch can build 1,500+ in a single
        pass) all share one real fetch instead of one each, while a call made minutes
        later during the same trading day still gets a fresh value. Returns None on any
        fetch failure (never fabricates a market condition) so callers can omit the
        context line entirely, same graceful-omission pattern used throughout this
        codebase's other optional prompt sections."""
        now = time.monotonic()
        if self._market_change_cache is not None:
            cached_value, fetched_at = self._market_change_cache
            if now - fetched_at < cache_seconds:
                return cached_value
        try:
            quote = await self.get_quote(index_ticker)
            value = quote.change_pct
        except Exception as e:
            logger.warning("Market change fetch failed for %s: %s", index_ticker, e)
            return None
        self._market_change_cache = (value, now)
        return value

    async def get_quote(self, ticker: str) -> StockQuote:
        try:
            def _fetch():
                stock = yf.Ticker(ticker)
                return stock.fast_info, stock.history(period="1d")
            info, hist = await asyncio.to_thread(_fetch)

            if hist.empty:
                return await self._finnhub_quote(ticker)

            row = hist.iloc[-1]
            # NaN guard (2026-08-03) -- yfinance can return a real float('nan') for a
            # not-yet-published close (confirmed live for the composition-weighted
            # benchmark feature, see CLAUDE.md); `float(nan)` doesn't raise, so this would
            # otherwise propagate silently into a StockQuote's price and, since NaN
            # comparisons are always False in Python, could bypass downstream risk-gate
            # checks (e.g. SignalGenerator's risk <= 0 / rr < min_rr_ratio rejections)
            # rather than being caught by them.
            if math.isnan(row["Close"]):
                return await self._finnhub_quote(ticker)
            prev_close = info.previous_close if hasattr(info, "previous_close") else row["Close"]
            # A bare truthiness check can never catch NaN -- bool(float('nan')) is True
            # (GitHub #112), so a NaN previous_close (fast_info is not NaN-checked
            # anywhere upstream) used to run the arithmetic and produce a NaN
            # change_pct. That value flows into get_market_change_pct(), gets cached,
            # and is handed straight to the live "is the market up or down today"
            # line in every real buy/sell prompt -- so it must degrade to 0.0 here
            # rather than silently reaching a trading decision as NaN.
            prev_close_usable = is_usable_price(prev_close) and prev_close != 0
            change_pct = (
                ((row["Close"] - prev_close) / prev_close * 100) if prev_close_usable else 0.0
            )

            return StockQuote(
                ticker=ticker,
                price=float(row["Close"]),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"]),
                timestamp=datetime.now(),
                change_pct=round(change_pct, 2),
            )
        except Exception as e:
            logger.warning("yfinance quote failed for %s: %s, trying Finnhub", ticker, e)
            return await self._finnhub_quote(ticker)

    async def _finnhub_quote(self, ticker: str) -> StockQuote:
        if not self.finnhub_key:
            raise RuntimeError(f"No quote data available for {ticker} — yfinance failed and no Finnhub key set")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ticker, "token": self.finnhub_key},
            )
            resp.raise_for_status()
            data = resp.json()

        prev = data.get("pc", 0)
        current = data.get("c", 0)
        change_pct = ((current - prev) / prev * 100) if prev else 0.0

        return StockQuote(
            ticker=ticker,
            price=float(current),
            open=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            volume=0,
            timestamp=datetime.now(),
            change_pct=round(change_pct, 2),
        )

    async def get_financials(self, ticker: str) -> Financials:
        info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info or {})

        def _g(key: str, default=0.0):
            # `v is not None` is True for float('nan') (GitHub #109), so a NaN field
            # from yfinance's .info dict -- routine for a sparse/missing fundamental --
            # used to become a real NaN on the Financials object instead of the
            # intended safe default, propagating into fundamental scoring and every
            # downstream AI prompt. Coerce first (so a numeric string still parses, as
            # it always did), then apply the shared finite-value guard; a genuinely
            # unparseable value now falls back to the default too, rather than raising
            # ValueError out of this whole method as it previously would have.
            v = info.get(key)
            if v is None:
                return default
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return v if is_usable_price(v) else default

        return Financials(
            ticker=ticker,
            sector=info.get("sector", "") or "",
            revenue=_g("totalRevenue"),
            revenue_growth=_g("revenueGrowth"),
            eps=_g("trailingEps"),
            pe_ratio=_g("trailingPE"),
            pb_ratio=_g("priceToBook"),
            ps_ratio=_g("priceToSalesTrailing12Months"),
            peg_ratio=_g("pegRatio"),
            gross_margin=_g("grossMargins"),
            operating_margin=_g("operatingMargins"),
            net_margin=_g("profitMargins"),
            free_cash_flow=_g("freeCashflow"),
            debt_to_equity=_g("debtToEquity"),
            current_ratio=_g("currentRatio"),
            quick_ratio=_g("quickRatio"),
            roe=_g("returnOnEquity"),
            roic=_g("returnOnInvestmentCapital") if info.get("returnOnInvestmentCapital") is not None else _g("returnOnAssets"),
            dividend_yield=_g("dividendYield"),
            market_cap=_g("marketCap"),
        )

    async def get_historical(
        self, ticker: str, period: str = "1y", interval: str = "1d", *, hist=None,
    ) -> list[dict]:
        """interval (added 2026-07-18) lets a caller ask for intraday bars (e.g. "15m", "1h")
        instead of just daily closes — used by On Deck's price-history backfill to get many
        more real data points than one-per-day. "timestamp" (a real Unix epoch, taken directly
        from the tz-aware pandas index) was added alongside "date" for this — a bare
        YYYY-MM-DD string collapses every intraday bar on the same day to an identical value,
        losing the whole point of finer granularity.

        hist (added 2026-08-18) lets a caller that already fetched this exact
        ticker/period/interval pass the DataFrame straight through instead of this method
        fetching it again — /api/stock-chart used to call this and get_technicals()
        concurrently for the identical yfinance history, doubling real network calls per
        chart open. Optional and additive; omitting it fetches fresh exactly as before."""
        if hist is None:
            hist = await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period=period, interval=interval))
        results = []
        for date, row in hist.iterrows():
            # yfinance occasionally reports NaN Volume on a thin-volume/incomplete bar
            # (int(nan) raises ValueError) -- the real OHLC price data for that bar is
            # still good, so report volume=0 rather than losing the whole bar (or, before
            # this fix, crashing every bar in this call).
            volume = row["Volume"]
            # Price fields get a stricter guard than Volume (GitHub #121): there is no
            # safe substitute value for a bad price the way 0 works for volume, so a bar
            # with any non-finite OHLC is DROPPED rather than emitted with a raw NaN
            # embedded in it. This is the same thing /api/stock-chart already does to
            # this function's output after the fact (2026-08-18 fix) -- doing it here
            # means every other/future caller (chart data, the composition-benchmark
            # feature, ...) gets that protection too, instead of only the one endpoint
            # that happened to be crashed into fixing it.
            ohlc = [float(row[k]) for k in ("Open", "High", "Low", "Close")]
            if not all(is_usable_price(v) for v in ohlc):
                continue
            open_, high, low, close = ohlc
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": date.timestamp(),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 0 if (isinstance(volume, float) and math.isnan(volume)) else int(volume),
            })
        return results

    async def get_recent_closes(self, ticker: str, minutes: int = 10) -> list[float]:
        """1-minute intraday closes for the last `minutes` minutes of today's session — used
        for a backward-looking momentum check before buying (is the stock still actively
        falling right now?). yfinance only supports 1m-interval bars for period<=7d, so this
        always requests period="1d". Returns [] on any error or if intraday data isn't
        available yet (e.g. first minute after open)."""
        try:
            hist = await asyncio.to_thread(
                lambda: yf.Ticker(ticker).history(period="1d", interval="1m"))
        except Exception:
            return []
        if hist.empty:
            return []
        return hist["Close"].tail(minutes).tolist()

    async def get_technicals(self, ticker: str, *, hist=None) -> TechnicalIndicators:
        """hist (added 2026-08-18): see get_historical()'s matching parameter -- lets a
        caller that already fetched the same 1y/1d history reuse it instead of this
        method independently re-fetching from yfinance."""
        if hist is None:
            hist = await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="1y"))

        if hist.empty:
            return TechnicalIndicators(ticker=ticker)

        closes = hist["Close"]
        volumes = hist["Volume"]

        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else float(closes.mean())

        rsi = self._compute_rsi(closes)

        avg_volume_30d = int(volumes.tail(30).mean()) if len(volumes) >= 30 else int(volumes.mean())

        recent = closes.tail(20)
        support_level = float(recent.min())
        resistance_level = float(recent.max())

        macd_line, macd_signal, macd_histogram = self._compute_macd(closes)

        return TechnicalIndicators(
            ticker=ticker,
            sma_50=round(sma_50, 2),
            sma_200=round(sma_200, 2),
            rsi=round(rsi, 2),
            avg_volume_30d=avg_volume_30d,
            support_level=round(support_level, 2),
            resistance_level=round(resistance_level, 2),
            atr_pct=round(self._compute_atr_pct(hist), 4),
            macd_line=round(macd_line, 4),
            macd_signal=round(macd_signal, 4),
            macd_histogram=round(macd_histogram, 4),
            adx=round(self._compute_adx(hist), 2),
            obv_trend_pct=round(self._compute_obv_trend_pct(closes, volumes), 2),
        )

    async def get_long_term_trend(
        self, ticker: str, current_price: float, years: int = 5,
    ) -> LongTermTrend:
        """Multi-year high/low context, deliberately separate from get_technicals' 1-year
        window (2026-07-29, BEN discussion) -- see LongTermTrend's docstring. Fails open
        (returns a zeroed LongTermTrend, same as an empty-history TechnicalIndicators
        above) on any fetch error, matching this file's existing convention of never
        letting a market-data lookup crash the caller's analysis."""
        try:
            hist = await asyncio.to_thread(
                lambda: yf.Ticker(ticker).history(period=f"{years}y"))
        except Exception as e:
            logger.warning("Long-term trend fetch failed for %s: %s", ticker, e)
            return LongTermTrend(ticker=ticker, years=years, current_price=current_price)

        if hist.empty:
            return LongTermTrend(ticker=ticker, years=years, current_price=current_price)

        return _compute_long_term_trend(ticker, years, _usable_closes(hist), current_price)

    async def get_sma_crossover_info(
        self, ticker: str, direction: str = "above", lookback_days: int = 252, *, hist=None,
    ) -> SmaCrossoverInfo:
        """Stage 2 of the SMA Trend-Confirmation Track (2026-08-24 design, ported from
        AIShortTrading) -- fetches 2 years of daily closes (enough to compute a SMA200
        series across a `lookback_days` search window, since SMA200 itself needs 200
        prior days just for its first point) and finds the most recent date the
        SMA50/SMA200 relationship flipped into the qualifying state (direction="above":
        a golden cross, this program's long-side signal). Fails open (returns a zeroed
        SmaCrossoverInfo) on any fetch error or empty/too-short history, same convention
        as get_long_term_trend/get_technicals above.

        hist= (optional) mirrors get_technicals'/get_historical's own shared-history
        kwarg -- lets a caller that already fetched this ticker's history pass it in
        directly instead of triggering a second real yfinance call, and lets tests
        supply a synthetic DataFrame with no network mocking required."""
        if hist is None:
            try:
                hist = await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="2y"))
            except Exception as e:
                logger.warning("SMA crossover fetch failed for %s: %s", ticker, e)
                return SmaCrossoverInfo(ticker=ticker)

        if hist.empty:
            return SmaCrossoverInfo(ticker=ticker)

        # Length is checked AFTER dropping non-finite closes (GitHub #104) -- the raw
        # sum()/N below has no skipna behaviour, so a single bad close would otherwise
        # poison both current SMA values as well as the crossover series.
        closes = _usable_closes(hist)
        if len(closes) < 201:
            return SmaCrossoverInfo(ticker=ticker)

        sma_50 = round(sum(c for _, c in closes[-50:]) / 50, 2)
        sma_200 = round(sum(c for _, c in closes[-200:]) / 200, 2)

        search_window = closes[-(200 + lookback_days):]
        crossover_date = find_sma_crossover_date(search_window, direction=direction)
        days_since_crossover = None
        if crossover_date:
            crossover_idx = next(i for i, (d, _) in enumerate(closes) if d == crossover_date)
            days_since_crossover = len(closes) - 1 - crossover_idx

        return SmaCrossoverInfo(
            ticker=ticker, sma_50=sma_50, sma_200=sma_200,
            crossover_date=crossover_date, days_since_crossover=days_since_crossover,
        )

    @staticmethod
    def _compute_rsi(closes, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0

        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = (-deltas).where(deltas < 0, 0.0)

        # Seed with SMA of first period bars, then apply Wilder's smoothing
        avg_gain = float(gains.iloc[:period].mean())
        avg_loss = float(losses.iloc[:period].mean())
        for gain, loss in zip(gains.iloc[period:], losses.iloc[period:]):
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _compute_atr_pct(hist, period: int = 14) -> float:
        """14-period Average True Range, as a percentage of the latest close --
        2026-08-28, volatility-bounded R/R design (see that design doc for the full
        rationale). True range per bar is max(high-low, |high-prev_close|,
        |low-prev_close|), matching the standard Wilder definition -- the
        prev-close terms matter on a gap day, where the bar's own high-low range
        understates how far price actually moved. Returns 0.0 (never raises) when
        there isn't enough history for even one 14-bar average, or the latest close
        is non-positive -- every consumer treats 0.0 as "cannot volatility-bound
        this one" and falls back to its existing flat behavior."""
        if len(hist) < period + 1:
            return 0.0
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(true_range.tail(period).mean())
        last_close = float(close.iloc[-1])
        # 2026-08-29: `last_close <= 0` alone never catches NaN -- NaN comparisons
        # are always False in Python, so a NaN latest close (confirmed live: a
        # real yfinance bar can come back with NaN High/Low/Close for an
        # incomplete/still-forming day) used to silently produce a NaN atr_pct
        # instead of failing safe to 0.0.
        if last_close <= 0 or math.isnan(last_close) or math.isnan(atr):
            return 0.0
        return atr / last_close * 100

    @staticmethod
    def _compute_macd(closes, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
        """Standard MACD: fast/slow EMA line, its own signal-line EMA, and their
        difference (histogram) -- 2026-08-29, complements SMA/RSI/ATR with a
        momentum-divergence signal none of those capture (see this project's
        CLAUDE.md "MACD/ADX/OBV" entry). Returns (0.0, 0.0, 0.0) rather than
        raising when there isn't enough history for the signal line's own EMA to
        have a real seed -- every consumer treats an all-zero triple as "no
        reading available" and omits the section, matching this file's existing
        atr_pct/rsi fail-open convention."""
        if len(closes) < slow + signal:
            return (0.0, 0.0, 0.0)
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1]))

    @staticmethod
    def _compute_adx(hist, period: int = 14) -> float:
        """Average Directional Index -- trend STRENGTH, not direction, 2026-08-29.
        Complements a raw SMA crossover: the same golden/death cross is far more
        reliable in a strong trend (high ADX) than a choppy one (low ADX). Uses a
        simple rolling-mean smoothing for +DM/-DM/TR and for the final DX average,
        matching this file's own existing simplification choice for
        _compute_atr_pct (a plain tail-mean of true range, not full recursive
        Wilder smoothing) -- consistent style over strict textbook fidelity.
        Needs roughly 2*period bars (period for the first smoothed +DM/-DM/TR
        point, period more for ADX's own averaging window); returns 0.0 rather
        than raising or NaN when there isn't enough history, or when a period has
        zero directional movement at all (a 0/0 DX, e.g. a completely flat
        high/low), matching this file's existing fail-open convention."""
        if len(hist) < period * 2:
            return 0.0
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        tr_smooth = true_range.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()

        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        di_sum = plus_di + minus_di
        dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0, 0.0)

        adx = dx.rolling(period).mean()
        result = adx.iloc[-1]
        return float(result) if not pd.isna(result) else 0.0

    @staticmethod
    def _compute_obv_trend_pct(closes, volumes, lookback: int = 20) -> float:
        """On-Balance Volume, expressed as net signed volume over the lookback
        window as a % of total volume traded in that window -- 2026-08-29,
        confirms whether a price move has real conviction behind it (a
        breakdown on high volume is far more trustworthy than one on thin
        volume). Deliberately NOT raw OBV's own % change: OBV's cumulative
        level is arbitrary and path-dependent (it can sit near zero or cross
        it), making a plain % change of the level itself unstable/meaningless.
        This framing is bounded [-100, 100] and comparable across tickers
        regardless of OBV's own cumulative history. Returns 0.0 (never
        raises) when there isn't enough history for a full window, or the
        window's total volume is zero."""
        if len(closes) < lookback + 1:
            return 0.0
        direction = closes.diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        signed_volume = (direction * volumes).tail(lookback)
        total_volume = volumes.tail(lookback).sum()
        if total_volume == 0:
            return 0.0
        return float(signed_volume.sum() / total_volume * 100)
