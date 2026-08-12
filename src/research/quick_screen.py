"""Fast pre-screen for universe candidates — yfinance only, no Claude API call.

Returns in ~2s per stock (up to ~2.7s for stocks that survive every free check
and reach the P/E lookup, which needs yfinance's slower full .info call).
Only passes stocks worth a full paid Claude analysis — a "best of the best"
liquidity/quality/valuation bar: genuine uptrend on both time frames (price
above both 50- and 200-day MA), a healthy (not extreme either direction) RSI
band, high liquidity, momentum that isn't actively weak, and a real valuation
sanity check. v1 (loose technical-only bar) passed ~67% of the universe but
only ~2.5% of survivors ever reached conviction 7+ in the full paid analysis.
v2 cut that to ~32%. v3-v5 tightened further (added a golden-cross requirement
and a required +2% momentum gain) to ~3%, but that combination was found live
to systematically select stocks the market had already recognized — high
recent momentum plus medium-term trend above long-term trend correlates with
efficient pricing, so R/R (which needs real undervaluation) kept failing even
with clean, low-noise analysis (0/46 candidates cleared R/R>=2.1 on two
consecutive live runs). v6 removed the golden-cross requirement and reverted
momentum to v2's "not actively weak" bar (allow up to 1% decline over 20 days)
while keeping the volume/trend/RSI/P/E filters, which are genuine quality
filters rather than "already-popular" filters. Deliberately still permissive
on judgment calls (missing P/E, e.g. a loss-making growth stock, is NOT
rejected — that nuance is left to the full Claude analysis rather than
guessed at here)."""

import logging
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

_MIN_AVG_VOLUME = 3_000_000  # v2: 500k -> v3: 1M -> v4: 2.5M -> v5: 3M — institutional-quality liquidity only
_RSI_HIGH = 65               # v2: 68 -> v3: 65 — healthy zone, not extended
_RSI_LOW = 35                # v2: 30 -> v3: 35 — healthy zone, not weak
_MOMENTUM_MIN_GAIN = 0.99    # v6 (2026-07-16): reverted v3's +2%-required momentum back to
                             # v2's "near-flat-or-up" (allow up to 1% decline over 20 days).
                             # v3-v5 + the golden-cross requirement below consistently
                             # selected stocks the market had already recognized (strong
                             # recent momentum + medium-term trend above long-term trend),
                             # which correlates with efficient pricing — thin margin of
                             # safety, so R/R kept failing even with clean, low-noise
                             # analysis (confirmed live: 0/46 candidates cleared R/R>=2.1
                             # on two consecutive fully-analyzed runs). Kept everything
                             # else (volume, dual-MA trend, RSI, P/E) — those are quality/
                             # liquidity/valuation filters, not "already-popular" filters.
_MAX_TRAILING_PE = 35        # v2: 60 -> v3: 35 — real valuation bar, not just "not
                             # absurd"; missing/negative P/E (e.g. a loss-making growth
                             # stock) is still NOT rejected — left to the full analysis


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's EMA: seed with SMA of first period bars, then smooth
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def quick_screen(ticker: str) -> tuple[bool, str]:
    """Return (passes, reason). Synchronous — run in executor from async code."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info

        avg_vol = getattr(info, "three_month_average_volume", 0) or 0
        if avg_vol < _MIN_AVG_VOLUME:
            return False, f"low volume ({avg_vol:,.0f} avg)"

        price = getattr(info, "last_price", None)
        if not price or price <= 0:
            return False, "no price data"

        ma50 = getattr(info, "fifty_day_average", None)
        if ma50 and price < ma50:
            return False, f"below 50-day trend (${price:.2f} < 50-MA ${ma50:.2f})"

        ma200 = getattr(info, "two_hundred_day_average", None)
        if ma200 and price < ma200:
            return False, f"below long-term trend (${price:.2f} < 200-MA ${ma200:.2f})"
        # Golden-cross (50-MA > 200-MA) requirement removed in v6 — see module docstring.

        hist = t.history(period="2mo", interval="1d")
        if hist.empty or len(hist) < 15:
            return False, "insufficient history"

        closes = hist["Close"].values
        rsi = _rsi(closes)

        if rsi > _RSI_HIGH:
            return False, f"overbought (RSI {rsi:.0f})"
        if rsi < _RSI_LOW:
            return False, f"oversold (RSI {rsi:.0f})"

        if len(closes) >= 20 and closes[-1] < closes[-20] * _MOMENTUM_MIN_GAIN:
            chg = (closes[-1] / closes[-20] - 1) * 100
            return False, f"insufficient momentum ({chg:+.1f}% vs 20d ago)"

        # P/E check last — needs yfinance's slower full .info call, so only pay that
        # cost for stocks that already survived every free/instant check above.
        try:
            pe = t.info.get("trailingPE")
        except Exception:
            pe = None
        if pe is not None and pe > _MAX_TRAILING_PE:
            return False, f"overvalued (P/E {pe:.0f})"

        return True, f"RSI {rsi:.0f} | vol {avg_vol / 1e6:.1f}M | momentum OK"

    except Exception as e:
        logger.debug("Quick screen error for %s: %s", ticker, e)
        return False, "data error"
