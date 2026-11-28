"""Short Mirror -- the Stress Dip-Buyer's rules turned upside down, traded only in SHORT MODE
(the market pause on). Owner, 2026-11-27: "turn the pause back on and use it to decide if were
in a long or short mode", "short would be the exact opposite of the long correct?",
"Stock-by-stock mirror", "Build it in AITrading". Design:
docs/superpowers/specs/2026-11-27-short-mirror-design.md.

  * Each evening, from COMPLETED daily bars only (the same snapshots as the dip-buyer): a stock
    qualifies when it closes BELOW its 200-day average, its 2-day RSI is ABOVE 90 (a sharp 2-3
    day bounce), it trades at least $20M/day (63 days), it closes at $5 or more, AND VIX closed
    at 20 or more. Ranked by highest RSI(2).
  * Next session, right after the open, only while the market pause is on: short up to 20
    slots (each ~1/20 of equity net of the cash reserve), whole shares (Alpaca has no
    fractional or notional shorts), easy-to-borrow only, after a Claude red-flag check.
  * Hard stop: a GTC BUY STOP at entry + 3 x ATR(14), clipped to 3%-15%.
  * Exit (exit_mode "trail", the default since 2026-11-27 -- owner: the short side mirrors the
    long, trailing stop included): the dip-buyer's stepped trail mirrored. It turns on once the
    price trades under the short, starts at the stock's own stop %, narrows (through -4.7% and
    -6.6%) to 0.5% over the lowest low at -8.4%, is lowered once per session from the completed
    daily lows since the short (never raised), and the resting buy stop is moved down to it.
    Cover at the close of the 20th session held if nothing else has. Backtest (exp16, short mode
    only): 0.2%/yr (2005-15) and -7.1%/yr (2016-26) vs 0.8% / -6.1% for exit_mode "average":
    the first close under the 5-day average (live price standing in for today's close), or the
    10th session.

Backtest (scratchpad exp16, 20 slots, dividends charged via adjusted prices, 0.5%/yr borrow):
0.8%/yr (2005-15) and -6.1%/yr (2016-26) on its own; long mode + this short mode 3.2%/yr and
1.0%/yr vs long mode alone 2.3% and 7.5%. The owner chose to build it anyway, switched on.

Pure (no I/O): web/app.py owns the loop, data, Claude call and orders."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from src.strategy import stress_dip as sdip
from src.strategy.stress_dip import TickerSnapshot, _as_bool, _finite

SHORT_MIRROR_DEFAULTS = {
    "enabled": True,                  # the owner's on/off switch ("i can turn it on or off")
    "slots": 20,
    "vix_min": 20.0,
    "rsi2_min": 90.0,                 # short when RSI(2) is ABOVE this
    "min_dollar_volume": 20_000_000.0,
    "min_price": 5.0,
    "stop_atr_mult": 3.0,
    "stop_min_pct": 3.0,
    "stop_max_pct": 15.0,
    "max_hold_sessions": 10,          # exit_mode "average"
    "exit_mode": "trail",             # "trail" (mirrored stepped trail) or "average" (5-day average)
    "trail_max_hold_sessions": 20,    # exit_mode "trail"
    "trail_final_pct": 0.5,           # the trail's width at -trail_t3_pct
    "trail_t1_pct": 4.7,
    "trail_t2_pct": 6.6,
    "trail_t3_pct": 8.4,
    "entry_delay_minutes": 1,
    "entry_window_minutes": 60,
    "exit_minutes_before_close": 10,
    "drawdown_halt_pct": 20.0,
    "max_per_sector": 3,
    "ex_dividend_block_days": 14,     # owner: "i dont want any dividened stocks" -- narrow form:
                                      # no short with an ex-dividend date inside the hold window
    "earnings_block_days": 2,
    # Pick order (2026-11-28): "rsi2" (highest first, the original rule), "volume_profile"
    # (furthest OVER the point of control in ATRs) or "move3" (biggest 3-day rise). The dip-buyer's
    # orders mirrored; only the order changes. The short side's order is untested (the research
    # ranked dips only) -- the owner: "we will backtest later for proper settings".
    "rank_by": "rsi2",
    # exit_mode "tighten" (2026-11-28): AIDayTrading's trail as it was built (for shorts): the buy
    # stop trails the lowest price since the short, its distance shrinking tighten_by_pct for
    # every tighten_every_pct the price falls, never closer than tighten_floor_pct. Live.
    "tighten_start_pct": 0.0,
    "tighten_every_pct": 0.1,
    "tighten_by_pct": 0.2,
    "tighten_floor_pct": 2.0,
    "tighten_min_move_pct": 0.2,
}


@dataclass(frozen=True)
class ShortMirrorConfig:
    enabled: bool
    slots: int
    vix_min: float
    rsi2_min: float
    min_dollar_volume: float
    min_price: float
    stop_atr_mult: float
    stop_min_pct: float
    stop_max_pct: float
    max_hold_sessions: int
    exit_mode: str
    trail_max_hold_sessions: int
    trail_final_pct: float
    trail_t1_pct: float
    trail_t2_pct: float
    trail_t3_pct: float
    entry_delay_minutes: int
    entry_window_minutes: int
    exit_minutes_before_close: int
    drawdown_halt_pct: float
    max_per_sector: int
    ex_dividend_block_days: int
    earnings_block_days: int
    rank_by: str
    tighten_start_pct: float
    tighten_every_pct: float
    tighten_by_pct: float
    tighten_floor_pct: float
    tighten_min_move_pct: float


def short_mirror_cfg(raw: dict | None) -> ShortMirrorConfig:
    """config['short_mirror'] merged over SHORT_MIRROR_DEFAULTS. A blank/None value keeps the
    default; a malformed value raises (a typo must not silently become a live number)."""
    merged = dict(SHORT_MIRROR_DEFAULTS)
    for k, v in (raw or {}).items():
        if k in merged and v is not None and v != "":
            merged[k] = v
    return ShortMirrorConfig(
        enabled=_as_bool(merged["enabled"]),
        slots=int(merged["slots"]),
        vix_min=float(merged["vix_min"]),
        rsi2_min=float(merged["rsi2_min"]),
        min_dollar_volume=float(merged["min_dollar_volume"]),
        min_price=float(merged["min_price"]),
        stop_atr_mult=float(merged["stop_atr_mult"]),
        stop_min_pct=float(merged["stop_min_pct"]),
        stop_max_pct=float(merged["stop_max_pct"]),
        max_hold_sessions=int(merged["max_hold_sessions"]),
        exit_mode=sdip._exit_mode(merged["exit_mode"]),
        trail_max_hold_sessions=int(merged["trail_max_hold_sessions"]),
        trail_final_pct=float(merged["trail_final_pct"]),
        trail_t1_pct=float(merged["trail_t1_pct"]),
        trail_t2_pct=float(merged["trail_t2_pct"]),
        trail_t3_pct=float(merged["trail_t3_pct"]),
        entry_delay_minutes=int(merged["entry_delay_minutes"]),
        entry_window_minutes=int(merged["entry_window_minutes"]),
        exit_minutes_before_close=int(merged["exit_minutes_before_close"]),
        drawdown_halt_pct=float(merged["drawdown_halt_pct"]),
        max_per_sector=int(merged["max_per_sector"]),
        ex_dividend_block_days=int(merged["ex_dividend_block_days"]),
        earnings_block_days=int(merged["earnings_block_days"]),
        rank_by=sdip._rank_by(merged["rank_by"]),
        tighten_start_pct=float(merged["tighten_start_pct"]),
        tighten_every_pct=float(merged["tighten_every_pct"]),
        tighten_by_pct=float(merged["tighten_by_pct"]),
        tighten_floor_pct=float(merged["tighten_floor_pct"]),
        tighten_min_move_pct=float(merged["tighten_min_move_pct"]),
    )


# ---------------------------------------------------------------- the rule
def qualifies_short(s: TickerSnapshot, vix_close: float | None, cfg: ShortMirrorConfig) -> bool:
    if not _finite(vix_close) or float(vix_close) < cfg.vix_min:
        return False
    if not all(_finite(x) for x in (s.close, s.sma200, s.rsi2, s.atr, s.dollar_volume)):
        return False
    return (s.close >= cfg.min_price
            and s.dollar_volume >= cfg.min_dollar_volume
            and s.close < s.sma200
            and s.rsi2 > cfg.rsi2_min)


def rank_short_candidates(snaps: dict[str, TickerSnapshot], vix_close: float | None,
                          cfg: ShortMirrorConfig) -> list[TickerSnapshot]:
    """Qualifiers in the configured pick order (cfg.rank_by), the dip-buyer's orders mirrored.
    "rsi2": highest RSI(2) first (ties: larger 3-day rise, then ticker). "volume_profile": furthest
    OVER the point of control in ATRs first; no profile after all that have one. "move3": biggest
    3-day rise first; an unknown change last. Ties: higher RSI(2), then ticker."""
    picks = [s for s in snaps.values() if qualifies_short(s, vix_close, cfg)]
    picks.sort(key=lambda s: sdip.rank_key(cfg.rank_by, "short", s.rsi2, s.change_3d_pct,
                                           s.vp.poc_atr if s.vp is not None else None, s.ticker))
    return picks


def stop_distance_pct(atr: float, ref_close: float, cfg: ShortMirrorConfig) -> float:
    """Stop distance in percent ABOVE the entry: stop_atr_mult x ATR as a % of the signal-day
    close, clipped to [stop_min_pct, stop_max_pct]."""
    if not (_finite(atr) and _finite(ref_close)) or ref_close <= 0 or atr <= 0:
        return cfg.stop_max_pct
    raw = 100.0 * cfg.stop_atr_mult * atr / ref_close
    return min(max(raw, cfg.stop_min_pct), cfg.stop_max_pct)


def short_stop_price(entry_price: float, atr: float, ref_close: float, cfg: ShortMirrorConfig) -> float:
    """The buy stop, rounded UP to the cent (never closer to the entry than the rule says)."""
    raw = entry_price * (1 + stop_distance_pct(atr, ref_close, cfg) / 100.0)
    return math.ceil(raw * 100 - 1e-9) / 100.0


def short_qty(slot_dollars: float, price: float) -> int:
    """Whole shares only (Alpaca has no fractional or notional shorts); 0 = can't size it."""
    if not (_finite(slot_dollars) and _finite(price)) or price <= 0 or slot_dollars <= 0:
        return 0
    return int(math.floor(slot_dollars / price + 1e-9))


def short_pnl(entry_price: float, cover_price: float, qty: float) -> float:
    return round((entry_price - cover_price) * abs(qty), 2)


def max_hold(cfg: ShortMirrorConfig) -> int:
    """Sessions until the time exit, for the configured exit mode."""
    return cfg.trail_max_hold_sessions if cfg.exit_mode in ("trail", "tighten") else cfg.max_hold_sessions


def cover_reason(live_price: float, last4_closes: list[float], sessions_held: int,
                 cfg: ShortMirrorConfig) -> str | None:
    """Why to cover now (near the close), or None to keep holding. sessions_held counts the
    entry session as 1, so the 10th (20th in trail mode) session held is the time exit (same as
    the backtest). In trail mode the trailing buy stop does the rest, at the broker."""
    if cfg.exit_mode == "average":
        avg = sdip.sma5_with_live(last4_closes, live_price)
        if avg is not None and live_price < avg:
            return f"closed under its 5-day average (${live_price:.2f} < ${avg:.2f})"
    limit = max_hold(cfg)
    if sessions_held >= limit:
        return f"{limit}-session time limit reached"
    return None


def short_trail_width(entry: float, low: float, start_pct: float, cfg: ShortMirrorConfig) -> float:
    """The trail's width in % (above the low) when the lowest low is `low`: start_pct at the
    short, a straight line through the -T1/-T2 checkpoints (each one step narrower) to
    trail_final_pct at -T3 and below. The dip-buyer's stepped_trail_width mirrored."""
    final = cfg.trail_final_pct
    step = (start_pct - final) / 3.0
    g = (entry - low) / entry
    ap = [0.0, cfg.trail_t1_pct / 100, cfg.trail_t2_pct / 100, cfg.trail_t3_pct / 100]
    av = [start_pct, start_pct - step, start_pct - 2 * step, final]
    if g <= ap[0]:
        return av[0]
    if g >= ap[3]:
        return av[3]
    for k in range(3):
        if g <= ap[k + 1]:
            return av[k] - (g - ap[k]) / (ap[k + 1] - ap[k]) * (av[k] - av[k + 1])
    return final


def short_trail_price(entry: float, lowest_low: float, start_pct: float, cfg: ShortMirrorConfig) -> float:
    """The trailing buy stop for the next session from the lowest completed daily low since the
    short (the short day included), rounded UP to the cent; 0.0 while the price hasn't traded
    under the short (not armed)."""
    if not (_finite(entry) and _finite(lowest_low) and _finite(start_pct)) or entry <= 0 or lowest_low <= 0:
        return 0.0
    if lowest_low >= entry:
        return 0.0
    w = short_trail_width(entry, lowest_low, start_pct, cfg)
    return math.ceil(lowest_low * (1 + w / 100.0) * 100 - 1e-9) / 100.0


def short_tighten_trail_price(entry: float, best_low: float, start_pct: float, cfg: ShortMirrorConfig) -> float:
    """exit_mode "tighten" (AIDayTrading's vwap_trail_stop "tighten" math): the buy stop sits
    `width`% over the lowest price since the short, width = start - by/every x the % fall, never
    under the floor. Rounded UP to the cent. 0.0 while the price hasn't traded under the short (not
    armed: the hard stop rules), or on any bad input."""
    vals = (entry, best_low, start_pct, cfg.tighten_every_pct, cfg.tighten_by_pct, cfg.tighten_floor_pct)
    if not all(_finite(x) for x in vals) or entry <= 0 or best_low <= 0 or start_pct <= 0 \
            or cfg.tighten_every_pct <= 0:
        return 0.0
    if best_low >= entry:
        return 0.0
    fall = (entry - best_low) / entry * 100.0
    width = max(cfg.tighten_floor_pct, start_pct - cfg.tighten_by_pct / cfg.tighten_every_pct * fall)
    if not 0 < width < 100:
        return 0.0
    return math.ceil(best_low * (1 + width / 100.0) * 100 - 1e-9) / 100.0


def short_tighten_should_move(current: float | None, new: float, min_move_pct: float) -> bool:
    """A short's live trail moves only DOWN, and only by min_move_pct% or more."""
    if not _finite(new) or new <= 0:
        return False
    if not _finite(current) or not current or current <= 0:
        return True
    return new <= current * (1 - max(0.0, min_move_pct) / 100.0) + 1e-9


def effective_stop(book_entry: dict) -> float:
    """The buy stop in force for a short: the hard stop or the trail, whichever is LOWER."""
    stop = float(book_entry.get("stop") or 0.0)
    trail = float(book_entry.get("trail") or 0.0)
    return min(stop, trail) if trail > 0 else stop


def date_within(day_iso: str, today_iso: str, days: int) -> bool:
    """True when day_iso falls from today through today + days (calendar days). Blank or
    unreadable dates, and past dates, are not 'within'."""
    try:
        d = date.fromisoformat(str(day_iso)[:10])
        t = date.fromisoformat(str(today_iso)[:10])
    except (TypeError, ValueError):
        return False
    return t <= d <= t + timedelta(days=days)


def entry_window_open(sessions: list[dict], day: str, now_hhmm: str, cfg: ShortMirrorConfig) -> bool:
    return sdip.entry_window_open(sessions, day, now_hhmm, cfg)


def exit_window_open(sessions: list[dict], day: str, now_hhmm: str, cfg: ShortMirrorConfig) -> bool:
    return sdip.exit_window_open(sessions, day, now_hhmm, cfg)


def unknown_shorts(positions: list[dict], book: dict, in_flight: set | frozenset) -> list[str]:
    """Short positions at the broker that the short mirror didn't open (and isn't opening) --
    e.g. a long-side double sell once shorting is on. The caller alerts and covers them."""
    return sorted(p["ticker"] for p in positions
                  if _finite(p.get("shares")) and float(p["shares"]) < 0
                  and p["ticker"] not in book and p["ticker"] not in in_flight)


def short_liability(positions: list[dict]) -> float:
    """What the account's short positions owe at market: the sum of |market value| over every
    position with negative shares. Portfolio.total_value subtracts it, because Alpaca's cash
    already includes the short-sale proceeds."""
    total = 0.0
    for p in positions:
        try:
            if float(p.get("shares", 0)) < 0 and _finite(p.get("market_value")):
                total += abs(float(p["market_value"]))
        except (TypeError, ValueError):
            continue
    return round(total, 2)


# ---------------------------------------------------------------- Claude veto
def build_short_veto_prompt(*, today: str, ticker: str, company: str, price: float,
                            snap: TickerSnapshot, vix_close: float, stop_pct: float,
                            earnings_date: str, ex_dividend_date: str, headlines: list[str],
                            max_hold: int) -> str:
    below200 = 100.0 * (snap.close / snap.sma200 - 1) if snap.sma200 else float("nan")
    news = "\n".join(f"- {h}" for h in headlines[:12]) or "- (no headlines found)"
    chg = f"{snap.change_3d_pct:+.1f}%" if snap.change_3d_pct is not None else "n/a"
    return f"""You are the risk check for a short-term mean-reversion SHORT-SELLING strategy.

TODAY'S DATE: {today}

THE STRATEGY (fixed rules, not yours to change): it sells short liquid US stocks that are in a
long-term downtrend (below their 200-day average) right after a sharp 2-3 day bounce, only while
the market is weak and stressed (S&P 500 below its 50-day average and VIX 20 or higher). It covers
on the first close back under the 5-day average, or after {max_hold} trading days at most, with a
hard buy stop about {stop_pct:.1f}% above the short price.

YOUR ONLY JOB: decide whether there is a SPECIFIC, CONCRETE reason this particular bounce is
not an ordinary dead-cat bounce and is likely to keep rising over the next 1-{max_hold} trading
days, or a reason shorting it is dangerous. Real reasons look like: a pending or rumored
acquisition, buyout or tender offer; activist involvement; very high short interest or a
squeeze setup (a heavily shorted or meme stock); an earnings report inside the holding window;
a fresh positive catalyst that CAUSED the bounce (regulatory approval, contract win, guidance
raise, upgrade cycle, index addition, buyback announcement); a special dividend or spin-off.
Do NOT veto for general market fear, valuation, the stock simply being up, or vague
uncertainty -- the bounce inside a downtrend IS the setup. If there is no specific red flag,
allow the trade.

STOCK: {ticker} - {company}
PRICE NOW: ${price:.2f}  (last close ${snap.close:.2f}, 3-day change {chg}, 2-day RSI {snap.rsi2:.1f})
TREND: {below200:+.1f}% vs its 200-day average
VIX (last close): {vix_close:.2f}
NEXT EARNINGS DATE: {earnings_date or "unknown"}
EX-DIVIDEND DATE: {ex_dividend_date or "none found"}
RECENT HEADLINES (newest first):
{news}

Respond with ONLY this JSON object, no other text:
{{"veto": <true or false>, "reason": "<one sentence>"}}
"""


parse_veto_response = sdip.parse_veto_response
