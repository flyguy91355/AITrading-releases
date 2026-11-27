"""Stress Dip-Buyer -- the buy strategy since 2026-11-26 (owner: "do all phases.. build this
strategy, turn off all other strategies that dont need to be on").

Research: docs/research/2026-11-26-swing-strategy-review/README.md. Rule, in plain words:

  * Each evening, from COMPLETED daily bars only: a stock qualifies when it closes above its
    200-day average, its 2-day RSI is below 10 (a sharp 2-3 day drop), it trades at least
    $20M/day on average (63 days), it closes at $5 or more, AND VIX closed at 20 or more that
    same day. Qualifiers are ranked by lowest RSI(2).
  * Next session, right after the open: fill free slots (20 slots since 2026-11-27, each ~1/20
    of equity net of the cash reserve; was 10) in rank order, after a Claude veto check and the
    hard safety gates.
  * Hard stop at entry - 3 x ATR(14) (clipped to 3%-15%), resting at the broker as usual.
  * Exit (exit_mode "trail", the default since 2026-11-26, owner: "Let winners run"): a stepped
    trailing stop replaces the old sell-on-a-close-over-the-5-day-average rule. It starts at the
    stock's own stop % and narrows (through +4.7% and +6.6%) to 0.5% under the high at +8.4%,
    the same curve as the old bot's graduated trail (web/app.py _graduated_trail_pct). It is
    raised once per session from the completed daily highs since the buy, never lowered, and
    the broker's resting stop moves to it. Sell at the close of the 20th session held if
    nothing else has. Backtest of this exact rule: 11.7%/yr and 11.5%/yr in 2016-2026 (all
    stocks / point-in-time S&P 500) vs 9.0%/7.6% for the old exit, but 4.6%/3.0% vs 5.7%/4.6%
    in 2005-2015 with deeper drawdowns (-44% vs -30%). exit_mode "average" restores the old
    exit: the first close above the 5-day average, or the 10th session.

Backtest (both 2005-2015 and 2016-2026, all-universe and point-in-time S&P 500): every signal
averaged +0.40% to +0.71% per ~3-day trade, 68-73% winners. A per-day buy cap HURT it in
2005-2015 (5.7%/yr -> -1.4%/yr at 2/day), so the slots are the only limit.

Everything here is pure (no I/O, no network) so it can be unit-tested; web/app.py owns the
loops, data fetches, Claude call and orders. Indicator math matches the research scripts
exactly (Wilder smoothing seeded with a simple mean of the first n values)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

STRESS_DIP_DEFAULTS = {
    "enabled": True,
    "slots": 20,                      # owner 2026-11-27: "make it 20" (was 10)
    "vix_min": 20.0,
    "rsi2_max": 10.0,
    "min_dollar_volume": 20_000_000.0,
    "min_price": 5.0,
    "stop_atr_mult": 3.0,
    "stop_min_pct": 3.0,
    "stop_max_pct": 15.0,
    "max_hold_sessions": 10,          # exit_mode "average"
    "exit_mode": "trail",             # "trail" (stepped trailing stop) or "average" (5-day average)
    "trail_max_hold_sessions": 20,    # exit_mode "trail"
    "trail_final_pct": 0.5,           # the trail's width at +trail_t3_pct
    "trail_t1_pct": 4.7,              # the curve's checkpoints (the old bot's T1/T2/T3)
    "trail_t2_pct": 6.6,
    "trail_t3_pct": 8.4,
    # Entries start this many minutes after the open and stop being attempted after
    # entry_window_minutes (a restart late in the morning must not buy at a stale price).
    "entry_delay_minutes": 1,
    "entry_window_minutes": 60,
    # Exit checks run in the last N minutes before the session's real close (Alpaca
    # calendar, so half-days close at 13:00 correctly).
    "exit_minutes_before_close": 10,
    # The strategy's own drawdown halt (owner choice 2026-11-26): buys stop when equity is
    # this far below the all-time peak. The global 1.9% daily-loss limit and the
    # risk-tier drawdown halt do NOT apply to this strategy's buys -- they would block it on
    # exactly the stress days it is built for. Hard stops on every position still apply.
    "drawdown_halt_pct": 20.0,
}

# The research engine's dial for the veto call (research.model_stress_dip_veto).
DEFAULT_VETO_MODEL = "claude-haiku-5-5"


@dataclass(frozen=True)
class StressDipConfig:
    enabled: bool
    slots: int
    vix_min: float
    rsi2_max: float
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


def stress_dip_cfg(raw: dict | None) -> StressDipConfig:
    """config['stress_dip'] merged over STRESS_DIP_DEFAULTS. A blank/None value keeps the
    default; a malformed value raises (a typo must not silently become a live number)."""
    merged = dict(STRESS_DIP_DEFAULTS)
    for k, v in (raw or {}).items():
        if k in merged and v is not None and v != "":
            merged[k] = v
    return StressDipConfig(
        enabled=_as_bool(merged["enabled"]),
        slots=int(merged["slots"]),
        vix_min=float(merged["vix_min"]),
        rsi2_max=float(merged["rsi2_max"]),
        min_dollar_volume=float(merged["min_dollar_volume"]),
        min_price=float(merged["min_price"]),
        stop_atr_mult=float(merged["stop_atr_mult"]),
        stop_min_pct=float(merged["stop_min_pct"]),
        stop_max_pct=float(merged["stop_max_pct"]),
        max_hold_sessions=int(merged["max_hold_sessions"]),
        exit_mode=_exit_mode(merged["exit_mode"]),
        trail_max_hold_sessions=int(merged["trail_max_hold_sessions"]),
        trail_final_pct=float(merged["trail_final_pct"]),
        trail_t1_pct=float(merged["trail_t1_pct"]),
        trail_t2_pct=float(merged["trail_t2_pct"]),
        trail_t3_pct=float(merged["trail_t3_pct"]),
        entry_delay_minutes=int(merged["entry_delay_minutes"]),
        entry_window_minutes=int(merged["entry_window_minutes"]),
        exit_minutes_before_close=int(merged["exit_minutes_before_close"]),
        drawdown_halt_pct=float(merged["drawdown_halt_pct"]),
    )


def _exit_mode(v) -> str:
    m = str(v).strip().lower()
    if m not in ("trail", "average"):
        raise ValueError(f"exit_mode must be 'trail' or 'average', not {v!r}")
    return m


def max_hold(cfg: "StressDipConfig") -> int:
    """Sessions until the time exit, for the configured exit mode."""
    return cfg.trail_max_hold_sessions if cfg.exit_mode == "trail" else cfg.max_hold_sessions


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {v!r}")
    return bool(v)


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- indicators

def _wilder(values: list[float], n: int) -> list[float | None]:
    """Wilder smoothing seeded with the simple mean of the first n values (same as the
    research scripts' _wilder). Output aligned to input; None until seeded."""
    out: list[float | None] = [None] * len(values)
    s = 0.0
    v = None
    for i, x in enumerate(values):
        if v is None:
            s += x
            if i == n - 1:
                v = s / n
                out[i] = v
        else:
            v = (v * (n - 1) + x) / n
            out[i] = v
    return out


def rsi_wilder(closes: list[float], n: int = 2) -> float | None:
    """RSI of the LAST close. Needs at least n+1 closes."""
    if len(closes) < n + 1:
        return None
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    ups = _wilder([max(d, 0.0) for d in diffs], n)
    dns = _wilder([max(-d, 0.0) for d in diffs], n)
    au, ad = ups[-1], dns[-1]
    if au is None or ad is None:
        return None
    if ad == 0:
        return 100.0 if au > 0 else 50.0
    rs = au / ad
    return 100.0 - 100.0 / (1.0 + rs)


def atr_wilder(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> float | None:
    """ATR of the last bar. True range of the first bar is high-low."""
    if len(closes) < n or not (len(highs) == len(lows) == len(closes)):
        return None
    trs = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            trs.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    return _wilder(trs, n)[-1]


def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


@dataclass
class TickerSnapshot:
    """Indicator values as of one completed session."""
    ticker: str
    date: str               # YYYY-MM-DD of the last completed bar
    close: float
    sma5: float | None
    sma200: float | None
    rsi2: float | None
    atr: float | None
    dollar_volume: float | None
    last4_closes: list[float] = field(default_factory=list)
    change_3d_pct: float | None = None

    @property
    def atr_pct(self) -> float | None:
        if self.atr is None or not self.close:
            return None
        return 100.0 * self.atr / self.close


def snapshot_from_bars(ticker: str, rows: list[dict]) -> TickerSnapshot | None:
    """rows: completed daily bars, oldest first, each {date, open, high, low, close, volume}.
    Rows with a non-finite OHLC are dropped first. Returns None when there isn't enough
    clean history to judge (fewer than 200 closes)."""
    clean = [r for r in rows if all(_finite(r.get(k)) for k in ("high", "low", "close"))]
    if len(clean) < 200:
        return None
    closes = [float(r["close"]) for r in clean]
    highs = [float(r["high"]) for r in clean]
    lows = [float(r["low"]) for r in clean]
    dv_window = [float(r["close"]) * float(r["volume"]) for r in clean[-63:]
                 if _finite(r.get("volume"))]
    dv = sum(dv_window) / len(dv_window) if len(dv_window) >= 45 else None
    c = closes[-1]
    return TickerSnapshot(
        ticker=ticker, date=str(clean[-1]["date"])[:10], close=c,
        sma5=sma(closes, 5), sma200=sma(closes, 200), rsi2=rsi_wilder(closes, 2),
        atr=atr_wilder(highs, lows, closes, 14), dollar_volume=dv,
        last4_closes=closes[-4:],
        change_3d_pct=(100.0 * (c / closes[-4] - 1)) if len(closes) >= 4 and closes[-4] else None,
    )


def completed_rows(rows: list[dict], today: str, session_closed_today: bool) -> list[dict]:
    """Drop today's still-forming bar. A bar dated today counts only once today's session
    has closed (Yahoo serves the in-progress bar during the day)."""
    if not rows:
        return rows
    if str(rows[-1]["date"])[:10] == today and not session_closed_today:
        return rows[:-1]
    return rows


# ---------------------------------------------------------------- signals

def qualifies(s: TickerSnapshot, vix_close: float | None, cfg: StressDipConfig) -> bool:
    if not _finite(vix_close) or float(vix_close) < cfg.vix_min:
        return False
    if not all(_finite(x) for x in (s.close, s.sma200, s.rsi2, s.atr, s.dollar_volume)):
        return False
    return (s.close >= cfg.min_price
            and s.dollar_volume >= cfg.min_dollar_volume
            and s.close > s.sma200
            and s.rsi2 < cfg.rsi2_max)


def rank_candidates(snaps: dict[str, TickerSnapshot], vix_close: float | None,
                    cfg: StressDipConfig) -> list[TickerSnapshot]:
    """Qualifiers, lowest RSI(2) first (ties: larger 3-day drop first, then ticker)."""
    picks = [s for s in snaps.values() if qualifies(s, vix_close, cfg)]
    picks.sort(key=lambda s: (s.rsi2, s.change_3d_pct if s.change_3d_pct is not None else 0.0, s.ticker))
    return picks


def stop_distance_pct(atr: float, ref_close: float, cfg: StressDipConfig) -> float:
    """Stop distance in percent: stop_atr_mult x ATR as a % of the signal-day close,
    clipped to [stop_min_pct, stop_max_pct]."""
    if not (_finite(atr) and _finite(ref_close)) or ref_close <= 0 or atr <= 0:
        return cfg.stop_max_pct
    raw = 100.0 * cfg.stop_atr_mult * atr / ref_close
    return min(max(raw, cfg.stop_min_pct), cfg.stop_max_pct)


def stop_price(entry_price: float, atr: float, ref_close: float, cfg: StressDipConfig) -> float:
    return round(entry_price * (1 - stop_distance_pct(atr, ref_close, cfg) / 100.0), 2)


def slot_dollars(equity: float, reserve_pct: float, slots: int) -> float:
    """One slot = equity net of the cash-reserve floor, split evenly across the slots."""
    if slots <= 0 or not _finite(equity) or equity <= 0:
        return 0.0
    return equity * (1 - reserve_pct / 100.0) / slots


def drawdown_halted(equity: float, peak: float, halt_pct: float) -> bool:
    if not (_finite(equity) and _finite(peak)) or peak <= 0:
        return False
    return (peak - equity) / peak * 100.0 >= halt_pct


# ---------------------------------------------------------------- exits

def sma5_with_live(last4_closes: list[float], live_price: float) -> float | None:
    """Today's 5-day average if today closed at live_price."""
    if len(last4_closes) < 4 or not _finite(live_price):
        return None
    return (sum(last4_closes[-4:]) + live_price) / 5.0


def exit_reason(live_price: float, last4_closes: list[float], sessions_held: int,
                cfg: StressDipConfig) -> str | None:
    """Why to sell now (near the close), or None to keep holding. sessions_held counts the
    entry session as 1, so the 10th (20th in trail mode) session held is the time exit (same as
    the backtest). In trail mode the trailing stop does the rest, at the broker."""
    if cfg.exit_mode == "average":
        avg = sma5_with_live(last4_closes, live_price)
        if avg is not None and live_price > avg:
            return f"closed above its 5-day average (${live_price:.2f} > ${avg:.2f})"
    limit = max_hold(cfg)
    if sessions_held >= limit:
        return f"{limit}-session time limit reached"
    return None


def stepped_trail_width(entry: float, high: float, start_pct: float, cfg: StressDipConfig) -> float:
    """The trail's width in % at `high`: start_pct at the buy, then a straight line through
    the T1/T2 checkpoints (each one step narrower) to trail_final_pct at T3 and above.
    Same curve as web/app.py _graduated_trail_pct with follow_tp_targets on."""
    final = cfg.trail_final_pct
    step = (start_pct - final) / 3.0
    ap = [entry, entry * (1 + cfg.trail_t1_pct / 100), entry * (1 + cfg.trail_t2_pct / 100),
          entry * (1 + cfg.trail_t3_pct / 100)]
    av = [start_pct, start_pct - step, start_pct - 2 * step, final]
    if high <= ap[0]:
        return av[0]
    if high >= ap[3]:
        return av[3]
    for k in range(3):
        if high <= ap[k + 1]:
            return av[k] - (high - ap[k]) / (ap[k + 1] - ap[k]) * (av[k] - av[k + 1])
    return final


def trail_price(entry: float, highest_high: float, start_pct: float, cfg: StressDipConfig) -> float:
    """The trailing stop for the next session from the highest completed daily high since the
    buy (the buy day included), or 0.0 while the price hasn't traded over the buy (not armed)."""
    if not (_finite(entry) and _finite(highest_high) and _finite(start_pct)) or entry <= 0:
        return 0.0
    if highest_high <= entry:
        return 0.0
    w = stepped_trail_width(entry, highest_high, start_pct, cfg)
    return round(highest_high * (1 - w / 100.0), 2)


def sessions_held(entry_date: str, completed_dates: list[str], today: str) -> int:
    """Sessions from entry_date through today inclusive: completed bars on/after the entry
    date (excluding today's bar if present) plus today itself."""
    prior = {d[:10] for d in completed_dates if entry_date <= d[:10] < today}
    return len(prior) + 1


def calendar_sessions_held(sessions: list[dict], entry_date: str, today: str) -> int:
    """Same count from the market calendar (authoritative; doesn't depend on a price
    download succeeding): sessions dated entry_date..today inclusive, at least 1."""
    return max(1, sum(1 for s in sessions if entry_date <= s.get("date", "") <= today))


# ---------------------------------------------------------------- session calendar
# sessions: Alpaca calendar rows as [{"date": "YYYY-MM-DD", "open": "HH:MM", "close": "HH:MM"}].
# Times are ET wall-clock, so plain zero-padded string comparison orders them correctly.

def add_minutes(hhmm: str, minutes: int) -> str:
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    total = max(0, min(23 * 60 + 59, h * 60 + m + minutes))
    return f"{total // 60:02d}:{total % 60:02d}"


def session_for(sessions: list[dict], day: str) -> dict | None:
    return next((s for s in sessions if s.get("date") == day), None)


def prior_session_date(sessions: list[dict], day: str) -> str | None:
    prior = [s["date"] for s in sessions if s.get("date") and s["date"] < day]
    return max(prior) if prior else None


def last_completed_session(sessions: list[dict], day: str, now_hhmm: str,
                           settle_minutes: int = 30) -> str | None:
    """The newest session whose daily bars (and VIX close) are final: today once
    `settle_minutes` have passed after today's real close, otherwise the prior session."""
    s = session_for(sessions, day)
    if s and now_hhmm >= add_minutes(s["close"], settle_minutes):
        return day
    return prior_session_date(sessions, day)


def entry_window_open(sessions: list[dict], day: str, now_hhmm: str, cfg: StressDipConfig) -> bool:
    s = session_for(sessions, day)
    if not s:
        return False
    start = add_minutes(s["open"], cfg.entry_delay_minutes)
    end = min(add_minutes(s["open"], cfg.entry_window_minutes), s["close"])
    return start <= now_hhmm < end


def exit_window_open(sessions: list[dict], day: str, now_hhmm: str, cfg: StressDipConfig) -> bool:
    s = session_for(sessions, day)
    if not s:
        return False
    return add_minutes(s["close"], -cfg.exit_minutes_before_close) <= now_hhmm < s["close"]


# ---------------------------------------------------------------- Claude veto

def build_veto_prompt(*, today: str, ticker: str, company: str, price: float,
                      snap: TickerSnapshot, vix_close: float, stop_pct: float,
                      earnings_date: str, headlines: list[str], max_hold: int) -> str:
    above200 = 100.0 * (snap.close / snap.sma200 - 1) if snap.sma200 else float("nan")
    news = "\n".join(f"- {h}" for h in headlines[:12]) or "- (no headlines found)"
    chg = f"{snap.change_3d_pct:+.1f}%" if snap.change_3d_pct is not None else "n/a"
    return f"""You are the risk check for a short-term mean-reversion trading strategy.

TODAY'S DATE: {today}

THE STRATEGY (fixed rules, not yours to change): it buys liquid US stocks that are in a
long-term uptrend (above their 200-day average) right after a sharp 2-3 day drop, only while
the market is stressed (VIX 20 or higher). It sells on the first close back above the 5-day
average, or after {max_hold} trading days at most, with a hard stop about {stop_pct:.1f}% below
the buy price. Historically about 70% of these trades bounce within a few days.

YOUR ONLY JOB: decide whether there is a SPECIFIC, CONCRETE reason this particular drop is
not an ordinary pullback and is likely to keep falling over the next 1-{max_hold} trading days.
Real reasons look like: an earnings report inside the holding window; a pending acquisition
or deal that pins the price; fraud, accounting problems, an SEC/DOJ investigation; going-
concern, bankruptcy or delisting risk; a guidance cut, dividend cut, failed trial, lost
lawsuit or other company-specific bad news that CAUSED the drop; removal from a major index.
Do NOT veto for general market fear, valuation, the stock simply being down, or vague
uncertainty -- the drop and the stressed market ARE the setup. If there is no specific red
flag, allow the trade.

STOCK: {ticker} - {company}
PRICE NOW: ${price:.2f}  (last close ${snap.close:.2f}, 3-day change {chg}, 2-day RSI {snap.rsi2:.1f})
TREND: {above200:+.1f}% above its 200-day average
VIX (last close): {vix_close:.2f}
NEXT EARNINGS DATE: {earnings_date or "unknown"}
RECENT HEADLINES (newest first):
{news}

Respond with ONLY this JSON object, no other text:
{{"veto": <true or false>, "reason": "<one sentence>"}}
"""


def parse_veto_response(text: str) -> tuple[bool, str] | None:
    """(veto, reason), or None when the answer can't be read -- the caller treats None as
    "no buy" (fail-closed, AI Data Integrity: a failed buy-trigger check never buys).
    A quoted "true"/"false" string is read as the boolean it names."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(t[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "veto" not in data:
        return None
    v = data["veto"]
    if isinstance(v, bool):
        veto = v
    elif isinstance(v, str) and v.strip().lower() in ("true", "false"):
        veto = v.strip().lower() == "true"
    else:
        return None
    return veto, str(data.get("reason", "")).strip()
