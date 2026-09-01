"""Portfolio state and tracking with SQLite persistence."""

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


def _format_trade_reason(reason: str | None) -> str:
    """Python port of formatSellReason() in dashboard.html (2026-07-23) -- kept in sync
    manually since one renders for a human (the Recent Sells table) and the other for a
    Claude prompt (see _format_trade_history_summary below). Falls back to the raw string
    unchanged for anything not recognized, so a future new reason is never hidden."""
    if not reason:
        return "Not recorded"

    def frac(n: str) -> str:
        return "1/3" if n == "1" else "2/3"

    m = re.match(r"^Take-Profit T(\d)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)})"

    m = re.match(r"^Take-Profit T(\d) \(estimated\)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)}, estimated)"

    if reason == "Take-Profit (final tranche)":
        return "Final 1/3 — Take-Profit"

    if reason == "Take-Profit (estimated, untracked fill)":
        return "Take-Profit (estimated — exact tranche unclear, filled while untracked)"

    if reason == "UNRECONCILED FILL (real order not found — needs manual review)":
        return "Unreconciled Fill — real order not found, needs manual review"

    if reason == "Stop Loss (gap-through market sell, full close)":
        return "Full Close — Stop Loss (price gapped through, sold at market)"
    if reason == "Stop Loss (gap-through market sell)":
        return "Partial — Stop Loss (price gapped through, sold at market)"

    if reason == "Broker-Detected Close (Stop/TP)":
        return "Full Close — Stop or Take-Profit (detected at broker)"

    if reason == "Stop loss hit":
        return "Full Close — Stop Loss"
    if reason == "Trailing stop hit":
        return "Full Close — Trailing Stop"
    if reason == "Manual sell":
        return "Full Close — Manual Sell"

    m = re.match(r"^Portfolio rotation: (.+)$", reason)
    if m:
        return f"Full Close — Rotation ({m.group(1)})"

    return reason


def _format_trade_history_summary(ticker: str, rows: list[tuple]) -> str:
    """Formats raw trade_history rows for one ticker into a compact, chronological summary
    for inclusion in a Claude prompt as minor supplementary context (2026-07-23) -- see
    docs/superpowers/specs/2026-07-23-trade-history-context-design.md. Each row is
    (ticker, action, shares, price, pnl, timestamp, reason), the exact column order of
    "SELECT ticker, action, shares, price, pnl, timestamp, reason FROM trade_history".
    Deliberately does NOT pair buys with their eventual sells into reconstructed "episodes"
    -- lists every row chronologically and lets Claude's own judgment synthesize the
    pattern. Returns an empty string for a ticker with no trade history at all -- callers
    must treat that as "omit this section entirely", not display filler text."""
    if not rows:
        return ""
    lines = [f"Prior trading history on {ticker}:"]
    for _ticker, action, shares, price, pnl, timestamp, reason in rows:
        date = timestamp.split("T")[0] if timestamp else "unknown date"
        if action == "BUY":
            lines.append(f"- {date}: Bought at ${price:.2f}")
            continue
        reason_str = _format_trade_reason(reason)
        pnl_str = ""
        if pnl is not None:
            sign = "+" if pnl >= 0 else "-"
            pnl_str = f", P&L {sign}${abs(pnl):.2f}"
            if shares:
                implied_entry = price - (pnl / shares)
                if implied_entry > 0:
                    pnl_pct = (pnl / (shares * implied_entry)) * 100
                    pct_sign = "+" if pnl_pct >= 0 else ""
                    pnl_str += f" ({pct_sign}{pnl_pct:.1f}%)"
        lines.append(f"- {date}: Sold ({reason_str}) at ${price:.2f}{pnl_str}")
    return "\n".join(lines)


def _format_sell_analysis_summary(rows: list[dict]) -> str:
    """Formats completed "Recent Sell" post-mortems (2026-08-21) for inclusion in a Claude
    prompt as minor supplementary context -- same "for reference only" framing as
    _format_trade_history_summary, appended alongside it rather than replacing it. Each
    dict is one row from Portfolio.get_recent_sell_analyses (already filtered to
    post_mortem_thesis IS NOT NULL, newest close first). Includes the delayed follow-up
    verdict when one has been generated, since it reflects real price action the
    immediate post-mortem couldn't have known about yet. Returns an empty string for no
    rows -- callers must treat that as "omit this section entirely"."""
    if not rows:
        return ""
    lines = ["Past sell post-mortem(s) for this ticker (for reference only):"]
    for r in rows:
        date = (r.get("closed_at") or "").split("T")[0] or "unknown date"
        lines.append(f"- {date}: {r.get('post_mortem_thesis', '')}")
        if r.get("followup_reasoning"):
            lines.append(f"  Follow-up: {r['followup_reasoning']}")
    return "\n".join(lines)


def _format_analysis_history_summary(ticker: str, rows: list[tuple]) -> str:
    """Formats every past real analysis for this ticker into a compact, chronological
    summary for _build_analysis_history_section (2026-08-21, "Analysis History"
    feed-forward feature; extended 2026-08-24 for the SMA Trend-Confirmation Track
    design, ported from AIShortTrading). Each row is (generated_at, conviction_score,
    signal, entry_price, fair_value_estimate, watch_condition, sma_relationship,
    trend_confirms_entry), the exact column order of get_analysis_history_summary's
    SELECT. sma_relationship/trend_confirms_entry are None for any Track 1-only row
    (the vast majority) -- omitted from that row's line entirely, not shown as
    "None". Deliberately includes ALL history the caller passes in (per owner
    direction: "make sure to prompt ai to use those previous (I think all)
    analysys"), not a capped recent window -- each line is kept compact specifically
    so a long history still stays a reasonable prompt size. Returns an empty string
    for a ticker with no history at all -- callers must treat that as "omit this
    section entirely"."""
    if not rows:
        return ""
    lines = [f"Prior analysis history on {ticker} (chronological):"]
    for (generated_at, conviction_score, signal, entry_price, fair_value_estimate,
         watch_condition, sma_relationship, trend_confirms_entry) in rows:
        date = generated_at.split("T")[0] if generated_at else "unknown date"
        line = f"- {date}: {signal}, conviction {conviction_score:.1f}/10"
        if entry_price:
            line += f", entry ${entry_price:.2f}"
        if fair_value_estimate:
            line += f", fair value ${fair_value_estimate:.2f}"
        if sma_relationship:
            confirmed = "confirmed" if trend_confirms_entry else "not confirmed"
            line += f", SMA {sma_relationship} ({confirmed})"
        line += f" — watch: {watch_condition}" if watch_condition else " — no watch condition stated"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_targets: list[float]
    sector: str
    opened_at: datetime
    trailing_stop: float | None = None
    # Snapshotted once per trading day (see Portfolio.new_trading_day) -- the price this
    # position was at when today's session started, used for a per-position "Daily P/L"
    # distinct from unrealized_pnl (which is total gain since entry_price, however many
    # days ago that was). None for a position that didn't exist yet at the last snapshot
    # (opened intraday, or the field predates this addition) -- day_pnl/day_pnl_pct fall
    # back to entry_price in that case, so a same-day position's "daily" P&L is correctly
    # identical to its total P&L rather than showing as 0 or crashing.
    day_open_price: float | None = None
    # Anchor price for the graduated final-tranche trailing stop (2026-07-22) -- captured
    # once, the first tick position_update_loop notices the position has entered its final
    # tranche (T1 and T2 both fired, one target left). Lets the trail interpolate smoothly
    # from 2% (right at this anchor) down to 0.5% (at the stored final target reference
    # price) instead of the old flat-3%-then-instant-0.5% step. None before the final
    # tranche begins, or for a position that predates this field.
    final_tranche_start_price: float | None = None
    # Cumulative $ / shares from every PARTIAL fill (T1, T2) on this still-open position
    # (2026-07-22). Needed because unrealized_pnl only ever reflects the REMAINING shares
    # vs entry_price -- once T1/T2 has fired, that real, already-banked profit (now sitting
    # in cash) becomes invisible from the position's own P/L, understating (or in some
    # cases overstating, if the later tranche moved against it) how the position has
    # actually performed since it was opened. See Position.lifetime_pnl below, which is
    # what the dashboard's "P/L" badge now shows for a held position instead of raw
    # unrealized_pnl. Incremented at every site that logs a partial (non-final) TP fill to
    # trade_history -- real fills in check_take_profits(), and the honest unreconciled-fill
    # record in OrderManager._log_unreconciled_fill() for fills detected only via a
    # share-count delta that couldn't be matched to a real order (offline/between-polls).
    # unrealized_pnl/unrealized_pnl_pct themselves are UNCHANGED
    # and still used everywhere they were before (e.g. logging "this sell's P&L" at the
    # moment a position closes) -- this is purely an additive display concept, not a
    # replacement.
    realized_pnl: float = 0.0
    shares_sold: float = 0.0
    # Original T1/T2 target prices, captured once at buy time (2026-07-23) and never
    # mutated afterward, even once those orders fire and are popped off
    # take_profit_targets. Needed so the graduated trailing-stop curve can still bend
    # through T1/T2 as fixed checkpoints (`risk_management.trailing_stop_follow_tp_targets`)
    # after they've fired — take_profit_targets[-1] remains a reliable live reference for
    # T3 on its own (it's never removed until the position fully closes), but T1/T2's
    # prices would otherwise be lost the moment each one fills. None for a position that
    # predates this field, or whose signal had fewer than 1/2 targets at buy time.
    t1_target_price: float | None = None
    t2_target_price: float | None = None
    # One-way latch (2026-07-24) -- once True, this position's trailing stop permanently
    # uses take_profit.dollar_target_trail_pct instead of the graduated entry->T3 curve
    # (_graduated_trail_pct), regardless of tranche stage. Never reset back to False once
    # set, even if lifetime_pnl later dips back below dollar_target_amount -- see
    # web/app.py's trailing-stop block and docs/superpowers/specs/
    # 2026-07-24-dollar-profit-target-trailing-stop-design.md.
    profit_target_hit: bool = False
    # Persistent identifier for this position's entire lifecycle (2026-07-27), assigned
    # once at the moment of the real buy fill and never regenerated -- lets every sell
    # tranche (T1/T2/final) that later closes this position be traced back to the lot it
    # came from, for accurate win/loss grouping and tax-lot tracking (see
    # docs/superpowers -- the win-loss-dashboard-stat and tax-tracking-database design
    # notes). None for a position that predates this field.
    trade_id: str | None = None
    # "Why AI Bought This" (2026-08-21, owner request) -- a snapshot of the AI's real
    # buy-time decision, captured once at the moment of purchase and never regenerated
    # or overwritten by a later re-analysis while the position is held. Deliberately a
    # SNAPSHOT, not a live reference to whatever's currently cached for this ticker --
    # the whole point is answering "why was THIS bought," which should stay fixed even
    # if the ticker's own thesis later drifts. buy_rr/buy_required_rr are the exact
    # numbers that cleared the gate at that moment (not recomputed later), so the
    # displayed R/R breakdown is always accurate to the real decision, not a
    # reconstruction from possibly-stale current data. None/"" for any position bought
    # before this field existed -- backfilled once via scripts/backfill_buy_rationale.py
    # rather than left silently blank forever (see that script's own docstring).
    buy_thesis: str = ""
    buy_reasoning: str = ""
    buy_conviction: int | None = None
    buy_signal: str = ""
    buy_rr: float | None = None
    buy_required_rr: float | None = None
    buy_fair_value: float | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    @property
    def lifetime_pnl(self) -> float:
        """Total gain/loss since this position was first opened: unrealized gain on
        whatever shares remain, PLUS every partial exit's real (or best-effort estimated)
        realized gain along the way. This is what the dashboard shows as "P/L" for a held
        position -- unrealized_pnl alone silently drops any profit/loss already banked from
        an earlier T1/T2 fill."""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def lifetime_pnl_pct(self) -> float:
        """lifetime_pnl as a % of the ORIGINAL investment (current remaining shares +
        every share already sold via a partial exit) -- not the shrunken current cost_basis
        unrealized_pnl_pct uses, which only reflects what's left invested right now."""
        original_cost = (self.shares + self.shares_sold) * self.entry_price
        if original_cost == 0:
            return 0.0
        return (self.lifetime_pnl / original_cost) * 100

    @property
    def day_pnl(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        return (self.current_price - ref) * self.shares

    @property
    def day_pnl_pct(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        if not ref:
            return 0.0
        return ((self.current_price - ref) / ref) * 100


class Portfolio:
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config["portfolio"]["initial_capital"]
        self.cash = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.peak_value = self.initial_capital
        self.day_start_value = self.initial_capital
        # Which calendar date (YYYY-MM-DD, ET) day_start_value was last snapshotted for
        # (2026-07-22) -- closes the long-deferred CR14 gap: without this, OrderManager.
        # _sync_portfolio() unconditionally overwrote day_start_value from Alpaca's
        # last_equity on EVERY restart, even an intraday one where new_trading_day() had
        # already correctly set it earlier that same day. Now _sync_portfolio only touches
        # it when day_start_date shows a stale (prior) date -- i.e. the process was down
        # across the day boundary and genuinely missed today's reset -- self-healing that
        # case while leaving an already-correct same-day value alone.
        self.day_start_date: str | None = None
        # Set while a rotation swap's sell is in flight (submitted but not yet filled) —
        # during this window the sold position and its replacement can be briefly
        # double-counted in total_value, so update_peak() must not ratchet on it.
        self._rotation_in_progress = False
        self.db_path = config.get("database", {}).get("path", "data/aitrading.db")
        self._db: aiosqlite.Connection | None = None
        # Most recent LOSING sale's timestamp per ticker (2026-07-27, wash-sale rebuy
        # block) -- kept in-memory, updated synchronously alongside every trade_history
        # SELL insert (see close_position_async below and OrderManager's other insert
        # sites) so RiskManager.check_wash_sale_cooldown can check it with a plain dict
        # lookup at buy-decision time, no DB round-trip needed. Loaded once from real
        # trade_history at startup by _load_recent_losses (called from initialize()).
        self.recent_losses: dict[str, datetime] = {}

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def cash_pct(self) -> float:
        if self.total_value == 0:
            return 0.0
        return (self.cash / self.total_value) * 100

    @property
    def day_pnl(self) -> float:
        return self.total_value - self.day_start_value

    async def initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # timeout=20.0 (2026-08-19, cross-module hardening after a real
        # sqlite3.OperationalError: database is locked crash in a sibling sync
        # connection to this same shared data/aitrading.db file -- see
        # src/utils/watchlist_manager.py's matching comment for the live incident).
        # aiosqlite passes this through to the underlying sqlite3 connection's own
        # busy-retry window, same mechanism as the sync connections elsewhere in this
        # app -- longer than the implicit 5.0s default.
        self._db = await aiosqlite.connect(self.db_path, timeout=20.0)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                shares REAL,
                entry_price REAL,
                current_price REAL,
                stop_loss REAL,
                take_profit_targets TEXT,
                sector TEXT,
                opened_at TEXT,
                trailing_stop REAL,
                day_open_price REAL,
                final_tranche_start_price REAL,
                realized_pnl REAL,
                shares_sold REAL,
                t1_target_price REAL,
                t2_target_price REAL,
                profit_target_hit INTEGER,
                trade_id TEXT
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL,
                peak_value REAL,
                day_start_value REAL,
                day_start_date TEXT
            )
        """)
        async with self._db.execute("PRAGMA table_info(portfolio_state)") as cur:
            _ps_cols = {row[1] for row in await cur.fetchall()}
        if "day_start_date" not in _ps_cols:
            await self._db.execute("ALTER TABLE portfolio_state ADD COLUMN day_start_date TEXT")
            await self._db.commit()
            logger.info("Migrated portfolio_state: added day_start_date column")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                pnl REAL,
                timestamp TEXT,
                reason TEXT,
                trade_id TEXT
            )
        """)
        await self._db.commit()

        # Add reason column to any pre-existing trade_history table that predates it
        # (2026-07-21) -- a plain ALTER TABLE ADD COLUMN, not the rename-recreate dance
        # used below for the shares INTEGER->REAL migration, since SQLite supports adding
        # a nullable column directly. Existing rows get reason=NULL (no reason was ever
        # recorded for them) -- the API layer treats a NULL/missing reason as "not
        # recorded", not as an error.
        async with self._db.execute("PRAGMA table_info(trade_history)") as cur:
            _th_cols = {row[1] for row in await cur.fetchall()}
        if "reason" not in _th_cols:
            await self._db.execute("ALTER TABLE trade_history ADD COLUMN reason TEXT")
            await self._db.commit()
            logger.info("Migrated trade_history: added reason column")

        # Add day_open_price to any pre-existing positions table that predates it
        # (2026-07-22) -- same plain ALTER TABLE ADD COLUMN pattern as the reason-column
        # migration above. Existing rows get NULL, which Position/_load_state already
        # treats as "no snapshot yet" (falls back to entry_price for day_pnl), so no
        # backfill is needed -- the next new_trading_day() call naturally populates it.
        async with self._db.execute("PRAGMA table_info(positions)") as cur:
            _pos_cols = {row[1] for row in await cur.fetchall()}
        if "day_open_price" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN day_open_price REAL")
            await self._db.commit()
            logger.info("Migrated positions: added day_open_price column")
        if "final_tranche_start_price" not in _pos_cols:
            await self._db.execute(
                "ALTER TABLE positions ADD COLUMN final_tranche_start_price REAL")
            await self._db.commit()
            logger.info("Migrated positions: added final_tranche_start_price column")
        # realized_pnl / shares_sold (2026-07-22) -- see Position.lifetime_pnl. Existing rows
        # get NULL; _load_state below treats that as 0.0 (no partial fills tracked yet for
        # a position that predates this column), same "no snapshot = fall back to a sane
        # default" pattern as day_open_price.
        if "realized_pnl" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN realized_pnl REAL")
            await self._db.commit()
            logger.info("Migrated positions: added realized_pnl column")
        if "shares_sold" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN shares_sold REAL")
            await self._db.commit()
            logger.info("Migrated positions: added shares_sold column")
        # t1_target_price / t2_target_price (2026-07-23) -- fixed T1/T2 reference prices for
        # the graduated trailing-stop curve (see Position.t1_target_price). Existing rows get
        # NULL, which the trailing-stop calc in position_update_loop treats as "fall back to
        # a straight entry->T3 line" for that position, same defensive pattern as the other
        # nullable columns here.
        if "t1_target_price" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN t1_target_price REAL")
            await self._db.commit()
            logger.info("Migrated positions: added t1_target_price column")
        if "t2_target_price" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN t2_target_price REAL")
            await self._db.commit()
            logger.info("Migrated positions: added t2_target_price column")
        # profit_target_hit (2026-07-24) -- see Position.profit_target_hit. Existing rows
        # get NULL, treated as 0/False by _load_state below (a position that predates this
        # feature has obviously never had it latch).
        if "profit_target_hit" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN profit_target_hit INTEGER")
            await self._db.commit()
            logger.info("Migrated positions: added profit_target_hit column")
        # trade_id (2026-07-27) -- see Position.trade_id. Existing rows get NULL (a
        # position opened before this field existed has no recoverable original id), same
        # "no snapshot = leave it None" pattern as the other nullable columns here.
        if "trade_id" not in _pos_cols:
            await self._db.execute("ALTER TABLE positions ADD COLUMN trade_id TEXT")
            await self._db.commit()
            logger.info("Migrated positions: added trade_id column")
        # "Why AI Bought This" snapshot (2026-08-21) -- see Position's own field
        # docstring. Existing rows get NULL/empty on every one of these 7 columns; a
        # position that predates this feature is exactly what
        # scripts/backfill_buy_rationale.py exists to fill in, once, retroactively.
        for _col, _type in (
            ("buy_thesis", "TEXT"), ("buy_reasoning", "TEXT"),
            ("buy_conviction", "INTEGER"), ("buy_signal", "TEXT"),
            ("buy_rr", "REAL"), ("buy_required_rr", "REAL"), ("buy_fair_value", "REAL"),
        ):
            if _col not in _pos_cols:
                await self._db.execute(f"ALTER TABLE positions ADD COLUMN {_col} {_type}")
                await self._db.commit()
                logger.info("Migrated positions: added %s column", _col)
        async with self._db.execute("PRAGMA table_info(trade_history)") as cur:
            _th_cols2 = {row[1] for row in await cur.fetchall()}
        if "trade_id" not in _th_cols2:
            await self._db.execute("ALTER TABLE trade_history ADD COLUMN trade_id TEXT")
            await self._db.commit()
            logger.info("Migrated trade_history: added trade_id column")

        # "Recent Sell" post-mortem (2026-08-21, owner request) -- a position's buy-side
        # thesis/reasoning/etc. is about to be lost the moment close_position_async
        # deletes the Position row, so it's snapshotted here (see close_position_async
        # below) alongside the raw close facts, one row per trade_id. post_mortem_*
        # fields start NULL and are filled in by a separate periodic job in web/app.py
        # (this file has no Claude client) once per closed trade; followup_* fields are
        # filled in by a second, delayed pass using price action after the sale.
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sell_analysis (
                trade_id TEXT PRIMARY KEY,
                ticker TEXT,
                buy_thesis TEXT,
                buy_reasoning TEXT,
                buy_conviction INTEGER,
                buy_signal TEXT,
                buy_rr REAL,
                buy_required_rr REAL,
                buy_fair_value REAL,
                entry_price REAL,
                exit_price REAL,
                opened_at TEXT,
                closed_at TEXT,
                post_mortem_thesis TEXT,
                post_mortem_reasoning TEXT,
                generated_at TEXT,
                followup_due_date TEXT,
                followup_reasoning TEXT,
                followup_generated_at TEXT
            )
        """)
        await self._db.commit()

        # "Analysis History" feed-forward (2026-08-21, owner request: "I see the same
        # stocks analyzed all the time.. same ones come up.. surely the most current
        # analysis should take all of the other analysis into account for a broder
        # picture") -- one row per real (non-fallback) analysis of a ticker, appended
        # (never overwritten, unlike the research_reports/reports_cache.json "latest
        # only" cache) so a recurring candidate's whole arc -- including any
        # previously-stated watch_condition -- survives being scanned again and again,
        # instead of the AI starting cold each time. See ResearchEngine.
        # explain_buy_decision's sibling docstring / docs/CLAUDE_HISTORY.md's
        # 2026-08-21 "Analysis History" entry for the full design.
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                generated_at TEXT,
                conviction_score REAL,
                signal TEXT,
                entry_price REAL,
                fair_value_estimate REAL,
                watch_condition TEXT
            )
        """)
        await self._db.commit()

        # sma_relationship / trend_confirms_entry (2026-08-24, SMA Trend-Confirmation
        # Track design, ported from AIShortTrading) -- captures the SMA50/SMA200
        # relationship and the AI's trend_confirms_entry judgment at the time of a
        # Track 2 analysis. Existing rows (and every Track 1 row, which never passes
        # these) get NULL on both -- same "NULL = predates this feature / not
        # applicable" convention as every other nullable column here.
        async with self._db.execute("PRAGMA table_info(analysis_history)") as cur:
            _ah_cols = {row[1] for row in await cur.fetchall()}
        if "sma_relationship" not in _ah_cols:
            await self._db.execute("ALTER TABLE analysis_history ADD COLUMN sma_relationship TEXT")
            await self._db.commit()
            logger.info("Migrated analysis_history: added sma_relationship column")
        if "trend_confirms_entry" not in _ah_cols:
            await self._db.execute("ALTER TABLE analysis_history ADD COLUMN trend_confirms_entry INTEGER")
            await self._db.commit()
            logger.info("Migrated analysis_history: added trend_confirms_entry column")

        # Migrate existing INTEGER shares columns to REAL so fractional shares are preserved
        for table in ("positions", "trade_history"):
            # Recovery: if a previous migration was interrupted (e.g. power loss between
            # RENAME and CREATE), restore the backup so the migration can retry cleanly.
            async with self._db.execute(f"PRAGMA table_info(_{table}_old)") as cur:
                orphan_exists = bool(await cur.fetchall())
            if orphan_exists:
                logger.warning("Found orphan _%s_old from interrupted migration — restoring", table)
                async with self._db.execute(f"PRAGMA table_info({table})") as cur:
                    partial_exists = bool(await cur.fetchall())
                if partial_exists:
                    await self._db.execute(f"DROP TABLE {table}")
                await self._db.execute(f"ALTER TABLE _{table}_old RENAME TO {table}")
                await self._db.commit()

            async with self._db.execute(f"PRAGMA table_info({table})") as cur:
                cols = await cur.fetchall()
            col_types = {row[1]: row[2].upper() for row in cols}
            if col_types.get("shares", "") == "INTEGER":
                await self._db.execute(f"ALTER TABLE {table} RENAME TO _{table}_old")
                # Rebuilt from the OLD table's own real column list (fixed 2026-09-01,
                # full-codebase review) rather than a hardcoded CREATE. The hardcoded
                # version was frozen at its mid-2026-07 shape and never updated as later
                # ALTERs accumulated, so it omitted trade_id and the seven buy_* columns
                # that the ALTER migrations directly above this loop had just guaranteed
                # exist. On a legacy DB that meant the copy silently DESTROYED the tax /
                # win-loss trade_id lineage and every buy-rationale snapshot, and then
                # every later _save_position and trade_history INSERT (both name those
                # columns) raised "no such column" -- which those callers' fail-soft
                # handlers misreport as "database locked", so nothing persisted at all
                # until the next restart re-added the columns as NULL. Deriving the
                # schema from what is actually present cannot drift again: whatever a
                # future ALTER adds is carried through automatically.
                async with self._db.execute(f"PRAGMA table_info(_{table}_old)") as cur:
                    _old_info = await cur.fetchall()
                _old_names = [row[1] for row in _old_info]
                _defs = []
                for _row in _old_info:
                    _name, _type, _is_pk = _row[1], (_row[2] or "TEXT"), _row[5]
                    if _name == "shares":
                        _type = "REAL"  # the entire point of this migration
                    _decl = f"{_name} {_type}"
                    if _is_pk:
                        # Preserve the original key shape: trade_history's synthetic
                        # INTEGER id keeps AUTOINCREMENT, positions' ticker stays the
                        # natural primary key.
                        _decl += " PRIMARY KEY"
                        if _name == "id" and _type.upper() == "INTEGER":
                            _decl += " AUTOINCREMENT"
                    _defs.append(_decl)
                await self._db.execute(f"CREATE TABLE {table} ({', '.join(_defs)})")
                # Columns named explicitly on BOTH sides -- never positional -- so the
                # CAST can't silently line up against a different column.
                _select_cols = ", ".join(
                    f"CAST({n} AS REAL)" if n == "shares" else n for n in _old_names)
                await self._db.execute(
                    f"INSERT INTO {table} ({', '.join(_old_names)}) "
                    f"SELECT {_select_cols} FROM _{table}_old"
                )
                await self._db.execute(f"DROP TABLE _{table}_old")
                await self._db.commit()
                logger.info("Migrated %s.shares from INTEGER to REAL", table)

        await self._load_state()
        await self._load_recent_losses()

    async def _load_recent_losses(self):
        """Populates self.recent_losses from real trade_history at startup (2026-07-27) --
        see the field's own docstring above. Looks back research.wash_sale_cooldown_days
        (fixed 2026-08-02, GitHub #43 -- was hardcoded to 30, which silently dropped any
        loss between 30 and the real configured value, e.g. the live 31-day setting, off
        a restart) since anything older than that is already outside the wash-sale window
        and irrelevant to check_wash_sale_cooldown regardless. Same config path and
        default RiskManager itself uses, so the two can't drift apart."""
        cooldown_days = self.config.get("research", {}).get("wash_sale_cooldown_days", 30)
        cutoff = (datetime.now() - timedelta(days=cooldown_days)).isoformat()
        try:
            async with self._db.execute(
                "SELECT ticker, MAX(timestamp) FROM trade_history "
                "WHERE action = 'SELL' AND pnl < 0 AND timestamp >= ? GROUP BY ticker",
                (cutoff,),
            ) as cur:
                rows = await cur.fetchall()
            for ticker, ts in rows:
                try:
                    parsed = datetime.fromisoformat(ts)
                    # Normalize to naive (2026-07-27 fix, caught by a real test run) --
                    # most trade_history timestamps are naive local (datetime.now()
                    # .isoformat()), but rows written from OrderManager._reconcile_
                    # untracked_fill use Alpaca's own real (UTC, timezone-aware)
                    # filled_at instead. Mixing the two crashes any later
                    # `datetime.now() - parsed` arithmetic (check_wash_sale_cooldown)
                    # with "can't subtract offset-naive and offset-aware datetimes" --
                    # every other datetime in this codebase is naive local, so that's
                    # the convention to normalize to, not the other way around.
                    if parsed.tzinfo is not None:
                        parsed = parsed.replace(tzinfo=None)
                    self.recent_losses[ticker] = parsed
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.warning("_load_recent_losses failed: %s", e)

    async def _load_state(self):
        async with self._db.execute(
            "SELECT cash, peak_value, day_start_value, day_start_date FROM portfolio_state WHERE id = 1"
        ) as cur:
            row = await cur.fetchone()
            if row:
                self.cash = row[0]
                self.peak_value = row[1]
                self.day_start_value = row[2]
                self.day_start_date = row[3] if len(row) > 3 else None

        async with self._db.execute("SELECT * FROM positions") as cur:
            async for row in cur:
                targets = json.loads(row[5]) if row[5] else []
                self.positions[row[0]] = Position(
                    ticker=row[0],
                    shares=row[1],
                    entry_price=row[2],
                    current_price=row[3],
                    stop_loss=row[4],
                    take_profit_targets=targets,
                    sector=row[6],
                    opened_at=datetime.fromisoformat(row[7]),
                    trailing_stop=row[8],
                    day_open_price=row[9] if len(row) > 9 else None,
                    final_tranche_start_price=row[10] if len(row) > 10 else None,
                    realized_pnl=row[11] if len(row) > 11 and row[11] is not None else 0.0,
                    shares_sold=row[12] if len(row) > 12 and row[12] is not None else 0.0,
                    t1_target_price=row[13] if len(row) > 13 else None,
                    t2_target_price=row[14] if len(row) > 14 else None,
                    profit_target_hit=bool(row[15]) if len(row) > 15 and row[15] is not None else False,
                    trade_id=row[16] if len(row) > 16 else None,
                    buy_thesis=row[17] if len(row) > 17 and row[17] is not None else "",
                    buy_reasoning=row[18] if len(row) > 18 and row[18] is not None else "",
                    buy_conviction=row[19] if len(row) > 19 else None,
                    buy_signal=row[20] if len(row) > 20 and row[20] is not None else "",
                    buy_rr=row[21] if len(row) > 21 else None,
                    buy_required_rr=row[22] if len(row) > 22 else None,
                    buy_fair_value=row[23] if len(row) > 23 else None,
                )

    async def _save_state(self):
        """Persists the portfolio-level snapshot (cash, peak_value, day_start_value/
        date) -- called from 15 real sites across order_manager.py/web/app.py/this
        file, several reachable from position_update_loop's own tick (the same
        loop _save_position's own 2026-08-25 fix covers -- see that method's
        docstring for the live incident: "Position update failed: database is
        locked" recurring during an unusually write-heavy afternoon). Same
        retry-then-warn treatment for the identical reason -- this is a real,
        continuously-rewritten snapshot (self-healing on the next successful
        save), not correctness-critical for any single tick, but worth one
        retry before giving up rather than silently losing it with no attempt."""
        if not self._db:
            return
        for attempt in range(2):
            try:
                await self._db.execute(
                    "INSERT OR REPLACE INTO portfolio_state (id, cash, peak_value, day_start_value, day_start_date) VALUES (1, ?, ?, ?, ?)",
                    (self.cash, self.peak_value, self.day_start_value, self.day_start_date),
                )
                await self._db.commit()
                return
            except sqlite3.OperationalError as e:
                if attempt == 0:
                    logger.warning(
                        "_save_state: database locked, retrying once in 2s: %s", e,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.error(
                    "_save_state: database still locked after one retry -- portfolio "
                    "snapshot (cash=%.2f) may be stale until a later save succeeds: %s",
                    self.cash, e,
                )

    async def _save_position(self, position: Position):
        """Persists the full Position row -- the canonical DB record for a held
        position's state (shares, stop_loss, take_profit_targets, trailing_stop,
        etc.), called from 15 real sites across order_manager.py/web/app.py/this
        file (trailing-stop ratchets, tranche fills, AI-chosen-stop updates, and
        more). Retries once on a locked database, then fails soft with a loud
        warning rather than raising (fixed 2026-08-25, live incident -- caught
        live as "Monitor re-analysis failed: database is locked" repeating
        across several held tickers during an unusually write-heavy afternoon).

        Deliberately retry-then-warn here, NOT the same silent skip-and-log
        pattern set_scan_cursor/save_analysis_history use -- this write is more
        consequential than either of those (a real position's persisted state,
        not a losable convenience/history record), so it's worth one real retry
        before giving up. Even after a failed retry, this is NOT a live
        protection gap: the caller's own in-memory `position` object (e.g.
        position.stop_loss, already updated by the caller before this method
        was ever invoked) is what sync_exit_orders/check_protection_gaps
        actually read on their own next cycle -- broker-side protection is
        governed by that live in-memory state, not by whether this specific DB
        write landed. The real, bounded risk is narrower: this one update could
        be lost if the process restarts before a later save of the same
        position succeeds."""
        if not self._db:
            return
        for attempt in range(2):
            try:
                await self._db.execute(
                    "INSERT OR REPLACE INTO positions (ticker, shares, entry_price, current_price, stop_loss, take_profit_targets, sector, opened_at, trailing_stop, day_open_price, final_tranche_start_price, realized_pnl, shares_sold, t1_target_price, t2_target_price, profit_target_hit, trade_id, buy_thesis, buy_reasoning, buy_conviction, buy_signal, buy_rr, buy_required_rr, buy_fair_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        position.ticker, position.shares, position.entry_price,
                        position.current_price, position.stop_loss,
                        json.dumps(position.take_profit_targets), position.sector,
                        position.opened_at.isoformat(), position.trailing_stop,
                        position.day_open_price, position.final_tranche_start_price,
                        position.realized_pnl, position.shares_sold,
                        position.t1_target_price, position.t2_target_price,
                        int(position.profit_target_hit), position.trade_id,
                        position.buy_thesis, position.buy_reasoning, position.buy_conviction,
                        position.buy_signal, position.buy_rr, position.buy_required_rr,
                        position.buy_fair_value,
                    ),
                )
                await self._db.commit()
                return
            except sqlite3.OperationalError as e:
                if attempt == 0:
                    logger.warning(
                        "_save_position(%s): database locked, retrying once in 2s: %s",
                        position.ticker, e,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.error(
                    "_save_position(%s): database still locked after one retry -- "
                    "this position's DB record may be stale (in-memory stop_loss=%.2f) "
                    "until a later save succeeds. Broker-side protection is UNAFFECTED "
                    "-- exit-order sync reads live in-memory state, not the DB: %s",
                    position.ticker, position.stop_loss, e,
                )

    async def _remove_position_db(self, ticker: str):
        """Same retry-then-warn treatment as _save_position (2026-08-25) -- a
        failed delete here leaves a stale row in the DB for a position that's
        genuinely closed in memory; the position's own in-memory removal from
        self.positions (done by the caller, not here) is what every real
        decision actually reads, so this isn't a live gap, just a DB-tidiness
        risk that self-heals on the next full portfolio sync."""
        if not self._db:
            return
        for attempt in range(2):
            try:
                await self._db.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
                await self._db.commit()
                return
            except sqlite3.OperationalError as e:
                if attempt == 0:
                    logger.warning(
                        "_remove_position_db(%s): database locked, retrying once in 2s: %s",
                        ticker, e,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.error(
                    "_remove_position_db(%s): database still locked after one retry -- "
                    "a stale row may remain until the next full portfolio sync: %s",
                    ticker, e,
                )

    async def get_trade_history_summary(self, ticker: str) -> str:
        """Every trade_history row for this ticker, formatted as compact context for a
        Claude prompt when reconsidering it as a buy candidate (2026-07-23) -- see
        docs/superpowers/specs/2026-07-23-trade-history-context-design.md. Fails open
        (returns "") on any DB error or if the DB hasn't been initialized yet, same as
        every other nice-to-have query in this file -- this must never be able to delay or
        break a real buy decision."""
        if not self._db:
            return ""
        try:
            async with self._db.execute(
                "SELECT ticker, action, shares, price, pnl, timestamp, reason "
                "FROM trade_history WHERE ticker = ? ORDER BY timestamp",
                (ticker,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_trade_history_summary failed for %s: %s", ticker, e)
            return ""
        summary = _format_trade_history_summary(ticker, rows)

        # "Recent Sell" post-mortem feed-forward (2026-08-21, owner request: "saved for
        # future analysis of that paticular stock... for future buying and selling
        # analysys") -- same minor, supplementary framing as the trade-history summary
        # above, appended rather than replacing it. Fails open exactly like the query
        # above -- a post-mortem lookup failure must never delay or break a real buy
        # decision either.
        try:
            sell_analyses = await self.get_recent_sell_analyses(ticker)
        except Exception as e:
            logger.warning("get_recent_sell_analyses failed for %s: %s", ticker, e)
            sell_analyses = []
        post_mortem_summary = _format_sell_analysis_summary(sell_analyses)
        if post_mortem_summary:
            summary = f"{summary}\n\n{post_mortem_summary}" if summary else post_mortem_summary
        return summary

    def update_peak(self):
        # Guarded here (not just at the order_manager.py call site) because
        # close_position()/close_position_async() also call this directly during a
        # rotation swap's sell execution — a second, previously-unguarded path that let
        # peak_value ratchet on a transiently double-counted total_value (see RS-1).
        if self._rotation_in_progress:
            return
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value

    def new_trading_day(self, today_str: str | None = None):
        """Snapshots the "start of today" baseline both portfolio-wide (day_start_value)
        and per-position (day_open_price) -- both measured vs. the LAST KNOWN price at the
        moment this fires (effectively "previous close", since it's called right after a
        midnight date-rollover while the market is closed), matching the standard
        brokerage convention for "Day P/L" (current value vs. previous close, which
        legitimately includes any overnight/pre-market gap as part of "today's" change --
        not vs. today's actual opening trade, which would hide that gap). `today_str`
        (YYYY-MM-DD, ET) is recorded as day_start_date (2026-07-22) so OrderManager.
        _sync_portfolio() can tell whether today's reset already happened and skip
        re-deriving it from Alpaca on an intraday restart -- omitted only by tests that
        don't care about that guard."""
        self.day_start_value = self.total_value
        if today_str is not None:
            self.day_start_date = today_str
        for pos in self.positions.values():
            pos.day_open_price = pos.current_price

    def add_position(self, position: Position):
        self.positions[position.ticker] = position
        self.cash -= position.cost_basis

    def close_position(self, ticker: str) -> float:
        position = self.positions.pop(ticker, None)
        if position is None:
            return 0.0
        self.cash += position.market_value
        self.update_peak()
        return position.unrealized_pnl

    async def add_position_async(self, position: Position):
        self.add_position(position)
        await self._save_position(position)
        await self._save_state()

    async def close_position_async(
        self, ticker: str, *, exit_shares: float = None, exit_price: float = None,
        reason: str = "",
    ) -> float:
        pos = self.positions.get(ticker)
        _exit_shares = exit_shares if exit_shares is not None else (pos.shares if pos else 0)
        _exit_price = exit_price if exit_price is not None else (pos.current_price if pos else 0)
        _trade_id = pos.trade_id if pos else None
        if exit_shares is not None and exit_price is not None:
            # Cash was already credited per-fill in check_take_profits.
            # Pop without crediting market_value to avoid double-counting.
            position = self.positions.pop(ticker, None)
            if position:
                self.update_peak()
            pnl = (_exit_price - position.entry_price) * _exit_shares if position else 0
        else:
            pnl = self.close_position(ticker)
        await self._remove_position_db(ticker)
        await self._save_state()
        if pnl < 0:
            # See self.recent_losses' docstring -- kept in sync with every SELL insert
            # across this file and order_manager.py, not just this one call site.
            self.recent_losses[ticker] = datetime.now()
        if self._db:
            await self._db.execute(
                "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, "SELL", _exit_shares, _exit_price, pnl, datetime.now().isoformat(), reason, _trade_id),
            )
            await self._db.commit()
            # "Recent Sell" post-mortem snapshot (2026-08-21) -- close_position_async is
            # always a FULL close (the position is gone from self.positions either way by
            # this point), so this is exactly the one moment pos's buy_thesis/etc is still
            # available before it's lost for good. Skipped for a ticker with no real
            # buy-time rationale (pre-feature position, or a non-AI-driven signal source
            # like the admin rebuy utility) -- nothing worth a post-mortem without it.
            await self.snapshot_sell_analysis(pos, ticker)
        return pnl

    async def snapshot_sell_analysis(self, pos, ticker: str) -> None:
        """Capture a closing position's buy-side rationale into sell_analysis, the row
        the "Recent Sell" post-mortem queue later drains.

        Extracted from close_position_async (2026-09-01, full-codebase review) because
        it was NOT the only full-close path: _execute_take_profit_tranche pops the
        position directly when a final tranche takes it to zero (deliberately, to avoid
        double-recording trade_history), so a trade that exited entirely through
        take-profits -- the cleanest wins, the ones most worth learning from -- never
        got a post-mortem or fed anything forward into a later analysis of the same
        ticker. Idempotent (INSERT OR IGNORE on trade_id), so calling it from both
        paths can never duplicate a row.

        Skipped for a ticker with no real buy-time rationale (pre-feature position, or
        a non-AI-driven signal source like the admin rebuy utility) -- nothing worth a
        post-mortem without it."""
        if not self._db or not pos or not pos.trade_id or not pos.buy_thesis:
            return
        try:
            await self._db.execute(
                "INSERT OR IGNORE INTO sell_analysis "
                "(trade_id, ticker, buy_thesis, buy_reasoning, buy_conviction, buy_signal, "
                "buy_rr, buy_required_rr, buy_fair_value, entry_price, opened_at, closed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pos.trade_id, ticker, pos.buy_thesis, pos.buy_reasoning,
                    pos.buy_conviction, pos.buy_signal, pos.buy_rr, pos.buy_required_rr,
                    pos.buy_fair_value, pos.entry_price,
                    pos.opened_at.isoformat() if pos.opened_at else None,
                    datetime.now().isoformat(),
                ),
            )
            await self._db.commit()
        except Exception as e:
            # Hot-path write, fail soft (same precedent as the other position writes) --
            # a missing post-mortem must never break a real close.
            logger.warning("Failed to snapshot sell_analysis for %s: %s", ticker, e)

    _SELL_ANALYSIS_COLUMNS = (
        "trade_id", "ticker", "buy_thesis", "buy_reasoning", "buy_conviction", "buy_signal",
        "buy_rr", "buy_required_rr", "buy_fair_value", "entry_price", "exit_price",
        "opened_at", "closed_at",
        "post_mortem_thesis", "post_mortem_reasoning", "generated_at",
        "followup_due_date", "followup_reasoning", "followup_generated_at",
    )

    async def get_pending_sell_analyses(self, limit: int = 5) -> list[dict]:
        """Closed trades whose immediate post-mortem hasn't been generated yet -- the
        queue a periodic job (web/app.py, which owns the real Claude client) drains one
        real API call at a time. Oldest first, same "process a bounded batch per pass"
        pattern as this project's other periodic queues."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE post_mortem_thesis IS NULL "
            "ORDER BY closed_at ASC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def get_sell_analysis(self, trade_id: str) -> dict | None:
        if not self._db:
            return None
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE trade_id = ?", (trade_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) if row else None

    async def save_sell_analysis_post_mortem(
        self, trade_id: str, thesis: str, reasoning: str, followup_due_date: str,
        exit_price: float | None = None,
    ):
        if not self._db:
            return
        await self._db.execute(
            "UPDATE sell_analysis SET post_mortem_thesis = ?, post_mortem_reasoning = ?, "
            "generated_at = ?, followup_due_date = ?, exit_price = ? WHERE trade_id = ?",
            (thesis, reasoning, datetime.now().isoformat(), followup_due_date,
             exit_price, trade_id),
        )
        await self._db.commit()

    async def get_due_sell_analysis_followups(self, today_str: str, limit: int = 5) -> list[dict]:
        """Trades whose delayed follow-up check (price action after the sale) is due --
        has a real post-mortem already, no follow-up yet, and its due date has arrived.
        Same bounded-queue pattern as get_pending_sell_analyses."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE post_mortem_thesis IS NOT NULL "
            "AND followup_generated_at IS NULL AND followup_due_date IS NOT NULL "
            "AND followup_due_date <= ? ORDER BY followup_due_date ASC LIMIT ?",
            (today_str, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def save_sell_analysis_followup(self, trade_id: str, followup_reasoning: str):
        if not self._db:
            return
        await self._db.execute(
            "UPDATE sell_analysis SET followup_reasoning = ?, followup_generated_at = ? "
            "WHERE trade_id = ?",
            (followup_reasoning, datetime.now().isoformat(), trade_id),
        )
        await self._db.commit()

    async def get_recent_sell_analyses(self, ticker: str, limit: int = 2) -> list[dict]:
        """Completed post-mortems for this ticker, newest close first -- feeds
        get_trade_history_summary's minor supplementary prompt context (2026-08-21) so a
        past sell's own AI post-mortem genuinely informs the next time this ticker is
        reconsidered as a buy candidate, not just a passive record. Excludes a still-
        pending row (no post_mortem_thesis yet) -- nothing useful to say yet."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        # rowid DESC tiebreaker (2026-08-31, full-codebase review) -- two closes
        # landing in the same datetime.now() tick (routine on Windows' coarser
        # clock; the exact intermittent failure tests/test_sell_analysis.py's
        # newest-first test showed) produce identical closed_at strings, and bare
        # ORDER BY closed_at DESC then degrades to insertion order -- OLDEST
        # first. A later insert is by construction the later close, so rowid is
        # the correct deterministic tiebreak.
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE ticker = ? AND post_mortem_thesis IS NOT NULL "
            "ORDER BY closed_at DESC, rowid DESC LIMIT ?",
            (ticker, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def save_analysis_history(
        self, ticker: str, generated_at: str, conviction_score: float, signal: str,
        entry_price: float | None, fair_value_estimate: float | None, watch_condition: str,
        sma_relationship: str | None = None, trend_confirms_entry: bool | None = None,
    ) -> None:
        """Appends one row per real analysis (2026-08-21, "Analysis History"
        feed-forward feature) -- called by DashboardState._persist_report (web/app.py)
        for every non-fallback report, fire-and-forget via asyncio.create_task since
        this must never delay or block the real analysis flow it's recording. Never
        overwrites (unlike research_reports/reports_cache.json, which only ever keeps
        the LATEST report per ticker) -- the whole point is preserving the arc across
        repeated analyses of the same recurring candidate.

        sma_relationship/trend_confirms_entry (2026-08-24, SMA Trend-Confirmation
        Track design, ported from AIShortTrading) are only ever non-None for a
        Track 2 candidate -- every existing Track 1 call site omits them, storing
        NULL, same as any pre-migration row.

        Fails soft on a locked database (fixed 2026-08-25, live incident) -- this
        call is already fire-and-forget (see above), so an unhandled
        sqlite3.OperationalError here doesn't block anything else, but it does
        surface as an ugly "Task exception was never retrieved" traceback in the
        logs and silently drops this one ticker's history row for good, with no
        retry. Same fail-soft precedent as watchlist_manager.set_scan_cursor's own
        2026-08-25 fix -- this row is a losable, best-effort feed-forward record
        (see get_analysis_history_summary below), not correctness-critical; the
        next real analysis of this same ticker just won't have this one row's
        worth of history to draw on, a trivial cost against a noisy traceback for
        every real contention event."""
        if not self._db:
            return
        try:
            await self._db.execute(
                "INSERT INTO analysis_history "
                "(ticker, generated_at, conviction_score, signal, entry_price, "
                "fair_value_estimate, watch_condition, sma_relationship, trend_confirms_entry) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, generated_at, conviction_score, signal, entry_price,
                 fair_value_estimate, watch_condition, sma_relationship,
                 None if trend_confirms_entry is None else int(trend_confirms_entry)),
            )
            await self._db.commit()
        except sqlite3.OperationalError as e:
            logger.warning(
                "save_analysis_history(%s): database still locked after the "
                "connection's own timeout -- skipping this history row rather than "
                "raising into the fire-and-forget task: %s", ticker, e,
            )

    async def get_analysis_history_summary(self, ticker: str) -> str:
        """Every past real analysis for this ticker, formatted as prompt context for
        the NEXT analysis of the same ticker (2026-08-21; extended 2026-08-24 to
        carry SMA Trend-Confirmation Track columns) -- see
        _format_analysis_history_summary's own docstring for the full design and why
        this deliberately includes ALL history, not a capped window. Fails open
        (returns "") on any DB error or if the DB hasn't been initialized yet, same as
        every other nice-to-have query in this file -- this must never delay or break a
        real buy decision."""
        if not self._db:
            return ""
        try:
            async with self._db.execute(
                "SELECT generated_at, conviction_score, signal, entry_price, "
                "fair_value_estimate, watch_condition, sma_relationship, "
                "trend_confirms_entry FROM analysis_history "
                "WHERE ticker = ? ORDER BY generated_at",
                (ticker,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_analysis_history_summary failed for %s: %s", ticker, e)
            return ""
        return _format_analysis_history_summary(ticker, rows)
