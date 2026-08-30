"""Dynamic watchlist — tracks 50 active stocks, evicts weak performers, pulls replacements from universe."""

import asyncio
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

WEAK_SIGNALS = {"HOLD", "SELL", "STRONG SELL", "NO ACTION"}


class WatchlistManager:
    def __init__(self, db_path: str, target_size: int = 50, weak_threshold: int = 3):
        self.db_path = db_path
        self.target_size = target_size
        self.weak_threshold = weak_threshold
        self._init_db()

    def _connect(self):
        # check_same_thread=False (2026-07-24, GitHub #25): every method already opens,
        # uses, and discards its own connection within one synchronous call -- never
        # shared/cached across calls -- so this isn't fixing an active bug (confirmed via
        # a test calling through asyncio.to_thread, which already worked fine). Purely
        # defensive: closes off the failure mode if a future caller ever starts reusing a
        # connection object across threads.
        #
        # timeout=20.0 (2026-08-19, live incident) -- this exact connection's
        # set_scan_cursor() commit is what crashed a real pre-open batch run with
        # sqlite3.OperationalError: database is locked, after only the implicit 5.0s
        # default retry window. data/aitrading.db is shared across several independent
        # async loops (position updates, ai_log persistence, trade history) with no
        # write coordination between them, so a longer retry window gives real
        # contention a chance to clear instead of failing fast. Matches web/app.py's
        # own _SQLITE_TIMEOUT_SECS constant -- not imported directly since this module
        # has no dependency on web/app.py (the reverse would be true), so the value is
        # duplicated here with this comment rather than adding that coupling for one
        # constant.
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=20.0)

    def _init_db(self):
        # Stays synchronous -- runs once from __init__ (which can't be async in Python
        # anyway) before the live event loop is servicing any real trading traffic, so
        # none of the "blocks position_update_loop" risk this file's other methods were
        # fixed for (GitHub #102) applies here.
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    added_date TEXT,
                    consecutive_weak_signals INTEGER DEFAULT 0,
                    last_signal TEXT,
                    last_scanned TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    signal TEXT,
                    conviction_score INTEGER,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit_targets TEXT,
                    screened_at TEXT,
                    batch_id TEXT
                )
            """)
            conn.commit()

    # ── Candidates (batch pre-screened stocks) ─────────────────────────────
    #
    # Every public method below is a thin async wrapper dispatching its actual
    # blocking DB work to a worker thread via asyncio.to_thread (fixed 2026-08-30,
    # GitHub #102) -- every one of these used to be a plain synchronous method doing
    # blocking sqlite3.connect()...execute()...commit() directly on the caller's own
    # coroutine. data/aitrading.db is shared with several other independent async
    # loops (position_update_loop's stop-loss/trailing-stop/protection-gap checks,
    # ai_log persistence, trade history, BenchmarkStore), so any real write
    # contention used to block the ENTIRE process event loop for the duration of
    # the call -- confirmed live for set_scan_cursor specifically (see its own
    # docstring below), which is called once per scanned ticker inside the
    # pre-open/mid-day-rescan hot loops, potentially hundreds of times per run.
    # The real DB logic itself is unchanged -- moved verbatim into a same-named
    # `_..._sync` private method, called via asyncio.to_thread instead of directly.
    #
    # get_last_signals is the one deliberate exception, left fully synchronous --
    # its only real caller is DashboardState.__init__ (a one-time startup event
    # before the live event loop is servicing any real trading traffic), and
    # Python constructors can't be async at all.

    async def add_candidate(self, ticker: str, company_name: str, signal: str,
                      conviction: int, entry_price: float, stop_loss: float,
                      take_profit_targets: list, batch_id: str = ""):
        await asyncio.to_thread(
            self._add_candidate_sync, ticker, company_name, signal,
            conviction, entry_price, stop_loss, take_profit_targets, batch_id)

    def _add_candidate_sync(self, ticker: str, company_name: str, signal: str,
                      conviction: int, entry_price: float, stop_loss: float,
                      take_profit_targets: list, batch_id: str = ""):
        import json as _json
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candidates
                (ticker, company_name, signal, conviction_score, entry_price,
                 stop_loss, take_profit_targets, screened_at, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker, company_name, signal, conviction, entry_price,
                  stop_loss, _json.dumps(take_profit_targets), now, batch_id))
            conn.commit()

    async def get_candidates(self, limit: int = 20, exclude: set | None = None) -> list[dict]:
        """Return top candidates sorted by conviction then risk/reward ratio."""
        exclude = exclude or set()
        exclude |= await self.get_active_tickers()
        return await asyncio.to_thread(self._get_candidates_sync, limit, exclude)

    def _get_candidates_sync(self, limit: int, exclude: set) -> list[dict]:
        import json as _json
        with self._connect() as conn:
            if exclude:
                placeholders = ",".join("?" * len(exclude))
                rows = conn.execute(f"""
                    SELECT ticker, company_name, signal, conviction_score,
                           entry_price, stop_loss, take_profit_targets, screened_at
                    FROM candidates
                    WHERE ticker NOT IN ({placeholders})
                """, tuple(exclude)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT ticker, company_name, signal, conviction_score,
                           entry_price, stop_loss, take_profit_targets, screened_at
                    FROM candidates
                """).fetchall()

        results = []
        for r in rows:
            targets = _json.loads(r[6]) if r[6] else []
            entry, stop = r[4], r[5]
            t3 = targets[2] if len(targets) >= 3 else (targets[-1] if targets else 0)
            risk = entry - stop
            rr = (t3 - entry) / risk if risk > 0 else 0.0
            results.append({
                "ticker": r[0], "company_name": r[1], "signal": r[2],
                "conviction_score": r[3], "entry_price": entry,
                "stop_loss": stop, "take_profit_targets": targets,
                "screened_at": r[7], "rr_ratio": round(rr, 2),
            })

        results.sort(key=lambda x: (-x["conviction_score"], -x["rr_ratio"]))
        return results[:limit]

    async def get_stock_summary(self, ticker: str) -> dict | None:
        """Return whatever analysis data we have for a ticker from candidates + watchlist tables."""
        return await asyncio.to_thread(self._get_stock_summary_sync, ticker)

    def _get_stock_summary_sync(self, ticker: str) -> dict | None:
        import json as _json
        with self._connect() as conn:
            row = conn.execute("""
                SELECT ticker, company_name, signal, conviction_score,
                       entry_price, stop_loss, take_profit_targets, screened_at
                FROM candidates WHERE ticker = ?
            """, (ticker,)).fetchone()
            if row:
                targets = _json.loads(row[6]) if row[6] else []
                return {
                    "ticker": row[0], "company_name": row[1], "signal": row[2],
                    "conviction": row[3], "entry_price": row[4], "stop_loss": row[5],
                    "take_profit_targets": targets, "generated_at": row[7],
                    "source": "quick_scan",
                    "thesis": None, "risk_level": None,
                    "fundamental_summary": None, "insider_summary": None,
                    "news_summary": None, "competitive_summary": None, "risk_factors": None,
                }
            wl = conn.execute(
                "SELECT ticker, name, last_signal FROM watchlist WHERE ticker = ?", (ticker,)
            ).fetchone()
            if wl:
                return {
                    "ticker": wl[0], "company_name": wl[1], "signal": wl[2],
                    "conviction": None, "entry_price": 0, "stop_loss": 0,
                    "take_profit_targets": [], "generated_at": None,
                    "source": "watchlist_only",
                    "thesis": None, "risk_level": None,
                    "fundamental_summary": None, "insider_summary": None,
                    "news_summary": None, "competitive_summary": None, "risk_factors": None,
                }
        return None

    async def remove_candidate(self, ticker: str):
        await asyncio.to_thread(self._remove_candidate_sync, ticker)

    def _remove_candidate_sync(self, ticker: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM candidates WHERE ticker = ?", (ticker,))
            conn.commit()

    def get_last_signals(self) -> dict[str, str]:
        """Return {ticker: last_signal} for all watchlist stocks that have been scanned.

        Deliberately left synchronous (2026-08-30, GitHub #102) -- its only real
        caller is DashboardState.__init__, a one-time startup read before the live
        event loop is servicing any real trading traffic, and Python constructors
        can't be async at all."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, last_signal FROM watchlist WHERE last_signal IS NOT NULL"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    async def get_scan_cursor(self) -> int:
        """Position in the universe list where the next replacement scan should resume."""
        return await asyncio.to_thread(self._get_scan_cursor_sync)

    def _get_scan_cursor_sync(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM scan_state WHERE key = 'universe_cursor'"
            ).fetchone()
        return int(row[0]) if row else 0

    async def set_scan_cursor(self, index: int):
        """Fails soft on a locked database (fixed 2026-08-25, live incident) --
        called once per scanned ticker during a pre-open batch (potentially hundreds
        of calls in one run), racing every other loop that also writes to this same
        shared database (position_update_loop, ai_log persistence, trade history,
        BenchmarkStore -- see the 2026-08-19 SQLite Concurrency Hardening note). The
        2026-08-19 fix (WAL mode + this connection's own 20.0s timeout) is real and
        confirmed still active, but a real pre-open batch run hit contention that
        outlasted even that generous window and raised sqlite3.OperationalError --
        which, being unhandled, killed the ENTIRE remaining batch scan outright (an
        unhandled asyncio Task exception), not just this one cursor update. The
        cursor is a pure "resume from here next time" convenience -- losing one
        update in the rare case of a still-locked database after 20s is a trivial,
        self-healing cost (the next pre-open batch just resumes from a slightly
        stale position) compared to abandoning an entire in-progress scan that
        already paid for real Claude analysis on however many tickers it had
        gotten through.

        Now dispatched via asyncio.to_thread (fixed 2026-08-30, GitHub #102) -- the
        try/except above only ever stopped a locked database from CRASHING the
        caller; it did nothing about the BLOCKING WAIT itself, which — being called
        once per scanned ticker in a 1000+-ticker pre-open batch — could freeze the
        entire process event loop (including position_update_loop's own real-time
        stop-loss/trailing-stop checks) for up to 20s per call under real
        contention."""
        await asyncio.to_thread(self._set_scan_cursor_sync, index)

    def _set_scan_cursor_sync(self, index: int):
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO scan_state (key, value) VALUES ('universe_cursor', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(index),),
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(
                "set_scan_cursor(%d): database still locked after the connection's "
                "own timeout -- skipping this cursor update rather than crashing "
                "the caller: %s", index, e,
            )

    async def size(self) -> int:
        return await asyncio.to_thread(self._size_sync)

    def _size_sync(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]

    async def get_active(self) -> list[dict]:
        return await asyncio.to_thread(self._get_active_sync)

    def _get_active_sync(self) -> list[dict]:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=48)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, name, sector, added_date, last_signal FROM watchlist ORDER BY ticker"
            ).fetchall()
        results = []
        for r in rows:
            is_new = False
            if r[3] and not r[4]:
                try:
                    added = datetime.fromisoformat(r[3])
                    is_new = added >= cutoff
                except (ValueError, TypeError):
                    pass
            results.append({"ticker": r[0], "name": r[1], "sector": r[2], "is_new": is_new})
        return results

    async def get_active_tickers(self) -> set[str]:
        return await asyncio.to_thread(self._get_active_tickers_sync)

    def _get_active_tickers_sync(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT ticker FROM watchlist").fetchall()
        return {r[0] for r in rows}

    async def update_signal(self, ticker: str, signal: str):
        """Increment or reset the consecutive-weak-signal counter after each scan."""
        await asyncio.to_thread(self._update_signal_sync, ticker, signal)

    def _update_signal_sync(self, ticker: str, signal: str):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT consecutive_weak_signals FROM watchlist WHERE ticker = ?", (ticker,)
            ).fetchone()
            if not row:
                return
            count = (row[0] + 1) if signal in WEAK_SIGNALS else 0
            conn.execute(
                "UPDATE watchlist SET consecutive_weak_signals=?, last_signal=?, last_scanned=? WHERE ticker=?",
                (count, signal, now, ticker),
            )
            conn.commit()
        if count >= self.weak_threshold:
            logger.info("%s flagged as underperformer (%d consecutive weak signals)", ticker, count)

    async def get_underperformers(self) -> list[str]:
        """Tickers that have hit the weak-signal eviction threshold."""
        return await asyncio.to_thread(self._get_underperformers_sync)

    def _get_underperformers_sync(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker FROM watchlist WHERE consecutive_weak_signals >= ?",
                (self.weak_threshold,),
            ).fetchall()
        return [r[0] for r in rows]

    async def remove(self, ticker: str):
        await asyncio.to_thread(self._remove_sync, ticker)

    def _remove_sync(self, ticker: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
            conn.commit()
        logger.info("Evicted %s from watchlist", ticker)

    async def add(self, ticker: str, name: str, sector: str):
        await asyncio.to_thread(self._add_sync, ticker, name, sector)

    def _add_sync(self, ticker: str, name: str, sector: str):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO watchlist
                   (ticker, name, sector, added_date, consecutive_weak_signals)
                   VALUES (?, ?, ?, ?, 0)""",
                (ticker, name, sector, now),
            )
            conn.commit()
        logger.info("Added %s (%s) to watchlist", ticker, name)

    async def slots_available(self) -> int:
        return max(0, self.target_size - await self.size())

    async def available_from_universe(self, universe: list[str]) -> list[str]:
        """Universe tickers not currently in the watchlist, starting from the saved
        scan cursor and wrapping around — so repeated scans cycle through the full
        universe before repeating, instead of always restarting at index 0."""
        if not universe:
            return []
        current = await self.get_active_tickers()
        cursor = (await self.get_scan_cursor()) % len(universe)
        rotated = universe[cursor:] + universe[:cursor]
        return [t for t in rotated if t not in current]
