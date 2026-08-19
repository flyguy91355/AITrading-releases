"""SQLite-backed storage for the composition-weighted benchmark (2026-07-29):
the ticker classification cache and the settled daily benchmark history.
Plain synchronous sqlite3, matching src/utils/watchlist_manager.py's
existing pattern for auxiliary/infrequent DB access -- not aiosqlite, which
is reserved for Portfolio's hot-path connection."""

import json
import sqlite3

from src.analytics.composition_benchmark import sector_to_etf, cap_tier_to_etf


class BenchmarkStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        # timeout=20.0 (2026-08-19, cross-module hardening after a real
        # sqlite3.OperationalError: database is locked crash in the sibling
        # watchlist_manager.py connection -- see that module's matching comment) --
        # longer retry window against the same shared, multi-writer data/aitrading.db
        # file than the implicit 5.0s default.
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=20.0)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_classification (
                    ticker TEXT PRIMARY KEY,
                    sector TEXT,
                    sector_etf TEXT,
                    cap_tier TEXT,
                    cap_tier_etf TEXT,
                    classified_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_composition_history (
                    date TEXT PRIMARY KEY,
                    benchmark_index_value REAL,
                    composition_json TEXT
                )
            """)
            conn.commit()
            # is_provisional (2026-07-29) -- a day computed while any held
            # ticker/ETF's real same-day close wasn't published yet (see
            # has_real_close) is only a placeholder, not real settled
            # history. Added after benchmark_composition_history already
            # existed and had real rows (both locally and on Hetzner), so
            # this must be a migration, not just part of the CREATE TABLE --
            # same ALTER TABLE pattern src/decision/portfolio.py already
            # uses for its own post-hoc columns. Existing rows default to 0
            # (not provisional) since they were computed before this
            # concept existed and were the best information available then.
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(benchmark_composition_history)").fetchall()}
            if "is_provisional" not in cols:
                conn.execute(
                    "ALTER TABLE benchmark_composition_history "
                    "ADD COLUMN is_provisional INTEGER NOT NULL DEFAULT 0")
                conn.commit()

    def get_cached_classification(self, ticker: str) -> tuple[str | None, str] | None:
        """Returns (sector_etf_or_None, cap_tier_etf) if this ticker has
        already been classified, else None (caller must classify fresh)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sector_etf, cap_tier_etf FROM ticker_classification WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        return (row[0], row[1])

    def save_classification(
        self, ticker: str, sector: str | None, sector_etf: str | None,
        cap_tier: str, cap_tier_etf: str,
    ) -> None:
        from datetime import datetime, timezone
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ticker_classification
                   (ticker, sector, sector_etf, cap_tier, cap_tier_etf, classified_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, sector, sector_etf, cap_tier, cap_tier_etf,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def get_settled_days(self) -> dict[str, float]:
        """Returns {date: benchmark_index_value} for every already-settled
        (backfilled or previously-captured) day."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, benchmark_index_value FROM benchmark_composition_history"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def save_settled_day(
        self, date: str, index_value: float, composition: dict, is_provisional: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_composition_history
                   (date, benchmark_index_value, composition_json, is_provisional)
                   VALUES (?, ?, ?, ?)""",
                (date, index_value, json.dumps(composition), int(is_provisional)),
            )
            conn.commit()

    def get_provisional_dates(self) -> set[str]:
        """Dates whose settled value was computed from carried-forward (not
        real same-day) data -- the backfill script recomputes these every
        run until real data is published, instead of freezing a lucky-timing
        gap in as permanent history."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date FROM benchmark_composition_history WHERE is_provisional = 1"
            ).fetchall()
        return {row[0] for row in rows}


def classify_ticker(ticker: str, store: "BenchmarkStore", get_sector, get_cap_tier_membership) -> tuple[str | None, str]:
    """Returns (sector_etf_or_None, cap_tier_etf) for a ticker, using the
    cache when available and only falling through to the real (slower)
    lookups -- get_sector(ticker) -> str | None and
    get_cap_tier_membership() -> (sp500_set, sp400_set, sp600_set) -- on a
    genuine cache miss. Always persists the result afterward so a given
    ticker is only ever looked up live once."""
    cached = store.get_cached_classification(ticker)
    if cached is not None:
        return cached

    sector = get_sector(ticker)
    sector_etf = sector_to_etf(sector)
    sp500, sp400, sp600 = get_cap_tier_membership()
    cap_tier_etf = cap_tier_to_etf(ticker, sp500, sp400, sp600)
    cap_tier_label = "S&P 500" if cap_tier_etf == "SPY" else ("S&P 400" if cap_tier_etf == "MDY" else "S&P 600")

    store.save_classification(ticker, sector, sector_etf, cap_tier_label, cap_tier_etf)
    return (sector_etf, cap_tier_etf)
