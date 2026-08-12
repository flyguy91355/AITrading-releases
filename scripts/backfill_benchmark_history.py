"""One-time backfill for the composition-weighted benchmark
(docs/superpowers/specs/2026-07-29-composition-weighted-benchmark-design.md).

Run manually: python scripts/backfill_benchmark_history.py

Reconstructs every settled trading day since the earliest performance_history
entry, computing what a sector+cap-tier-composition-matched passive
benchmark would have returned each day given what was ACTUALLY held that
day. Idempotent -- re-running skips any date already in
benchmark_composition_history, so a partial failure never corrupts already-
good days.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.analytics.composition_benchmark import (
    parse_trade_events, reconstruct_daily_holdings, weighted_daily_return, compound_index,
    carry_forward_price, seed_last_known_prices, has_real_close,
)
from src.analytics.benchmark_store import BenchmarkStore, classify_ticker
from src.data.market_data import MarketDataFetcher
from src.data.stock_universe import get_universe

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"]
CAP_TIER_ETFS = ["SPY", "MDY", "IJR"]  # IJR, not SLY -- see cap_tier_to_etf's docstring
ALL_ETFS = SECTOR_ETFS + CAP_TIER_ETFS

# Mirrors web/app.py's _LIVE_ACCOUNT_START -- this script runs standalone
# and can't import that module (DashboardState builds real broker/market
# connections at import time), so the value is duplicated here rather than
# imported. data/trade_history/*.jsonl (both local and Hetzner) contains
# ~3 weeks of leftover pre-migration dev/test pollution before this date
# (confirmed live 2026-07-29) -- see parse_trade_events' `since` docstring.
_LIVE_ACCOUNT_START = "2026-07-12"


def load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_performance_history_dates(project_root: Path) -> list[str]:
    import json
    path = project_root / "data" / "performance_history.json"
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    return sorted(e["date"] for e in entries)


async def fetch_etf_closes(market_data: MarketDataFetcher, period: str) -> dict[str, dict[str, float]]:
    """Returns {etf: {date: close}} for every ETF this benchmark needs."""
    result = {}
    for etf in ALL_ETFS:
        history = await market_data.get_historical(etf, period=period, interval="1d")
        result[etf] = {row["date"]: row["close"] for row in history}
    return result


async def fetch_ticker_closes(market_data: MarketDataFetcher, tickers: set[str], period: str) -> dict[str, dict[str, float]]:
    result = {}
    for ticker in tickers:
        try:
            history = await market_data.get_historical(ticker, period=period, interval="1d")
            result[ticker] = {row["date"]: row["close"] for row in history}
        except Exception as e:
            print(f"WARNING: failed to fetch history for {ticker}: {e} -- will carry-forward last known price")
            result[ticker] = {}
    return result


async def main():
    config = load_config()
    project_root = Path(__file__).resolve().parent.parent
    db_path = config.get("database", {}).get("path", "data/aitrading.db")

    store = BenchmarkStore(db_path)
    store.initialize()

    dates = load_performance_history_dates(project_root)
    if not dates:
        print("No performance_history.json found (or it's empty) -- nothing to backfill yet.")
        return
    # A provisional day (computed while some held ticker/ETF's real same-day
    # close wasn't published yet) is reprocessed on every run alongside
    # brand-new dates, not skipped -- self-healing once real data shows up,
    # rather than permanently freezing a lucky-timing gap as final history.
    provisional = store.get_provisional_dates()
    already_settled_final = set(store.get_settled_days().keys()) - provisional
    dates_to_process = [d for d in dates if d not in already_settled_final]
    if not dates_to_process:
        print("Nothing to backfill -- every performance_history date is already settled.")
        return
    reprocessing = [d for d in dates_to_process if d in provisional]
    if reprocessing:
        print(f"Reprocessing {len(reprocessing)} previously-provisional day(s): {reprocessing}")
    print(f"Backfilling {len(dates_to_process)} day(s): {dates_to_process[0]} to {dates_to_process[-1]}")

    events = parse_trade_events(project_root / "data" / "trade_history", since=_LIVE_ACCOUNT_START)
    daily_holdings = reconstruct_daily_holdings(events, dates)

    all_held_tickers = {t for day in daily_holdings.values() for t in day}

    market_data = MarketDataFetcher(config)
    # +30d safety margin past the oldest date we need, in yfinance's period-string format.
    from datetime import date as _date
    days_span = (_date.today() - _date.fromisoformat(dates[0])).days + 30
    period = f"{days_span}d"

    etf_closes = await fetch_etf_closes(market_data, period)
    ticker_closes = await fetch_ticker_closes(market_data, all_held_tickers, period)

    sp500 = set(get_universe(["S&P 500"]))
    sp400 = set(get_universe(["S&P 400"]))
    sp600 = set(get_universe(["S&P 600"]))

    def get_sector(ticker: str):
        import yfinance as yf
        try:
            return yf.Ticker(ticker).info.get("sector")
        except Exception:
            return None

    def get_cap_tier_membership():
        return (sp500, sp400, sp600)

    classifications = {
        t: classify_ticker(t, store, get_sector, get_cap_tier_membership)
        for t in all_held_tickers
    }

    # Seed from the last already-settled day if one exists, else start fresh at 100.0.
    settled = store.get_settled_days()
    prior_dates = [d for d in sorted(settled.keys()) if d < dates_to_process[0]]
    seed = settled[prior_dates[-1]] if prior_dates else 100.0

    # Seeded from already-fetched history strictly before this run's first
    # processed date (2026-07-29 fix) -- an incremental re-run (only
    # reprocessing newly-added dates) would otherwise start these caches
    # empty, unable to carry forward from real prior data it already has
    # just because that prior date isn't being reprocessed in THIS run.
    first_date = dates_to_process[0]
    last_known_price: dict[str, float] = seed_last_known_prices(ticker_closes, first_date)
    last_known_etf: dict[str, float] = seed_last_known_prices(etf_closes, first_date)
    prev_etf_close: dict[str, float] = dict(last_known_etf)

    # Pass 1: compute each day's own return and composition, in order, without
    # yet knowing the running index value -- keeps this pass's per-day work
    # (price carry-forward, classification lookup) separate from the actual
    # compounding step below.
    day_returns: list[float] = []
    day_compositions: list[dict] = []
    day_holding_counts: list[int] = []
    day_provisional: list[bool] = []
    for d in dates_to_process:
        holdings_value = {}
        day_is_provisional = False
        for ticker, shares in daily_holdings[d].items():
            price = carry_forward_price(ticker, d, ticker_closes, last_known_price)
            if price is not None:
                holdings_value[ticker] = shares * price
            if not has_real_close(ticker, d, ticker_closes):
                day_is_provisional = True

        etf_daily_returns = {}
        for etf in ALL_ETFS:
            close = carry_forward_price(etf, d, etf_closes, last_known_etf)
            prev = prev_etf_close.get(etf)
            etf_daily_returns[etf] = (close / prev - 1) if (prev and close) else 0.0
            if close is not None:
                prev_etf_close[etf] = close
            if not has_real_close(etf, d, etf_closes):
                day_is_provisional = True

        if holdings_value:
            day_returns.append(weighted_daily_return(holdings_value, classifications, etf_daily_returns))
            total = sum(holdings_value.values())
            day_compositions.append({t: v / total for t, v in holdings_value.items()})
        else:
            day_returns.append(0.0)  # no positions held this day -- benchmark holds flat
            day_compositions.append({})
        day_holding_counts.append(len(holdings_value))
        day_provisional.append(day_is_provisional)

    # Pass 2: compound_index does the actual chain-compounding in one shot;
    # index_values[0] is the seed itself, so index_values[i+1] is the value
    # AFTER dates_to_process[i]'s return -- zip against index_values[1:].
    index_values = compound_index(seed, day_returns)
    for d, value, composition, count, is_prov in zip(
        dates_to_process, index_values[1:], day_compositions, day_holding_counts, day_provisional,
    ):
        store.save_settled_day(d, value, composition, is_provisional=is_prov)
        flag = " [PROVISIONAL -- will retry next run]" if is_prov else ""
        print(f"{d}: benchmark={value:.4f} ({count} tickers held){flag}")

    print("Backfill complete.")


if __name__ == "__main__":
    asyncio.run(main())
