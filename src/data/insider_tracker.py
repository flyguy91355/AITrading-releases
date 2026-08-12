"""SEC Form 4 and insider transaction monitoring via Finnhub."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    BUY = "buy"
    SELL = "sell"
    OPTION_EXERCISE = "option_exercise"


@dataclass
class InsiderTransaction:
    ticker: str
    insider_name: str
    title: str
    transaction_type: TransactionType
    shares: int
    price: float
    total_value: float
    shares_held_after: int
    filing_date: datetime
    transaction_date: datetime
    is_10b5_1: bool = False


@dataclass
class InsiderSummary:
    ticker: str
    buy_count_90d: int = 0
    sell_count_90d: int = 0
    net_shares_90d: int = 0
    cluster_buying: bool = False
    notable_transactions: list[InsiderTransaction] = None

    def __post_init__(self):
        if self.notable_transactions is None:
            self.notable_transactions = []

    @property
    def buy_sell_ratio(self) -> float:
        if self.sell_count_90d == 0:
            return float(self.buy_count_90d) if self.buy_count_90d > 0 else 0.0
        return self.buy_count_90d / self.sell_count_90d


class InsiderTracker:
    def __init__(self, config: dict):
        self.config = config
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        # Finnhub's free tier is rate-limited to ~60 calls/min. Capping *concurrency*
        # (e.g. a semaphore of 5) was found live on 2026-07-15/16 to still let a burst
        # of near-simultaneous requests through — 5 concurrent tickers all reaching
        # this call within the same ~100ms window is a burst of 50 req/sec instantaneous,
        # which triggered cascading 429s and silently dropped candidates from batch
        # analysis (see CLAUDE.md, "Finnhub rate-limit cascade"). This lock + timestamp
        # pacer enforces a real minimum interval between successive Finnhub calls
        # regardless of how many concurrent callers there are — the actual constraint
        # (Finnhub's rate limit), not an approximation of it via caller concurrency.
        self._finnhub_lock = asyncio.Lock()
        self._finnhub_last_call: float = 0.0
        self._finnhub_min_interval = 1.1  # seconds — ~54 calls/min, under the 60/min limit

    async def _finnhub_throttle(self) -> None:
        async with self._finnhub_lock:
            wait = self._finnhub_min_interval - (time.monotonic() - self._finnhub_last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._finnhub_last_call = time.monotonic()

    async def get_recent_transactions(self, ticker: str, days: int = 90) -> list[InsiderTransaction]:
        if not self.finnhub_key:
            logger.warning("No FINNHUB_API_KEY set — returning empty insider transactions")
            return []

        cutoff = datetime.now() - timedelta(days=days)

        await self._finnhub_throttle()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/stock/insider-transactions",
                params={"symbol": ticker, "token": self.finnhub_key},
            )
            resp.raise_for_status()
            data = resp.json()

        transactions = []
        for item in data.get("data", []):
            try:
                tx_date = datetime.strptime(item.get("transactionDate", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if tx_date < cutoff:
                continue

            try:
                filing_date = datetime.strptime(item.get("filingDate", ""), "%Y-%m-%d")
            except (ValueError, TypeError):
                filing_date = tx_date

            change = item.get("change", 0) or 0
            price = item.get("transactionPrice", 0) or 0

            if change > 0:
                tx_type = TransactionType.BUY
            elif change < 0:
                tx_type = TransactionType.SELL
            else:
                # change=0 means grants, awards, or other non-directional events — skip
                continue

            transactions.append(InsiderTransaction(
                ticker=ticker,
                insider_name=item.get("name", "Unknown"),
                title=item.get("position", ""),
                transaction_type=tx_type,
                shares=abs(int(change)),
                price=float(price),
                total_value=abs(int(change)) * float(price),
                shares_held_after=int(item.get("share", 0) or 0),
                filing_date=filing_date,
                transaction_date=tx_date,
                is_10b5_1=False,
            ))

        transactions.sort(key=lambda t: t.transaction_date, reverse=True)
        return transactions

    async def get_insider_summary(self, ticker: str) -> InsiderSummary:
        transactions = await self.get_recent_transactions(ticker, days=90)

        buys = [t for t in transactions if t.transaction_type == TransactionType.BUY]
        sells = [t for t in transactions if t.transaction_type == TransactionType.SELL]
        net_shares = sum(t.shares for t in buys) - sum(t.shares for t in sells)

        notable = sorted(transactions, key=lambda t: t.total_value, reverse=True)[:5]

        cluster = await self.detect_cluster_buying(ticker, transactions=transactions)

        return InsiderSummary(
            ticker=ticker,
            buy_count_90d=len(buys),
            sell_count_90d=len(sells),
            net_shares_90d=net_shares,
            cluster_buying=cluster,
            notable_transactions=notable,
        )

    async def detect_cluster_buying(
        self, ticker: str, window_days: int = 14, transactions: list[InsiderTransaction] | None = None
    ) -> bool:
        if transactions is None:
            transactions = await self.get_recent_transactions(ticker, days=window_days)

        buys = [t for t in transactions if t.transaction_type == TransactionType.BUY]
        if len(buys) < 3:
            return False

        buys.sort(key=lambda t: t.transaction_date)
        for buy in buys:
            window_start = buy.transaction_date - timedelta(days=window_days)
            cluster_buys = [b for b in buys if window_start <= b.transaction_date <= buy.transaction_date]
            unique_buyers_in_window = {b.insider_name for b in cluster_buys}
            if len(unique_buyers_in_window) >= 3:
                return True

        return False
