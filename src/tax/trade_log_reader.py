"""Reads data/trade_history/*.jsonl for tax-lot reconstruction. A separate
reader from src/analytics/composition_benchmark.py's parse_trade_events() --
that function only extracts ticker/is_buy/shares/timestamp (enough to
reconstruct daily portfolio holdings), not price, trade_id, or is_paper,
which this feature needs for cost basis and the paper-exclusion guarantee.
Same underlying data source and BUY/SELL detection convention, different
extraction. See docs/superpowers/specs/2026-08-02-tax-lot-tracking-design.md."""

import json
import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class TaxTradeEvent(NamedTuple):
    ticker: str
    is_buy: bool
    shares: float
    price: float
    timestamp: str  # ISO 8601 string, sortable lexicographically as-is
    trade_id: str | None
    is_paper: bool


def read_tax_events(jsonl_dir: Path, since: str | None = None) -> list[TaxTradeEvent]:
    """Reads every *.jsonl file in jsonl_dir and returns all buy+sell events,
    sorted chronologically. A signal is a buy if "BUY" appears in it, a sell
    if "SELL" appears in it -- anything else (e.g. "HOLD") is skipped, not
    raised on. Any event missing "is_paper" defaults to True (paper,
    excluded) -- fail-safe default, never assumed live. since (optional ISO
    date/datetime string, inclusive) drops any event with an earlier
    timestamp."""
    events: list[TaxTradeEvent] = []
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
                    "read_tax_events: unrecognized signal %r for %s in %s -- skipping",
                    signal, row.get("ticker"), path.name,
                )
                continue
            if since is not None and row["timestamp"] < since:
                continue
            events.append(TaxTradeEvent(
                ticker=row["ticker"],
                is_buy=is_buy,
                shares=row["shares"],
                price=row.get("entry_price", 0.0),
                timestamp=row["timestamp"],
                trade_id=row.get("trade_id"),
                is_paper=row.get("is_paper", True),
            ))
    events.sort(key=lambda e: e.timestamp)
    return events
