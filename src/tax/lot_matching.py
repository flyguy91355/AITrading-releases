"""FIFO tax-lot reconstruction from a chronological trade event list."""

from collections import defaultdict, deque
from dataclasses import dataclass

from src.tax.trade_log_reader import TaxTradeEvent


@dataclass
class ClosedLot:
    ticker: str
    shares: float
    acquired_date: str
    sold_date: str
    cost_basis: float
    proceeds: float
    gain_loss: float


@dataclass
class UnmatchedSell:
    ticker: str
    shares: float
    sold_date: str
    price: float


@dataclass
class _OpenLot:
    shares: float
    acquired_date: str
    price: float


def match_lots(events: list[TaxTradeEvent]) -> tuple[list[ClosedLot], list[UnmatchedSell]]:
    """Replays events in the order given (callers must pass them already
    sorted chronologically -- read_tax_events() guarantees this) and
    FIFO-matches each ticker's sells against its own oldest open buys. A
    sell that exceeds every open lot's remaining shares for that ticker
    closes what it can and reports the unmatched remainder separately,
    rather than raising or fabricating a lot that was never bought."""
    open_lots: dict[str, deque[_OpenLot]] = defaultdict(deque)
    closed: list[ClosedLot] = []
    unmatched: list[UnmatchedSell] = []

    for event in events:
        if event.is_buy:
            open_lots[event.ticker].append(
                _OpenLot(shares=event.shares, acquired_date=event.timestamp, price=event.price)
            )
            continue

        remaining_to_sell = event.shares
        queue = open_lots[event.ticker]
        while remaining_to_sell > 1e-9 and queue:
            lot = queue[0]
            consumed = min(lot.shares, remaining_to_sell)
            closed.append(ClosedLot(
                ticker=event.ticker,
                shares=consumed,
                acquired_date=lot.acquired_date,
                sold_date=event.timestamp,
                cost_basis=consumed * lot.price,
                proceeds=consumed * event.price,
                gain_loss=consumed * (event.price - lot.price),
            ))
            lot.shares -= consumed
            remaining_to_sell -= consumed
            if lot.shares <= 1e-9:
                queue.popleft()

        if remaining_to_sell > 1e-9:
            unmatched.append(UnmatchedSell(
                ticker=event.ticker, shares=remaining_to_sell,
                sold_date=event.timestamp, price=event.price,
            ))

    return closed, unmatched
