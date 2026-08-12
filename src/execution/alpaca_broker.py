"""Alpaca broker implementation for paper and live trading via alpaca-trade-api SDK."""

import asyncio
import logging
import os
import random
import uuid
from datetime import datetime

import alpaca_trade_api as tradeapi

from src.execution.broker import Broker, Order, OrderSide, OrderType, OrderStatus, AccountInfo

logger = logging.getLogger(__name__)

ALPACA_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
    "pending_new": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    # Six more real, documented statuses (per Alpaca's own order-lifecycle docs,
    # confirmed 2026-08-08) that used to fall through to this function's own
    # OrderStatus.PENDING default undocumented -- that default happened to be safe
    # for all six (every one of them is genuinely non-terminal), but "safe by
    # accident of a fallback" isn't the same as verified and explicit. Mapped
    # deliberately rather than left implicit, per the reference-alpaca-api-expertise
    # memory's own standard: every status this system might ever see should be a
    # conscious choice, not an unlabeled default.
    "done_for_day": OrderStatus.SUBMITTED,  # "stops executing until the next trading session" -- still a live order, just paused for today
    "replaced": OrderStatus.SUBMITTED,      # this order's own id is superseded by a new one; a caller polling the OLD id should see it as still "in flight" rather than failed
    "accepted_for_bidding": OrderStatus.SUBMITTED,  # at the exchange, under pricing evaluation -- non-terminal
    "stopped": OrderStatus.SUBMITTED,       # "trade guaranteed but not executed" -- still working
    "suspended": OrderStatus.PENDING,       # "not eligible for trading" -- paused/blocked, not actively working
    "calculated": OrderStatus.PENDING,      # rare, under-documented even by Alpaca itself -- kept conservative (never assumed terminal/filled) rather than guessed
}

# Exponential backoff delays (seconds) for the rate-limit retry below, plus a small random
# jitter added at call time -- overridable in tests via monkeypatch to avoid real sleeps.
_RATE_LIMIT_RETRY_DELAYS = [0.5, 1.0, 2.0]


class AlpacaBroker(Broker):
    def __init__(self, config: dict):
        self.config = config
        self.paper = config["trading"]["paper_trading"]
        self.api: tradeapi.REST | None = None

    async def _call_with_rate_limit_retry(self, func, *args, **kwargs):
        """Runs a blocking Alpaca SDK call via asyncio.to_thread, with bounded
        exponential-backoff retry specifically on HTTP 429 (Alpaca's trading API rate
        limit: 200 requests/minute) -- 2026-07-24. Any other exception, or a 429 that
        persists past the retry budget, propagates unchanged -- every existing caller's
        error handling is preserved exactly."""
        from alpaca_trade_api.rest import APIError

        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except APIError as e:
                if e.status_code != 429 or attempt >= len(_RATE_LIMIT_RETRY_DELAYS):
                    raise
                delay = _RATE_LIMIT_RETRY_DELAYS[attempt] + random.uniform(0, 0.25)
                logger.warning(
                    "Alpaca rate limit hit (429) — retrying in %.2fs (attempt %d/%d)",
                    delay, attempt + 1, len(_RATE_LIMIT_RETRY_DELAYS),
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def connect(self):
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "")

        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env (copy config/credentials.env to .env at the repo root, then add your real keys)")

        if not base_url:
            base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")

        # _call_with_rate_limit_retry, not asyncio.to_thread directly (fixed 2026-08-08,
        # GitHub #56) -- this was the one Alpaca call in the whole class bypassing the
        # same 429 retry every other real call gets. Low real-world severity (connect()
        # fires once per process start, a low-volume moment), but a real, easy-to-close
        # inconsistency.
        account = await self._call_with_rate_limit_retry(self.api.get_account)
        mode = "PAPER" if self.paper else "LIVE"
        logger.info("Connected to Alpaca (%s) — Account: %s, Equity: $%s",
                     mode, account.account_number, account.equity)

    async def disconnect(self):
        self.api = None
        logger.info("Disconnected from Alpaca")

    def create_trade_updates_stream(self):
        """Returns a real-time Alpaca trade_updates WebSocket Stream, or None if
        credentials aren't available. Not part of the abstract Broker interface (same
        precedent as get_closed_orders) -- OrderManager accesses this defensively via
        getattr, so a broker implementation without stream support (e.g.
        RobinhoodBroker) needs no changes.

        Sourced from this project's own ALPACA_API_KEY/ALPACA_SECRET_KEY/ALPACA_BASE_URL
        env vars (2026-07-24) -- NOT the alpaca-trade-api SDK's own default env var
        names, which differ from this project's."""
        from alpaca_trade_api.stream import Stream

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "")
        if not api_key or not secret_key:
            return None
        if not base_url:
            base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        return Stream(key_id=api_key, secret_key=secret_key, base_url=base_url)

    async def get_account(self) -> AccountInfo:
        account = await self._call_with_rate_limit_retry(self.api.get_account)
        return AccountInfo(
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
        )

    async def submit_order(self, order: Order) -> Order:
        side = "buy" if order.side == OrderSide.BUY else "sell"

        # Use notional (dollar amount) for fractional market buys; qty for everything else
        if order.notional_value and order.side == OrderSide.BUY and order.order_type == OrderType.MARKET:
            kwargs = {
                "symbol": order.ticker,
                "notional": str(round(order.notional_value, 2)),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
        else:
            qty_str = str(round(order.quantity, 9)).rstrip("0").rstrip(".") if order.quantity % 1 else str(int(order.quantity))
            if not qty_str or qty_str == ".":
                qty_str = "0"
            kwargs = {
                "symbol": order.ticker,
                "qty": qty_str,
                "side": side,
            }

        if order.notional_value:
            pass  # kwargs already complete (notional market buy built above)

        elif order.order_type == OrderType.MARKET:
            kwargs["type"] = "market"
            kwargs["time_in_force"] = "day"

        elif order.order_type == OrderType.LIMIT:
            kwargs["type"] = "limit"
            # Alpaca requires DAY (not GTC) for fractional quantities
            kwargs["time_in_force"] = "day" if order.quantity % 1 != 0 else "gtc"
            kwargs["limit_price"] = str(round(order.limit_price, 2))

        elif order.order_type == OrderType.STOP:
            kwargs["type"] = "stop"
            # Alpaca requires DAY (not GTC) for fractional quantities
            kwargs["time_in_force"] = "day" if order.quantity % 1 != 0 else "gtc"
            kwargs["stop_price"] = str(round(order.stop_price, 2))

        elif order.order_type == OrderType.STOP_LIMIT:
            kwargs["type"] = "stop_limit"
            kwargs["time_in_force"] = "day" if order.quantity % 1 != 0 else "gtc"
            kwargs["stop_price"] = str(round(order.stop_price, 2))
            kwargs["limit_price"] = str(round(order.limit_price, 2))

        elif order.order_type == OrderType.TRAILING_STOP:
            kwargs["type"] = "trailing_stop"
            kwargs["time_in_force"] = "day" if order.quantity % 1 != 0 else "gtc"
            kwargs["trail_percent"] = str(order.trail_pct)

        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        if not order.client_order_id:
            order.client_order_id = (
                f"{order.ticker}-{order.side.value}-{order.order_type.value}-{uuid.uuid4().hex[:8]}"
            )
        kwargs["client_order_id"] = order.client_order_id

        result = await self._call_with_rate_limit_retry(lambda: self.api.submit_order(**kwargs))

        order.broker_order_id = result.id
        order.status = ALPACA_STATUS_MAP.get(result.status, OrderStatus.PENDING)
        order.submitted_at = datetime.now()

        if result.filled_avg_price is not None:
            order.filled_price = float(result.filled_avg_price)
        if result.filled_qty is not None:
            order.filled_quantity = float(result.filled_qty)
        if hasattr(result, "filled_at") and result.filled_at:
            try:
                order.filled_at = datetime.fromisoformat(str(result.filled_at).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass

        if order.notional_value:
            logger.info("Alpaca order submitted: %s %s $%.2f notional — ID: %s, Status: %s",
                        order.side.value, order.ticker, order.notional_value,
                        order.broker_order_id, order.status.value)
        else:
            logger.info("Alpaca order submitted: %s %s %.4g shares %s — ID: %s, Status: %s",
                        order.side.value, order.ticker, order.quantity, order.order_type.value,
                        order.broker_order_id, order.status.value)
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await self._call_with_rate_limit_retry(self.api.cancel_order, broker_order_id)
            logger.info("Alpaca order cancelled: %s", broker_order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel Alpaca order %s: %s", broker_order_id, e)
            return False

    async def replace_order(self, broker_order_id: str, qty: float | None = None,
                             stop_price: float | None = None, limit_price: float | None = None) -> str | None:
        """Replace an existing order's qty/stop_price/limit_price in place via Alpaca's
        PATCH /v2/orders/{id} (2026-07-28, MET incident follow-up) -- confirmed against
        Alpaca's own docs as the intended way to adjust an already-resting order's price/
        qty, unlike this codebase's existing cancel-then-resubmit pattern, which releases
        the order's reserved shares and re-reserves them, exposing a real settlement-lag
        race (a fresh placement can see qty_available == 0 for a few seconds after the
        cancel is accepted but before Alpaca actually frees the shares -- confirmed live,
        see CLAUDE.md). A same-order replace never releases the shares in between.

        Per Alpaca's docs, replace itself is rejected outright while the order's status
        is accepted/pending_new/pending_cancel/pending_replace -- so this does NOT bypass
        that specific stuck state if an order is already mid-cancel; it only helps avoid
        entering a cancel in the first place for the routine "same order, new price"
        case (see OrderManager._try_replace_stale_stop).

        Returns the NEW order's broker_order_id on success -- Alpaca always creates a new
        order ID on replace; the original transitions to status 'replaced'. Returns None
        on any failure (including a rejected replace) -- callers MUST treat None as
        "fall back to cancel+place", not a fatal error; this method never raises."""
        kwargs: dict = {}
        if qty is not None:
            qty_str = str(round(qty, 9)).rstrip("0").rstrip(".") if qty % 1 else str(int(qty))
            kwargs["qty"] = qty_str or "0"
        if stop_price is not None:
            kwargs["stop_price"] = str(round(stop_price, 2))
        if limit_price is not None:
            kwargs["limit_price"] = str(round(limit_price, 2))
        try:
            result = await self._call_with_rate_limit_retry(
                lambda: self.api.replace_order(broker_order_id, **kwargs))
            logger.info("Alpaca order replaced: %s -> %s", broker_order_id, result.id)
            return result.id
        except Exception as e:
            logger.warning("Failed to replace Alpaca order %s: %s", broker_order_id, e)
            return None

    async def get_order_status(self, broker_order_id: str) -> Order:
        result = await self._call_with_rate_limit_retry(self.api.get_order, broker_order_id)

        side = OrderSide.BUY if result.side == "buy" else OrderSide.SELL
        try:
            order_type = OrderType(str(result.type))
        except ValueError:
            order_type = OrderType.MARKET

        return Order(
            ticker=result.symbol,
            side=side,
            order_type=order_type,
            quantity=float(result.qty) if result.qty else 0.0,
            status=ALPACA_STATUS_MAP.get(result.status, OrderStatus.PENDING),
            filled_price=float(result.filled_avg_price) if result.filled_avg_price is not None else None,
            filled_quantity=float(result.filled_qty) if result.filled_qty is not None else 0.0,
            broker_order_id=result.id,
        )

    async def get_positions(self) -> list[dict]:
        positions = await self._call_with_rate_limit_retry(self.api.list_positions)
        return [
            {
                "ticker": p.symbol,
                "shares": float(p.qty),
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ]

    async def get_position(self, ticker: str) -> dict | None:
        """Single-ticker real-time position lookup (2026-07-29, BAC exit-order race) --
        GET /v2/positions/{symbol}, confirmed via Alpaca's own docs (docs.alpaca.markets):
        `qty` is the true current total; `qty_available` is "total number of shares
        available minus open orders" -- i.e. it already accounts for any resting order
        (stop, TP, or a market sell still settling) placed by ANY code path, not just the
        caller's own bookkeeping. Used to verify a fresh, trustworthy share count
        immediately before computing a new exit-order split, closing the exact gap that
        caused the BAC incident: sync_exit_orders' per-pass open_orders snapshot (fetched
        once, before its per-ticker loop) can't see an order a concurrent _execute_sell
        call placed moments later, so a subsequent placement attempt computed its split
        against a share count that was already partially or fully spoken for.
        Returns None if Alpaca has no open position for this ticker at all (already fully
        closed) -- Alpaca's docs don't specify the exact error for this case, so any
        exception from the lookup is treated as "no real position to protect right now"
        rather than assumed to be some other kind of failure."""
        try:
            p = await self._call_with_rate_limit_retry(self.api.get_position, ticker)
        except Exception:
            return None
        return {
            "ticker": p.symbol,
            "shares": float(p.qty),
            "qty_available": float(getattr(p, "qty_available", p.qty)),
        }

    async def get_open_orders(self) -> list[dict]:
        orders = await self._call_with_rate_limit_retry(lambda: self.api.list_orders(status="open"))
        return [
            {
                "order_id": o.id,
                "ticker": o.symbol,
                "side": o.side,
                "type": o.type,
                # Notional (dollar-amount) orders have qty=None until filled -- fall back to
                # notional so a still-open notional order isn't misread as qty=0/nonexistent.
                "qty": float(o.qty) if o.qty else (float(o.notional) if getattr(o, "notional", None) else 0.0),
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "stop_price": float(o.stop_price) if o.stop_price else None,
            }
            for o in orders
        ]

    async def get_closed_orders(self, symbols: list[str] | None = None, limit: int = 100) -> list[dict]:
        """Return filled/closed sell orders — used to verify apparent position closes.

        symbols (2026-07-24, INSW incident) -- when provided, scopes the query to just
        those tickers instead of the account-wide "most recent 100 orders" window.
        Alpaca sorts that window by submission time, not fill time, so a stop order
        placed early in the day that only fills later can get pushed out of a shared,
        unscoped window by unrelated order churn on OTHER tickers (this system
        cancels/replaces exit orders hourly across every held position) even though it
        filled recently. Callers verifying a specific ticker's apparent close should
        always pass symbols=[ticker, ...] so unrelated account-wide volume can never
        hide the one order that actually matters.

        limit (2026-08-04, NDAQ entry-time incident) -- defaults to the original 100,
        unchanged for every existing caller. A ticker with heavy cancel/replace churn
        (dozens of superseded stop-order replacements from before the 2026-07-28
        in-place-replace fix) can have well over 100 closed orders of its own, pushing
        a much older real fill (e.g. the original buy) out of even a single-ticker-
        scoped window. _infer_real_entry_time passes a larger limit explicitly when it
        needs to look further back than routine close-verification ever does."""
        orders = await self._call_with_rate_limit_retry(
            lambda: self.api.list_orders(
                status="closed", limit=limit, direction="desc",
                symbols=symbols if symbols else None,
            )
        )
        return [
            {
                "order_id": o.id, "symbol": o.symbol, "side": o.side, "status": o.status,
                # Added 2026-07-24 (untracked-TP-fill accuracy fix) -- lets a caller
                # reconstruct the REAL fill (price, quantity, order type, timing) instead
                # of guessing from popped target prices alone.
                "order_type": o.type,
                "qty": float(o.qty) if o.qty else 0.0,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            }
            for o in orders
            if o.status in ("filled", "partially_filled")
        ]

    async def get_portfolio_history(self) -> list[dict]:
        """Real account equity over time, via Alpaca's own portfolio-history endpoint —
        used by the dashboard's portfolio summary modal for the value-history chart.
        Confirmed live against the real account (2026-07-20): period="3M" was originally
        chosen but Alpaca rejects any intraday timeframe once period exceeds 30 days
        ("invalid timeframe provided: 1H. Valid timeframe for days > 30 is 1D"). Since the
        account is still under a month old, period="30D"/timeframe="1H" comfortably covers
        its full life so far with good resolution (confirmed: 147 real points, though the
        raw response pads roughly half of them with equity=0.0 for timestamps before the
        account existed / outside market hours — filtered out below, since a real active
        account never legitimately reports exactly $0 equity) and stays valid without
        changes until the account itself is older than 30 days — at that point this
        degrades to only the most recent 30 days of history rather than since-inception,
        which is an accepted, explicitly-deferred limitation (see CLAUDE.md), not a silent
        bug. Returns [] on any error — this is display-only, never allowed to affect
        trading."""
        try:
            history = await self._call_with_rate_limit_retry(
                lambda: self.api.get_portfolio_history(period="30D", timeframe="1H")
            )
            timestamps = history.timestamp or []
            equities = history.equity or []
            return [
                {"t": float(t), "equity": float(e)}
                for t, e in zip(timestamps, equities)
                if e is not None and float(e) != 0.0
            ]
        except Exception as e:
            logger.warning("get_portfolio_history failed: %s", e)
            return []

    async def get_quote(self, ticker: str) -> float | None:
        """Not currently called anywhere live (2026-08-08 audit confirmed) -- every real
        quote fetch in this codebase goes through MarketDataFetcher.get_quote() (yfinance)
        instead. Kept for interface completeness/future use, and hardened to match every
        other AlpacaBroker method's rate-limit retry for the same reason -- an unused
        method that silently skips this codebase's own established resilience pattern is
        exactly the kind of inconsistency a future caller would unknowingly inherit."""
        quote = await self._call_with_rate_limit_retry(self.api.get_latest_quote, ticker)
        price = quote.ap if quote.ap is not None else quote.bp
        return float(price) if price is not None else None
