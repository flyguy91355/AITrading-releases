"""Robinhood broker implementation with paper trading simulation."""

import asyncio
import logging
import os
import uuid
from datetime import datetime

import robin_stocks.robinhood as rh

from src.execution.broker import Broker, Order, OrderSide, OrderType, OrderStatus, AccountInfo

logger = logging.getLogger(__name__)


class RobinhoodBroker(Broker):
    def __init__(self, config: dict):
        self.config = config
        self.paper = config["trading"]["paper_trading"]
        self.logged_in = False
        self._paper_cash = config["portfolio"]["initial_capital"]
        self._paper_positions: dict[str, dict] = {}
        self._paper_orders: dict[str, Order] = {}

    async def connect(self):
        if self.paper:
            logger.info("Robinhood paper trading mode — no login required")
            return

        username = os.getenv("ROBINHOOD_USERNAME", "")
        password = os.getenv("ROBINHOOD_PASSWORD", "")
        mfa_code = os.getenv("ROBINHOOD_MFA_CODE", "")

        if not username or not password:
            raise RuntimeError("ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set")

        login = await asyncio.to_thread(
            rh.login, username, password, mfa_code=mfa_code if mfa_code else None
        )
        if login:
            self.logged_in = True
            logger.info("Connected to Robinhood")
        else:
            raise RuntimeError("Robinhood login failed")

    async def disconnect(self):
        if self.logged_in:
            await asyncio.to_thread(rh.logout)
            self.logged_in = False
            logger.info("Disconnected from Robinhood")

    async def get_account(self) -> AccountInfo:
        if self.paper:
            pos_value = sum(
                p["shares"] * p["current_price"] for p in self._paper_positions.values()
            )
            total = self._paper_cash + pos_value
            return AccountInfo(
                cash=self._paper_cash,
                portfolio_value=total,
                buying_power=self._paper_cash,
                equity=total,
            )

        profile = await asyncio.to_thread(rh.profiles.load_portfolio_profile)
        return AccountInfo(
            cash=float(profile.get("withdrawable_amount", 0)),
            portfolio_value=float(profile.get("market_value", 0)),
            buying_power=float(profile.get("excess_margin", 0)),
            equity=float(profile.get("equity", 0)),
        )

    async def submit_order(self, order: Order) -> Order:
        if self.paper:
            return await self._paper_submit(order)

        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                result = await asyncio.to_thread(rh.orders.order_buy_market, order.ticker, order.quantity)
            else:
                result = await asyncio.to_thread(rh.orders.order_sell_market, order.ticker, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                result = await asyncio.to_thread(rh.orders.order_buy_limit, order.ticker, order.quantity, order.limit_price)
            else:
                result = await asyncio.to_thread(rh.orders.order_sell_limit, order.ticker, order.quantity, order.limit_price)
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                result = await asyncio.to_thread(rh.orders.order_buy_stop_loss, order.ticker, order.quantity, order.stop_price)
            else:
                result = await asyncio.to_thread(rh.orders.order_sell_stop_loss, order.ticker, order.quantity, order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                result = await asyncio.to_thread(
                    rh.orders.order_buy_stop_limit, order.ticker, order.quantity, order.limit_price, order.stop_price
                )
            else:
                result = await asyncio.to_thread(
                    rh.orders.order_sell_stop_limit, order.ticker, order.quantity, order.limit_price, order.stop_price
                )
        else:
            raise ValueError(f"Unsupported order type for Robinhood: {order.order_type}")

        order.broker_order_id = result.get("id", "")
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        if result.get("average_price"):
            order.filled_price = float(result["average_price"])
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_at = datetime.now()

        logger.info("Robinhood order submitted: %s %s %d — ID: %s",
                     order.side.value, order.ticker, order.quantity, order.broker_order_id)
        return order

    async def _paper_submit(self, order: Order) -> Order:
        price = await self.get_quote(order.ticker)
        order.broker_order_id = str(uuid.uuid4())
        order.submitted_at = datetime.now()

        fill_price = order.limit_price if order.order_type == OrderType.LIMIT and order.limit_price else price
        cost = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            if cost > self._paper_cash:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected — insufficient cash: $%.2f needed, $%.2f available",
                               cost, self._paper_cash)
                return order

            self._paper_cash -= cost
            pos = self._paper_positions.get(order.ticker, {"shares": 0, "avg_price": 0, "current_price": price})
            total_shares = pos["shares"] + order.quantity
            if total_shares > 0:
                pos["avg_price"] = (pos["shares"] * pos["avg_price"] + cost) / total_shares
            pos["shares"] = total_shares
            pos["current_price"] = price
            self._paper_positions[order.ticker] = pos

        elif order.side == OrderSide.SELL:
            pos = self._paper_positions.get(order.ticker)
            if not pos or pos["shares"] < order.quantity:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected — insufficient shares for %s", order.ticker)
                return order

            self._paper_cash += fill_price * order.quantity
            pos["shares"] -= order.quantity
            if pos["shares"] == 0:
                del self._paper_positions[order.ticker]

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        self._paper_orders[order.broker_order_id] = order

        logger.info("Paper order filled: %s %s %d @ $%.2f",
                     order.side.value, order.ticker, order.quantity, fill_price)
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        if self.paper:
            order = self._paper_orders.get(broker_order_id)
            if order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                order.status = OrderStatus.CANCELLED
                return True
            return False

        try:
            await asyncio.to_thread(rh.orders.cancel_stock_order, broker_order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel Robinhood order %s: %s", broker_order_id, e)
            return False

    async def get_order_status(self, broker_order_id: str) -> Order:
        if self.paper:
            order = self._paper_orders.get(broker_order_id)
            if order:
                return order
            raise ValueError(f"Paper order {broker_order_id} not found")

        result = await asyncio.to_thread(rh.orders.get_stock_order_info, broker_order_id)
        return Order(
            ticker=result.get("instrument", "").split("/")[-2] if result.get("instrument") else "",
            side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=int(float(result.get("quantity", 0))),
            status=OrderStatus.FILLED if result.get("state") == "filled" else OrderStatus.SUBMITTED,
            filled_price=float(result["average_price"]) if result.get("average_price") else None,
            filled_quantity=int(float(result.get("cumulative_quantity", 0))),
            broker_order_id=broker_order_id,
        )

    async def get_positions(self) -> list[dict]:
        if self.paper:
            return [
                {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": pos["avg_price"],
                    "current_price": pos["current_price"],
                    "market_value": pos["shares"] * pos["current_price"],
                    "unrealized_pnl": pos["shares"] * (pos["current_price"] - pos["avg_price"]),
                    "unrealized_pnl_pct": ((pos["current_price"] - pos["avg_price"]) / pos["avg_price"] * 100)
                    if pos["avg_price"] > 0 else 0,
                }
                for ticker, pos in self._paper_positions.items()
            ]

        positions = await asyncio.to_thread(rh.account.get_open_stock_positions)
        results = []
        for p in positions:
            shares = float(p.get("quantity", 0))
            if shares <= 0:
                continue
            avg_price = float(p.get("average_buy_price", 0))
            instrument_url = p.get("instrument", "")

            try:
                instrument = await asyncio.to_thread(rh.stocks.get_instrument_by_url, instrument_url)
                ticker = instrument.get("symbol", "")
                quote = await asyncio.to_thread(rh.stocks.get_latest_price, ticker)
                current_price = float(quote[0]) if quote else avg_price
            except Exception:
                ticker = ""
                current_price = avg_price

            results.append({
                "ticker": ticker,
                "shares": int(shares),
                "entry_price": avg_price,
                "current_price": current_price,
                "market_value": shares * current_price,
                "unrealized_pnl": shares * (current_price - avg_price),
                "unrealized_pnl_pct": ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0,
            })
        return results

    async def get_open_orders(self) -> list[dict]:
        if self.paper:
            return [
                {
                    "order_id": oid,
                    "ticker": o.ticker,
                    "side": o.side.value,
                    "type": o.order_type.value,
                    "qty": o.quantity,
                    "limit_price": o.limit_price,
                    "stop_price": o.stop_price,
                }
                for oid, o in self._paper_orders.items()
                if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
            ]

        orders = await asyncio.to_thread(rh.orders.get_all_open_stock_orders)
        return [
            {
                "order_id": o.get("id", ""),
                "ticker": o.get("symbol", ""),
                "side": o.get("side", ""),
                "type": o.get("type", ""),
                "qty": float(o.get("quantity", 0)),
                "limit_price": float(o["price"]) if o.get("price") else None,
                "stop_price": float(o["stop_price"]) if o.get("stop_price") else None,
            }
            for o in orders
        ]

    async def get_closed_orders(self, symbols: list[str] | None = None, limit: int = 100) -> list[dict]:
        """Not implemented for Robinhood -- returns [] rather than raising (2026-08-03,
        full-codebase audit finding). OrderManager calls self.broker.get_closed_orders(...)
        unconditionally at several sites (share-count-drop and apparent-close
        reconciliation) with no getattr guard, unlike get_position/replace_order which
        are already accessed defensively elsewhere. Robinhood is currently dormant
        (config default is alpaca) so this has no live impact today, but would have
        raised AttributeError immediately if Robinhood were ever switched back in.
        A real implementation would need Robinhood's own filled-orders API; deferred
        until Robinhood support is actually revisited."""
        return []

    async def get_quote(self, ticker: str) -> float:
        if self.paper:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
            return 0.0

        quote = await asyncio.to_thread(rh.stocks.get_latest_price, ticker)
        return float(quote[0]) if quote else 0.0
