"""Order lifecycle management."""

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.execution.broker import Broker, Order, OrderSide, OrderType, OrderStatus
from src.execution.alpaca_broker import AlpacaBroker
from src.decision.portfolio import Portfolio, Position

logger = logging.getLogger(__name__)

# Matches Alpaca's "insufficient qty available for order (requested: X, available: Y)"
_INSUFFICIENT_QTY_RE = re.compile(r"available:\s*([\d.]+)")

# Backoff schedule (seconds) for consecutive sync_exit_orders remediation failures on the
# same ticker (2026-07-28) -- index 0 applies after the 1st failure, index 1 after the
# 2nd, etc.; the schedule's last value repeats for any further consecutive failures
# rather than growing unbounded. Detection (check_protection_gaps, every 10s) is
# unaffected by this -- only the retry attempt itself is throttled once it starts failing
# repeatedly, so a gap that resolves on the very next attempt is never delayed at all.
_EXIT_ORDER_RETRY_BACKOFF_SECONDS = [10, 30, 60, 120, 300]


class FillStillSettlingError(Exception):
    """Raised by _reconcile_untracked_fill when the real order behind an observed
    share-count delta is still partially_filled at Alpaca (2026-08-11, SBRA incident).
    That order's own filled_qty keeps changing between when the position-poller
    observed this delta and when this function queries Alpaca's real order history,
    so committing either a confident match or a guessed estimate right now risks
    recording the wrong number -- SBRA's real trailing-stop-triggered market sell was
    caught mid-fill this way and fabricated into two fake "Take-Profit" trade_history
    rows using stale target prices. Callers must catch this and skip BOTH the
    trade_history write and the pos.shares correction for this tick, so the next poll
    retries once the order reaches a real terminal state -- silently losing track of
    the gap would be worse than a one-tick delay."""


def _split_thirds(shares: float) -> tuple[float, float, float]:
    """Split shares into three tranches that sum exactly to shares."""
    t1 = round(shares / 3, 9)
    t2 = round(shares / 3, 9)
    t3 = round(shares - t1 - t2, 9)
    return t1, t2, t3


def _classify_stop_exit_reason(fill_price: float, entry_price: float, profit_target_hit: bool) -> str:
    """Classifies a CONFIRMED stop-type order fill into the same specific reason labels
    the live RISK-log path already uses in position_update_loop (web/app.py) -- "Stop
    loss hit" / "Trailing stop hit" / "Profit-target trailing stop hit" (2026-07-29, user
    asked to investigate why so many trade_history rows only had the vague reason
    "Broker-Detected Close (Stop/TP)"). The trade_updates-stream fill handler that calls
    this already knows FOR CERTAIN the fill was on the stop-type order (matched via
    _stop_order_ids, never a TP limit order), so "(Stop/TP)" was misleadingly uncertain
    about something already known -- only stop-loss vs trailing-stop (and whether the
    dollar profit-target trail was active) was genuinely unresolved there.

    Uses the same fill-price-vs-entry-price signal position_update_loop's own live
    classification already relies on: a stop-type order filling AT OR ABOVE entry means
    the trailing stop had armed and ratcheted up before triggering (a profit-preserving
    exit), while filling BELOW entry means the original stop-loss (never armed, or price
    gapped through too fast for the trail to matter)."""
    if fill_price >= entry_price:
        return "Profit-target trailing stop hit" if profit_target_hit else "Trailing stop hit"
    return "Stop loss hit"


def _compute_tranche_split(shares: float, targets: list[float]):
    """Pure tranche-size math. Originally computed a stop/TP split for
    _place_exit_orders' former inline branching; since the 2026-08-11 redesign (stop
    always covers 100% of the position, take-profit is a separate active action —
    see _place_exit_orders' docstring), only this function's `tp_shares` and
    `tranche` return values are actually used anymore, by
    _execute_take_profit_tranche, to size the ONE tranche about to be sold given the
    position's current holdings. `stop_shares` and `queued` are computed the same way
    as before but are no longer consumed by any real caller — kept rather than
    reworking this function's signature, since the underlying share-splitting math
    (how big is tranche N given `shares` currently held) is still exactly right and
    still worth having in one place, tested once.

    Returns (stop_shares, tp_shares, tp_price, queued, tranche). tp_shares is this
    tranche's size (e.g. ⅓ of `shares` when 3 targets remain, ½ when 2 remain);
    tranche is the 1-indexed tranche number (1 for T1, 2 for T2). n <= 1 (one or zero
    targets remain) returns tp_shares=0 — that state means "no more real tranches to
    sell, the rest rides the trailing stop alone," and check_take_profits never calls
    _execute_take_profit_tranche for it."""
    n = len(targets)
    if n == 0:
        return shares, 0.0, None, [], 3

    if n >= 3:
        t1, t2, t3 = _split_thirds(shares)
        stop_shares = round(t2 + t3, 9)
        tp_shares, tp_price = t1, targets[0]
        queued = [(t2, targets[1]), (t3, targets[2])]
        tranche = 1
    elif n == 2:
        # Resuming after TP1 — split remaining shares evenly for stop and TP2
        half = round(shares / 2, 9)
        remainder = round(shares - half, 9)
        stop_shares = remainder
        tp_shares, tp_price = half, targets[0]
        queued = [(remainder, targets[1])]
        tranche = 2
    else:  # n == 1
        # Final tranche (resuming after TP2) — no fixed target gets placed here at all.
        # The stop covers ALL remaining shares; the trailing stop (which tightens further
        # once price clears this stored target level) is the only exit.
        stop_shares = shares
        tp_shares, tp_price = 0, targets[0]
        queued = []
        tranche = 3

    return stop_shares, tp_shares, tp_price, queued, tranche


def _tranche_reason(tranche_number: int) -> str:
    """Maps a 1-indexed tranche number to this codebase's existing trade_history reason
    strings (2026-07-24, untracked-TP-fill accuracy fix) -- the same strings
    formatSellReason() in dashboard.html already knows how to render. Anything beyond T2
    is the final tranche, which never gets a real fixed order in normal operation (it
    rides the trailing stop), but IS reachable here if a reconciled fill happens to cover
    it too."""
    if tranche_number == 1:
        return "Take-Profit T1"
    if tranche_number == 2:
        return "Take-Profit T2"
    return "Take-Profit (final tranche)"


def _order_avg_fill_price(order, fallback_price: float) -> float:
    """Alpaca's trade_updates schema distinguishes the top-level per-event `price` (just
    THIS specific execution's price) from `order.filled_avg_price` (the order's true
    cumulative weighted-average fill price across every execution that order has had --
    confirmed against Alpaca's own docs, 2026-08-04). For a single-execution fill the two
    are identical, but an order that fills in multiple pieces at different prices means
    the terminal event's own `price` only reflects the LAST piece, not the true blended
    average paid -- not hypothetical for this system: the OVV incident (2026-07-28) is a
    real order that filled in exactly two pieces. Handles order arriving as either the
    SDK's typed Order object or a raw dict, same as the order_id extraction above this
    function's only caller already does. Falls back to fallback_price (the per-event
    tu.price) only if filled_avg_price is ever missing or not a valid number."""
    raw = order.get("filled_avg_price") if isinstance(order, dict) else getattr(order, "filled_avg_price", None)
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return fallback_price


class OrderManager:
    def __init__(self, config: dict, portfolio: Portfolio):
        self.config = config
        self.portfolio = portfolio
        self.broker: Broker | None = None
        self.active_orders: dict[str, Order] = {}

        # ticker → current stop-loss order ID. Take-profit has no resting-order
        # tracking dict anymore (2026-08-11 redesign) -- it's an active, independent
        # action (check_take_profits/_execute_take_profit_tranche), never a resting
        # order this file needs to remember across calls.
        self._stop_order_ids: dict[str, str] = {}
        # pending info for orders not yet filled (after-hours market orders)
        self._pending_stops: dict[str, dict] = {}
        # ticker -> broker_order_id for a position ALREADY created from a synchronous
        # PARTIAL fill result in _execute_buy (2026-07-29, BEN incident). Per Alpaca's own
        # docs, "partially_filled" is explicitly NOT a terminal state -- the order remains
        # open and can receive more fills -- so a position built from that snapshot alone
        # can be permanently undersized once the SAME order's remaining quantity clears
        # moments later. Confirmed live: BEN's notional buy synchronously reported PARTIAL
        # with a partial filled_qty (position built for 14.0 shares), while Alpaca's real
        # completing fill (20.79 shares, confirmed via get_closed_orders) arrived ~3s later
        # via the trade_updates stream -- but since this ticker was never registered in
        # _pending_stops (that dict is only for the create-fresh case, no position exists
        # yet), the stream's terminal "fill" event had nothing to reconcile against and was
        # silently dropped, leaving the position permanently stuck at the partial quantity
        # with exit orders undersized to match. _handle_trade_update now checks this dict
        # too and corrects the existing position's share count (never creates a second one)
        # when the real terminal fill arrives -- the already-existing quantity-based
        # protection-gap check (SCHW incident, _position_is_covered) then naturally resizes
        # the under-covered stop/TP on its own next cycle once shares is corrected.
        self._partial_fill_pending: dict[str, str] = {}
        # Alpaca trade_updates stream (2026-07-24) -- None until start_trade_updates_stream()
        # successfully sets it up. See docs/superpowers/specs/
        # 2026-07-24-alpaca-trade-updates-stream-design.md.
        self._trade_updates_stream = None
        # Closes detected via the stream, queued for web/app.py's position_update_loop to
        # drain and report through the SAME _report_alpaca_detected_close path
        # update_positions()'s poll-detected closes already use -- keeps the dashboard
        # experience (ai_log entry, WS broadcast, push notification) identical regardless
        # of which mechanism detected the close first.
        self._stream_closed_reports: list[dict] = []
        # One asyncio.Lock per ticker (2026-07-17 — see _lock_for's docstring). Replaces
        # the previous _in_progress/_sell_in_progress sets, which were two separately
        # maintained ad-hoc flags that every new order-mutating code path had to
        # remember to check AND set correctly — the root cause of a recurring class of
        # order-execution races across this project's history (most recently AMGN and
        # ECPG the same day this was built). Every function that mutates a ticker's
        # broker-side exit orders now holds this one lock for the duration.
        self._ticker_locks: dict[str, asyncio.Lock] = {}
        # prevents concurrent invocations of sync_exit_orders from racing each other
        self._sync_in_progress: bool = False
        # set by a sync_exit_orders() call that arrived while another was already running;
        # the in-progress run checks this and loops once more before releasing the lock,
        # instead of silently dropping tickers that were skipped this pass (2026-07-17 incident)
        self._sync_rerun_requested: bool = False
        # ticker -> (next_retry_allowed_at, consecutive_failure_count) for sync_exit_orders'
        # per-ticker remediation backoff (2026-07-28, MET incident: check_protection_gaps
        # fires sync_exit_orders() on every 10s cycle a gap stays open, with no throttle —
        # when the underlying cause doesn't clear inside 10s (e.g. Alpaca hasn't finished
        # settling a cancel yet), this hammered the identical cancel+place race every cycle
        # indefinitely). Detection (check_protection_gaps) deliberately stays fast/untouched
        # by this — only REMEDIATION attempts back off, and only after they've actually
        # failed; a ticker that's never failed, or that just succeeded, is never throttled.
        self._exit_order_retry_backoff: dict[str, tuple[datetime, int]] = {}

    def _lock_for(self, ticker: str) -> asyncio.Lock:
        """The single per-ticker coordination point for order mutation. Every code path
        that cancels or places exit orders (or closes a position) for a ticker acquires
        this lock for the duration of that work; every code path that only needs to check
        "is someone else currently touching this ticker's orders" reads lock.locked()
        (non-blocking) rather than acquiring it — checking-then-entering `async with lock`
        with no `await` between the two is safe in asyncio's single-threaded cooperative
        model (nothing else can run in that gap), so this never introduces a new race of
        its own. Callers that find the lock held skip the ticker for this cycle rather
        than block waiting for it — sync_exit_orders' rerun-pass loop, the hourly position
        monitor, and check_take_profits' own next cycle already exist to catch up once the
        lock frees, so blocking here would only stall unrelated tickers for no benefit."""
        if ticker not in self._ticker_locks:
            self._ticker_locks[ticker] = asyncio.Lock()
        return self._ticker_locks[ticker]

    async def connect(self):
        broker_name = self.config["trading"]["broker"]
        if broker_name == "alpaca":
            self.broker = AlpacaBroker(self.config)
        else:
            raise ValueError(f"Unknown broker: {broker_name}")
        await self.broker.connect()
        await self._sync_portfolio()
        await self.sync_exit_orders()

    async def start_trade_updates_stream(self) -> None:
        """Sets up the real-time Alpaca trade_updates WebSocket stream and launches it as
        a background task (2026-07-24). Purely additive -- every existing REST-polling
        mechanism in this file keeps running unchanged as a safety net. Any failure here
        (missing credentials, broker without stream support, connection setup error) is
        logged and swallowed; the app continues on polling alone, exactly as it did before
        this feature existed. See the design spec for why this must never block startup."""
        create_stream = getattr(self.broker, "create_trade_updates_stream", None)
        if create_stream is None:
            logger.info("Broker does not support trade_updates streaming — using polling only")
            return
        try:
            stream = create_stream()
            if stream is None:
                logger.info("trade_updates stream unavailable (missing credentials) — using polling only")
                return
            stream.subscribe_trade_updates(self._handle_trade_update)
            asyncio.create_task(stream._trading_ws._run_forever())
            self._trade_updates_stream = stream
            logger.info("Alpaca trade_updates stream started")
        except Exception as e:
            logger.warning("Failed to start trade_updates stream (%s) — using polling only", e)

    def pop_stream_closed_reports(self) -> list[dict]:
        """Drains and returns closes detected via the trade_updates stream since the last
        call, in the same {"ticker", "shares", "fill_price", "pnl"} shape
        update_positions() already returns for web/app.py to report identically."""
        reports = self._stream_closed_reports
        self._stream_closed_reports = []
        return reports

    async def _handle_trade_update(self, tu) -> None:
        """Routes a real-time fill event from the trade_updates stream. Only acts on the
        terminal "fill" event, never "partial_fill" (2026-07-28, OVV incident) -- a
        partial_fill's position_qty reflects only what's filled SO FAR, not the order's
        eventual final total. This function used to also react to partial_fill, and both
        branches below pop their tracking dict entry on the FIRST matching event they see
        -- so an order that filled in multiple pieces (a $674 notional buy for OVV filled
        in two: a first partial_fill reporting position_qty=1.126, then a second fill
        completing it at the real 11.126) got permanently locked in at the too-small
        partial quantity, since the tracking entry was already gone by the time the real
        completing fill arrived. Every other event type this system doesn't act on (new,
        canceled, rejected, etc.) is also ignored here. Any exception is caught and logged
        here -- it must never propagate into the SDK's own event loop, which would
        otherwise kill the stream task.

        Every recorded fill/exit price below goes through _order_avg_fill_price (2026-08-04,
        verified against Alpaca's own trade_updates schema docs), not the raw tu.price --
        Alpaca documents top-level `price` as just THIS execution's own price, while
        `order.filled_avg_price` is the order's true cumulative weighted-average across
        every execution. Identical for a single-fill order; only actually differs for a
        multi-piece fill like OVV's above, where using tu.price alone would silently
        record only the last piece's price instead of what was really paid on average."""
        try:
            if tu.event != "fill":
                return

            # tu.order arrives as the SDK's typed Order object most of the time, but has
            # been observed (2026-07-27, live) coming through as a raw dict instead --
            # handle both shapes rather than assuming .id is always present.
            _order = tu.order
            order_id = _order.get("id") if isinstance(_order, dict) else _order.id

            # Pending notional buy resolving for real (2026-07-24, SCHW incident fix) --
            # authoritative position_qty replaces the risky "whatever the next 10s poll
            # happens to see" mechanism entirely.
            for ticker, pending in list(self._pending_stops.items()):
                if pending.get("order_id") != order_id:
                    continue
                self._pending_stops.pop(ticker, None)
                real_shares = float(tu.position_qty)
                real_price = _order_avg_fill_price(_order, float(tu.price))
                await self.portfolio.add_position_async(Position(
                    ticker=ticker,
                    shares=real_shares,
                    entry_price=real_price,
                    current_price=real_price,
                    stop_loss=pending["stop_price"],
                    take_profit_targets=pending["take_profit_targets"],
                    sector=pending.get("sector", ""),
                    opened_at=datetime.now(),
                    t1_target_price=pending["take_profit_targets"][0] if len(pending["take_profit_targets"]) > 0 else None,
                    t2_target_price=pending["take_profit_targets"][1] if len(pending["take_profit_targets"]) > 1 else None,
                    trade_id=str(uuid.uuid4()),
                ))
                logger.info(
                    "%s pending buy resolved via trade_updates stream: %.4g shares @ $%.2f",
                    ticker, real_shares, real_price,
                )
                async with self._lock_for(ticker):
                    await self._place_exit_orders(
                        ticker, real_shares, pending["stop_price"], pending["take_profit_targets"],
                    )
                return

            # A position already exists for this ticker, built from a synchronous PARTIAL
            # fill result in _execute_buy (2026-07-29, BEN incident) -- see
            # _partial_fill_pending's docstring in __init__ for the full writeup. Corrects
            # the EXISTING position's share count to this order's real, final, cumulative
            # position_qty rather than creating a second position. Does not resize the
            # exit orders directly here -- the already-existing quantity-based
            # protection-gap check (_position_is_covered, SCHW incident) naturally detects
            # the now-undersized stop/TP against the corrected share count and resizes them
            # on its own next sync_exit_orders/check_protection_gaps cycle.
            for ticker, tracked_order_id in list(self._partial_fill_pending.items()):
                if tracked_order_id != order_id:
                    continue
                self._partial_fill_pending.pop(ticker, None)
                if ticker not in self.portfolio.positions:
                    logger.warning(
                        "%s partial-fill reconciliation: position no longer exists — skipping",
                        ticker,
                    )
                    return
                real_shares = float(tu.position_qty)
                pos = self.portfolio.positions[ticker]
                old_shares = pos.shares
                pos.shares = real_shares
                await self.portfolio._save_position(pos)
                logger.info(
                    "%s partial-fill reconciled via trade_updates stream: %.4g -> %.4g shares "
                    "(real completing fill) -- exit orders will resize on the next cycle",
                    ticker, old_shares, real_shares,
                )
                return

            # Stop-order fill (2026-07-24, corrected 2026-07-30 -- NDAQ incident). The
            # original assumption here ("this system's stops always cover the FULL
            # remaining position") was only ever true for the final tranche (0 or 1
            # remaining targets) -- a resting stop order is sized to just the "stop
            # tranche" portion once T1 (or T1+T2) has already fired
            # (_compute_tranche_split gives it 2/3 or half of what's left, never the
            # whole remaining amount until the final tranche). Blindly closing the
            # position for its FULL locally-tracked share count whenever this stop id
            # fired -- regardless of how many shares that stop order actually covered --
            # left the real remainder (the TP tranche) genuinely still held at Alpaca
            # with zero local tracking and zero protection, undetected until the next
            # full _sync_portfolio pass (normally startup-only). Alpaca's own
            # tu.position_qty is documented as the true CUMULATIVE remaining quantity
            # after this fill (the same attribute already relied on for the
            # partial-buy-fill case above) -- use it to tell a genuine full close apart
            # from a partial stop-tranche fill instead of assuming.
            for ticker, stop_id in list(self._stop_order_ids.items()):
                if stop_id != order_id:
                    continue
                if ticker not in self.portfolio.positions:
                    return
                async with self._lock_for(ticker):
                    self._stop_order_ids.pop(ticker, None)
                    pos = self.portfolio.positions[ticker]
                    fill_price = _order_avg_fill_price(_order, float(tu.price))
                    real_remaining = float(tu.position_qty) if tu.position_qty is not None else 0.0
                    if real_remaining > 0.001:
                        # Only the stop tranche filled -- correct the share count and
                        # leave the position open. Fixed 2026-08-02 (GitHub #39): this
                        # branch used to run _cancel_exit_orders(ticker) unconditionally
                        # BEFORE this check, which cancelled a still-valid, correctly-
                        # priced resting TP order that was never part of this fill --
                        # briefly leaving the remaining shares with zero protection until
                        # the next gap-check cycle re-placed it. Now leaves every other
                        # resting order untouched; only the fill's own stop tranche
                        # changed hands. The existing quantity-based protection-gap check
                        # (_position_is_covered, SCHW incident) naturally detects the
                        # now-undersized/missing exit orders against the corrected share
                        # count and resizes/replaces them on its own next
                        # sync_exit_orders/check_protection_gaps cycle if actually needed --
                        # same self-healing pattern as the BEN partial-fill fix above.
                        #
                        # But a stop firing at all -- even just for its own tranche --
                        # means the WHOLE position closes now (2026-08-11, FTV incident,
                        # see _liquidate_remainder_after_stop_fire's docstring) rather
                        # than leaving the remainder resting on whatever take-profit
                        # order happens to still cover it until a later cycle notices.
                        old_shares = pos.shares
                        pos.shares = real_remaining
                        await self.portfolio._save_position(pos)
                        logger.info(
                            "%s stop order filled for the stop tranche (%.4g -> %.4g "
                            "shares remaining) -- closing the rest immediately",
                            ticker, old_shares, real_remaining,
                        )
                        await self._liquidate_remainder_after_stop_fire(ticker, real_remaining)
                        return
                    # Full close -- now safe to cancel every remaining exit order for
                    # this ticker (deferred from before the branch above, GitHub #39).
                    await self._cancel_exit_orders(ticker)
                    closed_shares = pos.shares
                    # Specific reason (2026-07-29), not the vague "(Stop/TP)" -- this
                    # branch already knows for certain it's the stop-type order (matched
                    # via _stop_order_ids above), so only stop-loss-vs-trailing-stop was
                    # ever unresolved. See _classify_stop_exit_reason's docstring.
                    close_reason = _classify_stop_exit_reason(
                        fill_price, pos.entry_price, pos.profit_target_hit)
                    pnl = await self.portfolio.close_position_async(
                        ticker, exit_shares=closed_shares, exit_price=fill_price,
                        reason=close_reason,
                    )
                try:
                    account = await self.broker.get_account()
                    self.portfolio.cash = account.cash
                    await self.portfolio._save_state()
                except Exception as e:
                    logger.warning("Cash re-sync after stream-detected close failed: %s", e)
                self._stream_closed_reports.append({
                    "ticker": ticker, "shares": closed_shares,
                    "fill_price": fill_price, "pnl": pnl,
                })
                logger.info(
                    "%s position closed via trade_updates stream (stop filled): "
                    "%.4g shares @ $%.2f",
                    ticker, closed_shares, fill_price,
                )
                return
        except Exception as e:
            logger.warning("_handle_trade_update: unhandled error (%s) — event: %s", e, getattr(tu, "event", "?"))

    async def _log_unreconciled_fill(
        self, ticker: str, shares_sold: float, entry_price: float,
    ) -> tuple[float, float] | None:
        """Honest last-resort trade_history record for a detected share-count delta whose
        real Alpaca order could not be found or confidently matched (2026-08-11,
        DXCM/MA/VLY/GEN/IVZ/SBRA fabrication audit — owner asked "are any other stock
        trades fabricated" after catching SBRA's, and cross-checking every historical
        "estimated, untracked fill" row against Alpaca's real order history found 6 more:
        several of them a real STOP-LOSS loss relabeled as a fake profitable "Take-Profit").

        Replaces the old _log_estimated_tp_fill, which averaged the currently-open
        take_profit_targets into a guessed price and ALWAYS wrote the reason as
        "Take-Profit" — even when the real fill was a stop-loss, because this fallback has
        no real order data to tell the difference. That guessing is the root cause of every
        fabricated row this audit found, not any one ticker's specific trigger condition —
        each prior "fix" (SCHW/OVV/BEN/SBRA) closed off one more PATH into this fallback
        without ever removing the fallback's ability to invent a price and a wrong label
        once it's reached.

        This function never invents a specific fill price or claims a specific outcome
        (take-profit vs. stop). It records the real share count — the one number the
        position-poller's own observed delta actually confirms — against the CURRENT live
        quote (an honest "roughly where the stock was trading," not a target price that
        implies profit-taking happened), with a reason that says plainly the real order
        wasn't found and this needs manual review. Conservatively marks the ticker as a
        recent loss for wash-sale purposes regardless of the quote-based estimate's sign —
        since the real P&L here is genuinely unknown, blocking an unnecessary rebuy for
        wash_sale_cooldown_days is a far smaller cost than missing a real wash-sale
        violation. Callers must treat this as 0 targets fired (see _reconcile_untracked_fill)
        — an unconfirmed fill must never pop a take-profit target on a guess, the same RRC
        lesson already applied to confirmed stop-driven fills.

        Returns (price, pnl) on success (pnl here is an unreliable, clearly-labeled
        estimate, never treated as authoritative by anything reading it), None if it
        declined to log (no db / no shares) or the insert itself failed.
        """
        if not self.portfolio._db or shares_sold <= 0:
            return None
        price = None
        try:
            # AlpacaBroker.get_quote() returns a plain float (or None), not a dict --
            # fixed 2026-08-20, BEN incident. The old quote.get("price") call always
            # raised (a real float has no .get()), meaning this fallback's own live
            # quote attempt has never actually worked since it was written -- it
            # silently fell through to entry_price every single time, making the
            # logged "at current quote $X" line always just the entry price
            # relabeled, and guaranteeing pnl computed as exactly $0 no matter what
            # the position actually did.
            price = await self.broker.get_quote(ticker)
        except Exception as e:
            logger.warning("%s: live quote lookup for unreconciled fill failed (%s)", ticker, e)
        if not price or price <= 0:
            price = entry_price
        pnl = (price - entry_price) * shares_sold
        _pos = self.portfolio.positions.get(ticker)
        _trade_id = _pos.trade_id if _pos else None
        self.portfolio.recent_losses[ticker] = datetime.now()
        try:
            await self.portfolio._db.execute(
                "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, "SELL", shares_sold, price, pnl, datetime.now().isoformat(),
                 "UNRECONCILED FILL (real order not found — needs manual review)", _trade_id),
            )
            await self.portfolio._db.commit()
            logger.warning(
                "%s: could not match a %.4g-share fill to any real Alpaca order — logged "
                "as UNRECONCILED at current quote $%.2f (NOT a real fill price, NOT "
                "assumed take-profit). Needs manual review against Alpaca's real order "
                "history.",
                ticker, shares_sold, price,
            )
            return price, pnl
        except Exception as e:
            logger.warning("Failed to log unreconciled trade_history for %s: %s", ticker, e)
            return None

    async def _reconcile_untracked_fill(
        self, ticker: str, shares_sold: float, old_shares: float,
        current_targets: list[float], entry_price: float,
    ) -> tuple[float, float, int] | None:
        """Replaces the vague, averaged "estimated, untracked fill" trade_history record
        with accurate, real per-tranche entries by matching against Alpaca's own closed
        order history for this ticker (2026-07-24 -- user, looking at today's trade
        history, correctly rejected the vague reasons: "these were all filled today..
        please fix that so there accurate reasons.. i want it right"). _tp_orders is
        in-memory only and gets wiped on every restart, so a tranche that fills in the
        window right after a restart (before sync_exit_orders re-populates it) used to
        always fall back to this "untracked" guess, even though the real fill price/time
        was sitting at the broker the whole time, just never consulted.

        current_targets is the FULL take_profit_targets list as it stood before this
        share drop -- the caller must NOT pre-shrink it before calling this. Determining
        whether the real fill(s) behind the drop are take-profit or stop-driven has to
        happen BEFORE deciding how many targets to pop (2026-07-28, RRC incident): a
        single gap-through stop-loss can sell several tranches' worth of shares at once
        (the stop always covers the FULL currently-resting stop tranche, not one
        tranche-at-a-time like a real TP ladder), and the old code pre-guessed
        "shares_dropped / tranche_size" targets as fired and popped them BEFORE this
        function ever got a chance to determine the fill was actually a loss-taking stop,
        not a profit-taking TP -- RRC's single stop-loss sold exactly 2 tranches' worth
        and got mislabeled as "T1 and T2 both hit their targets" (visible on the dashboard
        as T1 CHECK T2 CHECK badges) when neither target had actually fired.

        Returns (avg_price, total_pnl, n_targets_fired) on success, None if it declined to
        log entirely. n_targets_fired is how many entries the CALLER should pop from the
        FRONT of current_targets -- always 0 when the matched fill(s) are stop-driven (a
        stop-loss never represents a take-profit tranche, regardless of quantity), or the
        real number of matched take-profit fills otherwise (never a pre-guess).

        Falls back to _log_unreconciled_fill whenever the real order history can't be
        cleanly, confidently matched (broker API failure, or the fill count doesn't line
        up with the available targets) — this must never log something wrong with false
        confidence. **Never guesses a price or a
        take-profit label (fixed 2026-08-11, DXCM/MA/VLY/GEN/IVZ/SBRA fabrication audit)**
        — the old version averaged popped target prices and always assumed take-profit,
        which is exactly what fabricated a fake profitable "Take-Profit" out of what were,
        in several real historical cases, actual stop-loss losses. Now defers to
        _log_unreconciled_fill, which records the real share count honestly against the
        current live quote with a plain "needs manual review" label, and always reports 0
        targets fired (never guesses which ones).

        Raises FillStillSettlingError (2026-08-11, SBRA incident) instead of returning
        None/falling back when the real order behind the delta is still
        partially_filled -- unlike a genuine no-match, this case must NOT reach the
        guess-based fallback at all, since the order's own quantity is still changing
        and any estimate committed now risks being wrong. Callers must catch this and
        defer the whole reconciliation (including the pos.shares correction) to the
        next poll.
        """
        if not self.portfolio._db or shares_sold <= 0 or not current_targets:
            return None

        async def _fallback_estimate() -> tuple[float, float, int] | None:
            # No guessing which targets fired (fixed 2026-08-11) -- an unconfirmed fill
            # always reports 0 targets fired, the same conservative default a confirmed
            # stop-driven fill already uses. Guessing a tranche count from shares_sold /
            # tranche_size was the other half of the fabrication bug: it could pop real
            # T1/T2 targets (wrong dashboard badges) for a fill that was never actually
            # verified as a take-profit at all.
            est = await self._log_unreconciled_fill(ticker, shares_sold, entry_price)
            if est is None:
                return None
            return est[0], est[1], 0

        async def _attempt_match() -> tuple[list, float] | None:
            """One attempt at fetching Alpaca's real closed-order history and matching it
            against shares_sold. Returns (matched, total) on a confident match, None
            otherwise (broker call failed, or nothing lines up)."""
            try:
                real_orders = await self.broker.get_closed_orders(symbols=[ticker])
            except Exception as e:
                logger.warning("%s: real-order lookup for untracked fill failed (%s)", ticker, e)
                return None

            # A still-settling order (2026-08-11, SBRA incident) -- get_closed_orders
            # already includes partially_filled orders (not just fully "filled" ones),
            # and such an order's own filled_qty keeps changing as it continues to
            # fill. That makes the greedy quantity-sum match below unreliable in both
            # directions: it can spuriously fail against a shares_sold snapshot the
            # order has since moved past, or spuriously "succeed" against a
            # coincidentally-matching intermediate quantity that isn't the order's
            # real final fill. Either way, matching or guessing right now risks
            # recording the wrong price/quantity for what is really just one order
            # still in the middle of filling -- bail out immediately (not just "no
            # match", which would fall through to the guess-based estimate) and let
            # the caller retry once the order reaches a real terminal state.
            if any(o.get("side") == "sell" and o.get("status") == "partially_filled"
                   for o in real_orders):
                raise FillStillSettlingError(ticker)

            sells = [
                o for o in real_orders
                if o.get("side") == "sell" and (o.get("filled_qty") or 0) > 0
                and o.get("filled_avg_price") is not None and o.get("filled_at")
            ]
            sells.sort(key=lambda o: o["filled_at"])

            # Greedily accumulate the MOST RECENT real fills (chronologically) whose combined
            # filled_qty sums to shares_sold within a small tolerance -- these are the actual
            # fill(s) behind this detected gap.
            _matched = []
            _total = 0.0
            for o in reversed(sells):
                if _total >= shares_sold - 0.01:
                    break
                _matched.append(o)
                _total += o["filled_qty"]
            _matched.reverse()  # chronological order again

            if not _matched or abs(_total - shares_sold) > 0.05:
                return None
            return _matched, _total

        result = await _attempt_match()
        if result is None:
            # Alpaca's closed-orders endpoint can lag a real fill's own filled_at by a
            # fraction of a second (confirmed live, 2026-08-07 EQR incident: the real
            # stop-loss fill's filled_at was ~300ms after this lookup first ran, so the
            # initial query returned an order list that didn't contain it yet, this
            # function gave up and fell through to the old guess-based estimate below,
            # and a real ~$6.51 loss got recorded as a fabricated $9.70 profit). One
            # short retry covers this the same way the settlement-lag retries elsewhere
            # in this file already handle Alpaca's eventual consistency for order
            # placement -- not a new pattern, just applied to this call site too.
            await asyncio.sleep(2.5)
            result = await _attempt_match()
        if result is None:
            # Still nothing after the retry -- fall back rather than risk mislabeling.
            return await _fallback_estimate()
        matched, total = result

        # A real fill whose order_type isn't "limit" (2026-07-24, ONB/GPN/EBAY incident)
        # means price already gapped through the stop before a resting order could be
        # placed, so _place_exit_orders' market-sell fallback sold the WHOLE stop-covered
        # portion in one order -- which can span multiple tranches' worth of shares, since
        # the fallback doesn't sell per-target the way a real take-profit ladder does.
        # Labeling it "Take-Profit" (or popping targets as if it were) would claim the
        # opposite of what actually happened, so this case doesn't need (or expect) a 1:1
        # order-to-tranche count match at all -- only the total quantity matters, and NO
        # targets get popped regardless of how many "tranches worth" the quantity spans.
        is_stop_driven = any(o.get("order_type") != "limit" for o in matched)

        if not is_stop_driven and len(matched) > len(current_targets):
            # More real fills than we have known targets for -- something's inconsistent,
            # fall back rather than guess which fill maps to which target.
            return await _fallback_estimate()

        n_fired = 0 if is_stop_driven else len(matched)
        starting_tranche = (3 - len(current_targets)) + 1
        # A gap-through stop can either close the WHOLE remaining position (no TP has
        # fired yet, so the resting stop covers 100% of shares) or just the currently
        # resting "stop tranche" (T1/T2 already fired, more of the position stays open)
        # -- these read identically apart from the quantity, so the reason string must
        # say which one actually happened (fixed 2026-08-11, owner live-caught SBRA's
        # correction reading "Partial" on the dashboard when it was a full close).
        stop_is_full_close = is_stop_driven and total >= old_shares - 0.01

        total_pnl = 0.0
        _pos = self.portfolio.positions.get(ticker)
        _trade_id = _pos.trade_id if _pos else None
        for i, o in enumerate(matched):
            if is_stop_driven:
                reason = ("Stop Loss (gap-through market sell, full close)" if stop_is_full_close
                          else "Stop Loss (gap-through market sell)")
            else:
                reason = _tranche_reason(starting_tranche + i)
            real_qty = o["filled_qty"]
            real_price = o["filled_avg_price"]
            pnl = (real_price - entry_price) * real_qty
            total_pnl += pnl
            if pnl < 0:
                self.portfolio.recent_losses[ticker] = datetime.now()
            try:
                await self.portfolio._db.execute(
                    "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticker, "SELL", real_qty, real_price, pnl, o["filled_at"], reason, _trade_id),
                )
            except Exception as e:
                logger.warning("Failed to log reconciled trade_history for %s: %s", ticker, e)
                return await _fallback_estimate()
        await self.portfolio._db.commit()
        logger.info(
            "%s: reconciled %d untracked fill(s) against real order history (%s, %d target(s) fired) — %s",
            ticker, len(matched), "stop-driven" if is_stop_driven else "take-profit", n_fired,
            [(round(o["filled_qty"], 4), o["filled_avg_price"]) for o in matched],
        )
        avg_price = sum(o["filled_avg_price"] * o["filled_qty"] for o in matched) / total
        return avg_price, total_pnl, n_fired

    async def _infer_real_entry_time(self, ticker: str, target_shares: float) -> datetime:
        """Best-effort recovery of a position's real original entry timestamp when it's
        being reconstructed from Alpaca's account state alone, with no local DB record to
        read it from (2026-08-04, NDAQ incident) -- the position-detail chart's entry
        marker was 6 days off because the one call site that does this (below, "In Alpaca
        but not in DB") stamped opened_at=datetime.now(), the moment of RECONCILIATION,
        not the real original buy. Root cause traced via journalctl: a restart 13 minutes
        after NDAQ's stop-loss fill found Alpaca still reporting a residual NDAQ position
        (not yet reflected locally) and rebuilt it here, 6 days after the real 2026-07-24
        buy.

        Mirrors _reconcile_untracked_fill's "accumulate the most recent real fills whose
        combined quantity matches" pattern (same file, above), applied to buys instead of
        sells: walks real closed buy orders backward from most recent, accumulating
        filled_qty until it covers target_shares (within the same 0.05-share tolerance
        used there), then returns the EARLIEST fill's time among that matched set -- the
        real moment the currently-held lot was actually opened. Passes a much larger
        limit than get_closed_orders' 100-order default, since a heavily cancel/replace-
        churned ticker (the routine hourly exit-order renewal, before the 2026-07-28
        in-place-replace fix cut this down a lot) can push a genuinely old buy fill out
        of even a single-ticker-scoped 100-order window -- confirmed live for NDAQ itself,
        whose original 7/24 buy fill was NOT present in a limit=100 query by the time of
        this fix.

        Falls back to datetime.now() (the old behavior) whenever the real history can't
        be fetched or cleanly matched -- an approximate-but-recent timestamp is strictly
        better than raising and losing the whole reconciliation."""
        try:
            real_orders = await self.broker.get_closed_orders(symbols=[ticker], limit=500)
        except Exception as e:
            logger.warning("%s: real-order lookup for entry time failed (%s) — using now()", ticker, e)
            return datetime.now()

        buys = [
            o for o in real_orders
            if o.get("side") == "buy" and (o.get("filled_qty") or 0) > 0
            and o.get("filled_at")
        ]
        buys.sort(key=lambda o: o["filled_at"])

        matched = []
        total = 0.0
        for o in reversed(buys):
            if total >= target_shares - 0.01:
                break
            matched.append(o)
            total += o["filled_qty"]

        if not matched or abs(total - target_shares) > 0.05:
            return datetime.now()

        # Alpaca's filled_at is UTC and tz-aware; every other opened_at in this codebase
        # is naive datetime.now() (the server itself runs in UTC, confirmed via
        # timedatectl, so the two are already the same wall-clock time) -- strip tzinfo
        # so this stays comparable/consistent with the naive convention used everywhere
        # else opened_at is read (e.g. web/app.py's opened_at.date() checks).
        earliest = min(matched, key=lambda o: o["filled_at"])
        return datetime.fromisoformat(earliest["filled_at"]).replace(tzinfo=None)

    async def _sync_portfolio(self):
        try:
            account = await self.broker.get_account()
            self.portfolio.cash = account.cash
            # Only derive day_start_value from Alpaca's last_equity ("previous trading
            # day's close") when today's reset genuinely hasn't happened yet -- closes the
            # long-deferred CR14 gap (2026-07-05): this used to run unconditionally on
            # EVERY restart, re-anchoring the whole day's P&L baseline to last_equity even
            # on an intraday restart where Portfolio.new_trading_day() had already
            # correctly set day_start_value (and day_start_date) earlier that same day.
            # Gating on day_start_date makes this purely self-healing: a restart that
            # missed the midnight rollover entirely (process was down across the day
            # boundary) still gets corrected here, but a same-day restart leaves the
            # already-correct value untouched instead of clobbering it.
            _tz_name = self.config.get("research", {}).get("market_timezone", "America/New_York")
            _today_str = datetime.now(ZoneInfo(_tz_name)).strftime("%Y-%m-%d")
            if account.last_equity > 0 and self.portfolio.day_start_date != _today_str:
                self.portfolio.day_start_value = account.last_equity
                self.portfolio.day_start_date = _today_str
            await self.portfolio._save_state()

            tp_cfg = self.config.get("take_profit", {})
            sl_mult = 1 - tp_cfg.get("stop_loss_pct", 5.0) / 100
            t1_mult = 1 + tp_cfg.get("t1_pct",  5.0) / 100
            t2_mult = 1 + tp_cfg.get("t2_pct", 10.0) / 100
            t3_mult = 1 + tp_cfg.get("t3_pct", 17.0) / 100

            positions = await self.broker.get_positions()
            for p in positions:
                if p["ticker"] in self.portfolio.positions:
                    # Loaded from DB — update live price and actual share count, then persist
                    pos = self.portfolio.positions[p["ticker"]]
                    pos.current_price = p["current_price"]
                    old_shares = pos.shares
                    new_shares = p["shares"]

                    # Detect a share-count drop while the system was down or between polls:
                    # if Alpaca reports significantly fewer shares than our DB, either a real
                    # take-profit fired untracked, or a stop-loss (possibly a gap-through
                    # market-sell covering multiple tranches' worth at once) did.
                    # _reconcile_untracked_fill determines which against Alpaca's own real
                    # order history and returns how many targets to pop -- 0 for a stop-driven
                    # closure, since a stop-loss never represents a take-profit tranche firing
                    # no matter how many tranches' worth of shares it happened to cover
                    # (2026-07-28, RRC incident: a single gap-through stop sold exactly 2
                    # tranches' worth and the OLD pre-guess-then-pop logic here assumed that
                    # meant "2 TPs fired," mislabeling a real loss as two banked profits and
                    # showing T1/T2 checkmarks for a position where neither target had
                    # actually been hit). Must NOT pre-shrink pos.take_profit_targets before
                    # this call -- the whole point is deciding how much (if any) to shrink
                    # only after knowing what the fill actually was.
                    #
                    # pos.shares is deliberately NOT committed until AFTER this call returns
                    # (2026-08-11, SBRA incident) -- see the matching comment in
                    # update_positions(). A restart landing mid-fill of a real order leaves
                    # pos.shares at its old (stale) value for this one pass; the routine 10s
                    # update_positions() loop retries and self-heals once the order settles.
                    _est = None
                    _settling = False
                    if (old_shares > 0.001
                            and new_shares < old_shares - 0.01
                            and pos.take_profit_targets):
                        shares_dropped = old_shares - new_shares
                        try:
                            _est = await self._reconcile_untracked_fill(
                                p["ticker"], shares_dropped, old_shares,
                                pos.take_profit_targets, pos.entry_price,
                            )
                        except FillStillSettlingError:
                            _settling = True
                    if _settling:
                        logger.info(
                            "_sync_portfolio: %s — real order still settling, "
                            "deferring share-count correction to next poll",
                            p["ticker"],
                        )
                    else:
                        pos.shares = new_shares
                        if _est is not None:
                            _avg_price, _pnl, _n_fired = _est
                            pos.realized_pnl += _pnl
                            pos.shares_sold += shares_dropped
                            if _n_fired > 0:
                                logger.warning(
                                    "%s shares dropped %.4g→%.4g during downtime — "
                                    "%d TP fill(s) confirmed, updating targets from %s",
                                    p["ticker"], old_shares, pos.shares,
                                    _n_fired, pos.take_profit_targets,
                                )
                                pos.take_profit_targets = pos.take_profit_targets[_n_fired:]
                                # Same breakeven-protection rule as the live
                                # check_take_profits() path — a TP fill detected after the
                                # fact still means remaining shares should never be protected
                                # at less than breakeven. Deliberately NOT applied when
                                # _n_fired == 0 (a stop-driven closure) -- forcing the
                                # trailing stop up to breakeven right after a loss makes no
                                # sense for shares that weren't part of any profit-taking.
                                if pos.trailing_stop is None:
                                    pos.trailing_stop = pos.entry_price
                                    logger.info(
                                        "%s trailing stop initialized to breakeven $%.2f "
                                        "on downtime-detected TP fill",
                                        p["ticker"], pos.entry_price,
                                    )
                            else:
                                logger.warning(
                                    "%s shares dropped %.4g→%.4g during downtime — "
                                    "confirmed stop-loss (not a TP fill), targets unchanged: %s",
                                    p["ticker"], old_shares, pos.shares,
                                    pos.take_profit_targets,
                                )
                                # A stop firing closes the WHOLE position immediately,
                                # not just its own tranche (2026-08-11, FTV incident —
                                # see _liquidate_remainder_after_stop_fire's docstring).
                                if pos.shares > 0.001:
                                    await self._liquidate_remainder_after_stop_fire(
                                        p["ticker"], pos.shares,
                                    )

                    # Self-healing invariant check, independent of the delta-detection above:
                    # any position holding fewer than 3 targets has necessarily had at least
                    # one TP fill, by any mechanism, at any point in its history — so it must
                    # have a trailing stop at or above breakeven. Catches cases (like a fill
                    # detected before this breakeven-protection fix existed) that the
                    # reactive delta-detection above only catches at the moment of detection.
                    if (pos.trailing_stop is None
                            and pos.take_profit_targets is not None
                            and len(pos.take_profit_targets) < 3):
                        pos.trailing_stop = pos.entry_price
                        logger.info(
                            "%s trailing stop initialized to breakeven $%.2f "
                            "(startup invariant check — %d target(s) already fired)",
                            p["ticker"], pos.entry_price, 3 - len(pos.take_profit_targets),
                        )

                    await self.portfolio._save_position(pos)
                else:
                    # In Alpaca but not in DB — compute TP targets from config and persist
                    entry = p["entry_price"]
                    targets = [
                        round(entry * t1_mult, 2),
                        round(entry * t2_mult, 2),
                        round(entry * t3_mult, 2),
                    ]
                    # 2026-08-04, NDAQ incident: this used to stamp opened_at=datetime.now()
                    # unconditionally -- the moment of RECONCILIATION, not the real original
                    # buy, which corrupted the chart's entry-date marker by however long the
                    # position sat "missing" before a restart rediscovered it (6 days for
                    # NDAQ). _infer_real_entry_time looks up the real fill via Alpaca's own
                    # order history and falls back to datetime.now() (the old behavior)
                    # itself whenever that can't be found or cleanly matched.
                    real_opened_at = await self._infer_real_entry_time(p["ticker"], p["shares"])
                    pos = Position(
                        ticker=p["ticker"],
                        shares=p["shares"],
                        entry_price=entry,
                        current_price=p["current_price"],
                        stop_loss=round(entry * sl_mult, 2),
                        take_profit_targets=targets,
                        sector="",
                        opened_at=real_opened_at,
                        # 2026-07-24: every other Position(...) construction site sets
                        # these from the real buy signal's targets -- this one (a position
                        # Alpaca reports that never made it into our own DB, e.g. a pending
                        # notional buy whose _pending_stops entry was lost across a
                        # restart) was missed, leaving the graduated trailing-stop curve's
                        # follow-T1/T2 mode with no anchors for any position reconstructed
                        # this way. targets here is always a fresh 3-element list.
                        t1_target_price=targets[0],
                        t2_target_price=targets[1],
                        # 2026-07-27: this recovery path has no way to know the real
                        # original trade_id (never tracked), so a fresh one is generated
                        # here as the best available option -- same "can't recover, don't
                        # guess the un-recoverable parts" spirit as t1/t2 above being left
                        # accurate rather than fake for genuinely lost data, except here a
                        # fresh id is strictly better than leaving it permanently None.
                        trade_id=str(uuid.uuid4()),
                    )
                    # Set directly — cash already reflects this position from Alpaca account
                    self.portfolio.positions[p["ticker"]] = pos
                    await self.portfolio._save_position(pos)
                    logger.info(
                        "Synced missing position %s from Alpaca — stop $%.2f, TPs $%.2f/$%.2f/$%.2f",
                        p["ticker"], pos.stop_loss, *targets,
                    )

            logger.info("Portfolio synced — Cash: $%.2f, Positions: %d, Total: $%.2f",
                        self.portfolio.cash, len(self.portfolio.positions), self.portfolio.total_value)
        except Exception as e:
            logger.warning("Portfolio sync failed: %s — using local state", e)

    async def disconnect(self):
        if self.broker:
            await self.broker.disconnect()

    async def execute(self, signal) -> Order | None:
        if signal.signal.value in ("SELL", "STRONG SELL"):
            return await self._execute_sell(signal)
        return await self._execute_buy(signal)

    async def _execute_buy(self, signal) -> Order | None:
        order = Order(
            ticker=signal.ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional_value=round(signal.position_size_dollars, 2),
        )

        result = await self.broker.submit_order(order)

        actual_shares = result.filled_quantity if result.filled_quantity is not None else signal.shares

        if result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            # PARTIAL: only some shares filled — protect what Alpaca actually bought
            if result.status == OrderStatus.PARTIAL:
                logger.warning(
                    "_execute_buy: PARTIAL fill for %s — creating position for %.4g filled shares",
                    signal.ticker, actual_shares,
                )
            self.active_orders[result.broker_order_id] = result
            filled_price = result.filled_price if result.filled_price is not None else signal.entry_price
            # Generated once here (2026-07-27) and echoed back onto the signal so the
            # caller's trade_logger.log_trade(signal) call (the JSONL buy record) carries
            # the same id every sell tranche of this position will later be stamped with —
            # see Position.trade_id.
            _trade_id = str(uuid.uuid4())
            if hasattr(signal, "trade_id"):
                signal.trade_id = _trade_id
            # "Why AI Bought This" snapshot (2026-08-21) -- captured once, here, from
            # whatever real report/numbers this signal actually carries. research_report
            # is None for a handful of non-AI-driven signal sources (e.g. a legacy
            # rebuy) -- falls back to empty/None fields rather than guessing, same
            # precedent as every other nullable Position field.
            _report = getattr(signal, "research_report", None)
            await self.portfolio.add_position_async(Position(
                ticker=signal.ticker,
                shares=actual_shares,
                entry_price=filled_price,
                current_price=filled_price,
                stop_loss=signal.stop_loss,
                take_profit_targets=signal.take_profit_targets,
                sector=getattr(signal, 'sector', ''),
                opened_at=datetime.now(),
                t1_target_price=signal.take_profit_targets[0] if len(signal.take_profit_targets) > 0 else None,
                t2_target_price=signal.take_profit_targets[1] if len(signal.take_profit_targets) > 1 else None,
                trade_id=_trade_id,
                buy_thesis=_report.thesis if _report else "",
                buy_reasoning=_report.reasoning if _report else "",
                buy_conviction=signal.conviction,
                buy_signal=signal.signal.value if hasattr(signal.signal, "value") else str(signal.signal),
                buy_rr=getattr(signal, "rr", None),
                buy_required_rr=getattr(signal, "required_rr", None),
                buy_fair_value=_report.fair_value_estimate if _report else None,
            ))
            # Locked so sync_exit_orders can't independently discover this brand-new
            # position mid-placement and race to "cover" it a second time — a gap that
            # existed even before today's other races (this ticker was never in
            # portfolio.positions until add_position_async just above, so nothing
            # previously guarded this specific window).
            async with self._lock_for(signal.ticker):
                await self._place_exit_orders(
                    signal.ticker, actual_shares,
                    signal.stop_loss, signal.take_profit_targets,
                )
            if result.status == OrderStatus.PARTIAL:
                # Register for reconciliation against this SAME order's real completing
                # fill (2026-07-29, BEN incident) -- see this dict's own docstring in
                # __init__ for the full incident writeup. Per Alpaca's docs, PARTIAL is
                # explicitly not terminal; without this, the order's later terminal "fill"
                # event (which reports the true cumulative position_qty) has nothing to
                # reconcile against and gets silently dropped.
                self._partial_fill_pending[signal.ticker] = result.broker_order_id
        elif result.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            self.active_orders[result.broker_order_id] = result
            # After-hours / just-accepted order — store for later fill detection.
            # order_id (2026-07-24) lets the trade_updates stream match its fill event to
            # this exact pending buy — see _handle_trade_update.
            self._pending_stops[signal.ticker] = {
                "shares": actual_shares,
                "stop_price": round(signal.stop_loss, 2),
                "take_profit_targets": signal.take_profit_targets,
                "sector": getattr(signal, 'sector', ''),
                "order_id": result.broker_order_id,
            }
        else:
            # REJECTED, CANCELLED — no shares were purchased
            logger.warning(
                "_execute_buy: order for %s ended with status %s — no position created",
                signal.ticker, result.status,
            )

        return result

    async def _place_exit_orders(
        self, ticker: str, shares: float,
        stop_price: float, targets: list[float],
    ) -> bool:
        """Places ONLY the protective stop, sized for the FULL current position
        (2026-08-11, FTV incident -- owner: "ITS NOT STOP LOSS/TAKE PROFIT...... ITS A
        STOP LOSS, THERE SEPARATE ENTIRELY"). Take-profit is no longer a resting order
        placed alongside the stop at all -- it's a fully separate, independent action
        (see check_take_profits' rewrite and _execute_take_profit_tranche below) that
        actively watches price against pos.take_profit_targets[0] and, when reached,
        cancels the current stop, sells that one tranche, and places a fresh stop for
        whatever remains. Verified against Alpaca's own docs before this redesign:
        Alpaca has no native mechanism for "one full-position stop + a separate resting
        partial take-profit" (resting orders reserve shares, so a 100%-sized stop and
        any additional resting sell order for the same shares can't coexist) -- this is
        the correct, and only achievable, way to guarantee a stop always covers
        everything currently held.

        The old version placed a stop for only part of the position (leaving the rest
        resting on a take-profit limit order) specifically so a partial profit-take
        could happen without ever exceeding 100% committed quantity. That's exactly
        what let FTV exit as two separate trades: the undersized stop fired for its own
        tranche, and the remainder — sitting on a now-irrelevant TP order — was left
        exposed until a later cycle noticed. A stop that always covers 100% has no such
        gap, because there's nothing else ever resting on the same shares to leave
        exposed.

        `targets` is accepted for backward compatibility with every existing call site
        (which still track/pass take_profit_targets for check_take_profits' own use)
        but no longer affects what gets placed here -- the stop is always sized for the
        full `shares` passed in, regardless of how many targets remain.

        Returns True once the stop is confirmed placed (2026-07-28 return value,
        preserved for sync_exit_orders' per-ticker retry backoff below) -- False if it
        never got a working order this call."""
        return await self._place_stop_only(ticker, shares, stop_price)

    async def _build_and_submit_sell(
        self, ticker: str, order_type: OrderType, quantity: float,
        stop_price: float | None = None,
    ) -> Order:
        """Builds a SELL Order and submits it, registering the result in
        self.active_orders (and self._stop_order_ids too, for a STOP order) on
        success. Raises whatever self.broker.submit_order raises on failure --
        callers own their own try/except for logging text and retry decisions;
        this extracts only the mechanical build+submit+register step, which
        _place_stop_only repeated 7 times near-identically across its initial
        attempts and every retry tier (2026-08-24, GitHub #88 -- a fix applied
        to one copy had no mechanism to reach the other 6, exactly the class of
        drift this file's own comments elsewhere already warn about)."""
        order = Order(
            ticker=ticker, side=OrderSide.SELL, order_type=order_type,
            quantity=quantity,
            stop_price=round(stop_price, 2) if stop_price is not None else None,
        )
        result = await self.broker.submit_order(order)
        self.active_orders[result.broker_order_id] = result
        if order_type == OrderType.STOP:
            self._stop_order_ids[ticker] = result.broker_order_id
        return result

    async def _place_stop_only(self, ticker: str, shares: float, stop_price: float) -> bool:
        """The stop-placement half of the old _place_exit_orders, extracted verbatim
        (2026-08-11) so both _place_exit_orders (the normal "no exit orders yet" path)
        and _execute_take_profit_tranche (which replaces the stop after every TP fires)
        share one implementation of this hard-won retry logic, rather than risking the
        two drifting apart. `shares` here is always the FULL quantity the stop should
        cover -- callers never pass a tranche-sized amount.

        Returns True once the stop (or, if the stop price was already breached, the
        entire-position market-sell fallback) is confirmed placed; False if every
        attempt failed."""
        stop_shares = shares
        stop_ok = True   # trivially satisfied unless the block below actually attempts one
        # Set True the moment the stop is found already breached (price already moved
        # past it) -- 2026-07-30, ADC/SCHW incident, still relevant now that stop_shares
        # is always the full position anyway: the market-sell fallback below sells
        # `shares` (100%) either way, this flag just distinguishes which path placed it
        # for the caller's own logging/backoff purposes.
        stop_breached = False

        # ── Stop loss ──
        if stop_price is not None and stop_price > 0 and stop_shares > 0:
            stop_ok = False  # about to attempt; only True once a submit actually succeeds
            try:
                await self._build_and_submit_sell(
                    ticker, OrderType.STOP, stop_shares, stop_price=stop_price)
                logger.info("Stop placed for %s: %.4g shares @ $%.2f", ticker, stop_shares, stop_price)
                stop_ok = True
            except Exception as e:
                err_str = str(e).lower()
                if "stop price must be" in err_str and "current price" in err_str:
                    # Price has already moved through the intended stop level (e.g. a
                    # trailing-stop trigger held off pre-market, then price kept falling
                    # before sync_exit_orders got a chance to re-place a valid stop —
                    # see CLAUDE.md "Auto-close market-open race"). A sell-stop must sit
                    # below current price to be valid, so a rejection with this exact
                    # error means the position should already be sold, not left with an
                    # unplaceable stop.
                    #
                    # Sell the ENTIRE remaining position (`shares`), not just
                    # `stop_shares` (2026-07-30, ADC/SCHW incident) -- once the stop
                    # itself has been breached, there's no good reason to leave the other
                    # half sitting on a take-profit limit order above a price that has
                    # already fallen through the risk-management exit level. The old
                    # "only stop_shares, TP portion still gets placed below" behavior
                    # caused a runaway loop: every subsequent sync_exit_orders pass saw
                    # the same (already-covered) TP portion, miscounted it as a fresh,
                    # unprotected position because pos.shares hadn't been corrected yet,
                    # and re-ran this exact split on the leftover -- market-selling half
                    # of it AGAIN and leaving a smaller TP resting, repeating every ~10s
                    # until the position was consumed via dozens of shrinking fragmented
                    # sells (ADC: 13 fills; SCHW: 4 fills) instead of exiting cleanly
                    # once. stop_breached is set here so the TP block below is skipped
                    # entirely for this call.
                    stop_breached = True
                    logger.warning(
                        "Stop price $%.2f for %s already invalid (price moved past it) — "
                        "selling entire remaining position (%.4g shares) at market instead",
                        stop_price, ticker, shares,
                    )
                    try:
                        await self._build_and_submit_sell(ticker, OrderType.MARKET, shares)
                        logger.info(
                            "Market sell submitted for %s (%.4g shares) — stop was already breached",
                            ticker, shares,
                        )
                        stop_ok = True
                    except Exception as e2:
                        e2_str = str(e2).lower()
                        # Same tiny float-accumulation drift the stop-order retry above
                        # already handles -- confirmed live (HBAN, 2026-07-23): this
                        # fallback kept retrying with the same stale computed quantity
                        # every 10s sync_exit_orders cycle, hitting the identical
                        # sub-billionth-of-a-share mismatch every time instead of
                        # self-correcting, leaving the position genuinely unprotected
                        # for over a minute until an unrelated trigger happened to close
                        # it via a different code path. Parse the broker's own
                        # "available: N" and retry once with that exact quantity, same
                        # fix already proven for the stop-order path above.
                        if "insufficient qty" in e2_str or "insufficient quantity" in e2_str:
                            match = _INSUFFICIENT_QTY_RE.search(str(e2))
                            available = float(match.group(1)) if match else 0.0
                            if available > 0:
                                logger.warning(
                                    "Market-sell fallback for %s rejected on qty (requested "
                                    "%.9f, available %s) — retrying with broker's exact "
                                    "available quantity",
                                    ticker, shares, match.group(1),
                                )
                                try:
                                    await self._build_and_submit_sell(ticker, OrderType.MARKET, available)
                                    logger.info(
                                        "Market sell submitted for %s (%.4g shares, "
                                        "qty-corrected retry) — stop was already breached",
                                        ticker, available,
                                    )
                                    stop_ok = True
                                except Exception as e3:
                                    logger.error(
                                        "Market-sell fallback qty-corrected retry also failed "
                                        "for %s: %s — position remains unprotected until next "
                                        "sync_exit_orders cycle",
                                        ticker, e3,
                                    )
                            else:
                                # available == 0 -- same settlement-lag case as the stop-order
                                # path's available:0 retry, just one level deeper (stop
                                # rejected on price -> market-sell fallback -> THAT rejected
                                # on qty too). Wait and retry once at the original quantity
                                # instead of giving up after a single attempt.
                                logger.warning(
                                    "Market-sell fallback for %s rejected with available:0 "
                                    "(requested %.9f) — waiting 2s for settlement then "
                                    "retrying once",
                                    ticker, shares,
                                )
                                await asyncio.sleep(2)
                                try:
                                    await self._build_and_submit_sell(ticker, OrderType.MARKET, shares)
                                    logger.info(
                                        "Market sell submitted for %s (%.4g shares, "
                                        "available:0 retry) — stop was already breached",
                                        ticker, shares,
                                    )
                                    stop_ok = True
                                except Exception as e3:
                                    logger.error(
                                        "Market-sell fallback available:0 retry also failed "
                                        "for %s: %s — position remains unprotected until next "
                                        "sync_exit_orders cycle",
                                        ticker, e3,
                                    )
                        else:
                            logger.error(
                                "Market-sell fallback also failed for %s: %s — "
                                "position remains unprotected until next sync_exit_orders cycle",
                                ticker, e2,
                            )
                elif "wash trade" in err_str:
                    # Alpaca rejects a sell order submitted immediately after its entry
                    # fills, before Alpaca's own internal settlement has caught up
                    # (2026-07-24, SCHW incident: "potential wash trade detected. use
                    # complex orders" -- bracket/OCO/OTO orders were investigated as the
                    # suggested fix and confirmed incompatible with this system's
                    # notional/fractional trading model, so a short wait-then-retry is
                    # the compatible fix instead). A brief pause is normally enough for
                    # the settlement lag to clear.
                    logger.warning(
                        "Stop order for %s rejected as a potential wash trade — "
                        "waiting 3s for settlement then retrying once", ticker,
                    )
                    await asyncio.sleep(3)
                    try:
                        await self._build_and_submit_sell(
                            ticker, OrderType.STOP, stop_shares, stop_price=stop_price)
                        logger.info(
                            "Stop placed for %s: %.4g shares @ $%.2f (wash-trade retry)",
                            ticker, stop_shares, stop_price,
                        )
                        stop_ok = True
                    except Exception as e2:
                        logger.warning(
                            "Stop order wash-trade retry also failed for %s: %s — "
                            "position remains unprotected until next sync_exit_orders cycle",
                            ticker, e2,
                        )
                elif "insufficient qty" in err_str or "insufficient quantity" in err_str:
                    # Tiny float-accumulation drift between our locally-computed share count
                    # and Alpaca's true available balance (e.g. requesting 2.485746103 shares
                    # when Alpaca's real available is 2.485746102 — a sub-billionth-of-a-share
                    # difference) can make Alpaca reject an otherwise-correct stop order
                    # outright. Parse the broker's own "available: N" figure from the error
                    # and retry once with that exact quantity instead of our computed one.
                    match = _INSUFFICIENT_QTY_RE.search(str(e))
                    available = float(match.group(1)) if match else 0.0
                    if available > 0:
                        logger.warning(
                            "Stop order for %s rejected on qty (requested %.9f, available %s) — "
                            "retrying with broker's exact available quantity",
                            ticker, stop_shares, match.group(1),
                        )
                        try:
                            await self._build_and_submit_sell(
                                ticker, OrderType.STOP, available, stop_price=stop_price)
                            logger.info(
                                "Stop placed for %s: %.4g shares @ $%.2f (qty-corrected retry)",
                                ticker, available, stop_price,
                            )
                            stop_ok = True
                        except Exception as e2:
                            logger.warning("Stop order qty-corrected retry also failed for %s: %s", ticker, e2)
                    else:
                        # available == 0 (not just a tiny float-drift mismatch) -- Alpaca
                        # hasn't finished freeing the shares yet (2026-07-28, MET incident).
                        # The qty-corrected retry above can't help here since Alpaca gave no
                        # usable number to retry with; wait a bit longer for settlement and
                        # retry once at the ORIGINAL requested quantity instead of giving up
                        # after a single attempt.
                        logger.warning(
                            "Stop order for %s rejected with available:0 (requested %.9f) — "
                            "waiting 2s for settlement then retrying once",
                            ticker, stop_shares,
                        )
                        await asyncio.sleep(2)
                        try:
                            await self._build_and_submit_sell(
                                ticker, OrderType.STOP, stop_shares, stop_price=stop_price)
                            logger.info(
                                "Stop placed for %s: %.4g shares @ $%.2f (available:0 retry)",
                                ticker, stop_shares, stop_price,
                            )
                            stop_ok = True
                        except Exception as e2:
                            logger.warning(
                                "Stop order available:0 retry also failed for %s: %s — "
                                "position remains unprotected until next sync_exit_orders cycle",
                                ticker, e2,
                            )
                else:
                    logger.warning("Stop order failed for %s: %s", ticker, e)

        return stop_ok

    async def _execute_sell(self, signal) -> Order | None:
        position = self.portfolio.positions.get(signal.ticker)
        if not position:
            logger.warning("No position to sell for %s", signal.ticker)
            return None

        lock = self._lock_for(signal.ticker)
        if lock.locked():
            logger.warning("_execute_sell: already in progress for %s — dropping duplicate", signal.ticker)
            return None

        # Capture shares before any await — concurrent TP fills can mutate position.shares
        shares_to_sell = position.shares

        async with lock:
            # Clear any pending after-hours buy for this ticker so it isn't re-opened
            self._pending_stops.pop(signal.ticker, None)

            # Cancel any open TP and stop orders for this position
            await self._cancel_exit_orders(signal.ticker)
            # Brief pause so Alpaca finishes processing the cancels before the market sell arrives
            await asyncio.sleep(0.75)

            order = Order(
                ticker=signal.ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=shares_to_sell,
            )

            try:
                result = await self.broker.submit_order(order)
            except Exception as _sell_err:
                _err_str = str(_sell_err).lower()
                # "sold short" (2026-07-22, GitHub #29) -- Alpaca returns a DIFFERENT error
                # string, "fractional orders cannot be sold short", for the exact same
                # underlying problem (selling more shares than actually held) whenever the
                # position involves fractional shares -- confirmed a real, previously-seen
                # message in this codebase's own incident history (the AMGN race, 2026-07-17).
                # A whole-share position hits "insufficient qty" instead, already handled
                # below; fractional positions fell through to the bare re-raise in the `else`
                # branch with no retry, leaving a conviction-drop-to-0 auto-close silently
                # failed and the position open with no completed sell.
                if ("insufficient qty" in _err_str or "insufficient quantity" in _err_str
                        or "sold short" in _err_str):
                    # Local share count is stale (TP fills reduced Alpaca shares while
                    # system was offline or before our state updated).  Re-sync from
                    # Alpaca and retry once with the real available quantity so the
                    # position is actually closed and exit orders can be correctly sized.
                    try:
                        _alpaca_pos = await self.broker.get_positions()
                        _real_shares = next(
                            (ap["shares"] for ap in _alpaca_pos
                             if ap["ticker"] == signal.ticker),
                            None,
                        )
                        if _real_shares and _real_shares > 0.001:
                            logger.warning(
                                "_execute_sell: share-count mismatch for %s "
                                "(local=%.4g Alpaca=%.4g) — correcting and retrying",
                                signal.ticker, shares_to_sell, _real_shares,
                            )
                            shares_to_sell = _real_shares
                            order.quantity = _real_shares
                            if signal.ticker in self.portfolio.positions:
                                self.portfolio.positions[signal.ticker].shares = _real_shares
                                await self.portfolio._save_position(
                                    self.portfolio.positions[signal.ticker])
                            result = await self.broker.submit_order(order)
                        else:
                            # Not found in Alpaca — position may already be closed externally
                            logger.warning(
                                "_execute_sell: %s not found in Alpaca — "
                                "treating as already closed; restoring exit orders",
                                signal.ticker,
                            )
                            asyncio.create_task(self.sync_exit_orders())
                            raise _sell_err
                    except Exception as _retry_err:
                        # Retry also failed — restore exit orders with corrected share count
                        asyncio.create_task(self.sync_exit_orders())
                        raise _retry_err
                else:
                    # Non-qty error — exit orders already cancelled; restore protection
                    asyncio.create_task(self.sync_exit_orders())
                    raise
            self.active_orders[result.broker_order_id] = result

            if result.status == OrderStatus.FILLED:
                fill_price = result.filled_price if result.filled_price is not None else position.current_price
                # signal.reasoning already carries the real, specific reason this sell was
                # initiated (e.g. "Stop loss hit", "Trailing stop hit", "Manual sell",
                # "Portfolio rotation: ...") -- reuse it directly rather than a second,
                # potentially-drifting copy of the same text (2026-07-21).
                await self.portfolio.close_position_async(
                    signal.ticker, exit_shares=shares_to_sell, exit_price=fill_price,
                    reason=getattr(signal, "reasoning", ""))
                try:
                    account = await self.broker.get_account()
                    self.portfolio.cash = account.cash
                    await self.portfolio._save_state()
                except Exception as _e:
                    logger.warning("Cash re-sync after sell of %s failed: %s", signal.ticker, _e)
            elif result.status == OrderStatus.PARTIAL:
                # Some shares sold — re-sync cash and update local share count from Alpaca so
                # sync_exit_orders places exit orders for the correct remaining quantity.
                logger.warning("_execute_sell: PARTIAL fill for %s — re-syncing shares and cash", signal.ticker)
                _partial_qty = result.filled_quantity if result.filled_quantity is not None else 0.0
                _partial_price = result.filled_price if result.filled_price is not None else position.current_price
                if self.portfolio._db and _partial_qty > 0:
                    try:
                        _partial_pnl = (_partial_price - position.entry_price) * _partial_qty
                        if _partial_pnl < 0:
                            self.portfolio.recent_losses[signal.ticker] = datetime.now()
                        await self.portfolio._db.execute(
                            "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (signal.ticker, "SELL", _partial_qty, _partial_price,
                             _partial_pnl, datetime.now().isoformat(),
                             getattr(signal, "reasoning", ""), position.trade_id),
                        )
                        await self.portfolio._db.commit()
                    except Exception as _te:
                        logger.warning("Failed to write partial sell trade_history for %s: %s", signal.ticker, _te)
                try:
                    account = await self.broker.get_account()
                    self.portfolio.cash = account.cash
                    alpaca_positions = await self.broker.get_positions()
                    for ap in alpaca_positions:
                        if ap["ticker"] == signal.ticker and signal.ticker in self.portfolio.positions:
                            self.portfolio.positions[signal.ticker].shares = ap["shares"]
                            await self.portfolio._save_position(self.portfolio.positions[signal.ticker])
                            break
                    await self.portfolio._save_state()
                except Exception as _e:
                    logger.warning("Re-sync after partial sell of %s failed: %s", signal.ticker, _e)
                asyncio.create_task(self.sync_exit_orders())
            elif result.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                # Exit orders were already cancelled, so protection does need restoring if
                # this sell doesn't resolve — but firing sync_exit_orders() immediately races
                # the sell itself: a market sell to a liquid stock can fill within a couple
                # hundred ms, faster than Alpaca's own open-orders listing reflects it (observed
                # live on AMGN, 2026-07-17 — the immediate recovery task tried to re-place a
                # stop, hit "price already invalid" since the fill had already happened, fell
                # back to a second market sell, and that failed with "cannot be sold short"
                # since the position was already gone by then). Harmless that time only because
                # the first sell had already succeeded; not guaranteed in general. Give the
                # order a brief window to resolve on its own before deciding recovery is needed.
                asyncio.create_task(self._delayed_sync_after_pending_sell(result.broker_order_id))
            else:
                # REJECTED, CANCELLED — no order in flight to wait for; restore protection now
                asyncio.create_task(self.sync_exit_orders())

            return result

    async def _delayed_sync_after_pending_sell(self, broker_order_id: str):
        """Wait briefly for a just-submitted PENDING/SUBMITTED sell to resolve on its own
        before running sync_exit_orders() — closes the race where recovery fires so fast it
        collides with the sell's own near-instant fill (see the call site's comment for the
        live incident this was found from). If the order filled during the wait, there's
        nothing to recover — update_positions()'s own poll will pick up the closed position
        normally. If the order can't be confirmed filled (still pending, or the status check
        itself fails), fall through to the original behavior and run sync_exit_orders().
        """
        await asyncio.sleep(1.5)
        try:
            status_order = await self.broker.get_order_status(broker_order_id)
            if status_order.status == OrderStatus.FILLED:
                return
        except Exception:
            pass
        await self.sync_exit_orders()

    async def _cancel_exit_orders(self, ticker: str, open_orders: list | None = None):
        """Cancel all open stop and take-profit orders for a ticker.

        Fetches live Alpaca orders when open_orders is not provided so this
        works correctly after a restart (local dicts are not persisted to DB
        and are empty on restart, making dict-only cancel a no-op).
        """
        if open_orders is None:
            try:
                open_orders = await self.broker.get_open_orders()
            except Exception as e:
                logger.warning("_cancel_exit_orders: get_open_orders failed for %s (%s) — relying on local dicts", ticker, e)
                open_orders = []
        for o in open_orders:
            # Never cancel a pending market sell — that's an active close attempt
            if o["ticker"] == ticker and o["side"] == "sell" and o["type"] != "market":
                await self.broker.cancel_order(o["order_id"])
        # Also clear the local stop tracking entry (may overlap with above — cancel is
        # idempotent). No take-profit dict to clear anymore (2026-08-11 redesign) --
        # there's never a resting TP order for this to find.
        stop_id = self._stop_order_ids.pop(ticker, None)
        if stop_id:
            await self.cancel(stop_id)

    async def _liquidate_remainder_after_stop_fire(self, ticker: str, remaining_shares: float) -> bool:
        """A stop-loss/trailing-stop firing is a "get out" signal, not a partial
        profit-take like a TP fill -- the instant ANY stop-type order fires, even just
        for its own tranche, the rest of the position must close too (2026-08-11, FTV
        incident: owner caught this live -- "if we hit a stop loss, all of the position
        needs to be sold. 100% of it... tp and stop loss is not the same thing").

        Cancels every other resting exit order for this ticker (typically an unfired
        take-profit limit order still covering the remainder) and market-sells the
        entire remainder immediately, instead of leaving it open for
        sync_exit_orders/check_protection_gaps' next ~10s pass to try placing a fresh
        stop and only fall back to a market sell after Alpaca rejects that (price has
        usually already moved past it by then) -- the exact gap that let FTV's
        remaining shares sit exposed for several seconds and exit as a second, separate
        fill instead of closing immediately alongside the first.

        Can't be merged into a single broker-level order with the stop's own fill --
        that fill already happened as its own real event before this ever runs -- but
        this removes the multi-second reactive delay and the wasted doomed-stop
        placement attempt in between. Returns True if the market sell was submitted
        successfully; False (logged, non-fatal) falls through to the existing
        sync_exit_orders/check_protection_gaps safety net exactly as before this fix
        existed."""
        try:
            await self._cancel_exit_orders(ticker)
        except Exception as e:
            logger.warning(
                "%s: failed to cancel resting exit orders before stop-triggered "
                "full close (%s)", ticker, e,
            )
        try:
            market_order = Order(
                ticker=ticker, side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=round(remaining_shares, 9),
            )
            result = await self.broker.submit_order(market_order)
            self.active_orders[result.broker_order_id] = result
            logger.info(
                "%s: stop-type order fired — immediately closing the remaining %.4g "
                "shares at market (a stop-loss/trailing-stop always closes the whole "
                "position, never leaves a fraction resting on a take-profit order)",
                ticker, remaining_shares,
            )
            return True
        except Exception as e:
            logger.warning(
                "%s: immediate stop-triggered full close failed (%s) — falling back "
                "to the normal sync_exit_orders/check_protection_gaps cycle",
                ticker, e,
            )
            return False

    async def cancel(self, broker_order_id: str) -> bool:
        success = await self.broker.cancel_order(broker_order_id)
        if success:
            self.active_orders.pop(broker_order_id, None)
        return success

    @staticmethod
    def _position_is_covered(pos, stop_orders: dict) -> bool:
        """True if `pos` currently has correct broker-side protection (2026-08-11
        redesign — see _place_exit_orders' docstring): a stop order at the right
        price, sized for the FULL position. There's no separate take-profit order to
        account for anymore, since take-profit is no longer a resting order at all —
        it's an active, independent action (check_take_profits/
        _execute_take_profit_tranche) that never leaves anything resting except the
        stop. Extracted from sync_exit_orders' per-ticker check (2026-07-21) so the
        same logic backs both the hourly reconcile-and-fix pass and the 10s read-only
        verification in check_protection_gaps below — a single source of truth for "is
        this covered"."""
        stop_order = stop_orders.get(pos.ticker)
        if stop_order is None:
            return False
        intended_stop = round(max(pos.stop_loss, pos.trailing_stop or 0), 2)
        stop_price_ok = (
            stop_order.get("stop_price") is not None
            and abs(round(stop_order["stop_price"], 2) - intended_stop) < 0.005
        )
        # Sized for the whole position, not just part of it (2026-07-24, SCHW incident
        # — a position whose share count was captured wrong could sit with a
        # correctly-priced but undersized stop indefinitely without this check).
        covered_qty = stop_order.get("qty") or 0.0
        qty_ok = covered_qty >= pos.shares - 0.05
        return stop_price_ok and qty_ok

    async def check_protection_gaps(self) -> list[dict]:
        """Read-only verification: does every held position actually have correct
        broker-side protection right now? Runs on the 10s position_update_loop cadence
        (2026-07-21) instead of only being caught by sync_exit_orders' hourly pass —
        EPRT sat with zero real stop/TP orders at Alpaca for roughly a day before this
        existed. Returns [] on any broker error (fail-open — can't verify, so don't
        raise a false alarm on top of an API hiccup). Never places or cancels an order
        itself; the caller decides whether/how to alert and can fire sync_exit_orders()
        as an immediate follow-up fix."""
        if not self.broker:
            return []
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as e:
            logger.debug("check_protection_gaps: get_open_orders failed (%s) — skipping this cycle", e)
            return []

        stop_orders = {o["ticker"]: o for o in open_orders
                       if o["side"] == "sell" and o["type"] in ("stop", "stop_limit")}
        pending_market_sell = {o["ticker"] for o in open_orders
                                if o["side"] == "sell" and o["type"] == "market"}

        gaps: list[dict] = []
        for ticker, pos in list(self.portfolio.positions.items()):
            if self._lock_for(ticker).locked():
                continue  # mid-operation elsewhere (buy/sell/sync in flight) — not a gap
            if ticker in pending_market_sell:
                continue  # active close attempt already in flight — not a gap
            if pos.stop_loss <= 0:
                gaps.append({"ticker": ticker, "reason": f"stop_loss is {pos.stop_loss} — no downside protection configured"})
                continue
            if not self._position_is_covered(pos, stop_orders):
                gaps.append({"ticker": ticker, "reason": "missing or mispriced stop order at the broker"})
        return gaps

    async def _try_replace_stale_stop(self, ticker: str, pos: Position, existing_stop: dict | None) -> bool:
        """Attempts the routine case first (2026-07-28, MET incident): if a stop order
        already exists for this ticker at the RIGHT quantity and only its PRICE is stale
        (the trailing stop ratcheted up since it was placed), replace it in place via
        Alpaca's PATCH /v2/orders/{id} instead of the heavier cancel-then-resubmit
        sequence sync_exit_orders otherwise always uses. This is Alpaca's own documented
        mechanism for adjusting an existing order's price, and it sidesteps the
        release-then-reserve race that left MET briefly unprotected: cancel+resubmit
        briefly shows the just-cancelled order's shares as unavailable to the very next
        placement attempt (confirmed live), while a same-order replace never releases
        them in the first place.

        Deliberately narrow, by design: only handles "stop exists, qty already correct,
        price wrong". Every other case (no existing stop, wrong quantity, replace
        unsupported by this broker, or the replace call itself failing for any reason —
        including Alpaca rejecting it because the order is already
        accepted/pending_new/pending_cancel/pending_replace, none of which are
        replaceable per Alpaca's docs) falls through untouched to the existing
        cancel-then-place path, exactly as it worked before this method existed. This
        makes the method fail-safe by construction: its only way to report "handled" is
        a confirmed-successful replace that directly set the correct price, so a bug here
        can make sync_exit_orders do MORE work (an unnecessary cancel+place) but can
        never leave a position less protected than the pre-existing path already did.

        Returns True if the replace succeeded (caller should treat this ticker as handled
        for this pass) or False if this case doesn't apply / the replace failed (caller
        must fall back to the existing cancel+place path)."""
        if not existing_stop or not hasattr(self.broker, "replace_order"):
            return False
        intended_stop = round(max(pos.stop_loss, pos.trailing_stop or 0), 2)
        current_stop_price = existing_stop.get("stop_price")
        if current_stop_price is not None and abs(round(current_stop_price, 2) - intended_stop) < 0.005:
            # Price is already correct -- whatever made this ticker "not covered" isn't a
            # stale stop price, so a price-only replace can't be the fix.
            return False
        # The stop always covers the FULL position now (2026-08-11 redesign) -- no
        # tranche split to compute anymore.
        stop_shares = pos.shares
        existing_qty = existing_stop.get("qty") or 0.0
        if stop_shares <= 0 or abs(existing_qty - stop_shares) > 1e-6:
            return False  # quantity itself needs to change too -- not a simple price edit
        order_id = existing_stop.get("order_id")
        if not order_id:
            return False
        # Stale-broker-read guard (2026-07-28, DV incident) -- self._stop_order_ids is
        # updated synchronously the instant OUR OWN replace succeeds, so it can never lag
        # reality. Alpaca's open-orders LISTING (what existing_stop was built from) can
        # briefly still show the just-superseded id for a second or more afterward
        # (confirmed live). If our own tracking already moved past this id, a replace
        # against it is doomed and would otherwise fall through to a cancel+place that
        # treats real, still-held shares as free -- the ticker is already covered under
        # the newer id, so skip entirely instead of attempting it.
        if ticker in self._stop_order_ids and self._stop_order_ids[ticker] != order_id:
            return True
        new_id = await self.broker.replace_order(order_id, stop_price=intended_stop)
        if not new_id:
            return False
        self._stop_order_ids[ticker] = new_id
        logger.info(
            "Stop replaced in place for %s: %.4g shares, $%s -> $%.2f (order %s -> %s)",
            ticker, stop_shares, current_stop_price, intended_stop, order_id, new_id,
        )
        return True

    async def sync_exit_orders(self):
        """Place exit orders for any held position that has no open sell orders in Alpaca."""
        if not self.broker:
            return
        if self._sync_in_progress:
            # A pass is already running system-wide (_sync_in_progress is a single global
            # lock, not per-ticker — a real single pass through all positions can't safely
            # run concurrently with another, since both would independently fetch/cancel/
            # place). But simply dropping this call is what caused the 2026-07-17 incident:
            # a ticker whose own _execute_sell was still mid-flight when the in-progress
            # pass reached it gets correctly skipped for THAT pass (see the per-ticker
            # lock check below), and if this was the only other trigger for that ticker,
            # it never got a real recovery attempt at all. Instead of no-op'ing, request a
            # follow-up pass — the in-progress run loops once more before releasing the
            # lock, so any ticker skipped this time gets picked up once its blocker clears.
            self._sync_rerun_requested = True
            logger.debug("sync_exit_orders: already running — requesting a follow-up pass")
            return
        self._sync_in_progress = True
        try:
            max_passes = 5  # bounded — a ticker stuck locked by another operation
            # indefinitely (should never happen, but must not be able to spin this loop
            # forever) falls through to the next external trigger (hourly loop, or the
            # blocked call's own recovery task) instead of hammering the broker API.
            for _pass_num in range(max_passes):
                self._sync_rerun_requested = False
                if _pass_num > 0:
                    await asyncio.sleep(1)  # give the blocker a moment to actually clear
                try:
                    open_orders = await self.broker.get_open_orders()
                except Exception as e:
                    # Incomplete order list would cause us to place duplicate orders — skip entirely
                    logger.warning("sync_exit_orders: get_open_orders failed (%s) — skipping to avoid orphaning orders", e)
                    return

                # Only stop-type orders provide downside protection (2026-08-11 redesign
                # — take-profit is no longer a resting order at all, so it's never part
                # of coverage). Keep the actual order (not just presence) so we can also
                # verify its price — a stop order existing at a stale price (e.g. the
                # trailing stop ratcheted up since the order was placed) must NOT count
                # as covered, or it silently never gets updated to reflect the current
                # protective level.
                stop_orders = {o["ticker"]: o for o in open_orders
                               if o["side"] == "sell" and o["type"] in ("stop", "stop_limit")}
                for ticker, pos in list(self.portfolio.positions.items()):
                    # Shared with check_protection_gaps' 10s read-only verification
                    # (2026-07-21) — one source of truth for "is this position covered."
                    if self._position_is_covered(pos, stop_orders):
                        continue
                    # Fall through: missing or undersized stop — re-place it
                    lock = self._lock_for(ticker)
                    if lock.locked():
                        logger.debug("sync_exit_orders: skipping %s — locked by another operation", ticker)
                        self._sync_rerun_requested = True
                        continue
                    # Market-sell-in-flight check -- refined 2026-07-30 (BEN incident): a
                    # resting market sell only means "skip this ticker entirely" if it
                    # covers the WHOLE remaining position (the original 2026-07-28
                    # assumption, true for _execute_sell's manual/auto-close full-position
                    # sells -- cancelling one of those really would leave the position
                    # briefly unprotected). _place_exit_orders' OWN stop-fallback market
                    # sell only ever covers the STOP TRANCHE portion, though -- when it
                    # doesn't cover everything, the remaining (TP-tranche) shares still
                    # need their own protection, and skipping the whole ticker here left
                    # them permanently uncovered until the pending sell resolved (confirmed
                    # live: BEN sat with 6.93 of 20.79 shares completely unprotected for
                    # several minutes, since every subsequent pass hit this same skip).
                    # Only the genuinely-still-uncovered remainder is placed for below --
                    # _cancel_exit_orders already never touches a market-type order, so the
                    # resting partial sell is left alone either way.
                    pending_market_sell_qty = sum(
                        o.get("qty") or 0.0 for o in open_orders
                        if o["ticker"] == ticker and o["side"] == "sell" and o["type"] == "market"
                    )
                    if pending_market_sell_qty >= pos.shares - 0.001:
                        logger.debug(
                            "sync_exit_orders: skipping %s — pending market sell already "
                            "covers the full position", ticker,
                        )
                        continue
                    remaining_shares = pos.shares - pending_market_sell_qty
                    # No early skip for an empty take_profit_targets list (2026-07-30, DV
                    # incident) -- 0 remaining targets means the same thing 1 remaining
                    # target already meant (final tranche, stop-only, ride the trailing
                    # stop -- see _compute_tranche_split), not "nothing can be done."
                    # _place_exit_orders below now places a stop for the full remaining
                    # position in that case instead of being called at all here.
                    if pos.stop_loss <= 0:
                        logger.warning(
                            "Skipping exit orders for %s — stop_loss is %.2f, no downside protection",
                            ticker, pos.stop_loss or 0,
                        )
                        continue
                    # Per-ticker remediation backoff (2026-07-28, MET incident) -- skip this
                    # cycle's attempt entirely if a recent attempt already failed and the
                    # backoff window hasn't elapsed yet. Placed after every guard above so
                    # those keep taking priority exactly as before; this only throttles
                    # genuine repeated placement failures, never detection itself (that stays
                    # on check_protection_gaps' unmodified 10s cadence).
                    _backoff = self._exit_order_retry_backoff.get(ticker)
                    if _backoff:
                        _next_ok_at, _fail_count = _backoff
                        if datetime.now() < _next_ok_at:
                            logger.debug(
                                "sync_exit_orders: skipping %s — backing off after %d consecutive "
                                "placement failure(s), next retry at %s",
                                ticker, _fail_count, _next_ok_at.isoformat(timespec="seconds"),
                            )
                            continue
                    # Hold the ticker's lock for the whole replace-or-cancel-then-replace
                    # sequence so any other function (execute_sell, check_take_profits) that
                    # checks .locked() sees this ticker as busy for the entire span, not just
                    # individual calls. Safe to enter directly here (no await since the
                    # lock.locked() check above) — nothing else can have grabbed it in that gap.
                    async with lock:
                        # Use whichever is higher: original stop or trailing stop.
                        # Trailing stop only moves up, so this safely ratchets the
                        # Alpaca hard stop to reflect locked-in profit protection.
                        effective_stop = max(pos.stop_loss, pos.trailing_stop or 0)
                        # Try the routine "stop exists, just needs a fresh price" case via an
                        # in-place replace first (2026-07-28) -- Alpaca's own documented way
                        # to adjust an existing order's price, and it sidesteps the cancel+
                        # place release-then-reserve race entirely for the common trailing-
                        # stop-ratchet renewal. Falls through unchanged to the existing
                        # cancel+place path for every other case (missing order, wrong qty,
                        # replace unsupported/rejected).
                        if await self._try_replace_stale_stop(ticker, pos, stop_orders.get(ticker)):
                            self._exit_order_retry_backoff.pop(ticker, None)
                            continue
                        logger.info(
                            "No exit orders found for %s (%.4g shares, %d targets) — placing now",
                            ticker, remaining_shares, len(pos.take_profit_targets),
                        )
                        # Cancel ALL open Alpaca sell orders for this ticker — including any
                        # that existed before the last restart (order IDs are not persisted to
                        # DB, so _cancel_exit_orders alone is a no-op after a restart and
                        # leaves stale orders that cause "insufficient qty" errors otherwise).
                        await self._cancel_exit_orders(ticker, open_orders=open_orders)
                        # Settle pause (2026-07-28, MET incident) -- Alpaca confirms a cancel
                        # before it actually frees the shares as "available" (a real,
                        # measurable lag, confirmed live), so placing immediately after a
                        # cancel can see available:0 and fail. Same pattern already
                        # established twice elsewhere in this file (the 0.75s stop-then-TP
                        # pause, the 3s wash-trade retry pause) -- this doesn't GUARANTEE
                        # Alpaca has settled by the time it elapses, which is why the
                        # insufficient-qty retry paths below still exist as a second layer,
                        # but it closes the gap for the overwhelming majority of cases instead
                        # of racing it on every single cancel+place cycle.
                        await asyncio.sleep(1.0)
                        # Fresh single-ticker share-count verification right before
                        # computing the split (2026-07-29, BAC incident) -- this pass's
                        # own open_orders snapshot was fetched ONCE, before the per-ticker
                        # loop started, so it can't see a fill or a new order that a
                        # CONCURRENT _execute_sell call (a different code path, correctly
                        # serialized via the same _lock_for for CODE EXECUTION, but not
                        # for how stale this pass's already-fetched data is) produced
                        # after that snapshot was taken. Confirmed live: BAC's local
                        # pos.shares (3.709) was still the pre-fill value when this
                        # function computed a split, while Alpaca's real qty had already
                        # dropped to 1.854 because an earlier order from the SAME cascade
                        # had already filled -- producing a stop-fallback sized for
                        # shares that no longer existed and a rejected "cannot be sold
                        # short" TP. get_position (Alpaca-specific, not part of the
                        # shared Broker ABC) gives one final, real-time qty check;
                        # it only CORRECTS a stale local
                        # count, never skips the placement outright based on
                        # qty_available -- Alpaca's exact settlement-lag timing isn't
                        # fully documented, and the existing insufficient-qty/
                        # available:0 retry tiers below already own the "still
                        # settling" case; this only fixes the "our number was flat-out
                        # wrong" case.
                        _get_position = getattr(self.broker, "get_position", None)
                        if _get_position is not None:
                            _real_pos = await _get_position(ticker)
                            if _real_pos is None:
                                logger.info(
                                    "sync_exit_orders: %s has no real Alpaca position "
                                    "anymore — skipping placement this pass (already "
                                    "closed by another operation)", ticker,
                                )
                                continue
                            if abs(_real_pos["shares"] - pos.shares) > 0.001:
                                logger.warning(
                                    "sync_exit_orders: %s local share count (%.4g) was "
                                    "stale vs Alpaca's real-time %.4g — correcting "
                                    "before placing exit orders",
                                    ticker, pos.shares, _real_pos["shares"],
                                )
                                pos.shares = _real_pos["shares"]
                                await self.portfolio._save_position(pos)
                                # Re-derive the still-uncovered remainder against the
                                # corrected total (2026-07-30, BEN incident) -- pos.shares
                                # may have just changed above; pending_market_sell_qty
                                # itself is still accurate (computed from this same pass's
                                # open_orders, and _cancel_exit_orders never touches a
                                # market-type order, so nothing has changed it since).
                                remaining_shares = pos.shares - pending_market_sell_qty
                        placed_ok = await self._place_exit_orders(
                            ticker, remaining_shares,
                            effective_stop, pos.take_profit_targets,
                        )
                        if placed_ok:
                            self._exit_order_retry_backoff.pop(ticker, None)
                        else:
                            # 2026-07-28 -- record the failure and back off, instead of
                            # letting the next 10s check_protection_gaps cycle immediately
                            # re-trigger the identical cancel+place sequence into the same
                            # unresolved race (this is exactly what happened to MET: Alpaca
                            # hadn't finished settling the cancel, so every 10s retry hit
                            # "insufficient qty available (available: 0)" again).
                            _prev_fail_count = self._exit_order_retry_backoff.get(ticker, (None, 0))[1]
                            _fail_count = _prev_fail_count + 1
                            _delay = _EXIT_ORDER_RETRY_BACKOFF_SECONDS[
                                min(_fail_count - 1, len(_EXIT_ORDER_RETRY_BACKOFF_SECONDS) - 1)]
                            self._exit_order_retry_backoff[ticker] = (
                                datetime.now() + timedelta(seconds=_delay), _fail_count)
                            logger.warning(
                                "sync_exit_orders: placement still incomplete for %s after %d "
                                "consecutive attempt(s) — backing off %ds before retrying",
                                ticker, _fail_count, _delay,
                            )
                if not self._sync_rerun_requested:
                    break
                logger.debug("sync_exit_orders: running a follow-up pass (rerun requested during this run)")
        finally:
            self._sync_in_progress = False

    async def check_take_profits(self):
        """Actively checks every held position's current price against its next
        take-profit target and executes that tranche if reached (2026-08-11 redesign
        -- see _place_exit_orders' docstring). Take-profit is no longer a resting
        order this function polls the status of; there IS no resting TP order to poll
        anymore. `pos.take_profit_targets[-1]` (when exactly one target remains) is
        deliberately never checked here -- it's a reference price for the graduated
        trailing stop's own curve, not a real, actionable target (same "final tranche
        rides the trailing stop alone" design this codebase already had before this
        redesign, just no longer expressed as tp_shares == 0 in a tranche split)."""
        if not self.broker:
            return
        for ticker, pos in list(self.portfolio.positions.items()):
            if len(pos.take_profit_targets) < 2:
                continue
            if pos.current_price is None or pos.current_price < pos.take_profit_targets[0]:
                continue
            await self._execute_take_profit_tranche(ticker)

    async def _execute_take_profit_tranche(self, ticker: str) -> bool:
        """Executes exactly one take-profit tranche as its own fully independent
        action (2026-08-11 redesign, FTV incident — owner: "ITS NOT STOP LOSS/TAKE
        PROFIT...... ITS A STOP LOSS, THERE SEPARATE ENTIRELY"). Cancels the current
        stop (which was covering 100% of the position, per _place_exit_orders),
        market-sells just this one tranche, records the fill directly, and places a
        fresh 100%-of-remainder stop.

        Deliberately does NOT rely on the generic _reconcile_untracked_fill fallback
        to record this fill -- that logic distinguishes a real take-profit from a
        stop-driven closure by checking order_type == "limit", and this design
        deliberately never places a limit order for a take-profit anymore (everything
        here is a market order, consistent with "the stop is the only thing that ever
        rests"). Relying on that check here would misclassify every genuine
        take-profit as a stop-loss — recording it directly, the way this function
        already knows for certain what just happened, avoids that entirely.

        Returns True if the tranche sold and (if shares remain) the new stop was
        placed; False on any failure, in which case the position is left exactly as
        protected as sync_exit_orders/check_protection_gaps' existing safety net would
        restore it to on its own next cycle -- same fail-safe-by-construction pattern
        as every other order-placement path in this file."""
        if self._lock_for(ticker).locked():
            return False
        async with self._lock_for(ticker):
            pos = self.portfolio.positions.get(ticker)
            if pos is None or len(pos.take_profit_targets) < 2:
                return False  # re-check under the lock -- state may have changed
            target_price = pos.take_profit_targets[0]
            tranche_number = 4 - len(pos.take_profit_targets)  # 3 remaining -> T1, 2 -> T2
            _, tranche_shares, _, _, _ = _compute_tranche_split(pos.shares, pos.take_profit_targets)
            if tranche_shares <= 0.001:
                return False

            # Cancel the current stop first -- it's resting for 100% of pos.shares,
            # which includes the tranche about to sell; Alpaca reserves shares against
            # a resting order, so the tranche sell would be rejected for insufficient
            # quantity otherwise (verified against Alpaca's real docs, 2026-08-11).
            old_stop = self._stop_order_ids.pop(ticker, None)
            if old_stop:
                await self.cancel(old_stop)
                await asyncio.sleep(0.75)  # let Alpaca settle the cancellation

            sell_qty = round(tranche_shares, 9)
            try:
                tp_order = Order(
                    ticker=ticker, side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=sell_qty,
                )
                result = await self.broker.submit_order(tp_order)
                self.active_orders[result.broker_order_id] = result
            except Exception as e:
                # Same settlement-lag "insufficient qty" class this codebase has hit
                # repeatedly for the stop leg (see _place_stop_only) -- the stop was
                # JUST cancelled above, and Alpaca doesn't always free those shares as
                # immediately available. One retry with the broker's own exact
                # available quantity (or, if unparseable, a short wait and a retry at
                # the original quantity) before giving up and restoring protection.
                err_str = str(e).lower()
                retried = False
                if "insufficient qty" in err_str or "insufficient quantity" in err_str:
                    match = _INSUFFICIENT_QTY_RE.search(str(e))
                    available = float(match.group(1)) if match else 0.0
                    if available > 0:
                        sell_qty = available
                    else:
                        await asyncio.sleep(2)
                    try:
                        tp_order = Order(
                            ticker=ticker, side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=sell_qty,
                        )
                        result = await self.broker.submit_order(tp_order)
                        self.active_orders[result.broker_order_id] = result
                        retried = True
                    except Exception as e2:
                        logger.warning(
                            "Take-profit T%d retry also failed for %s: %s",
                            tranche_number, ticker, e2,
                        )
                if not retried:
                    logger.warning(
                        "Take-profit T%d execution failed for %s: %s — restoring a "
                        "fresh 100%% stop for the untouched position",
                        tranche_number, ticker, e,
                    )
                    await self._place_stop_only(
                        ticker, pos.shares, max(pos.stop_loss, pos.trailing_stop or 0))
                    return False
            tranche_shares = sell_qty

            # Same fallback convention as _execute_sell's own market-sell handling —
            # a market order to a liquid position usually fills fast enough that
            # Alpaca's synchronous response already carries filled_price; when it
            # doesn't, the last known quote is the best available honest estimate.
            fill_price = result.filled_price if result.filled_price is not None else pos.current_price
            entry_price = pos.entry_price
            pnl = (fill_price - entry_price) * tranche_shares
            self.portfolio.cash += tranche_shares * fill_price
            pos.realized_pnl += pnl
            pos.shares_sold += tranche_shares
            pos.shares = round(pos.shares - tranche_shares, 9)
            pos.take_profit_targets = pos.take_profit_targets[1:]
            if pos.trailing_stop is None:
                pos.trailing_stop = entry_price
                logger.info(
                    "%s trailing stop initialized to breakeven $%.2f on TP fill",
                    ticker, entry_price,
                )
            await self.portfolio._save_position(pos)
            await self.portfolio._save_state()

            if self.portfolio._db:
                try:
                    if pnl < 0:
                        self.portfolio.recent_losses[ticker] = datetime.now()
                    await self.portfolio._db.execute(
                        "INSERT INTO trade_history "
                        "(ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (ticker, "SELL", tranche_shares, fill_price, pnl,
                         datetime.now().isoformat(), f"Take-Profit T{tranche_number}",
                         pos.trade_id),
                    )
                    await self.portfolio._db.commit()
                except Exception as e:
                    logger.warning(
                        "Failed to log TP%d trade_history for %s: %s",
                        tranche_number, ticker, e,
                    )

            logger.info(
                "Take-profit T%d executed for %s: %.4g shares @ $%.2f (target was "
                "$%.2f) — %.4g shares remain",
                tranche_number, ticker, tranche_shares, fill_price, target_price, pos.shares,
            )

            if pos.shares <= 0.001:
                # Nothing left to protect -- remove the position entirely. Not
                # close_position_async: this tranche's trade_history row was already
                # written above with the correct T{n} reason: a second call would
                # double-record (and close_position_async's own docstring/comment
                # confirms cash was already credited per-fill, matching this design).
                self.portfolio.positions.pop(ticker, None)
                self.portfolio.update_peak()
                await self.portfolio._remove_position_db(ticker)
                await self.portfolio._save_state()
                logger.info("Position %s fully closed via take-profit", ticker)
                return True

            stop_placed = await self._place_stop_only(
                ticker, pos.shares, max(pos.stop_loss, pos.trailing_stop or 0))
            if not stop_placed:
                logger.error(
                    "%s: fresh 100%% stop failed to place after TP T%d — position "
                    "temporarily unprotected until sync_exit_orders' next cycle",
                    ticker, tranche_number,
                )
            return stop_placed

    async def update_positions(self) -> list[dict]:
        """Returns a list of {ticker, shares, fill_price, pnl} dicts, one per position
        this call detected closed directly by a standing Alpaca order (stop/TP fill) —
        as opposed to a close the app itself initiated. OrderManager has no reference to
        the dashboard's ai_log/broadcast/notify machinery (by design — avoids a circular
        import), so it cannot surface these itself; the caller (DashboardState) uses this
        return value to log/broadcast/notify exactly as it does for app-initiated closes."""
        closed_reports: list[dict] = []
        if not self.broker:
            return closed_reports
        try:
            positions = await self.broker.get_positions()
            alpaca_tickers = {p["ticker"] for p in positions}

            shares_corrected = False
            for p in positions:
                ticker = p["ticker"]
                if ticker in self.portfolio.positions:
                    pos = self.portfolio.positions[ticker]
                    pos.current_price = p["current_price"]
                    # Sync share count when this ticker isn't locked elsewhere (fixed
                    # 2026-08-02, GitHub #40) -- an in-flight _execute_sell or (since the
                    # 2026-08-11 redesign) _execute_take_profit_tranche holds this exact
                    # lock for its whole duration, so checking .locked() alone is
                    # sufficient to avoid racing either one's own share-count update; the
                    # sibling apparent_closes block below already guards the same way.
                    # Deferring here just means the next 10s cycle picks it up once the
                    # lock is free.
                    if (abs(p["shares"] - pos.shares) > 0.01
                            and not self._lock_for(ticker).locked()):
                        old_shares = pos.shares
                        new_shares = p["shares"]
                        # Determine what actually happened (real TP fill vs stop-driven
                        # closure) via Alpaca's own order history BEFORE deciding how many
                        # targets to pop -- same fix as _sync_portfolio above, and for the
                        # same reason (2026-07-28, RRC incident): a stop-loss can cover
                        # multiple tranches' worth of shares in one gap-through market sell,
                        # and pre-guessing "shares_dropped / tranche_size" targets as fired
                        # mislabels a loss as profit-taking. Must NOT pre-shrink
                        # pos.take_profit_targets before calling _reconcile_untracked_fill.
                        #
                        # pos.shares itself is deliberately NOT committed until AFTER this
                        # call returns (2026-08-11, SBRA incident) -- if the real order
                        # behind this delta is still partially_filled,
                        # _reconcile_untracked_fill raises FillStillSettlingError instead of
                        # guessing, and this whole correction (share count included) must be
                        # skipped for this tick so the next poll retries once the order
                        # actually settles. Committing pos.shares here regardless (the old
                        # behavior) would have made the gap permanently unrecoverable: the
                        # next poll would see p["shares"] already matching pos.shares and
                        # never call this again for the same fill.
                        _est = None
                        _settling = False
                        if (old_shares > 0.001
                                and pos.take_profit_targets
                                and new_shares < old_shares - 0.01):
                            _shares_dropped_upd = old_shares - new_shares
                            try:
                                _est = await self._reconcile_untracked_fill(
                                    ticker, _shares_dropped_upd, old_shares,
                                    pos.take_profit_targets, pos.entry_price,
                                )
                            except FillStillSettlingError:
                                _settling = True
                        if _settling:
                            logger.info(
                                "update_positions: %s — real order still settling, "
                                "deferring share-count correction to next poll",
                                ticker,
                            )
                        else:
                            pos.shares = new_shares
                            if _est is not None:
                                _avg_price, _pnl, n_fired = _est
                                pos.realized_pnl += _pnl
                                pos.shares_sold += _shares_dropped_upd
                                if n_fired > 0:
                                    logger.warning(
                                        "update_positions: %s shares %.4g→%.4g — "
                                        "%d TP fill(s) confirmed, trimming targets from %s",
                                        ticker, old_shares, pos.shares,
                                        n_fired, pos.take_profit_targets,
                                    )
                                    pos.take_profit_targets = pos.take_profit_targets[n_fired:]
                                else:
                                    logger.warning(
                                        "update_positions: %s shares %.4g→%.4g — confirmed "
                                        "stop-loss (not a TP fill), targets unchanged: %s",
                                        ticker, old_shares, pos.shares,
                                        pos.take_profit_targets,
                                    )
                                    # A stop firing closes the WHOLE position immediately,
                                    # not just its own tranche (2026-08-11, FTV incident —
                                    # see _liquidate_remainder_after_stop_fire's docstring).
                                    # Only fires if genuinely still open after this poll's
                                    # own correction -- nothing to liquidate on an already-
                                    # full close.
                                    if pos.shares > 0.001:
                                        await self._liquidate_remainder_after_stop_fire(
                                            ticker, pos.shares,
                                        )
                                # Same breakeven-protection rule as the live check_take_profits()
                                # path — a TP fill detected after the fact still means remaining
                                # shares should never be protected at less than breakeven.
                                # Deliberately NOT applied when n_fired == 0 (a stop-driven
                                # closure) -- see _sync_portfolio's matching comment above.
                                if n_fired > 0 and pos.trailing_stop is None:
                                    pos.trailing_stop = pos.entry_price
                                    logger.info(
                                        "%s trailing stop initialized to breakeven $%.2f "
                                        "on untracked TP fill",
                                        ticker, pos.entry_price,
                                    )
                            await self.portfolio._save_position(pos)
                            shares_corrected = True
                elif ticker in self._pending_stops:
                    # After-hours buy just filled — place exit orders now
                    #
                    # 2026-08-05, SBRA incident: this used to trust p["shares"] (from
                    # get_positions(), a POSITIONS-endpoint snapshot) directly, and popped
                    # _pending_stops[ticker] unconditionally. Per Alpaca's documented order
                    # lifecycle, a fractional/notional market order can settle across
                    # multiple pieces at the exchange, and the positions snapshot can
                    # reflect an INTERMEDIATE quantity mid-settlement, before the order's
                    # own status ever reaches a genuinely terminal "filled" state with its
                    # true final cumulative filled_qty (the same distinction already
                    # trusted elsewhere in this file, e.g. the OVV/BEN fixes' terminal-
                    # fill-event handling). Confirmed live: SBRA's real fill was 32.54
                    # shares, but this poll caught the positions snapshot at 7 shares mid-
                    # settlement, created the position and sized every exit order to that
                    # wrong number — and since popping _pending_stops here removes the only
                    # thing the trade_updates stream's own (correct) resolution keys off
                    # of, nothing ever went back to fix it. 25.54 real shares sat
                    # completely unprotected at Alpaca until caught live and manually
                    # corrected.
                    #
                    # Fixed by checking the ORDER's own status first (the authoritative
                    # source for whether a fill is truly final, not the positions
                    # snapshot) -- only resolves once genuinely `filled`; otherwise leaves
                    # _pending_stops in place so a later poll (or the stream) can resolve
                    # it correctly once settlement actually completes.
                    pending = self._pending_stops[ticker]
                    order_id = pending.get("order_id")
                    order_status = None
                    if order_id:
                        try:
                            order_status = await self.broker.get_order_status(order_id)
                        except Exception as e:
                            logger.debug("%s: order status check failed (%s) — will retry next cycle", ticker, e)
                    if order_status is None:
                        continue  # status check failed -- leave pending, retry later
                    if order_status.status in (
                        OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL,
                    ):
                        continue  # not genuinely settled yet -- leave pending, retry later
                    # Reaching here means a TERMINAL status: FILLED, or CANCELLED/REJECTED/
                    # EXPIRED (2026-08-08, GitHub #49) -- a day order that partially fills
                    # before Alpaca auto-cancels/expires the remainder at end of day leaves
                    # real, nonzero shares held with a status that will NEVER become FILLED.
                    # Treating that identically to "still pending" (the old `!= FILLED`
                    # check) meant this ticker looped here forever every cycle, with those
                    # shares sitting completely unprotected (no _place_exit_orders ever
                    # called) until a restart happened to catch it via _sync_portfolio's
                    # separate reconciliation path. Now resolves immediately with whatever
                    # genuinely filled, rather than waiting for a status that can't arrive.
                    #
                    # pop(ticker, None) not pop(ticker) (2026-08-08, GitHub #47) -- the
                    # concurrent trade_updates stream handler can resolve this same pending
                    # buy first (it does its own pop(ticker, None) safely) while this
                    # get_order_status await was in flight; a bare pop(ticker) then raises
                    # KeyError, aborting the rest of this poll cycle (remaining tickers,
                    # apparent_closes detection, check_take_profits, peak-value update) for
                    # no reason -- the position was already resolved correctly by the stream.
                    pending = self._pending_stops.pop(ticker, None)
                    if pending is None:
                        continue  # already resolved by the trade_updates stream while we awaited above
                    real_shares = order_status.filled_quantity or 0.0
                    if order_status.status != OrderStatus.FILLED and real_shares <= 0:
                        logger.info(
                            "%s: pending order %s ended %s with no fill at all -- nothing to track",
                            ticker, order_id, order_status.status.value,
                        )
                        continue
                    if order_status.status != OrderStatus.FILLED:
                        logger.warning(
                            "%s: pending order %s ended %s after a partial fill (%.4g shares) -- "
                            "resolving the position with what actually filled instead of "
                            "waiting indefinitely for a status that will never arrive",
                            ticker, order_id, order_status.status.value, real_shares,
                        )
                    _pending_targets = pending.get("take_profit_targets", [])
                    # order_status.filled_price is the order's true weighted-average fill
                    # price (Alpaca's real filled_avg_price, mapped in get_order_status) --
                    # p["entry_price"] is a positions-endpoint snapshot that can reflect an
                    # intermediate mid-settlement price for a multi-piece fill (2026-08-08,
                    # GitHub #48; same class of race as the SBRA share-count fix above, just
                    # for entry_price instead of share count). Falls back to the snapshot
                    # only if the order's own fill price is ever missing, same pattern
                    # already used at every other real-fill-price call site in this file.
                    entry_price = (
                        order_status.filled_price if order_status.filled_price is not None
                        else p["entry_price"]
                    )
                    await self.portfolio.add_position_async(Position(
                        ticker=ticker,
                        shares=real_shares,
                        entry_price=entry_price,
                        current_price=p["current_price"],
                        stop_loss=pending["stop_price"],
                        take_profit_targets=_pending_targets,
                        sector=pending.get("sector", ""),
                        opened_at=datetime.now(),
                        t1_target_price=_pending_targets[0] if len(_pending_targets) > 0 else None,
                        t2_target_price=_pending_targets[1] if len(_pending_targets) > 1 else None,
                        trade_id=str(uuid.uuid4()),
                    ))
                    # Alpaca deducts cash at order submission, so _sync_portfolio already
                    # captured the reduced balance. add_position_async deducted it again —
                    # re-fetch the authoritative cash balance to correct any double-deduction.
                    try:
                        account = await self.broker.get_account()
                        self.portfolio.cash = account.cash
                        await self.portfolio._save_state()
                    except Exception as e:
                        logger.warning("Cash re-sync after pending fill failed for %s: %s", ticker, e)
                    # Same reasoning as the _execute_buy call site — the ticker was only
                    # just added to portfolio.positions above, so it needs the lock held
                    # for the placement to be safe against a concurrent sync_exit_orders.
                    async with self._lock_for(ticker):
                        await self._place_exit_orders(
                            ticker, real_shares,
                            pending["stop_price"],
                            pending.get("take_profit_targets", []),
                        )

            # If any share counts were corrected, re-sync cash from Alpaca so portfolio.cash
            # reflects the actual proceeds from untracked TP fills rather than a stale value.
            if shares_corrected:
                try:
                    account = await self.broker.get_account()
                    self.portfolio.cash = account.cash
                    await self.portfolio._save_state()
                    logger.info(
                        "Cash re-synced after share-count correction(s): $%.2f",
                        self.portfolio.cash,
                    )
                except Exception as _e:
                    logger.warning("Cash re-sync after share correction failed: %s", _e)

            # Clean up orphaned _pending_stops entries (fixed 2026-08-08, GitHub #58) --
            # a buy order that ends CANCELLED/REJECTED/EXPIRED with ZERO shares ever
            # filled never causes its ticker to appear in Alpaca's positions list at all
            # (a never-filled order simply isn't a position) -- so the main loop above,
            # which only visits tickers currently IN `positions`, never runs for it, and
            # the trade_updates stream only reacts to "fill" events. Without this, the
            # dict entry lingers forever (until a later buy attempt for the same ticker
            # happens to overwrite it). Low impact -- no real shares are ever left
            # unprotected, since nothing was ever bought -- but a real state-hygiene gap.
            # Checks each pending entry directly against its own order status,
            # independent of whether the ticker shows up in `positions` at all --
            # deliberately scoped to tickers NOT in alpaca_tickers, so this can never
            # race or double-handle a ticker the main loop above is already resolving.
            for _pending_ticker in list(self._pending_stops.keys()):
                if _pending_ticker in alpaca_tickers:
                    continue  # has real shares -- the main loop above owns this one
                # .get(), not [] (caught 2026-08-08 by adversarial self-review, same
                # race class as GitHub #47 fixed elsewhere today): this loop awaits
                # get_order_status per-ticker, so the concurrent trade_updates stream
                # handler can pop a LATER ticker's entry between iterations while an
                # EARLIER ticker's await is in flight -- direct indexing would KeyError.
                _pending = self._pending_stops.get(_pending_ticker)
                if _pending is None:
                    continue  # already resolved by the stream while we were iterating
                _order_id = _pending.get("order_id")
                if not _order_id:
                    continue
                try:
                    _order_status = await self.broker.get_order_status(_order_id)
                except Exception:
                    continue  # transient lookup failure -- leave it, retry next cycle
                if (_order_status.status in (
                        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
                        and (_order_status.filled_quantity or 0) <= 0):
                    logger.info(
                        "%s: pending buy order %s ended %s with zero fill and never "
                        "became a position -- clearing stale _pending_stops entry",
                        _pending_ticker, _order_id, _order_status.status.value,
                    )
                    self._pending_stops.pop(_pending_ticker, None)

            # Detect positions closed by Alpaca (stop/TP filled).
            # Guard: empty list — likely transient API error, not simultaneous close of all positions.
            apparent_closes = [t for t in self.portfolio.positions if t not in alpaca_tickers]
            if not positions and self.portfolio.positions:
                logger.warning(
                    "Alpaca returned 0 positions but we have %d locally — "
                    "skipping close-sync to avoid false position wipes",
                    len(self.portfolio.positions),
                )
            elif apparent_closes:
                # Fetch live open orders once (for orphan-cancel) and recent closed orders
                # (to verify each apparent close has an actual fill before acting on it).
                try:
                    live_open_orders = await self.broker.get_open_orders()
                except Exception as _e:
                    logger.warning("update_positions: get_open_orders failed (%s) — orphan cancel skipped", _e)
                    live_open_orders = []

                try:
                    # Scoped to just the tickers apparently closed (2026-07-24, INSW
                    # incident) -- the old unscoped call verified against an account-wide
                    # "most recent 100 orders" window, sorted by submission time. A stop
                    # order placed early in the day that only fills later can get pushed
                    # out of that shared window by unrelated order churn on OTHER
                    # tickers, even though it filled recently. Querying just these
                    # specific tickers means unrelated account-wide volume can never hide
                    # the one order that actually matters for verifying THIS close.
                    recent_closed = await self.broker.get_closed_orders(symbols=apparent_closes)
                    closed_tickers = {o["symbol"] for o in recent_closed if o.get("side") == "sell"}
                except Exception as _e:
                    logger.warning("update_positions: get_closed_orders failed (%s) — skipping close-sync this cycle", _e)
                    closed_tickers = set()  # empty = treat all apparent closes as unverified, skip them

                closed_by_alpaca = []
                for ticker in list(self.portfolio.positions.keys()):
                    if ticker not in alpaca_tickers:
                        # Verify with a real filled sell order — avoids acting on Alpaca glitches
                        if closed_tickers is not None and ticker not in closed_tickers:
                            logger.warning(
                                "%s missing from Alpaca positions but NO filled sell order found — "
                                "likely transient API glitch, skipping close", ticker)
                            continue
                        # If something else is actively working this ticker right now (a sell,
                        # a TP placement), don't rip state out from under it — the next 10s
                        # update_positions cycle will catch the close once that finishes.
                        lock = self._lock_for(ticker)
                        if lock.locked():
                            logger.debug(
                                "update_positions: %s locked elsewhere — deferring Alpaca-detected close", ticker)
                            continue
                        logger.info("%s position closed in Alpaca (stop/TP filled) — syncing local state", ticker)
                        stop_id = self._stop_order_ids.get(ticker)
                        stop_fill_price = None
                        if stop_id:
                            try:
                                stop_order = await self.broker.get_order_status(stop_id)
                                stop_fill_price = stop_order.filled_price
                            except Exception:
                                pass
                        async with lock:
                            await self._cancel_exit_orders(ticker, open_orders=live_open_orders)
                            pos = self.portfolio.positions[ticker]
                            fill_price = stop_fill_price if stop_fill_price is not None else pos.current_price
                            closed_shares = pos.shares
                            # Specific reason (2026-07-29) instead of the vague
                            # "(Stop/TP)" -- recent_closed (fetched above to verify this
                            # apparent close is real) already has every closed order for
                            # this ticker, so reuse it to distinguish a genuine
                            # take-profit fill (order_type == "limit") from a
                            # stop-driven one, same technique already proven in
                            # _reconcile_untracked_fill, instead of guessing.
                            ticker_closes = [
                                o for o in recent_closed
                                if o.get("symbol") == ticker and o.get("side") == "sell"
                                and (o.get("filled_qty") or 0) > 0 and o.get("filled_at")
                            ]
                            ticker_closes.sort(key=lambda o: o["filled_at"])
                            last_close = ticker_closes[-1] if ticker_closes else None
                            if last_close is not None and last_close.get("order_type") == "limit":
                                close_reason = "Take-Profit (final tranche)"
                            elif last_close is not None:
                                close_reason = _classify_stop_exit_reason(
                                    fill_price, pos.entry_price, pos.profit_target_hit)
                            else:
                                # Couldn't confidently determine which real order this
                                # was -- fall back to the honest, ambiguous label
                                # rather than guess.
                                close_reason = "Broker-Detected Close (Stop/TP)"
                            pnl = await self.portfolio.close_position_async(
                                ticker, exit_shares=pos.shares, exit_price=fill_price,
                                reason=close_reason)
                        closed_by_alpaca.append(ticker)
                        closed_reports.append({
                            "ticker": ticker, "shares": closed_shares,
                            "fill_price": fill_price, "pnl": pnl,
                        })
                if closed_by_alpaca:
                    # Re-sync cash from Alpaca so the actual fill price (not stale current_price)
                    # is reflected — especially important for gap-down stop fills.
                    try:
                        account = await self.broker.get_account()
                        self.portfolio.cash = account.cash
                        await self.portfolio._save_state()
                        logger.info("Cash re-synced after %d Alpaca-detected close(s): $%.2f",
                                    len(closed_by_alpaca), self.portfolio.cash)
                    except Exception as e:
                        logger.warning("Cash re-sync after position close failed: %s", e)

            await self.check_take_profits()
            # update_peak() now guards on _rotation_in_progress internally (see portfolio.py)
            _old_peak = self.portfolio.peak_value
            self.portfolio.update_peak()
            if self.portfolio.peak_value > _old_peak:
                await self.portfolio._save_state()
        except Exception as e:
            logger.warning("Position update failed: %s", e)
        return closed_reports
