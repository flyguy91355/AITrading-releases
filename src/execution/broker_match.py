"""Bot vs Alpaca match check (2026-11-27, owner: "it should check after every trade that the bot and alpaca
matches"). Pure: given what the bot thinks it holds and what Alpaca reports, list every mismatch in plain words.
web/app.py runs it ~20s after every fill on the trade_updates stream (and every 15 minutes in market hours as a
backstop) and puts a ✓ or ✗ line in the AI log. Read-only: it never places or changes an order.

Checked:
  * every long the bot holds (portfolio.positions) exists at Alpaca with the same share count;
  * every short in the short mirror's book exists at Alpaca with the same (negative) share count, and its buy stop
    (the book's stop_order_id) is open for the whole quantity;
  * in regular hours, each long's resting sell stops cover all its shares (they're DAY orders, so not overnight);
  * Alpaca holds nothing the bot doesn't track.
Tickers with an order still working (a market/limit order open, a buy waiting on its fill, a short being opened or
covered) are skipped -- their numbers are mid-change."""
from __future__ import annotations

STOP_TYPES = ("stop", "stop_limit", "trailing_stop")
_TOL = 0.0011


def _side(o: dict) -> str:
    return str(o.get("side", "")).lower().split(".")[-1]


def _type(o: dict) -> str:
    return str(o.get("type", "")).lower().split(".")[-1]


def working_tickers(open_orders: list[dict]) -> set[str]:
    """Tickers with a non-stop order still open (a market or limit order mid-fill)."""
    return {o["ticker"] for o in open_orders if _type(o) not in STOP_TYPES}


def compare_with_broker(bot_longs: dict[str, float], bot_shorts: dict[str, dict], positions: list[dict],
                        open_orders: list[dict], in_flight: set[str], check_long_stops: bool) -> list[str]:
    """Every mismatch as a sentence ([] = everything matches). bot_longs: {ticker: shares};
    bot_shorts: {ticker: {"qty", "stop_order_id", "stop"}}; positions: broker.get_positions();
    open_orders: broker.get_open_orders()."""
    alp = {p["ticker"]: float(p["shares"]) for p in positions}
    skip = set(in_flight) | working_tickers(open_orders)
    problems: list[str] = []
    for t, shares in sorted(bot_longs.items()):
        if t in skip:
            continue
        a = alp.get(t)
        if a is None:
            problems.append(f"{t}: the bot holds {shares:g} shares long, Alpaca has no position")
            continue
        if a < 0:
            problems.append(f"{t}: the bot holds it long, Alpaca has it SHORT ({a:g})")
            continue
        if abs(a - float(shares)) > _TOL:
            problems.append(f"{t}: the bot has {float(shares):g} shares, Alpaca has {a:g}")
        if check_long_stops:
            covered = sum(float(o.get("qty") or 0) for o in open_orders
                          if o["ticker"] == t and _side(o) == "sell" and _type(o) in STOP_TYPES)
            if covered < a - _TOL:
                problems.append(f"{t}: its sell stops cover {covered:g} of {a:g} shares")
    for t, b in sorted(bot_shorts.items()):
        if t in skip:
            continue
        q = float(b.get("qty") or 0)
        a = alp.get(t)
        if a is None:
            problems.append(f"{t}: the short book has {q:g} short, Alpaca has no position")
            continue
        if a > 0:
            problems.append(f"{t}: the bot has it short, Alpaca has it LONG ({a:g})")
            continue
        if abs(-a - q) > _TOL:
            problems.append(f"{t}: the short book has {q:g} short, Alpaca has {-a:g} short")
        stop = next((o for o in open_orders if o.get("order_id") == b.get("stop_order_id")), None)
        if stop is None:
            problems.append(f"{t}: its buy stop isn't open at Alpaca")
        elif _side(stop) != "buy" or abs(float(stop.get("qty") or 0) - q) > _TOL:
            problems.append(f"{t}: its stop order is {_side(stop)} {float(stop.get('qty') or 0):g}, "
                            f"expected buy {q:g}")
    for t, a in sorted(alp.items()):
        if t in skip or t in bot_longs or t in bot_shorts:
            continue
        problems.append(f"{t}: Alpaca has {abs(a):g} shares {'short' if a < 0 else 'long'} the bot doesn't track")
    return problems


def describe(ticker: str, bot_longs: dict[str, float], bot_shorts: dict[str, dict], positions: list[dict],
             open_orders: list[dict]) -> str:
    """One ticker's state at Alpaca in plain words, for the ✓ line."""
    a = next((float(p["shares"]) for p in positions if p["ticker"] == ticker), None)
    if a is None:
        return f"{ticker} is flat at Alpaca and in the bot"
    stops = [o for o in open_orders if o["ticker"] == ticker and _type(o) in STOP_TYPES]
    if a < 0:
        st = ", ".join(f"buy stop {float(o.get('qty') or 0):g} @ ${float(o.get('stop_price') or 0):.2f}"
                       for o in stops if _side(o) == "buy") or "no buy stop"
        return f"{ticker} short {-a:g} at Alpaca and in the short book; {st} resting"
    st = ", ".join(f"sell stop {float(o.get('qty') or 0):g} @ ${float(o.get('stop_price') or 0):.2f}"
                   for o in stops if _side(o) == "sell") or "no sell stop resting (DAY stops renew in market hours)"
    return f"{ticker} long {a:g} at Alpaca and in the bot; {st}"
