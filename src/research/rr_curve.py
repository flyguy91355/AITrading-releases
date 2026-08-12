"""Pure R/R-over-time reconstruction for On Deck candidates.

Shared by the near-miss list's card-level sparkline and the full-history chart
endpoint in web/app.py. Kept dependency-free (no FastAPI/DashboardState import)
so it's directly unit-testable without importing the web app module.
"""


def dip_summary(price_history: list[tuple[float, float]], retracement_pct: float) -> dict | None:
    """Peak-to-low dip depth and the retracement price target — the exact same
    previous-peak/low/depth math near_miss_monitor_loop uses to decide the buy trigger,
    extracted here (2026-07-18) so a card-level "here's what's happening" summary can show
    the identical numbers instead of a second, easily-drifting copy of the calculation.

    low_ts/low_price is the FIRST (earliest) occurrence of price_history's minimum price —
    the actual bottom of the dip, not a later repeat of the same low if one exists. "Previous
    peak" is the highest price strictly BEFORE that timestamp, deliberately excluding any
    price after the low (a post-low spike isn't part of the dip being measured). peak_t is
    that peak's own timestamp (2026-07-28, RRC/OVV incident) -- added so a caller can tell
    Claude how many days ago the peak/low actually happened, not just their bare prices; see
    recommend_dip_entry's docstring for why that mattered (a low sitting near the edge of a
    long trending window still passes the "there's a peak before the low" check below even
    when it's weeks old and no longer represents a real, current dip).

    Returns None if there's no measurable dip yet (empty history, the low is the oldest point
    in the window with nothing before it to compare against, or the "peak" turns out to be at
    or below the low) — same "not enough data, don't guess" case the buy-trigger check itself
    treats as unable to promote, so a caller can render "not enough history yet" instead of a
    misleading number.
    """
    if not price_history:
        return None
    low_ts, low_price = min(price_history, key=lambda p: p[1])
    prior_points = [(ts, p) for ts, p in price_history if ts < low_ts]
    if not prior_points or low_price <= 0:
        return None
    peak_ts, peak = max(prior_points, key=lambda point: point[1])
    depth = peak - low_price
    if depth <= 0:
        return None
    target = low_price + depth * (retracement_pct / 100)
    return {
        "peak": round(peak, 2),
        "peak_t": peak_ts,
        "low": round(low_price, 2),
        "low_t": low_ts,
        "depth": round(depth, 2),
        "depth_pct": round(depth / peak * 100, 1),
        "retracement_target": round(target, 2),
    }


def rr_at_price(price: float, fair_value: float, stop_loss_pct: float) -> float | None:
    """The same stop-trailing R/R formula near_miss_monitor_loop uses live, applied to a
    single price rather than a whole history — shared by rr_points (below) and by the
    "where would the buy-target price land on the R/R axis" calculation used to plot the
    entry-target line directly on the R/R chart. None if the ratio can't be computed
    (invalid price/fair_value), same as rr_points' per-point behavior.
    """
    stop_pct = stop_loss_pct / 100
    stop = price * (1 - stop_pct)
    risk = price - stop
    return (fair_value - price) / risk if risk > 0 and fair_value > 0 else None


def rr_points(price_history: list[tuple[float, float]], fair_value: float, stop_loss_pct: float) -> list[dict]:
    """Reconstructs R/R at each recorded (timestamp, price) sample using rr_at_price.
    Points where the ratio can't be computed (invalid price/fair_value) get rr=None rather
    than being silently dropped, so callers can decide how to handle gaps.
    """
    points = []
    for ts, price in price_history:
        rr = rr_at_price(price, fair_value, stop_loss_pct)
        points.append({
            "t": ts,
            "price": round(price, 2),
            "rr": round(rr, 3) if rr is not None else None,
        })
    return points


def rr_sparkline(
    price_history: list[tuple[float, float]],
    fair_value: float,
    stop_loss_pct: float,
    max_points: int = 40,
) -> list[dict]:
    """Compact R/R series for a card-level sparkline — {"t": ts, "rr": rr} per point. Capped
    at max_points so a full multi-week history of 60-second live samples plus daily backfill
    bars doesn't bloat the /api/near-miss list payload.

    Downsampling is bucketed by TIME, not by array index (changed 2026-07-18 — fixing a real
    bug, not a style choice). price_history mixes sparse daily backfill (~1 point/day) with
    dense 60-second live ticks once market monitoring starts on a candidate. Picking every
    Nth *item* (the original approach) meant the dense recent cluster dominated the output —
    e.g. 40 points could end up being 2 points covering 30 days of backfill and 38 points
    covering the last few hours of ticks, visually squeezing older dips into nothing and
    making the whole chart read as minutes-scale noise even though the underlying window
    spans weeks. Splitting [t_min, t_max] into max_points equal-width TIME buckets and taking
    the most recent sample within each bucket represents the full time range proportionally
    regardless of how sample density varies across it. Buckets with no sample are simply
    absent from the result (never fabricated), so the true output length can be less than
    max_points for a sparse/gappy history — that's honest, not a bug.
    """
    points = [p for p in rr_points(price_history, fair_value, stop_loss_pct) if p["rr"] is not None]
    if len(points) <= max_points:
        return [{"t": p["t"], "rr": p["rr"]} for p in points]
    t_min, t_max = points[0]["t"], points[-1]["t"]
    span = t_max - t_min
    if span <= 0:
        # All samples share one timestamp (shouldn't normally happen) -- time-bucketing is
        # meaningless here, fall back to plain index slicing rather than dividing by zero.
        step = len(points) / max_points
        return [{"t": points[int(i * step)]["t"], "rr": points[int(i * step)]["rr"]} for i in range(max_points)]
    bucket_width = span / max_points
    buckets: dict[int, dict] = {}
    for p in points:
        idx = min(int((p["t"] - t_min) / bucket_width), max_points - 1)
        buckets[idx] = p  # last (most recent) sample in the bucket wins
    return [{"t": buckets[i]["t"], "rr": buckets[i]["rr"]} for i in sorted(buckets)]
