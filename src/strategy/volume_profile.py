"""Volume profile (volume traded at each price) for the dip-buyer's and the short mirror's pick
order -- 2026-11-28, owner: "I want it built and put on the charts. i can turn it on or off in
settings. make sure its built for both long and short."

Research: docs/research/2026-11-28-volume-profile-video/README.md. Ranking stressed dip signals
by "furthest under the point of control" beat today's lowest-RSI(2)-first order in both
2016-20 and 2021-26 (forward returns only, not yet a portfolio backtest). Design:
docs/superpowers/specs/2026-11-28-volume-profile-pick-order-design.md.

The math matches the research script (scripts/dip_volume.py::profile) so a later backtest tunes
the thing that trades:
  * window = the `lookback` completed sessions BEFORE the signal session;
  * each day's volume is spread evenly over its low-high range across `rows` equal price rows
    from the window's lowest low to its highest high (a day with high == low puts all of it in
    its own row -- the research gave those rare days nothing);
  * point of control (POC) = the busiest row (ties: the lowest row);
  * value area = from the POC, add one neighbouring row at a time, the busier side first (ties
    go UP), until it holds value_area_pct% of the volume -- the same rule as Quant Terminal's.
Pure: no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass

VOLUME_PROFILE_DEFAULTS = {
    "lookback_sessions": 63,
    "rows": 100,
    "value_area_pct": 70,
}
_RANGES = {"lookback_sessions": (20, 250), "rows": (20, 400), "value_area_pct": (50, 95)}


@dataclass(frozen=True)
class VpConfig:
    lookback_sessions: int
    rows: int
    value_area_pct: float


@dataclass(frozen=True)
class VolumeProfile:
    poc: float              # the point of control's price (middle of its row)
    vah: float              # value area high (top edge of its top row)
    val: float              # value area low (bottom edge of its bottom row)
    poc_atr: float          # (close - poc) / ATR: negative = under the POC
    zone: str               # the close vs the value area: "below" / "inside" / "above"
    start_date: str         # the window's first and last session
    end_date: str


def vp_cfg(raw: dict | None) -> VpConfig:
    """config['volume_profile'] merged over VOLUME_PROFILE_DEFAULTS. Unlike the strategy configs
    (where a typo raises), an unparsable or out-of-range value falls back to its default: these
    only order the qualifiers, and a bad number must never stop the evening refresh."""
    out = {}
    for k, default in VOLUME_PROFILE_DEFAULTS.items():
        v = (raw or {}).get(k)
        lo, hi = _RANGES[k]
        try:
            x = float(v)
            ok = math.isfinite(x) and lo <= x <= hi
        except (TypeError, ValueError):
            ok = False
        out[k] = x if ok else default
    return VpConfig(lookback_sessions=int(out["lookback_sessions"]), rows=int(out["rows"]),
                    value_area_pct=float(out["value_area_pct"]))


def _coerce_range(key: str, v, as_int: bool):
    """Settings-page coercion: a number inside the field's range, else ValueError (which aborts
    the whole save, so nothing half-applies)."""
    x = float(str(v).strip())
    lo, hi = _RANGES[key]
    if not math.isfinite(x) or not lo <= x <= hi or (as_int and x != int(x)):
        raise ValueError(f"{key} must be {'a whole number ' if as_int else ''}from {lo} to {hi}, got {v!r}")
    return int(x) if as_int else x


def coerce_lookback(v) -> int:
    return _coerce_range("lookback_sessions", v, True)


def coerce_rows(v) -> int:
    return _coerce_range("rows", v, True)


def coerce_value_area(v) -> float:
    return _coerce_range("value_area_pct", v, False)


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def profile(rows: list[dict], close: float | None, atr: float | None, cfg: VpConfig) -> VolumeProfile | None:
    """rows: completed daily bars, oldest first, INCLUDING the signal session (its own bar is not
    in the window). None when there isn't enough to judge: fewer than 5/7 of the window usable
    (45 of 63), a flat window, no volume, or a non-finite close / non-positive ATR."""
    if not (_finite(close) and _finite(atr)) or float(atr) <= 0:
        return None
    n_look, n = cfg.lookback_sessions, cfg.rows
    window = rows[-n_look - 1:-1]
    days = [r for r in window
            if _finite(r.get("high")) and _finite(r.get("low")) and _finite(r.get("volume"))
            and float(r["high"]) >= float(r["low"]) and float(r["volume"]) >= 0]
    if len(days) < math.ceil(n_look * 5 / 7):
        return None
    lo = min(float(r["low"]) for r in days)
    hi = max(float(r["high"]) for r in days)
    if not hi > lo:
        return None
    edges = [lo + (hi - lo) * k / n for k in range(n + 1)]
    hist = [0.0] * n
    for r in days:
        h, l, v = float(r["high"]), float(r["low"]), float(r["volume"])
        if v == 0:
            continue
        if h == l:
            hist[min(n - 1, int((h - lo) / (hi - lo) * n))] += v
            continue
        # one row of slack each side: float rounding at a row edge must not drop a sliver
        a = max(0, min(n - 1, int((l - lo) / (hi - lo) * n) - 1))
        z = max(0, min(n - 1, int((h - lo) / (hi - lo) * n) + 1))
        for k in range(a, z + 1):
            ov = min(h, edges[k + 1]) - max(l, edges[k])
            if ov > 0:
                hist[k] += ov / (h - l) * v
    tot = sum(hist)
    if not tot > 0:
        return None
    poc = max(range(n), key=lambda k: (hist[k], -k))
    acc, a, b = hist[poc], poc, poc
    need = cfg.value_area_pct / 100.0 * tot
    while acc < need and not (a == 0 and b == n - 1):
        up = hist[b + 1] if b + 1 < n else -1.0
        dn = hist[a - 1] if a > 0 else -1.0
        if up >= dn:
            b += 1
            acc += up
        else:
            a -= 1
            acc += dn
    val, vah = edges[a], edges[b + 1]
    poc_px = (edges[poc] + edges[poc + 1]) / 2
    c = float(close)
    return VolumeProfile(
        poc=poc_px, vah=vah, val=val, poc_atr=(c - poc_px) / float(atr),
        zone="below" if c < val else ("above" if c > vah else "inside"),
        start_date=str(days[0].get("date", ""))[:10], end_date=str(days[-1].get("date", ""))[:10])


def as_dict(p: VolumeProfile | None) -> dict | None:
    """The JSON-safe form stored on candidate cards and in the state files."""
    if p is None:
        return None
    return {"poc": round(p.poc, 4), "vah": round(p.vah, 4), "val": round(p.val, 4),
            "poc_atr": round(p.poc_atr, 3), "zone": p.zone, "start": p.start_date, "end": p.end_date}
