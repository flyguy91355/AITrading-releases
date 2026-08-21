"""Web dashboard server for AITrading system."""

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, time as dtime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import requests

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.market_data import MarketDataFetcher
from src.data.insider_tracker import InsiderTracker
from src.data.news_feed import NewsFeed
from src.research.engine import ResearchEngine, _clamp_ai_stop_loss
from src.research.fundamental import FundamentalAnalyzer
from src.research.sentiment import SentimentAnalyzer
from src.research.insider_analysis import InsiderAnalyzer
from src.research.competitor import CompetitorAnalyzer
from src.research.quick_screen import quick_screen

# quick_screen() makes several yfinance calls with no timeout of their own (2026-08-03,
# live incident: a Full Scan run stalled indefinitely on a single ticker, no exception,
# no log line -- the try/except around each call site only catches a raised error, not a
# hang). asyncio.wait_for bounds the await itself; the underlying thread may linger a bit
# longer until the network call eventually gives up on its own, but the scan LOOP is no
# longer blocked waiting on it.
_QUICK_SCREEN_TIMEOUT_SECS = 15

# SQLite busy-retry window (2026-08-19, live incident: _run_pre_open_batch crashed with
# sqlite3.OperationalError: database is locked ~19 minutes into a live pre-open batch
# run, silently killing that morning's On Deck refresh -- the crashing write was a plain
# sqlite3 connection with no explicit `timeout=`, so it only internally retried for the
# module's implicit 5.0s default before giving up. This app has several independent
# async loops touching the same data/aitrading.db file (position updates, ai_log
# persistence, trade history, the watchlist cursor write that actually crashed) with no
# coordination between them, so a 5s window can genuinely be too short under real
# concurrent load. 20s gives real headroom without risking a hung request feeling stuck
# to a human waiting on it. Applied everywhere this app opens a sqlite3/aiosqlite
# connection to that shared file -- see also _ensure_wal_mode below, the other half of
# this same fix.
_SQLITE_TIMEOUT_SECS = 20.0
from src.research.rr_curve import dip_summary, price_sparkline, rr_at_price, rr_points, rr_sparkline


async def _quick_screen_with_timeout(ticker: str) -> tuple[bool, str] | None:
    """Shared wrapper for quick_screen() with the 15s hang-protection timeout above --
    extracted (2026-08-03) after the identical try/except block was independently
    copy-pasted into both _run_pre_open_batch's and _run_midday_rescan's inner chunk
    generators, including the same comment -- the exact "duplicated logic drifts"
    failure class this project has hit before (population floor, composite score,
    admission gates). Returns None on any error or timeout (caller should `continue`
    to the next ticker); returns quick_screen()'s real (passes, reason) tuple otherwise."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(quick_screen, ticker), timeout=_QUICK_SCREEN_TIMEOUT_SECS)
    except Exception as _qs_err:
        # Exception already covers asyncio.TimeoutError (an alias for the builtin
        # TimeoutError, itself an OSError subclass) -- no separate branch needed.
        logger.debug("Quick screen error/timeout for %s: %s", ticker, _qs_err)
        return None
from src.decision.signal_generator import SignalGenerator
from src.decision.risk_manager import RiskManager
from src.decision.portfolio import Portfolio
from src.decision.risk_tier import (
    apply_risk_tier_to_settings,
    compute_risk_tier_settings,
    risk_tier_label,
    RISK_TIER_DOTKEYS,
)
from src.execution.order_manager import OrderManager
from src.execution.broker import OrderStatus
from src.reporting.trade_logger import TradeLogger
from src.utils.watchlist_manager import WatchlistManager
from src.data.stock_universe import get_universe
from src.analytics.composition_benchmark import weighted_daily_return
from src.update.version import read_local_version, write_local_version, is_newer
from src.update.release_client import fetch_latest_release, fetch_recent_releases
from src.update.apply import extract_release_archive, copy_updatable_files, requirements_changed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Real go-live date for THIS install, self-initializing rather than hardcoded
# (2026-08-20, closed at the source after a real incident on the AIShortTrading
# fork — see that project's CLAUDE_HISTORY.md 2026-08-20 entry). This constant used
# to be a literal date string; when AIShortTrading forked this file, that literal
# copied over verbatim and silently named AITrading's own 2026-07-12 inception
# instead of AIShortTrading's real one, distorting its annualized-P&L display AND
# its trade-history/win-rate filtering for a full day before anyone noticed. This
# file's own value (2026-07-12) is still correct for THIS install — the fix below
# doesn't change today's behavior here at all — but it closes the landmine at its
# source: the *next* fork of this file (whichever sibling it becomes) now inherits
# a self-initializing mechanism instead of a copy-pasted literal to forget to
# update. `_ACCOUNT_GENESIS_PATH` is a marker file (same presence-only-marker idea
# as `_FIRST_SCAN_MARKER` below): on a fresh clone's very first startup, no such
# file exists yet, so `_get_or_init_account_genesis()` writes TODAY's date and that
# becomes that install's own permanent genesis from then on. Both
# /api/portfolio-summary and /api/portfolio-health filter trade data to this
# cutoff — everything before it is leftover local-machine dev/test data that got
# rsync'd over during the initial deploy (both data/trade_history/*.jsonl and
# data/aitrading.db itself), never cleaned up, and would otherwise wildly distort
# both endpoints' numbers.
_ACCOUNT_GENESIS_PATH = Path("data/.account_genesis")


def _get_or_init_account_genesis(today_str: str) -> str:
    """Pure apart from the one file read/write: returns the persisted genesis date if
    `_ACCOUNT_GENESIS_PATH` already holds one, else writes `today_str` there and
    returns it -- so a fresh clone's very first startup permanently stamps its own
    real go-live date with zero manual configuration. `today_str` is passed in
    (rather than computed here) so this stays testable without mocking the clock."""
    try:
        existing = _ACCOUNT_GENESIS_PATH.read_text().strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    _ACCOUNT_GENESIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ACCOUNT_GENESIS_PATH.write_text(today_str)
    return today_str

# The Watchlist-removal redesign (2026-07-17, see CLAUDE.md) completely replaced the
# buy/sell decision mechanism — On Deck's continuous near-miss monitoring became the sole
# buy path, superseding the old scheduled-scan Watchlist logic entirely. /api/portfolio-
# health uses this to split win-rate into "all trades since go-live" vs "current
# architecture only", since trades before this date were decided by a fundamentally
# different, now-replaced system and blending them into one figure understates how the
# CURRENT logic is actually performing. Confirmed live (2026-07-20): all-time win rate was
# 42.9% (6/14), but trades since this date alone were 80% (4/5) — a large, real difference.
_CURRENT_ARCHITECTURE_START = "2026-07-17"

# Position.trade_id (the column _group_closed_trades groups by) didn't exist until
# 2026-07-27 -- any SELL row from a position that was still open when that column
# shipped has trade_id=NULL for its entire lifetime, and _group_closed_trades correctly
# has no choice but to count each such row as its own standalone trade (there's no id to
# group it by). Confirmed live (2026-07-30, user report): 69 of 105 real SELL rows have
# NULL trade_id, several of them genuinely fragmented tranches of the SAME original
# position (e.g. ONB: 4 separate "Stop loss hit" rows within 5 minutes on 2026-07-21) --
# each counting as its own win/loss instead of one combined trade, meaningfully
# distorting both win-rate figures. The LATEST real NULL-trade_id sell on record is
# 2026-07-27T17:55 -- every sell since has a real trade_id, confirming the migration
# fully self-healed by then. _closed_trades_since floors whatever cutoff it's given to
# this date so win-rate figures only ever include trades that CAN be accurately grouped,
# rather than silently inflating the trade count with known-fragmentable legacy rows.
_TRADE_ID_RELIABLE_SINCE = "2026-07-28"


async def _notify(title: str, message: str, priority: str = "default", tags: str = "") -> None:
    """Fire-and-forget push notification via ntfy.sh."""
    topic = os.getenv("NTFY_TOPIC", "")
    if not topic:
        return
    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    try:
        async with httpx.AsyncClient(timeout=5.0) as _hc:
            await _hc.post(f"https://ntfy.sh/{topic}", content=message, headers=headers)
    except Exception as _ne:
        logger.debug("ntfy notification failed: %s", _ne)


# Load .env explicitly here (2026-07-22, auth gate) -- load_config() also calls
# load_dotenv() internally, but that only happens later when DashboardState() is
# constructed (after `app = FastAPI()` below), which would be too late for the
# SESSION_SECRET_KEY/DASHBOARD_PASSWORD reads a few lines down. load_dotenv() is
# idempotent, so this early call and the later one inside load_config() don't conflict.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="Hilton's AITrading Dashboard")
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ── Auth gate (2026-07-22 -- GitHub issues #13/#14/#15/#16) ──────────────────────
# This dashboard places real Alpaca orders and holds Anthropic/Alpaca/Finnhub/NewsAPI
# credentials with zero authentication anywhere. The Hetzner firewall already restricts
# inbound to SSH + the tailscale0 interface (not exposed to the raw internet), but anything
# on the tailnet previously had full unauthenticated control. Adds a single shared-password
# login gate in front of every route (except /login itself and static assets) plus the
# WebSocket, using a signed session cookie (Starlette's SessionMiddleware, itsdangerous
# under the hood -- no server-side session store needed).
_SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "")
_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
if not _SESSION_SECRET_KEY or not _DASHBOARD_PASSWORD:
    raise RuntimeError(
        "SESSION_SECRET_KEY and DASHBOARD_PASSWORD must both be set in .env before the "
        "dashboard can start -- see CLAUDE.md 'Dashboard Login Gate' for how these were "
        "generated and where to change them."
    )

_LOGIN_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hilton's AITrading — Login</title>
<style>
  body { background:#0a0e17; color:#e5e7eb; font-family:system-ui,-apple-system,sans-serif;
         display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
  form { background:#1a2332; border:1px solid #2a3a4e; border-radius:10px; padding:32px;
         width:280px; box-shadow:0 8px 24px rgba(0,0,0,0.4); }
  h1 { font-size:18px; margin:0 0 20px; text-align:center; }
  input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:6px;
          border:1px solid #2a3a4e; background:#0a0e17; color:#e5e7eb; font-size:14px;
          margin-bottom:14px; }
  button { width:100%; padding:10px; border-radius:6px; border:none; background:#3987e5;
           color:#fff; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#2a78d6; }
  .error { color:#e66767; font-size:12.5px; margin-bottom:12px; text-align:center; }
</style></head>
<body>
  <form method="post" action="/login">
    <h1>Hilton's AITrading Dashboard</h1>
    __ERROR_HTML__
    <input type="password" name="password" placeholder="Password" autofocus required>
    <button type="submit">Log In</button>
  </form>
</body></html>"""


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse("/")
    return _LOGIN_PAGE_HTML.replace("__ERROR_HTML__", "")


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    if password and password == _DASHBOARD_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=303)
    error_html = '<div class="error">Incorrect password</div>'
    return HTMLResponse(_LOGIN_PAGE_HTML.replace("__ERROR_HTML__", error_html), status_code=401)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


# Paths reachable with no session -- the login page itself (or the redirect loop would
# never terminate) and static assets (PWA manifest/icons; harmless, and gating them would
# break "Add to Home Screen" before a first login on that device).
_AUTH_EXEMPT_PATHS = {"/login"}


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or path.startswith("/static/"):
        return await call_next(request)
    if not request.session.get("authenticated"):
        if path.startswith("/api/"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse("/login")
    return await call_next(request)


# Registered AFTER _auth_gate above so it becomes the OUTER middleware layer (Starlette
# applies middleware in reverse registration order -- the most recently added wraps
# everything registered before it) and therefore runs BEFORE _auth_gate on every request,
# populating request.session in time for _auth_gate to read it. https_only follows the same
# SSL_CERTFILE/SSL_KEYFILE env vars start.py checks (2026-07-23, real Tailscale-issued cert
# now available) -- True only when this process is actually being served over HTTPS by
# uvicorn, since a Secure-flagged cookie is never sent back by the browser over plain HTTP
# (would silently break login for local dev, which has no cert configured).
app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET_KEY,
    https_only=bool(os.environ.get("SSL_CERTFILE") and os.environ.get("SSL_KEYFILE")),
    max_age=30 * 24 * 60 * 60,  # 30 days -- a personal, trusted-tailnet-only dashboard
)

INTER_STOCK_DELAY = 3
_DD_CACHE_PATH = Path("data/deep_dive_cache.json")
_REPORT_CACHE_PATH = Path("data/reports_cache.json")
_PRICE_DIRECTION_CACHE_PATH = Path("data/price_direction_cache.json")
_ON_DECK_CACHE_PATH = Path("data/on_deck_cache.json")
# Marks that a full pre-open/universe scan has completed at least once, ever, for this
# install (2026-07-21) -- presence alone matters, contents are just a human-readable
# timestamp for anyone poking at the filesystem, never read back programmatically.
_FIRST_SCAN_MARKER = Path("data/.first_scan_completed")
_ON_DECK_BLOCKED_CACHE_PATH = Path("data/on_deck_blocked_cache.json")
_ON_DECK_NOTES_CACHE_PATH = Path("data/on_deck_notes_cache.json")
_BUY_REASONING_CACHE_PATH = Path("data/buy_reasoning_cache.json")
_EVENT_MONITOR_COOLDOWN_CACHE_PATH = Path("data/event_monitor_cooldown_cache.json")
_MIDDAY_SCAN_FIRED_CACHE_PATH = Path("data/midday_scan_fired_cache.json")
_ON_DECK_STALE_DIP_LOW_CACHE_PATH = Path("data/on_deck_stale_dip_low_cache.json")
# Log of every real promotion ATTEMPT (2026-07-21) -- repurposes the old "Candidates" tab,
# whose original data source (run_deep_dives -> _collect_buy_candidates) had zero live
# callers left anywhere in this file, a dead leftover from the pre-2026-07-17 Watchlist-era
# pipeline. Answers "why didn't X get bought" without needing to SSH and grep logs -- every
# time _attempt_near_miss_promotion actually fires, win or lose, one entry lands here.
_PROMOTION_ATTEMPTS_CACHE_PATH = Path("data/promotion_attempts_cache.json")
# Recent actionable signals (Signals / Action Items tabs), persisted across restarts
# (2026-07-21, per explicit request) -- previously reset to empty on every restart since
# active_signals was never saved anywhere.
_ACTIVE_SIGNALS_CACHE_PATH = Path("data/active_signals_cache.json")
# Daily portfolio-vs-market performance history (2026-07-21) -- one entry per trading day,
# captured once (piggybacking the existing daily-report trigger) so a single day's
# under/over-performance can be told apart from a genuine multi-week pattern. See
# CLAUDE.md for why this exists (a real day the portfolio badly lagged the market, and no
# way to tell if that was one-off or a trend). Kept forever, not capped -- this is exactly
# the kind of small (~1 row/day), long-lived series worth never trimming.
_PERFORMANCE_HISTORY_PATH = Path("data/performance_history.json")

# Populated in DashboardState.__init__ using configured universe_indexes
STOCK_UNIVERSE: list[str] = []


def _save_dd_cache(reports: dict) -> None:
    try:
        _DD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DD_CACHE_PATH.write_text(json.dumps(reports, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save deep dive cache: %s", e)


def _load_dd_cache() -> dict:
    try:
        if _DD_CACHE_PATH.exists():
            return json.loads(_DD_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load deep dive cache: %s", e)
    return {}


def _save_report_cache(reports: dict) -> None:
    try:
        _REPORT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_CACHE_PATH.write_text(json.dumps(reports, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save report cache: %s", e)


def _load_report_cache() -> dict:
    try:
        if _REPORT_CACHE_PATH.exists():
            return json.loads(_REPORT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load report cache: %s", e)
    return {}


def _save_price_direction_cache(directions: dict) -> None:
    try:
        _PRICE_DIRECTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PRICE_DIRECTION_CACHE_PATH.write_text(json.dumps(directions), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save price direction cache: %s", e)


def _load_price_direction_cache() -> dict:
    try:
        if _PRICE_DIRECTION_CACHE_PATH.exists():
            return json.loads(_PRICE_DIRECTION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load price direction cache: %s", e)
    return {}


def _derive_stop_pct(entry_price: float, stop_loss: float, default_pct: float) -> float:
    """Each On Deck candidate's own stop-loss %, derived from Claude's real stop_loss
    recommendation at analysis time rather than one flat config value shared by every
    stock — added 2026-07-18 after confirming the real buy order uses report.stop_loss
    directly (TradeSignal(stop_loss=report.stop_loss, ...) in _attempt_near_miss_promotion),
    so the R/R gate that decides WHETHER to buy should reflect the same stop the trade will
    actually place, not an unrelated mechanical percentage. Returns default_pct (the global
    take_profit.stop_loss_pct config, as a whole percentage e.g. 5.0) if entry_price/stop_loss
    aren't usable — same "fall back to the flat rule rather than guess" behavior as before
    this field existed, for a fallback report or any malformed input."""
    if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
        return default_pct
    return (entry_price - stop_loss) / entry_price * 100


def _graduated_trail_pct(
    entry_price: float, current_price: float, start_pct: float, final_pct: float,
    t1_price: float | None, t2_price: float | None, t3_price: float | None,
    follow_tp_targets: bool,
) -> float:
    """Trail width (as a whole percentage, e.g. 2.0 = 2%) for the graduated trailing stop
    (2026-07-23), interpolated along a single curve from entry_price (start_pct — the
    stock's own derived stop-loss %) to t3_price (final_pct — final_tranche_trail_pct).
    Replaces the old 5%->4%->final-tranche-only-interpolation step function with one
    continuous curve covering the whole life of the trade, so a fast mover that reaches T1
    quickly no longer gets the same trail width as one that ground there over weeks.

    follow_tp_targets=False draws a straight line with just two anchors (entry, T3).
    follow_tp_targets=True additionally bends the curve through T1/T2 as checkpoints
    (trail width split evenly across the four anchor points) when both are available and
    sit strictly between entry and T3 — falls back to the straight line otherwise (missing
    data, or a corrupted/out-of-order target), same "don't guess" pattern as the rest of
    this file's defensive fallbacks. Returns start_pct (the widest, safest end of the
    range) if t3_price is missing or the entry->T3 range itself isn't valid."""
    if t3_price is None or entry_price <= 0 or t3_price <= entry_price:
        return start_pct

    if (follow_tp_targets and t1_price is not None and t2_price is not None
            and entry_price < t1_price < t2_price < t3_price):
        step = (start_pct - final_pct) / 3
        anchors = [
            (entry_price, start_pct),
            (t1_price, start_pct - step),
            (t2_price, start_pct - 2 * step),
            (t3_price, final_pct),
        ]
    else:
        anchors = [(entry_price, start_pct), (t3_price, final_pct)]

    if current_price <= anchors[0][0]:
        return anchors[0][1]
    if current_price >= anchors[-1][0]:
        return anchors[-1][1]

    for (p_lo, pct_lo), (p_hi, pct_hi) in zip(anchors, anchors[1:]):
        if p_lo <= current_price <= p_hi:
            progress = (current_price - p_lo) / (p_hi - p_lo)
            return pct_lo - progress * (pct_lo - pct_hi)

    return final_pct  # unreachable given the bounds checks above -- safe fallback regardless


def _profit_target_trail_pct(
    profit_target_hit: bool, dollar_target_trail_pct: float, graduated_pct: float,
) -> float:
    """Trail width (whole percentage) for the trailing-stop block (2026-07-24) -- once a
    position's dollar profit target has permanently latched (Position.profit_target_hit),
    this fixed, tighter percentage replaces the graduated entry->T3 curve for the rest of
    that position's life. See docs/superpowers/specs/
    2026-07-24-dollar-profit-target-trailing-stop-design.md."""
    return dollar_target_trail_pct if profit_target_hit else graduated_pct


def _on_deck_cooldown_active(cooldown: dict, ticker: str, now: datetime) -> bool:
    """Whether ticker is still within a stored cooldown-until timestamp (2026-08-18) --
    shared by both the On Deck backfill's generic reject cooldown and the longer
    above-gate-decline cooldown, so the two checks can't drift apart on comparison
    direction/inclusivity. A missing ticker (never cooled down) is never active. An
    expiry exactly equal to `now` is treated as already expired (strict `>`, not `>=`),
    matching the pre-existing `cooldown.get(ticker, now) <= now` check this replaces."""
    return cooldown.get(ticker, now) > now


def _dip_low_changed_meaningfully(old_low: float | None, new_low: float, refresh_pct: float) -> bool:
    """Whether new_low is a genuinely deeper dip low than old_low, worth firing a fresh AI
    dip-entry recommendation for (2026-07-23) -- a plain != comparison treated ANY drift
    (even a few cents of ordinary price noise as the rolling observation window slides) as
    new information, causing a real Claude call (_compute_ai_dip_entry) to re-fire
    repeatedly for what was essentially the same answer. Confirmed live: ALLY fired 4 times
    in under 2 hours with near-identical reasoning each time, its tracked low drifting by
    single-digit cents ($43.59 -> $43.44 -> $43.42 -> $43.40). Returns True only if new_low
    is at least refresh_pct% below old_low -- a genuinely deeper low, not noise. A missing or
    invalid old_low (no recommendation exists yet) always counts as meaningfully different,
    and a new_low that hasn't actually gotten lower never does."""
    if old_low is None or old_low <= 0:
        return True
    if new_low >= old_low:
        return False
    return (old_low - new_low) / old_low * 100 >= refresh_pct


def _ai_entry_initially_armed(price: float, ai_entry_price: float, arm_band_pct: float) -> bool:
    """Whether price counts as "close enough" to ai_entry_price to arm the promotion
    trigger -- shared threshold formula used both for a freshly-computed recommendation's
    starting state and for every later monitoring tick in _ai_entry_trigger below, so the
    two can't drift apart. True the instant price is below ai_entry_price scaled up by
    arm_band_pct% (0 means "must be strictly below the entry itself," matching the
    original 2026-07-28 DV-incident behavior before arm_band_pct existed)."""
    return price < ai_entry_price * (1 + arm_band_pct / 100)


def _ai_entry_trigger(
    price: float, ai_entry_price: float, seen_below: bool, arm_band_pct: float = 0.0,
) -> tuple[bool, bool]:
    """Whether an AI dip-entry promotion should fire this tick, and the updated
    ai_entry_seen_below tracking value to store (2026-07-28, DV incident).

    Claude is asked for a good entry PRICE, not necessarily one above the current price --
    it commonly recommends a support level BELOW where price has already recovered to by the
    time the recommendation is computed. Real incident: DV's AI recommendation reasoned about
    a low from 5.1 days ago and recommended entry $10.65 "near initial support," while price
    had ALREADY recovered to $11.27 by the time that recommendation was made. The old trigger
    (`price >= ai_entry_price`) was already satisfied the instant the recommendation existed,
    so the system bought immediately at the current $11.28 market price -- 6% above the level
    Claude actually reasoned about -- instead of ever waiting for a real pullback there.

    Requires a genuine rise UP THROUGH ai_entry_price: seen_below must have been True at some
    point (price observed close enough to the entry since the recommendation was made) before
    price reaching/crossing the entry counts as a real signal. A recommendation where price is
    already below the entry (or within arm_band_pct of it) at computation time starts
    pre-armed, matching the ordinary "waiting for it to rise" case this feature was designed
    for; a recommendation like DV's, where price is already well above the entry, starts
    unarmed and requires price to genuinely approach the entry first.

    arm_band_pct (2026-08-04, "trading a range" discussion) -- widens what counts as "close
    enough to arm" from requiring a full cross below ai_entry_price to getting within this
    many percent ABOVE it. 0 (default) preserves the original exact DV-incident-fix behavior.
    Real motivation: every dip-entry recommendation observed live in this project's first
    weeks of trading started unarmed (price already above entry when computed, since Claude
    reasons about a support level below wherever price has already recovered to) -- in a
    persistently rising market that never gives back even 1-3%, an unarmed candidate could
    wait indefinitely. This still requires PROOF price got genuinely close to the reasoned
    support level, just not the full round-trip back through it -- and if price is already
    within the band (at/above entry) the moment it arms, it fires that same tick rather than
    requiring an additional future rise past a level it's already sitting above.

    Returns (should_promote, updated_seen_below) — the caller stores the second value back
    onto the candidate regardless of the first, since "price got close enough to the entry"
    is itself worth remembering even on a tick that doesn't yet promote."""
    if _ai_entry_initially_armed(price, ai_entry_price, arm_band_pct):
        seen_below = True
    if price < ai_entry_price:
        return False, seen_below
    return seen_below, seen_below


def _not_yet_analyzed_today(report: dict | None, today_str: str) -> bool:
    """True if `report` (a single research_reports value, or None if the ticker has no
    entry at all) was NOT generated today -- the "genuinely new" filter for the
    mid-day re-scan feature (2026-07-31). A ticker already analyzed today by ANY pass
    (this morning's pre-open batch, the 12:30 re-analysis, an On Deck backfill check,
    a prior mid-day re-scan slot the same day) is skipped, so a mid-day re-scan only
    ever spends a real Claude call on tickers genuinely untouched today -- this is
    what keeps the feature's cost scoped to new opportunities rather than repeating
    analysis already done. See docs/superpowers/specs/2026-07-31-midday-rescan-design.md."""
    if report is None:
        return True
    return not report.get("generated_at", "").startswith(today_str)


def _on_deck_population_floor(min_conviction: float, conviction_band: float) -> float:
    """The conviction floor for ADDING a candidate to On Deck -- deliberately below the
    real buy gate (min_conviction), so a "close but not quite" stock can be watched for
    a later re-analysis to push it over the line, per research.on_deck_conviction_band
    (2026-07-31, XRAY incident). Extracted into one shared function after this exact
    one-line calculation (min_conviction - conviction_band) was found independently
    duplicated in three separate places -- the pre-open population fill, the 60s
    intraday On-Shore backfill, and _refill_on_deck_from_shore (fired when a Settings
    save raises on_deck_max_size) -- and the third had silently drifted to not applying
    it at all, letting a 5.2-conviction stock (XRAY) onto On Deck with no floor check
    whatsoever. A single shared function means a future fourth call site can't
    independently forget this check the way that one did. Every real buy decision
    (_attempt_near_miss_promotion) still re-checks conviction fresh against the
    UNCHANGED min_conviction, not this floor -- being on On Deck only means "watched,"
    never "cleared to buy."
    """
    return min_conviction - conviction_band


def _on_deck_rr_floor_not_met(rr: float, required_rr: float, floor_margin) -> bool:
    """True if rr falls below THIS CANDIDATE'S OWN real, conviction-scaled buy gate
    (required_rr, from _required_rr) by more than research.on_deck_rr_floor_margin.
    Deliberately still keyed off the per-candidate required_rr rather than the flat
    research.min_risk_reward_ratio base value -- a stock's real gate can sit
    meaningfully above or below that flat base depending on its own conviction
    score. floor_margin of None (the default, unset) means no floor -- existing
    behavior (any R/R, however low, gets tracked as long as conviction and signal
    qualify) is unchanged until this is explicitly configured.

    Additive formula (floor = required_rr + floor_margin), not subtractive -- fixed
    2026-08-05, owner report: the original subtractive formula meant a POSITIVE
    floor_margin was needed to place the floor BELOW the gate, which read backwards.
    A negative floor_margin now means "this far below gate," matching intuition
    directly. There is no separate ceiling on the high side (removed the same
    evening) -- a candidate above its own gate is judged individually, see
    _on_deck_rr_above_gate and _on_deck_ai_gate_above_gate."""
    return floor_margin is not None and rr < required_rr + floor_margin


def _on_deck_rr_above_gate(rr: float, required_rr: float) -> bool:
    """True if rr has risen above THIS CANDIDATE'S OWN real, conviction-scaled buy
    gate (required_rr) -- gates whether a real Claude call
    (recommend_on_deck_retention, via the shared _on_deck_ai_gate_above_gate helper)
    is worth spending to judge whether this is still a good buy, rather than a fixed
    numeric ceiling or a grace-period timer (both tried and explicitly rejected the
    same evening -- owner: "i dont like the timer.. lets do this instead...
    whenever it isnt a good buy any more.. then evict it"). No margin parameter --
    the real gate is the one and only threshold here, unlike the floor check above.

    Used at exactly 2 real call sites: the persist-check retention sweep (any
    already-listed candidate), and _backfill_on_deck_from_on_shore specifically (the
    one admission path where a candidate has a genuine track record of having been
    watched rise past its own gate before getting bumped for an unrelated reason).
    The other 3 admission sites check this too, but mechanically exclude on True
    rather than asking Claude -- a candidate found above its own gate there has no
    such track record (owner: "ai is for a stock that has risen up past the gate,"
    not one simply discovered already above it)."""
    return rr > required_rr


def _on_deck_rr_ceiling_exceeded(rr: float, required_rr: float, ceiling_margin: float) -> bool:
    """True if rr exceeds required_rr by more than ceiling_margin -- a small tolerance
    band a first-look candidate (no track record) is mechanically admitted within,
    just above its own real gate, before the harder mechanical exclude kicks in
    (2026-08-20, owner request/design). Reintroduces the SHAPE of the flat
    ceiling-margin design removed 2026-08-05 (that one was found broken -- KEY evicted
    for landing exactly on its own gate, EQR/ALLY/SBRA evicted despite their real gates
    being far higher, since it compared against one ABSOLUTE flat number rather than
    each candidate's own conviction-scaled gate) but brought back as a per-candidate
    RELATIVE margin (matching how the floor margin below already works) and
    deliberately much smaller in magnitude (owner-set default 0.15, was 0.3) --
    scoped narrowly to catch genuine "basically at the gate" noise, confirmed against a
    real day's rejections before this was built: only 2 of 20 above-gate first-look
    rejections that day were within 0.10 of their own gate; the rest were 15-120% over
    -- the genuine "price already fell toward the stop, mechanically inflating the
    ratio" pattern this design must keep excluding regardless of the new tolerance.

    Used ONLY at the 3 mechanical-exclude admission sites _on_deck_rr_above_gate's own
    docstring names (the fresh universe-scan result, the startup cache restore, the
    on-demand Settings-triggered refill) -- replaces a bare _on_deck_rr_above_gate
    check as the reject condition at exactly those 3 sites. The AI-judgment sites
    (persist-check retention, On-Shore backfill, the buy trigger, the continuous
    above-gate recheck) are deliberately untouched -- those already ask a real,
    per-candidate judgment call instead of applying a blunt numeric line, so a small
    tolerance band adds nothing there."""
    return rr > required_rr + ceiling_margin


def _on_deck_composite_score(
    conviction: float, margin_of_safety_pct: float, rr: float, required_rr: float,
) -> float:
    """The valuation/R/R-aware composite score used to rank On Deck and On Shore
    candidates against each other (2026-07-31) -- conviction plus a margin-of-safety
    and R/R-vs-gate bonus. Extracted so DashboardState._on_deck_candidate_score and
    _backfill_on_deck_from_on_shore's local _shore_score share one formula instead of
    each maintaining their own copy that could drift -- the exact failure class the
    XRAY population-floor incident (above) was caused by, just in the composite-score
    formula instead of the population floor.

    The (rr - required_rr) term is capped at 0 once rr clears required_rr (fixed
    2026-08-12, OKE/BMY incident) -- previously uncapped, so a candidate far above its
    own gate (rr=2.43 vs. required 2.04, say) scored unboundedly higher the further
    above it sat, purely from valuation math, ahead of a candidate still genuinely in
    the qualifying watch range working toward its gate. This contradicted this
    project's own established stance elsewhere that "above the gate" is ambiguous, not
    simply better -- see _on_deck_rr_above_gate and the "AI is for a stock that has
    risen up past the gate" judgment-call design a few sections up in CLAUDE.md's
    On Deck Buy Pipeline section. Reaching the gate is now this score's ceiling for
    the R/R term; a candidate already past it gets no further bonus for having run
    even further, so ranking among already-qualified candidates falls to conviction
    and margin of safety instead of an unbounded R/R blowout. Below the gate, the
    term is unchanged (still a real, uncapped penalty -- further below the gate is
    still genuinely worse)."""
    rr_vs_gate = min(rr - required_rr, 0.0)
    return conviction + margin_of_safety_pct / 10 + rr_vs_gate * 2


def _on_deck_ranking_key(
    conviction: float, margin_of_safety_pct: float, rr: float, required_rr: float,
    min_conviction: float,
) -> tuple[bool, float]:
    """Ranking key for filling/trimming/swapping On Deck (2026-07-31, XRAY-adjacent
    fix) -- a candidate that actually clears the real buy gate (conviction >=
    min_conviction) must always outrank one that doesn't, regardless of composite
    score. Without this, a watch-only candidate (population-floor-eligible per
    _on_deck_population_floor above, but below the real buy gate) with a strong R/R or
    margin of safety could crowd out, or even bump via the weakest-member swap, a
    candidate that would actually clear a real promotion attempt right now -- a real,
    live risk once research.on_deck_conviction_band is widened enough to admit
    watch-only candidates that Claude's own scores actually produce (see the XRAY
    incident above for why a narrow band never triggered this in practice before).
    Returns (is_buy_eligible, composite_score); Python tuple comparison naturally
    ranks every True ahead of every False, falling back to the composite score only to
    break ties within the same tier -- this holds for a plain sort/min() as well as a
    threshold check like `challenger_key < (weakest_tier, weakest_score + margin)`,
    since the tier (first element) is always compared before the score."""
    is_buy_eligible = conviction >= min_conviction
    composite = _on_deck_composite_score(conviction, margin_of_safety_pct, rr, required_rr)
    return (is_buy_eligible, composite)


def _stop_loss_tightened(new_stop: float, current_stop: float) -> bool:
    """True if new_stop is strictly tighter (closer to the market, i.e. higher) than
    current_stop -- 2026-07-31, AI-chosen stop-loss/TP feature. A stop-loss is a price
    floor below the market; raising it means less room before it triggers, i.e. MORE
    protective. This is the tighten-only guardrail for AI re-analysis adjustments on an
    already-held position: a looser (lower) suggestion is never applied, matching the
    same one-way-ratchet invariant the existing graduated trailing stop already
    follows. See docs/superpowers/specs/2026-07-31-ai-chosen-stop-loss-tp-design.md."""
    return new_stop > current_stop


def _reconcile_ai_take_profit_targets(
    current_targets: list[float], fresh_targets: list[float],
) -> list[float]:
    """Applies a fresh AI-recommended take-profit ladder to an already-held position
    without corrupting the tranche-count invariant len(pos.take_profit_targets) encodes
    (fixed 2026-08-08, GitHub #55) -- a fresh analyze_stock() call always returns a
    full-length ladder computed off the current price, with no concept of which
    tranches THIS position has already sold. The old code (`pos.take_profit_targets =
    report.take_profit_targets`, unconditional) reset an already-partially-exited
    position's target list back to full length on every re-analysis (the periodic
    sweep, or either event trigger) -- the next sync_exit_orders pass then treated it
    as an untouched, freshly-bought position and re-split the already-reduced
    remaining shares into thirds again, replacing correctly-sized post-T1/T2 orders
    with wrong-sized ones, and silently un-firing the dashboard's T1/T2 badges for a
    tranche that had genuinely already sold and banked its proceeds. Same class of
    corruption as the RRC/ONB incident (stop-driven closures popping targets
    incorrectly), reintroduced through this different, newer feature.

    Preserves the CURRENT remaining-tranche count, refreshed with the AI's new price
    estimates for however many tranches remain -- mirrors this codebase's own existing
    front-popped-as-tranches-fire convention (targets shrink from the front as each
    tranche fires; the LAST element is always the final-tranche price) by taking the
    fresh ladder's own trailing N entries, so the prices update but the count doesn't.

    A position already down to 0 remaining targets (final tranche, riding the
    graduated trailing stop only per this codebase's own Exit Order System design --
    "After TP2 fires, NO fixed TP order is placed") never gets a target re-injected
    here regardless of what the AI recommends -- that's a deliberate, permanent state
    for that position, not something a routine re-analysis should accidentally
    resurrect a TP order for."""
    remaining = len(current_targets)
    if remaining == 0:
        return []
    if len(fresh_targets) < remaining:
        return current_targets
    return fresh_targets[-remaining:]


def _dip_recovery_dedup_cleared(
    failed_low: float | None, current_low: float,
    failed_at_price: float | None, current_price: float,
    recovery_retry_pct: float,
) -> bool:
    """Whether a dip-recovery promotion attempt should be allowed to re-fire for a candidate
    that already had one fail (2026-07-23) -- replaces a plain equality check
    (failed_low == current_low) that left a recovering candidate stuck indefinitely whenever
    its dip's own low hadn't changed, no matter how far price had since recovered past the
    level where the last attempt failed (confirmed live: DV, conviction and R/R both
    clearing their gates, stuck for hours since its low never printed a new value). Mirrors
    the escape hatch the no-dip trigger already had (no_dip_failed_at_pct_gain/
    no_dip_failed_at_up_count require the tracked measure to grow further, not just stay
    elevated) — the dip-recovery trigger had no equivalent until now.

    Returns True (a fresh attempt IS allowed) if EITHER the low has genuinely moved (a new
    low forming is real new information on its own, regardless of price), OR price has
    recovered at least recovery_retry_pct% further past the price recorded at the moment of
    the last failed attempt. Returns False (stay deduped) if the low is unchanged and either
    there's no recorded failed-attempt price to compare against (a candidate that predates
    this field) or price hasn't recovered far enough past it yet."""
    if failed_low is None or failed_low != current_low:
        return True
    if failed_at_price is None or failed_at_price <= 0:
        return False
    return current_price >= failed_at_price * (1 + recovery_retry_pct / 100)


def _dip_low_too_stale(low_t: float, now_ts: float, max_age_days: float) -> bool:
    """Whether dip_summary's tracked low is too old to represent a genuine, CURRENT dip
    (2026-07-28, RRC/OVV incident) — the mechanical "retracement" on_deck_entry_mode's own
    version of the staleness judgment "ai" mode now gets from Claude (see
    recommend_dip_entry's docstring for the full incident writeup). Mechanical mode has no
    Claude call in its path at all to reason about elapsed time the way the AI mode's prompt
    now does, so there's no way to let it "judge" staleness — a hardcoded recency floor is
    the only fix available here, same idea as this mode's other hardcoded knobs
    (on_deck_retracement_pct, on_deck_no_dip_pct_gain, on_deck_up_ticks_needed). Deliberately
    NOT applied to "ai" mode's own trigger path — that mode's whole point is letting Claude
    judge genuineness dynamically per the user's explicit direction ("it needs to be in an
    uptrend, how many days AI can figure out"), and gating it with a second, hardcoded cutoff
    on top of that judgment would just reintroduce the fixed-cutoff approach already rejected
    there. max_age_days <= 0 disables the check entirely (every low counts as fresh) — same
    "0 means off" convention as wash_sale_cooldown_days."""
    if max_age_days <= 0:
        return False
    return (now_ts - low_t) / 86400 > max_age_days


def _price_clears_block_breakout(
    current_price: float, ref_peak: float | None, breakout_pct: float,
) -> bool:
    """Whether a manually-removed On Deck ticker has demonstrably broken out of its stale
    situation, for free (2026-07-29, FNB discussion) -- user pushback on the removal
    dialog's blunt "N days"/permanent block: "can we use no cost price adjustments to see
    if its out of the long dip?" ref_peak is the resistance level (the dip_summary peak,
    or the ticker's own price if no dip was tracked) captured at removal time -- clearing
    it by a real margin, not just noise, confirms the stock is no longer stuck recovering
    toward that old level. None/zero/negative ref_peak means no reference was ever
    captured (e.g. no price history existed yet at removal time) -- can't auto-clear,
    falls back to the existing time-based block only."""
    if ref_peak is None or ref_peak <= 0:
        return False
    return current_price >= ref_peak * (1 + breakout_pct / 100)


def _normalize_block_entry(raw) -> dict:
    """Normalizes a loaded on_deck_blocked value into the current {"until", "ref_peak"}
    shape (2026-07-29) -- the stored value used to be a bare ISO string (temporary block)
    or None (permanent block) before the price-based breakout check above needed
    somewhere to keep a reference peak. Old cache files on disk still have the legacy
    shape, so this keeps them working without a migration script: a legacy None/string
    becomes a dict with ref_peak=None (no reference ever captured, matching pre-2026-07-29
    behavior exactly -- time-based expiry only), and an already-current dict passes
    through, defaulting a missing ref_peak key to None."""
    if isinstance(raw, dict):
        return {"until": raw.get("until"), "ref_peak": raw.get("ref_peak")}
    return {"until": raw, "ref_peak": None}


def _group_closed_trades(rows: list[tuple]) -> list[dict]:
    """Groups raw trade_history SELL rows into one logical trade per real buy (2026-07-29,
    Win/Loss dashboard stat fix) -- direct user request, stated twice: "no dont do each
    1/3 2/3 trades.. thats not accurate" / "make sure the 1/3 2/3 and full sale all go
    the 1 buy". A single position that banks a profitable T1, a profitable T2, then a
    losing final-tranche close used to count as 2 wins + 1 loss (3 separate "trades")
    instead of being judged by its combined outcome. Position.trade_id (shipped
    2026-07-27) is stamped on every tranche of the same original buy, so this sums pnl
    per trade_id and classifies the GROUP's total as one win or one loss.

    rows: (trade_id, ticker, pnl) tuples, straight from the trade_history table.
    Rows with trade_id is None (pre-migration legacy rows, before this column existed)
    each count as their own standalone trade, same as the original per-row behavior --
    there's no id to group them by. A None pnl contributes nothing to its group's total
    (treated as 0, not as a loss); a standalone (no trade_id) row with no pnl at all has
    nothing usable to judge and is dropped entirely, not silently counted as a loss.

    Order is preserve-first-seen per trade_id -- callers that want most-recent-trade-
    first should query rows already sorted timestamp DESC (see _closed_trades_since),
    so each trade's own most recent tranche is the first row encountered for its id.

    Returns one dict per closed trade: {trade_id, ticker, total_pnl, is_win}."""
    grouped: dict[str, dict] = {}
    trades: list[dict] = []
    for trade_id, ticker, pnl in rows:
        if trade_id is None:
            if pnl is not None:
                trades.append({"trade_id": None, "ticker": ticker,
                                "total_pnl": pnl, "is_win": pnl > 0})
            continue
        if trade_id not in grouped:
            grouped[trade_id] = {"trade_id": trade_id, "ticker": ticker, "total_pnl": 0.0}
        if pnl is not None:
            grouped[trade_id]["total_pnl"] += pnl
    for trade in grouped.values():
        trade["is_win"] = trade["total_pnl"] > 0
        trades.append(trade)
    return trades


def _loss_retrigger_should_fire(
    pnl_pct: float,
    loss_trigger_pct: float,
    retrigger_step_pct: float,
    worst_pct_since_last_fire: float | None,
    cooldown_active: bool,
) -> bool:
    """Whether an underwater position's event-triggered re-analysis should fire right now
    (2026-07-29, user request following the IVZ incident: "when a stock gets under entry
    by 2-3%... does it do an analysis... to see if it's worth keeping"). cooldown_active
    (position_monitor_event_cooldown_minutes, shared with the existing profitable-side
    event trigger) is the SOLE primary gate — mirrors that profitable-side trigger's own
    design exactly. The only way to fire again before the cooldown elapses is a
    materially deeper loss: at least retrigger_step_pct further past whatever loss level
    last triggered a fire, since that's a genuinely new situation the AI hasn't weighed in
    on yet (explicit user request: "maybe another [check] at 4% if it reaches that.. ai
    might say sell") — never a substitute for the cooldown itself.

    **Fixed same day (IVZ incident)** — an earlier version treated
    worst_pct_since_last_fire=None as an automatic bypass ("never fired before, so always
    fire"), paired with a caller that cleared that tracking to None the instant a
    position recovered even slightly above the raw threshold. Live-caught: IVZ's P&L
    oscillated right around -3.0% (noise between roughly -2.9% and -4.1%) for hours —
    every tiny tick back above -3.0% cleared the tracking, and every dip back below it
    was then treated as a brand-new "first crossing," firing immediately and bypassing an
    ALREADY-ACTIVE cooldown from the previous fire. 17 real Claude calls fired on IVZ
    alone in one day, nearly all of them wasteful repeats of the same unchanged
    situation. None now means only "no deepening reference to compare against" — it can
    no longer bypass an active cooldown by itself.

    pnl_pct: the position's current unrealized P/L % (negative when underwater).
    worst_pct_since_last_fire: the pnl_pct value recorded at the last fire, or None if
    this ticker has never fired yet."""
    if pnl_pct > -loss_trigger_pct:
        return False
    if not cooldown_active:
        return True
    return worst_pct_since_last_fire is not None and pnl_pct <= worst_pct_since_last_fire - retrigger_step_pct


def _in_exit_order_maintenance_window(now, pre_open_batch_time, market_close, is_holiday: bool) -> bool:
    """Pure decision behind DashboardState._exit_order_maintenance_window_open (2026-07-30,
    BEN incident) -- True during real market hours OR the pre-open window leading up to
    it, False on a weekend/holiday or during genuinely dead overnight hours. See the
    method's own docstring for the full incident (a stale/degenerate overnight Alpaca
    quote drove a real, if ultimately mitigated, false stop-loss cascade on BEN with no
    genuine price move behind it at all).

    now: an ET-aware datetime (from DashboardState._now_et()).
    pre_open_batch_time / market_close: plain time objects.
    is_holiday: DashboardState._is_holiday, already resolved by the caller."""
    if now.weekday() >= 5 or is_holiday:
        return False
    return pre_open_batch_time <= now.time() < market_close


def _required_rr(conviction: int, min_conviction: int, base_rr: float, step: float, floor: float) -> float:
    """Conviction-scaled R/R threshold (2026-07-18) — a flat min_risk_reward_ratio applied
    identically to every stock doesn't account for how much more (or less) confident Claude's
    thesis is; a stock right at the conviction minimum still needs the full base_rr margin of
    safety, but each conviction point above that earns a small reduction, since a stronger
    thesis needs less reward-to-risk cushion to be worth taking. Reduction is deliberately
    modest (small step, floored well above 1.0) — this narrows the gap for an
    already-qualified high-conviction stock, it never turns R/R into a rubber-stamp. Never
    adjusts for conviction below min_conviction (that path is already rejected by the
    conviction gate itself before this would ever be evaluated)."""
    extra_conviction = max(0, conviction - min_conviction)
    return max(floor, base_rr - extra_conviction * step)


def _yf_interval_and_period_for_days(days: int) -> tuple[str, str]:
    """Picks the finest intraday granularity yfinance actually supports for the requested
    window, falling back to coarser bars once the window exceeds each granularity's own
    lookback cap — verified live against the real API rather than assumed: 15-minute bars
    are capped at 60 days of history, hourly at 730 days, and daily has no practical cap for
    On Deck's purposes. Added 2026-07-18 per explicit request: daily closes gave only ~20-30
    points for a 30-day window; 15-minute bars give ~500+ for the same window, letting the
    chart actually show real recovery shape instead of one dot per day. Intraday intervals
    take an exact "Nd" day-count period string directly (confirmed working live) rather than
    yfinance's coarser named buckets (1mo/3mo/...) that _yf_period_for_days maps to — that
    function is still used here for the daily-fallback case, where day-count strings aren't
    the convention."""
    if days <= 60:
        return f"{days}d", "15m"
    if days <= 730:
        return f"{days}d", "1h"
    return _yf_period_for_days(days), "1d"


def _yf_period_for_days(days: int) -> str:
    """Maps a day count to yfinance's fixed set of supported periods, rounding UP so the
    fetch always covers at least the requested window (yfinance doesn't take an arbitrary
    day count — only 1d/5d/1mo/3mo/6mo/1y/2y/5y/10y/ytd/max)."""
    if days <= 5:
        return "5d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"


def _save_on_deck_cache(candidates: dict) -> None:
    """Persists On Deck list membership (ticker, conviction, fair value, thesis, etc.)
    across restarts and calendar days — added 2026-07-17 so a candidate isn't silently
    dropped by a restart or the next morning's pre-open wipe; it now only leaves the list
    when its conviction genuinely falls below on_deck_removal_conviction.

    price_history is INCLUDED here (changed 2026-07-18, was previously excluded and reset
    every day/restart) — per explicit user direction, the retracement check should
    track a candidate's full dip-and-recovery cycle for as long as it's continuously on the
    list, since a large, slower-moving stock can take multiple days or weeks to bottom and
    recover, not just one trading session. Now kept in full forever, never trimmed (a second,
    same-day direction — "we want to keep it all"), backed by 15-minute intraday bars rather
    than daily closes (a third same-day change) — together these mean real, unbounded growth
    per long-lived candidate (~2-3MB/year at 15-minute granularity, not the "few hundred KB"
    this comment originally estimated for daily-only, never-growing history), accepted
    knowingly as trivial at the scale of a handful of On Deck candidates. A candidate's
    fair_value_estimate can drift from what it was when the current low was recorded
    (mitigated by the persist-check phase re-analyzing and refreshing fair_value_estimate
    every pre-open regardless). Resets only
    when a candidate is newly added after being fully removed (see _candidate_entry) — not
    on every restart or every pre-open re-vet."""
    try:
        _ON_DECK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ON_DECK_CACHE_PATH.write_text(json.dumps(candidates, default=str), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save On Deck cache: %s", e)


def _load_on_deck_cache() -> dict:
    try:
        if _ON_DECK_CACHE_PATH.exists():
            data = json.loads(_ON_DECK_CACHE_PATH.read_text(encoding="utf-8"))
            for nm in data.values():
                nm.setdefault("price_history", [])
                nm["direction"] = None
                nm["streak"] = 0
                # ai_entry_pending never survives a restart -- any async recommendation call
                # in flight when the process stopped is simply gone, not resumable. Reset so
                # the next qualifying tick can fire a fresh one rather than staying stuck
                # thinking a call is still pending forever.
                nm["ai_entry_pending"] = False
                nm.setdefault("ai_entry_price", None)
                nm.setdefault("ai_entry_low_ref", None)
                nm.setdefault("ai_entry_reasoning", "")
                nm.setdefault("ai_entry_seen_below", False)
                # recent_directions (2026-07-21) resets same as direction/streak above — a
                # restart can't trust in-memory momentum tracked across the gap.
                nm["recent_directions"] = []
                # _debug_price_history deliberately left untouched if present — see
                # _save_on_deck_cache's docstring.
            return data
    except Exception as e:
        logger.warning("Failed to load On Deck cache: %s", e)
    return {}


def _save_on_deck_blocked(blocked: dict) -> None:
    """Manual On Deck removals (2026-07-18) — ticker -> ISO block-until timestamp, or None
    for a permanent block. Separate cache file from on_deck_cache.json since this is a small,
    independently-updated set (one write per manual removal, not per pre-open run)."""
    try:
        _ON_DECK_BLOCKED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ON_DECK_BLOCKED_CACHE_PATH.write_text(json.dumps(blocked), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save On Deck blocklist: %s", e)


def _load_on_deck_blocked() -> dict:
    try:
        if _ON_DECK_BLOCKED_CACHE_PATH.exists():
            return json.loads(_ON_DECK_BLOCKED_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load On Deck blocklist: %s", e)
    return {}


def _save_on_deck_notes(notes: dict) -> None:
    """Persistent per-ticker notes attached to a manual On Deck removal (2026-07-29) --
    deliberately a SEPARATE store from on_deck_blocked: a note is meant to survive even
    after the block itself clears (whether by time-based expiry or the price-based
    breakout check), so a ticker's re-analysis carries the context forward regardless of
    when/how it becomes eligible again. Never auto-purged (no un-note UI built yet, same
    precedent as on_deck_blocked's own "no un-block UI yet" gap)."""
    try:
        _ON_DECK_NOTES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ON_DECK_NOTES_CACHE_PATH.write_text(json.dumps(notes), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save On Deck notes: %s", e)


def _load_on_deck_notes() -> dict:
    try:
        if _ON_DECK_NOTES_CACHE_PATH.exists():
            return json.loads(_ON_DECK_NOTES_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load On Deck notes: %s", e)
    return {}


def _save_on_deck_stale_dip_low(stale_lows: dict) -> None:
    """Persists the dip low Claude has already judged genuinely stale for a ticker
    (2026-07-31, BRO repeat-decline incident) -- deliberately a SEPARATE store from
    on_deck_blocked, same precedent as on_deck_notes: this memory must survive even
    after the block itself clears (time-based expiry or the price-based breakout
    check), since the underlying reference low doesn't become fresh again just because
    the block lifted or price ticked up a bit. Without this, BRO was declined twice in
    one day (2026-07-29, ~6 minutes apart, identical stale-low reasoning both times) --
    the auto-eviction that fires on a stale verdict deletes the candidate's whole
    near_miss_candidates entry, including ai_entry_low_ref (the in-memory guard that
    already exists to stop exactly this kind of repeat ask), so a restore starts fresh
    with no memory of the low already having been judged dead. Never auto-purged (no
    un-note UI built yet, same precedent as on_deck_blocked/on_deck_notes) -- superseded
    naturally the moment a genuinely different (deeper) low is detected, via the same
    _dip_low_changed_meaningfully threshold already used for the in-memory guard."""
    try:
        _ON_DECK_STALE_DIP_LOW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ON_DECK_STALE_DIP_LOW_CACHE_PATH.write_text(json.dumps(stale_lows), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save On Deck stale dip low cache: %s", e)


def _load_on_deck_stale_dip_low() -> dict:
    try:
        if _ON_DECK_STALE_DIP_LOW_CACHE_PATH.exists():
            return json.loads(_ON_DECK_STALE_DIP_LOW_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load On Deck stale dip low cache: %s", e)
    return {}


def _save_event_monitor_cooldown(cooldown: dict, worst_pct: dict) -> None:
    """Persists the event-triggered position re-analysis cooldown (2026-07-30) --
    previously purely in-memory, so ANY restart (even one needed to deploy an urgent,
    unrelated fix) silently reset every ticker's cooldown clock to zero, causing an
    immediate real Claude re-fire for a condition that had just been checked minutes
    earlier and hadn't changed. Live-caught the same day: 3 restarts deploying real
    safety fixes each re-fired ONB's and GEN's event-triggered re-analysis within
    seconds of startup, on top of their normal cooldown-driven fires -- real, avoidable
    spend directly caused by deployment activity, not a genuine new trigger. cooldown's
    datetime values are serialized as ISO strings; worst_pct is saved alongside since
    the loss-retrigger logic depends on both surviving together."""
    try:
        _EVENT_MONITOR_COOLDOWN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cooldown": {k: v.isoformat() for k, v in cooldown.items()},
            "worst_pct": worst_pct,
        }
        _EVENT_MONITOR_COOLDOWN_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save event-monitor cooldown cache: %s", e)


def _load_event_monitor_cooldown() -> tuple[dict, dict]:
    """Returns (cooldown, worst_pct) -- see _save_event_monitor_cooldown's docstring.
    A cooldown timestamp already in the past (the process was down longer than the
    cooldown window) is kept as-is rather than dropped -- the caller's own
    `datetime.now() >= _cd` check already treats an expired entry as "not on
    cooldown," so restoring it changes nothing except avoiding an unnecessary write."""
    try:
        if _EVENT_MONITOR_COOLDOWN_CACHE_PATH.exists():
            data = json.loads(_EVENT_MONITOR_COOLDOWN_CACHE_PATH.read_text(encoding="utf-8"))
            cooldown = {k: datetime.fromisoformat(v) for k, v in data.get("cooldown", {}).items()}
            worst_pct = data.get("worst_pct", {})
            return cooldown, worst_pct
    except Exception as e:
        logger.warning("Failed to load event-monitor cooldown cache: %s", e)
    return {}, {}


def _save_midday_scan_fired(fired: dict) -> None:
    """Persists the mid-day re-scan per-slot firing tracker (2026-07-31 incident, same
    day the feature shipped) -- previously purely in-memory, so a restart AFTER a
    configured slot's time had already passed for the day made the restart-catch-up
    logic re-fire that slot again, exactly the same "wiped by restart" bug already
    fixed for _event_monitor_cooldown two days earlier. Live-caught deploying this
    very feature: restarting at 3:18 PM ET (past both default 10:30/13:30 slots) fired
    both simultaneously, and a manual verification trigger a minute later added a
    third concurrent scan on top. See docs/CLAUDE_HISTORY.md."""
    try:
        _MIDDAY_SCAN_FIRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MIDDAY_SCAN_FIRED_CACHE_PATH.write_text(json.dumps(fired), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save mid-day scan fired cache: %s", e)


def _load_midday_scan_fired() -> dict:
    try:
        if _MIDDAY_SCAN_FIRED_CACHE_PATH.exists():
            return json.loads(_MIDDAY_SCAN_FIRED_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load mid-day scan fired cache: %s", e)
    return {}


def _save_buy_reasoning(reasoning: dict) -> None:
    """The AI-entry recommendation that triggered a near-miss promotion buy (2026-07-20) —
    see DashboardState.buy_reasoning's docstring for why this exists. Small, independently-
    updated set (one write per promotion buy), same pattern as on_deck_blocked_cache.json."""
    try:
        _BUY_REASONING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BUY_REASONING_CACHE_PATH.write_text(json.dumps(reasoning), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save buy reasoning cache: %s", e)


def _load_buy_reasoning() -> dict:
    try:
        if _BUY_REASONING_CACHE_PATH.exists():
            return json.loads(_BUY_REASONING_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load buy reasoning cache: %s", e)
    return {}


def _save_promotion_attempts(attempts: list) -> None:
    try:
        _PROMOTION_ATTEMPTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROMOTION_ATTEMPTS_CACHE_PATH.write_text(json.dumps(attempts), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save promotion attempts cache: %s", e)


def _load_promotion_attempts() -> list:
    try:
        if _PROMOTION_ATTEMPTS_CACHE_PATH.exists():
            return json.loads(_PROMOTION_ATTEMPTS_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load promotion attempts cache: %s", e)
    return []


def _save_active_signals(signals: list) -> None:
    try:
        _ACTIVE_SIGNALS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_SIGNALS_CACHE_PATH.write_text(json.dumps(signals), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save active signals cache: %s", e)


def _load_active_signals() -> list:
    try:
        if _ACTIVE_SIGNALS_CACHE_PATH.exists():
            return json.loads(_ACTIVE_SIGNALS_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load active signals cache: %s", e)
    return []


def _save_performance_history(history: list) -> None:
    try:
        _PERFORMANCE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERFORMANCE_HISTORY_PATH.write_text(json.dumps(history), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save performance history: %s", e)


def _load_performance_history() -> list:
    try:
        if _PERFORMANCE_HISTORY_PATH.exists():
            return json.loads(_PERFORMANCE_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load performance history: %s", e)
    return []


def _annualized_pct(pnl_pct: float | None, days: int) -> float | None:
    """Naively compounds a real % gain over `days` out to a full year (2026-07-21, user's
    own request) -- e.g. +2.82% over 9 days compounds to roughly +209%/year. Explicitly a
    small-sample extrapolation, not a projection: a single unusually good or bad stretch
    gets blown up dramatically by this math, which is exactly why the frontend must always
    show the day count alongside it (get_portfolio_snapshot already includes
    total_annualized_days/ytd_annualized_days for this) rather than presenting the bare
    percentage as if it were a real expectation. Returns None if pnl_pct is None (no YTD
    baseline yet) or the gain is <= -100% (mathematically undefined to annualize a total
    wipeout)."""
    if pnl_pct is None or pnl_pct <= -100:
        return None
    daily_rate = (1 + pnl_pct / 100) ** (1 / days) - 1
    return round(((1 + daily_rate) ** 365 - 1) * 100, 1)


class DashboardState:
    def __init__(self):
        self.config = load_config()
        # Build universe from configured indexes (blocks briefly at startup — acceptable)
        enabled_indexes = self.config.get("research", {}).get("universe_indexes")
        STOCK_UNIVERSE[:] = get_universe(enabled_indexes)
        self.market_data = MarketDataFetcher(self.config)
        self.insider_tracker = InsiderTracker(self.config)
        self.news_feed = NewsFeed(self.config)
        self.research_engine = ResearchEngine(
            self.config, self.market_data, self.insider_tracker, self.news_feed
        )
        self.risk_manager = RiskManager(self.config)
        self.portfolio = Portfolio(self.config)
        self.signal_generator = SignalGenerator(
            self.config, self.research_engine, self.risk_manager, self.portfolio
        )
        self.fund_analyzer = FundamentalAnalyzer(self.config)
        self.sent_analyzer = SentimentAnalyzer(self.config)
        self.ins_analyzer = InsiderAnalyzer(self.config)
        self.comp_analyzer = CompetitorAnalyzer(self.config)

        self.ai_log: list[dict] = []
        self.trade_history: list[dict] = []
        self.active_signals: list[dict] = []
        self.ticker_signals: dict[str, dict] = {}
        # ticker -> "up"/"down" — tracked server-side (not per-browser-tab) so the price
        # direction arrow survives page refreshes and shows correctly on first mobile load,
        # instead of resetting to blank every time a client reconnects.
        self.price_direction: dict[str, str] = {}
        self.deep_dive_reports: dict[str, dict] = {}
        self.research_reports: dict[str, dict] = {}
        self.connected_clients: list[WebSocket] = []

        self.current_ticker: str = ""
        self.cycle_count: int = 0
        self.scan_index: int = 0       # which stock in the 50 we're on
        self.next_cycle_at: str = ""
        # Pause/Stop (2026-08-20, owner request) -- two independent severity levels,
        # both persisted to disk together (an in-memory-only pause silently un-pauses
        # on every restart/Apply Update with no warning -- a real gap this fix closes):
        #   paused=True  -- stops every AI-spend loop (position_monitor_loop,
        #                    position_deep_dive_loop, auto_scan_loop, watchlist_rr_loop,
        #                    near_miss_monitor_loop, the pre-open/midday batch paths).
        #                    Zero Claude calls. position_update_loop keeps running --
        #                    held positions stay fully protected (stop-loss/
        #                    trailing-stop/protection-gap checks/exit-order sync).
        #   stopped=True -- everything above PLUS position_update_loop itself stops.
        #                    No broker-side management of any kind. Deliberately does
        #                    NOT kill the process -- keeps the dashboard reachable so a
        #                    Start System click brings everything back with no
        #                    SSH/support needed (same reasoning as AICryptoTrading's
        #                    identical feature, built the same day).
        self._run_state_path = "data/run_state.json"
        self.paused: bool
        self.stopped: bool
        self.paused, self.stopped = self._load_run_state()
        # Concurrent On Deck promotion cash-reserve guard (fixed 2026-08-02, GitHub #44)
        # -- near_miss_monitor_loop can fire multiple _attempt_near_miss_promotion tasks
        # in the same tick with no shared lock, so each independently checked
        # self.portfolio.cash (only actually decremented once a buy round-trip completes)
        # and could jointly overspend past the reserve floor. _promotion_cash_lock
        # serializes the check-then-reserve step; _reserved_cash tracks how much is
        # provisionally committed by in-flight promotion attempts, released in each
        # attempt's own finally block regardless of how it exits.
        self._promotion_cash_lock = asyncio.Lock()
        self._reserved_cash: float = 0.0
        # Fresh-install banner (2026-07-21) -- True until a pre-open/universe scan has
        # completed at least once, ever (see _FIRST_SCAN_MARKER, checked in startup()).
        # Never auto-fires a scan on its own; only true to let the dashboard show a
        # "run your first scan?" banner the user can accept or dismiss on their own terms.
        self.needs_first_scan: bool = False

        self.has_claude = bool(os.getenv("ANTHROPIC_API_KEY", ""))
        self.stock_delay = 15 if self.has_claude else INTER_STOCK_DELAY

        self.order_manager = OrderManager(self.config, self.portfolio)
        self.trade_logger = TradeLogger(self.config)
        self.broker_connected: bool = False
        self.pending_confirmations: dict[str, dict] = {}
        # Per-ticker guard for the confirm_buy critical section (GitHub issue #26) — two
        # concurrent confirm_buy requests for the same ticker (e.g. two browser tabs) could
        # both pass check_all_rules before either one's execute() actually deducts cash,
        # since a separate confirmation_id from each preview lets both requests reach the
        # order-placement call independently. A plain guard set (not order_manager's own
        # per-ticker asyncio.Lock) is used deliberately: execute() -> _execute_buy() already
        # acquires that lock internally, so reusing it here for the outer confirm_buy section
        # would self-deadlock the moment execute() tried to acquire it a second time.
        self._confirm_in_progress: set[str] = set()
        self._replacement_scan_running: bool = False
        # Per-ticker cooldown for auto-close: once fired, suppress re-trigger for 5 min.
        # Prevents repeated market-sell submissions when a pre-market trailing stop keeps
        # the condition true across multiple 30-second position-monitor cycles.
        self._auto_close_cooldown: dict[str, datetime] = {}
        # Tracks which stop/trailing-stop conditions are CURRENTLY active per ticker
        # (2026-07-23, user request) -- lets the position-monitor loop detect the
        # transition back to "price recovered above the stop" and log a matching
        # recovery message, symmetric to the existing "triggered" one. Without this,
        # a position sitting below its stop pre-market (correctly held back from an
        # actual close by the market-hours gate) could recover before 9:30 with no
        # visible confirmation that the condition had cleared.
        #
        # Also now the ONLY gate on the "triggered" log line itself (fixed 2026-07-29,
        # MET incident -- overnight, MET's price froze at one stale value below its
        # trailing stop for 4+ hours with no new quotes coming in, and the RISK log
        # re-fired the identical message every 5 minutes the whole time, ~48 lines for
        # one unchanged condition). A prior fix (removed) used a time-based 5-minute
        # cooldown to cut this down from every-10-seconds to every-5-minutes, which
        # helped but didn't solve it for a long-lived condition. Membership in this set
        # is a true one-shot: the log line only fires on the tick a condition transitions
        # from absent to present, never again until the matching recovery branch clears
        # it -- exactly one "triggered" + one "recovered" message per episode, regardless
        # of how many hours it persists or whether the market is even open.
        self._risk_condition_active: dict[str, set[str]] = {}
        # Per-ticker cooldown for the event-triggered position re-analysis (2026-07-27) --
        # a comfortably profitable position only gets re-analyzed when price nears its
        # trailing stop or next target (see position_monitor_loop's docstring), checked
        # every 10s alongside the trailing-stop calc in position_update_loop. This
        # cooldown stops price lingering near a trigger zone from firing a real Claude
        # call on every single tick.
        self._event_monitor_cooldown: dict[str, datetime] = {}
        # Deepest unrealized_pnl_pct that has already fired the underwater-position event
        # trigger (2026-07-29) -- see _loss_retrigger_should_fire's docstring. Cleared for
        # a ticker once it recovers back above -position_monitor_loss_trigger_pct, so a
        # future new decline starts fresh at the base threshold.
        self._loss_event_worst_pct: dict[str, float] = {}
        # Current graduated trailing-stop trail width (%) per ticker (2026-07-23), purely
        # for display — the Positions table shows this in parens next to the stop figure
        # (e.g. "$17.86 (3%)") so the user can see where on the entry->T3 curve a position
        # currently sits. Recomputed every position_update_loop tick; never read back into
        # any trading decision.
        self._trailing_stop_pct_display: dict[str, float] = {}
        # Edge-triggered alert tracking for check_protection_gaps() (2026-07-21) — first
        # detection of a real stop/TP gap fires immediately; re-alerts every 5 min while it
        # persists (same cooldown window as _auto_close_cooldown above, not batched/silent
        # for up to an hour like EPRT's gap was before this existed). Deliberately still a
        # repeating alert, unlike the stop-loss/trailing-stop RISK log above (2026-07-29) --
        # a real, unresolved missing-order gap stays actionable and worth re-surfacing every
        # few minutes, whereas a stop/trailing-stop breach that's already fully logged once
        # has nothing new to say until it actually changes. Cleared for a ticker the moment
        # it's covered again, so a future recurrence re-alerts.
        self._protection_gap_alerted: dict[str, datetime] = {}
        # First-detection timestamp per ticker (2026-07-23) -- the VISIBLE alert above is
        # now delayed until a gap has been continuously open for
        # risk_management.protection_gap_alert_delay_seconds (default 30s, chosen from
        # live observation: routine DAY-order-renewal gaps self-heal in ~10-20s). Most
        # protection gaps are exactly this kind of routine, near-instant self-heal -- the
        # 10s detection/remediation cadence was never the problem, but showing every one
        # of them as an urgent "⚠️ PROTECTION GAP" alert (with a push notification) made a
        # normal, already-self-correcting event look alarming. Remediation
        # (sync_exit_orders) still fires every single cycle regardless of this delay --
        # only the user-facing alert is deferred.
        self._protection_gap_first_seen: dict[str, datetime] = {}
        # Near-miss candidates: BUY-signal, conviction-qualified stocks rejected at
        # pre-open only for R/R (too expensive relative to fair value right now).
        # Rebuilt fresh every pre-open scan (cleared at the top of _run_pre_open_batch).
        # Monitored for free (yfinance, no Claude) during market hours by
        # near_miss_monitor_loop; promoted straight to a buy (not the watchlist) when
        # R/R recovers past the gate AND a confirmed uptick shows the price has
        # actually stopped falling. See CLAUDE.md "Near-Miss Candidates" for the design.
        self.near_miss_candidates: dict[str, dict] = {}
        # Manual On Deck removals (2026-07-18) — user-initiated, distinct from the automatic
        # conviction-based removal above. ticker -> ISO block-until timestamp, or None for a
        # permanent block. Checked by every candidate-creation site so a manually-removed
        # stock doesn't silently reappear on the very next pre-open scan. Purely additive to
        # the existing removal path — a blocked ticker is never added in the first place, it's
        # not a filter applied after the fact.
        self.on_deck_blocked: dict[str, dict] = {}
        # Persistent per-ticker notes attached to a manual On Deck removal (2026-07-29) --
        # separate from on_deck_blocked above so a note survives even after the block
        # itself clears (time-based or price-based) — see remove_on_deck_candidate and
        # user_note_summary (src/research/engine.py).
        self.on_deck_notes: dict[str, str] = {}
        # ticker -> the dip low Claude has already judged genuinely stale (2026-07-31) --
        # see _save_on_deck_stale_dip_low's docstring. Separate from on_deck_blocked/
        # on_deck_notes: must survive a block clearing, since the underlying low doesn't
        # become fresh again just because the block lifted.
        self.on_deck_stale_dip_low: dict[str, float] = {}
        # Cached AI portfolio health assessment (2026-07-20) — one entry, not per-ticker.
        # In-memory only, no disk persistence — cheap to regenerate, non-critical.
        # 30-minute TTL enforced at the /api/portfolio-health endpoint.
        self.portfolio_health_cache: dict = {}
        # Win/Loss dashboard stat cache (2026-07-29) -- get_portfolio_snapshot() is
        # synchronous and called on every portfolio broadcast (as often as every ~10s),
        # so the real trade_id-grouped DB query (_closed_trades_since) can't run inline
        # there. Refreshed once at startup and every near_miss_monitor_loop tick (60s) --
        # cheap enough for a plain SELECT, and a stat like this doesn't need split-second
        # freshness the way Day P/L does. Empty dict means "not computed yet"; the
        # frontend/get_portfolio_snapshot both treat that the same as zero trades.
        self._win_rate_cache: dict = {}
        # AI-entry recommendation that actually triggered a near-miss promotion buy
        # (2026-07-20) — ticker -> {ai_entry_price, ai_entry_reasoning, recorded_at}.
        # Without this, the specific recommendation/reasoning that led to the buy
        # (visible on the On Deck card right up until the moment of purchase) simply
        # vanished the instant the ticker left near_miss_candidates and became a held
        # position, with nothing in the position detail view explaining why it was
        # bought. Keyed by ticker, not position id -- if the same ticker is later
        # bought again via a different path (manual, a fresh promotion) without this
        # entry being cleared first, the position detail view could show stale
        # reasoning from the earlier buy; accepted as a minor, purely informational
        # limitation rather than plumbing position-instance tracking for it.
        self.buy_reasoning: dict[str, dict] = {}

        # Real-time log of every promotion attempt, win or lose (2026-07-21) -- repurposes
        # the old Candidates tab; see _PROMOTION_ATTEMPTS_CACHE_PATH's comment for why.
        # Capped at 100 entries (most-recent-first) so the cache file and WS payload can't
        # grow unbounded over a long-running install.
        self.promotion_attempts: list[dict] = []

        research_cfg = self.config.get("research", {})
        self.scans_per_day = research_cfg.get("scans_per_day", 3)
        h, m = research_cfg.get("market_open", "09:30").split(":")
        self.market_open = dtime(int(h), int(m))
        h, m = research_cfg.get("market_close", "16:00").split(":")
        self.market_close = dtime(int(h), int(m))
        self.market_tz = ZoneInfo(research_cfg.get("market_timezone", "America/New_York"))
        # This install's own real go-live date -- see _get_or_init_account_genesis's
        # docstring. Computed here (not module-level, unlike the constant it replaces)
        # since it needs a real ET "today," which needs market_tz just set above.
        self.live_account_start = _get_or_init_account_genesis(
            datetime.now(self.market_tz).strftime("%Y-%m-%d"))

        self.explicit_scan_times: list[dtime] = []
        for t_str in research_cfg.get("scan_times", []):
            sh, sm = t_str.split(":")
            self.explicit_scan_times.append(dtime(int(sh), int(sm)))

        self.midday_scan_times: list[dtime] = []
        for t_str in research_cfg.get("midday_scan_times", []):
            sh, sm = t_str.split(":")
            self.midday_scan_times.append(dtime(int(sh), int(sm)))
        # scan-time-string ("10:30") -> ISO date last fired, e.g. {"10:30": "2026-07-31"}.
        # Per-slot (not a single date flag, since there are 2+ configured times/day) --
        # see auto_scan_loop's firing check. Persisted (2026-07-31 incident) -- see
        # _save_midday_scan_fired's docstring.
        self._midday_scan_fired: dict[str, str] = {}
        # Concurrency guard (2026-07-31 incident) -- set by whichever caller (the
        # scheduled auto_scan_loop firing check, or the manual /api/trigger-midday-rescan
        # endpoint) wins the race to start a mid-day scan; _run_midday_rescan itself clears
        # it in a finally block when done. Prevents 2+ mid-day scans running concurrently,
        # which caused real duplicate Claude spend when a restart's catch-up logic fired
        # both configured slots at once and a manual trigger added a third on top.
        self._midday_rescan_in_progress: bool = False

        # Full-scan-on-demand concurrency guard (2026-08-03) -- the dashboard's manual
        # "Full Scan" button now runs the same full universe pipeline as the pre-open batch
        # (_run_pre_open_batch), not just an On Deck re-vet, so it needs the same
        # already-in-progress protection _midday_rescan_in_progress gives that feature.
        # Owned by _run_full_scan_on_demand (see /api/trigger-batch-scan), which wraps
        # _run_pre_open_batch itself unchanged rather than adding a try/finally inside that
        # already-long, already-live function.
        self._full_scan_in_progress: bool = False

        # Update-available apply guard (2026-08-12, owner concern: "someone would push
        # the button again") -- the frontend already disables the clicked button for
        # the duration of one apply, but closing and reopening the update panel rebuilds
        # a fresh, un-disabled button with no memory of an apply already running. Same
        # in-progress-flag pattern as _full_scan_in_progress above, so a second click
        # can't launch a second concurrent download/extract/copy/restart cycle.
        self._apply_update_in_progress: bool = False

        # Update-status GitHub-fetch cache (2026-08-12, owner report: "if i have to
        # refresh to get it thats a problem" -- the dashboard badge needed to poll
        # periodically, not just check once on page load. Polling this endpoint every
        # ~60s from an open dashboard tab is fine on its own, but calling
        # fetch_latest_release() (a real GitHub API hit) on every single one of those
        # polls, from potentially several open tabs/devices at once, risks GitHub's
        # 60-req/hour-per-IP unauthenticated limit. This caches the real GitHub result
        # for update.check_interval_minutes (default 60) -- the frontend can poll the
        # cheap local endpoint often; the actual GitHub lookup stays rare.
        self._update_status_cache: dict | None = None
        self._update_status_cache_time: datetime | None = None

        # On Deck backfill reject cooldown (2026-08-03, owner request) -- a ticker that just
        # failed a real fresh re-check inside _backfill_on_deck_from_on_shore's _try_add
        # (no longer qualifies, or outside the R/R band) is the top-ranked On Shore
        # candidate by definition, so without this it gets re-tried -- another real Claude
        # call -- on literally every 60s near_miss_monitor_loop tick until it either
        # stabilizes into range or something else outranks it. Confirmed live: APLE was
        # re-checked 7+ times in ~5 minutes while its R/R swung around outside the 1.8-2.5
        # band. In-memory only (deliberately not persisted) -- the cost this protects
        # against is same-session tick-to-tick repetition, not something worth surviving a
        # restart for.
        self._on_deck_backfill_reject_cooldown: dict[str, datetime] = {}

        # On Deck continuous above-gate re-check cooldown (2026-08-18, owner request) --
        # near_miss_monitor_loop's 60s tick already sweep-evicts on the FLOOR side for free
        # (no Claude call, just live-price math -- see the 2026-08-10 to_evict_rr_floor
        # entry below). The ABOVE-gate side never had a continuous equivalent: a candidate
        # that drifts above its own gate only gets AI-rejudged (_on_deck_ai_gate_above_gate)
        # at the twice-daily persist-check, so it can sit visibly above-gate for hours with
        # nothing re-checking it. Unlike the floor side, above-gate judgment is a real,
        # billed Claude call, so making it continuous needs a cooldown -- without one, a
        # candidate that stays above gate would re-fire the same judgment every single 60s
        # tick indefinitely. In-memory only (deliberately not persisted), same precedent as
        # _on_deck_backfill_reject_cooldown just above -- a restart losing a few minutes of
        # this cooldown is a trivial cost, not worth surviving a restart for.
        self._on_deck_above_gate_cooldown: dict[str, datetime] = {}

        # On Deck backfill above-gate DECLINE cooldown (2026-08-18, SNDK incident) --
        # separate from, and much longer than, the generic reject cooldown just above.
        # Confirmed live: SNDK got sent through the real above-gate AI retention judgment
        # (_on_deck_ai_gate_above_gate) inside _backfill_on_deck_from_on_shore's _try_add
        # ~23 times in one afternoon, every ~5-6 minutes, declined the same way every
        # time ("R/R inflated by continued decline, no longer a good buy"). The generic
        # 5-minute reject cooldown was working exactly as designed -- it just isn't a
        # long enough gap for a stock in ongoing decline to stop being a stock in ongoing
        # decline. A value-change threshold (like _dip_low_changed_meaningfully's) doesn't
        # fit here the way it does for dip lows: SNDK's own R/R genuinely moved a fair
        # amount between checks (mechanical inflation as price kept falling), yet the
        # real answer never changed, so gating on "did R/R move enough" wouldn't have
        # caught this. A dedicated longer cooldown, set only on an above-gate decline
        # specifically (not on the other _try_add_inner failure reasons, which already
        # tend to resolve faster), is the more direct fix. In-memory only, same
        # not-worth-persisting-across-a-restart precedent as its two siblings above.
        self._on_deck_backfill_above_gate_cooldown: dict[str, datetime] = {}

        self.position_monitor_interval = research_cfg.get("position_monitor_interval_minutes", 60)
        self._is_holiday: bool = False
        self._holiday_check_date: str = ""
        pre_open_hours = research_cfg.get("pre_open_batch_hours", 2)
        from datetime import datetime as _dt, timedelta as _td
        _open_dt = _dt.combine(_dt.today(), self.market_open)
        self.pre_open_batch_time: dtime = (_open_dt - _td(hours=pre_open_hours)).time()
        # If restarting after the trigger time, mark today as already scanned so we don't re-run
        from zoneinfo import ZoneInfo as _ZI
        _now_et = _dt.now(_ZI("America/New_York"))
        self._pre_open_batch_date: str = (
            _now_et.strftime("%Y-%m-%d") if _now_et.time() >= self.pre_open_batch_time else ""
        )
        # Daily recap — fires once per weekday shortly after market close
        self.daily_report_time: dtime = (
            _dt.combine(_dt.today(), self.market_close) + _td(minutes=5)
        ).time()
        self._daily_report_date: str = (
            _now_et.strftime("%Y-%m-%d") if _now_et.time() >= self.daily_report_time else ""
        )
        # Daily portfolio-vs-market performance snapshot (2026-07-21) -- same once-per-day
        # guard pattern as the daily report right above, fired from the same trigger point.
        self._performance_snapshot_date: str = self._daily_report_date
        self.performance_history: list[dict] = []

        db_path = self.config.get("database", {}).get("path", "data/aitrading.db")
        self.watchlist_manager = WatchlistManager(
            db_path=db_path,
            target_size=research_cfg.get("watchlist_size", 50),
            weak_threshold=research_cfg.get("weak_signal_threshold", 3),
        )

        # Restore last-known signals from DB so badges show immediately after restart
        for ticker, sig in self.watchlist_manager.get_last_signals().items():
            if sig:
                self.ticker_signals[ticker] = {
                    "ticker": ticker, "signal": sig, "conviction": 0, "price": 0, "time": "--",
                }

        # AI log persistence — init table and reload last 300 entries
        self._log_db_path = db_path
        self._init_log_db()
        self.ai_log = self._load_log_from_db()

    def _load_run_state(self) -> tuple[bool, bool]:
        try:
            data = json.loads(Path(self._run_state_path).read_text(encoding="utf-8"))
            return bool(data.get("paused", False)), bool(data.get("stopped", False))
        except Exception:
            return False, False

    def _save_run_state(self):
        try:
            Path(self._run_state_path).write_text(
                json.dumps({"paused": self.paused, "stopped": self.stopped}), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save run state: %s", e)

    def run_status(self) -> str:
        """Single source of truth for the 3-way UI state -- 'stopped' takes priority
        over 'paused' since it's the stronger condition (see the paused/stopped
        __init__ comment for exactly what each level gates)."""
        if self.stopped:
            return "stopped"
        if self.paused:
            return "paused"
        return "running"

    def _init_log_db(self):
        import sqlite3 as _sqlite3
        with _sqlite3.connect(self._log_db_path, timeout=_SQLITE_TIMEOUT_SECS) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    phase TEXT,
                    content TEXT,
                    level TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def _load_log_from_db(self) -> list[dict]:
        import sqlite3 as _sqlite3
        with _sqlite3.connect(self._log_db_path, timeout=_SQLITE_TIMEOUT_SECS) as conn:
            rows = conn.execute(
                "SELECT timestamp, ticker, phase, content, level FROM ai_log "
                "ORDER BY id DESC LIMIT 300"
            ).fetchall()
        return [
            {"timestamp": r[0], "ticker": r[1], "phase": r[2], "content": r[3], "level": r[4]}
            for r in reversed(rows)
        ]

    def _persist_log_entry(self, entry: dict):
        import sqlite3 as _sqlite3
        try:
            with _sqlite3.connect(self._log_db_path, timeout=_SQLITE_TIMEOUT_SECS) as conn:
                conn.execute(
                    "INSERT INTO ai_log (timestamp, ticker, phase, content, level, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entry["timestamp"], entry["ticker"], entry["phase"],
                     entry["content"], entry["level"], datetime.now().isoformat()),
                )
                # Keep table from growing forever — prune to 1000 rows
                conn.execute(
                    "DELETE FROM ai_log WHERE id NOT IN "
                    "(SELECT id FROM ai_log ORDER BY id DESC LIMIT 1000)"
                )
                conn.commit()
        except Exception as e:
            logger.debug("ai_log persist error: %s", e)

    # ── WebSocket helpers ──────────────────────────────────────────────────

    async def broadcast(self, msg: dict):
        dead = []
        # Snapshot the list before iterating -- send_json's await yields control, and a
        # concurrent disconnect handler mutating self.connected_clients mid-loop can shift
        # list indices and silently skip a client (GitHub issue #24).
        for ws in list(self.connected_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.connected_clients:
                self.connected_clients.remove(ws)

    def add_ai_log(self, ticker: str, phase: str, content: str, level: str = "info") -> dict:
        entry = {
            "timestamp": self._now_et().strftime("%b %d %H:%M:%S"),
            "ticker": ticker,
            "phase": phase,
            "content": content,
            "level": level,
        }
        self.ai_log.append(entry)
        if len(self.ai_log) > 500:
            self.ai_log = self.ai_log[-500:]
        # Fire-and-forget (2026-07-18) — _persist_log_entry does a synchronous sqlite3
        # connect + INSERT + prune-DELETE + commit, blocking the event loop for its full
        # duration on every single call. add_ai_log is called ~hundreds of times per busy
        # pre-open scan and every 60s from near_miss_monitor_loop, so this was a real,
        # confirmed-current instance of the CR8-6 finding (deferred since 2026-07-05,
        # never revisited). Same asyncio.create_task(asyncio.to_thread(...)) pattern
        # already used for _save_dd_cache/_save_report_cache (CR16-4) — add_ai_log's
        # signature and return value are unchanged, so none of its ~118 call sites needed
        # to change.
        asyncio.create_task(asyncio.to_thread(self._persist_log_entry, entry))
        return entry

    # ── Portfolio snapshot ─────────────────────────────────────────────────

    def get_portfolio_snapshot(self) -> dict:
        p = self.portfolio
        positions = [
            {
                "ticker": pos.ticker,
                "shares": pos.shares,
                "entry_price": round(pos.entry_price, 2),
                "current_price": round(pos.current_price, 2),
                "market_value": round(pos.market_value, 2),
                "day_pnl": round(pos.day_pnl, 2),
                "day_pnl_pct": round(pos.day_pnl_pct, 2),
                # "pnl"/"pnl_pct" switched from raw unrealized_pnl to lifetime_pnl (2026-07-22)
                # -- a position with a fired T1/T2 was silently showing ONLY the remaining
                # shares' gain, dropping the real profit/loss already banked (and sitting in
                # cash) from those earlier partial exits. See Position.lifetime_pnl.
                "pnl": round(pos.lifetime_pnl, 2),
                "pnl_pct": round(pos.lifetime_pnl_pct, 2),
                "stop_loss": round(pos.stop_loss, 2),
                "trailing_stop": round(pos.trailing_stop, 2) if pos.trailing_stop is not None else None,
                "trailing_stop_pct": self._trailing_stop_pct_display.get(pos.ticker),
                "sector": pos.sector,
                "take_profit_targets": [round(t, 2) for t in (pos.take_profit_targets or [])],
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                "price_direction": self.price_direction.get(pos.ticker),
                # 2026-07-27, for the Positions table's ID/Targets columns.
                "trade_id": pos.trade_id,
                "t1_target_price": round(pos.t1_target_price, 2) if pos.t1_target_price is not None else None,
                "t2_target_price": round(pos.t2_target_price, 2) if pos.t2_target_price is not None else None,
                # "Why AI Bought This" (2026-08-21) -- see Position's own field docstring.
                "buy_thesis": pos.buy_thesis,
                "buy_reasoning": pos.buy_reasoning,
                "buy_conviction": pos.buy_conviction,
                "buy_signal": pos.buy_signal,
                "buy_rr": round(pos.buy_rr, 2) if pos.buy_rr is not None else None,
                "buy_required_rr": round(pos.buy_required_rr, 2) if pos.buy_required_rr is not None else None,
                "buy_fair_value": round(pos.buy_fair_value, 2) if pos.buy_fair_value is not None else None,
            }
            for pos in p.positions.values()
        ]
        ytd_pnl, ytd_pnl_pct = self._ytd_pnl()
        week_pnl, week_pnl_pct = self._week_pnl()
        week_number = self._now_et().date().isocalendar()[1]
        return {
            "total_value": round(p.total_value, 2),
            "cash": round(p.cash, 2),
            "cash_pct": round(p.cash_pct, 1),
            "day_pnl": round(p.day_pnl, 2),
            "day_pnl_pct": round((p.day_pnl / p.day_start_value * 100) if p.day_start_value else 0, 2),
            "total_pnl": round(p.total_pnl, 2),
            "total_pnl_pct": round(p.total_pnl_pct, 2),
            "total_annualized_pct": _annualized_pct(
                p.total_pnl_pct, self._days_since(self.live_account_start)),
            "total_annualized_days": self._days_since(self.live_account_start),
            "ytd_pnl": ytd_pnl,
            "ytd_pnl_pct": ytd_pnl_pct,
            "ytd_annualized_pct": _annualized_pct(ytd_pnl_pct, self._days_since(
                max(self.live_account_start, f"{self._now_et().year}-01-01"))),
            "ytd_annualized_days": self._days_since(
                max(self.live_account_start, f"{self._now_et().year}-01-01")),
            # No annualized figure for week_pnl (2026-07-27) -- a 7-day window extrapolated
            # to a full year is even noisier than the existing "low confidence" Total/YTD
            # annualized figures, so this follows Day P/L's precedent (also no annualized)
            # rather than Total/YTD's.
            "week_pnl": week_pnl,
            "week_pnl_pct": week_pnl_pct,
            "week_number": week_number,
            "positions": positions,
            "position_count": len(positions),
            # Win/Loss stat tile (2026-07-29) -- read from the cache refreshed by
            # _refresh_win_rate_cache (startup + every near_miss_monitor_loop tick), not
            # computed inline here -- this method is synchronous and called on every
            # portfolio broadcast, too often for a live DB query.
            "win_rate_current_arch_pct": self._win_rate_cache.get("win_rate_current_arch_pct", 0.0),
            "closed_current_arch": self._win_rate_cache.get("closed_current_arch", 0),
            "win_rate_all_time_pct": self._win_rate_cache.get("win_rate_all_time_pct", 0.0),
            "closed_all_time": self._win_rate_cache.get("closed_all_time", 0),
        }

    def get_init_payload(self) -> dict:
        """Everything websocket_endpoint sends as its first message on a fresh connection —
        factored out (2026-07-23) so /api/dashboard-poll (the HTTP-polling fallback used
        when a client's WebSocket can't stay connected — see GET /api/dashboard-poll's own
        docstring) can serve byte-identical data through a completely different transport,
        rather than drifting out of sync with a second hand-maintained copy."""
        return {
            "type": "init",
            "portfolio": self.get_portfolio_snapshot(),
            "ai_log": self.ai_log[-150:],
            "signals": self.active_signals,
            "ticker_signals": self.ticker_signals,
            "deep_dive_reports": self.deep_dive_reports,
            "broker_status": {
                "connected": self.broker_connected,
                "broker": self.config["trading"]["broker"],
                "paper_trading": self.config["trading"]["paper_trading"],
            },
            "stocks": self.watchlist_manager.get_active(),
            "watchlist_size": self.watchlist_manager.size(),
            "watchlist_target": self.watchlist_manager.target_size,
            "scan_status": self.get_scan_status(),
            "max_positions": self.config.get("portfolio", {}).get("max_positions", 10),
            "needs_first_scan": self.needs_first_scan,
            "promotion_attempts": self.promotion_attempts,
            "market_open": self._is_market_open(),
            # Reconciles the "Full Scan" button's disabled/"Scanning..." state on every
            # fresh WS connection and HTTP-polling-fallback tick (2026-08-09, GitHub #61)
            # -- the button previously only ever re-enabled via the one-shot
            # full_scan_done broadcast, which a client disconnected during the ~20-minute
            # scan window (an ordinary, expected drop this dashboard already has a whole
            # disconnect-banner/polling-fallback system for) would simply never receive,
            # leaving it stuck disabled indefinitely after reconnect.
            "full_scan_in_progress": self._full_scan_in_progress,
        }

    def _days_since(self, date_str: str) -> int:
        """Whole days between date_str (YYYY-MM-DD) and today, floored at 1 so a same-day
        calculation never divides by zero in _annualized_pct below."""
        try:
            start = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return 1
        return max(1, (self._now_et().replace(tzinfo=None) - start).days)

    async def _closed_trades_since(self, cutoff: str) -> list[dict]:
        """Real trade_id-grouped closed trades since cutoff (2026-07-29 Win/Loss stat
        fix) -- shared by get_portfolio_health's win-rate figures and
        _refresh_win_rate_cache, so both read the identical grouping logic
        (_group_closed_trades) instead of two copies that could drift. See that
        function's docstring for why grouping by trade_id matters (a position's T1/T2/
        final tranches must count as ONE trade, not three).

        Floors the effective cutoff to _TRADE_ID_RELIABLE_SINCE (2026-07-30 fix) --
        callers may legitimately ask for an earlier date (self.live_account_start,
        _CURRENT_ARCHITECTURE_START), but trade_id itself didn't exist before
        2026-07-27, so honoring an earlier request as-is would include known
        NULL-trade_id rows that _group_closed_trades can only count individually,
        re-introducing the exact per-tranche-counted-as-a-separate-trade distortion
        this whole feature was built to eliminate."""
        effective_cutoff = max(cutoff, _TRADE_ID_RELIABLE_SINCE)
        if not self.portfolio._db:
            return []
        async with self.portfolio._db.execute(
            "SELECT trade_id, ticker, pnl FROM trade_history "
            "WHERE action = 'SELL' AND timestamp >= ? "
            "ORDER BY timestamp DESC",
            (effective_cutoff,),
        ) as cur:
            rows = await cur.fetchall()
        return _group_closed_trades(rows)

    async def _refresh_win_rate_cache(self) -> None:
        """Recomputes the Win/Loss dashboard stat tile's cached figures (2026-07-29).
        Uses the CURRENT-architecture cutoff as the tile's headline figure, same
        reasoning as get_portfolio_health's existing split: a stat blending in trades
        decided by the pre-2026-07-17 Watchlist-based system (which no longer exists)
        isn't a fair read on how the current logic performs. All-time is kept alongside
        it for the tile's popup detail view."""
        current_arch_trades = await self._closed_trades_since(_CURRENT_ARCHITECTURE_START)
        all_time_trades = await self._closed_trades_since(self.live_account_start)
        closed_current_arch = len(current_arch_trades)
        wins_current_arch = sum(1 for t in current_arch_trades if t["is_win"])
        closed_all_time = len(all_time_trades)
        wins_all_time = sum(1 for t in all_time_trades if t["is_win"])
        self._win_rate_cache = {
            "win_rate_current_arch_pct": round(
                (wins_current_arch / closed_current_arch * 100) if closed_current_arch else 0.0, 1),
            "closed_current_arch": closed_current_arch,
            "win_rate_all_time_pct": round(
                (wins_all_time / closed_all_time * 100) if closed_all_time else 0.0, 1),
            "closed_all_time": closed_all_time,
            # Already most-recent-first: _closed_trades_since queries ORDER BY timestamp
            # DESC, and _group_closed_trades preserves first-seen order per trade_id --
            # since DESC means each trade's own most recent (final) tranche is the first
            # row encountered for that id, groups end up ordered by their latest activity.
            "trades": current_arch_trades,
        }

    async def _process_sell_analysis_queue(self) -> None:
        """"Recent Sell" post-mortem (2026-08-21, owner request: "an analisys of what
        the program could have done for a better sell if warented... saved for future
        analysis"). Drains at most one pending immediate post-mortem and one due delayed
        follow-up per call (bounded real Claude spend per tick, same "small bounded batch
        per pass" pattern as this project's other periodic queues, e.g. On Deck
        backfill). The raw facts were already captured synchronously by
        Portfolio.close_position_async at the moment of close (see that method) --
        this only ever generates the AI judgment on top of already-safe, already-
        persisted data, so a failure here just leaves a row pending for the next tick,
        never loses anything."""
        pending = await self.portfolio.get_pending_sell_analyses(limit=1)
        for row in pending:
            ticker = row["ticker"]
            company_name = self.research_reports.get(ticker, {}).get("company_name", ticker)
            async with self.portfolio._db.execute(
                "SELECT price, reason, pnl, timestamp FROM trade_history "
                "WHERE trade_id = ? AND action = 'SELL' ORDER BY timestamp ASC",
                (row["trade_id"],),
            ) as cur:
                tranche_rows = await cur.fetchall()
            tranches = [
                {"date": ts.split("T")[0] if ts else "unknown", "price": price,
                 "reason": reason or "Not recorded", "pnl": pnl}
                for price, reason, pnl, ts in tranche_rows
            ]
            opened_date = (row["opened_at"] or "").split("T")[0]
            closed_date = (row["closed_at"] or "").split("T")[0]
            price_history = []
            if opened_date:
                try:
                    hist = await self.market_data.get_historical(ticker, period="1y")
                    price_history = [
                        {"date": p["date"], "close": p["close"]}
                        for p in hist if opened_date <= p["date"] <= closed_date
                    ]
                except Exception as e:
                    logger.warning("sell-analysis price history fetch failed for %s: %s", ticker, e)
            days_ago = None
            if row["opened_at"]:
                try:
                    opened_dt = datetime.fromisoformat(row["opened_at"])
                    days_ago = (datetime.now() - opened_dt).total_seconds() / 86400
                except ValueError:
                    pass
            result = await self.research_engine.analyze_sell_decision(
                ticker=ticker, company_name=company_name,
                buy_thesis=row["buy_thesis"], buy_reasoning=row["buy_reasoning"],
                buy_conviction=row["buy_conviction"], buy_rr=row["buy_rr"],
                buy_required_rr=row["buy_required_rr"], entry_price=row["entry_price"],
                opened_at_days_ago=days_ago, tranches=tranches, price_history=price_history,
            )
            if result is None:
                continue  # leaves the row pending -- retried on a later tick
            followup_due = (self._now_et() + timedelta(
                days=self.config.get("research", {}).get("sell_analysis_followup_days", 5)
            )).strftime("%Y-%m-%d")
            exit_price = tranches[-1]["price"] if tranches else None
            await self.portfolio.save_sell_analysis_post_mortem(
                row["trade_id"], result["thesis"], result["reasoning"], followup_due,
                exit_price=exit_price)
            entry = self.add_ai_log(
                ticker, "SELL_ANALYSIS", f"Post-mortem: {result['thesis']}", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

        today_str = self._now_et().strftime("%Y-%m-%d")
        due = await self.portfolio.get_due_sell_analysis_followups(today_str, limit=1)
        for row in due:
            ticker = row["ticker"]
            company_name = self.research_reports.get(ticker, {}).get("company_name", ticker)
            closed_date = (row["closed_at"] or "").split("T")[0]
            price_history_since = []
            if closed_date:
                try:
                    hist = await self.market_data.get_historical(ticker, period="3mo")
                    price_history_since = [
                        {"date": p["date"], "close": p["close"]}
                        for p in hist if p["date"] > closed_date
                    ]
                except Exception as e:
                    logger.warning("sell-analysis followup price fetch failed for %s: %s", ticker, e)
            closed_days_ago = None
            if row["closed_at"]:
                try:
                    closed_dt = datetime.fromisoformat(row["closed_at"])
                    closed_days_ago = (datetime.now() - closed_dt).total_seconds() / 86400
                except ValueError:
                    pass
            followup = await self.research_engine.analyze_sell_followup(
                ticker=ticker, company_name=company_name,
                post_mortem_thesis=row["post_mortem_thesis"],
                post_mortem_reasoning=row["post_mortem_reasoning"],
                exit_price=row["exit_price"] if row["exit_price"] is not None else row["entry_price"],
                closed_at_days_ago=closed_days_ago,
                price_history_since=price_history_since,
            )
            if followup is None:
                continue  # retried on a later due check -- followup_due_date stays in the past
            await self.portfolio.save_sell_analysis_followup(row["trade_id"], followup)
            entry = self.add_ai_log(
                ticker, "SELL_ANALYSIS", f"Post-mortem follow-up: {followup}", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

    def _ytd_pnl(self) -> tuple[float | None, float | None]:
        """Year-to-date P&L. When the account's entire real history began within the current
        year (true here since go-live 2026-07-12), YTD is definitionally identical to Total
        P&L -- so this uses the same initial_capital origin Total P&L uses (`Portfolio.
        total_pnl`), not the first performance_history snapshot.

        Bug found live 2026-07-21 (user: "i think its off"): the daily-snapshot mechanism
        didn't exist at go-live, so the earliest entry in performance_history is dated
        2026-07-13 -- one day AFTER inception -- and by then the account had already made
        $133.12 of real gains. Using that entry as the YTD baseline silently understated
        true YTD gain by that first day's gain (2.91% Total vs 1.56% YTD at the time this
        was caught, a $133.12 / 1.35pp gap that should not have existed). Falls back to a
        real Jan-1 performance_history snapshot only once the account has survived past its
        first calendar-year boundary, since at that point initial_capital is no longer a
        correct YTD anchor (it would include prior years' gains too)."""
        year_start = f"{self._now_et().year}-01-01"
        if self.live_account_start >= year_start:
            base = self.portfolio.initial_capital
        else:
            candidates = [p for p in self.performance_history if p["date"] >= year_start]
            if not candidates:
                return None, None
            base = candidates[0]["portfolio_value"]
        if not base:
            return None, None
        current = self.portfolio.total_value
        pnl = current - base
        return round(pnl, 2), round(pnl / base * 100, 2)

    def _week_start_value(self, iso_year: int, iso_week: int) -> float:
        """Portfolio value baseline for one ISO calendar week: the last
        performance_history snapshot strictly before that week's Monday, or
        initial_capital if no earlier snapshot exists yet (the very first
        week real data covers). Shared by _weekly_pnl_buckets() (historical
        breakdown) and _week_pnl() (live tile) so both agree on week
        boundaries (2026-07-27)."""
        monday_str = date.fromisocalendar(iso_year, iso_week, 1).isoformat()
        candidates = [p for p in self.performance_history if p["date"] < monday_str]
        if not candidates:
            return self.portfolio.initial_capital
        return max(candidates, key=lambda p: p["date"])["portfolio_value"]

    def _weekly_pnl_buckets(self) -> list[dict]:
        """Groups performance_history into real ISO calendar weeks (Monday-Sunday,
        date.isocalendar()) for the Week P/L popup's historical breakdown
        (2026-07-27 redesign -- was a single trailing-7-day figure). Each bucket's
        P&L is settled-snapshot-to-settled-snapshot (end_value is the last
        snapshot captured so far, not a live quote) -- the live tile itself
        (_week_pnl below) is what stays intraday-accurate for the current week."""
        if not self.performance_history:
            return []
        groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for p in self.performance_history:
            d = date.fromisoformat(p["date"])
            iso_year, iso_week, _ = d.isocalendar()
            groups[(iso_year, iso_week)].append(p)

        today = self._now_et().date()
        current_key = today.isocalendar()[:2]
        # weekday() < 5 (Mon=0..Fri=4) -- fixed 2026-08-15, owner report: the ISO week
        # doesn't calendar-roll to the next one until Monday, so a plain iso-week-number
        # match kept showing "(in progress)" all through Saturday/Sunday even though the
        # market only trades Mon-Fri and nothing can change that week's P&L again until
        # the next trading week starts. "In progress" now means "this ISO week AND
        # today's still a weekday" -- turns off at midnight Friday night, not at the
        # actual ISO week boundary (midnight Sunday).
        is_current_week_and_weekday = today.weekday() < 5
        buckets = []
        for (iso_year, iso_week), entries in groups.items():
            entries_sorted = sorted(entries, key=lambda p: p["date"])
            start_value = self._week_start_value(iso_year, iso_week)
            end_value = entries_sorted[-1]["portfolio_value"]
            pnl = end_value - start_value
            buckets.append({
                "iso_year": iso_year,
                "iso_week": iso_week,
                "start_date": entries_sorted[0]["date"],
                "end_date": entries_sorted[-1]["date"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / start_value * 100, 2) if start_value else None,
                "is_current": (iso_year, iso_week) == current_key and is_current_week_and_weekday,
            })
        buckets.sort(key=lambda b: (b["iso_year"], b["iso_week"]), reverse=True)
        return buckets

    def _daily_pnl_buckets(self) -> list[dict]:
        """Backs the Day P/L popup's running per-day history list (2026-08-14, owner
        request: "a running history of every days P/L... backdate all the way to
        inception"). Same day-over-day-diff shape as _weekly_pnl_buckets() above, just
        against consecutive performance_history entries instead of ISO week boundaries
        -- each settled day's P&L is that day's own portfolio_value minus the PRIOR
        entry's (the very first entry in the file diffs against initial_capital, same
        "no earlier snapshot yet" fallback _week_start_value uses).

        Deliberately reads the full, unfiltered performance_history list, no recent-
        window truncation -- checked live (2026-08-14) whether Alpaca's own portfolio-
        history API could backfill further back than this file's own first entry
        (2026-07-13): it can, down to 2026-07-07, but everything before
        self.live_account_start (2026-07-12, the Sunday before that first Monday --
        now sourced from the self-initializing _get_or_init_account_genesis rather
        than a hardcoded literal, see that function's docstring) is the
        already-documented pre-migration dev/test window this codebase filters out
        everywhere else -- so this file's own first entry already IS genuine day one,
        nothing to backfill.

        Today itself is intentionally NOT included here -- it's still in progress and
        has no settled performance_history entry yet; the live figure for "today"
        comes from Portfolio.day_pnl/day_pnl_pct directly (the same numbers already
        driving the Day P/L tile), stitched in by the caller so today always shows
        instantly rather than only appearing after tonight's snapshot job runs."""
        if not self.performance_history:
            return []
        entries_sorted = sorted(self.performance_history, key=lambda p: p["date"])
        buckets = []
        prev_value = self.portfolio.initial_capital
        for entry in entries_sorted:
            value = entry["portfolio_value"]
            pnl = value - prev_value
            buckets.append({
                "date": entry["date"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / prev_value * 100, 2) if prev_value else None,
            })
            prev_value = value
        buckets.reverse()
        return buckets

    def _week_pnl(self) -> tuple[float | None, float | None]:
        """Live P&L for the CURRENT ISO calendar week -- same (pnl, pnl_pct) shape
        as before, but "week" now means a real Monday-Sunday ISO week (matching
        _weekly_pnl_buckets()'s historical breakdown) instead of a trailing 7
        calendar days (2026-07-27 redesign, see
        docs/superpowers/specs/2026-07-27-week-pnl-redesign-design.md). Diffs
        against the LIVE portfolio value (not the latest stored snapshot) so the
        tile stays intraday-accurate, unlike the settled end-of-day figures
        _weekly_pnl_buckets() returns for the popup."""
        today = self._now_et().date()
        iso_year, iso_week, _ = today.isocalendar()
        base = self._week_start_value(iso_year, iso_week)
        if not base:
            return None, None
        current = self.portfolio.total_value
        pnl = current - base
        return round(pnl, 2), round(pnl / base * 100, 2)

    def get_scan_status(self) -> dict:
        return {
            "current_ticker": self.current_ticker,
            "cycle": self.cycle_count,
            "index": self.scan_index,
            "total": self.watchlist_manager.size(),
            "next_cycle": self.next_cycle_at,
            "run_status": self.run_status(),
        }

    def _rank_position(self, ticker: str) -> float:
        """Score a held position — lower score = weaker holding, better swap candidate."""
        pos = self.portfolio.positions.get(ticker)
        if not pos:
            return float("inf")

        report = self.research_engine.reports.get(ticker)
        conviction = report.conviction_score if report else 5

        return conviction + (pos.unrealized_pnl_pct / 10)

    async def _try_rotation_swap(self, new_candidate: dict, rotated_out: set | None = None) -> tuple[bool, float]:
        """Sell weakest holding if new candidate scores higher. Returns (swapped, proceeds)."""
        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        today = self._now_et().date()
        already_sold = rotated_out or set()

        # Only consider positions not already rotated out this cycle and not bought today
        eligible = [
            t for t in self.portfolio.positions
            if t not in already_sold
            and (self.portfolio.positions[t].opened_at is None
                 or self.portfolio.positions[t].opened_at.date() != today)
        ]

        ranked = sorted(eligible, key=lambda t: self._rank_position(t))

        if not ranked:
            return False, 0.0

        weakest_ticker = ranked[0]
        weakest_score = self._rank_position(weakest_ticker)
        new_score = new_candidate["score"]

        weakest_pos = self.portfolio.positions[weakest_ticker]
        weakest_report = self.research_engine.reports.get(weakest_ticker)
        weakest_conviction = weakest_report.conviction_score if weakest_report else 5

        if new_score <= weakest_score:
            entry = self.add_ai_log(new_candidate["ticker"], "ROTATION",
                f"New candidate (score {new_score:.1f}) does not beat weakest holding "
                f"{weakest_ticker} (score {weakest_score:.1f}) — keeping current portfolio", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False, 0.0

        entry = self.add_ai_log(weakest_ticker, "ROTATION",
            f"SWAPPING {weakest_ticker} (conviction {weakest_conviction}, P&L {weakest_pos.unrealized_pnl_pct:+.1f}%) "
            f"→ {new_candidate['ticker']} (conviction {new_candidate['conviction']}, "
            f"margin of safety {new_candidate['margin']:.0f}%)", "sell")
        await self.broadcast({"type": "ai_log", "entry": entry})

        sell_signal = TradeSignal(
            ticker=weakest_ticker, signal=Sig.SELL, conviction=10,
            entry_price=weakest_pos.current_price, stop_loss=0,
            take_profit_targets=[], position_size_pct=0,
            position_size_dollars=weakest_pos.market_value,
            shares=weakest_pos.shares,
            reasoning=f"Portfolio rotation: replacing with stronger candidate {new_candidate['ticker']}",
            research_report=weakest_report, generated_at=datetime.now(),
            should_execute=True,
        )

        proceeds = weakest_pos.market_value
        try:
            order = await self.order_manager.execute(sell_signal)
            if order and order.status == OrderStatus.FILLED:
                # Log the real fill price, not the pre-trade pos.current_price snapshot
                # sell_signal was built from (2026-08-08, GitHub #53).
                if order.filled_price is not None:
                    sell_signal.entry_price = order.filled_price
                self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                result = {
                    "ticker": weakest_ticker, "status": order.status.value,
                    "filled_price": order.filled_price,
                    "shares": weakest_pos.shares,
                    "pnl": round(weakest_pos.unrealized_pnl, 2),
                }
                await self.broadcast({"type": "trade_executed", "trade": result})
                entry = self.add_ai_log(weakest_ticker, "ROTATION",
                    f"SOLD {weakest_pos.shares} shares — P&L: ${weakest_pos.unrealized_pnl:+.2f} "
                    f"— making room for {new_candidate['ticker']}", "sell")
                await self.broadcast({"type": "ai_log", "entry": entry})
                logger.info("Rotation: sold %s (score %.1f) to make room for %s (score %.1f)",
                            weakest_ticker, weakest_score, new_candidate["ticker"], new_score)
                if rotated_out is not None:
                    rotated_out.add(weakest_ticker)
                return True, 0.0  # portfolio.cash already updated by _execute_sell
            elif order and order.status == OrderStatus.PARTIAL:
                if order.filled_quantity is not None:
                    sell_signal.shares = order.filled_quantity
                if order.filled_price is not None:
                    sell_signal.entry_price = order.filled_price
                self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                # Broadcast trade_executed for PARTIAL fills too (2026-07-21) -- the frontend
                # no longer reads this message's payload at all (just uses it as a trigger to
                # re-fetch /api/trade-history), so the original "phantom full sale" risk that
                # kept PARTIAL fills off this broadcast no longer applies. Without this, a
                # partial fill was correctly logged to trade_history but invisible on the
                # Trades tab until the next full page reload.
                await self.broadcast({"type": "trade_executed", "trade": {
                    "ticker": weakest_ticker, "status": order.status.value,
                    "filled_price": order.filled_price, "shares": sell_signal.shares,
                }})
                await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                entry = self.add_ai_log(weakest_ticker, "ROTATION",
                    f"Rotation sell PARTIAL for {weakest_ticker} — position partially reduced, "
                    f"slot not freed for {new_candidate['ticker']}", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                return False, 0.0  # slot not freed — don't proceed with new buy
            elif order and order.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                # PENDING / SUBMITTED (after-hours) — proceeds not credited yet
                if rotated_out is not None:
                    rotated_out.add(weakest_ticker)
                return True, proceeds  # report uncredited proceeds to effective_cash
        except Exception as e:
            entry = self.add_ai_log(weakest_ticker, "ERROR", f"Rotation sell failed: {e}", "error")
            await self.broadcast({"type": "ai_log", "entry": entry})

        return False, 0.0

    async def _auto_execute_signal(self, report, ticker: str, is_buy: bool, is_sell: bool, held: bool):
        """Auto-sell during scan. Buys are deferred until after deep-dive confirmation."""
        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        if is_sell and held:
            pos = self.portfolio.positions.get(ticker)
            if not pos:
                return
            sell_signal = TradeSignal(
                ticker=ticker, signal=Sig.SELL, conviction=report.conviction_score,
                entry_price=pos.current_price, stop_loss=0,
                take_profit_targets=[], position_size_pct=0,
                position_size_dollars=pos.market_value,
                shares=pos.shares, reasoning=f"Auto-sell: signal dropped to {report.signal.value}",
                research_report=report, generated_at=datetime.now(),
                should_execute=True,
            )
            try:
                order = await self.order_manager.execute(sell_signal)
                if order and order.status == OrderStatus.FILLED:
                    # Log the real fill price, not the pre-trade pos.current_price snapshot
                    # sell_signal was built from (2026-08-08, GitHub #53).
                    if order.filled_price is not None:
                        sell_signal.entry_price = order.filled_price
                    self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                    result = {
                        "ticker": ticker,
                        "status": order.status.value,
                        "filled_price": order.filled_price,
                        "shares": pos.shares,
                        "pnl": round(pos.unrealized_pnl, 2),
                    }
                    await self.broadcast({"type": "trade_executed", "trade": result})
                    await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    entry = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"AUTO SELL {pos.shares} shares @ ${order.filled_price or pos.current_price:.2f} "
                        f"P&L: ${pos.unrealized_pnl:+.2f}", "sell")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    logger.info("Auto-executed SELL %s — %.4g shares, P&L $%.2f",
                                ticker, pos.shares, pos.unrealized_pnl)
                    _fp = order.filled_price or pos.current_price
                    asyncio.create_task(_notify(
                        f"SELL {ticker}",
                        f"{sell_signal.shares:.4g} shares @ ${_fp:.2f} | P&L ${pos.unrealized_pnl:+.2f}",
                        priority="high", tags="red_circle"))
                elif order and order.status == OrderStatus.PARTIAL:
                    if order.filled_quantity is not None:
                        sell_signal.shares = order.filled_quantity
                    if order.filled_price is not None:
                        sell_signal.entry_price = order.filled_price
                    self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                    # Broadcast trade_executed too (2026-07-21) -- see the matching comment
                    # in _try_rotation_swap for why this is now safe for PARTIAL fills.
                    await self.broadcast({"type": "trade_executed", "trade": {
                        "ticker": ticker, "status": order.status.value,
                        "filled_price": order.filled_price, "shares": sell_signal.shares,
                    }})
                    await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    entry = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"Auto-sell PARTIAL for {ticker} — position partially reduced", "warning")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                elif order:
                    entry = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"Auto-sell rejected by broker: {order.status.value}", "error")
                    await self.broadcast({"type": "ai_log", "entry": entry})
            except Exception as e:
                entry = self.add_ai_log(ticker, "AUTO_TRADE", f"Auto-sell failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})

    # ── Broker integration ─────────────────────────────────────────────

    async def connect_broker(self):
        try:
            await self.order_manager.connect()
            self.broker_connected = True
            broker = self.config["trading"]["broker"]
            mode = "PAPER" if self.config["trading"]["paper_trading"] else "LIVE"
            entry = self.add_ai_log("SYSTEM", "BROKER", f"Connected to {broker.upper()} ({mode})", "success")
            await self.broadcast({"type": "ai_log", "entry": entry})
            await self.broadcast({"type": "broker_status", "connected": True, "broker": broker, "mode": mode})
        except Exception as e:
            self.broker_connected = False
            logger.warning("Broker connection failed: %s — trading disabled", e)
            entry = self.add_ai_log("SYSTEM", "BROKER", f"Connection failed: {e}", "error")
            await self.broadcast({"type": "ai_log", "entry": entry})
            await self.broadcast({"type": "broker_status", "connected": False})

    async def position_update_loop(self):
        _current_day = self._now_et().date()
        while True:
            await asyncio.sleep(10)

            # Reset daily P&L at the start of each new trading day — regardless of broker state
            today = self._now_et().date()
            if today != _current_day:
                _current_day = today
                self.portfolio.new_trading_day(today.strftime("%Y-%m-%d"))
                await self.portfolio._save_state()
                # Persist each position's freshly-snapshotted day_open_price (2026-07-22) --
                # new_trading_day() only updates the in-memory values, matching its existing
                # synchronous signature; the actual DB write needs to happen here.
                for _pos in list(self.portfolio.positions.values()):
                    await self.portfolio._save_position(_pos)
                logger.info("New trading day %s — daily P&L reset", today)
                entry = self.add_ai_log("SYSTEM", "PORTFOLIO",
                    f"New trading day — daily P&L reset to $0", "neutral")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await self.broadcast({"type": "portfolio",
                                     "portfolio": self.get_portfolio_snapshot()})

            if not self.broker_connected:
                continue
            try:
                _prev_prices = {t: pos.current_price for t, pos in self.portfolio.positions.items()}
                _closed_at_broker = await self.order_manager.update_positions()
                for _c in _closed_at_broker:
                    await self._report_alpaca_detected_close(
                        _c["ticker"], _c["shares"], _c["fill_price"], _c["pnl"])
                for _c in self.order_manager.pop_stream_closed_reports():
                    await self._report_alpaca_detected_close(
                        _c["ticker"], _c["shares"], _c["fill_price"], _c["pnl"])

                # ── Protection-gap verification, every 10s (2026-07-21) ──
                # Previously only sync_exit_orders' hourly pass would ever notice a held
                # position missing real stop/TP orders at the broker — EPRT sat unprotected
                # for roughly a day because of exactly this gap. check_protection_gaps() is
                # read-only (no order calls), so this costs one extra get_open_orders() call
                # per 10s cycle.
                #
                # Alert visibility is delayed (2026-07-23) — see _protection_gap_first_seen's
                # docstring in __init__ — but remediation is NOT: sync_exit_orders() fires on
                # EVERY cycle a gap is open, independent of whether the alert has fired yet.
                # This also fixes a real bug found live the same night: the old code fired
                # sync_exit_orders() from INSIDE the 5-minute re-alert cooldown's guard, so a
                # persisting gap could go up to 5 minutes between retry attempts even though
                # nothing was stopping it from retrying sooner (confirmed: INSW needed 3
                # attempts over ~7 minutes, spaced by the alert cooldown, not by anything
                # about the gap itself).
                #
                # Gated on the pre-open-or-market-open window (2026-07-30, BEN incident) --
                # this whole block used to run unconditionally, 24/7, every 10s. During
                # genuinely dead overnight hours, a missing DAY order is completely
                # expected (the prior session's already expired), not a real gap, and
                # attempting remediation risks acting on a stale/degenerate overnight quote
                # with no genuine price move behind it -- see
                # _exit_order_maintenance_window_open's docstring for the exact incident.
                # Scoped to just this block (not a bare `continue`) -- the rest of this
                # tick (stop-loss/trailing-stop tracking, etc. below) must keep running
                # regardless of this window.
                if self._exit_order_maintenance_window_open():
                    _gaps = await self.order_manager.check_protection_gaps()
                    _gap_tickers = {g["ticker"] for g in _gaps}
                    _gap_alert_delay = self.config.get("risk_management", {}).get(
                        "protection_gap_alert_delay_seconds", 30)
                    for _t in list(self._protection_gap_first_seen):
                        if _t not in _gap_tickers:
                            del self._protection_gap_first_seen[_t]
                            # Only announce a resolution if the gap was actually announced in
                            # the first place — a gap that self-healed inside the alert delay
                            # was never shown, so "resolved" would be an orphaned message with
                            # no matching "detected" line above it.
                            was_alerted = self._protection_gap_alerted.pop(_t, None) is not None
                            if was_alerted:
                                _resolved_entry = self.add_ai_log(_t, "RISK",
                                    f"✅ Protection gap resolved for {_t} — real stop/TP order "
                                    "confirmed at the broker again", "success")
                                await self.broadcast({"type": "ai_log", "entry": _resolved_entry})
                    for _g in _gaps:
                        _ticker = _g["ticker"]
                        if _ticker not in self._protection_gap_first_seen:
                            self._protection_gap_first_seen[_ticker] = self._now_et()
                        _elapsed = (self._now_et() - self._protection_gap_first_seen[_ticker]).total_seconds()
                        if _elapsed >= _gap_alert_delay:
                            _last = self._protection_gap_alerted.get(_ticker)
                            if _last is None or (self._now_et() - _last).total_seconds() >= 300:
                                self._protection_gap_alerted[_ticker] = self._now_et()
                                _entry = self.add_ai_log(_ticker, "RISK",
                                    f"⚠️ PROTECTION GAP — {_g['reason']}", "error")
                                await self.broadcast({"type": "ai_log", "entry": _entry})
                                logger.error("Protection gap detected for %s: %s", _ticker, _g["reason"])
                                asyncio.create_task(_notify(
                                    f"⚠️ {_ticker} unprotected",
                                    _g["reason"],
                                    priority="urgent", tags="rotating_light"))
                    if _gaps:
                        # Attempt an immediate fix on every cycle regardless of alert-display
                        # gating above — most gaps (missing/mispriced order) are exactly what
                        # sync_exit_orders already knows how to correct. A data corruption gap
                        # (0 targets, stop_loss<=0) can't be auto-fixed by it — it'll log its
                        # own warning and skip.
                        #
                        # Fired ONCE per tick regardless of how many tickers have a gap
                        # (fixed 2026-07-28, DV incident) — sync_exit_orders() already walks
                        # every held position in one pass; the old code created one task PER
                        # gap ticker, so any tick with 2+ simultaneous gaps (routine right
                        # after a restart, when several trailing stops need a ratchet at once)
                        # spawned redundant concurrent calls. sync_exit_orders' own
                        # _sync_in_progress guard can't just drop a concurrent call outright
                        # (a real fix for the 2026-07-17 ECPG incident — see
                        # TestSyncExitOrdersRerun), so each redundant call instead forced an
                        # immediate extra pass ~1s later, re-touching tickers the first pass
                        # had already fixed. That extra pass raced Alpaca's own eventual-
                        # consistency window on the replace-in-place path (a stale order id
                        # can still list as "open" for a second or more after being
                        # superseded), producing a doomed re-replace, a failed cancel, and
                        # finally a spurious available:0 on a brand-new placement — the shares
                        # were never actually free, they were still legitimately held by the
                        # first pass's still-live stop. DV was never actually unprotected;
                        # this just made the system fight itself and log a false alarm.
                        asyncio.create_task(self.order_manager.sync_exit_orders())

                _direction_changed = False
                for t, pos in self.portfolio.positions.items():
                    _old = _prev_prices.get(t)
                    if _old is not None and pos.current_price != _old:
                        self.price_direction[t] = "up" if pos.current_price > _old else "down"
                        _direction_changed = True
                if _direction_changed:
                    asyncio.create_task(
                        asyncio.to_thread(_save_price_direction_cache, self.price_direction))

                for ticker, pos in list(self.portfolio.positions.items()):
                    market_open = self._is_market_open()

                    # ── Auto-close cooldown: suppress repeated close attempts for 5 minutes ──
                    # Without this, a trailing-stop or stop-loss condition that stays true
                    # across 30-second cycles submits a new market sell every 30 seconds
                    # (e.g. pre-market PENDING orders accumulate), flooding Alpaca's order
                    # engine and potentially corrupting the paper/live account state.
                    _cooldown_until = self._auto_close_cooldown.get(ticker)
                    _on_cooldown = (
                        _cooldown_until is not None
                        and datetime.now() < _cooldown_until
                    )

                    # ── Stop loss ──
                    _active_conditions = self._risk_condition_active.setdefault(ticker, set())
                    if pos.stop_loss > 0 and pos.current_price <= pos.stop_loss:
                        _was_active = "stop_loss" in _active_conditions
                        _active_conditions.add("stop_loss")
                        if not _was_active:
                            entry = self.add_ai_log(ticker, "RISK",
                                f"STOP LOSS triggered at ${pos.current_price:.2f}", "sell")
                            await self.broadcast({"type": "ai_log", "entry": entry})
                        if self.config["trading"].get("auto_execute", False) and market_open and not _on_cooldown:
                            self._auto_close_cooldown[ticker] = datetime.now() + timedelta(minutes=5)
                            await self._auto_close_position(ticker, pos, "Stop loss hit")
                        elif _on_cooldown:
                            logger.debug("Stop-loss auto-close suppressed for %s — cooldown active until %s",
                                         ticker, _cooldown_until)
                        if ticker not in self.portfolio.positions:
                            continue  # stop-loss already closed the position — skip trailing stop
                    elif "stop_loss" in _active_conditions and pos.stop_loss > 0 and pos.current_price > pos.stop_loss:
                        # Recovered back above the stop -- log it once, symmetric to the
                        # "triggered" message above, then clear so a future re-breach logs
                        # fresh immediately as its own new one-shot.
                        _active_conditions.discard("stop_loss")
                        entry = self.add_ai_log(ticker, "RISK",
                            f"✅ Price recovered above stop loss (${pos.stop_loss:.2f}) — "
                            f"now ${pos.current_price:.2f}", "success")
                        await self.broadcast({"type": "ai_log", "entry": entry})

                    # ── Trailing stop ──
                    if self.config["risk_management"].get("trailing_stop_enabled", True):
                        # Dollar profit-target latch (2026-07-24) -- checked before the
                        # graduated curve below, arms regardless of market hours (it's not
                        # a trade action, just a permanent tightening of the trail used
                        # from here on). See docs/superpowers/specs/
                        # 2026-07-24-dollar-profit-target-trailing-stop-design.md.
                        if (self.config["take_profit"].get("dollar_target_enabled", False)
                                and not pos.profit_target_hit
                                and pos.lifetime_pnl >= self.config["take_profit"].get(
                                    "dollar_target_amount", 30.0)):
                            pos.profit_target_hit = True
                            await self.portfolio._save_position(pos)
                            _target_amt = self.config["take_profit"].get("dollar_target_amount", 30.0)
                            _target_trail = self.config["take_profit"].get("dollar_target_trail_pct", 1.0)
                            entry = self.add_ai_log(ticker, "RISK",
                                f"🎯 Profit target hit (${_target_amt:.2f} lifetime P&L) — "
                                f"tightening trailing stop to {_target_trail:.1f}%", "success")
                            await self.broadcast({"type": "ai_log", "entry": entry})

                        # Single graduated curve from entry (start_pct = this stock's own
                        # derived stop-loss %) down to final_tranche_trail_pct at T3
                        # (2026-07-23, replaces the old 5%->4%->final-tranche-only-interp
                        # step function — see _graduated_trail_pct). trailing_stop_pct is
                        # cached on the position purely for display (the Positions table
                        # shows it in parens next to the stop figure) — it is NOT used to
                        # decide anything; the actual trailing_stop price below is always
                        # recomputed fresh from trail_pct_value directly.
                        start_pct = _derive_stop_pct(
                            pos.entry_price, pos.stop_loss,
                            self.config["take_profit"].get("stop_loss_pct", 5.0))
                        final_trail_pct = self.config["take_profit"].get(
                            "final_tranche_trail_pct", 0.5)
                        t3_price = pos.take_profit_targets[-1] if pos.take_profit_targets else None
                        graduated_pct = _graduated_trail_pct(
                            pos.entry_price, pos.current_price, start_pct, final_trail_pct,
                            pos.t1_target_price, pos.t2_target_price, t3_price,
                            self.config["risk_management"].get(
                                "trailing_stop_follow_tp_targets", False))
                        trail_pct_value = _profit_target_trail_pct(
                            pos.profit_target_hit,
                            self.config["take_profit"].get("dollar_target_trail_pct", 1.0),
                            graduated_pct)
                        trail_pct = 1 - (trail_pct_value / 100)
                        self._trailing_stop_pct_display[ticker] = round(trail_pct_value, 2)
                        if pos.trailing_stop is None:
                            # Arm the moment the position goes positive (2026-07-23,
                            # replaces the old >5%-gain gate) -- the standing hard stop_loss
                            # order is unaffected either way; this only changes when this
                            # second, ratcheting layer starts contributing.
                            if pos.current_price > pos.entry_price:
                                pos.trailing_stop = pos.current_price * trail_pct
                                await self.portfolio._save_position(pos)
                        else:
                            # Already trailing — ratchet up only, using the current graduated trail_pct
                            new_trailing = pos.current_price * trail_pct
                            if new_trailing > pos.trailing_stop:
                                pos.trailing_stop = new_trailing
                                await self.portfolio._save_position(pos)
                        if pos.trailing_stop is not None and pos.current_price <= pos.trailing_stop:
                            _was_active = "trailing_stop" in _active_conditions
                            _active_conditions.add("trailing_stop")
                            _pt_hit = pos.profit_target_hit
                            _label = "PROFIT-TARGET TRAILING STOP" if _pt_hit else "TRAILING STOP"
                            if not _was_active:
                                entry = self.add_ai_log(ticker, "RISK",
                                    f"{_label} triggered at ${pos.current_price:.2f} "
                                    f"(trail: ${pos.trailing_stop:.2f})", "sell")
                                await self.broadcast({"type": "ai_log", "entry": entry})
                            if self.config["trading"].get("auto_execute", False) and market_open and not _on_cooldown:
                                self._auto_close_cooldown[ticker] = datetime.now() + timedelta(minutes=5)
                                _close_reason = ("Profit-target trailing stop hit" if _pt_hit
                                                 else "Trailing stop hit")
                                await self._auto_close_position(ticker, pos, _close_reason)
                            elif _on_cooldown:
                                logger.debug("Trailing-stop auto-close suppressed for %s — cooldown active until %s",
                                             ticker, _cooldown_until)
                            if ticker not in self.portfolio.positions:
                                continue  # trailing stop closed the position — skip conviction-drop
                        elif ("trailing_stop" in _active_conditions and pos.trailing_stop is not None
                                and pos.current_price > pos.trailing_stop):
                            _active_conditions.discard("trailing_stop")
                            entry = self.add_ai_log(ticker, "RISK",
                                f"✅ Price recovered above trailing stop (${pos.trailing_stop:.2f}) — "
                                f"now ${pos.current_price:.2f}", "success")
                            await self.broadcast({"type": "ai_log", "entry": entry})

                        # ── Event-triggered re-analysis for profitable positions (2026-07-27) ──
                        # These are intentionally skipped by position_monitor_loop's periodic
                        # cycle (see _is_position_profitable_enough_to_skip) -- this is their
                        # only re-analysis path, firing only when price actually nears a real
                        # decision point rather than on a fixed schedule. Explicitly excludes
                        # the "already breached" cases above (those already trigger their own
                        # RISK log + auto-close directly) -- only the "approaching but not
                        # there yet" zone fires this.
                        # market_open gate (fixed 2026-07-27, same day, live incident -- this
                        # fired real Claude calls after hours on every restart, since the
                        # periodic path already checks market hours but this one didn't): no
                        # real decision gets made after close (auto-close itself is already
                        # gated on market_open above), so there's nothing for a re-analysis to
                        # usefully inform until the next session anyway.
                        if (ticker in self.portfolio.positions and market_open
                                and self._is_position_profitable_enough_to_skip(pos)):
                            proximity = self.config["research"].get(
                                "position_monitor_event_proximity_pct", 2.0) / 100
                            nearing_stop = (
                                pos.trailing_stop is not None
                                and pos.current_price > pos.trailing_stop
                                and pos.current_price <= pos.trailing_stop * (1 + proximity)
                            )
                            next_target = pos.take_profit_targets[0] if pos.take_profit_targets else None
                            nearing_target = (
                                next_target is not None
                                and pos.current_price < next_target
                                and pos.current_price >= next_target * (1 - proximity)
                            )
                            if nearing_stop or nearing_target:
                                _cd = self._event_monitor_cooldown.get(ticker)
                                cooldown_min = self.config["research"].get(
                                    "position_monitor_event_cooldown_minutes", 60)
                                if _cd is None or datetime.now() >= _cd:
                                    self._event_monitor_cooldown[ticker] = (
                                        datetime.now() + timedelta(minutes=cooldown_min))
                                    asyncio.create_task(asyncio.to_thread(
                                        _save_event_monitor_cooldown,
                                        dict(self._event_monitor_cooldown), dict(self._loss_event_worst_pct)))
                                    _reason = "nearing trailing stop" if nearing_stop else "nearing next target"
                                    asyncio.create_task(
                                        self._reanalyze_held_position(ticker, trigger_label=f"Event ({_reason})"))

                    # ── Event-triggered re-analysis for positions crossing a loss threshold
                    # (2026-07-29) ── Complements position_monitor_loop's periodic sweep
                    # (which already re-checks every underwater/flat position, just on a
                    # fixed up-to-120-minute cadence) with an immediate check the moment a
                    # position crosses meaningfully further underwater -- per user request
                    # (watching IVZ sit underwater with no fresh re-check), waiting up to two
                    # hours to find out whether a losing position is still worth holding is
                    # too slow. Deliberately independent of trailing_stop_enabled -- unlike
                    # the profitable-side event trigger above, this only depends on P/L, not
                    # on a trailing-stop price existing at all. Shares the same per-ticker
                    # cooldown dict/setting as that profitable-side trigger (a position is
                    # never in both the profitable-skip and underwater buckets at once, so
                    # there's no risk of the two competing over the same cooldown entry).
                    # cooldown_active is the SOLE primary gate (see
                    # _loss_retrigger_should_fire) -- a deepening loss (a full
                    # retrigger_step_pct past whatever level last fired) is the only thing
                    # that can bypass an already-active cooldown, never a substitute for
                    # it. Deliberately does NOT clear _loss_event_worst_pct on recovery
                    # above threshold (fixed same day, IVZ incident) -- an earlier version
                    # did, which combined with a since-removed "never fired before ->
                    # always fire" special case in the pure function caused IVZ's P&L
                    # oscillating right around -3.0% for hours to re-fire a real Claude
                    # call every time it dipped back below the threshold, bypassing the
                    # cooldown every single time (17 real calls in one day, nearly all
                    # wasteful repeats). Leaving the stale value in place is harmless --
                    # it only means a future dip needs to be either past cooldown or
                    # deeper than history's worst to bypass it, exactly the intended
                    # behavior.
                    if (ticker in self.portfolio.positions and market_open
                            and not self._is_position_profitable_enough_to_skip(pos)):
                        loss_trigger_pct = self.config["research"].get(
                            "position_monitor_loss_trigger_pct", 3.0)
                        retrigger_pct = self.config["research"].get(
                            "position_monitor_loss_retrigger_pct", 1.0)
                        _cd = self._event_monitor_cooldown.get(ticker)
                        cooldown_min = self.config["research"].get(
                            "position_monitor_event_cooldown_minutes", 60)
                        cooldown_active = _cd is not None and datetime.now() < _cd
                        if _loss_retrigger_should_fire(
                                pos.unrealized_pnl_pct, loss_trigger_pct, retrigger_pct,
                                self._loss_event_worst_pct.get(ticker), cooldown_active):
                            self._event_monitor_cooldown[ticker] = (
                                datetime.now() + timedelta(minutes=cooldown_min))
                            self._loss_event_worst_pct[ticker] = pos.unrealized_pnl_pct
                            asyncio.create_task(asyncio.to_thread(
                                _save_event_monitor_cooldown,
                                dict(self._event_monitor_cooldown), dict(self._loss_event_worst_pct)))
                            asyncio.create_task(
                                self._reanalyze_held_position(
                                    ticker,
                                    trigger_label=f"Event (down {pos.unrealized_pnl_pct:.1f}%)"))

                    # ── Conviction-drop auto-sell ──
                    # Gated on the same profitable/skip bucket (2026-07-27) as
                    # position_monitor_loop -- a comfortably profitable position's cached
                    # report is intentionally NOT kept fresh by the periodic cycle anymore
                    # (see position_monitor_loop's docstring), so its conviction_score here
                    # could be stale by hours or days; letting a stale number drive a real
                    # sell on a winner would defeat the whole point of skipping it above.
                    if (self.config["trading"].get("auto_execute", False)
                            and ticker in self.portfolio.positions
                            and not self._is_position_profitable_enough_to_skip(pos)):
                        report = self.research_engine.reports.get(ticker)
                        if report and report.conviction_score <= 4:
                            entry = self.add_ai_log(ticker, "RISK",
                                f"Conviction dropped to {report.conviction_score}/10 — auto-selling", "sell")
                            await self.broadcast({"type": "ai_log", "entry": entry})
                            await self._auto_close_position(ticker, pos,
                                f"Conviction dropped to {report.conviction_score}/10")

                await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
            except Exception as e:
                logger.warning("Position update failed: %s", e)

    def _is_position_profitable_enough_to_skip(self, pos) -> bool:
        """True when a held position is comfortably profitable enough that the periodic
        position-monitor cycle should skip it (2026-07-27) -- see position_monitor_loop's
        docstring for the full reasoning. Shared with the conviction-drop auto-sell check
        in position_update_loop so that check never acts on a report this system has
        deliberately stopped keeping fresh for a winning position."""
        threshold = self.config["research"].get("position_monitor_profitable_skip_pct", 2.0)
        return pos.unrealized_pnl_pct >= threshold

    async def _reanalyze_held_position(self, ticker: str, trigger_label: str = "Periodic"):
        """One held position's re-analysis + logging + auto-execute -- the actual work
        behind both position_monitor_loop's periodic cycle (underwater/flat positions) and
        the event-triggered check in position_update_loop (profitable positions nearing
        their trailing stop or next target). Extracted (2026-07-27) so both callers share
        one implementation rather than drifting apart. trigger_label only affects the
        ai_log wording so the AI Research Engine feed makes clear which path fired it."""
        if ticker not in self.portfolio.positions:
            return
        try:
            entry = self.add_ai_log(ticker, "MONITOR", f"{trigger_label} re-analysis starting...")
            await self.broadcast({"type": "ai_log", "entry": entry})

            # model_position_monitor (2026-07-24) -- split off from the shared
            # model_quick_scan default so the sell-decision engine on holdings can
            # be tuned independently of the live buy pipeline.
            report = await self.research_engine.analyze_stock(
                ticker, model=self.config["research"].get("model_position_monitor", "claude-haiku-4-5"))

            pos = self.portfolio.positions.get(ticker)
            if not pos:
                return

            if getattr(report, "is_fallback", False):
                entry = self.add_ai_log(ticker, "MONITOR",
                    f"⚠ Re-analysis failed — AI unavailable. Holding position, manual review recommended.", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                return

            # Persist the fresh report (2026-07-20 fix) — this re-analysis used to only
            # log to ai_log and update the lightweight ticker_signals badge (signal/
            # conviction/price, no thesis/fair-value/sector), leaving state.research_reports
            # frozen at whatever was cached at original buy time no matter how many
            # re-analyses ran since. Found via the portfolio health assessment feature
            # flagging stale fair_value_estimate data on long-held positions.
            self._persist_report(report)
            asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))

            # AI-chosen stop-loss/TP (2026-07-31) -- see docs/superpowers/specs/
            # 2026-07-31-ai-chosen-stop-loss-tp-design.md. Tighten-only for stop_loss
            # (never widen an already-held position's risk); take_profit_targets update
            # freely since they don't carry the same downside-risk concern. No new
            # order-placement code here -- a changed pos.stop_loss/take_profit_targets
            # is picked up by the existing sync_exit_orders/check_protection_gaps cycle
            # on its own next pass, the same mechanism already driving the graduated
            # trailing-stop ratchet.
            stop_tightened_note = ""
            research_cfg = self.config.get("research", {})
            if research_cfg.get("ai_chosen_stop_tp_enabled", False):
                clamped_stop = _clamp_ai_stop_loss(
                    report.stop_loss, pos.entry_price,
                    research_cfg.get("ai_stop_loss_min_pct", 2.0),
                    research_cfg.get("ai_stop_loss_max_pct", 10.0),
                )
                if _stop_loss_tightened(clamped_stop, pos.stop_loss):
                    old_stop = pos.stop_loss
                    pos.stop_loss = clamped_stop
                    stop_tightened_note = f" | Stop tightened ${old_stop:.2f}→${clamped_stop:.2f}"
                if report.take_profit_targets:
                    pos.take_profit_targets = _reconcile_ai_take_profit_targets(
                        pos.take_profit_targets, report.take_profit_targets)
                await self.portfolio._save_position(pos)

            level = "buy" if "BUY" in report.signal.value else "sell" if "SELL" in report.signal.value else "neutral"
            entry = self.add_ai_log(ticker, "MONITOR",
                f"Updated: {report.signal.value} | Conviction {report.conviction_score}/10 | "
                f"P&L {pos.unrealized_pnl_pct:+.1f}%{stop_tightened_note}", level)
            await self.broadcast({"type": "ai_log", "entry": entry})

            badge = {
                "ticker": ticker, "signal": report.signal.value,
                "conviction": report.conviction_score,
                "price": round(report.entry_price, 2),
                "time": self._now_et().strftime("%H:%M"),
            }
            self.ticker_signals[ticker] = badge
            await self.broadcast({"type": "ticker_signal", "badge": badge})

            if self.config["trading"].get("auto_execute", False) and self.broker_connected:
                held = ticker in self.portfolio.positions
                is_sell = "SELL" in report.signal.value
                await self._auto_execute_signal(report, ticker, False, is_sell, held)

        except Exception as e:
            entry = self.add_ai_log(ticker, "ERROR", f"Monitor re-analysis failed: {e}", "error")
            await self.broadcast({"type": "ai_log", "entry": entry})

    async def position_monitor_loop(self):
        """Periodic re-analysis of held positions that are underwater or flat (2026-07-27,
        replaces the old blanket-hourly design that re-ran this on EVERY position
        regardless of P/L). Comfortably profitable positions (see
        _is_position_profitable_enough_to_skip) are intentionally skipped here — their
        downside is already mechanically protected by the ratcheting trailing stop
        (position_update_loop, independent of any Claude call), so a periodic thesis
        re-check on a winner adds real false-positive-sell risk (the conviction-drop
        auto-sell trigger doesn't otherwise care whether a position is up or down) with no
        offsetting benefit. Those get event-triggered checks instead, fired from
        position_update_loop when price actually nears the trailing stop or the next
        target — see _reanalyze_held_position, shared by both paths."""
        while True:
            interval = self.position_monitor_interval * 60
            await asyncio.sleep(interval)
            if self.paused or self.stopped or not self._is_market_open():
                continue
            if not self.portfolio.positions:
                continue

            held_tickers = list(self.portfolio.positions.keys())
            to_check = [
                t for t in held_tickers
                if (pos := self.portfolio.positions.get(t)) is not None
                and not self._is_position_profitable_enough_to_skip(pos)
            ]
            skipped = len(held_tickers) - len(to_check)
            if not to_check:
                continue
            entry = self.add_ai_log("SYSTEM", "MONITOR",
                f"Position monitor: re-analyzing {len(to_check)} underwater/flat position(s)"
                f"{f' ({skipped} profitable position(s) skipped — event-only)' if skipped else ''}",
                "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

            for ticker in to_check:
                if ticker not in self.portfolio.positions:
                    continue
                await self._reanalyze_held_position(ticker, trigger_label="Periodic")
                await asyncio.sleep(self.stock_delay)

            entry = self.add_ai_log("SYSTEM", "MONITOR",
                f"Position monitor complete — {len(to_check)} position(s) updated", "success")
            await self.broadcast({"type": "ai_log", "entry": entry})

            # Renew any DAY exit orders that expired at market close
            if self.order_manager:
                await self.order_manager.sync_exit_orders()

            # Open slot-filling no longer needs an hourly nudge here (2026-07-17) —
            # near_miss_monitor_loop checks every candidate every 60s continuously, so a slot
            # missed by a blocked scan (drawdown halt, restart, etc.) gets picked up on the
            # very next tick rather than waiting for this hourly cycle.

    async def position_deep_dive_loop(self):
        """Periodic Deep Dive (Sonnet, richer valuation/moat/growth/catalysts prompt) on every
        held position (2026-07-20) — display/context only, does NOT feed any automatic sell
        decision. That's a deliberate scope limit, discussed with the user: using deep-dive
        output to actually DRIVE a sell is a separate, higher-stakes design question (which
        specific signal, how it coexists with the existing hourly quick-scan HOLD/SELL
        auto-close logic so the two don't fight or double-trigger) that needs its own careful
        design pass, not a same-day addition. This loop only populates state.deep_dive_reports
        so the richer analysis is visible in the position detail view (see
        showPositionDetail's deepDiveReports-first lookup) — the same "auto deep dive, purely
        informational" precedent already established for On Deck entries earlier today.

        Deliberately a SEPARATE loop from position_monitor_loop (the existing, critical hourly
        quick-scan + stop-loss-adjacent logic) rather than piggybacking on it — isolates any
        risk in this new, lower-priority addition from the loop that actually matters for
        capital safety. Fires every `position_deep_dive_interval_hours` (default 3, ~2x during
        a 6.5h trading day) rather than a fixed schedule, matching the user's own framing ("a
        couple times a day or more")."""
        while True:
            interval_hours = self.config["research"].get("position_deep_dive_interval_hours", 3)
            await asyncio.sleep(max(1, interval_hours) * 3600)
            if self.paused or self.stopped or not self._is_market_open():
                continue
            if not self.config["research"].get("position_deep_dive_enabled", True):
                continue
            held_tickers = list(self.portfolio.positions.keys())
            if not held_tickers:
                continue

            entry = self.add_ai_log("SYSTEM", "DEEP DIVE",
                f"Periodic deep dive: re-analyzing {len(held_tickers)} held position(s)", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

            for ticker in held_tickers:
                if ticker not in self.portfolio.positions:
                    continue
                try:
                    # model_periodic_deep_dive (2026-07-24) -- split off from the shared
                    # model_deep_dive default (which stays for manual Deep Dive clicks)
                    # specifically because this recurring every-few-hours re-check on
                    # every holding was the actual cost driver the user flagged, not the
                    # occasional manual click.
                    report = await self.research_engine.deep_dive_analysis(
                        ticker, model=self.config["research"].get("model_periodic_deep_dive", "claude-haiku-4-5"))
                    if report is None or getattr(report, "is_fallback", False):
                        continue
                    self.deep_dive_reports[ticker] = _deep_dive_report_dict(report)
                    asyncio.create_task(asyncio.to_thread(_save_dd_cache, self.deep_dive_reports))
                    entry = self.add_ai_log(ticker, "DEEP DIVE",
                        f"Periodic deep dive — fair value ${report.fair_value_estimate:.2f}, "
                        f"margin {report.margin_of_safety_pct:.0f}%", "neutral")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    await self.broadcast(
                        {"type": "deep_dive_report", "ticker": ticker, "report": self.deep_dive_reports[ticker]})
                except Exception as e:
                    logger.warning("Periodic position deep-dive failed for %s: %s", ticker, e)
                await asyncio.sleep(self.stock_delay)

    async def _report_alpaca_detected_close(self, ticker: str, shares: float, fill_price: float, pnl: float):
        """Surface a close that Alpaca's own standing stop/TP order executed directly —
        as opposed to a close _auto_close_position itself decided to submit. OrderManager
        already made local state (trade_history SQL table, cash, positions) correct by the
        time this fires; this adds every OTHER user-visible record that path has no way to
        produce itself (no reference to DashboardState, by design): the ai_log entry, the
        dashboard broadcast, the push notification, AND — separate from the SQL table —
        the JSONL trade log (data/trade_history/*.jsonl via trade_logger.log_trade) that
        actually backs the dashboard's Trades tab and the Daily Recap. Mirrors
        _auto_close_position's FILLED branch exactly, including that second log call."""
        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        result = {
            "ticker": ticker, "status": "filled", "filled_price": fill_price,
            "shares": shares, "pnl": round(pnl, 2),
        }
        await self.broadcast({"type": "trade_executed", "trade": result})
        sell_signal = TradeSignal(
            ticker=ticker, signal=Sig.SELL, conviction=10,
            entry_price=fill_price, stop_loss=0,
            take_profit_targets=[], position_size_pct=0,
            position_size_dollars=shares * fill_price,
            shares=shares, reasoning="Stop/TP filled at broker",
            research_report=None, generated_at=datetime.now(),
            should_execute=True,
        )
        self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
        entry = self.add_ai_log(ticker, "AUTO_TRADE",
            f"Position closed at broker — stop/TP filled — {shares:.4g} shares @ "
            f"${fill_price:.2f} — P&L: ${pnl:+.2f}", "sell")
        await self.broadcast({"type": "ai_log", "entry": entry})
        logger.info("Alpaca-detected close reported for %s — P&L $%.2f", ticker, pnl)
        asyncio.create_task(_notify(
            f"CLOSED {ticker} — stop/TP filled at broker",
            f"{shares:.4g} shares @ ${fill_price:.2f} | P&L ${pnl:+.2f}",
            priority="urgent", tags="warning"))

    async def _auto_close_position(self, ticker: str, pos, reason: str):
        """Close entire position automatically."""
        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        if ticker not in self.portfolio.positions:
            return

        sell_signal = TradeSignal(
            ticker=ticker, signal=Sig.SELL, conviction=10,
            entry_price=pos.current_price, stop_loss=0,
            take_profit_targets=[], position_size_pct=0,
            position_size_dollars=pos.market_value,
            shares=pos.shares, reasoning=reason,
            research_report=None, generated_at=datetime.now(),
            should_execute=True,
        )

        try:
            order = await self.order_manager.execute(sell_signal)
            if order and order.status == OrderStatus.FILLED:
                # Order confirmed filled — clear the cooldown so future genuine signals work
                self._auto_close_cooldown.pop(ticker, None)
                # Log the real fill price, not the pre-trade pos.current_price snapshot
                # sell_signal was built from (2026-08-08, GitHub #53).
                if order.filled_price is not None:
                    sell_signal.entry_price = order.filled_price
                self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                result = {
                    "ticker": ticker, "status": order.status.value,
                    "filled_price": order.filled_price,
                    "shares": pos.shares, "pnl": round(pos.unrealized_pnl, 2),
                }
                await self.broadcast({"type": "trade_executed", "trade": result})
                entry = self.add_ai_log(ticker, "AUTO_TRADE",
                    f"AUTO SELL {pos.shares} shares — {reason} — "
                    f"P&L: ${pos.unrealized_pnl:+.2f}", "sell")
                await self.broadcast({"type": "ai_log", "entry": entry})
                logger.info("Auto-closed %s — %s — P&L $%.2f", ticker, reason, pos.unrealized_pnl)
                _fp = order.filled_price or pos.current_price
                asyncio.create_task(_notify(
                    f"CLOSED {ticker} — {reason}",
                    f"{pos.shares:.4g} shares @ ${_fp:.2f} | P&L ${pos.unrealized_pnl:+.2f}",
                    priority="urgent", tags="warning"))
            elif order and order.status == OrderStatus.PARTIAL:
                if order.filled_quantity is not None:
                    sell_signal.shares = order.filled_quantity
                if order.filled_price is not None:
                    sell_signal.entry_price = order.filled_price
                self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                # Broadcast trade_executed too (2026-07-21) -- see the matching comment
                # in _try_rotation_swap for why this is now safe for PARTIAL fills.
                await self.broadcast({"type": "trade_executed", "trade": {
                    "ticker": ticker, "status": order.status.value,
                    "filled_price": order.filled_price, "shares": sell_signal.shares,
                }})
                await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                entry = self.add_ai_log(ticker, "AUTO_TRADE",
                    f"Auto-close PARTIAL for {ticker} — {reason} — position partially reduced", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
            elif order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                # Order submitted but not yet filled — sync_exit_orders restores protection
                # while we wait; update_positions will detect the fill and close locally.
                entry = self.add_ai_log(ticker, "AUTO_TRADE",
                    f"Auto-close order pending fill — {reason} — monitoring for fill", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
            elif order:
                entry = self.add_ai_log(ticker, "ERROR",
                    f"Auto-close rejected by broker: {order.status.value} — {reason}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
        except Exception as e:
            entry = self.add_ai_log(ticker, "ERROR", f"Auto-close failed: {e}", "error")
            await self.broadcast({"type": "ai_log", "entry": entry})

    async def handle_trade_command(self, data: dict, websocket: WebSocket):
        import uuid as _uuid
        cmd = data.get("command")

        if cmd == "execute_buy":
            ticker = data.get("ticker", "").upper().strip()
            valid_tickers = self.watchlist_manager.get_active_tickers()
            if ticker not in valid_tickers:
                await websocket.send_json({"type": "trade_error", "error": f"Invalid ticker: {ticker}"})
                return
            if not self.broker_connected:
                await websocket.send_json({"type": "trade_error", "error": "Broker not connected"})
                return
            report = self.research_engine.reports.get(ticker)
            if not report:
                await websocket.send_json({"type": "trade_error", "error": f"No report for {ticker}"})
                return

            signal = self.signal_generator._evaluate_report(report)
            if not signal:
                await websocket.send_json({"type": "trade_error", "error": f"Signal rejected by risk checks for {ticker}"})
                return

            risk_ok = self.risk_manager.check_all_rules(report, self.portfolio)
            risk_msg = "" if risk_ok else "One or more risk rules failed"
            earnings_soon, earnings_date = await self._earnings_soon(ticker)
            conf_id = str(_uuid.uuid4())
            preview = {
                "confirmation_id": conf_id,
                "ticker": ticker,
                "shares": signal.shares,
                "entry_price": round(signal.entry_price, 2),
                "stop_loss": round(signal.stop_loss, 2),
                "estimated_cost": round(signal.position_size_dollars, 2),
                "position_pct": round(signal.position_size_pct, 1),
                "risk_check_passed": risk_ok,
                "risk_message": risk_msg,
                "earnings_warning": f"Earnings on {earnings_date} — gap risk is high" if earnings_soon else "",
            }
            self.pending_confirmations[conf_id] = {
                "signal": signal,
                "created_at": datetime.now(),
            }
            await websocket.send_json({"type": "trade_preview", "preview": preview})

        elif cmd == "confirm_buy":
            conf_id = data.get("confirmation_id", "")
            # Evict all expired entries while we're here
            now_ts = datetime.now()
            expired = [k for k, v in self.pending_confirmations.items()
                       if (now_ts - v["created_at"]).total_seconds() > 120]
            for k in expired:
                self.pending_confirmations.pop(k, None)

            pending = self.pending_confirmations.pop(conf_id, None)
            if not pending:
                await websocket.send_json({"type": "trade_error", "error": "Confirmation expired or invalid"})
                return
            elapsed = (now_ts - pending["created_at"]).total_seconds()
            if elapsed > 60:
                await websocket.send_json({"type": "trade_error", "error": "Confirmation expired (60s)"})
                return

            signal = pending["signal"]
            # Reject a second concurrent confirm for the same ticker outright (GitHub issue
            # #26) rather than letting both reach check_all_rules/execute() independently —
            # two browser tabs confirming the same ticker at once could otherwise both pass
            # the cash-reserve check before either actually deducted cash.
            if signal.ticker in self._confirm_in_progress:
                await websocket.send_json({"type": "trade_error",
                    "error": f"Another confirmation for {signal.ticker} is already in progress"})
                return
            self._confirm_in_progress.add(signal.ticker)
            try:
                # Re-validate: reject if ticker already held or risk rules now fail
                if signal.ticker in self.portfolio.positions:
                    await websocket.send_json({"type": "trade_error",
                        "error": f"{signal.ticker} already in portfolio (auto-buy may have fired)"})
                    return
                if not self.risk_manager.check_all_rules(signal.research_report, self.portfolio):
                    await websocket.send_json({"type": "trade_error",
                        "error": "Risk rules no longer pass (market conditions changed since preview)"})
                    return
                try:
                    order = await self.order_manager.execute(signal)
                    if order and order.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                        # Log the real fill price, not the pre-trade recommendation
                        # (2026-08-08, GitHub #53).
                        if order.filled_price is not None:
                            signal.entry_price = order.filled_price
                        self.trade_logger.log_trade(signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                        entry = self.add_ai_log(signal.ticker, "TRADE",
                            f"BUY {signal.shares} shares @ ${order.filled_price or signal.entry_price:.2f}", "buy")
                        await self.broadcast({"type": "ai_log", "entry": entry})
                        result = {
                            "ticker": signal.ticker,
                            "status": order.status.value,
                            "filled_price": order.filled_price,
                            "shares": signal.shares,
                        }
                        await self.broadcast({"type": "trade_executed", "trade": result})
                        await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    else:
                        status = order.status.value if order else "FAILED"
                        await websocket.send_json({"type": "trade_error",
                                                   "error": f"Buy order failed: {status}"})
                except Exception as e:
                    await websocket.send_json({"type": "trade_error", "error": str(e)})
            finally:
                self._confirm_in_progress.discard(signal.ticker)

        elif cmd == "execute_sell":
            ticker = data.get("ticker", "")
            if not self.broker_connected:
                await websocket.send_json({"type": "trade_error", "error": "Broker not connected"})
                return
            pos = self.portfolio.positions.get(ticker)
            if not pos:
                await websocket.send_json({"type": "trade_error", "error": f"No position in {ticker}"})
                return

            from src.decision.signal_generator import TradeSignal
            from src.research.engine import Signal as Sig
            sell_signal = TradeSignal(
                ticker=ticker, signal=Sig.SELL, conviction=10,
                entry_price=pos.current_price, stop_loss=0,
                take_profit_targets=[], position_size_pct=0,
                position_size_dollars=pos.market_value,
                shares=pos.shares, reasoning="Manual sell",
                research_report=None, generated_at=datetime.now(),
                should_execute=True,
            )
            try:
                order = await self.order_manager.execute(sell_signal)
                result = {
                    "ticker": ticker,
                    "status": order.status.value if order else "FAILED",
                    "filled_price": order.filled_price if order else None,
                    "shares": pos.shares,
                    "pnl": round(pos.unrealized_pnl, 2),
                }
                if order and order.status == OrderStatus.FILLED:
                    # Log the real fill price, not the pre-trade pos.current_price snapshot
                    # sell_signal was built from (2026-08-08, GitHub #53).
                    if order.filled_price is not None:
                        sell_signal.entry_price = order.filled_price
                    self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                    await self.broadcast({"type": "trade_executed", "trade": result})
                    await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    entry = self.add_ai_log(ticker, "TRADE",
                        f"SELL {pos.shares} shares @ ${result['filled_price'] or pos.current_price:.2f} "
                        f"P&L: ${pos.unrealized_pnl:+.2f}", "sell")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                elif order and order.status == OrderStatus.PARTIAL:
                    if order.filled_quantity is not None:
                        sell_signal.shares = order.filled_quantity
                    if order.filled_price is not None:
                        sell_signal.entry_price = order.filled_price
                    self.trade_logger.log_trade(sell_signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                    # Broadcast trade_executed too (2026-07-21) -- see the matching comment
                    # in _try_rotation_swap for why this is now safe for PARTIAL fills.
                    await self.broadcast({"type": "trade_executed", "trade": {
                        "ticker": ticker, "status": order.status.value,
                        "filled_price": order.filled_price, "shares": sell_signal.shares,
                    }})
                    await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    await websocket.send_json({"type": "trade_error",
                                               "error": "Partial fill — not all shares sold. Position updated."})
                elif order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                    # Not a failure — order is genuinely still in flight (e.g. after-hours,
                    # or momentarily queued during a busy market). update_positions() will
                    # detect the fill and close the position normally once it completes.
                    await websocket.send_json({"type": "trade_error",
                        "error": f"Sell order submitted, waiting for fill (status: {result['status']})"})
                else:
                    await websocket.send_json({"type": "trade_error",
                                               "error": f"Sell failed: {result['status']}"})
            except Exception as e:
                await websocket.send_json({"type": "trade_error", "error": str(e)})

        elif cmd == "cancel_order":
            order_id = data.get("order_id", "")
            success = await self.order_manager.cancel(order_id)
            await websocket.send_json({"type": "order_cancelled", "order_id": order_id, "success": success})

    # ── Market hours helpers ─────────────────────────────────────────────

    def _now_et(self) -> datetime:
        return datetime.now(self.market_tz)

    def _drawdown_diagnostic(self) -> str:
        """Total value / peak value / computed drawdown %, for appending to any halt log
        message. Added 2026-07-18 after the MSCI incident (2026-07-16) where a halt fired
        four times with no way to tell after the fact whether it reflected genuine Alpaca
        equity movement or a transient bad total_value read — see CLAUDE.md.

        Appends a recovery hint once the gap is large enough to plausibly be an external
        balance change rather than genuine trading losses (2026-08-08, GitHub #46) —
        peak_value only ever ratchets upward and nothing resets it automatically, so a
        manual deposit/withdrawal/balance reset reads as a catastrophic loss and
        permanently blocks buying with no self-service way to recover short of a raw DB
        edit. This doesn't try to detect the real cause (that's a judgment call for
        whoever reads it) — just points at the safe, explicit fix that exists now
        (POST /api/recalibrate-drawdown-baseline) instead of leaving someone to
        rediscover that a DB edit was ever the only option."""
        peak = self.portfolio.peak_value
        total = self.portfolio.total_value
        pct = (peak - total) / peak * 100 if peak else 0.0
        diag = f"total=${total:,.2f} peak=${peak:,.2f} drawdown={pct:.2f}%"
        if pct >= 50.0:
            diag += (" — if this doesn't reflect real trading losses (e.g. a manual "
                      "balance change), use Settings > Recalibrate Drawdown Baseline")
        return diag

    async def _earnings_soon(self, ticker: str, days: int = 2) -> tuple[bool, str]:
        """Return (True, date_str) if earnings fall within `days` calendar days, else (False, '')."""
        def _fetch():
            try:
                import yfinance as yf
                from datetime import date, timedelta
                cal = yf.Ticker(ticker).calendar
                if not cal:
                    return False, ""
                dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
                if not dates:
                    return False, ""
                today = date.today()
                cutoff = today + timedelta(days=days)
                for d in dates:
                    try:
                        ed = d.date() if hasattr(d, "date") else d
                        if today <= ed <= cutoff:
                            return True, str(ed)
                    except Exception:
                        pass
                return False, ""
            except Exception:
                return False, ""  # fail open — never block a buy due to data errors
        return await asyncio.to_thread(_fetch)

    async def _recent_momentum_ok(self, ticker: str, minutes: int = 10, tolerance: float = 0.997) -> bool:
        """Backward-looking pre-buy gate: is the stock flat-or-up over the last ~10 minutes,
        or still actively declining right now? Uses 1-minute intraday bars that already exist
        (market_data.get_recent_closes) — no waiting, no continuous monitoring needed, unlike
        near_miss_monitor_loop's forward-looking uptick confirmation (which serves the same
        purpose but has to wait for future ticks since near-miss candidates aren't otherwise
        polled). Added 2026-07-16 after VIRT was bought via the normal watchlist scan while
        actively crashing and stopped out ~1% lower within 10 minutes — real evidence the
        falling-knife risk isn't unique to near-miss, it applies to any buy at any moment the
        stock happens to be sliding. tolerance=0.997 allows ~0.3% noise so ordinary bid/ask
        wobble doesn't block a buy that's genuinely flat. Fails open (True) on any data error
        or insufficient history (e.g. first few minutes after open) — this is a risk-reduction
        gate, not a hard requirement, and a data hiccup should never block an otherwise-good buy."""
        try:
            closes = await self.market_data.get_recent_closes(ticker, minutes=minutes)
        except Exception as e:
            logger.debug("%s: momentum check failed, failing open: %s", ticker, e)
            return True
        if len(closes) < 3:
            return True
        return closes[-1] >= closes[0] * tolerance

    def _is_market_open(self) -> bool:
        now = self._now_et()
        if now.weekday() >= 5:
            return False
        if self._is_holiday:
            return False
        return self.market_open <= now.time() < self.market_close

    def _exit_order_maintenance_window_open(self) -> bool:
        """True during real market hours OR the pre-open window leading up to it
        (2026-07-30, BEN incident) -- exit-order protection-gap detection/remediation
        doesn't need to run during genuinely dead overnight hours, when a missing DAY
        order is completely expected (the previous session's orders already expired,
        not a real gap) and quote feeds can return stale/degenerate data with no
        business driving a real stop-loss decision. Confirmed live: BEN's own
        renewal attempt hit a real Alpaca quote reading ask=$0 and a bid $2.80 below
        the actual last trade, sitting right after the prior session's close --
        Alpaca's own order validation used that bad reading to reject a perfectly
        legitimate stop placement, triggering a real (if ultimately mitigated) false
        "price already breached" cascade with no genuine price move behind it at
        all. Reuses research.pre_open_batch_hours (already the "how far ahead of
        open should real activity start" dial for the scan pipeline) so protection
        is still fully rebuilt with the same lead time before open as everything
        else, just not hours earlier than that for no benefit."""
        return _in_exit_order_maintenance_window(
            self._now_et(), self.pre_open_batch_time, self.market_close, self._is_holiday)

    async def _update_holiday_flag(self):
        today_str = self._now_et().strftime("%Y-%m-%d")
        if self._holiday_check_date == today_str:
            return
        try:
            broker = self.order_manager.broker
            if hasattr(broker, "api") and broker.api:
                cal = await asyncio.to_thread(broker.api.get_calendar, today_str, today_str)
                self._is_holiday = len(cal) == 0
                if self._is_holiday:
                    logger.info("Market holiday (%s) — scanning paused for the day", today_str)
        except Exception as e:
            logger.warning("Could not check Alpaca market calendar: %s", e)
        # Always advance the check date — prevents log spam every 60s when broker is offline
        self._holiday_check_date = today_str

    def _scan_times_today(self) -> list[dtime]:
        """Build the scheduled scan times across market hours."""
        if self.explicit_scan_times:
            return sorted(self.explicit_scan_times)
        open_mins = self.market_open.hour * 60 + self.market_open.minute
        # Cap last scan at 15:30 so it always falls inside market hours
        # (_is_market_open uses strict '< 16:00', so a 16:00 scan is silently skipped)
        last_mins = self.market_close.hour * 60 + self.market_close.minute - 30
        total = last_mins - open_mins
        count = max(self.scans_per_day, 2)
        times = []
        for i in range(count):
            offset = open_mins + (total * i) // (count - 1)
            times.append(dtime(offset // 60, offset % 60))
        return times

    def _seconds_until_next_scan(self) -> tuple[float, str]:
        """Return (seconds_to_wait, formatted_time) for the next scan."""
        now = self._now_et()
        today_times = self._scan_times_today()

        if now.weekday() < 5:
            for t in today_times:
                scan_dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                if scan_dt > now:
                    delta = (scan_dt - now).total_seconds()
                    return delta, scan_dt.strftime("%H:%M ET")

        days_ahead = 1
        while True:
            next_day = now + timedelta(days=days_ahead)
            if next_day.weekday() < 5:
                break
            days_ahead += 1

        next_open = now.replace(
            hour=self.market_open.hour, minute=self.market_open.minute,
            second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        delta = (next_open - now).total_seconds()
        return delta, next_open.strftime("%a %H:%M ET")

    # ── Auto-scan loop (runs during market hours) ─────────────────────

    async def auto_scan_loop(self):
        scan_times = self._scan_times_today()
        scan_labels = [t.strftime("%H:%M") for t in scan_times]
        logger.info("Auto-scan loop started — %d scans/day at %s ET",
                    self.scans_per_day, ", ".join(scan_labels))

        _market_was_open = False

        while True:
            await self._update_holiday_flag()

            if self.paused or self.stopped:
                await asyncio.sleep(5)
                continue

            if not self._is_market_open():
                now_et = self._now_et()
                today_str = now_et.strftime("%Y-%m-%d")
                # Fire the pre-open batch once per day when we reach the trigger time (default 7:30 AM ET)
                if (self._pre_open_batch_date != today_str
                        and not self._is_holiday
                        and now_et.weekday() < 5
                        and now_et.time() >= self.pre_open_batch_time):
                    asyncio.create_task(self._run_pre_open_batch())
                    self._pre_open_batch_date = today_str
                # Fire the daily recap once per weekday shortly after market close
                if (self._daily_report_date != today_str
                        and not self._is_holiday
                        and now_et.weekday() < 5
                        and now_et.time() >= self.daily_report_time):
                    asyncio.create_task(self._generate_daily_report())
                    self._daily_report_date = today_str
                # Same trigger point, same once-per-day guard (2026-07-21) -- portfolio-
                # vs-market performance snapshot for the new Day/Total/YTD chart popups.
                if (self._performance_snapshot_date != today_str
                        and not self._is_holiday
                        and now_et.weekday() < 5
                        and now_et.time() >= self.daily_report_time):
                    asyncio.create_task(self._capture_daily_performance_snapshot())
                    asyncio.create_task(self._capture_benchmark_snapshot())
                    self._performance_snapshot_date = today_str
                # Dashboard market-open/closed badge (2026-08-04, owner request) --
                # broadcast only on the actual open->closed transition, not every closed-
                # loop tick (this branch re-runs roughly every 60s while closed).
                if _market_was_open:
                    await self.broadcast({"type": "market_status", "open": False})
                _market_was_open = False
                wait_secs, next_label = self._seconds_until_next_scan()
                self.next_cycle_at = next_label
                await self.broadcast({
                    "type": "market_closed",
                    "next_scan": next_label,
                })
                logger.info("Market closed — next scan at %s (%.0f min)",
                            next_label, wait_secs / 60)
                await asyncio.sleep(min(wait_secs, 60))
                continue

            # Market just opened — renew DAY exit orders that expired at prior close
            if not _market_was_open:
                _market_was_open = True
                # Dashboard market-open/closed badge (2026-08-04) -- symmetric transition
                # broadcast for the closed->open direction above.
                await self.broadcast({"type": "market_status", "open": True})
                if self.order_manager:
                    asyncio.create_task(self.order_manager.sync_exit_orders())

            # Mid-day re-scan firing check (2026-07-31) -- independent of scan_times'
            # own 30s-window trigger below (midday_scan_times isn't tied to
            # _seconds_until_next_scan), so checked directly here on every loop pass
            # instead. Per-slot tracking (not a single date flag, since there are 2+
            # configured times/day) mirrors _pre_open_batch_date's restart-catch-up
            # behavior: a slot whose time already passed while the process was down
            # still fires once noticed, since the guard is "already fired TODAY", not
            # "currently at this exact minute". _midday_rescan_in_progress and firing at
            # most one slot per tick (2026-07-31 incident fix) prevent 2+ catch-up slots
            # from firing concurrently on the same restart -- confirmed live to have
            # caused real duplicate Claude spend before this fix. A slot that's skipped
            # this tick because another scan is still running is simply picked up on a
            # later tick, since it isn't marked fired until it actually starts.
            # Also checks _full_scan_in_progress (2026-08-03, owner request) -- confirmed
            # live twice in one day: the manual "Full Scan" dashboard button and this
            # scheduled slot ran concurrently, both hitting the same universe/Claude API
            # (and, confirmed via real logs, exhausting NewsAPI's free-tier rate limit
            # between the two). A slot skipped for this reason is simply picked up on a
            # later tick, same as the existing _midday_rescan_in_progress skip already does.
            if (self.config.get("research", {}).get("midday_scan_enabled", True)
                    and not self._midday_rescan_in_progress
                    and not self._full_scan_in_progress):
                _now_et_midday = self._now_et()
                _today_str_midday = _now_et_midday.strftime("%Y-%m-%d")
                for _slot in self.midday_scan_times:
                    _slot_str = _slot.strftime("%H:%M")
                    if (self._midday_scan_fired.get(_slot_str) != _today_str_midday
                            and not self._is_holiday
                            and _now_et_midday.weekday() < 5
                            and _now_et_midday.time() >= _slot):
                        self._midday_scan_fired[_slot_str] = _today_str_midday
                        self._midday_rescan_in_progress = True
                        asyncio.create_task(self._run_midday_rescan(_slot_str))
                        asyncio.create_task(asyncio.to_thread(
                            _save_midday_scan_fired, dict(self._midday_scan_fired)))
                        break

            wait_secs, next_label = self._seconds_until_next_scan()
            if wait_secs > 30:
                self.next_cycle_at = next_label
                await self.broadcast({
                    "type": "waiting",
                    "next_scan": next_label,
                    "seconds": int(wait_secs),
                })
                await asyncio.sleep(min(wait_secs, 30))
                continue

            self.cycle_count += 1
            now_et = self._now_et()
            # Buying now happens exclusively through near_miss_monitor_loop's continuous
            # On Deck monitoring (see CLAUDE.md "Watchlist Replaced Entirely" 2026-07-17) —
            # this scheduled slot no longer does any scanning or buying of its own. Kept as
            # a lightweight heartbeat (cycle broadcasts, portfolio refresh) since the
            # dashboard's "next scan" / cycle-count UI still reads off it, and this loop
            # also owns the pre-open-scan and daily-report triggers above.
            logger.info("Scan cycle #%d at %s ET", self.cycle_count, now_et.strftime("%H:%M"))
            await self.broadcast({"type": "cycle_start", "cycle": self.cycle_count, "total": 0})

            # Repurpose the LAST scheduled scan of the day (2026-07-19) — with 2 scans/day
            # this is the 12:30 slot — to re-analyze every On Deck candidate a second time.
            # The FIRST slot (9:45) is deliberately skipped: it fires too soon after
            # pre-open (~90 min) to have earned a re-check yet. Without this, fair_value_
            # estimate (which drives every R/R gate decision) sits unrefreshed for the
            # entire trading day after pre-open, up to 8-9 hours stale by market close.
            scan_times_today = self._scan_times_today()
            if scan_times_today:
                closest = min(scan_times_today, key=lambda t:
                    abs((now_et.hour * 60 + now_et.minute) - (t.hour * 60 + t.minute)))
                if closest == scan_times_today[-1]:
                    asyncio.create_task(self._run_midday_reanalysis())

            await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})

            # Sleep 60s after each scan so the loop skips past the 30s trigger window
            # and doesn't re-fire the same scan time repeatedly (price-check scans
            # complete in <1s, so without this the loop fires ~30-60 times per scan slot).
            await asyncio.sleep(60)

            wait_secs, next_label = self._seconds_until_next_scan()
            self.next_cycle_at = next_label
            await self.broadcast({
                "type": "cycle_end",
                "cycle": self.cycle_count,
                "next_at": next_label,
            })

            logger.info("Cycle #%d complete. Next scan at %s", self.cycle_count, next_label)

    async def _bootstrap_watchlist(self):
        """First-run: scan universe once to populate an empty watchlist.
        Runs immediately at startup regardless of market hours so new installs
        have a populated watchlist before the first scheduled scan fires."""
        slots = self.watchlist_manager.slots_available()
        if slots == 0:
            return

        entry = self.add_ai_log("SYSTEM", "SETUP",
            f"First-run setup detected — scanning universe to populate {slots} watchlist slots. "
            "Stocks will appear as they are found. This runs once and takes 15–30 min.", "neutral")
        await self.broadcast({"type": "ai_log", "entry": entry})
        logger.info("Bootstrap: starting first-run watchlist population (%d slots)", slots)

        held_tickers = set(self.portfolio.positions.keys())
        min_conviction = self.config.get("research", {}).get("min_conviction_score", 7)
        # Use a faster delay for bootstrap (runs alone, not competing with scan loop)
        bootstrap_delay = 5 if self.has_claude else 1
        filled = 0

        for ticker in STOCK_UNIVERSE:
            if self.watchlist_manager.slots_available() == 0:
                break
            if ticker in held_tickers or ticker in self.watchlist_manager.get_active_tickers():
                continue
            try:
                passes, _ = await asyncio.to_thread(quick_screen, ticker)
                if not passes:
                    await asyncio.sleep(0.5)
                    continue
            except Exception:
                await asyncio.sleep(0.5)
                continue
            try:
                report = await self.research_engine.analyze_stock(ticker)
            except Exception as e:
                logger.debug("Bootstrap: analysis error for %s: %s", ticker, e)
                await asyncio.sleep(bootstrap_delay)
                continue
            if getattr(report, "is_fallback", False):
                await asyncio.sleep(bootstrap_delay)
                continue
            if report.signal.value in ("BUY", "STRONG BUY") and report.conviction_score >= min_conviction:
                self.watchlist_manager.add(ticker, report.company_name, "")
                filled += 1
                size_now = self.watchlist_manager.size()
                entry = self.add_ai_log(ticker, "SETUP",
                    f"Added — {report.signal.value} | Conviction {report.conviction_score}/10 "
                    f"| Watchlist: {size_now}/{self.watchlist_manager.target_size}", "buy")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await self.broadcast({"type": "stocks_update",
                                      "stocks": self.watchlist_manager.get_active(),
                                      "watchlist_size": self.watchlist_manager.size(),
                                      "watchlist_target": self.watchlist_manager.target_size})
            await asyncio.sleep(bootstrap_delay)

        final_size = self.watchlist_manager.size()
        level = "success" if final_size >= self.watchlist_manager.target_size // 2 else "warning"
        entry = self.add_ai_log("SYSTEM", "SETUP",
            f"First-run setup complete — {filled} stocks added "
            f"({final_size}/{self.watchlist_manager.target_size} watchlist slots filled). "
            "Scheduled scans will maintain and improve the watchlist going forward.", level)
        await self.broadcast({"type": "ai_log", "entry": entry})
        logger.info("Bootstrap complete: %d stocks added to watchlist", filled)

    async def _replace_one_watchlist_slot(self):
        """Find one BUY/STRONG BUY from the universe to fill a freed watchlist slot.

        Its only caller was _auto_buy_after_deep_dive, deleted 2026-08-09 (GitHub #72,
        the dead deep-dive-confirmation auto-buy subsystem) -- this is now itself dead
        (zero callers). Left in place undeleted, same precedent as this project's other
        already-documented dead Watchlist-era functions (_bootstrap_watchlist,
        run_replacement_scan, _evict_underperformers, _buy_from_watchlist_by_price --
        see CLAUDE.md's On Deck Buy Pipeline section), pending an explicit owner
        decision on whether to remove it too.
        """
        min_conviction = self.config.get("research", {}).get("min_conviction_score", 7)
        held_tickers = set(self.portfolio.positions.keys())

        # Batch candidates first (best R/R), then universe
        batch_cands = self.watchlist_manager.get_candidates(limit=10, exclude=held_tickers)
        batch_tickers = [c["ticker"] for c in batch_cands]
        batch_set = set(batch_tickers)
        universe_tickers = [t for t in self.watchlist_manager.available_from_universe(STOCK_UNIVERSE)
                            if t not in held_tickers and t not in batch_set]
        available = batch_tickers + universe_tickers

        for ticker in available:
            is_batch = ticker in batch_set
            if not is_batch:
                passes, reason = await asyncio.to_thread(quick_screen, ticker)
                if not passes:
                    logger.debug("Quick screen rejected %s: %s", ticker, reason)
                    if ticker in STOCK_UNIVERSE:
                        self.watchlist_manager.set_scan_cursor(
                            (STOCK_UNIVERSE.index(ticker) + 1) % len(STOCK_UNIVERSE))
                    await asyncio.sleep(0.5)
                    continue
            try:
                report = await self.research_engine.analyze_stock(ticker)
            except Exception:
                if is_batch:
                    self.watchlist_manager.remove_candidate(ticker)
                await asyncio.sleep(self.stock_delay)
                continue
            if is_batch:
                self.watchlist_manager.remove_candidate(ticker)
            elif ticker in STOCK_UNIVERSE:
                self.watchlist_manager.set_scan_cursor(
                    (STOCK_UNIVERSE.index(ticker) + 1) % len(STOCK_UNIVERSE))
            if getattr(report, "is_fallback", False):
                entry = self.add_ai_log(ticker, "WATCHLIST",
                    f"⚠ AI unavailable for {ticker} — skipping watchlist add", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await asyncio.sleep(self.stock_delay)
                continue
            if report.signal.value in ("BUY", "STRONG BUY") and report.conviction_score >= min_conviction:
                self.watchlist_manager.add(ticker, report.company_name, "")
                entry = self.add_ai_log(ticker, "WATCHLIST",
                    f"Added to watchlist (replaced bought position) — "
                    f"{report.signal.value} conviction {report.conviction_score}/10", "buy")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await self.broadcast({"type": "stocks_update",
                                      "stocks": self.watchlist_manager.get_active(),
                                      "watchlist_size": self.watchlist_manager.size(),
                                      "watchlist_target": self.watchlist_manager.target_size})
                logger.info("Watchlist slot filled by %s after buy", ticker)
                return
            await asyncio.sleep(self.stock_delay)
        logger.warning("Could not find a replacement watchlist candidate from universe")

    async def run_replacement_scan(self):
        """Evict underperformers and any held positions, scan universe until all slots filled."""
        if self._replacement_scan_running:
            logger.info("Replacement scan already running — skipping duplicate trigger")
            return
        self._replacement_scan_running = True
        try:
            await self._run_replacement_scan_body()
        finally:
            self._replacement_scan_running = False

    async def _run_replacement_scan_body(self):
        underperformers = self.watchlist_manager.get_underperformers()

        # Also remove any watchlist stocks that are now held — they're monitored hourly
        held_in_watchlist = [t for t in self.watchlist_manager.get_active_tickers()
                             if t in self.portfolio.positions]
        for ticker in held_in_watchlist:
            self.watchlist_manager.remove(ticker)

        if not underperformers and not held_in_watchlist and self.watchlist_manager.slots_available() == 0:
            logger.info("End-of-day replacement scan: watchlist is full and clean — no changes needed")
            return

        for ticker in underperformers:
            self.watchlist_manager.remove(ticker)

        slots = self.watchlist_manager.slots_available()
        logger.info("Weekly replacement scan: evicted %s — scanning universe for %d replacement(s)",
                    ", ".join(underperformers), slots)
        entry = self.add_ai_log("SYSTEM", "WATCHLIST",
            f"Evicted {len(underperformers)} underperformer(s): {', '.join(underperformers)}. "
            f"Scanning for {slots} replacement(s).", "warning")
        await self.broadcast({"type": "ai_log", "entry": entry})

        filled = 0
        screened_out = 0
        analyzed = 0
        held_tickers = set(self.portfolio.positions.keys())
        min_conviction = self.config.get("research", {}).get("min_conviction_score", 7)

        # ── Build scan order: batch candidates first (skip quick screen), then universe ──
        batch_cands = self.watchlist_manager.get_candidates(limit=slots * 3, exclude=held_tickers)
        batch_tickers = [c["ticker"] for c in batch_cands]
        batch_set = set(batch_tickers)
        if batch_tickers:
            entry = self.add_ai_log("SYSTEM", "WATCHLIST",
                f"{len(batch_tickers)} pre-open scan candidates queued first — skipping quick screen", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})

        universe_tickers = [t for t in self.watchlist_manager.available_from_universe(STOCK_UNIVERSE)
                            if t not in held_tickers and t not in batch_set]
        available = batch_tickers + universe_tickers
        last_scanned = None

        total_available = len(available)
        scanned_count = 0
        for ticker in available:
            # Live check so the loop stops immediately if target_size was lowered mid-scan
            # or if concurrent additions (held-slot release) brought the watchlist to target.
            if self.watchlist_manager.slots_available() <= 0:
                break
            # Stop if market closes mid-scan — cursor is saved, resumes next open
            if not self._is_market_open():
                logger.info("Market closed mid-replacement scan — pausing at %s, cursor saved", ticker)
                entry = self.add_ai_log("SYSTEM", "WATCHLIST",
                    f"Market closed — replacement scan paused at {ticker}. "
                    f"Resumes next market open ({filled} added this session).", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                break
            last_scanned = ticker
            scanned_count += 1
            is_batch = ticker in batch_set

            # Update the middle panel scan strip so universe scan is visible there too
            self.current_ticker = ticker
            await self.broadcast({"type": "scan_progress", "status": {
                "current_ticker": ticker,
                "index": scanned_count,
                "total": total_available,
                "cycle": self.cycle_count,
                "label": f"{'Pre-screened' if is_batch else 'Universe'} scan — {self.watchlist_manager.slots_available()} slot(s) | {analyzed} analyzed, {screened_out} skipped",
            }})

            if is_batch:
                # Already quick-screened nightly — go straight to full Claude analysis
                entry = self.add_ai_log(ticker, "UNIVERSE SCAN",
                    "Batch pre-screened — running full analysis", "neutral")
                await self.broadcast({"type": "ai_log", "entry": entry})
            else:
                # Quick screen first — 2s vs 25s; skip Claude analysis for obvious non-candidates
                passes, reason = await asyncio.to_thread(quick_screen, ticker)
                if not passes:
                    screened_out += 1
                    logger.debug("Quick screen rejected %s: %s", ticker, reason)
                    entry = self.add_ai_log(ticker, "UNIVERSE SCAN",
                        f"Screened out — {reason}", "neutral")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    if ticker in STOCK_UNIVERSE:
                        self.watchlist_manager.set_scan_cursor(
                            (STOCK_UNIVERSE.index(ticker) + 1) % len(STOCK_UNIVERSE))
                    await asyncio.sleep(0.5)
                    continue
            analyzed += 1

            try:
                report = await self.research_engine.analyze_stock(ticker)
            except Exception as e:
                logger.warning("Replacement scan error on %s: %s", ticker, e)
                if is_batch:
                    self.watchlist_manager.remove_candidate(ticker)
                await asyncio.sleep(self.stock_delay)
                continue

            if is_batch:
                self.watchlist_manager.remove_candidate(ticker)

            if getattr(report, "is_fallback", False):
                entry = self.add_ai_log(ticker, "UNIVERSE SCAN",
                    f"⚠ AI unavailable for {ticker} — result discarded", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await asyncio.sleep(self.stock_delay)
                continue

            level = "buy" if report.signal.value in ("BUY", "STRONG BUY") else "neutral"
            entry = self.add_ai_log(ticker, "UNIVERSE SCAN",
                f"{report.signal.value} | Conviction {report.conviction_score}/10", level)
            await self.broadcast({"type": "ai_log", "entry": entry})

            if report.signal.value in ("BUY", "STRONG BUY") and report.conviction_score >= min_conviction:
                self.watchlist_manager.add(ticker, report.company_name, "")
                filled += 1
                entry = self.add_ai_log(ticker, "WATCHLIST",
                    f"Added to watchlist — {report.signal.value} conviction {report.conviction_score}/10",
                    "buy")
                await self.broadcast({"type": "ai_log", "entry": entry})
                await self.broadcast({"type": "stocks_update",
                                      "stocks": self.watchlist_manager.get_active(),
                                      "watchlist_size": self.watchlist_manager.size(),
                                      "watchlist_target": self.watchlist_manager.target_size})
                logger.info("Replacement: added %s (%s, conviction %d)",
                            ticker, report.signal.value, report.conviction_score)

            await asyncio.sleep(self.stock_delay)

        if last_scanned and last_scanned in STOCK_UNIVERSE:
            next_cursor = (STOCK_UNIVERSE.index(last_scanned) + 1) % len(STOCK_UNIVERSE)
            self.watchlist_manager.set_scan_cursor(next_cursor)

        # Clear the scan strip
        self.current_ticker = ""
        await self.broadcast({"type": "scan_progress", "status": {
            "current_ticker": "",
            "index": scanned_count,
            "total": total_available,
            "cycle": self.cycle_count,
            "label": "Universe scan complete",
        }})

        logger.info("Replacement scan complete: %d/%d slots filled. Watchlist now %d stocks.",
                    filled, slots, self.watchlist_manager.size())
        entry = self.add_ai_log("SYSTEM", "WATCHLIST",
            f"Universe scan complete — {filled}/{slots} slots filled. "
            f"Watchlist: {self.watchlist_manager.size()} stocks.", "success")
        await self.broadcast({"type": "ai_log", "entry": entry})

    async def _rebuy_legacy_positions(self):
        """Sell round-share (legacy) positions and immediately rebuy notionally to get T1/T2/T3."""
        import alpaca_trade_api as tradeapi
        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        api = tradeapi.REST(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
            os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )

        # Legacy = round share count (bought by qty, not notional)
        all_positions = await asyncio.to_thread(api.list_positions)
        legacy = [
            (p.symbol, float(p.qty), float(p.market_value))
            for p in all_positions
            if float(p.qty) == int(float(p.qty))
        ]

        if not legacy:
            entry = self.add_ai_log("SYSTEM", "REBUY", "No legacy positions found", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return

        entry = self.add_ai_log("SYSTEM", "REBUY",
            f"Found {len(legacy)} legacy position(s): {', '.join(t for t,_,_ in legacy)} — selling and rebuying notionally",
            "neutral")
        await self.broadcast({"type": "ai_log", "entry": entry})

        skipped = []
        for ticker, qty, mkt_value in legacy:
            # Only rebuy if signal is BUY or STRONG BUY
            sig_info = self.ticker_signals.get(ticker, {})
            sig_str = sig_info.get("signal", "").upper()
            # Fall back to research report if no cached signal
            if not sig_str:
                rpt = self.research_engine.reports.get(ticker)
                sig_str = rpt.signal.value.upper() if rpt else ""
            if sig_str not in ("BUY", "STRONG BUY"):
                entry = self.add_ai_log(ticker, "REBUY",
                    f"Skipping — signal is {sig_str or 'unknown'}, not a buy", "neutral")
                await self.broadcast({"type": "ai_log", "entry": entry})
                skipped.append(ticker)
                continue

            # Cancel existing orders for this ticker
            try:
                open_orders = await asyncio.to_thread(api.list_orders, status="open")
                for o in open_orders:
                    if o.symbol == ticker:
                        await asyncio.to_thread(api.cancel_order, o.id)
                        await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("Cancel orders failed for %s: %s", ticker, e)

            # Sell all shares at market
            try:
                await asyncio.to_thread(
                    api.submit_order,
                    symbol=ticker, qty=int(qty), side="sell",
                    type="market", time_in_force="day",
                )
                entry = self.add_ai_log(ticker, "REBUY",
                    f"Sold {int(qty)} share(s) at market — waiting for fill", "sell")
                await self.broadcast({"type": "ai_log", "entry": entry})
            except Exception as e:
                entry = self.add_ai_log(ticker, "REBUY", f"Sell failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                continue

            # Remove old position BEFORE sleeping so position_update_loop cannot concurrently
            # detect the Alpaca close and call close_position_async (double-cash-credit / phantom trade).
            self.portfolio.positions.pop(ticker, None)
            await self.portfolio._remove_position_db(ticker)
            self.order_manager._stop_order_ids.pop(ticker, None)
            self.order_manager._tp_orders.pop(ticker, None)
            self.order_manager._queued_tps.pop(ticker, None)
            self.order_manager._pending_stops.pop(ticker, None)

            # Wait for fill (paper trading fills near-instantly)
            await asyncio.sleep(6)

            # Re-sync cash from Alpaca so the rebuy check uses the actual post-sell balance
            try:
                account = await asyncio.to_thread(api.get_account)
                self.portfolio.cash = float(account.cash)
                await self.portfolio._save_state()
            except Exception as _e:
                logger.warning("Cash re-sync after rebuy sell of %s failed: %s", ticker, _e)

            # Get current price for stop/TP levels
            try:
                latest = await asyncio.to_thread(api.get_latest_trade, ticker)
                current_price = float(latest.price)
            except Exception:
                current_price = mkt_value / qty

            # Use existing research report levels if available, else calculate
            report = self.research_engine.reports.get(ticker)
            if report and report.stop_loss > 0 and len(report.take_profit_targets) >= 3:
                stop_loss = report.stop_loss
                take_profits = list(report.take_profit_targets)
            else:
                tp_cfg = self.config.get("take_profit", {})
                sl  = tp_cfg.get("stop_loss_pct", 7.0)  / 100
                t1p = tp_cfg.get("t1_pct",  5.0)  / 100
                t2p = tp_cfg.get("t2_pct", 10.0)  / 100
                t3p = tp_cfg.get("t3_pct", 17.0)  / 100
                stop_loss = round(current_price * (1 - sl), 2)
                take_profits = [
                    round(current_price * (1 + t1p), 2),
                    round(current_price * (1 + t2p), 2),
                    round(current_price * (1 + t3p), 2),
                ]

            # Rebuy notionally for the same dollar amount
            # Respect drawdown halt — don't rebuy into a halted portfolio
            drawdown_status = self.risk_manager.check_drawdown(self.portfolio)
            if drawdown_status in ("halt", "exit_review", "defensive"):
                entry = self.add_ai_log(ticker, "REBUY",
                    f"Skipping rebuy — portfolio in {drawdown_status} state ({self._drawdown_diagnostic()})", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                skipped.append(ticker)
                continue

            signal = TradeSignal(
                ticker=ticker, signal=Sig.BUY, conviction=8,
                entry_price=current_price, stop_loss=stop_loss,
                take_profit_targets=take_profits,
                position_size_pct=7.0, position_size_dollars=round(mkt_value, 2),
                shares=0,
                reasoning="Rebuy notionally to establish T1/T2/T3 take-profit orders",
                research_report=report, generated_at=datetime.now(), should_execute=True,
            )
            try:
                order = await self.order_manager.execute(signal)
                entry = self.add_ai_log(ticker, "REBUY",
                    f"Rebuying ${mkt_value:.0f} notionally — "
                    f"stop ${stop_loss:.2f} | T1 ${take_profits[0]:.2f} | "
                    f"T2 ${take_profits[1]:.2f} | T3 ${take_profits[2]:.2f}", "buy")
                await self.broadcast({"type": "ai_log", "entry": entry})
            except Exception as e:
                entry = self.add_ai_log(ticker, "REBUY", f"Rebuy failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})

            await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
            await asyncio.sleep(4)

        rebuyed = len(legacy) - len(skipped)
        entry = self.add_ai_log("SYSTEM", "REBUY",
            f"Done — {rebuyed} rebuyed, {len(skipped)} skipped ({', '.join(skipped) or 'none'}). "
            "Placing exit orders now...", "success")
        await self.broadcast({"type": "ai_log", "entry": entry})

        # Re-sync everything from Alpaca — fixes cash that got corrupted by sell/rebuy accounting
        await asyncio.sleep(12)
        if self.order_manager:
            await self.order_manager._sync_portfolio()   # authoritative cash + positions from Alpaca
            _closed_at_broker = await self.order_manager.update_positions()  # process any pending fills, check TPs
            for _c in _closed_at_broker:
                await self._report_alpaca_detected_close(
                    _c["ticker"], _c["shares"], _c["fill_price"], _c["pnl"])
            for _c in self.order_manager.pop_stream_closed_reports():
                await self._report_alpaca_detected_close(
                    _c["ticker"], _c["shares"], _c["fill_price"], _c["pnl"])
            await self.order_manager.sync_exit_orders()  # place any missing exit orders

        await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})

    async def _evict_underperformers(self):
        """Evict held positions and weak-signal stocks from watchlist.
        Does NOT fill slots — that happens exclusively at pre-open."""
        held_in_watchlist = [t for t in self.watchlist_manager.get_active_tickers()
                             if t in self.portfolio.positions]
        for ticker in held_in_watchlist:
            self.watchlist_manager.remove(ticker)

        underperformers = self.watchlist_manager.get_underperformers()
        for ticker in underperformers:
            self.watchlist_manager.remove(ticker)
            self.deep_dive_reports.pop(ticker, None)

        if held_in_watchlist or underperformers:
            entry = self.add_ai_log("SYSTEM", "WATCHLIST",
                f"Evicted {len(underperformers)} underperformer(s)"
                + (f" + {len(held_in_watchlist)} now-held tickers" if held_in_watchlist else "")
                + f": {', '.join(underperformers + held_in_watchlist)}. "
                "Slots will be refilled at next pre-open scan.", "warning")
            await self.broadcast({"type": "ai_log", "entry": entry})
            await self.broadcast({"type": "stocks_update",
                                  "stocks": self.watchlist_manager.get_active(),
                                  "watchlist_size": self.watchlist_manager.size(),
                                  "watchlist_target": self.watchlist_manager.target_size})

    async def _buy_from_watchlist_by_price(self):
        """Daytime buy: check current price against pre-open deep-dive data, buy if still valid.
        Zero Claude calls — all analysis was done pre-open. One re-dive allowed per stock
        if price moved more than 5% above the pre-open entry price."""
        _now = self._now_et()
        if _now.hour >= 14:
            entry = self.add_ai_log("SYSTEM", "AUTO_TRADE",
                f"Auto-buy skipped — past 2:00 PM ET ({_now.strftime('%H:%M')}); "
                "next window opens at market open", "warning")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return

        if not self.config["trading"].get("auto_execute", False) or not self.broker_connected:
            return

        from src.decision.signal_generator import TradeSignal
        from src.research.engine import Signal as Sig

        max_positions = self.config.get("portfolio", {}).get("max_positions", 10)
        min_conviction = self.config["research"]["min_conviction_score"]
        min_rr = self.config["research"]["min_risk_reward_ratio"]
        price_tolerance_pct = 5.0  # re-analyze if current price rose > this % above pre-open entry

        # Fetch current price for every watchlist candidate once, up front, so the R/R
        # tiebreaker below reflects where the stock actually trades right now rather than
        # its stale pre-open entry price. Free (yfinance, no Claude cost) — the main loop
        # below still re-fetches its own fresh quote per ticker at actual decision time,
        # since some tickers may sit through slow reconfirms before their turn comes up.
        watchlist_ticker_list = [s["ticker"] for s in self.watchlist_manager.get_active()]
        ranking_quotes: dict[str, float] = {}
        for _t in watchlist_ticker_list:
            try:
                _q = await self.market_data.get_quote(_t)
                if _q.price > 0:
                    ranking_quotes[_t] = _q.price
            except Exception as e:
                logger.debug("%s: price fetch failed during ranking: %s", _t, e)

        # Rank by the same quality score used for rotation swaps (conviction + margin/10
        # + R/R tiebreaker, using live price) so limited open slots go to the strongest
        # candidates first, not alphabetical order.
        def _candidate_score(t: str) -> float:
            cand = self.deep_dive_reports.get(t)
            if not cand:
                return -999.0
            base = cand.get("conviction_score", 0) + (cand.get("margin_of_safety_pct", 0) / 10)
            stop = cand.get("stop_loss", 0)
            fair_value = cand.get("fair_value_estimate", 0)
            live_price = ranking_quotes.get(t)
            if live_price and stop > 0 and fair_value and fair_value > 0:
                risk = live_price - stop
                if risk > 0:
                    rr = (fair_value - live_price) / risk
                    base += (rr - min_rr) * 2
            return base

        watchlist_tickers = sorted(watchlist_ticker_list, key=_candidate_score, reverse=True)
        pending_tickers: set[str] = set()
        pending_cash_reserved: float = 0.0
        rotation_proceeds: float = 0.0
        rotated_out: set[str] = set()

        # Suppress peak_value ratcheting while buys/rotation-sells are in flight — a
        # sold position can remain in portfolio.positions (not yet Alpaca-confirmed
        # closed) at the same moment its replacement is already bought, transiently
        # double-counting both in total_value.
        self.portfolio._rotation_in_progress = True

        for ticker in watchlist_tickers:
            if ticker in self.portfolio.positions or ticker in pending_tickers:
                continue

            dd = self.deep_dive_reports.get(ticker)
            if not dd or dd.get("is_fallback"):
                logger.debug("%s: no pre-open deep dive cached — skipping daytime buy", ticker)
                continue

            if dd.get("signal", "") not in ("BUY", "STRONG BUY"):
                continue
            conviction = dd.get("conviction_score", 0)
            if conviction < min_conviction:
                continue

            entry_price = dd.get("entry_price", 0.0)
            stop_loss = dd.get("stop_loss", 0.0)
            targets = dd.get("take_profit_targets") or []
            margin = dd.get("margin_of_safety_pct", 0.0)
            fair_value = dd.get("fair_value_estimate", 0.0)

            if not entry_price or not stop_loss or not targets:
                logger.debug("%s: incomplete pre-open data — skipping", ticker)
                continue

            # Current market price
            try:
                quote = await self.market_data.get_quote(ticker)
                current_price = quote.price
            except Exception as e:
                logger.warning("%s: price fetch failed: %s", ticker, e)
                continue

            if current_price <= 0:
                continue

            # Below stop — don't buy (would immediately stop out)
            if current_price <= stop_loss:
                logger.info("%s: price $%.2f at/below stop $%.2f — skipping", ticker, current_price, stop_loss)
                continue

            pct_from_entry = (current_price - entry_price) / entry_price * 100

            # Price moved up significantly — re-verify with a fresh analysis
            if pct_from_entry > price_tolerance_pct:
                entry_obj = self.add_ai_log(ticker, "SCAN",
                    f"Price +{pct_from_entry:.1f}% from pre-open entry — re-verifying...",
                    "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                try:
                    new_report = await self.research_engine.analyze_stock(ticker)
                    if (getattr(new_report, "is_fallback", True)
                            or new_report.signal.value not in ("BUY", "STRONG BUY")
                            or new_report.conviction_score < min_conviction):
                        reason = (f"{new_report.signal.value} | "
                                  f"Conviction {new_report.conviction_score}/10")
                        entry_obj = self.add_ai_log(ticker, "SCAN",
                            f"Re-verify: no longer buyable — {reason}. Removing from watchlist.", "warning")
                        await self.broadcast({"type": "ai_log", "entry": entry_obj})
                        self.watchlist_manager.remove(ticker)
                        self.deep_dive_reports.pop(ticker, None)
                        continue

                    # Use fresh targets (from Claude's new analysis, or compute from config)
                    new_targets = (list(new_report.take_profit_targets)
                                   if new_report.take_profit_targets
                                   else [
                                       round(new_report.entry_price * (1 + self.config["take_profit"]["t1_pct"] / 100), 2),
                                       round(new_report.entry_price * (1 + self.config["take_profit"]["t2_pct"] / 100), 2),
                                       round(new_report.entry_price * (1 + self.config["take_profit"]["t3_pct"] / 100), 2),
                                   ])
                    targets = new_targets
                    entry_price = new_report.entry_price
                    stop_loss = new_report.stop_loss
                    conviction = new_report.conviction_score
                    margin = new_report.margin_of_safety_pct
                    fair_value = new_report.fair_value_estimate
                    self.deep_dive_reports[ticker] = {**dd,
                        "signal": new_report.signal.value,
                        "conviction_score": conviction,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "fair_value_estimate": new_report.fair_value_estimate,
                        "margin_of_safety_pct": new_report.margin_of_safety_pct,
                        "take_profit_targets": targets,
                        "is_fallback": False,
                        "generated_at": new_report.generated_at.isoformat(),
                    }
                    asyncio.create_task(asyncio.to_thread(_save_dd_cache, self.deep_dive_reports))
                    entry_obj = self.add_ai_log(ticker, "SCAN",
                        f"Re-verify OK — {new_report.signal.value} | Conviction {conviction}/10 | "
                        f"Entry ${entry_price:.2f}", "success")
                    await self.broadcast({"type": "ai_log", "entry": entry_obj})
                except Exception as e:
                    logger.error("%s: re-verify analysis failed: %s", ticker, e)
                    continue

            buy_price = current_price

            # R/R check with current buy price — reward side is Claude's fair_value_estimate,
            # not a fixed % target, so it actually discriminates between stocks by real
            # estimated upside rather than producing the same ratio for every candidate.
            risk = buy_price - stop_loss
            if not fair_value or fair_value <= 0:
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                    "Skipping — no valid fair value estimate for R/R check", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                continue
            rr = (fair_value - buy_price) / risk if risk > 0 else 0
            if rr < min_rr:
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                    f"Skipping — R/R {rr:.2f} < {min_rr} at current price ${buy_price:.2f} "
                    f"(fair value ${fair_value:.2f}, stop ${stop_loss:.2f})", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                continue

            # Position sizing
            position_size = self.risk_manager.calculate_position_size(
                buy_price, stop_loss, self.portfolio.total_value)
            shares = position_size / buy_price if buy_price > 0 else 0
            if shares < 0.001:
                logger.info("%s: position size too small ($%.2f) — skipping", ticker, position_size)
                continue

            # Portfolio capacity
            current_count = len(self.portfolio.positions) + len(pending_tickers)
            if current_count >= max_positions:
                candidate_meta = {
                    "ticker": ticker,
                    "conviction": conviction,
                    "margin": margin,
                    "fair_value": fair_value,
                    "score": conviction + (margin / 10) + (rr - min_rr) * 2,
                }
                swapped, swap_proceeds = await self._try_rotation_swap(candidate_meta, rotated_out)
                if not swapped:
                    entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"Portfolio full ({max_positions}) — no weaker holding to swap", "warning")
                    await self.broadcast({"type": "ai_log", "entry": entry_obj})
                    continue
                rotation_proceeds += swap_proceeds

            # Per-candidate halt guards
            if not self.risk_manager.check_daily_loss(self.portfolio):
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                    "Auto-buy halted — daily loss limit reached during scan", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                break
            dd_state = self.risk_manager.check_drawdown(self.portfolio)
            if dd_state in ("halt", "exit_review", "defensive"):
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                    f"Auto-buy halted — portfolio in {dd_state} state ({self._drawdown_diagnostic()})", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                if dd_state in ("halt", "exit_review"):
                    asyncio.create_task(_notify(
                        f"Drawdown {dd_state.upper()}",
                        f"Auto-buys suspended — portfolio drawdown exceeded threshold",
                        priority="urgent", tags="rotating_light"))
                break

            # Cash reserve check
            effective_cash = self.portfolio.cash - pending_cash_reserved + rotation_proceeds
            required_reserve = self.portfolio.total_value * (
                self.config["risk_management"]["min_cash_reserve_pct"] / 100)
            if effective_cash - position_size < required_reserve:
                logger.info("%s: insufficient cash (need $%.0f, have $%.0f free)",
                            ticker, position_size, effective_cash - required_reserve)
                continue

            # Momentum check — don't buy into an active decline (see _recent_momentum_ok)
            if not await self._recent_momentum_ok(ticker):
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                    "Skipping this scan — still actively declining over the last 10 min "
                    "(will re-check next scan)", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})
                continue

            # Build and execute trade signal
            sig_str = dd.get("signal", "BUY")
            sig_enum = Sig.STRONG_BUY if sig_str == "STRONG BUY" else Sig.BUY
            signal = TradeSignal(
                ticker=ticker,
                signal=sig_enum,
                conviction=conviction,
                entry_price=buy_price,
                stop_loss=stop_loss,
                take_profit_targets=targets,
                position_size_pct=dd.get("position_size_pct",
                                         self.config["risk_management"].get("starting_position_pct", 3.0)),
                position_size_dollars=position_size,
                shares=shares,
                reasoning=(f"Pre-open deep-dive confirmed BUY; "
                           f"price ${buy_price:.2f} ({pct_from_entry:+.1f}% from pre-open entry)"),
                research_report=None,
                generated_at=datetime.now(),
                should_execute=True,
            )

            try:
                order = await self.order_manager.execute(signal)
                if order and order.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                    pending_tickers.add(ticker)
                    pending_cash_reserved += position_size
                    # Log the real fill price, not the pre-trade recommendation
                    # (2026-08-08, GitHub #53).
                    if order.filled_price is not None:
                        signal.entry_price = order.filled_price
                    self.trade_logger.log_trade(signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                    result = {
                        "ticker": ticker, "status": order.status.value,
                        "filled_price": order.filled_price, "shares": shares,
                    }
                    await self.broadcast({"type": "trade_executed", "trade": result})
                    await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                    _fp = order.filled_price or buy_price
                    entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"AUTO BUY {shares:.4g} shares @ "
                        f"${_fp:.2f} | "
                        f"Conviction {conviction}/10 | Margin {margin:.0f}% | "
                        f"Stop ${stop_loss:.2f} | T3 ${targets[-1]:.2f}", "buy")
                    await self.broadcast({"type": "ai_log", "entry": entry_obj})
                    asyncio.create_task(_notify(
                        f"BUY {ticker}",
                        f"{shares:.4g} shares @ ${_fp:.2f} | Conviction {conviction}/10 | Stop ${stop_loss:.2f} | T3 ${targets[-1]:.2f}",
                        priority="high", tags="white_check_mark"))
                    logger.info("Auto-buy %s — %.4g shares @ $%.2f (margin %.0f%%)",
                                ticker, shares, _fp, margin)
                    # Remove from watchlist — position is monitored hourly
                    self.watchlist_manager.remove(ticker)
                    await self.broadcast({"type": "stocks_update",
                        "stocks": self.watchlist_manager.get_active(),
                        "watchlist_size": self.watchlist_manager.size(),
                        "watchlist_target": self.watchlist_manager.target_size})
                elif order:
                    entry_obj = self.add_ai_log(ticker, "AUTO_TRADE",
                        f"Auto-buy rejected by broker: {order.status.value}", "error")
                    await self.broadcast({"type": "ai_log", "entry": entry_obj})
            except Exception as e:
                entry_obj = self.add_ai_log(ticker, "AUTO_TRADE", f"Auto-buy failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry_obj})

        self.portfolio._rotation_in_progress = False

    def _is_on_deck_blocked(self, ticker: str) -> bool:
        """True if ticker was manually removed from On Deck and the block hasn't expired.
        Lazily purges an expired temporary block on the same lookup that finds it — no
        separate cleanup pass needed since every candidate-creation site already calls this
        before adding a ticker. The price-based breakout check (2026-07-29) is checked
        separately and more cheaply by _check_price_based_unblocks (a single free quote per
        blocked ticker per near_miss_monitor_loop tick would be wasteful to also run on
        every one of THIS function's many population-gate call sites)."""
        raw = self.on_deck_blocked.get(ticker, "__absent__")
        if raw == "__absent__":
            return False
        entry = _normalize_block_entry(raw)
        block_until = entry["until"]
        if block_until is None:
            return True  # permanent block
        try:
            if self._now_et() < datetime.fromisoformat(block_until):
                return True
        except Exception:
            pass  # unparseable timestamp — treat as expired rather than stuck blocked forever
        self.on_deck_blocked.pop(ticker, None)
        asyncio.create_task(asyncio.to_thread(_save_on_deck_blocked, dict(self.on_deck_blocked)))
        return False

    def _wash_sale_blocked(self, ticker: str) -> bool:
        """True if ticker currently can't be bought due to the wash-sale rebuy cooldown
        (2026-07-28, user request) -- same underlying check _attempt_near_miss_promotion
        already runs before spending a real Claude call (RiskManager.check_wash_sale_cooldown
        against Portfolio.recent_losses, a pure in-memory lookup). Used to keep a cooled-down
        ticker off On Deck/On Shore entirely, not just block it at promotion time -- a
        candidate that can't legally be bought for the next N days shouldn't sit on the
        dashboard showing a live "Buy" badge as if it's a real prospect."""
        return not self.risk_manager.check_wash_sale_cooldown(ticker, self.portfolio)

    async def remove_on_deck_candidate(
        self, ticker: str, permanent: bool, days, note: str = "", initiated_by: str = "user",
    ) -> None:
        """User-initiated removal (2026-07-18) — "if I feel it's not a good stock for any
        reason." Distinct from the automatic conviction-based removal: this is a judgment
        call the user is explicitly allowed to make regardless of what the numbers say.
        Removes immediately and blocks re-adding — permanently (until manually un-blocked via
        a future removal-list UI, not built yet) or for a number of days (default 1, per the
        user's own default — "if we have 3 we don't need 1" was the reasoning for a short
        default rather than always-permanent).

        Two additions (2026-07-29, FNB discussion), both scoped to this same call: (1) a
        reference peak — the recent window high, see _recent_window_high, falling back to
        the candidate's current price if no price history was tracked yet — is captured so
        _check_price_based_unblocks can restore eligibility for free the moment price
        genuinely breaks out, regardless of the days/permanent choice above — user
        pushback: "ya never block it permanently.. why would you do that? ... can we use
        no cost price adjustments." (2) an optional note persists in on_deck_notes
        (separate from on_deck_blocked, and never auto-cleared by either the time-based or
        price-based unblock) so a future re-analysis of this ticker — via the promotion
        attempt or On Deck backfill/swap re-check, see user_note_summary in
        src/research/engine.py — carries this context forward, per explicit request
        ("if analisys ever occurs again it needs to have that side note").

        ref_peak was originally sourced from _windowed_dip's dip_summary "peak" field —
        fixed same day (BRO incident, live-caught by the user: "its supoose to not be
        allowed back for a day" but kept reappearing within a minute) after that peak,
        measured strictly BEFORE the tracked low, turned out to be a stale, near-the-
        window-boundary artifact for a STALE low specifically (BRO's own dip_summary peak
        came back $63.95 while the stock was already trading at ~$74.6 — trivially
        "broke out" the instant the block was set). Since a stale low is exactly the
        scenario "AI declined this entry" (and therefore this removal feature) exists for,
        this wasn't a rare edge case — every stale-dip removal was affected. See
        _recent_window_high's docstring for the full incident."""
        ticker = ticker.upper().strip()
        if not ticker or ticker not in self.near_miss_candidates:
            return
        nm = self.near_miss_candidates.pop(ticker, None)
        history_days = self.config["research"].get("on_deck_history_days", 30)
        windowed = _windowed_price_history(nm, history_days) if nm else []
        ref_peak = _recent_window_high(windowed) or (nm.get("last_price") if nm else None)
        if permanent:
            self.on_deck_blocked[ticker] = {"until": None, "ref_peak": ref_peak}
            block_desc = "permanently"
        else:
            try:
                days_n = max(1, int(days))
            except (TypeError, ValueError):
                days_n = 1
            block_until = self._now_et() + timedelta(days=days_n)
            self.on_deck_blocked[ticker] = {"until": block_until.isoformat(), "ref_peak": ref_peak}
            block_desc = f"for {days_n} day(s)"
        if note:
            self.on_deck_notes[ticker] = note
            asyncio.create_task(asyncio.to_thread(_save_on_deck_notes, dict(self.on_deck_notes)))
        asyncio.create_task(asyncio.to_thread(_save_on_deck_blocked, dict(self.on_deck_blocked)))
        asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))
        # initiated_by (2026-07-30, On Deck auto-removal feature) -- distinguishes the
        # user's own ✕-button click from the automated stale-decline eviction below so
        # the ai_log line attributes the removal correctly instead of always claiming
        # "by user" for something the system did on its own.
        removed_by_desc = "by user" if initiated_by == "user" else "automatically by AI"
        entry = self.add_ai_log(ticker, "ON_DECK",
            f"Removed from On Deck {removed_by_desc} — blocked {block_desc}"
            f"{' (will auto-restore early if price breaks out)' if ref_peak else ''}", "warning")
        await self.broadcast({"type": "ai_log", "entry": entry})
        await self.broadcast({"type": "on_deck_removed", "ticker": ticker})

    async def _fetch_price_history(
        self, ticker: str, current_price: float | None = None,
    ) -> list[tuple[float, float]]:
        """Real intraday bars (15-minute by default, falling back to hourly/daily for longer
        windows — see _yf_interval_and_period_for_days) covering the currently-configured
        on_deck_history_days window (default 30). Read from config at call time, not
        hardcoded, so raising the setting (e.g. to 60) takes effect on the very next candidate
        added or restart-catch-up, not just for future live monitor ticks. Called the moment a
        candidate is added to On Deck (2026-07-18) so its R/R chart and window-low recovery
        check both start with a real, richly-detailed multi-week picture instead of an empty
        chart that only fills in over the following days/weeks of live 60s ticks. Switched
        from once-a-day closes to 15-minute bars the same day (~25x more real points for the
        same 30-day window) per explicit request — daily closes made for a very sparse chart.
        The bar fetch fails open on error (falls back to an empty history rather than
        raising) — callers already treat an empty/short price_history as the pre-existing
        "nothing yet" state. If current_price is available even when the fetch itself fails,
        it's still appended below, so a fetch error yields one real point instead of none.

        current_price, when given, is appended as one final (now, price) sample — fixing a
        real bug found live (2026-07-18): yfinance's own daily close and whatever quote
        Claude's analysis captured (seeded into last_price/rr/the R/R badge) are two
        independent real prices for the same day, and they can simply disagree by a real
        amount ($1+, not just rounding noise) — NOT because either one is stale (re-verified
        directly: yfinance's last bar was correctly the most recent trading day's close, not
        lagged). The R/R badge and the chart's own rightmost point disagreed as a result
        ($74.73 → R/R 2.08 badge vs. $73.38 → R/R 2.49 chart) even though both were genuinely
        "current" by their own source. Appending current_price guarantees the chart's
        rightmost point always matches whatever price the rest of the candidate (last_price,
        rr, the badge) was actually computed from, so the two views of "now" can't disagree
        regardless of which of the two real sources happens to differ on a given day."""
        history_days = self.config["research"].get("on_deck_history_days", 30)
        period, interval = _yf_interval_and_period_for_days(history_days)
        try:
            hist = await self.market_data.get_historical(ticker, period=period, interval=interval)
        except Exception as e:
            logger.debug("%s: price history backfill failed: %s", ticker, e)
            hist = []
        out: list[tuple[float, float]] = []
        for row in hist:
            try:
                out.append((row["timestamp"], float(row["close"])))
            except Exception:
                continue
        if current_price is not None and current_price > 0:
            out.append((self._now_et().timestamp(), current_price))
        return out

    async def _backfill_near_miss_from_cache(self) -> int:
        """Repopulate near_miss_candidates from the restored research_reports cache after a
        restart — near_miss_candidates is pure in-memory and doesn't otherwise survive one.
        Only considers today's reports (a fair_value_estimate from days ago isn't a reliable
        basis — the world's moved on) that clear conviction/signal and aren't already held.
        Includes every qualifying stock regardless of R/R (2026-07-17 — On Deck is the sole
        buy list now, not just R/R-rejects; already-qualifying stocks belong here too, sorted
        to the top by the dashboard). Returns the count added, for a startup log line.

        Cap-aware (fixed 2026-07-27, live complaint: "every time the server is reset" the
        AI Research Engine feed floods with ~20 identical "Dropped from On Deck — over the cap"
        lines) -- previously added EVERY qualifying candidate here unconditionally, then
        relied on the separate _enforce_on_deck_cap() trim step to immediately drop most
        of them back out again on every single restart, re-fetching price_history and
        re-logging the same drops for candidates that were never going to survive anyway.
        Now scores and ranks candidates with the same composite _on_deck_candidate_score
        the trim step itself uses, and only actually adds (and fetches price_history for)
        however many slots are genuinely open -- so a normal restart with On Deck already
        at full strength adds nothing and logs nothing, instead of overshooting and
        immediately correcting itself every time."""
        min_conviction = self.config["research"]["min_conviction_score"]
        base_rr = self.config["research"]["min_risk_reward_ratio"]
        floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
        ceiling_margin = self.config["research"].get("on_deck_rr_ceiling_margin", 0.15)
        rr_step = self.config["research"].get("on_deck_rr_conviction_step", 0.1)
        rr_floor = self.config["research"].get("on_deck_rr_floor", 1.5)
        # Same population floor as the live Phase 2 fill (_run_pre_open_batch) — a candidate
        # in the conviction-watch band should survive a restart the same way it would survive
        # a normal pre-open cycle, not vanish just because the process happened to bounce.
        conviction_band = self.config["research"].get("on_deck_conviction_band", 0.0)
        population_floor = _on_deck_population_floor(min_conviction, conviction_band)
        today_str = self._now_et().strftime("%Y-%m-%d")
        held = set(self.portfolio.positions.keys())

        candidates: list[tuple[str, dict]] = []
        for ticker, d in self.research_reports.items():
            if (ticker in held or ticker in self.near_miss_candidates
                    or self._is_on_deck_blocked(ticker) or self._wash_sale_blocked(ticker)):
                continue
            if not d.get("generated_at", "").startswith(today_str):
                continue
            if d.get("signal") not in ("BUY", "STRONG BUY"):
                continue
            conviction = d.get("conviction", 0)
            if conviction < population_floor:
                continue
            entry_price = d.get("entry_price", 0.0)
            stop_loss = d.get("stop_loss", 0.0)
            fair_value = d.get("fair_value_estimate", 0.0)
            if entry_price <= 0 or stop_loss <= 0 or fair_value <= 0:
                continue
            risk = entry_price - stop_loss
            if risk <= 0:
                continue
            rr = (fair_value - entry_price) / risk
            required_rr = _required_rr(conviction, min_conviction, base_rr, rr_step, rr_floor)
            if _on_deck_rr_floor_not_met(rr, required_rr, floor_margin):
                continue
            if _on_deck_rr_ceiling_exceeded(rr, required_rr, ceiling_margin):
                # Above its own gate by more than the small tolerance margin -- mechanical
                # exclude, no AI call (2026-08-05, owner design; ceiling margin added
                # 2026-08-20). The AI-judgment exception is reserved for a candidate that
                # was actually watched rising past its own gate while tracked (the
                # On-Shore backfill path specifically) -- this restore path has no such
                # track record for any given ticker, so it gets no exception. A candidate
                # only just above its own gate (within ceiling_margin) is still admitted
                # here -- see _on_deck_rr_ceiling_exceeded's docstring for why.
                continue
            entry = {
                "ticker": ticker,
                "company_name": d.get("company_name", ticker),
                "sector": d.get("sector", ""),
                "business_summary": d.get("business_summary", ""),
                "thesis": d.get("thesis", ""),
                "signal": d.get("signal", ""),
                "conviction_score": conviction,
                "fair_value_estimate": fair_value,
                "margin_of_safety_pct": d.get("margin_of_safety_pct", 0.0),
                "last_price": entry_price,
                "rr": rr,
                "required_rr": required_rr,
                "stop_loss_pct": _derive_stop_pct(
                    entry_price, stop_loss, self.config["take_profit"]["stop_loss_pct"]),
                "direction": None,
                "streak": 0,
                "ai_entry_price": None,
                "ai_entry_low_ref": None,
                "ai_entry_reasoning": "",
                "ai_entry_seen_below": False,
                "ai_entry_pending": False,
                "added_at": self._now_et().isoformat(),
            }
            candidates.append((ticker, entry))

        max_size = self.config["research"].get("on_deck_max_size", 0)
        slots_open = (max_size - len(self.near_miss_candidates)) if max_size else len(candidates)
        if slots_open <= 0:
            return 0
        # Tiered ranking (2026-07-31, XRAY-adjacent fix) -- see
        # _on_deck_ranking_key's docstring: a buy-eligible candidate must always fill a
        # slot before a watch-only one, regardless of composite score.
        candidates.sort(key=lambda kv: self._on_deck_ranking_key_for(kv[1]), reverse=True)
        to_add = candidates[:slots_open]

        added = 0
        for ticker, entry in to_add:
            # Real trailing-~30-day daily closes (2026-07-18), anchored with entry_price
            # as the final point so the chart's rightmost sample always matches this same
            # entry_price the rest of the candidate was just computed from (fixes a real
            # bug — see _fetch_price_history's docstring). Only fetched for candidates
            # actually being kept, not the ones about to be capped out.
            entry["price_history"] = await self._fetch_price_history(ticker, entry["last_price"])
            self.near_miss_candidates[ticker] = entry
            added += 1
        return added

    async def watchlist_rr_loop(self):
        """Free (yfinance, no Claude) live R/R refresh for watchlist cards. The watchlist R/R
        badge previously used the cached pre-open entry_price, which can drift far from the
        real value over the course of a trading day (caught live 2026-07-16: GPN showed R/R
        3.14 on its stale pre-open price while its actual live-price R/R was ~1.40) — this is
        purely a display feed to keep that badge honest; it does not affect any buy/promotion
        decision, since every actual buy path already fetches its own live price at the moment
        it decides, independent of this loop."""
        while True:
            await asyncio.sleep(60)
            try:
                if self.paused or self.stopped or not self._is_market_open():
                    continue
                tickers = [s["ticker"] for s in self.watchlist_manager.get_active()]
                if not tickers:
                    continue
                updates: dict[str, float] = {}
                for ticker in tickers:
                    dd = self.deep_dive_reports.get(ticker)
                    if not dd:
                        continue
                    fair_value = dd.get("fair_value_estimate", 0.0)
                    stop_loss = dd.get("stop_loss", 0.0)
                    if not fair_value or not stop_loss:
                        continue
                    try:
                        quote = await self.market_data.get_quote(ticker)
                        price = quote.price
                    except Exception as e:
                        logger.debug("%s: watchlist R/R live price fetch failed: %s", ticker, e)
                        continue
                    if price <= 0 or price <= stop_loss:
                        continue
                    updates[ticker] = (fair_value - price) / (price - stop_loss)
                if updates:
                    await self.broadcast({"type": "watchlist_rr_update", "rr": updates})
            except Exception as e:
                logger.error("watchlist_rr_loop error: %s", e)

    async def _check_price_based_unblocks(self) -> None:
        """Free (one quote per blocked ticker, no Claude) re-eligibility check for manually
        removed On Deck candidates (2026-07-29, FNB discussion) -- user pushback on the
        removal dialog's blunt N-days/permanent choice: "can we use no cost price
        adjustments to see if its out of the long dip?" A ticker with a captured ref_peak
        (see remove_on_deck_candidate) is restored to eligibility the moment its price
        clears that peak by research.on_deck_block_breakout_pct, regardless of whether its
        time-based block (temporary OR permanent) has expired yet -- a real breakout is a
        more meaningful signal than a blind clock either way. The persistent note in
        on_deck_notes is deliberately left untouched here (see its own docstring) so a
        future re-analysis still carries the context forward even though the block itself
        just cleared."""
        if not self.on_deck_blocked:
            return
        breakout_pct = self.config["research"].get("on_deck_block_breakout_pct", 2.0)
        for ticker, raw in list(self.on_deck_blocked.items()):
            entry = _normalize_block_entry(raw)
            ref_peak = entry["ref_peak"]
            if ref_peak is None:
                continue
            try:
                quote = await self.market_data.get_quote(ticker)
            except Exception:
                continue
            if _price_clears_block_breakout(quote.price, ref_peak, breakout_pct):
                self.on_deck_blocked.pop(ticker, None)
                asyncio.create_task(asyncio.to_thread(_save_on_deck_blocked, dict(self.on_deck_blocked)))
                log_entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Auto-restored to On Deck eligibility — price (${quote.price:.2f}) broke "
                    f"out above its stale-dip reference peak (${ref_peak:.2f}) by "
                    f"{breakout_pct:.1f}%+", "success")
                await self.broadcast({"type": "ai_log", "entry": log_entry})

    async def near_miss_monitor_loop(self):
        """Free (yfinance, no Claude) price monitor for near-miss candidates — BUY-signal,
        conviction-qualified stocks rejected at pre-open only for R/R (too expensive relative
        to fair_value_estimate right now). stop is recomputed off the LIVE price each poll
        (price * (1 - stop_loss_pct)), not held at the pre-open level — otherwise, as price
        drops toward a fixed stale stop, (price - stop) shrinks toward zero and R/R would
        blow up or go negative for the wrong reason. Because stop trails price proportionally,
        R/R rising as price drops is the intended, correct behavior (more margin of safety),
        not a bug.

        Promotes on R/R clearing the gate AND a confirmed entry-price recovery, using one of
        two research.on_deck_entry_mode strategies (2026-07-18):

        - "retracement": current price has recovered at least research.on_deck_retracement_pct
          (default 20%) of the dip's own depth, measured from the previous peak (the highest
          price in the window before the low) down to the low. Replaced an earlier flat
          %-off-the-low check per the user's own reasoning: a flat percent asks the same
          absolute bounce regardless of how far a stock actually fell, so a deep,
          still-crashing stock could clear it on a trivial bounce while a shallow dip needed a
          comparatively large one. Measuring the bounce as a % of the dip itself scales
          naturally to each stock's own move (same idea as a technical-analysis retracement
          level, e.g. Fibonacci retracements).
        - "ai": once a real dip has started recovering (2+ consecutive upticks off the low),
          fires a ONE-TIME Claude call (research_engine.recommend_dip_entry, via
          _compute_ai_dip_entry) with the actual observed peak/low/current price and asks for
          a specific recommended entry price, then promotes once price reaches that level.
          Per the user's own reasoning: Claude can't meaningfully predict a pullback entry
          before the dip has happened — a recommendation is only informed once there's a real
          peak, low, and recovery already visible. The result is cached against the low it was
          computed from (ai_entry_low_ref) and invalidated (recomputed on the next qualifying
          uptick) if a new, deeper low forms afterward, making the old number stale.

        Both modes share the same underlying peak/low/depth math (src.research.rr_curve.
        dip_summary) — "retracement" uses its retracement_target directly; "ai" only uses it
        to find peak/low/depth to hand to Claude and to detect when the low has moved (staling
        the cached recommendation).

        price_history persists across pre-open re-vets and restarts (changed 2026-07-18, was
        previously wiped every day/restart) and is now kept in full forever, never trimmed
        (changed again same day — "we want to keep it all") — the chart shows a candidate's
        complete history for as long as it's been on On Deck. The BUY-TRIGGER decision still
        only looks within the rolling on_deck_history_days window (default 30), via a
        windowed slice built fresh each tick rather than by deleting anything from storage —
        a large, slower-moving stock can genuinely take multiple days or weeks to bottom and
        recover, so anchoring "the low" to a single trading day was too short a memory, but a
        months-old low shouldn't keep anchoring a real buy decision long after it's stopped
        being a meaningful reference point for the candidate's current fair_value_estimate,
        even though it's still worth keeping around to look at. Polling every 60s keeps
        detection fast (free) without inflating Finnhub/yfinance load."""
        while True:
            await asyncio.sleep(60)
            try:
                # Win/Loss dashboard stat refresh (2026-07-29) -- deliberately BEFORE the
                # market-hours gate below, unlike the On Deck monitoring this loop is
                # otherwise dedicated to: a viewer checking the dashboard after close or
                # over a weekend shouldn't see a stale win rate just because nothing else
                # in this loop runs outside market hours.
                await self._refresh_win_rate_cache()

                # "Recent Sell" post-mortem queue (2026-08-21) -- also deliberately
                # before the market-hours gate, same reasoning as the win-rate refresh
                # above: a position can close (and need its post-mortem generated)
                # outside market hours (e.g. a stop firing right at the close), and the
                # delayed follow-up check has nothing to do with whether the market is
                # open right now either.
                await self._process_sell_analysis_queue()

                if self.paused or self.stopped or not self._is_market_open():
                    continue

                # Free price-based On Deck re-eligibility check (2026-07-29) -- runs before
                # backfill so a ticker that clears its breakout this same tick can
                # immediately participate in backfill/persist-check again, rather than
                # waiting a full extra 60s cycle. See _check_price_based_unblocks.
                await self._check_price_based_unblocks()

                # Runs even when near_miss_candidates is empty (2026-07-23) -- that's exactly
                # the scenario backfill exists for; placed BEFORE the emptiness check below so
                # it isn't skipped. See _backfill_on_deck_from_on_shore's own docstring.
                await self._backfill_on_deck_from_on_shore()
                if not self.near_miss_candidates:
                    continue

                base_rr = self.config["research"]["min_risk_reward_ratio"]
                min_conviction = self.config["research"]["min_conviction_score"]
                rr_step = self.config["research"].get("on_deck_rr_conviction_step", 0.1)
                rr_floor = self.config["research"].get("on_deck_rr_floor", 1.5)
                # Whole-percentage units (e.g. 5.0 for 5%), matching the convention every
                # other consumer of stop_loss_pct/_derive_stop_pct uses (see
                # /api/today-scan-rejects' identical `stop_pct / 100` at point of use below) —
                # NOT divided by 100 here. Previously divided here, which silently corrupted
                # every R/R calculation whenever a real per-candidate nm["stop_loss_pct"]
                # (whole-percentage) overrode this default via the .get() below, producing a
                # deeply negative stop price and a near-zero R/R that always failed the gate.
                default_stop_pct = self.config["take_profit"]["stop_loss_pct"]
                history_window_secs = self.config["research"].get("on_deck_history_days", 30) * 86400
                now_ts = datetime.now().timestamp()
                floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
                to_promote: list[tuple[str, float]] = []
                to_evict_stale: list[str] = []
                to_evict_rr_floor: list[tuple[str, float, float]] = []
                to_evict_above_gate: list[tuple[str, float, float, str]] = []

                for ticker, nm in list(self.near_miss_candidates.items()):
                    if ticker in self.portfolio.positions:
                        self.near_miss_candidates.pop(ticker, None)
                        continue
                    try:
                        quote = await self.market_data.get_quote(ticker)
                        price = quote.price
                    except Exception as e:
                        logger.debug("%s: near-miss price fetch failed: %s", ticker, e)
                        continue
                    if price <= 0:
                        continue

                    prev_price = nm.get("last_price")
                    nm["price_history"].append((now_ts, price))
                    # price_history is kept in full forever now (changed 2026-07-18, per
                    # explicit request — "don't delete data when it becomes older than 30,
                    # we want to keep it all") — nothing is ever trimmed from the stored
                    # list, so the chart shows the candidate's complete history for as long
                    # as it's stayed on On Deck. The on_deck_history_days window still
                    # matters for the BUY-TRIGGER decision specifically (see the windowed
                    # slice built just before the dip_summary() call below) — a months-old
                    # low shouldn't keep anchoring a real buy decision long after it's
                    # stopped being a relevant reference point, even though it's still worth
                    # keeping around to look at. Storage growth is accepted knowingly: ~390
                    # points/trading day kept indefinitely is roughly 2-3MB/year per
                    # candidate — real but trivial at the scale of a handful of On Deck
                    # candidates at a time, especially since most are bought or removed
                    # within days/weeks rather than staying on the list for years.
                    nm["last_price"] = price

                    # Direction + consecutive-tick streak — same up/down arrow concept as
                    # positions' price_direction, plus a streak count so the dashboard can
                    # show visual momentum building toward the 2-tick uptick confirmation.
                    if prev_price is not None and prev_price > 0 and price != prev_price:
                        new_direction = "up" if price > prev_price else "down"
                        nm["streak"] = (nm.get("streak", 0) + 1
                                         if new_direction == nm.get("direction") else 1)
                        nm["direction"] = new_direction
                        # Rolling window of recent tick directions (2026-07-21) — backs the
                        # no-dip up-tick-count trigger below, a second, independent measure
                        # from the streak above: "how many of the last N changes were up"
                        # rather than "how many in a row right now." A single down-tick barely
                        # moves this count the way it fully resets the streak, so it catches a
                        # steady, low-volatility grind that never produces a long same-direction
                        # streak or a large % price move, without needing either of those.
                        up_ratio_window = self.config["research"].get("on_deck_up_ratio_window", 10)
                        recent = nm.setdefault("recent_directions", [])
                        recent.append(new_direction)
                        if len(recent) > up_ratio_window:
                            del recent[:len(recent) - up_ratio_window]

                    # Per-candidate stop %, not the one global config value (2026-07-18) —
                    # derived once from Claude's own stop_loss recommendation at analysis
                    # time (see _candidate_entry/_persist_on_result), since the real order
                    # this candidate would eventually place uses report.stop_loss directly,
                    # not a flat mechanical percentage. Still trailed live off the current
                    # price every tick (not held as a fixed dollar stop) for the same
                    # stability reason the mechanical version always was: a fixed absolute
                    # stop would make (price - stop) shrink toward zero as price approaches
                    # it, causing R/R to blow up or go negative for the wrong reason. Falls
                    # back to the global default for any candidate that predates this field.
                    stop_pct = nm.get("stop_loss_pct", default_stop_pct)
                    stop = round(price * (1 - stop_pct / 100), 2)
                    fair_value = nm["fair_value_estimate"]
                    risk = price - stop
                    rr = (fair_value - price) / risk if risk > 0 and fair_value > 0 else -999.0
                    nm["rr"] = rr
                    nm["stop_loss"] = stop
                    # Conviction-scaled gate (2026-07-18), not one flat number for every
                    # stock — see _required_rr's docstring. Stored on the candidate so the
                    # dashboard can show exactly what threshold this specific stock actually
                    # needs to clear, not the flat config base value.
                    min_rr = _required_rr(nm["conviction_score"], min_conviction, base_rr, rr_step, rr_floor)
                    nm["required_rr"] = min_rr

                    # Continuous, free R/R-floor eviction (2026-08-10, owner request) — this
                    # loop already recomputes live R/R every 60s tick for free (no Claude
                    # call, just the quote already fetched above), but until now only the
                    # twice-daily persist-check swept out a candidate whose R/R had fallen
                    # below its own floor; between those sweeps a candidate could sit on the
                    # dashboard for hours showing a badly stale, well-below-floor number
                    # (confirmed live: COF/COP/CVX all sitting at R/R 1.0-1.2 against a
                    # ~1.6-1.7 floor mid-morning, hours before the next scheduled sweep would
                    # have caught them). Deferred into to_evict_rr_floor and processed after
                    # the loop, same reason to_promote/to_evict_stale are — avoid mutating
                    # near_miss_candidates while other tickers are still being processed in
                    # this same pass. Uses the identical _on_deck_rr_floor_not_met check the
                    # persist-check sweep already uses, just evaluated far more often since
                    # it costs nothing extra to run.
                    if _on_deck_rr_floor_not_met(rr, min_rr, floor_margin):
                        to_evict_rr_floor.append((ticker, rr, min_rr))
                        continue

                    if rr < min_rr:
                        continue

                    # Continuous above-gate AI re-judgment (2026-08-18, owner request) --
                    # mirrors the persist-check retention sweep's own above-gate check
                    # (_on_deck_ai_gate_above_gate) below, just running every tick instead of
                    # only 2x/day. Cooldown-gated (on_deck_above_gate_recheck_cooldown_minutes),
                    # unlike the floor-side eviction above -- this one IS a real, billed Claude
                    # call, so re-firing it every single tick for a candidate that just sits
                    # above gate for hours would be real, avoidable spend for a judgment that
                    # barely changes tick to tick. A "still good" verdict (or a not-yet-due
                    # cooldown) falls through to the entry-price trigger logic below exactly as
                    # before -- this only ever short-circuits the loop on eviction.
                    #
                    # Also checks _on_deck_backfill_above_gate_cooldown, not just this path's
                    # own dict (fixed 2026-08-18, cost audit) -- this path and the On Shore
                    # backfill path (_try_add_inner below) ask the identical real question
                    # ("is this candidate, now above its own gate, still a good buy") but
                    # previously tracked their cooldowns in two entirely separate dicts. If
                    # this loop evicted a candidate for being above gate, it could land back
                    # on On Shore and get immediately re-asked the same just-answered question
                    # through the backfill path's own cooldown, which had no record of this
                    # loop's very recent ask. Now stores a cooldown-UNTIL timestamp in
                    # tz-aware ET (self._now_et()), matching the backfill dict's own
                    # representation, instead of the previous naive-local "last checked" time
                    # -- comparing a naive and a tz-aware datetime directly raises TypeError,
                    # so unifying the representation was required to safely check both dicts
                    # together, not just a style preference.
                    if _on_deck_rr_above_gate(rr, min_rr):
                        cooldown_min = self.config["research"].get(
                            "on_deck_above_gate_recheck_cooldown_minutes", 20)
                        now_et = self._now_et()
                        due = not (
                            _on_deck_cooldown_active(self._on_deck_above_gate_cooldown, ticker, now_et)
                            or _on_deck_cooldown_active(
                                self._on_deck_backfill_above_gate_cooldown, ticker, now_et))
                        if due:
                            self._on_deck_above_gate_cooldown[ticker] = now_et + timedelta(minutes=cooldown_min)
                            still_good_buy, reasoning = await self._on_deck_ai_gate_above_gate(
                                ticker=ticker, company_name=nm.get("company_name", ""),
                                thesis=nm.get("thesis", ""), price=price,
                                fair_value_estimate=nm["fair_value_estimate"], stop_loss=stop,
                                rr=rr, required_rr=min_rr, conviction_score=nm["conviction_score"],
                                fail_default=True,
                            )
                            # Re-fetch guard (same pattern as _compute_ai_dip_entry and the
                            # persist-check sweep's own above-gate check): a concurrent
                            # promotion/removal could have popped this candidate while the
                            # real Claude call above was in flight.
                            if ticker not in self.near_miss_candidates:
                                continue
                            if not still_good_buy:
                                to_evict_above_gate.append((ticker, rr, min_rr, reasoning))
                                continue

                    # Entry-price check (2026-07-18) — see this method's docstring for the
                    # full "retracement" vs "ai" design. Both modes start from the same shared
                    # peak/low/depth math (dip_summary, src/research/rr_curve.py) rather than a
                    # second copy of it — that same function backs the card-level "here's
                    # what's happening" summary in /api/near-miss, and a second inline copy
                    # here would risk the two drifting out of sync with each other over future
                    # edits.
                    retracement_pct = self.config["research"].get("on_deck_retracement_pct", 20.0)
                    # Windowed slice, not the full (now-permanent) price_history — the buy
                    # decision should only look as far back as on_deck_history_days, even
                    # though the stored history itself is kept forever for display. Built
                    # fresh each tick, never mutates nm["price_history"] itself.
                    windowed_history = [p for p in nm["price_history"] if p[0] >= now_ts - history_window_secs]
                    dip = dip_summary(windowed_history, retracement_pct)
                    if dip is None:
                        # No measurable peak-to-low dip in this window -- the candidate has
                        # simply been trending the whole time, with no pullback to confirm a
                        # recovery from. Second, independent trigger (2026-07-21, user-approved
                        # after reviewing VLY's case): R/R already enforces "don't overpay" on
                        # its own (measured against fair_value_estimate regardless of
                        # dip/no-dip), so the dip-recovery path's real extra job -- confirming
                        # price has actually stopped falling before buying -- has nothing to
                        # add for a stock that's already been rising the whole time it's been
                        # tracked. A large enough % gain off the window's own low is treated as
                        # a real sustained-uptrend signal here.
                        #
                        # Measured as a % gain, not a consecutive-uptick streak (same-day
                        # revision, 2026-07-21) -- a streak-based version was tried first and
                        # rejected: a single down tick resets a streak counter to zero
                        # regardless of how strong the trend was building beforehand (e.g. 9
                        # straight up-ticks, one down, then 8 more up only reaches streak 8,
                        # never re-crossing a 10-tick bar), which punishes ordinary noise as
                        # harshly as a real reversal. % gain off the window low doesn't have
                        # this problem -- one noisy down tick barely moves the number, so
                        # genuine sustained progress isn't wiped out by it.
                        pct_gain_ok = False
                        low_entry = min(windowed_history, key=lambda p: p[1], default=None)
                        # Staleness guard (2026-07-29, IVZ incident) -- reuses
                        # _dip_low_too_stale, originally the mechanical retracement mode's
                        # own fix for the identical flaw (RRC/OVV, 2026-07-28): measuring
                        # "gain" off the single lowest price anywhere in the full 30-day
                        # window, with no check on how old that low actually is, lets a
                        # stock that's been genuinely declining for days still pass this
                        # check just because it hasn't yet fallen all the way back down to
                        # some much older low from weeks ago. Confirmed live: IVZ bought
                        # while still in a real multi-day decline (a Dip Recovery attempt
                        # for it had failed 2 days earlier) because today's price was still
                        # above a stale 30-day window low.
                        if (low_entry is not None
                                and not _dip_low_too_stale(
                                    low_entry[0], now_ts,
                                    self.config["research"].get("on_deck_max_dip_low_age_days", 14.0))):
                            window_low = low_entry[1]
                            if window_low > 0:
                                pct_gain = (price - window_low) / window_low * 100
                                # Stored on the candidate (not just a local variable) so a
                                # snapshot popped for this attempt carries the exact value that
                                # triggered it — the finally block below needs it to record
                                # no_dip_failed_at_pct_gain on a failed attempt.
                                nm["no_dip_pct_gain"] = round(pct_gain, 2)
                                no_dip_pct_gain = self.config["research"].get("on_deck_no_dip_pct_gain", 10.0)
                                # Dedup mirrors the dip-path's promotion_failed_low: retry only once
                                # price has risen further past whatever gain the last failed
                                # attempt saw (real new information), not just because the gain is
                                # still sitting at the same already-tried level.
                                pct_gain_ok = (pct_gain >= no_dip_pct_gain
                                               and pct_gain > nm.get("no_dip_failed_at_pct_gain", 0.0))

                        # Second, independent no-dip trigger (2026-07-21, user's own design):
                        # a raw count of upticks out of the last N changes, e.g. 7 out of a
                        # window of 10 — catches a steady, low-volatility grind that may never
                        # rack up a large % price gain (the trigger above) but is still a real,
                        # confirmed uptrend by this different measure. Either trigger alone is
                        # sufficient; both are independently useful for different price shapes.
                        up_ratio_ok = False
                        recent = nm.get("recent_directions", [])
                        up_ratio_window = self.config["research"].get("on_deck_up_ratio_window", 10)
                        if len(recent) >= up_ratio_window:
                            up_count = sum(1 for d in recent if d == "up")
                            nm["no_dip_up_count"] = up_count
                            up_ticks_needed = self.config["research"].get("on_deck_up_ticks_needed", 7)
                            up_ratio_ok = (up_count >= up_ticks_needed
                                           and up_count > nm.get("no_dip_failed_at_up_count", 0))

                        if pct_gain_ok or up_ratio_ok:
                            to_promote.append((ticker, None))
                        continue

                    # Don't re-attempt promotion for a dip that already had a failed attempt
                    # (2026-07-20) — the trigger condition below (price vs a stored entry
                    # level) has nothing to do with WHY a promotion actually fails (conviction,
                    # signal, R/R on fresh data) — only price. Retrying purely because price is
                    # still elevated would re-run the same fresh Claude check against a
                    # situation that hasn't meaningfully changed, indefinitely. Cleared the
                    # moment either a genuinely new (deeper) low forms, OR price has recovered
                    # meaningfully further past the level where the last attempt failed
                    # (2026-07-23 — see _dip_recovery_dedup_cleared for why a plain equality
                    # check on the low alone left a recovering candidate stuck for hours), or
                    # the next scheduled persist-check refreshes conviction and resets the
                    # AI-entry state anyway (see _persist_on_result).
                    recovery_retry_pct = self.config["research"].get("on_deck_recovery_retry_pct", 2.0)
                    if not _dip_recovery_dedup_cleared(
                            nm.get("promotion_failed_low"), dip["low"],
                            nm.get("promotion_failed_at_price"), price, recovery_retry_pct):
                        continue

                    entry_mode = self.config["research"].get("on_deck_entry_mode", "ai")
                    if entry_mode == "ai":
                        # Fire a one-time recommendation once a real recovery off the low is
                        # visible (2+ consecutive upticks) and there isn't already a current
                        # one for THIS dip low (a MEANINGFULLY deeper low invalidates any
                        # prior recommendation, since it was reasoned from a now-outdated
                        # bottom -- see _dip_low_changed_meaningfully for why this isn't a
                        # plain != comparison: on_deck_ai_entry_low_refresh_pct filters out
                        # ordinary noise-level drift in the tracked low so it doesn't burn a
                        # real Claude call for essentially the same answer every cycle).
                        refresh_pct = self.config["research"].get(
                            "on_deck_ai_entry_low_refresh_pct", 1.0)
                        low_changed = _dip_low_changed_meaningfully(
                            nm.get("ai_entry_low_ref"), dip["low"], refresh_pct)
                        # Skip re-asking Claude about a low it has already judged
                        # genuinely stale (2026-07-31, BRO repeat-decline incident) --
                        # see _save_on_deck_stale_dip_low's docstring. The in-memory
                        # ai_entry_low_ref guard above is wiped the moment a real stale
                        # verdict's auto-eviction removes this candidate's whole
                        # near_miss_candidates entry, so it can't protect against a
                        # repeat ask once the candidate is restored -- this separate,
                        # never-auto-cleared store is what actually survives that cycle.
                        # Reuses the same "genuinely deeper low" threshold as low_changed
                        # above; a low that's still essentially the one Claude already
                        # declined is treated as still-stale for free, mirroring exactly
                        # what a real repeat call would have said.
                        remembered_stale_low = self.on_deck_stale_dip_low.get(ticker)
                        still_known_stale = (
                            remembered_stale_low is not None
                            and not _dip_low_changed_meaningfully(
                                remembered_stale_low, dip["low"], refresh_pct))
                        if still_known_stale:
                            to_evict_stale.append(ticker)
                            continue
                        if (nm.get("direction") == "up" and nm.get("streak", 0) >= 2
                                and not nm.get("ai_entry_pending")
                                and low_changed):
                            nm["ai_entry_pending"] = True
                            asyncio.create_task(self._compute_ai_dip_entry(ticker))
                        if low_changed or nm.get("ai_entry_price") is None:
                            continue  # no current AI recommendation for this dip yet
                        # Must genuinely rise UP THROUGH the entry, not merely already sit at
                        # or above it (2026-07-28, DV incident) -- see _ai_entry_trigger's
                        # docstring for the full incident writeup, including the 2026-08-04
                        # arm_band_pct addition that widens "below" to "within this % above."
                        arm_band_pct = self.config["research"].get("on_deck_ai_entry_arm_band_pct", 2.0)
                        should_promote, nm["ai_entry_seen_below"] = _ai_entry_trigger(
                            price, nm["ai_entry_price"], nm.get("ai_entry_seen_below", False), arm_band_pct)
                        if not should_promote:
                            continue  # hasn't genuinely reached the AI-recommended entry yet
                    else:
                        # Mechanical mode has no Claude call to judge staleness the way "ai"
                        # mode now does (2026-07-28, RRC/OVV incident) -- a hardcoded recency
                        # floor on the low itself is the only fix available here. See
                        # _dip_low_too_stale's docstring.
                        max_low_age_days = self.config["research"].get(
                            "on_deck_max_dip_low_age_days", 14.0)
                        if _dip_low_too_stale(dip["low_t"], now_ts, max_low_age_days):
                            continue  # low is too old to represent a genuine, current dip
                        if price < dip["retracement_target"]:
                            continue  # hasn't retraced enough of the dip yet

                    to_promote.append((ticker, dip["low"]))

                for ticker, dip_low in to_promote:
                    nm_snapshot = self.near_miss_candidates.pop(ticker, None)
                    asyncio.create_task(
                        self._attempt_near_miss_promotion(ticker, nm_snapshot, dip_low))

                if to_promote:
                    # Keep the persisted cache in sync so a restart right after a promotion
                    # doesn't resurrect a ticker that's already been bought (or attempted).
                    asyncio.create_task(
                        asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

                # Continuous R/R-floor eviction (2026-08-10) -- deferred out of the main loop
                # above for the same reason to_promote/to_evict_stale are: avoid mutating
                # near_miss_candidates while other tickers are still being read from it in
                # this same pass. Mirrors the persist-check sweep's own eviction message
                # exactly (same wording, same "warning" level) so the two are indistinguishable
                # in the log except for how quickly each one can catch a given candidate.
                for ticker, rr_val, required_rr in to_evict_rr_floor:
                    self.near_miss_candidates.pop(ticker, None)
                    self._mark_universe_reject(ticker)
                    entry = self.add_ai_log(ticker, "ON_DECK",
                        f"Removed from On Deck — R/R {rr_val:.2f} below min R/R floor "
                        f"({required_rr + floor_margin:.2f}, its own gate {required_rr:.2f} "
                        f"+ {floor_margin:.2f} margin)", "warning")
                    asyncio.create_task(self.broadcast({"type": "ai_log", "entry": entry}))
                if to_evict_rr_floor:
                    asyncio.create_task(
                        asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

                # Continuous above-gate AI eviction (2026-08-18) -- deferred out of the main
                # loop above for the same reason every other eviction/promotion list here is:
                # avoid mutating near_miss_candidates while other tickers are still being read
                # from it in this same pass. Mirrors the persist-check sweep's own above-gate
                # eviction message exactly (same wording, same "warning" level).
                for ticker, rr_val, required_rr, reasoning in to_evict_above_gate:
                    self.near_miss_candidates.pop(ticker, None)
                    self._mark_universe_reject(ticker)
                    entry = self.add_ai_log(ticker, "ON_DECK",
                        f"Removed from On Deck — R/R {rr_val:.2f} above its own gate "
                        f"({required_rr:.2f}), AI judged it's no longer a good buy: "
                        f"{reasoning}", "warning")
                    asyncio.create_task(self.broadcast({"type": "ai_log", "entry": entry}))
                if to_evict_above_gate:
                    asyncio.create_task(
                        asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

                # Free re-eviction of a candidate still sitting on a previously-declined
                # stale low (2026-07-31) -- same consequence a real repeat Claude call
                # would have produced, without spending it. Deferred out of the main loop
                # above, same reason to_promote is: avoid mutating near_miss_candidates
                # (remove_on_deck_candidate pops it) while iterating over it.
                block_days = self.config["research"].get("on_deck_ai_stale_decline_block_days", 1)
                for ticker in to_evict_stale:
                    asyncio.create_task(self.remove_on_deck_candidate(
                        ticker, permanent=False, days=block_days,
                        note="Still the same previously-declined stale dip low — skipped re-asking AI",
                        initiated_by="ai",
                    ))
            except Exception as e:
                logger.error("near_miss_monitor_loop error: %s", e)

    async def _compute_ai_dip_entry(self, ticker: str) -> None:
        """Fires once near_miss_monitor_loop sees a real dip that's started recovering (2+
        consecutive upticks off the low) when research.on_deck_entry_mode == "ai" — gives
        Claude the actual observed peak/low/current price and asks for a recommended entry,
        rather than asking it to predict a pullback level before any dip has happened (it
        can't meaningfully do that with no real price action to reason about yet — see the
        2026-07-18 discussion in CLAUDE.md). Runs as a background task, not awaited by the
        monitor loop itself, so one in-flight Claude call never blocks that loop's 60s tick
        for every other candidate."""
        nm = self.near_miss_candidates.get(ticker)
        if nm is None:
            return
        try:
            retracement_pct = self.config["research"].get("on_deck_retracement_pct", 20.0)
            # Same windowed slice near_miss_monitor_loop's trigger check uses (not the full,
            # now-permanent price_history) — this call only ever fires because that check
            # already found a windowed dip, so it must agree on the same peak/low here.
            history_window_secs = self.config["research"].get("on_deck_history_days", 30) * 86400
            now_ts = datetime.now().timestamp()
            windowed_history = [p for p in nm["price_history"] if p[0] >= now_ts - history_window_secs]
            dip = dip_summary(windowed_history, retracement_pct)
            if dip is None:
                return
            # How long ago the peak/low actually happened (2026-07-28, RRC/OVV incident) --
            # lets Claude judge for itself whether this is a genuine, current dip or just
            # wherever a long trend happened to start, instead of the old prompt's three
            # bare prices with no sense of time at all. See recommend_dip_entry's docstring
            # for the full incident writeup.
            peak_days_ago = (now_ts - dip["peak_t"]) / 86400
            low_days_ago = (now_ts - dip["low_t"]) / 86400
            # Market cap (2026-08-04) -- lets the staleness judgment below calibrate by
            # company size instead of applying one generic day-cutoff instinct to every
            # stock. Best-effort: a fetch failure just omits the context, same as an
            # unknown/zero market cap already does inside recommend_dip_entry itself.
            try:
                market_cap = (await self.research_engine.market_data.get_financials(ticker)).market_cap
            except Exception:
                market_cap = None
            result = await self.research_engine.recommend_dip_entry(
                ticker, nm.get("company_name", ticker), nm.get("thesis", ""),
                nm.get("fair_value_estimate", 0.0), dip["peak"], dip["low"],
                nm.get("last_price", 0.0), peak_days_ago, low_days_ago, market_cap,
            )
            # Re-fetch: the candidate may have been bought, manually removed, or re-analyzed
            # (resetting these same fields) while this call was in flight.
            nm = self.near_miss_candidates.get(ticker)
            if nm is None:
                return
            if result is None:
                entry = self.add_ai_log(ticker, "ON_DECK",
                    "AI dip-entry recommendation unavailable — will retry on next confirmed uptick",
                    "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                return
            entry_price, reasoning, stale = result
            if entry_price is None:
                # Claude explicitly declined -- a real, reasoned "no" (stale low, not a
                # genuine current uptrend), not a failure. Record ai_entry_low_ref anyway so
                # _dip_low_changed_meaningfully doesn't treat the still-missing ai_entry_price
                # as "no recommendation exists yet" and immediately re-fire a fresh (billed)
                # call on the very next tick for the exact same low -- same wasteful-refire
                # class of bug already fixed once for this feature (see that function's
                # ALLY docstring). ai_entry_price deliberately stays unset, which keeps the
                # promotion trigger's own "no current AI recommendation" check blocking a
                # buy on this dip, same as an outright failure would.
                nm["ai_entry_low_ref"] = dip["low"]
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"AI declined this entry — {reasoning}", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                if stale:
                    # Genuinely stale reference point (2026-07-30, user request: a
                    # candidate AI keeps declining on the same old, never-going-to-be-
                    # fresh-again low was supposed to give way to a fresh candidate, not
                    # sit on On Deck taking up a slot indefinitely -- confirmed live for
                    # OKE/OVV, both declining daily on a ~29-day-old already-recovered
                    # low with no automatic consequence). This is Claude's own judgment
                    # (recommend_dip_entry's "stale" field), not a hardcoded proxy --
                    # consistent with this project's standing preference for trusting
                    # real AI judgment over mechanical rules (see CLAUDE.md). Reuses the
                    # exact same manual ✕-button mechanism (remove_on_deck_candidate) --
                    # same free price-based auto-restore if price genuinely breaks out
                    # later, same persisted note -- just triggered automatically instead
                    # of requiring the user to notice and click it themselves. A decline
                    # that's merely "too early" (stale=False) is deliberately left alone;
                    # that candidate is still legitimately developing and this whole
                    # monitoring loop exists to keep watching it.
                    # Persist the declined low separately from the block itself
                    # (2026-07-31, BRO repeat-decline incident) -- the removal below
                    # deletes this candidate's whole near_miss_candidates entry,
                    # including ai_entry_low_ref just set above, so THAT guard can't
                    # survive to protect against a repeat ask once this ticker is
                    # restored. See _save_on_deck_stale_dip_low's docstring.
                    self.on_deck_stale_dip_low[ticker] = dip["low"]
                    asyncio.create_task(asyncio.to_thread(
                        _save_on_deck_stale_dip_low, dict(self.on_deck_stale_dip_low)))
                    block_days = self.config["research"].get(
                        "on_deck_ai_stale_decline_block_days", 1)
                    await self.remove_on_deck_candidate(
                        ticker, permanent=False, days=block_days,
                        note=f"AI auto-removed (stale dip): {reasoning}",
                        initiated_by="ai",
                    )
                    return
                asyncio.create_task(
                    asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))
                return
            nm["ai_entry_price"] = entry_price
            nm["ai_entry_low_ref"] = dip["low"]
            nm["ai_entry_reasoning"] = reasoning
            # A fresh, valid (non-stale) recommendation supersedes any earlier stale
            # verdict remembered for this ticker (2026-07-31) -- the situation has
            # genuinely moved past whatever old low that memory was protecting against.
            if self.on_deck_stale_dip_low.pop(ticker, None) is not None:
                asyncio.create_task(asyncio.to_thread(
                    _save_on_deck_stale_dip_low, dict(self.on_deck_stale_dip_low)))
            # Whether the promotion trigger already starts "armed" (2026-07-28, DV incident)
            # -- Claude is asked for a good entry PRICE, not necessarily one above the
            # current price; it commonly recommends a support level BELOW where price has
            # already recovered to (DV: recommended $10.65 with price already at $11.27).
            # The trigger below requires a genuine rise up through ai_entry_price, so if
            # current price is already at/above the recommendation, that hasn't happened yet
            # -- starts unarmed, requiring a real pullback back below entry first. Only
            # starts pre-armed when price is already below the recommendation, matching the
            # ordinary "waiting for it to rise" case this feature was originally designed for.
            # arm_band_pct (2026-08-04) widens "below" to "within this % above" -- see
            # _ai_entry_initially_armed/_ai_entry_trigger's docstrings for the full reasoning.
            arm_band_pct = self.config["research"].get("on_deck_ai_entry_arm_band_pct", 2.0)
            nm["ai_entry_seen_below"] = _ai_entry_initially_armed(
                nm.get("last_price", 0.0), entry_price, arm_band_pct)
            entry = self.add_ai_log(ticker, "ON_DECK",
                f"AI recommended entry ${entry_price:.2f} — {reasoning}", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            asyncio.create_task(
                asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))
        finally:
            nm2 = self.near_miss_candidates.get(ticker)
            if nm2 is not None:
                nm2["ai_entry_pending"] = False

    def _record_promotion_attempt(
        self, ticker: str, dip_low: float | None, outcome: str,
        conviction: float | None = None, rr: float | None = None, price: float | None = None,
    ) -> None:
        """One entry per real _attempt_near_miss_promotion call, win or lose (2026-07-21).
        Fire-and-forget cache save, same pattern as every other small JSON cache in this
        file. Capped at 100 (most-recent-first) -- this is a diagnostic log, not a permanent
        trading record (that's trade_history), so unbounded growth isn't worth guarding
        against with anything fancier than a simple cap."""
        entry = {
            "ticker": ticker,
            "timestamp": self._now_et().isoformat(),
            "trigger": "Dip Recovery" if dip_low is not None else "No-Dip Uptrend",
            "outcome": outcome,
            "conviction": conviction,
            "rr": round(rr, 2) if rr is not None else None,
            "price": round(price, 2) if price is not None else None,
        }
        self.promotion_attempts.insert(0, entry)
        del self.promotion_attempts[100:]
        asyncio.create_task(
            asyncio.to_thread(_save_promotion_attempts, list(self.promotion_attempts)))
        asyncio.create_task(
            self.broadcast({"type": "promotion_attempt", "entry": entry}))

    async def _attempt_near_miss_promotion(
        self, ticker: str, nm_snapshot: dict | None = None, dip_low: float | None = None,
    ):
        """Fired when a candidate clears R/R + a confirmed uptick. Runs one fresh Claude
        re-analysis (pre-open data can be hours old) and, if it still passes every normal buy
        gate, executes the buy immediately — the confirmation moment itself is the entry
        signal; waiting for anything else would let it go stale before being acted on. This is
        now the sole buy path (2026-07-17) — if the portfolio is full, attempts a rotation
        swap against the weakest current holding (same pattern _buy_from_watchlist_by_price
        used to use) rather than skipping outright.

        `nm_snapshot` (2026-07-20) — the candidate's dict as it was the instant this attempt
        fired, popped from near_miss_candidates by the caller before this coroutine started.
        If the attempt fails for ANY reason short of an actual buy, this dict is restored back
        onto near_miss_candidates in the `finally` block below — a failed attempt (conviction
        just missed 7.0, signal slipped to HOLD, R/R fell back below gate, a rejected order)
        doesn't mean the stock is bad, just that this specific attempt didn't land; the same
        conviction-collapse-below-3.0 removal floor everywhere else in this codebase already
        governs whether a genuinely weak stock eventually leaves the list, via the next
        persist-check.

        `dip_low` — the peak-to-low dip's low price that triggered THIS attempt, or None if
        this attempt came from the no-dip sustained-uptrend trigger instead (2026-07-21 — see
        near_miss_monitor_loop). On failure, a real dip_low is stored as
        nm["promotion_failed_low"] so near_miss_monitor_loop won't re-trigger another attempt
        for the same dip (the trigger condition is price-based and has nothing to do with why
        a promotion actually fails — conviction, signal, R/R — so retrying purely because
        price is still elevated would just re-run the same check against an unchanged
        real-world situation, indefinitely). Becomes eligible again the moment either a
        genuinely new, deeper low forms, or the next persist-check refreshes conviction and
        resets the AI-entry state anyway. A None dip_low means this attempt came from one of
        the two no-dip triggers instead (see near_miss_monitor_loop) — on failure, both
        nm["no_dip_failed_at_pct_gain"] and nm["no_dip_failed_at_up_count"] are refreshed to
        their current values, same idea as promotion_failed_low but keyed on either measure
        growing further past that point rather than a new low forming."""
        if ticker in self.portfolio.positions:
            return
        bought = False
        reserved_amount = 0.0
        evicted = False
        try:
            if not self.config["trading"].get("auto_execute", False) or not self.broker_connected:
                return
            # Auto-buy cutoff, now a real Settings value (2026-07-20, moved from a hardcoded
            # 2:00 PM) — read fresh from config each call rather than parsed once at startup,
            # matching the pattern most other settings already use, so a change takes effect
            # immediately without a restart. Read live, not cached: this loop already re-reads
            # config every tick for other values (min_rr, retracement_pct, etc.).
            _cutoff_str = self.config["research"].get("auto_buy_cutoff_time", "14:00")
            _ch, _cm = _cutoff_str.split(":")
            if self._now_et().time() >= dtime(int(_ch), int(_cm)):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"R/R + uptick confirmed but past {_cutoff_str} ET cutoff — skipping", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(ticker, dip_low, f"Past {_cutoff_str} ET cutoff")
                return

            # Wash-sale rebuy block (2026-07-27) — pure in-memory dict lookup (see
            # Portfolio.recent_losses), so this runs before the cash pre-check below and
            # well before the real Claude re-analysis: no point paying for either if this
            # ticker is going to be rejected regardless. check_all_rules (used by the
            # manual-buy and rotation-swap paths) applies the identical gate via
            # RiskManager.check_wash_sale_cooldown — this is the equivalent check for the
            # automated On Deck promotion path specifically, which doesn't route through
            # check_all_rules.
            if not self.risk_manager.check_wash_sale_cooldown(ticker, self.portfolio):
                _last_loss = self.portfolio.recent_losses.get(ticker)
                _cd_days = self.risk_manager.wash_sale_cooldown_days
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Skipped — wash-sale cooldown active (lost money on this ticker "
                    f"{_last_loss.date() if _last_loss else '?'}, {_cd_days}-day rebuy block)", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(ticker, dip_low, "Wash-sale cooldown active")
                # Evict from On Deck too (2026-07-28, user request) -- this ticker just proved
                # it can't legally be bought for the rest of the cooldown, so it shouldn't keep
                # sitting on the dashboard showing a live "Buy" badge. Every population site
                # that could otherwise re-add it (_wash_sale_blocked) is also guarded, so this
                # eviction sticks until the cooldown actually expires.
                self.near_miss_candidates.pop(ticker, None)
                # evicted=True (fixed 2026-08-02, GitHub #41) -- without this, the shared
                # finally block below can't tell this eviction apart from an ordinary
                # "attempt failed, keep watching it" case, and was putting the candidate
                # straight back into near_miss_candidates before this coroutine even finished.
                evicted = True
                asyncio.create_task(
                    asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))
                return

            # Cheap cash pre-check (2026-07-22) — using the candidate's own cached
            # last-known price/stop (not yet the fresh re-analysis below), BEFORE paying for
            # a real Claude call that would be pointless if there's clearly not enough cash
            # regardless of what the fresh numbers turn out to be. Deliberately approximate:
            # entry/stop can shift a little once the real re-analysis lands, so this only
            # rejects a CLEAR shortfall against current data, not a marginal one — the exact,
            # authoritative cash check below (using the real post-analysis numbers) still
            # runs and is still what actually gates the order. Skipped entirely if the cached
            # price/stop aren't available (a brand-new or stale-data candidate) rather than
            # guessing.
            approx_price = nm_snapshot.get("last_price") if nm_snapshot else None
            approx_stop = nm_snapshot.get("stop_loss") if nm_snapshot else None
            if approx_price and approx_stop and approx_price > 0 and approx_stop > 0:
                approx_size = self.risk_manager.calculate_position_size(
                    approx_price, approx_stop, self.portfolio.total_value)
                approx_reserve = self.portfolio.total_value * (
                    self.config["risk_management"]["min_cash_reserve_pct"] / 100)
                if self.portfolio.cash - approx_size < approx_reserve:
                    entry = self.add_ai_log(ticker, "ON_DECK",
                        "Insufficient cash reserve (pre-check against cached price) — "
                        "skipping before spending on a fresh AI re-analysis", "warning")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    self._record_promotion_attempt(
                        ticker, dip_low, "Insufficient cash reserve (pre-check)")
                    return

            # Earnings blackout (fixed 2026-08-08, GitHub #54) -- CLAUDE.md has documented
            # this as a hard auto-buy block since _earnings_soon() was first built, but its
            # only 2 call sites (the pre-2026-07-17 watchlist auto-buy path, and the
            # WS execute_buy handler behind a frontend function with zero real callers) were
            # both unreachable dead code -- this is the sole live buy path and never checked
            # it at all. Placed here, before the real Claude re-analysis call, for the same
            # reason as the cash pre-check just above: no point paying for a re-analysis on a
            # ticker that's about to be blocked regardless. Fails open on a data error (see
            # _earnings_soon's own docstring) -- never blocks a buy just because the earnings
            # calendar lookup itself failed.
            earnings_soon, earnings_date = await self._earnings_soon(ticker)
            if earnings_soon:
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Skipped — earnings on {earnings_date}, within the auto-buy blackout window",
                    "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"Earnings blackout ({earnings_date})")
                return

            entry = self.add_ai_log(ticker, "ON_DECK",
                "R/R recovered + confirmed uptick — running fresh analysis...", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

            try:
                trade_history_summary = await self.portfolio.get_trade_history_summary(ticker)
                analysis_history_summary = await self.portfolio.get_analysis_history_summary(ticker)
                report = await self.research_engine.analyze_stock(
                    ticker, trade_history_summary, self.on_deck_notes.get(ticker, ""),
                    analysis_history_summary=analysis_history_summary)
            except Exception as e:
                entry = self.add_ai_log(ticker, "ON_DECK", f"Re-analysis failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(ticker, dip_low, f"Re-analysis failed: {e}")
                return

            min_conviction = self.config["research"]["min_conviction_score"]
            base_rr = self.config["research"]["min_risk_reward_ratio"]

            def _refresh_nm_from_report(rr_val=None, required_rr_val=None):
                # 2026-07-28, user request: a failed promotion attempt used to discard the
                # fresh report it just paid for, leaving the On Deck card showing whatever
                # stale conviction/R/R/fair_value it had BEFORE this re-analysis -- so a
                # candidate that had genuinely fallen off (like NEE repeatedly failing R/R)
                # kept looking as strong as it did originally, sorted near the top by the
                # same stale numbers, and kept attracting re-analysis every time price
                # ticked up 2%. Writing the real numbers back here means the existing
                # composite scoring (_on_deck_candidate_score) sorts it to where it actually
                # belongs on the very next render, same as a real persist-check does.
                if nm_snapshot is None:
                    return
                nm_snapshot["company_name"] = report.company_name
                nm_snapshot["sector"] = getattr(report, "sector", "")
                nm_snapshot["business_summary"] = getattr(report, "business_summary", "")
                nm_snapshot["thesis"] = report.thesis
                nm_snapshot["signal"] = report.signal.value
                nm_snapshot["conviction_score"] = report.conviction_score
                nm_snapshot["fair_value_estimate"] = report.fair_value_estimate
                nm_snapshot["margin_of_safety_pct"] = report.margin_of_safety_pct
                if report.entry_price > 0:
                    nm_snapshot["last_price"] = report.entry_price
                if report.entry_price > 0 and report.stop_loss > 0:
                    nm_snapshot["stop_loss_pct"] = _derive_stop_pct(
                        report.entry_price, report.stop_loss,
                        self.config["take_profit"]["stop_loss_pct"])
                if rr_val is not None:
                    nm_snapshot["rr"] = rr_val
                if required_rr_val is not None:
                    nm_snapshot["required_rr"] = required_rr_val
                # A full re-analysis invalidates the old AI-entry recommendation (it was
                # reasoned from the pre-re-analysis fair_value/thesis) -- reset the same way
                # a real persist-check does, forcing a fresh one once a new uptick streak is
                # observed rather than keeping a now-stale entry price/reasoning displayed.
                nm_snapshot["direction"] = None
                nm_snapshot["streak"] = 0
                nm_snapshot["ai_entry_price"] = None
                nm_snapshot["ai_entry_low_ref"] = None
                nm_snapshot["ai_entry_reasoning"] = ""
                nm_snapshot["ai_entry_seen_below"] = False

            if (getattr(report, "is_fallback", True)
                    or report.signal.value not in ("BUY", "STRONG BUY")
                    or report.conviction_score < min_conviction
                    or report.entry_price <= 0 or report.stop_loss <= 0):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"No longer qualifies on fresh data — {report.signal.value} | "
                    f"Conviction {report.conviction_score}/10", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"No longer qualifies — {report.signal.value}",
                    conviction=report.conviction_score, price=report.entry_price)
                if not getattr(report, "is_fallback", True):
                    _refresh_nm_from_report()
                return

            risk = report.entry_price - report.stop_loss
            fair_value = report.fair_value_estimate
            rr = (fair_value - report.entry_price) / risk if risk > 0 and fair_value > 0 else 0.0
            rr_step = self.config["research"].get("on_deck_rr_conviction_step", 0.1)
            rr_floor = self.config["research"].get("on_deck_rr_floor", 1.5)
            min_rr = _required_rr(report.conviction_score, min_conviction, base_rr, rr_step, rr_floor)
            if rr < min_rr:
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"R/R fell back below gate on fresh data — {rr:.2f} < {min_rr:.2f}", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"R/R fell below gate ({rr:.2f} < {min_rr:.2f})",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            # Above-gate ambiguity check at the actual buy trigger (2026-08-13, owner
            # design after the OXY incident) -- clearing the gate here can mean a genuine
            # opportunity, or the same ambiguity _on_deck_ai_gate_above_gate already
            # judges for retention/backfill: a tight stop sitting close to price
            # mechanically inflating the ratio without the setup actually improving. That
            # judgment already ran on OXY itself twice the same morning (both declines,
            # at persist-check and at an On Shore backfill check) but never at the actual
            # moment of spending real money -- the highest-stakes place to ask it.
            # fail_default=False: a call failure blocks the buy rather than letting a
            # missing AI judgment wave one through, same AI Data Integrity principle as
            # every other real trading figure in this codebase. Not an eviction on "no" --
            # treated the same as any other gate failure in this function (this specific
            # attempt didn't land, the candidate keeps watching); _refresh_nm_from_report
            # resets the uptick streak so it can't immediately re-fire on the same
            # unchanged setup next tick.
            if _on_deck_rr_above_gate(rr, min_rr):
                still_good_buy, reasoning = await self._on_deck_ai_gate_above_gate(
                    ticker=ticker, company_name=report.company_name, thesis=report.thesis,
                    price=report.entry_price, fair_value_estimate=fair_value,
                    stop_loss=report.stop_loss, rr=rr, required_rr=min_rr,
                    conviction_score=report.conviction_score, fail_default=False,
                )
                if not still_good_buy:
                    entry = self.add_ai_log(ticker, "ON_DECK",
                        f"R/R {rr:.2f} above its own gate ({min_rr:.2f}) at the buy trigger "
                        f"— AI judged it's not a genuine opportunity: {reasoning}", "warning")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    self._record_promotion_attempt(
                        ticker, dip_low,
                        f"R/R above gate, AI declined at buy trigger ({rr:.2f} > {min_rr:.2f})",
                        conviction=report.conviction_score, rr=rr, price=report.entry_price)
                    _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                    return

            if ticker in self.portfolio.positions:  # re-check post-await race
                return

            # max_positions is now purely the sizing guideline used by calculate_position_size
            # (below) — it no longer gates whether a new buy can happen at all. Previously,
            # hitting the headcount cap forced a rotation-swap attempt (sell the weakest
            # current holding) before any buy could proceed, and abandoned the buy entirely
            # if no weak-enough holding qualified to swap — even when real deployable cash
            # already existed (e.g. from partial T1/T2 proceeds sitting idle). Agreed
            # 2026-07-20, built 2026-07-21: the real constraint is cash above the reserve
            # floor, already checked correctly below (required_reserve) — that's now the
            # sole gate. Position count can grow past max_positions as long as cash allows;
            # no automatic ceiling is added on top of that by design (user manages headcount
            # manually via max_positions if the real count drifts too far past the intended
            # average, since they're actively watching the dashboard).

            if not self.risk_manager.check_daily_loss(self.portfolio):
                entry = self.add_ai_log(ticker, "ON_DECK", "Daily loss limit reached — skipping", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, "Daily loss limit reached",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return
            dd_state = self.risk_manager.check_drawdown(self.portfolio)
            if dd_state in ("halt", "exit_review", "defensive"):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Portfolio in {dd_state} state — skipping ({self._drawdown_diagnostic()})", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"Portfolio in {dd_state} state",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            # Sector concentration (fixed 2026-08-02, GitHub #42) -- check_all_rules already
            # enforces this on the manual-buy and rotation-swap paths, but this function (the
            # sole automated On Deck promotion path since 2026-07-17) never called it, so the
            # user-enabled sector_concentration_enabled/max_sector_positions cap silently did
            # nothing here despite being turned on in Settings.
            sector = getattr(report, "sector", "")
            if not self.risk_manager.check_sector_concentration(self.portfolio, sector):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Sector concentration limit reached ({sector}) — skipping", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"Sector concentration limit reached ({sector})",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            position_size = self.risk_manager.calculate_position_size(
                report.entry_price, report.stop_loss, self.portfolio.total_value)
            shares = position_size / report.entry_price if report.entry_price > 0 else 0
            if shares < 0.001:
                entry = self.add_ai_log(ticker, "ON_DECK", "Position size too small — skipping", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, "Position size too small",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            required_reserve = self.portfolio.total_value * (
                self.config["risk_management"]["min_cash_reserve_pct"] / 100)
            # Lock scope is deliberately tight -- just the shared-state check-then-reserve,
            # not the failure-path logging/broadcast below (which await and don't need to
            # be serialized against other concurrent promotion attempts).
            async with self._promotion_cash_lock:
                insufficient = (self.portfolio.cash - self._reserved_cash - position_size
                                < required_reserve)
                if not insufficient:
                    self._reserved_cash += position_size
                    reserved_amount = position_size
            if insufficient:
                entry = self.add_ai_log(ticker, "ON_DECK", "Insufficient cash reserve — skipping", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, "Insufficient cash reserve",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            targets = (list(report.take_profit_targets) if report.take_profit_targets else [
                round(report.entry_price * (1 + self.config["take_profit"]["t1_pct"] / 100), 2),
                round(report.entry_price * (1 + self.config["take_profit"]["t2_pct"] / 100), 2),
                round(report.entry_price * (1 + self.config["take_profit"]["t3_pct"] / 100), 2),
            ])

            from src.decision.signal_generator import TradeSignal
            from src.research.engine import Signal as Sig
            sig_enum = Sig.STRONG_BUY if report.signal.value == "STRONG BUY" else Sig.BUY
            signal = TradeSignal(
                ticker=ticker, signal=sig_enum, conviction=report.conviction_score,
                entry_price=report.entry_price, stop_loss=report.stop_loss,
                take_profit_targets=targets,
                position_size_pct=(report.position_size_pct
                                    or self.config["risk_management"].get("starting_position_pct", 3.0)),
                position_size_dollars=position_size, shares=shares,
                reasoning=f"On Deck Deploy — R/R recovered to {rr:.2f} with confirmed uptick",
                # research_report used to be hardcoded None here (2026-08-21 fix) --
                # this is the sole live buy path, and the real, fresh `report` this
                # whole function already re-analyzed with is right here in scope; the
                # old None discarded the AI's real thesis/reasoning for every single
                # live buy, the exact data "Why AI Bought This" (Position.buy_thesis
                # etc.) needs. rr/required_rr are the exact numbers that just cleared
                # the gate a few lines above, not a later reconstruction.
                research_report=report, generated_at=datetime.now(), should_execute=True,
                sector=getattr(report, "sector", ""),
                rr=rr, required_rr=min_rr,
            )

            try:
                order = await self.order_manager.execute(signal)
            except Exception as e:
                entry = self.add_ai_log(ticker, "ON_DECK", f"Buy failed: {e}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"Buy failed: {e}",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
                return

            if order and order.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                bought = True
                # Log the real fill price, not the AI's pre-trade recommendation
                # (2026-08-08, GitHub #53) -- this is the primary live buy path (On Deck
                # promotion), so this specific call site is the highest-value fix of the
                # 12 in this batch.
                if order.filled_price is not None:
                    signal.entry_price = order.filled_price
                self.trade_logger.log_trade(signal, is_paper=getattr(self.order_manager.broker, "paper", True))
                result = {"ticker": ticker, "status": order.status.value,
                          "filled_price": order.filled_price, "shares": shares}
                await self.broadcast({"type": "trade_executed", "trade": result})
                await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
                _fp = order.filled_price or report.entry_price
                # Distinct phase tag for the actual buy-execution line (2026-07-21, a
                # backlog item from 2026-07-16 finally done) -- ON_DECK_DEPLOY, separate
                # from the generic ON_DECK tag every other recommendation/uptick/rejection
                # entry in this function uses, so the real buy moment stands out in the
                # AI Research Engine feed instead of blending into routine On Deck chatter.
                entry = self.add_ai_log(ticker, "ON_DECK_DEPLOY",
                    f"BUY (On Deck Deploy) {shares:.4g} shares @ ${_fp:.2f} | "
                    f"Conviction {report.conviction_score}/10 | R/R {rr:.2f} | "
                    f"Stop ${report.stop_loss:.2f}", "buy")
                await self.broadcast({"type": "ai_log", "entry": entry})
                asyncio.create_task(_notify(
                    f"BUY {ticker} (On Deck Deploy)",
                    f"{shares:.4g} shares @ ${_fp:.2f} | R/R recovered to {rr:.2f} after confirmed uptick",
                    priority="high", tags="white_check_mark"))
                logger.info("On Deck Deploy buy %s — %.4g shares @ $%.2f (R/R %.2f)",
                            ticker, shares, _fp, rr)
                self._record_promotion_attempt(
                    ticker, dip_low, "Bought",
                    conviction=report.conviction_score, rr=rr, price=_fp)
            elif order:
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"Buy rejected by broker: {order.status.value}", "error")
                await self.broadcast({"type": "ai_log", "entry": entry})
                self._record_promotion_attempt(
                    ticker, dip_low, f"Rejected by broker: {order.status.value}",
                    conviction=report.conviction_score, rr=rr, price=report.entry_price)
                _refresh_nm_from_report(rr_val=rr, required_rr_val=min_rr)
        finally:
            if reserved_amount:
                self._reserved_cash -= reserved_amount
            if bought and nm_snapshot is not None:
                # Capture whatever AI-entry recommendation/reasoning actually triggered THIS
                # attempt (2026-07-20) -- visible on the On Deck card right up until the
                # moment of purchase, otherwise lost the instant the ticker leaves
                # near_miss_candidates with nothing in the position detail view explaining
                # why it was bought. Retracement-mode promotions have no ai_entry_price/
                # reasoning to capture (both None/"" on the snapshot) -- stored anyway so the
                # position detail view can at least show "no AI entry reasoning recorded"
                # accurately rather than the key being entirely absent.
                self.buy_reasoning[ticker] = {
                    "ai_entry_price": nm_snapshot.get("ai_entry_price"),
                    "ai_entry_reasoning": nm_snapshot.get("ai_entry_reasoning", ""),
                    "entry_mode": self.config["research"].get("on_deck_entry_mode", "ai"),
                    "recorded_at": self._now_et().isoformat(),
                }
                asyncio.create_task(
                    asyncio.to_thread(_save_buy_reasoning, dict(self.buy_reasoning)))
            elif (not bought and not evicted and nm_snapshot is not None
                    and ticker not in self.portfolio.positions):
                nm_snapshot["ai_entry_pending"] = False
                if dip_low is not None:
                    nm_snapshot["promotion_failed_low"] = dip_low
                    # Price at the moment of THIS failed attempt (2026-07-23) -- lets the
                    # dedup guard require price to recover meaningfully further past this
                    # exact level before retrying, not just stay stuck forever whenever the
                    # dip's own low happens to never print a new value. See
                    # _dip_recovery_dedup_cleared.
                    nm_snapshot["promotion_failed_at_price"] = nm_snapshot.get("last_price")
                else:
                    # Both no-dip triggers are independently evaluated every tick regardless
                    # of which one actually caused this attempt (see near_miss_monitor_loop),
                    # so both get their dedup value refreshed on a failed attempt — whichever
                    # one(s) were past their bar at this moment are now suppressed until they
                    # grow further, same as if only one had fired.
                    nm_snapshot["no_dip_failed_at_pct_gain"] = nm_snapshot.get("no_dip_pct_gain", 0.0)
                    nm_snapshot["no_dip_failed_at_up_count"] = nm_snapshot.get("no_dip_up_count", 0)
                self.near_miss_candidates[ticker] = nm_snapshot

    async def _capture_daily_performance_snapshot(self):
        """One entry per trading day: portfolio value + the 3 major index closes, so a
        single bad day (like today's real 0.27% vs SPY's 0.86%) can be told apart from a
        genuine multi-week pattern (2026-07-21, user's own concern). Fired once per weekday
        from the same trigger point as _generate_daily_report, a few minutes after market
        close so the day's real closing prices are available. Read-only against live
        quotes, never touches trading state -- wrapped in its own broad try/except so a
        failure here can never affect the daily report or anything else on that trigger."""
        try:
            today_str = self._now_et().strftime("%Y-%m-%d")
            spy = await self.market_data.get_quote("SPY")
            qqq = await self.market_data.get_quote("QQQ")
            dia = await self.market_data.get_quote("DIA")
            entry = {
                "date": today_str,
                "portfolio_value": round(self.portfolio.total_value, 2),
                "spy_close": round(spy.price, 2) if spy and spy.price else None,
                "qqq_close": round(qqq.price, 2) if qqq and qqq.price else None,
                "dia_close": round(dia.price, 2) if dia and dia.price else None,
            }
            # Replace any existing entry for today rather than accumulating duplicates --
            # this can legitimately fire more than once for the same date if the process
            # restarts after the trigger already fired earlier that day.
            self.performance_history = [
                p for p in self.performance_history if p["date"] != today_str]
            self.performance_history.append(entry)
            asyncio.create_task(
                asyncio.to_thread(_save_performance_history, list(self.performance_history)))
            logger.info("Captured daily performance snapshot for %s: portfolio $%.2f, "
                        "SPY $%s, QQQ $%s, DIA $%s",
                        today_str, entry["portfolio_value"], entry["spy_close"],
                        entry["qqq_close"], entry["dia_close"])
        except Exception as e:
            logger.warning("Failed to capture daily performance snapshot: %s", e)

    async def _capture_benchmark_snapshot(self):
        """Extends _capture_daily_performance_snapshot (same trigger, same
        few-minutes-after-close timing) with today's composition-weighted
        benchmark row. See docs/superpowers/specs/2026-07-29-composition-weighted-benchmark-design.md.
        Wrapped in its own broad try/except, same as the sibling snapshot --
        a failure here must never block the daily report or anything else."""
        try:
            from src.analytics.benchmark_store import BenchmarkStore, classify_ticker
            from src.analytics.composition_benchmark import has_real_close

            db_path = self.config.get("database", {}).get("path", "data/aitrading.db")
            store = BenchmarkStore(db_path)
            store.initialize()

            today_str = self._now_et().strftime("%Y-%m-%d")
            settled = store.get_settled_days()
            provisional = store.get_provisional_dates()
            if today_str in settled and today_str not in provisional:
                return  # already captured today for real -- restart re-fired this trigger
            # A provisional today (rare here -- only if an ETF's prev-day close
            # wasn't published yet even by evening) is recomputed, same
            # self-healing behavior as the backfill script's reprocessing.

            holdings_value = _live_holdings_value()
            if not holdings_value:
                return

            sp500 = set(get_universe(["S&P 500"]))
            sp400 = set(get_universe(["S&P 400"]))
            sp600 = set(get_universe(["S&P 600"]))

            def get_sector(ticker):
                pos = self.portfolio.positions.get(ticker)
                return pos.sector if pos and pos.sector else None

            def get_cap_tier_membership():
                return (sp500, sp400, sp600)

            classifications = {
                t: classify_ticker(t, store, get_sector, get_cap_tier_membership)
                for t in holdings_value
            }

            sector_etfs = {c[0] for c in classifications.values() if c[0]}
            cap_tier_etfs = {c[1] for c in classifications.values()}
            etf_daily_returns = {}
            today_is_provisional = False
            for etf in sector_etfs | cap_tier_etfs:
                quote = await self.market_data.get_quote(etf)
                history = await self.market_data.get_historical(etf, period="5d", interval="1d")
                real_prev_close = (
                    len(history) >= 2 and has_real_close(etf, history[-2]["date"],
                        {etf: {row["date"]: row["close"] for row in history}})
                )
                if quote and quote.price and real_prev_close:
                    prev_close = history[-2]["close"]
                    etf_daily_returns[etf] = (quote.price / prev_close - 1) if prev_close else 0.0
                else:
                    etf_daily_returns[etf] = 0.0
                    today_is_provisional = True

            day_return = weighted_daily_return(holdings_value, classifications, etf_daily_returns)

            prior_dates = sorted(d for d in settled if d != today_str)
            prior_value = settled[prior_dates[-1]] if prior_dates else 100.0
            new_value = prior_value * (1 + day_return)

            total = sum(holdings_value.values())
            composition = {t: v / total for t, v in holdings_value.items()}
            store.save_settled_day(today_str, new_value, composition, is_provisional=today_is_provisional)
            logger.info("Captured benchmark composition snapshot for %s: index=%.4f%s",
                        today_str, new_value, " [provisional]" if today_is_provisional else "")
        except Exception as e:
            logger.warning("Failed to capture benchmark composition snapshot: %s", e)

    async def _generate_daily_report(self):
        """End-of-day recap — read-only against the DB, one Claude call, one ntfy push.
        Wrapped in a broad try/except so a failure here can never affect trading."""
        try:
            today_str = self._now_et().strftime("%Y-%m-%d")

            trade_file = Path("data/trade_history") / f"{today_str}.jsonl"
            trades = []
            if trade_file.exists():
                for line in trade_file.read_text(encoding="utf-8").splitlines():
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        continue

            import sqlite3 as _sqlite3
            with _sqlite3.connect(self._log_db_path, timeout=_SQLITE_TIMEOUT_SECS) as conn:
                rows = conn.execute(
                    "SELECT timestamp, ticker, phase, content, level FROM ai_log "
                    "WHERE date(created_at) = date('now', 'localtime') "
                    # 'buy'/'sell' added (2026-07-22) -- the actual trade-fill confirmation
                    # log lines (e.g. "Position closed at broker — stop/TP filled...") use
                    # these levels for dashboard color-coding, not 'success'. The old filter
                    # silently excluded them, so the recap could only ever see an earlier
                    # "pending fill" (level='warning') entry for a trade and never its real
                    # resolution -- confirmed live 2026-07-22 when GEN's recap line quoted
                    # a stale "pending fill" status hours after it had actually filled.
                    "AND level IN ('error', 'warning', 'success', 'buy', 'sell') "
                    "ORDER BY id ASC LIMIT 200"
                ).fetchall()
            notable = [
                f"{r[0]} {r[1]} {r[2]} ({r[4]}): {r[3]}" for r in rows
                if "Quick screen passed" not in r[3] and not r[3].startswith("Not added")
            ][:80]

            total_value = self.portfolio.cash + sum(
                p.shares * p.current_price for p in self.portfolio.positions.values()
            )
            day_start = self.portfolio.day_start_value
            day_pnl = total_value - day_start if day_start else 0.0
            day_pnl_pct = (day_pnl / day_start * 100) if day_start else 0.0

            trades_summary = "\n".join(
                f"- {t.get('signal')} {t.get('ticker')} — {t.get('shares')} sh @ "
                f"${t.get('entry_price', 0):.2f}"
                for t in trades
            ) or "No trades executed today."

            log_summary = "\n".join(notable) or "No notable errors or warnings."

            prompt = f"""Summarize today's automated stock trading activity in 3-5 sentences, \
plain English, for a push notification. Be concise and flag anything that looks wrong or \
needs attention.

DATE: {today_str}
PORTFOLIO: ${total_value:,.2f} total, day P&L ${day_pnl:+,.2f} ({day_pnl_pct:+.1f}%), \
{len(self.portfolio.positions)} open positions

TRADES EXECUTED TODAY:
{trades_summary}

NOTABLE LOG ENTRIES (errors/warnings/confirmations, chronological):
{log_summary}

IMPORTANT — read entries chronologically before concluding anything is wrong:
- A "PROTECTION GAP" line is only a real, ongoing problem if there's no later "Protection
  gap resolved" line for that same ticker. This system's stop/take-profit orders routinely
  get replaced (at market open, market close, and whenever a trailing stop ratchets), which
  briefly shows as a gap and self-heals within seconds — that is normal, not a critical issue.
  Only call this out as needing attention if a ticker's gap has NO matching resolution
  anywhere later in today's log.
- A trade can show a "pending fill" / "monitoring for fill" line followed later by its real
  outcome (a fill confirmation, or a different error). Always use the LATEST entry for a
  given ticker/order as the true status, never an earlier in-flight one.

Respond with ONLY the summary text, no preamble, no markdown."""

            if not self.research_engine.client:
                logger.warning("Daily report skipped — no ANTHROPIC_API_KEY")
                return

            model = self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5")
            message = await asyncio.to_thread(
                lambda: self.research_engine.client.messages.create(
                    model=model, max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            text_block = next((b for b in message.content if hasattr(b, "text")), None)
            summary = text_block.text.strip() if text_block else "Daily report generation failed — no text in response."

            entry = self.add_ai_log("SYSTEM", "DAILY_REPORT", summary, "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

            await _notify(f"📊 Daily Report — {today_str}", summary,
                          priority="default", tags="bar_chart")
            logger.info("Daily report generated and sent for %s", today_str)
        except Exception as e:
            logger.error("Daily report generation failed: %s", e)

    async def _run_batched_chunk_loop(
        self, chunk_source, on_result, should_stop=None,
        analysis_history_summaries: dict[str, str] | None = None,
    ):
        """Adaptive Batch-API chunk orchestrator shared by pre-open Phase 1 (watchlist
        re-vet — a static pre-sliced chunk source) and Phase 2 (universe fill — a chunk
        source that lazily runs quick_screen to build each ~100-ticker chunk on demand).
        `chunk_source` is an async iterable yielding already-sized `list[str]` chunks.
        Submits the first chunk via research_engine.submit_analysis_batch(). While a
        chunk is in flight, if it has been running longer than
        max(3 minutes, 2x the fastest chunk completed so far this run) and neither
        self.paused, self.stopped, nor the caller's `should_stop()` says to stop, pulls
        and submits the next chunk from `chunk_source` concurrently (capped at 2 chunks
        in flight at once) rather than waiting further — a slow batch doesn't stall the
        whole scan, but a normal-speed run stays sequential with no wasted concurrent
        spend. Calls `on_result(ticker, report)` (async) for every ticker as soon as its
        batch's results are ready, in whatever order batches complete. New chunks stop
        being pulled once `should_stop()` returns True or self.paused/self.stopped is
        set (2026-08-20), but any already
        in-flight batch is always allowed to finish and its results are always
        processed — already-submitted batch work is already paid for and is never
        discarded outright (the caller's own on_result can still choose not to act on a
        given result, e.g. Phase 2 re-checking slots_available() before adding).
        SAFETY NET: if a submitted batch shows zero progress (no succeeded/errored count
        movement) for longer than FALLBACK_TIMEOUT, it's treated as stuck — the batch is
        cancelled and every ticker in that chunk is instead analyzed via the proven
        sequential analyze_stock() path, so a degraded/stalled Batch API queue (observed
        live on 2026-07-15 — see CLAUDE.md) can never leave the pre-open scan hung with
        zero results. Confirmed via direct account history that a real batch has never
        taken longer than ~7 minutes even at 977 requests, so this timeout has generous
        margin above normal variance while still recovering well before market open.

        analysis_history_summaries (2026-08-21, optional, keyed by ticker) is passed
        straight through to submit_analysis_batch/analyze_stock -- only the persist-check
        re-vet (a small chunk of RECURRING On Deck candidates) pre-fetches and passes
        one; the full universe scan doesn't, same cost/relevance-scoped exclusion as
        trade_history_summary. See _build_analysis_history_section's own docstring in
        src/research/engine.py for the full design."""
        should_stop = should_stop or (lambda: False)
        chunk_iter = chunk_source.__aiter__()
        exhausted = False

        async def _next_chunk() -> list[str] | None:
            nonlocal exhausted
            if exhausted:
                return None
            try:
                return await chunk_iter.__anext__()
            except StopAsyncIteration:
                exhausted = True
                return None

        MIN_ADAPTIVE_WAIT = 180.0  # 3 minute floor
        MAX_CONCURRENT = 2
        FALLBACK_TIMEOUT = 600.0  # 10 minutes of genuinely zero progress -> assume stuck
        ABSOLUTE_MAX_WAIT = 1800.0  # 30 min hard cap even if slowly, truly progressing
        fastest_seen: float | None = None
        in_flight: dict[asyncio.Task, float] = {}

        async def _sequential_fallback(chunk: list[str]) -> None:
            _history = analysis_history_summaries or {}
            for ticker in chunk:
                try:
                    report = await self.research_engine.analyze_stock(
                        ticker, analysis_history_summary=_history.get(ticker, ""))
                    await on_result(ticker, report)
                except Exception as e:
                    logger.error("Sequential fallback failed for %s: %s", ticker, e)
                await asyncio.sleep(1)

        async def _run_one_batch(chunk: list[str]) -> float | None:
            batch_id, inputs_by_ticker = await self.research_engine.submit_analysis_batch(
                chunk, analysis_history_summaries=analysis_history_summaries)
            if not batch_id:
                logger.warning(
                    "Batch submission produced no batch_id for a %d-ticker chunk — "
                    "falling back to sequential analysis", len(chunk))
                await _sequential_fallback(chunk)
                return None
            start = time.monotonic()
            # Fixed 2026-07-19 — the old check only compared elapsed time against
            # FALLBACK_TIMEOUT, regardless of whether the batch was actually moving. Confirmed
            # live: a 92-ticker chunk had genuinely reached 57/92 succeeded when this fired and
            # cancelled it anyway, losing the batch discount on the other 35 for no reason —
            # the log even claimed "zero progress" without ever actually checking for movement.
            # Now tracks the sum of terminal-state counts (succeeded+errored+canceled+expired)
            # between polls and only resets the stuck-clock when that number actually grows, so
            # a genuinely frozen batch is still caught in FALLBACK_TIMEOUT, but a slowly-but-
            # truly-progressing one is left alone. ABSOLUTE_MAX_WAIT is a hard backstop so a
            # batch that trickles just fast enough to keep resetting the clock can't stall the
            # whole pre-open scan indefinitely.
            last_progress_ts = start
            last_done_count = 0
            stuck = False
            stuck_reason = ""
            while True:
                await asyncio.sleep(15)
                try:
                    status = await self.research_engine.poll_batch_status(batch_id)
                except Exception as e:
                    logger.warning("Batch poll error for %s: %s", batch_id, e)
                    continue
                if status.processing_status == "ended":
                    break
                counts = status.request_counts
                done_count = counts.succeeded + counts.errored + counts.canceled + counts.expired
                now = time.monotonic()
                if done_count > last_done_count:
                    last_done_count = done_count
                    last_progress_ts = now
                if now - last_progress_ts > FALLBACK_TIMEOUT:
                    stuck_reason = (f"genuinely zero progress for {FALLBACK_TIMEOUT / 60:.0f} min "
                                     f"({counts})")
                elif now - start > ABSOLUTE_MAX_WAIT:
                    stuck_reason = (f"still running after {ABSOLUTE_MAX_WAIT / 60:.0f} min despite "
                                     f"ongoing progress ({counts}) — hard cap reached")
                if stuck_reason:
                    logger.error(
                        "Batch %s %s — cancelling and falling back to sequential "
                        "analyze_stock() for the remaining tickers in this %d-ticker chunk",
                        batch_id, stuck_reason, len(chunk))
                    await self.research_engine.cancel_batch(batch_id)
                    stuck = True
                    break
            if stuck:
                # Whatever already succeeded before the cutoff is real, paid-for work —
                # fetch and process it before falling back, rather than discarding it and
                # redoing those tickers sequentially too. Cancellation isn't instantaneous
                # (the batch transitions through its own "canceling" state before reaching
                # "ended"), and results aren't retrievable until it's genuinely ended — so
                # wait for that first, bounded to 60s, rather than fetching immediately and
                # getting nothing back.
                cancel_wait_start = time.monotonic()
                while time.monotonic() - cancel_wait_start < 60:
                    await asyncio.sleep(5)
                    try:
                        cstatus = await self.research_engine.poll_batch_status(batch_id)
                    except Exception:
                        continue
                    if cstatus.processing_status == "ended":
                        break
                try:
                    reports = await self.research_engine.fetch_batch_results(batch_id, inputs_by_ticker)
                except Exception as e:
                    logger.warning("Fetching partial results for cancelled batch %s failed: %s", batch_id, e)
                    reports = {}
                # fetch_batch_results returns a fallback (is_fallback=True) entry for EVERY
                # ticker in the batch, including the ones that were canceled/errored, not
                # just the truly successful ones — so only a genuinely non-fallback result
                # counts as "done"; a canceled ticker still needs a real sequential retry,
                # not to be silently treated as already handled.
                done_tickers = set()
                for ticker, report in reports.items():
                    if getattr(report, "is_fallback", True):
                        continue
                    done_tickers.add(ticker)
                    try:
                        await on_result(ticker, report)
                    except Exception as e:
                        logger.error("Batch on_result failed for %s: %s", ticker, e)
                remaining = [t for t in chunk if t not in done_tickers]
                if remaining:
                    await _sequential_fallback(remaining)
                return None
            elapsed = time.monotonic() - start
            reports = await self.research_engine.fetch_batch_results(batch_id, inputs_by_ticker)
            for ticker, report in reports.items():
                try:
                    await on_result(ticker, report)
                except Exception as e:
                    logger.error("Batch on_result failed for %s: %s", ticker, e)
            return elapsed

        async def _try_submit_next() -> bool:
            if self.paused or self.stopped or should_stop():
                return False
            chunk = await _next_chunk()
            if not chunk:
                return False
            t = asyncio.create_task(_run_one_batch(chunk))
            in_flight[t] = time.monotonic()
            return True

        if not await _try_submit_next():
            return

        while in_flight:
            done, _ = await asyncio.wait(
                in_flight.keys(), timeout=15, return_when=asyncio.FIRST_COMPLETED)

            for t in done:
                del in_flight[t]
                try:
                    elapsed = t.result()
                except Exception as e:
                    logger.error("Batch chunk failed: %s", e)
                    elapsed = None
                if elapsed is not None and (fastest_seen is None or elapsed < fastest_seen):
                    fastest_seen = elapsed

            threshold = max(MIN_ADAPTIVE_WAIT, (fastest_seen or MIN_ADAPTIVE_WAIT) * 2)
            oldest_start = min(in_flight.values()) if in_flight else None
            if len(in_flight) < MAX_CONCURRENT and (
                oldest_start is None or time.monotonic() - oldest_start > threshold
            ):
                await _try_submit_next()

    def _passes_on_deck_rr_gate(self, report) -> tuple[bool, float, float]:
        """Same fair-value-based R/R formula used at buy-time (_attempt_near_miss_promotion
        and _evaluate_report). Requires a valid (>0) fair_value_estimate; candidates without
        one are rejected outright rather than silently falling back to a misleading
        percentage-based target. Gate itself is conviction-scaled (2026-07-18, see
        _required_rr) rather than one flat number for every stock — returns the actual
        required threshold too (not just pass/fail) so callers can log the real per-stock
        number instead of the flat config base value.

        Extracted (2026-07-23) from two near-identical local closures in _run_pre_open_batch
        and _run_on_deck_persist_check into one shared method — the exact kind of drift
        _persist_report's own docstring already flagged as a real risk (a third near-copy
        going unnoticed), now needed by a third real caller (_backfill_on_deck_from_on_shore)."""
        min_conviction = self.config["research"]["min_conviction_score"]
        base_rr = self.config["research"]["min_risk_reward_ratio"]
        rr_step = self.config["research"].get("on_deck_rr_conviction_step", 0.1)
        rr_floor = self.config["research"].get("on_deck_rr_floor", 1.5)
        if not report.fair_value_estimate or report.fair_value_estimate <= 0:
            return False, 0.0, base_rr
        risk = report.entry_price - report.stop_loss
        rr = (report.fair_value_estimate - report.entry_price) / risk if risk > 0 else 0
        required = _required_rr(report.conviction_score, min_conviction, base_rr, rr_step, rr_floor)
        return rr >= required, rr, required

    def _build_on_deck_entry(self, report, rr_val: float) -> dict:
        """A BUY-signal, conviction-qualified stock — added regardless of whether it already
        clears R/R; the dashboard sorts by R/R so already-attractive stocks sit at the top.
        price_history is set to [] here as a placeholder — every caller awaits
        self._fetch_price_history(ticker, ...) and overwrites it right after, so every new
        candidate gets a real ~30-day chart immediately instead of an empty one. required_rr
        is also a placeholder (base_rr) — callers overwrite with the real conviction-scaled
        value from _passes_on_deck_rr_gate. Extracted (2026-07-23) alongside
        _passes_on_deck_rr_gate — see that method's docstring for why."""
        base_rr = self.config["research"]["min_risk_reward_ratio"]
        return {
            "ticker": report.ticker,
            "company_name": report.company_name,
            "sector": getattr(report, "sector", ""),
            "business_summary": getattr(report, "business_summary", ""),
            "thesis": report.thesis,
            "signal": report.signal.value,
            "conviction_score": report.conviction_score,
            "fair_value_estimate": report.fair_value_estimate,
            "margin_of_safety_pct": report.margin_of_safety_pct,
            "last_price": report.entry_price,
            "rr": rr_val,
            "required_rr": base_rr,  # placeholder — caller overwrites with the real value
            "stop_loss_pct": _derive_stop_pct(
                report.entry_price, report.stop_loss,
                self.config["take_profit"]["stop_loss_pct"]),
            "direction": None,
            "streak": 0,
            "ai_entry_price": None,
            "ai_entry_low_ref": None,
            "ai_entry_reasoning": "",
            "ai_entry_seen_below": False,
            "ai_entry_pending": False,
            "price_history": [],
            "added_at": self._now_et().isoformat(),
        }

    def _on_deck_ranking_key_for(self, nm: dict) -> tuple[bool, float]:
        """Wraps _on_deck_ranking_key with this candidate's own conviction and the
        live min_conviction_score setting (2026-07-31, XRAY-adjacent fix) -- see that
        function's docstring for why every fill/trim/swap decision must rank a
        buy-eligible candidate ahead of a watch-only one regardless of composite
        score."""
        min_conviction = self.config["research"]["min_conviction_score"]
        return _on_deck_ranking_key(
            nm.get("conviction_score", 0) or 0,
            nm.get("margin_of_safety_pct", 0) or 0,
            nm.get("rr", 0) or 0,
            nm.get("required_rr") or 0.0,
            min_conviction,
        )

    async def _on_deck_ai_gate_above_gate(
        self, *, ticker: str, company_name: str, thesis: str, price: float,
        fair_value_estimate: float, stop_loss: float, rr: float, required_rr: float,
        conviction_score: float, fail_default: bool,
    ) -> tuple[bool, str]:
        """Shared AI judgment for a candidate whose R/R sits above its own real gate
        (2026-08-05, owner design). Used for RETENTION -- both at the twice-daily
        persist-check, and (as of 2026-08-18) continuously every 60s tick inside
        near_miss_monitor_loop, cooldown-gated via on_deck_above_gate_recheck_cooldown_minutes
        so the same still-above-gate candidate can't re-fire this real Claude call every
        single tick -- for one admission path -- `_backfill_on_deck_from_on_shore` -- the
        only one where a candidate genuinely has a track record of being watched rise past
        its own gate while listed before getting bumped for an unrelated reason, and, as of
        2026-08-13, for the actual PROMOTION/BUY TRIGGER itself
        (`_attempt_near_miss_promotion`) -- see that call site's own comment for the OXY
        incident that prompted extending this here: the identical ambiguity this function
        already judged for admission/retention was, until this date, never asked at the one
        moment that actually spends real money. Deliberately NOT used at the other 3
        admission sites (a fresh universe-scan result, the startup cache restore,
        the on-demand Settings-triggered refill): a candidate found above its own
        gate there has zero track record, and owner explicitly wants those excluded
        mechanically, no AI call, no exception -- "ai is for a stock that has risen
        up past the gate," not one that simply happened to already be above it the
        first time it was ever looked at.

        R/R above the gate is ambiguous on its own -- it can mean genuine
        undervaluation, or it can simply mean price kept falling toward the stop
        (or, per the OXY incident, a fresh re-analysis simply plants a tight stop
        right under a recent low), which mechanically inflates the ratio without
        the setup improving. Owner explicitly rejected a hardcoded numeric ceiling
        for "how far above is too far" ("i dont know when it would not be a good
        buy.. maybe something for ai to choose"), so this asks Claude directly,
        with real reasoning, rather than gating on a margin.

        fail_default controls what happens on any call failure (no API key,
        malformed response, API error) -- the caller passes True for retention (fail
        toward keeping the status quo of being listed) and False for both admission
        call sites and the buy trigger (fail toward keeping the status quo of NOT
        being listed / NOT buying), so a failure never actively changes anything or
        spends money either direction -- same AI Data Integrity principle as every
        other real trading figure in this codebase."""
        retention = await self.research_engine.recommend_on_deck_retention(
            ticker=ticker, company_name=company_name, thesis=thesis, price=price,
            fair_value_estimate=fair_value_estimate, stop_loss=stop_loss, rr=rr,
            required_rr=required_rr, conviction_score=conviction_score,
        )
        if retention is None:
            return fail_default, ""
        return retention

    def _mark_universe_reject(self, ticker: str) -> None:
        """Tags ticker's research_reports entry (if one exists) as a same-moment universe-scan
        reject (2026-07-19) so it appears on On Shore immediately once removed from On Deck,
        instead of only reappearing there whenever tomorrow's Phase 2 universe scan happens to
        re-touch it -- which might never happen again if it fails quick_screen tomorrow, in
        which case it would otherwise just silently vanish with no trace anywhere in the UI.
        Called from every AUTOMATIC On Deck removal path (conviction-drop in the persist-check
        and in "Scan Now", over-cap trim) -- deliberately NOT called for a successful buy (the
        ticker is now held, correctly excluded from both lists) or for the user's own manual
        removal via the On Deck card's X button (that's a deliberate "get rid of this, I don't
        want to see it" judgment call with its own block mechanism -- reappearing on On Shore
        immediately would defeat the point; see the on_deck_blocked filter in
        /api/today-scan-rejects instead, which keeps a manually-removed ticker off both lists
        for as long as its block lasts)."""
        r = self.research_reports.get(ticker)
        if r is None:
            return
        r["source"] = "universe_scan"
        r["generated_at"] = self._now_et().isoformat()
        asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))

    async def _enforce_on_deck_cap(self, phase_tag: str = "PRE-OPEN") -> int:
        """Trims near_miss_candidates down to research.on_deck_max_size (2026-07-19), keeping
        only the top-scoring candidates when over the cap. Deliberately does NOT limit how
        many stocks get scanned/analyzed -- quick_screen and analyze_stock still run against
        the full universe regardless (see _run_pre_open_batch) -- this only decides which
        already-analyzed candidates stay tracked afterward. Same "full-scan-then-rank"
        pattern used once before in this codebase (2026-07-15, before the Watchlist-removal
        redesign), reintroduced now because candidates persist across days and only ever
        leave via a buy or a genuine conviction collapse, so without a cap the list only
        ever grows. Returns the number of candidates dropped, for the caller's log line."""
        max_size = self.config["research"].get("on_deck_max_size", 0)
        if not max_size or len(self.near_miss_candidates) <= max_size:
            return 0
        # Tiered ranking (2026-07-31, XRAY-adjacent fix) -- a buy-eligible candidate
        # must always survive a trim before a watch-only one, regardless of composite
        # score. See _on_deck_ranking_key's docstring.
        ranked = sorted(self.near_miss_candidates.items(),
                         key=lambda kv: self._on_deck_ranking_key_for(kv[1]), reverse=True)
        keep = dict(ranked[:max_size])
        dropped = [t for t in self.near_miss_candidates if t not in keep]
        for ticker in dropped:
            entry = self.add_ai_log(ticker, phase_tag,
                f"Dropped from On Deck — over the {max_size}-stock cap, ranked below the cutoff",
                "warning")
            await self.broadcast({"type": "ai_log", "entry": entry})
            self._mark_universe_reject(ticker)
        self.near_miss_candidates = keep
        return len(dropped)

    async def _backfill_on_deck_from_on_shore(self) -> None:
        """Actively maintains On Deck at its configured setpoint (research.on_deck_max_size)
        throughout the day (2026-07-23) — previously a slot that opened up (a buy, or a
        candidate failing a persist-check) just stayed open until the next day's pre-open
        batch, even though On Shore already holds real, analyzed candidates that only missed
        out on ranking, not on the underlying gate. Called every near_miss_monitor_loop tick
        (60s); the overwhelming majority of ticks exit immediately with zero cost. Gated on
        research.on_deck_backfill_enabled (default true) so this real, recurring Claude
        spend can be switched off independently of adjusting the size itself.

        Two modes, both real Claude spend beyond pre-open/midday: (1) an open slot gets
        filled with the best-ranked eligible On Shore candidate; (2) once full, the
        BEST eligible On Shore candidate is compared against the WEAKEST current On Deck
        member (2026-07-23, user request: "the best candidates should be in On Deck... if
        it needs to switch out it should") — if it beats the incumbent by at least
        research.on_deck_swap_margin on the same composite score used for cap-trimming, it
        swaps in. That margin exists specifically to prevent thrashing: R/R shifts
        continuously with live price, so without a real margin, ordinary noise could flip
        candidates in and out (and burn a Claude call) on every single tick. Either way,
        never reuses several-hours-old On Shore data directly to actually join On Deck — one
        fresh analyze_stock() re-check first, since a thesis can go stale over a few hours
        (confirmed same-day: MRP and DV both failed a promotion attempt on fresh data after
        looking fine that morning)."""
        if not self.config["research"].get("on_deck_backfill_enabled", True):
            return
        max_size = self.config["research"].get("on_deck_max_size", 0)
        if not max_size:
            return

        min_conviction = self.config["research"]["min_conviction_score"]
        conviction_band = self.config["research"].get("on_deck_conviction_band", 0.0)
        population_floor = _on_deck_population_floor(min_conviction, conviction_band)
        today_str = self._now_et().strftime("%Y-%m-%d")
        held = set(self.portfolio.positions.keys())

        def _shore_score(d: dict) -> tuple[bool, float]:
            """Tiered ranking key (2026-07-31, XRAY-adjacent fix) -- see
            _on_deck_ranking_key's docstring: a buy-eligible candidate must always
            outrank a watch-only one here too, regardless of composite score, or a
            watch-only challenger with a great R/R could bump a genuinely buy-eligible
            On Deck member in the swap check below."""
            entry_price = d.get("entry_price", 0.0)
            stop_loss = d.get("stop_loss", 0.0)
            fair_value = d.get("fair_value_estimate", 0.0)
            conviction = d.get("conviction", 0)
            if entry_price <= 0 or stop_loss <= 0 or fair_value <= 0 or entry_price <= stop_loss:
                return (False, float("-inf"))
            rr = (fair_value - entry_price) / (entry_price - stop_loss)
            required_rr = _required_rr(
                conviction, min_conviction,
                self.config["research"]["min_risk_reward_ratio"],
                self.config["research"].get("on_deck_rr_conviction_step", 0.1),
                self.config["research"].get("on_deck_rr_floor", 1.5))
            margin = d.get("margin_of_safety_pct", 0.0) or 0.0
            return _on_deck_ranking_key(conviction, margin, rr, required_rr, min_conviction)

        _now_backfill = self._now_et()
        candidates = [
            (ticker, d) for ticker, d in self.research_reports.items()
            if (ticker not in held and ticker not in self.near_miss_candidates
                and not self._is_on_deck_blocked(ticker)
                and not self._wash_sale_blocked(ticker)
                and d.get("source") == "universe_scan"
                and d.get("generated_at", "").startswith(today_str)
                and d.get("signal") in ("BUY", "STRONG BUY")
                and d.get("conviction", 0) >= population_floor
                and not d.get("is_fallback", False)
                and not _on_deck_cooldown_active(
                    self._on_deck_backfill_reject_cooldown, ticker, _now_backfill)
                and not _on_deck_cooldown_active(
                    self._on_deck_backfill_above_gate_cooldown, ticker, _now_backfill)
                # Also checks the continuous above-gate loop's own cooldown (fixed
                # 2026-08-18, cost audit) -- see near_miss_monitor_loop's matching
                # comment. Without this, a candidate that loop just evicted for being
                # above gate (and asked Claude about) could be immediately reconsidered
                # here as a fresh backfill candidate, re-asking the identical question
                # this dict has no record of.
                and not _on_deck_cooldown_active(
                    self._on_deck_above_gate_cooldown, ticker, _now_backfill))
        ]
        if not candidates:
            return
        candidates.sort(key=lambda kv: _shore_score(kv[1]), reverse=True)

        async def _try_add_inner(ticker: str, log_prefix: str) -> bool:
            """One fresh Claude re-check; adds to near_miss_candidates and returns True only
            if it still clears every gate on live data — never trusts the stale On Shore
            snapshot that picked it as a candidate in the first place."""
            try:
                trade_history_summary = await self.portfolio.get_trade_history_summary(ticker)
                analysis_history_summary = await self.portfolio.get_analysis_history_summary(ticker)
                report = await self.research_engine.analyze_stock(
                    ticker, trade_history_summary, self.on_deck_notes.get(ticker, ""),
                    analysis_history_summary=analysis_history_summary)
            except Exception as e:
                logger.debug("%s: On Deck backfill re-analysis failed: %s", ticker, e)
                return False

            self._persist_report(report, source="universe_scan")
            asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))

            if (getattr(report, "is_fallback", True)
                    or report.signal.value not in ("BUY", "STRONG BUY")
                    or report.conviction_score < population_floor
                    or report.entry_price <= 0 or report.stop_loss <= 0
                    or not report.fair_value_estimate or report.fair_value_estimate <= 0):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    "On Shore backfill candidate no longer qualifies on fresh data — "
                    f"{report.signal.value} | Conviction {report.conviction_score}/10", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                return False

            rr_ok, rr_val, required_rr = self._passes_on_deck_rr_gate(report)
            floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
            if _on_deck_rr_floor_not_met(rr_val, required_rr, floor_margin):
                entry = self.add_ai_log(ticker, "ON_DECK",
                    f"On Shore backfill candidate below min R/R floor — R/R {rr_val:.2f} "
                    f"< {required_rr + floor_margin:.2f} (its own gate {required_rr:.2f} + "
                    f"{floor_margin:.2f} margin)", "warning")
                await self.broadcast({"type": "ai_log", "entry": entry})
                return False
            if _on_deck_rr_above_gate(rr_val, required_rr):
                # Above its own gate -- ask the same shared AI judgment retention uses,
                # rather than a flat ceiling reject (2026-08-05, owner design). This is
                # exactly the re-admission path a candidate needs after getting
                # evicted for an unrelated reason (conviction, cap-trim) while above
                # gate -- it can always find its way back as long as Claude still
                # judges it a good buy. fail_default=False: a call failure keeps the
                # status quo of NOT re-adding it.
                still_good_buy, reasoning = await self._on_deck_ai_gate_above_gate(
                    ticker=ticker, company_name=report.company_name, thesis=report.thesis,
                    price=report.entry_price, fair_value_estimate=report.fair_value_estimate,
                    stop_loss=report.stop_loss, rr=rr_val, required_rr=required_rr,
                    conviction_score=report.conviction_score, fail_default=False,
                )
                if not still_good_buy:
                    # Longer, dedicated cooldown on top of the generic reject cooldown
                    # _try_add's wrapper already applies below (2026-08-18, SNDK
                    # incident) -- see _on_deck_backfill_above_gate_cooldown's own
                    # comment in __init__ for why this decline specifically needs a
                    # much longer gap before being real-Claude-re-asked again.
                    above_gate_cooldown_min = self.config["research"].get(
                        "on_deck_backfill_above_gate_decline_cooldown_minutes", 60)
                    self._on_deck_backfill_above_gate_cooldown[ticker] = (
                        self._now_et() + timedelta(minutes=above_gate_cooldown_min))
                    entry = self.add_ai_log(ticker, "ON_DECK",
                        f"On Shore backfill candidate above its own gate — AI judged it's "
                        f"no longer a good buy: {reasoning}", "warning")
                    await self.broadcast({"type": "ai_log", "entry": entry})
                    return False

            entry_dict = self._build_on_deck_entry(report, rr_val)
            entry_dict["required_rr"] = required_rr
            entry_dict["price_history"] = await self._fetch_price_history(ticker, report.entry_price)
            # Re-check right before writing (2026-08-08, GitHub #50) -- a different
            # concurrent admission path (e.g. a mid-day rescan's _process_universe_scan_result)
            # could have independently added this same ticker while the two awaits above
            # were in flight; an unconditional overwrite here would silently discard
            # whichever one finished first. First-to-land wins instead.
            if ticker in self.near_miss_candidates:
                logger.debug(
                    "%s: On Deck backfill skipped its own write — already added by a "
                    "concurrent path while this re-check was in flight", ticker)
                return False
            self.near_miss_candidates[ticker] = entry_dict
            asyncio.create_task(self._maybe_auto_deep_dive(ticker, "ON_DECK"))

            status = "clears R/R now" if rr_ok else f"R/R {rr_val:.2f} < {required_rr:.2f} — watching"
            entry = self.add_ai_log(ticker, "ON_DECK",
                f"{log_prefix} — Conviction {report.conviction_score}/10 | R/R {rr_val:.2f} | "
                f"{status}", "success")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return True

        async def _try_add(ticker: str, log_prefix: str) -> bool:
            """Wraps _try_add_inner with a reject cooldown (2026-08-03, owner request) --
            see _on_deck_backfill_reject_cooldown's own comment in __init__ for why. A
            failure here means this ticker was the top-ranked On Shore candidate but didn't
            hold up on fresh data; without this, it's simply the top-ranked candidate again
            next tick and gets re-tried immediately, real Claude spend, every 60s until it
            either stabilizes or something else outranks it."""
            ok = await _try_add_inner(ticker, log_prefix)
            if not ok:
                cooldown_min = self.config["research"].get(
                    "on_deck_backfill_retry_cooldown_minutes", 5)
                self._on_deck_backfill_reject_cooldown[ticker] = (
                    self._now_et() + timedelta(minutes=cooldown_min))
            return ok

        open_slots = max_size - len(self.near_miss_candidates)
        if open_slots > 0:
            for ticker, _ in candidates[:open_slots]:
                await _try_add(ticker, "Backfilled to On Deck from On Shore (slot opened up)")
            asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))
            return

        # Full — consider swapping the weakest current member for a stronger challenger.
        if not self.near_miss_candidates:
            return
        swap_margin = self.config["research"].get("on_deck_swap_margin", 2.0)
        weakest_ticker, weakest_nm = min(
            self.near_miss_candidates.items(), key=lambda kv: self._on_deck_ranking_key_for(kv[1]))
        # Tiered threshold (2026-07-31, XRAY-adjacent fix): only the numeric half of
        # the weakest member's key gets the margin added -- the tier (first element)
        # stays intact, so tuple comparison still refuses a watch-only challenger
        # against a buy-eligible weakest member no matter how large swap_margin is.
        # See _on_deck_ranking_key's docstring.
        weakest_tier, weakest_score = self._on_deck_ranking_key_for(weakest_nm)
        challenger_ticker, challenger_d = candidates[0]
        if _shore_score(challenger_d) < (weakest_tier, weakest_score + swap_margin):
            return

        added = await _try_add(
            challenger_ticker,
            f"Swapped into On Deck from On Shore, replacing {weakest_ticker} (weaker-ranked)")
        if not added:
            return
        self.near_miss_candidates.pop(weakest_ticker, None)
        self._mark_universe_reject(weakest_ticker)
        entry = self.add_ai_log(weakest_ticker, "ON_DECK",
            f"Dropped from On Deck — replaced by stronger On Shore candidate {challenger_ticker}",
            "warning")
        await self.broadcast({"type": "ai_log", "entry": entry})
        asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

    async def _maybe_auto_deep_dive(self, ticker: str, phase_tag: str) -> None:
        """Fires one real Deep Dive (Sonnet, the richer DCF/moat/growth/catalysts/scenarios
        prompt) automatically the moment a stock ENTERS On Deck (2026-07-19) — once per entry
        event, not a recurring refresh, since On Deck is deliberately capped small (10-25
        stocks via on_deck_max_size) so the added per-candidate Sonnet cost stays bounded.
        Called from both places a ticker is newly added to near_miss_candidates: Phase 2's
        universe-scan additions and _refill_on_deck_from_shore's restores. Deliberately NOT
        called from the persist-check's "kept, still on the list" branch — that would re-run
        this on every existing candidate twice a day, which is a different, much larger cost
        the user didn't ask for.

        Purely a DISPLAY-layer enrichment — populates state.deep_dive_reports[ticker] so
        clicking into the stock shows the full Deep Dive modal (richer thesis, valuation
        reasoning, competitive moat, growth outlook, catalysts, bull/base/bear scenarios)
        instead of the thinner quick-scan research modal, via clickStock's existing
        deepDiveReports-first lookup — no frontend change needed. Deliberately does NOT feed
        the deep dive's own fair_value_estimate/margin_of_safety_pct back into the R/R
        buy-gate math anywhere; that stays driven by the quick-scan tier's fair_value alone
        (refreshed twice daily by the persist-check/midday re-analysis), per the explicit
        design discussion — introducing a second, separately-timed fair-value source into the
        same buy decision would let the two drift out of sync with each other for no benefit.

        deep_dive_analysis() reuses the SAME fundamental/insider/news/competitive data the
        quick-scan already gathered (no new market-data fetch) — it only makes one additional,
        pricier Claude call asking a much bigger question about that same snapshot. Requires
        research_engine.reports[ticker] to already hold a report, which it always does by the
        time this fires: analyze_stock() sets it for a fresh Phase 2 addition, and it's
        rebuilt from data/reports_cache.json at every startup for anything already in
        research_reports — which every refill-eligible On Shore ticker necessarily has.

        Gated on research.on_deck_auto_deep_dive (default True, added 2026-07-19 right after
        this feature shipped, at the user's request for an easy off-switch) — checked here
        rather than at each call site so any future call site inherits the same gate for
        free. Disabling this has no effect on the manual "Deep Dive" button, which always
        works regardless."""
        if not self.config["research"].get("on_deck_auto_deep_dive", True):
            return
        try:
            report = await self.research_engine.deep_dive_analysis(ticker)
        except Exception as e:
            logger.warning("Auto deep-dive failed for %s: %s", ticker, e)
            return
        if report is None or getattr(report, "is_fallback", False):
            return  # AI unavailable or errored -- leave whatever cached deep dive already exists
        self.deep_dive_reports[ticker] = _deep_dive_report_dict(report)
        asyncio.create_task(asyncio.to_thread(_save_dd_cache, self.deep_dive_reports))
        await self.broadcast(
            {"type": "deep_dive_report", "ticker": ticker, "report": self.deep_dive_reports[ticker]})
        entry = self.add_ai_log(ticker, phase_tag,
            f"Auto deep dive complete — fair value ${report.fair_value_estimate:.2f}, "
            f"margin {report.margin_of_safety_pct:.0f}%", "success")
        await self.broadcast({"type": "ai_log", "entry": entry})

    async def _refill_on_deck_from_shore(self, need: int, phase_tag: str = "SETTINGS") -> int:
        """Pulls the top-`need` highest-scored eligible On Shore candidates back onto On Deck
        (2026-07-19) — fires when a settings save raises on_deck_max_size above the current
        count. Raising the cap has no way to conjure new candidates out of thin air (that
        needs a real scan), but there's usually a ready supply sitting on On Shore already:
        the exact stocks that cleared quick_screen and got a real analysis today, just didn't
        make the cut at whatever the cap used to be. Reuses get_today_scan_rejects()'s own
        ranking (same composite score, same eligibility filters — source==universe_scan,
        today's date, not blocked, not held, not already on-deck) rather than duplicating
        that logic a second time; that endpoint does a live quote+history fetch for every
        eligible ticker just to rank them, so this carries the same cost as opening the On
        Shore tab once — acceptable here since it's a rare, user-initiated event, not a
        background loop.

        Safe to reuse today's frozen conviction/fair_value/thesis without firing a second
        fresh Claude call: being on On Deck only means "watched," not "about to be bought" —
        the real buy trigger (_attempt_near_miss_promotion) always re-analyzes fresh before
        ever executing an order, regardless of how the candidate got onto the list. Refetches
        full price_history for just the tickers actually restored (the On Shore list itself
        only carries a downsampled sparkline, to keep that endpoint's own payload small) so
        each one gets a real chart immediately, same as a brand-new candidate would. Returns
        the number actually restored (may be less than `need` if On Shore doesn't have that
        many eligible candidates)."""
        if need <= 0:
            return 0
        shore = await get_today_scan_rejects()
        # Population floor (2026-07-31, XRAY incident) -- get_today_scan_rejects()
        # itself applies no conviction floor at all (On Shore intentionally includes
        # every real analysis that didn't make On Deck, regardless of score), so this
        # call site must apply the same floor its sibling population paths
        # (_run_pre_open_batch, _backfill_on_deck_from_on_shore) already do. Missing
        # here previously -- confirmed live: XRAY (5.2 conviction) was pulled straight
        # onto On Deck with zero conviction check when a Settings save (that merely
        # included on_deck_max_size in its payload, unchanged) found an open slot.
        # Already sorted by score descending (get_today_scan_rejects' own docstring),
        # so filtering before slicing preserves that order.
        min_conviction = self.config["research"]["min_conviction_score"]
        conviction_band = self.config["research"].get("on_deck_conviction_band", 0.0)
        population_floor = _on_deck_population_floor(min_conviction, conviction_band)
        base_rr = self.config["research"]["min_risk_reward_ratio"]
        rr_step = self.config["research"].get("on_deck_rr_conviction_step", 0.1)
        rr_floor = self.config["research"].get("on_deck_rr_floor", 1.5)
        floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
        ceiling_margin = self.config["research"].get("on_deck_rr_ceiling_margin", 0.15)

        def _required_rr_for(r: dict) -> float:
            return _required_rr(r.get("conviction_score", 0), min_conviction, base_rr, rr_step, rr_floor)

        eligible = [(t, r) for t, r in shore.items()
                    if r.get("conviction_score", 0) >= population_floor
                    and not _on_deck_rr_floor_not_met(r.get("rr", 0.0), _required_rr_for(r), floor_margin)]
        default_stop_pct = self.config["take_profit"]["stop_loss_pct"]
        restored = 0
        for ticker, r in eligible:
            if restored >= need:
                break
            raw = self.research_reports.get(ticker, {})
            stop_loss = raw.get("stop_loss", 0.0)
            stop_pct = _derive_stop_pct(raw.get("entry_price", 0.0), stop_loss, default_stop_pct)
            required_rr = _required_rr_for(r)
            rr_val = r.get("rr", 0.0)
            if _on_deck_rr_ceiling_exceeded(rr_val, required_rr, ceiling_margin):
                # Above its own gate by more than the small tolerance margin -- mechanical
                # exclude, no AI call (2026-08-05, owner design; ceiling margin added
                # 2026-08-20). This rare, on-demand refill deliberately reuses today's
                # frozen cached data without a fresh Claude call (see this function's own
                # docstring) -- the AI-judgment exception is reserved for the continuous
                # On-Shore backfill path specifically, which is the one actually watching
                # a candidate's R/R move over time.
                continue
            entry = {
                "ticker": ticker,
                "company_name": r.get("company_name", ticker),
                "sector": r.get("sector", ""),
                "business_summary": r.get("business_summary", ""),
                "thesis": r.get("thesis", ""),
                "signal": r.get("signal", ""),
                "conviction_score": r.get("conviction_score", 0),
                "fair_value_estimate": r.get("fair_value_estimate", 0.0),
                "margin_of_safety_pct": r.get("margin_of_safety_pct", 0.0),
                "last_price": r.get("last_price", 0.0),
                "rr": r.get("rr", 0.0),
                "required_rr": r.get("required_rr", 0.0),
                "stop_loss_pct": stop_pct,
                "direction": None,
                "streak": 0,
                "ai_entry_price": None,
                "ai_entry_low_ref": None,
                "ai_entry_reasoning": "",
                "ai_entry_seen_below": False,
                "ai_entry_pending": False,
                "price_history": [],
                "added_at": self._now_et().isoformat(),
            }
            entry["price_history"] = await self._fetch_price_history(ticker, entry["last_price"])
            self.near_miss_candidates[ticker] = entry
            restored += 1
            log_entry = self.add_ai_log(ticker, phase_tag,
                f"Restored to On Deck from On Shore — Conviction {entry['conviction_score']}/10 "
                f"| R/R {entry['rr']:.2f}", "success")
            await self.broadcast({"type": "ai_log", "entry": log_entry})
            asyncio.create_task(self._maybe_auto_deep_dive(ticker, phase_tag))
        return restored

    def _persist_report(self, report, source: str = "") -> None:
        """Save a ResearchReport to the research_reports cache (dashboard-visible, disk-
        persisted via the caller's own asyncio.to_thread(_save_report_cache, ...) —
        deliberately NOT done here, since call sites differ on whether they want to save
        to disk immediately or batch it) and broadcast it live so the Reports tab and any
        open modal patch in place without a refresh.

        Consolidated (2026-07-20) from two near-identical local closures that used to live
        inside _run_on_deck_persist_check and _run_pre_open_batch — differing only in an
        optional `source` tag. That duplication is exactly the kind of drift that let a
        third, real gap go unnoticed: position_monitor_loop's hourly re-analysis of every
        HELD position recomputes a fresh report every hour but never called either
        closure, so research_reports stayed frozen at whatever was cached at original buy
        time — discovered via the portfolio health assessment feature flagging stale
        fair_value_estimate/margin_of_safety_pct data on several long-held positions.
        position_monitor_loop now calls this shared method too, closing that gap without
        adding a third duplicate."""
        report_data = {
            "ticker": report.ticker,
            "company_name": report.company_name,
            "signal": report.signal.value,
            "conviction": report.conviction_score,
            "risk_level": report.risk_level.value,
            "business_summary": getattr(report, "business_summary", ""),
            "thesis": report.thesis,
            "entry_price": round(report.entry_price, 2),
            "stop_loss": round(report.stop_loss, 2),
            "take_profit_targets": [round(t, 2) for t in report.take_profit_targets],
            "position_size_pct": report.position_size_pct,
            "time_horizon": report.time_horizon,
            "reasoning": report.reasoning,
            "fundamental_summary": report.fundamental_summary,
            "insider_summary": report.insider_summary,
            "news_summary": report.news_summary,
            "competitive_summary": report.competitive_summary,
            "risk_factors": report.risk_factors,
            "sector": getattr(report, "sector", ""),
            "fair_value_estimate": report.fair_value_estimate,
            "margin_of_safety_pct": report.margin_of_safety_pct,
            "generated_at": report.generated_at.isoformat(),
            "source": source,
        }
        self.research_reports[report.ticker] = report_data
        asyncio.create_task(self.broadcast({"type": "report", "report": report_data}))

        # "Analysis History" feed-forward (2026-08-21) -- appends a permanent row
        # (never overwritten, unlike research_reports above) so the NEXT analysis of
        # this same ticker can see the whole arc, including any watch_condition this
        # call stated. Skipped for a fallback report (no real Claude data -- nothing
        # worth remembering) -- same AI Data Integrity principle as every other real
        # trading figure in this codebase. Fire-and-forget: this must never delay or
        # block the report-persistence flow it's recording.
        if not getattr(report, "is_fallback", False):
            asyncio.create_task(self.portfolio.save_analysis_history(
                report.ticker, report.generated_at.isoformat(), report.conviction_score,
                report.signal.value, report.entry_price, report.fair_value_estimate,
                getattr(report, "watch_condition", ""),
            ))

    async def _run_on_deck_persist_check(self, phase_tag: str = "PRE-OPEN") -> tuple[int, int]:
        """Re-analyze every current On Deck candidate with a fresh, real Claude call —
        extracted (2026-07-19) from _run_pre_open_batch so the exact same logic can also
        run at the repurposed midday scan slot (see _run_midday_reanalysis), not just once
        a day at pre-open. phase_tag tags the ai_log entries ("PRE-OPEN" vs "MIDDAY") so the
        two call sites are distinguishable in the activity feed. Returns (kept, removed)."""
        removal_conviction = self.config["research"].get("on_deck_removal_conviction", 3)

        async def _log(ticker: str, msg: str, level: str = "neutral"):
            entry = self.add_ai_log(ticker, phase_tag, msg, level)
            await self.broadcast({"type": "ai_log", "entry": entry})

        held_tickers = set(self.portfolio.positions.keys())
        to_persist_check = [t for t in list(self.near_miss_candidates.keys())
                             if t not in held_tickers and not self._wash_sale_blocked(t)]
        for t in list(self.near_miss_candidates.keys()):
            if t in held_tickers:
                self.near_miss_candidates.pop(t, None)
            elif self._wash_sale_blocked(t):
                # Sweep eviction (2026-07-28, user request) -- catches a ticker whose
                # wash-sale cooldown started (or was already active) since it was last
                # added, independent of the promotion-time eviction in
                # _attempt_near_miss_promotion. Skipping it from to_persist_check above also
                # saves the real Claude call this pass would otherwise spend re-analyzing a
                # candidate that can't legally be bought right now.
                self.near_miss_candidates.pop(t, None)
                entry = self.add_ai_log(t, phase_tag,
                    "Removed from On Deck — wash-sale cooldown active, can't be bought", "warning")
                asyncio.create_task(self.broadcast({"type": "ai_log", "entry": entry}))

        kept = 0
        removed = 0
        already_gone = 0

        async def _persist_on_result(ticker: str, report) -> None:
            nonlocal kept, removed, already_gone
            nm = self.near_miss_candidates.get(ticker)
            if nm is None:
                # Removed concurrently (e.g. bought, or evicted by a swap/backfill) since
                # to_persist_check was built -- this candidate's fresh analysis result has
                # nowhere to land, so it's correctly neither "kept" nor "removed" by THIS
                # function. Counted separately (2026-08-03) so the completion summary's math
                # actually adds up to the original candidate count -- previously, if every
                # queued candidate happened to be concurrently removed this way (confirmed
                # live: a real "5 candidates" in, "0 kept, 0 removed" out mismatch), the
                # summary gave no indication why the numbers didn't sum to the original count.
                already_gone += 1
                return
            if getattr(report, "is_fallback", True):
                await _log(ticker, "Persist-check: AI unavailable — keeping unchanged", "warning")
                kept += 1
                return

            self._persist_report(report)
            asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))

            if report.conviction_score < removal_conviction:
                self.near_miss_candidates.pop(ticker, None)
                self._mark_universe_reject(ticker)
                removed += 1
                await _log(ticker,
                    f"Removed from On Deck — conviction fell to {report.conviction_score}/10 "
                    f"(below {removal_conviction})", "warning")
                return

            rr_ok, rr_val, required_rr = self._passes_on_deck_rr_gate(report)

            # R/R band sweep-eviction (2026-08-03, owner request) -- same precedent as the
            # wash-sale sweep above: only re-evaluated here, at persist-check time (real
            # fresh Claude data), not continuously every 60s tick off live price -- a
            # continuous version would risk a candidate bouncing on/off the list purely
            # from ordinary price noise near a band edge. Checked after the conviction
            # removal above but before updating nm, mirroring that check's own
            # pop/mark_universe_reject/removed/log/return pattern.
            #
            # Above-gate retention judged by a real Claude call, not a fixed timer
            # (2026-08-05, owner design -- a grace-period timer was built and deployed
            # earlier the same evening, then explicitly replaced: "i dont like the
            # timer.. lets do this instead... whenever it isnt a good buy any
            # more.. then evict it."). R/R exceeding a candidate's own gate is
            # ambiguous on its own -- it can mean genuine undervaluation, or it can
            # simply mean price kept falling toward the stop, which mechanically
            # inflates the ratio without the setup actually improving. The SAME
            # shared judgment (_on_deck_ai_gate_above_gate) is also used by
            # _backfill_on_deck_from_on_shore specifically (not the other 3
            # admission call sites, which mechanically exclude above-gate
            # candidates instead -- see that helper's docstring for why the scope
            # is narrow) -- so a candidate evicted here for something unrelated
            # (conviction, cap-trim) while above gate isn't permanently locked out
            # of returning via that one path, as long as Claude still judges it a
            # good buy. fail_default=True here: a call failure keeps the status quo
            # of already being listed, never evicting on missing/failed AI data.
            if _on_deck_rr_above_gate(rr_val, required_rr):
                still_good_buy, reasoning = await self._on_deck_ai_gate_above_gate(
                    ticker=ticker, company_name=report.company_name, thesis=report.thesis,
                    price=report.entry_price, fair_value_estimate=report.fair_value_estimate,
                    stop_loss=report.stop_loss, rr=rr_val, required_rr=required_rr,
                    conviction_score=report.conviction_score, fail_default=True,
                )
                # Re-fetch (2026-08-08, GitHub #50): the candidate may have been bought,
                # manually removed, or already re-analyzed and restored by a concurrent
                # promotion attempt's own finally block (_refresh_nm_from_report) while
                # this real Claude call was in flight -- same race class, same fix
                # pattern as _compute_ai_dip_entry's own re-fetch above. Without this, a
                # stale `nm` reference below could either evict a candidate a concurrent
                # path just freshly re-verified, or overwrite its just-restored fresh
                # fields with this call's older report data.
                nm = self.near_miss_candidates.get(ticker)
                if nm is None:
                    already_gone += 1
                    return
                if not still_good_buy:
                    self.near_miss_candidates.pop(ticker, None)
                    self._mark_universe_reject(ticker)
                    removed += 1
                    await _log(ticker,
                        f"Removed from On Deck — R/R {rr_val:.2f} above its own gate "
                        f"({required_rr:.2f}), AI judged it's no longer a good buy: "
                        f"{reasoning}", "warning")
                    return
                if reasoning:
                    await _log(ticker,
                        f"R/R {rr_val:.2f} above its own gate ({required_rr:.2f}), AI "
                        f"judged it's still a genuine opportunity: {reasoning}", "neutral")
            floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
            if _on_deck_rr_floor_not_met(rr_val, required_rr, floor_margin):
                self.near_miss_candidates.pop(ticker, None)
                self._mark_universe_reject(ticker)
                removed += 1
                await _log(ticker,
                    f"Removed from On Deck — R/R {rr_val:.2f} below min R/R floor "
                    f"({required_rr + floor_margin:.2f}, its own gate {required_rr:.2f} "
                    f"+ {floor_margin:.2f} margin)", "warning")
                return

            nm["company_name"] = report.company_name
            nm["sector"] = getattr(report, "sector", "")
            nm["business_summary"] = getattr(report, "business_summary", "")
            nm["thesis"] = report.thesis
            nm["signal"] = report.signal.value
            nm["conviction_score"] = report.conviction_score
            nm["fair_value_estimate"] = report.fair_value_estimate
            nm["margin_of_safety_pct"] = report.margin_of_safety_pct
            nm["last_price"] = report.entry_price
            nm["rr"] = rr_val
            nm["required_rr"] = required_rr
            nm["stop_loss_pct"] = _derive_stop_pct(
                report.entry_price, report.stop_loss, self.config["take_profit"]["stop_loss_pct"])
            nm["direction"] = None
            nm["streak"] = 0
            nm["ai_entry_price"] = None
            nm["ai_entry_low_ref"] = None
            nm["ai_entry_reasoning"] = ""
            nm["ai_entry_seen_below"] = False
            # price_history deliberately NOT reset here — see near_miss_monitor_loop's
            # docstring; the recovery check tracks a multi-day dip cycle, not just today.
            # DO append one fresh point reflecting THIS refresh, though (2026-07-22 fix) —
            # near_miss_monitor_loop only appends ticks during market hours, but this persist-
            # check runs regardless of market hours (pre-open, ~8am; midday, ~12:30pm) and can
            # change fair_value_estimate enough to move the badge's live R/R across its gate.
            # Without a matching price_history point, the chart's last plotted R/R stays
            # anchored to whatever the market-hours loop last recorded — possibly the previous
            # trading day's close — visibly disagreeing with the just-refreshed badge for
            # however long remains until the monitor loop resumes (up to ~1.5h at pre-open).
            nm.setdefault("price_history", []).append(
                (datetime.now().timestamp(), report.entry_price))
            kept += 1
            status = "clears R/R now" if rr_ok else f"R/R {rr_val:.2f} < {required_rr:.2f} — watching"
            await _log(ticker,
                f"Persist-check — Conviction {report.conviction_score}/10 | R/R {rr_val:.2f} | "
                f"{status}", "neutral")

        async def _persist_chunks():
            for i in range(0, len(to_persist_check), 100):
                if self.paused or self.stopped:
                    return
                yield to_persist_check[i:i + 100]

        if to_persist_check and not (self.paused or self.stopped):
            await _log("SYSTEM",
                f"Persist-check — re-analyzing {len(to_persist_check)} existing On Deck "
                "candidate(s)")
            # "Analysis History" feed-forward (2026-08-21) -- pre-fetched here since
            # to_persist_check is always small (bounded by on_deck_max_size, typically
            # ~5), unlike the full universe scan this same _run_batched_chunk_loop
            # orchestrator also drives elsewhere (which never pre-fetches this). These
            # are exactly the RECURRING candidates the owner's request was about ("I
            # see the same stocks analyzed all the time").
            analysis_history_summaries = {
                t: await self.portfolio.get_analysis_history_summary(t)
                for t in to_persist_check
            }
            await self._run_batched_chunk_loop(
                _persist_chunks(), _persist_on_result,
                analysis_history_summaries=analysis_history_summaries)
            _already_gone_str = (f", {already_gone} already gone (bought/evicted "
                                  "concurrently)" if already_gone else "")
            await _log("SYSTEM",
                f"Persist-check complete — {kept} kept, {removed} removed (conviction below "
                f"{removal_conviction}, or outside the configured R/R band){_already_gone_str}")
        return kept, removed

    async def _run_midday_reanalysis(self) -> None:
        """Repurposes the otherwise-idle midday scan slot (2026-07-19 — auto_scan_loop's
        scheduled cycles do no scanning/buying of their own since the 2026-07-17 On Deck
        redesign) to refresh every On Deck candidate's fair_value_estimate/conviction a
        second time each day. Without this, fair_value_estimate — which drives every R/R
        gate decision for the rest of the day — is up to ~8-9 hours stale by market close,
        entirely unrefreshed between one morning's pre-open and the next. Real added Claude
        cost (one call per current On Deck candidate, same as the daily persist-check) —
        user explicitly accepted this, planning to manage total spend via an On Deck size
        cap rather than by skipping the second look."""
        if not os.getenv("ANTHROPIC_API_KEY", ""):
            return
        entry = self.add_ai_log("SYSTEM", "MIDDAY",
            f"Midday re-analysis starting — refreshing {len(self.near_miss_candidates)} "
            "On Deck candidate(s)", "info")
        await self.broadcast({"type": "ai_log", "entry": entry})
        kept, removed = await self._run_on_deck_persist_check("MIDDAY")
        entry = self.add_ai_log("SYSTEM", "MIDDAY",
            f"Midday re-analysis complete — {kept} kept, {removed} removed "
            f"({len(self.near_miss_candidates)} total on On Deck)", "success")
        await self.broadcast({"type": "ai_log", "entry": entry})
        asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

    async def _process_universe_scan_result(
        self, ticker: str, report, phase_tag: str, population_floor: float,
    ) -> bool:
        """Shared per-ticker result processing for any universe-scan-style Claude
        analysis pass (2026-07-31) -- persists the report, applies the standard On
        Deck admission gates (signal, population floor, wash-sale, valid entry/stop/
        fair-value), and adds a qualifying candidate to near_miss_candidates. Extracted
        from _run_pre_open_batch's original inline _on_result closure so the new
        mid-day re-scan (_run_midday_rescan) can share the exact same admission logic
        instead of maintaining a second copy that could drift -- the same "duplicated
        logic drifts" lesson as the population-floor and composite-score fixes earlier
        the same day (see CLAUDE.md). Returns True if the candidate was added to On
        Deck, False otherwise (caller tracks its own added-count)."""
        if getattr(report, "is_fallback", True):
            entry = self.add_ai_log(ticker, phase_tag, "AI unavailable — skipping", "warning")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        self._persist_report(report, source="universe_scan")
        asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))

        min_conviction = self.config["research"]["min_conviction_score"]
        signal_ok = (report.signal.value in ("BUY", "STRONG BUY")
                     and report.conviction_score >= population_floor
                     and report.entry_price > 0 and report.stop_loss > 0)
        if not signal_ok:
            entry = self.add_ai_log(ticker, phase_tag,
                f"Not added — {report.signal.value} | Conviction {report.conviction_score}/10",
                "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        if self._wash_sale_blocked(ticker):
            entry = self.add_ai_log(ticker, phase_tag,
                "Not added — wash-sale cooldown active, can't be bought yet", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        rr_ok, rr_val, required_rr = self._passes_on_deck_rr_gate(report)
        if not report.fair_value_estimate or report.fair_value_estimate <= 0:
            entry = self.add_ai_log(ticker, phase_tag,
                f"Not added — {report.signal.value} | Conviction {report.conviction_score}/10 | "
                "no valid fair value estimate for R/R", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        floor_margin = self.config["research"].get("on_deck_rr_floor_margin")
        if _on_deck_rr_floor_not_met(rr_val, required_rr, floor_margin):
            entry = self.add_ai_log(ticker, phase_tag,
                f"Not added — {report.signal.value} | Conviction {report.conviction_score}/10 | "
                f"R/R {rr_val:.2f} below min R/R floor ({required_rr + floor_margin:.2f}, "
                f"its own gate {required_rr:.2f} + {floor_margin:.2f} margin)", "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        ceiling_margin = self.config["research"].get("on_deck_rr_ceiling_margin", 0.15)
        if _on_deck_rr_ceiling_exceeded(rr_val, required_rr, ceiling_margin):
            # Above its own gate by more than the small tolerance margin -- mechanical
            # exclude, no AI call (2026-08-05, owner design; ceiling margin added
            # 2026-08-20). Shared by both the pre-open batch and the mid-day rescan; a
            # candidate found here has zero track record of being watched rise past its
            # own gate, unlike the On-Shore backfill path, which is where the
            # AI-judgment exception is reserved for instead. A candidate only just above
            # its own gate (within ceiling_margin) is still admitted here -- see
            # _on_deck_rr_ceiling_exceeded's docstring for why.
            entry = self.add_ai_log(ticker, phase_tag,
                f"Not added — R/R {rr_val:.2f} above ceiling ({required_rr + ceiling_margin:.2f}, "
                f"its own gate {required_rr:.2f} + {ceiling_margin:.2f} margin) on first look",
                "neutral")
            await self.broadcast({"type": "ai_log", "entry": entry})
            return False

        entry_dict = self._build_on_deck_entry(report, rr_val)
        entry_dict["required_rr"] = required_rr
        entry_dict["price_history"] = await self._fetch_price_history(ticker, entry_dict["last_price"])
        self.near_miss_candidates[ticker] = entry_dict
        status = "clears R/R now" if rr_ok else f"R/R {rr_val:.2f} < {required_rr:.2f} — watching"
        below_entry = (" (below entry gate " f"{min_conviction} — conviction-watch only)"
                        if report.conviction_score < min_conviction else "")
        entry = self.add_ai_log(ticker, phase_tag,
            f"Added — {report.signal.value} | Conviction {report.conviction_score}/10 | "
            f"R/R {rr_val:.2f} | {status}{below_entry}", "success")
        await self.broadcast({"type": "ai_log", "entry": entry})
        asyncio.create_task(self._maybe_auto_deep_dive(ticker, phase_tag))
        return True

    async def _run_midday_rescan(self, slot_label: str = "") -> None:
        """Mid-day re-scan (2026-07-31) — re-runs the free quick_screen() universe check
        outside the once-a-day pre-open batch, so a stock that develops a qualifying
        setup after that morning scan isn't stuck waiting until the next day's pre-open
        to get picked up. Deliberately narrow: only tickers with NO research_reports
        entry generated today from ANY source (_not_yet_analyzed_today) get a real
        Claude call — this is what keeps the feature's spend scoped to genuinely new
        opportunities rather than repeating today's already-done analysis. Reuses
        _run_batched_chunk_loop (the same adaptive Batch-API-with-sequential-fallback
        orchestrator the pre-open batch already uses) and _process_universe_scan_result
        (the same admission-gate logic, extracted in the same commit series as this
        feature specifically so this function and _run_pre_open_batch can't drift apart
        — see CLAUDE.md). Does NOT touch the shared universe scan cursor
        (watchlist_manager.set_scan_cursor) or re-vet existing On Deck candidates —
        both are the pre-open batch's job, not this supplementary pass's.

        Concurrency (2026-07-31, added after a live incident deploying this same
        feature): the caller (auto_scan_loop's firing check, or the manual
        /api/trigger-midday-rescan endpoint) is responsible for checking and setting
        self._midday_rescan_in_progress BEFORE calling this method — this function owns
        clearing it in a finally block once done, regardless of how it exits. Without
        this, a restart occurring after a configured slot's time had already passed for
        the day fired that slot immediately via the catch-up logic below, and a second
        slot (or a manual trigger) racing in at the same time started a fully redundant
        concurrent scan over the same 1,500+ ticker universe -- real, confirmed
        duplicate Claude spend, not just a theoretical risk. See
        docs/superpowers/specs/2026-07-31-midday-rescan-design.md and
        docs/CLAUDE_HISTORY.md for the incident."""
        try:
            if not os.getenv("ANTHROPIC_API_KEY", ""):
                logger.warning("No Anthropic API key — mid-day re-scan skipped")
                return

            min_conviction = self.config["research"]["min_conviction_score"]
            conviction_band = self.config["research"].get("on_deck_conviction_band", 0.0)
            population_floor = _on_deck_population_floor(min_conviction, conviction_band)
            today_str = self._now_et().strftime("%Y-%m-%d")
            tag = f"MID-DAY{f' {slot_label}' if slot_label else ''}"

            entry = self.add_ai_log("SYSTEM", tag,
                f"Mid-day re-scan starting — checking {len(STOCK_UNIVERSE):,} universe stocks "
                "for setups not already covered today...", "info")
            await self.broadcast({"type": "ai_log", "entry": entry})

            held_tickers = set(self.portfolio.positions.keys())
            universe_candidates = self.watchlist_manager.available_from_universe(STOCK_UNIVERSE)

            added = 0
            screened_out = 0
            already_covered = 0
            scanned = 0

            async def _on_result(ticker: str, report) -> None:
                nonlocal added
                if await self._process_universe_scan_result(ticker, report, tag, population_floor):
                    added += 1

            async def _chunks():
                nonlocal screened_out, scanned, already_covered
                buffer: list[str] = []
                for ticker in universe_candidates:
                    if self.paused:
                        break
                    if (ticker in held_tickers or ticker in self.near_miss_candidates
                            or self._is_on_deck_blocked(ticker)):
                        continue
                    if not _not_yet_analyzed_today(self.research_reports.get(ticker), today_str):
                        already_covered += 1
                        continue

                    scanned += 1
                    # Deliberately does NOT advance watchlist_manager's scan cursor -- that
                    # cursor tracks progress through the ONE canonical daily sweep
                    # (_run_pre_open_batch); this is a supplementary pass, not a second
                    # sweep, so it must not interfere with where the cursor resumes tomorrow.

                    result = await _quick_screen_with_timeout(ticker)
                    if result is None:
                        continue
                    passes, _reason = result

                    if not passes:
                        screened_out += 1
                        await asyncio.sleep(0.2)
                        continue

                    buffer.append(ticker)
                    if len(buffer) >= 100:
                        yield buffer
                        buffer = []
                if buffer:
                    yield buffer

            if self.paused:
                logger.info("Mid-day re-scan skipped — paused")
            else:
                await self._run_batched_chunk_loop(_chunks(), _on_result)

            trimmed = await self._enforce_on_deck_cap(tag)

            asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))
            asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

            trimmed_str = f"; {trimmed} trimmed (over cap)" if trimmed else ""
            entry = self.add_ai_log("SYSTEM", tag,
                f"Mid-day re-scan complete — {added} new On Deck candidate(s) "
                f"({len(self.near_miss_candidates)} total on On Deck); {scanned:,} genuinely-new "
                f"tickers checked, {screened_out:,} screened out, {already_covered:,} already "
                f"covered today{trimmed_str}.",
                "success")
            await self.broadcast({"type": "ai_log", "entry": entry})
        finally:
            self._midday_rescan_in_progress = False

    async def _run_pre_open_batch(self, source_label: str = "PRE-OPEN"):
        """Pre-open universe scan — quick-screens the full universe, runs full analyze_stock()
        on survivors, and adds every conviction/signal-qualifying stock (regardless of R/R) to
        near_miss_candidates for continuous live monitoring by near_miss_monitor_loop.

        Replaced the old two-phase design (2026-07-17) — Phase 1 existed to re-vet a
        persistent watchlist, but there is no more persistent watchlist to re-vet: buying no
        longer depends on a cached, twice-daily-scanned list, since near_miss_monitor_loop
        watches every qualifying stock continuously and _attempt_near_miss_promotion buys the
        instant a candidate clears R/R + a confirmed uptick (with rotation-swap if the
        portfolio is full). There is also no more slot-based cap on how many candidates get
        tracked — every qualifying stock is watched, sorted by R/R on the dashboard so
        already-attractive stocks sit at the top. All Claude spend is here so daytime scans
        are free (price-check only).

        Candidates now persist across days (2026-07-17) instead of being wiped every morning
        — a stock added yesterday stays on the list, re-analyzed each pre-open alongside new
        universe survivors, and only drops off once its conviction genuinely falls below
        on_deck_removal_conviction. Deliberately set well below the entry gate
        (min_conviction_score, 7) — not just to avoid noise-driven flapping near the entry
        threshold, but per the user's explicit reasoning (2026-07-18): a stock that once
        scored 7+/10 is a genuinely good company, and a dip that temporarily drags its
        conviction down (say, to 4 or 5) is exactly the kind of opportunity On Deck exists to
        keep watching, not a reason to drop it. Only a much deeper conviction collapse (below
        3/10 — a real thesis breakdown, not routine noise) removes it. See the persist-check
        phase below, which runs before the universe fill and re-uses the exact same
        _run_batched_chunk_loop/analyze_stock() machinery — same cost/consistency profile as
        scanning a new stock, not a cheaper shortcut.

        source_label (2026-08-03) tags every ai_log entry this run produces — defaults to
        "PRE-OPEN" for the real scheduled morning run, overridden to "FULL SCAN" by
        _run_full_scan_on_demand() (the manual dashboard button) so the log correctly
        distinguishes a manual mid-day trigger from the actual daily batch, rather than
        every manual run confusingly claiming to be "PRE-OPEN"."""
        if not os.getenv("ANTHROPIC_API_KEY", ""):
            logger.warning("No Anthropic API key — pre-open scan skipped")
            return

        min_conviction = self.config["research"]["min_conviction_score"]
        removal_conviction = self.config["research"].get("on_deck_removal_conviction", 3)
        # Population floor sits BELOW the real entry gate (2026-07-19, user's design — a
        # differential off min_conviction_score rather than an independent absolute number,
        # so it automatically follows if min_conviction_score is ever changed). A stock
        # scoring in this band gets tracked/watched but still can't buy — every real buy
        # decision (_attempt_near_miss_promotion) re-checks conviction fresh against the
        # UNCHANGED min_conviction_score, not this floor. Purpose: a stock at 6.8 is a real,
        # close call worth watching for a later re-analysis to push it over the line, rather
        # than being silently dropped and never reconsidered until the next full universe scan
        # happens to rediscover it from scratch.
        conviction_band = self.config["research"].get("on_deck_conviction_band", 0.0)
        population_floor = _on_deck_population_floor(min_conviction, conviction_band)

        async def _log(ticker: str, msg: str, level: str = "neutral"):
            entry = self.add_ai_log(ticker, source_label, msg, level)
            await self.broadcast({"type": "ai_log", "entry": entry})

        _scan_verb = "Pre-open scan" if source_label == "PRE-OPEN" else "Full scan"
        entry = self.add_ai_log("SYSTEM", source_label,
            f"{_scan_verb} starting — scanning {len(STOCK_UNIVERSE):,} universe stocks...", "info")
        await self.broadcast({"type": "ai_log", "entry": entry})

        held_tickers = set(self.portfolio.positions.keys())
        universe_candidates = self.watchlist_manager.available_from_universe(STOCK_UNIVERSE)

        added = 0
        screened_out = 0
        scanned = 0

        # Re-analyze every candidate already on the list before touching the universe —
        # extracted (2026-07-19) into _run_on_deck_persist_check so the identical logic
        # can also run at the repurposed midday scan slot, not just once a day here.
        kept, removed = await self._run_on_deck_persist_check(source_label)

        async def _on_result(ticker: str, report) -> None:
            nonlocal added
            if await self._process_universe_scan_result(ticker, report, source_label, population_floor):
                added += 1

        async def _chunks():
            nonlocal screened_out, scanned
            buffer: list[str] = []
            _last_progress_log = scanned
            for ticker in universe_candidates:
                if self.paused:
                    break
                if (ticker in held_tickers or ticker in self.near_miss_candidates
                        or self._is_on_deck_blocked(ticker)):
                    continue

                scanned += 1

                # Advance cursor regardless of outcome so next pre-open resumes from here
                if ticker in STOCK_UNIVERSE:
                    self.watchlist_manager.set_scan_cursor(
                        (STOCK_UNIVERSE.index(ticker) + 1) % len(STOCK_UNIVERSE))

                # Quick screen first — free, no Claude, filters ~97% of universe now
                result = await _quick_screen_with_timeout(ticker)
                if result is None:
                    continue
                passes, _reason = result

                if not passes:
                    screened_out += 1
                    await asyncio.sleep(0.2)
                    continue

                buffer.append(ticker)
                if len(buffer) >= 100:
                    yield buffer
                    buffer = []

                # Periodic progress visibility — a scan can otherwise go completely silent
                # for 10+ minutes with no way to tell it's still working.
                if scanned - _last_progress_log >= 200:
                    _last_progress_log = scanned
                    await _log("SYSTEM",
                        f"Universe scan progress — {scanned:,} scanned, {screened_out:,} "
                        f"screened out, {len(buffer)} quick-screen survivor(s) buffered so far")
            if buffer:
                yield buffer

        if self.paused:
            logger.info("Pre-open scan skipped — paused")
        else:
            await self._run_batched_chunk_loop(_chunks(), _on_result)

        trimmed = await self._enforce_on_deck_cap(source_label)

        asyncio.create_task(asyncio.to_thread(_save_report_cache, self.research_reports))
        asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(self.near_miss_candidates)))

        trimmed_str = f"; {trimmed} trimmed (over cap)" if trimmed else ""
        # On Shore count (2026-07-23, user request) -- same core filter
        # get_today_scan_rejects() uses (source=="universe_scan", today's date, not held,
        # not on-deck), but without that endpoint's live quote/R/R fetch per ticker -- this
        # is purely a count for the completion summary, so no need to pay for 80+ live
        # yfinance calls just to report a number.
        _today_str = self._now_et().strftime("%Y-%m-%d")
        _held = set(self.portfolio.positions.keys())
        on_shore_count = sum(
            1 for _t, _r in self.research_reports.items()
            if _r.get("source") == "universe_scan"
            and _r.get("generated_at", "").startswith(_today_str)
            and _t not in self.near_miss_candidates and _t not in _held
        )
        _complete_verb = "Pre-open complete" if source_label == "PRE-OPEN" else "Full scan complete"
        _ready_suffix = " Ready for market open." if source_label == "PRE-OPEN" else ""
        entry = self.add_ai_log("SYSTEM", source_label,
            f"{_complete_verb} — {added} new, {kept} carried over, {removed} removed "
            f"({len(self.near_miss_candidates)} total on On Deck, {on_shore_count} on On "
            f"Shore; {scanned:,} scanned, {screened_out:,} screened out{trimmed_str})."
            f"{_ready_suffix}",
            "success")
        await self.broadcast({"type": "ai_log", "entry": entry})
        asyncio.create_task(_notify(
            "Pre-open scan complete",
            f"{len(self.near_miss_candidates)} candidate(s) on On Deck — market opens soon",
            priority="default", tags="sunrise"))

        # First-scan-ever marker (2026-07-21) -- see startup()'s fresh-install banner logic.
        # Written on any successful completion of this function, whether triggered by the
        # daily schedule, the user's manual "Run Scan Now" banner button, or the existing
        # /api/trigger-batch-scan endpoint -- any of these mean data now exists, so the
        # banner should never show again after this point.
        if not self.needs_first_scan:
            pass  # already known-populated, nothing to update
        else:
            self.needs_first_scan = False
            try:
                _FIRST_SCAN_MARKER.parent.mkdir(parents=True, exist_ok=True)
                _FIRST_SCAN_MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to write first-scan marker: %s", e)
            await self.broadcast({"type": "first_scan_done"})

    async def _run_full_scan_on_demand(self):
        """Wraps _run_pre_open_batch() for the manual "Full Scan" dashboard button
        (2026-08-03) -- deliberately a thin wrapper rather than editing that already-long,
        already-live function to add its own try/finally, so this change can't risk any
        regression in the pre-open batch's own tested behavior. Owns setting/clearing
        self._full_scan_in_progress (caller -- the /api/trigger-batch-scan route -- checks
        and sets it before calling this) and broadcasts an unconditional completion event
        so the button can reset even though _run_pre_open_batch's own completion signal
        (first_scan_done) only ever fires once, the very first time it's ever run."""
        try:
            await self._run_pre_open_batch(source_label="FULL SCAN")
        finally:
            self._full_scan_in_progress = False
            await self.broadcast({"type": "full_scan_done"})


state = DashboardState()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse("/static/icon-192.png")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html = (Path(__file__).parent / "templates" / "dashboard.html").read_text(encoding="utf-8")
    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    cfg = state.config
    template = (Path(__file__).parent / "templates" / "settings.html").read_text(encoding="utf-8")

    def _v(section, key, default=""):
        return cfg.get(section, {}).get(key, default)

    # Build a simple namespace object the template can reference
    class _NS:
        pass
    ns = _NS()
    for section in ("portfolio", "take_profit", "risk_management", "research", "trading", "dashboard", "risk_tier"):
        sub = _NS()
        for k, v in cfg.get(section, {}).items():
            setattr(sub, k, v)
        setattr(ns, section, sub)

    # Load API key status for server-side rendering (avoids JS-fetch dependency on fresh install)
    env_path = Path(__file__).parent.parent / ".env"
    _env_keys: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            _env_keys[k.strip()] = v.strip().strip('"').strip("'")
    _paper_url = "https://paper-api.alpaca.markets"
    api_keys = {
        "ANTHROPIC_API_KEY":  {"label": "Anthropic API Key",  "value": _env_keys.get("ANTHROPIC_API_KEY", ""),  "is_set": bool(_env_keys.get("ANTHROPIC_API_KEY"))},
        "ALPACA_API_KEY":     {"label": "Alpaca API Key",     "value": _env_keys.get("ALPACA_API_KEY", ""),     "is_set": bool(_env_keys.get("ALPACA_API_KEY"))},
        "ALPACA_BASE_URL":    {"label": "Alpaca Base URL",    "value": _env_keys.get("ALPACA_BASE_URL", _paper_url), "is_set": bool(_env_keys.get("ALPACA_BASE_URL"))},
        "FINNHUB_API_KEY":    {"label": "Finnhub API Key",    "value": _env_keys.get("FINNHUB_API_KEY", ""),    "is_set": bool(_env_keys.get("FINNHUB_API_KEY"))},
        "NEWSAPI_API_KEY":    {"label": "NewsAPI Key",        "value": _env_keys.get("NEWSAPI_API_KEY", ""),    "is_set": bool(_env_keys.get("NEWSAPI_API_KEY"))},
    }

    # Render Jinja2-style template
    from jinja2 import Template
    rendered = Template(template).render(cfg=ns, api_keys=api_keys)
    return Response(
        content=rendered,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"},
    )


def _coerce_settings_payload(payload: dict, coercions: dict) -> dict:
    """Validates/coerces every COERCIONS-covered key in a /api/settings payload and
    returns the coerced values as a plain dict -- WITHOUT mutating anything itself
    (fixed 2026-08-09, GitHub #68). Extracted as its own pure function specifically so
    it's testable via this codebase's established AST-extraction technique (web/app.py
    can't be imported directly -- it builds a real DashboardState() at module import
    time). Raises ValueError on the first field that fails coercion, with nothing in
    the return dict for a payload that fails partway through -- the caller can then
    reject the whole save with confidence that NOTHING was ever applied, rather than
    the old behavior of mutating state.config field-by-field inside the same loop that
    could still raise partway through, which left every field processed before the bad
    one silently live while config/settings.yaml was never written and the user was
    told the save failed."""
    coerced: dict = {}
    for dotkey, raw_val in payload.items():
        if dotkey not in coercions:
            continue
        try:
            coerced[dotkey] = coercions[dotkey](raw_val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {dotkey}: {e}")
    return coerced


@app.post("/api/settings")
async def save_settings(payload: dict):
    """Persist settings to settings.yaml and update live config."""
    import yaml

    config_path = Path("config/settings.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    COERCIONS = {
        "take_profit.stop_loss_pct": float,
        "take_profit.t1_pct": float,
        "take_profit.t2_pct": float,
        "take_profit.t3_pct": float,
        "take_profit.final_tranche_trail_pct": float,
        "take_profit.dollar_target_enabled": lambda v: v == "true",
        "take_profit.dollar_target_amount": float,
        "take_profit.dollar_target_trail_pct": float,
        "portfolio.max_positions": int,
        "portfolio.initial_capital": float,
        "risk_tier.value": float,
        "risk_management.max_position_pct": float,
        "risk_management.starting_position_pct": float,
        "risk_management.min_cash_reserve_pct": float,
        "risk_management.daily_loss_limit_pct": float,
        "risk_management.max_loss_per_trade_pct": float,
        "risk_management.sector_concentration_enabled": lambda v: v == "true",
        "risk_management.max_sector_positions": int,
        "risk_management.drawdown_halt_pct": float,
        "risk_management.drawdown_defensive_pct": float,
        "risk_management.drawdown_exit_review_pct": float,
        "risk_management.trailing_stop_enabled": lambda v: v == "true",
        "risk_management.trailing_stop_follow_tp_targets": lambda v: v == "true",
        "risk_management.protection_gap_alert_delay_seconds": int,
        "research.min_conviction_score": float,
        "research.on_deck_conviction_band": float,
        "research.on_deck_removal_conviction": float,
        "research.min_risk_reward_ratio": float,
        "research.on_deck_rr_floor_margin": lambda v: (float(v) if v not in ("", None) else None),
        "research.on_deck_rr_ceiling_margin": float,
        "research.on_deck_rr_conviction_step": float,
        "research.on_deck_rr_floor": float,
        "research.watchlist_size": int,
        "research.long_term_trend_years": int,
        "research.on_deck_entry_mode": str,
        "research.on_deck_ai_entry_low_refresh_pct": float,
        "research.on_deck_ai_entry_arm_band_pct": float,
        "research.on_deck_retracement_pct": float,
        "research.on_deck_no_dip_pct_gain": float,
        "research.on_deck_recovery_retry_pct": float,
        "research.on_deck_max_dip_low_age_days": float,
        "research.on_deck_ai_stale_decline_block_days": int,
        "research.on_deck_block_breakout_pct": float,
        "research.ai_chosen_stop_tp_enabled": lambda v: v == "true",
        "research.ai_stop_loss_min_pct": float,
        "research.ai_stop_loss_max_pct": float,
        "research.midday_scan_enabled": lambda v: v == "true",
        "research.on_deck_up_ticks_needed": int,
        "research.on_deck_up_ratio_window": int,
        "research.on_deck_history_days": int,
        "research.on_deck_max_size": int,
        "research.on_deck_backfill_enabled": lambda v: v == "true",
        "research.on_deck_swap_margin": float,
        "research.on_deck_backfill_retry_cooldown_minutes": int,
        "research.on_deck_above_gate_recheck_cooldown_minutes": int,
        "research.on_deck_backfill_above_gate_decline_cooldown_minutes": int,
        "research.wash_sale_cooldown_days": int,
        "research.on_deck_auto_deep_dive": lambda v: v == "true",
        "research.position_deep_dive_enabled": lambda v: v == "true",
        "research.position_deep_dive_interval_hours": float,
        "research.auto_buy_cutoff_time": str,
        "research.position_monitor_interval_minutes": int,
        "research.position_monitor_profitable_skip_pct": float,
        "research.position_monitor_event_proximity_pct": float,
        "research.position_monitor_event_cooldown_minutes": int,
        "research.position_monitor_loss_trigger_pct": float,
        "research.position_monitor_loss_retrigger_pct": float,
        "research.pre_open_batch_hours": float,
        "research.model_quick_scan": str,
        "research.model_deep_dive": str,
        "research.model_deeper_dive": str,
        "research.model_pre_open_scan": str,
        "research.model_position_monitor": str,
        "research.model_dip_entry": str,
        "research.model_periodic_deep_dive": str,
        "research.model_portfolio_health": str,
        "trading.paper_trading": lambda v: v == "true",
        "trading.broker": str,
        "trading.auto_execute": lambda v: v == "true",
        "dashboard.ticker_tape_crypto_count": int,
        "dashboard.chart_show_price": lambda v: v == "true",
        "dashboard.chart_show_gate": lambda v: v == "true",
        "dashboard.chart_show_targets": lambda v: v == "true",
        "dashboard.chart_show_highlow": lambda v: v == "true",
        "dashboard.chart_show_markers": lambda v: v == "true",
    }

    # Validate/coerce the WHOLE payload first, before mutating anything (fixed
    # 2026-08-09, GitHub #68) -- this used to write each field into raw/state.config
    # INSIDE the same loop that could still raise HTTPException(422) on a LATER field.
    # Since state.config is read live (not cached) by most of the real buy/sell
    # decision code throughout this file, a single bad field anywhere in the
    # ~82-setting page-wide payload used to leave every field processed before it
    # silently applied and live -- changing real gate thresholds mid-request -- while
    # the 422 response told the user the save failed and config/settings.yaml was never
    # written, with no record anywhere that anything had changed. Now nothing is
    # mutated until every field in the payload has already passed coercion.
    try:
        coerced_values = _coerce_settings_payload(payload, COERCIONS)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))

    if "risk_tier.value" in coerced_values:
        _risk_tier_anchors = state.config.get("risk_tier", {}).get("anchors", {})
        coerced_values = apply_risk_tier_to_settings(coerced_values, _risk_tier_anchors)
        # The _rm_map resync block below reads directly from `payload`, not
        # `coerced_values` -- write the freshly tier-computed values back into
        # `payload` too so that block picks up the real numbers instead of whatever
        # stale value the browser's own (un-updated) form fields held at submit time.
        # See CLAUDE.md's "Risk-Tier Slider" section for the full incident this
        # avoids (found during implementation planning, not a live bug).
        for _dotkey in RISK_TIER_DOTKEYS.values():
            if _dotkey in coerced_values:
                payload[_dotkey] = str(coerced_values[_dotkey])

    for dotkey, value in coerced_values.items():
        section, key = dotkey.split(".", 1)
        raw.setdefault(section, {})[key] = value
        state.config.setdefault(section, {})[key] = value

    # Handle universe_indexes list (sent as JSON array from the frontend)
    if "research.universe_indexes" in payload:
        import json as _json_u
        try:
            indexes = _json_u.loads(payload["research.universe_indexes"])
            raw.setdefault("research", {})["universe_indexes"] = indexes
            state.config.setdefault("research", {})["universe_indexes"] = indexes
            STOCK_UNIVERSE[:] = await asyncio.to_thread(get_universe, indexes)
            logger.info("Universe rebuilt: %d stocks from %s", len(STOCK_UNIVERSE), indexes)
        except Exception as e:
            logger.warning("Failed to rebuild universe: %s", e)

    # Handle scan_times list (sent as JSON array from the frontend)
    if "research.scan_times" in payload:
        import json as _json
        try:
            times_list = _json.loads(payload["research.scan_times"])
            raw.setdefault("research", {})["scan_times"] = times_list
            state.config.setdefault("research", {})["scan_times"] = times_list
        except Exception:
            pass

    if "research.midday_scan_times" in payload:
        import json as _json_mds
        try:
            times_list = _json_mds.loads(payload["research.midday_scan_times"])
            raw.setdefault("research", {})["midday_scan_times"] = times_list
            state.config.setdefault("research", {})["midday_scan_times"] = times_list
        except Exception:
            pass

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Sync cached instance variables that aren't read from config dynamically
    if "research.position_monitor_interval_minutes" in payload:
        state.position_monitor_interval = int(payload["research.position_monitor_interval_minutes"])
    if "research.watchlist_size" in payload:
        state.watchlist_manager.target_size = int(payload["research.watchlist_size"])
    if "research.weak_signal_threshold" in payload:
        state.watchlist_manager.weak_threshold = int(payload["research.weak_signal_threshold"])
    if "research.wash_sale_cooldown_days" in payload:
        state.risk_manager.wash_sale_cooldown_days = int(payload["research.wash_sale_cooldown_days"])
    if "research.scan_times" in payload:
        import json as _json2
        try:
            times_list = _json2.loads(payload["research.scan_times"])
            state.explicit_scan_times = sorted([
                dtime(int(t.split(":")[0]), int(t.split(":")[1]))
                for t in times_list if t
            ])
        except Exception:
            pass
    if "research.midday_scan_times" in payload:
        import json as _json_mds2
        try:
            times_list = _json_mds2.loads(payload["research.midday_scan_times"])
            state.midday_scan_times = sorted([
                dtime(int(t.split(":")[0]), int(t.split(":")[1]))
                for t in times_list if t
            ])
        except Exception:
            pass
    if "research.pre_open_batch_hours" in payload:
        from datetime import timedelta as _tdd, datetime as _dtm
        _open_dt = _dtm.combine(_dtm.today(), state.market_open)
        state.pre_open_batch_time = (_open_dt - _tdd(hours=float(payload["research.pre_open_batch_hours"]))).time()
    _rm = state.risk_manager
    _rm_map = {
        "risk_management.drawdown_halt_pct":       ("drawdown_halt_pct",       100),
        "risk_management.drawdown_defensive_pct":  ("drawdown_defensive_pct",  100),
        "risk_management.drawdown_exit_review_pct":("drawdown_exit_review_pct",100),
        "risk_management.max_position_pct":        ("max_position_pct",        100),
        "risk_management.max_loss_per_trade_pct":  ("max_loss_per_trade_pct",  100),
        "risk_management.min_cash_reserve_pct":    ("min_cash_reserve_pct",    100),
        "risk_management.daily_loss_limit_pct":    ("daily_loss_limit_pct",    100),
        "risk_management.max_sector_positions":    ("max_sector_positions",      1),
    }
    for key, (attr, divisor) in _rm_map.items():
        if key in payload:
            setattr(_rm, attr, float(payload[key]) / divisor)
    if "portfolio.max_positions" in payload:
        _rm.max_positions = max(1, int(payload["portfolio.max_positions"]))
    if "risk_management.sector_concentration_enabled" in payload:
        _rm.sector_concentration_enabled = (
            payload["risk_management.sector_concentration_enabled"] == "true"
        )
    if "portfolio.initial_capital" in payload:
        state.portfolio.initial_capital = float(payload["portfolio.initial_capital"])
    if "trading.paper_trading" in payload:
        paper = payload["trading.paper_trading"] == "true"
        broker = state.order_manager.broker
        if broker is not None:
            broker.paper = paper
            import alpaca_trade_api as _tradeapi
            _api_key  = os.getenv("ALPACA_API_KEY", "")
            _secret   = os.getenv("ALPACA_SECRET_KEY", "")
            _base_url = os.getenv("ALPACA_BASE_URL", "") or (
                "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
            )
            broker.api = _tradeapi.REST(_api_key, _secret, _base_url, api_version="v2")

    # _enforce_on_deck_cap otherwise only runs at the end of a pre-open scan or once at
    # startup (2026-07-19 gap, caught live: lowering on_deck_max_size via Settings updated
    # state.config immediately but the already-over-cap list just sat there untrimmed until
    # the next scan/restart). Trim right away if this save actually touched the cap, rather
    # than making the change wait for a scan that might be many hours off.
    if "research.on_deck_max_size" in payload:
        max_size = state.config["research"].get("on_deck_max_size", 0)
        current = len(state.near_miss_candidates)
        if max_size and current > max_size:
            trimmed = await state._enforce_on_deck_cap("SETTINGS")
            if trimmed:
                # No dedicated WS push here -- matches the existing pre-open-scan/startup
                # call sites, which also don't broadcast a trim; the dashboard's own 60s
                # fetchNearMiss() poll picks up the shorter list on its own next cycle.
                asyncio.create_task(
                    asyncio.to_thread(_save_on_deck_cache, dict(state.near_miss_candidates)))
        elif max_size and current < max_size:
            # Raising the cap has no way to conjure brand-new candidates -- that needs a
            # real scan -- but On Shore usually has a ready supply already (see
            # _refill_on_deck_from_shore's docstring). Caught live (2026-07-19): without
            # this, raising the cap back up did nothing until the next scan/restart, which
            # read as "the setting doesn't work" even though the lower-cap direction (the
            # trim above) worked immediately.
            restored = await state._refill_on_deck_from_shore(max_size - current, "SETTINGS")
            if restored:
                asyncio.create_task(
                    asyncio.to_thread(_save_on_deck_cache, dict(state.near_miss_candidates)))

    return {"status": "ok", "saved": list(payload.keys())}


@app.get("/api/stocks")
async def get_stocks():
    return state.watchlist_manager.get_active()


@app.get("/api/portfolio")
async def get_portfolio():
    return state.get_portfolio_snapshot()


@app.get("/api/signals")
async def get_signals():
    return state.active_signals


@app.get("/api/ticker-signals")
async def get_ticker_signals():
    return state.ticker_signals


@app.get("/api/reports")
async def get_reports():
    out = []
    for ticker, report in state.research_engine.reports.items():
        out.append({
            "ticker": report.ticker,
            "company_name": report.company_name,
            "signal": report.signal.value,
            "conviction": report.conviction_score,
            "risk_level": report.risk_level.value,
            "entry_price": round(report.entry_price, 2),
            "stop_loss": round(report.stop_loss, 2),
            "take_profit_targets": [round(t, 2) for t in report.take_profit_targets],
            "thesis": report.thesis,
            "reasoning": report.reasoning,
            "fundamental_summary": report.fundamental_summary,
            "insider_summary": report.insider_summary,
            "news_summary": report.news_summary,
            "competitive_summary": report.competitive_summary,
            "risk_factors": report.risk_factors,
            "generated_at": report.generated_at.isoformat(),
        })
    return out


@app.post("/api/trigger-batch-scan")
async def trigger_batch_scan():
    """Manually kick off the full pre-open-style universe scan -- both the fresh-install
    banner's "Run Scan Now" button and the main dashboard's "Full Scan" button (renamed
    from "Scan Now" 2026-08-03, which used to only re-vet existing On Deck candidates)
    call this same endpoint. Guarded by _full_scan_in_progress so a second click can't
    launch a duplicate, wastefully concurrent full universe scan. Also checks
    _midday_rescan_in_progress (2026-08-03, owner request, after a real live overlap
    hit twice in one day) -- both hit the same universe/Claude API and, confirmed via
    real logs, exhausted NewsAPI's free-tier rate limit between the two."""
    if state._full_scan_in_progress:
        return {"status": "already_running",
                "message": "A full scan is already in progress — check AI Research Engine for progress"}
    if state._midday_rescan_in_progress:
        return {"status": "already_running",
                "message": "A mid-day re-scan is already in progress — try again once it completes"}
    state._full_scan_in_progress = True
    asyncio.create_task(state._run_full_scan_on_demand())
    return {"status": "started", "message": "Full scan launched — watch AI Research Engine for progress"}


@app.post("/api/trigger-midday-rescan")
async def trigger_midday_rescan():
    """Manually kick off a mid-day re-scan pass, independent of the scheduled
    10:30/13:30 ET times -- exists so the feature can be tried on demand right after
    deploy without waiting for the next scheduled slot. Respects the same
    _midday_rescan_in_progress guard the scheduled firing check uses (2026-07-31
    incident fix) -- refuses to start a second concurrent scan rather than silently
    duplicating real Claude spend against an already-running one. Also checks
    _full_scan_in_progress (2026-08-03, owner request, after a real live overlap with
    the manual "Full Scan" button hit twice in one day)."""
    if state._midday_rescan_in_progress:
        return {"status": "already_running",
                "message": "A mid-day re-scan is already in progress — try again once it completes"}
    if state._full_scan_in_progress:
        return {"status": "already_running",
                "message": "A full scan is already in progress — try again once it completes"}
    state._midday_rescan_in_progress = True
    asyncio.create_task(state._run_midday_rescan("MANUAL"))
    return {"status": "started", "message": "Mid-day re-scan launched — watch AI Research Engine for progress"}


@app.post("/api/trigger-daily-report")
async def trigger_daily_report():
    """Manually generate today's recap (normally fires automatically shortly after market close)."""
    asyncio.create_task(state._generate_daily_report())
    return {"status": "started", "message": "Daily report generating — watch AI Research Engine / ntfy for the result"}


@app.get("/api/research-reports")
async def get_research_reports():
    """Bulk research_reports fetch — moved out of the WS init payload (2026-07-13) since
    it had grown to several MB and was delaying every page load by multiple seconds.
    Frontend fetches this lazily after init instead."""
    return state.research_reports


@app.get("/api/ticker-tape-config")
async def get_ticker_tape_config():
    """Crypto symbol count for the TradingView ticker tape strip — read fresh from
    state.config each call (already updated in-memory by /api/settings, no restart needed)
    so the dashboard picks up a changed count on its next page load or periodic refresh."""
    count = state.config.get("dashboard", {}).get("ticker_tape_crypto_count", 5)
    return {"crypto_count": max(0, min(5, int(count)))}


@app.get("/api/chart-display-config")
async def get_chart_display_config():
    """Which optional overlays the On Deck R/R chart draws (price trace, gate line, target
    lines, high/low labels, per-point markers) — user-toggleable checkboxes on the Settings
    page (2026-07-18), so a stock's chart can be decluttered without touching the underlying
    data. Read fresh from state.config each call, same pattern as /api/ticker-tape-config."""
    d = state.config.get("dashboard", {})
    return {
        "price": d.get("chart_show_price", True),
        "gate": d.get("chart_show_gate", True),
        "targets": d.get("chart_show_targets", True),
        "highlow": d.get("chart_show_highlow", True),
        "markers": d.get("chart_show_markers", True),
    }


@app.get("/api/conviction-gate-config")
async def get_conviction_gate_config():
    """Real-time min_conviction_score + on_deck_rr_floor_margin, for the dashboard's On Deck/
    On Shore card coloring — read fresh from state.config each call so the JS color thresholds
    can never drift out of sync with the live settings again (previously a hardcoded JS
    constant, _MIN_CONVICTION_GATE, that needed a manual re-sync every time this setting
    changed — bit us twice on 2026-07-21). Same fetch-once-on-load pattern as
    /api/ticker-tape-config and /api/chart-display-config."""
    return {
        "min_conviction_score": state.config.get("research", {}).get(
            "min_conviction_score", 7.0
        ),
        "on_deck_rr_floor_margin": state.config.get("research", {}).get(
            "on_deck_rr_floor_margin"
        ),
    }


@app.get("/api/risk-tier-preview")
async def risk_tier_preview(value: float):
    """Read-only -- computes what the 11 real risk-tier factors WOULD become at the
    given tier value, without saving anything. Backs the Settings page's live preview
    table (owner: "i would know more about the risk" -- see
    docs/superpowers/specs/2026-08-21-risk-tier-design.md), so there's no surprise
    between moving the slider and clicking Save."""
    anchors = state.config.get("risk_tier", {}).get("anchors", {})
    computed = compute_risk_tier_settings(value, anchors)
    return {"label": risk_tier_label(value), **computed}


_VERSION_FILE_PATH = str(Path(__file__).resolve().parent.parent / "VERSION")


@app.get("/api/update-status")
async def get_update_status():
    """Compares this install's local VERSION against the latest release on
    the distribution repo (update.releases_repo in config) — no credential
    needed, since that repo is public. The frontend polls this endpoint
    periodically (2026-08-12, owner: "if i have to refresh to get it thats
    a problem" — was fetch-once-on-load only) so the badge can appear on an
    already-open dashboard without a manual reload. The real GitHub lookup
    itself is cached for update.check_interval_minutes (default 60) via
    state._update_status_cache — frequent client polling hits this cheap
    local endpoint, not GitHub, keeping this well under GitHub's
    60-req/hour-per-IP unauthenticated limit even with several tabs/devices
    open at once. A fetch failure degrades to the last good cached result if
    one exists, or to 'no update info available' on the very first call —
    this endpoint must never break the dashboard's own load."""
    current = read_local_version(_VERSION_FILE_PATH) or "v0.0.0"
    repo = state.config.get("update", {}).get("releases_repo", "")
    market_open = state._is_market_open()

    if not repo:
        return {
            "current": current,
            "latest": None,
            "available": False,
            "notes": "",
            "severity": "routine",
            "market_open": market_open,
        }

    check_interval = timedelta(
        minutes=state.config.get("update", {}).get("check_interval_minutes", 60)
    )
    cache_is_fresh = (
        state._update_status_cache is not None
        and state._update_status_cache_time is not None
        and datetime.now() - state._update_status_cache_time < check_interval
    )
    if not cache_is_fresh:
        try:
            state._update_status_cache = fetch_latest_release(repo)
            state._update_status_cache_time = datetime.now()
        except Exception:
            pass  # keep serving the last known-good cached release, if any

    release = state._update_status_cache
    if release is None:
        return {
            "current": current,
            "latest": None,
            "available": False,
            "notes": "",
            "severity": "routine",
            "market_open": market_open,
        }

    return {
        "current": current,
        "latest": release["tag_name"],
        "available": is_newer(current, release["tag_name"]),
        "notes": release["notes"],
        "severity": release["severity"],
        "market_open": market_open,
    }


_ABOUT_NAME = "Hilton's AITrading"
_ABOUT_DESCRIPTION = (
    "AI-powered research and autonomous trading for long-only U.S. equities. "
    "Claude continuously screens the S&P 500/400/600 universe, tracks qualifying "
    "setups on an On Deck watchlist, and buys automatically once conviction, "
    "risk/reward, and a confirmed technical trigger all align — with a graduated "
    "trailing stop protecting every position from entry to exit."
)


@app.get("/api/about")
async def get_about():
    """Backs the About panel (2026-08-20, owner request — "i see no update
    buttons" led to "maybe i need an about button... so i know what version
    we are running"). Deliberately reuses fetch_recent_releases against the
    same releases_repo Apply Update already reads from, rather than
    authoring/maintaining a separate changelog — the real release notes
    already exist the moment a release is cut. No caching (unlike
    /api/update-status above): this is a manual, user-initiated, low-
    frequency action, not something polled every 60s from an open tab, so
    the request-volume concern that caching exists for there doesn't apply
    here. Degrades to an empty release list (never raises) if the releases
    repo is unreachable, same fail-open philosophy as the update-status
    endpoint — a GitHub hiccup should never break this panel from at least
    showing the current version."""
    current = read_local_version(_VERSION_FILE_PATH) or "v0.0.0"
    repo = state.config.get("update", {}).get("releases_repo", "")
    releases: list[dict] = []
    if repo:
        try:
            releases = fetch_recent_releases(repo, limit=15)
        except Exception:
            pass
    return {
        "name": _ABOUT_NAME,
        "description": _ABOUT_DESCRIPTION,
        "current_version": current,
        "releases": releases,
    }


_INSTALL_ROOT = str(Path(__file__).resolve().parent.parent)


def _restart_service_after_delay():
    """Runs in a background thread so the HTTP response for the apply
    request can reach the client before the service (and this very
    process) restarts."""
    time.sleep(2)
    subprocess.run(["systemctl", "restart", "aitrading"], check=False)


@app.post("/api/apply-update")
async def apply_update():
    """Manually-triggered only — never called automatically. Downloads the
    latest release, replaces only is_path_updatable()-allowed paths,
    reinstalls dependencies if requirements.txt changed, updates VERSION,
    and restarts. No automatic rollback in v1 (see the design spec's Error
    Handling section) — a download/extract failure aborts before touching
    the live install at all; a pip install failure aborts before
    restarting, so a bad release never leaves the service down from this
    endpoint's own actions.

    Guarded by _apply_update_in_progress (2026-08-12, owner concern: "someone
    would push the button again") — the frontend already disables the button
    it was clicked from, but closing and reopening the update panel rebuilds
    a fresh, un-disabled one with no memory of an apply already running.
    Same in-progress-flag pattern as _full_scan_in_progress
    (/api/trigger-batch-scan)."""
    if state._apply_update_in_progress:
        return {"status": "already_applying",
                "detail": "An update is already being applied — wait for it to finish."}
    state._apply_update_in_progress = True
    try:
        repo = state.config.get("update", {}).get("releases_repo", "")
        if not repo:
            return {"status": "error", "detail": "update.releases_repo not configured"}

        try:
            release = fetch_latest_release(repo)
        except Exception as exc:
            return {"status": "error", "detail": f"could not fetch latest release: {exc}"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = str(Path(tmp_dir) / "release.tar.gz")
            try:
                response = requests.get(release["download_url"], timeout=60)
                response.raise_for_status()
                Path(archive_path).write_bytes(response.content)
            except Exception as exc:
                return {"status": "error", "detail": f"could not download release archive: {exc}"}

            try:
                extract_dir = str(Path(tmp_dir) / "extracted")
                Path(extract_dir).mkdir()
                extracted_root = extract_release_archive(archive_path, extract_dir)
            except Exception as exc:
                return {"status": "error", "detail": f"could not extract release archive: {exc}"}

            old_requirements_path = Path(_INSTALL_ROOT) / "requirements.txt"
            old_requirements = (
                old_requirements_path.read_text() if old_requirements_path.exists() else ""
            )
            new_requirements_path = Path(extracted_root) / "requirements.txt"
            new_requirements = (
                new_requirements_path.read_text() if new_requirements_path.exists() else old_requirements
            )
            needs_pip_install = requirements_changed(old_requirements, new_requirements)

            copy_updatable_files(extracted_root, _INSTALL_ROOT)

            if needs_pip_install:
                pip_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    cwd=_INSTALL_ROOT,
                    capture_output=True,
                    text=True,
                )
                if pip_result.returncode != 0:
                    return {
                        "status": "error",
                        "detail": f"pip install failed, service NOT restarted: {pip_result.stderr}",
                    }

            write_local_version(_VERSION_FILE_PATH, release["tag_name"])

        threading.Thread(target=_restart_service_after_delay, daemon=True).start()

        return {"status": "applying", "target_version": release["tag_name"]}
    finally:
        state._apply_update_in_progress = False


def _chart_price_history(nm: dict) -> list:
    """Price series used for chart display only — real price_history whenever it has
    enough points to plot, otherwise falls back to _debug_price_history if present. Real
    data always wins the moment it exists (>=2 points), so this self-resolves the instant
    near_miss_monitor_loop appends genuine ticks — no cleanup needed once the market opens.
    _debug_price_history is written only by /api/debug/seed-on-deck-chart and is never read
    by near_miss_monitor_loop's promotion logic (window_low / recovery-check), which uses
    nm["price_history"] directly — so seeded chart-preview data can never influence a real
    buy/sell decision, only what these two display endpoints return."""
    real = nm.get("price_history", [])
    if len(real) >= 2:
        return real
    return nm.get("_debug_price_history", []) or real


def _windowed_price_history(nm: dict, history_days: int) -> list[tuple[float, float]]:
    """Chart price history trimmed to the last history_days -- shared cutoff logic used by
    _windowed_dip (below) and by remove_on_deck_candidate's reference-peak capture
    (2026-07-29, BRO fix -- see _recent_window_high's docstring)."""
    now_ts = datetime.now().timestamp()
    cutoff = now_ts - history_days * 86400
    return [p for p in _chart_price_history(nm) if p[0] >= cutoff]


def _windowed_dip(nm: dict, retracement_pct: float, history_days: int) -> dict | None:
    """dip_summary() computed over the SAME windowed slice near_miss_monitor_loop's actual
    buy-trigger check uses — real bug found live (2026-07-18): the display endpoints below
    were calling dip_summary() on the full, now-permanently-kept _chart_price_history(nm)
    (see the 'keep it all' change earlier the same day), so the displayed dip/retracement
    line could reference a low from well outside the configured on_deck_history_days window
    (caught directly: WMB's displayed low was dated over a month back, 39 days, against a
    30-day window) — showing a target line that didn't match what was actually deciding the
    buy. The R/R curve/sparkline itself is correctly still drawn from the FULL history (that
    part of "keep it all" was intentional and correct); only the dip/target-line calculation
    needed to stay windowed to agree with the real trigger logic."""
    return dip_summary(_windowed_price_history(nm, history_days), retracement_pct)


def _recent_window_high(price_history: list[tuple[float, float]]) -> float | None:
    """The highest close anywhere in a windowed price-history slice (2026-07-29, BRO
    incident) -- the reference peak a manually-removed On Deck ticker must clear before
    _check_price_based_unblocks restores its eligibility. Deliberately NOT dip_summary's
    "peak" field: that peak is measured strictly BEFORE the tracked low, which for a
    STALE low (the exact situation this feature exists for -- "AI declined this entry"
    fires specifically when the low is old and mostly recovered from already) can sit
    right at the edge of the history window, leaving almost no real data before it. Real
    incident: BRO's dip had a 29.2-day-old low, and its own dip_summary peak came back
    $63.95 -- a stale, artifact-of-the-window-boundary value -- while BRO was already
    trading at ~$74.6, comfortably ABOVE that "peak" from the moment the block was set.
    _price_clears_block_breakout cleared on the very next tick (~60s later), auto-
    restoring a ticker the user had just explicitly removed, then it got backfilled and
    AI-declined again -- a real, wasteful loop, not a rare edge case, since a stale-dip
    decline is BY DEFINITION the scenario this feature runs against. Taking the max close
    over the whole window instead (not just before some low) directly answers "has this
    stock made a genuinely new high recently" regardless of where any dip's low sits."""
    if not price_history:
        return None
    return max(p[1] for p in price_history)


@app.get("/api/near-miss")
async def get_near_miss():
    """Near-miss candidates — BUY-signal, conviction-qualified stocks rejected at pre-open
    only for R/R, currently being monitored (free, no Claude) for a price recovery + confirmed
    uptick. Fetched lazily by the dashboard, same pattern as /api/research-reports.

    Includes a compact rr_sparkline (R/R values only, downsampled — see
    src.research.rr_curve.rr_sparkline) for each candidate's card-level chart. The full
    price_history stays internal-only; a whole trading day of 60s samples would bloat this
    list payload, so only the derived, capped series is exposed here — the uncapped version
    is available per-ticker via /api/near-miss/{ticker}/history for the detail-view chart.

    Also includes "dip" (see src.research.rr_curve.dip_summary) — the same peak/low/depth
    numbers that actually decide the buy trigger in "retracement" mode, added 2026-07-18 so
    the card can show a plain-English "here's what's happening" line instead of just the raw
    R/R chart. None when there's no measurable dip yet (e.g. a brand-new candidate).

    "entry_mode" echoes the current research.on_deck_entry_mode setting. "dip_target_rr" and
    "ai_entry_target_rr" convert each mode's target PRICE into its equivalent point on the R/R
    axis (via rr_at_price — the same formula the chart itself plots), so the frontend can draw
    a dashed target line directly on the R/R sparkline instead of needing a second, separate
    price chart. Both are included regardless of which mode is active — a deliberate choice
    (2026-07-18) so a user running "ai" mode can still see how the AI's recommended entry
    compares to the flat retracement percentage over time, and vice versa. ai_entry_target_rr
    is None whenever there's no CURRENT AI recommendation (never computed yet, or stale
    against a new deeper low since it was computed) — same "don't show a stale number" rule
    the buy-trigger check itself follows.
    """
    default_stop_pct = state.config["take_profit"]["stop_loss_pct"]
    retracement_pct = state.config["research"].get("on_deck_retracement_pct", 20.0)
    entry_mode = state.config["research"].get("on_deck_entry_mode", "ai")
    history_days = state.config["research"].get("on_deck_history_days", 30)
    base_rr = state.config["research"]["min_risk_reward_ratio"]
    min_conviction = state.config["research"]["min_conviction_score"]
    rr_step = state.config["research"].get("on_deck_rr_conviction_step", 0.1)
    rr_floor = state.config["research"].get("on_deck_rr_floor", 1.5)
    result = {}
    for ticker, nm in state.near_miss_candidates.items():
        fair_value = nm.get("fair_value_estimate", 0.0)
        # Per-candidate stop % (2026-07-18), not the global default — see near_miss_monitor_
        # loop's matching comment. Keeps the chart's R/R in sync with the badge's, both now
        # reflecting the same stock-specific stop Claude actually recommended (and the real
        # order would actually place), not one flat percentage applied to every stock.
        stop_loss_pct = nm.get("stop_loss_pct", default_stop_pct)
        dip = _windowed_dip(nm, retracement_pct, history_days)
        dip_target_rr = rr_at_price(dip["retracement_target"], fair_value, stop_loss_pct) if dip else None
        ai_entry_target_rr = None
        if (dip is not None and nm.get("ai_entry_price") is not None
                and nm.get("ai_entry_low_ref") == dip["low"]):
            ai_entry_target_rr = rr_at_price(nm["ai_entry_price"], fair_value, stop_loss_pct)
        # required_rr (2026-07-18): this candidate's own conviction-scaled threshold, falling
        # back to computing it fresh for any candidate created before the field existed.
        required_rr = nm.get("required_rr")
        if required_rr is None:
            required_rr = _required_rr(nm.get("conviction_score", min_conviction), min_conviction, base_rr, rr_step, rr_floor)
        result[ticker] = {
            **{k: v for k, v in nm.items() if k not in ("price_history", "_debug_price_history")},
            "rr_sparkline": rr_sparkline(_chart_price_history(nm), fair_value, stop_loss_pct),
            "price_sparkline": price_sparkline(_chart_price_history(nm)),
            "dip": dip,
            "dip_target_rr": dip_target_rr,
            "ai_entry_target_rr": ai_entry_target_rr,
            "required_rr": required_rr,
            "entry_mode": entry_mode,
        }
    return result


@app.get("/api/today-scan-rejects")
async def get_today_scan_rejects():
    """"Today's Scan" tab (2026-07-19) — stocks that cleared quick_screen and got a real
    analysis via the pre-open Phase 2 universe scan TODAY, but didn't qualify for On Deck
    (conviction/signal/population-floor gate). Distinct from every other reason a
    research_reports entry might carry today's date (persist-check re-vet of an existing
    candidate, an hourly position-monitor re-check, a manual Scan Now or Deep Dive) via the
    "source": "universe_scan" tag Phase 2's _persist_report sets — see that call site for
    why a plain date check alone isn't enough to distinguish these.

    Computes rr/required_rr fresh here (not stored — these reports never went through
    _build_on_deck_entry/_passes_on_deck_rr_gate since they were rejected before reaching
    that step) so the tab can sort by the same tiered (is_buy_eligible, composite_score)
    key used for ranking everywhere else in the app (see _on_deck_ranking_key — fixed
    2026-08-12, this endpoint originally sorted by the flat composite score alone, which
    let a non-buy-eligible candidate with an inflated R/R outlier top the list ahead of
    genuinely strong ones; see docs/CLAUDE_HISTORY.md's 2026-08-12 COTY entry).

    R/R uses a LIVE price (2026-07-19 follow-up), not the frozen entry_price from whenever
    this morning's scan happened to analyze this ticker — same free (yfinance, no Claude)
    trick near_miss_monitor_loop already uses for On Deck: derive this stock's own stop % from
    Claude's original entry/stop recommendation, then apply that % to the CURRENT price to get
    a trailing stop, rather than using the original absolute stop_loss forever (which would
    make R/R blow up or go negative for the wrong reason as price moves away from this
    morning's entry point). Conviction/fair_value/thesis stay frozen either way — only the
    price side of R/R is live. Fetched fresh on every request (no scheduled background loop),
    so this is only ever as expensive as someone actually opening the tab, and always
    reflects the current price the moment they do.

    A ticker kicked off On Deck (conviction drop, over-cap trim) reappears here immediately
    via _mark_universe_reject — see that method's docstring. A ticker the user manually
    removed via the On Deck card's X button is explicitly excluded instead (on_deck_blocked
    check below) — that's a deliberate "don't show me this" judgment call, not a quality
    verdict, and should stay off both lists for as long as its block lasts."""
    today_str = state._now_et().strftime("%Y-%m-%d")
    base_rr = state.config["research"]["min_risk_reward_ratio"]
    min_conviction = state.config["research"]["min_conviction_score"]
    rr_step = state.config["research"].get("on_deck_rr_conviction_step", 0.1)
    rr_floor = state.config["research"].get("on_deck_rr_floor", 1.5)
    default_stop_pct = state.config["take_profit"]["stop_loss_pct"]
    held = set(state.portfolio.positions.keys())

    candidates = {}
    for ticker, r in state.research_reports.items():
        if r.get("source") != "universe_scan":
            continue
        if not r.get("generated_at", "").startswith(today_str):
            continue
        if ticker in state.near_miss_candidates or ticker in held:
            continue
        if state._is_on_deck_blocked(ticker):
            continue  # user explicitly removed this one -- respect the block, don't resurface it
        if state._wash_sale_blocked(ticker):
            continue  # can't legally be bought right now -- see _wash_sale_blocked's docstring
        candidates[ticker] = r

    # Concurrent, capped quote + chart-history fetch (2026-07-19 follow-up, full parity with
    # On Deck cards including the R/R sparkline). Measured directly before building this: a
    # single 30-day/15-min history fetch (0.24s) is actually FASTER than a live quote (0.67s)
    # on this box, so doing both per ticker (sequentially within one semaphore slot) adds
    # roughly 4-5s on top of the quote-only ~7s for ~72 tickers, not the much bigger jump
    # that seemed likely before actually measuring it. Reuses _fetch_price_history verbatim
    # (the exact same real-history-plus-live-price-anchor helper On Deck uses when a new
    # candidate is first added) rather than re-implementing the fetch/fallback logic here.
    sem = asyncio.Semaphore(10)
    async def _quote_and_history(ticker):
        async with sem:
            try:
                q = await state.market_data.get_quote(ticker)
                price = q.price
            except Exception:
                price = None
            history = await state._fetch_price_history(ticker, price)
            return ticker, price, history
    fetched = await asyncio.gather(*(_quote_and_history(t) for t in candidates))
    quotes = {t: p for t, p, _h in fetched}
    histories = {t: h for t, _p, h in fetched}

    result = {}
    for ticker, r in candidates.items():
        conviction = r.get("conviction", 0) or 0
        entry_price = r.get("entry_price", 0.0) or 0.0
        stop_loss = r.get("stop_loss", 0.0) or 0.0
        fair_value = r.get("fair_value_estimate", 0.0) or 0.0
        # Fall back to this morning's entry_price if the live quote failed -- same fail-open
        # principle used everywhere else a live price feed can be unavailable.
        live_price = quotes.get(ticker) or entry_price
        stop_pct = _derive_stop_pct(entry_price, stop_loss, default_stop_pct)
        live_stop = live_price * (1 - stop_pct / 100)
        risk = live_price - live_stop
        rr = (fair_value - live_price) / risk if risk > 0 and fair_value > 0 else 0.0
        required_rr = _required_rr(conviction, min_conviction, base_rr, rr_step, rr_floor)
        margin = r.get("margin_of_safety_pct", 0.0) or 0.0
        score = conviction + margin / 10 + (rr - required_rr) * 2
        result[ticker] = {
            "ticker": ticker,
            "company_name": r.get("company_name", ticker),
            "sector": r.get("sector", ""),
            "business_summary": r.get("business_summary", ""),
            "thesis": r.get("thesis", ""),
            "signal": r.get("signal", ""),
            "conviction_score": conviction,
            "rr": rr,
            "required_rr": required_rr,
            "last_price": live_price,
            "fair_value_estimate": fair_value,
            "margin_of_safety_pct": margin,
            "generated_at": r.get("generated_at", ""),
            "rr_sparkline": rr_sparkline(histories.get(ticker, []), fair_value, stop_pct),
            "price_sparkline": price_sparkline(histories.get(ticker, [])),
            "_score": score,
        }
    # Sorted by the same tiered (is_buy_eligible, composite_score) key On Deck's own
    # ranking uses (fixed 2026-08-12, COTY incident — conviction 3.2, signal NO ACTION,
    # but an inflated R/R of 12.00 from a tight stop against a big fair-value gap put it
    # at #1 on this list under the old flat-_score sort, ahead of genuinely strong,
    # buy-eligible candidates). _on_deck_ranking_key ranks every buy-eligible candidate
    # ahead of every non-eligible one regardless of composite score, falling back to the
    # flat score only to break ties within the same tier — same fix already applied to
    # every On Deck ranking site (_on_deck_ranking_key's docstring, 2026-07-31). rank
    # added post-sort for the frontend's #N badge, matching On Deck's card numbering.
    ranked = sorted(
        result.items(),
        key=lambda kv: _on_deck_ranking_key(
            kv[1]["conviction_score"], kv[1]["margin_of_safety_pct"],
            kv[1]["rr"], kv[1]["required_rr"], min_conviction,
        ),
        reverse=True,
    )
    for i, (_ticker, entry) in enumerate(ranked):
        entry["rank"] = i + 1
    return dict(ranked)


@app.get("/api/near-miss/{ticker}/history")
async def get_near_miss_history(ticker: str):
    """R/R-over-time chart data for an active On Deck candidate — reconstructs R/R at each
    recorded price point using the same stop-trailing formula near_miss_monitor_loop uses
    live, so the chart matches exactly what has been driving (or not yet driving) a promotion
    decision. 404s once the candidate is bought or otherwise leaves near_miss_candidates —
    the chart is only meaningful while a candidate is actively being watched."""
    ticker = ticker.upper()
    nm = state.near_miss_candidates.get(ticker)
    if not nm:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not an active On Deck candidate")
    stop_loss_pct = nm.get("stop_loss_pct", state.config["take_profit"]["stop_loss_pct"])
    fair_value = nm.get("fair_value_estimate", 0.0)
    # This candidate's own conviction-scaled threshold (2026-07-18), not the flat config
    # base value — see _required_rr. Falls back to computing it fresh for any candidate
    # that predates the field being stored.
    base_rr = state.config["research"]["min_risk_reward_ratio"]
    min_rr = nm.get("required_rr")
    if min_rr is None:
        min_conviction = state.config["research"]["min_conviction_score"]
        rr_step = state.config["research"].get("on_deck_rr_conviction_step", 0.1)
        rr_floor = state.config["research"].get("on_deck_rr_floor", 1.5)
        min_rr = _required_rr(nm.get("conviction_score", min_conviction), min_conviction, base_rr, rr_step, rr_floor)
    points = rr_points(_chart_price_history(nm), fair_value, stop_loss_pct)
    # Same dip/entry-target fields as /api/near-miss (2026-07-18) so the bigger modal chart
    # can draw the identical dashed target line the card sparkline does — see that
    # endpoint's docstring for why both modes' targets are always included regardless of
    # which one is active.
    retracement_pct = state.config["research"].get("on_deck_retracement_pct", 20.0)
    entry_mode = state.config["research"].get("on_deck_entry_mode", "ai")
    history_days = state.config["research"].get("on_deck_history_days", 30)
    dip = _windowed_dip(nm, retracement_pct, history_days)
    dip_target_rr = rr_at_price(dip["retracement_target"], fair_value, stop_loss_pct) if dip else None
    ai_entry_target_rr = None
    if (dip is not None and nm.get("ai_entry_price") is not None
            and nm.get("ai_entry_low_ref") == dip["low"]):
        ai_entry_target_rr = rr_at_price(nm["ai_entry_price"], fair_value, stop_loss_pct)
    return {
        "ticker": ticker, "fair_value_estimate": fair_value, "min_rr": min_rr, "points": points,
        "dip": dip, "dip_target_rr": dip_target_rr, "ai_entry_target_rr": ai_entry_target_rr,
        "entry_mode": entry_mode, "ai_entry_price": nm.get("ai_entry_price"),
        "ai_entry_reasoning": nm.get("ai_entry_reasoning", ""),
    }


@app.get("/api/today-scan-rejects/{ticker}/history")
async def get_today_scan_reject_history(ticker: str):
    """Full-resolution R/R-over-time chart for an On Shore ticker (2026-07-19) — the click-
    through modal's chart and its "expand" view were both hardcoded to only work for On Deck
    candidates (fetching /api/near-miss/{ticker}/history, which 404s for anything not
    currently in near_miss_candidates), so opening either for an On Shore stock silently
    showed nothing. Mirrors that endpoint's response shape (points/min_rr, dip/ai_entry
    fields present but always None since retracement/AI-entry tracking is an On-Deck-only
    concept that doesn't apply to an untracked snapshot) so the same frontend rendering code
    handles both without a fork. Unlike /api/today-scan-rejects (which returns a capped/
    downsampled rr_sparkline for the whole list), this fetches the FULL undownsampled
    history for just the one requested ticker, same "list is capped, per-ticker detail isn't"
    split already used for On Deck's own two endpoints."""
    ticker = ticker.upper()
    r = state.research_reports.get(ticker)
    if not r or r.get("source") != "universe_scan":
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not a today's-scan reject")
    entry_price = r.get("entry_price", 0.0) or 0.0
    stop_loss = r.get("stop_loss", 0.0) or 0.0
    fair_value = r.get("fair_value_estimate", 0.0) or 0.0
    default_stop_pct = state.config["take_profit"]["stop_loss_pct"]
    stop_loss_pct = _derive_stop_pct(entry_price, stop_loss, default_stop_pct)
    try:
        quote = await state.market_data.get_quote(ticker)
        live_price = quote.price
    except Exception:
        live_price = entry_price
    history = await state._fetch_price_history(ticker, live_price)
    base_rr = state.config["research"]["min_risk_reward_ratio"]
    min_conviction = state.config["research"]["min_conviction_score"]
    rr_step = state.config["research"].get("on_deck_rr_conviction_step", 0.1)
    rr_floor = state.config["research"].get("on_deck_rr_floor", 1.5)
    conviction = r.get("conviction", 0) or 0
    min_rr = _required_rr(conviction, min_conviction, base_rr, rr_step, rr_floor)
    points = rr_points(history, fair_value, stop_loss_pct)
    return {
        "ticker": ticker, "fair_value_estimate": fair_value, "min_rr": min_rr, "points": points,
        "dip": None, "dip_target_rr": None, "ai_entry_target_rr": None,
        "entry_mode": "none", "ai_entry_price": None, "ai_entry_reasoning": "",
    }


@app.get("/api/stock-chart/{ticker}")
async def get_stock_chart(ticker: str):
    """Daily OHLCV + technicals for any ticker — backs the candlestick chart in the
    manual Deep Dive modal.  Works for any ticker the owner types, not just held
    positions.  ref_lines come from the most-recent cached research report when one
    exists, so entry/stop/TP/fair-value price lines show up automatically."""
    import math as _math

    import yfinance as yf

    def _fin(v):
        """Return None for NaN/Inf (yfinance occasionally returns these); otherwise v.
        JSON cannot serialize NaN — a single bad bar crashes the whole endpoint with 500."""
        return None if (v is None or (isinstance(v, float) and not _math.isfinite(v))) else v

    ticker = ticker.upper()
    try:
        # Fetch the 1y/1d yfinance history exactly once and hand it to both calls below
        # (fixed 2026-08-18, code-review finding) -- they used to each independently
        # re-fetch the identical history, doubling real yfinance network calls per chart
        # open (Deep Dive modal, position detail, every ticker switch).
        hist = await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="1y", interval="1d"))
        bars, technicals = await asyncio.gather(
            state.market_data.get_historical(ticker, period="1y", interval="1d", hist=hist),
            state.market_data.get_technicals(ticker, hist=hist),
        )
    except Exception as e:
        logger.warning("stock_chart failed for %s: %s", ticker, e)
        return {"points": [], "technicals": None, "ref_lines": None}
    points = [
        {"time": b["date"], "open": b["open"], "high": b["high"],
         "low": b["low"], "close": b["close"], "volume": b.get("volume", 0)}
        for b in bars
        if all(_fin(b.get(k)) is not None for k in ("open", "high", "low", "close"))
    ]
    report = state.research_reports.get(ticker)
    ref_lines = None
    if report:
        ref_lines = {
            "entry": report.get("entry_price"),
            "stop": report.get("stop_loss"),
            "take_profits": report.get("take_profit_targets") or [],
            "fair_value": report.get("fair_value_estimate"),
        }
    return {
        "points": points,
        "technicals": {
            "sma_50": _fin(technicals.sma_50),
            "sma_200": _fin(technicals.sma_200),
            "support_level": _fin(technicals.support_level),
            "resistance_level": _fin(technicals.resistance_level),
        } if technicals else None,
        "ref_lines": ref_lines,
    }


@app.get("/api/position/{ticker}/history")
async def get_position_history(ticker: str):
    """Price-over-time chart data for a currently HELD position (2026-07-20) — deliberately a
    PRICE chart with real entry/stop/take-profit reference lines, not the R/R curve On Deck
    and On Shore use. Those two charts answer "should this become a buy" (R/R is the relevant
    axis, since the stop/target are only proposed). A held position already answers that
    question — what's actually useful now is what the real, currently-live broker-facing
    levels are relative to where price has actually gone, which is a price chart, not R/R.

    404s the moment the ticker is no longer held — same "only meaningful while it's still the
    thing it claims to be" rule as the On Deck/On Shore history endpoints.

    Reuses _fetch_price_history verbatim (same real ~30-day/15-min history + live-price
    anchor already used for every other chart in this app) rather than a third copy of the
    fetch/fallback logic."""
    ticker = ticker.upper()
    pos = state.portfolio.positions.get(ticker)
    if not pos:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not a currently held position")
    history = await state._fetch_price_history(ticker, pos.current_price)
    points = [{"t": ts, "price": round(price, 2)} for ts, price in history]
    reasoning = state.buy_reasoning.get(ticker, {})
    return {
        "ticker": ticker,
        "points": points,
        "entry_price": pos.entry_price,
        "stop_loss": pos.stop_loss,
        "trailing_stop": pos.trailing_stop,
        "take_profit_targets": list(pos.take_profit_targets or []),
        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
        "ai_entry_price": reasoning.get("ai_entry_price"),
        "ai_entry_reasoning": reasoning.get("ai_entry_reasoning", ""),
    }


@app.post("/api/debug/seed-on-deck-chart/{ticker}")
async def debug_seed_on_deck_chart(ticker: str):
    """TEST-ONLY, display-safe: seeds a synthetic dip-and-recovery price series, anchored to
    a real live quote, into nm["_debug_price_history"] — a field near_miss_monitor_loop's
    promotion logic never reads (it uses nm["price_history"] directly). This lets the card
    sparkline and modal chart be visually verified before real market-hours ticks exist,
    with zero possibility of influencing a real buy/sell decision. Does not touch
    price_history, last_price, direction, or streak — none of the real monitoring state.
    Real price_history automatically takes over display (see _chart_price_history) the
    moment near_miss_monitor_loop records 2+ genuine ticks, so this needs no cleanup."""
    ticker = ticker.upper()
    nm = state.near_miss_candidates.get(ticker)
    if not nm:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not an active On Deck candidate")
    quote = await state.market_data.get_quote(ticker)
    price = quote.price
    if price <= 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="Could not fetch a valid quote")
    # ~5 hours at 5-minute bars (matches a realistic morning session length), a smooth
    # dip-and-recovery shape (peak decline ~3.5%, small wobble so it doesn't look too
    # perfectly synthetic) ending exactly at today's real quote.
    n = 60
    offsets = []
    for i in range(n):
        t = i / (n - 1)
        dip = math.sin(t * math.pi) * 0.035  # 0 -> peak mid-session -> 0
        wobble = math.sin(t * 23) * 0.0015
        offsets.append(dip + wobble)
    offsets[-1] = 0.0  # last point lands exactly on the real quote
    now_ts = datetime.now().timestamp()
    nm["_debug_price_history"] = [
        (now_ts - (n - 1 - i) * 300, round(price * (1 + off), 2))
        for i, off in enumerate(offsets)
    ]
    # _save_on_deck_cache already keeps _debug_price_history (it only strips the real
    # price_history field) and _load_on_deck_cache never resets it on load — so persisting
    # here is enough for the seeded preview to survive both a page refresh (already true,
    # it's server-side state) and a full service restart.
    asyncio.create_task(asyncio.to_thread(_save_on_deck_cache, dict(state.near_miss_candidates)))
    return {"ticker": ticker, "seeded_points": n, "span_hours": round(n * 5 / 60, 1),
            "final_price": price,
            "note": "display-only — real price_history and promotion logic untouched; persisted to survive restarts"}


@app.get("/api/stock-report/{ticker}")
async def get_stock_report(ticker: str):
    """Return best available analysis for a ticker: research cache → deep dive → DB summary."""
    ticker = ticker.upper()
    if ticker in state.research_reports:
        return state.research_reports[ticker]
    if ticker in state.deep_dive_reports:
        dd = state.deep_dive_reports[ticker]
        return {**dd, "source": "deep_dive"}
    summary = state.watchlist_manager.get_stock_summary(ticker)
    if summary:
        return summary
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="No analysis found")


@app.post("/api/rebuy-legacy-positions")
async def rebuy_legacy_positions():
    """Sell round-share (legacy) positions and rebuy as notional dollar orders to get T1/T2/T3."""
    asyncio.create_task(state._rebuy_legacy_positions())
    return {"status": "started", "message": "Legacy position rebuy initiated — check AI Research Engine feed"}


@app.post("/api/recalibrate-drawdown-baseline")
async def recalibrate_drawdown_baseline():
    """Explicit, owner-triggered recovery for a false-positive drawdown halt (2026-08-08,
    GitHub #46) — Portfolio.peak_value only ever ratchets upward; there's no mechanism
    anywhere that lowers it, so a manual balance change (a deliberate paper-account reset,
    or a real deposit/withdrawal) reads as a catastrophic trading loss and permanently
    blocks every automated buy path (RiskManager.check_drawdown), with the only previous
    recovery being a raw DB edit -- something CLAUDE.md's own operational rules already
    flag as dangerous to do without care. This is the safe, auditable, in-app alternative:
    fetches the REAL current equity directly from Alpaca (never trusts local state,
    per the same rule) and resets peak_value to it, so drawdown tracking starts fresh
    from right now. Deliberately does NOT touch day_start_value/day_start_date (a
    separate check, check_daily_loss, not implicated in this bug) or attempt to guess
    WHY the balance changed -- that judgment call belongs to whoever clicks the button,
    which is exactly why this is a manual, explicit, confirmable action rather than
    something that fires automatically."""
    if not state.order_manager.broker:
        return {"status": "error", "message": "No broker connected"}
    try:
        account = await state.order_manager.broker.get_account()
    except Exception as e:
        return {"status": "error", "message": f"Could not fetch real Alpaca balance: {e}"}

    old_peak = state.portfolio.peak_value
    new_peak = account.equity
    old_dd_state = state.risk_manager.check_drawdown(state.portfolio)
    state.portfolio.peak_value = new_peak
    await state.portfolio._save_state()
    new_dd_state = state.risk_manager.check_drawdown(state.portfolio)

    entry = state.add_ai_log("SYSTEM", "RISK",
        f"Drawdown baseline recalibrated by owner — peak_value ${old_peak:,.2f} -> "
        f"${new_peak:,.2f} (real Alpaca equity). State: {old_dd_state} -> {new_dd_state}.",
        "warning")
    await state.broadcast({"type": "ai_log", "entry": entry})
    return {
        "status": "ok", "old_peak_value": old_peak, "new_peak_value": new_peak,
        "old_drawdown_state": old_dd_state, "new_drawdown_state": new_dd_state,
    }


@app.get("/api/broker-status")
async def get_broker_status():
    return {
        "connected": state.broker_connected,
        "broker": state.config["trading"]["broker"],
        "paper_trading": state.config["trading"]["paper_trading"],
    }


@app.get("/api/dashboard-poll")
async def get_dashboard_poll():
    """HTTP-polling fallback for a client whose WebSocket can't stay connected (2026-07-23,
    real incident: a user's browser silently blocked the /ws upgrade handshake specifically —
    confirmed the server and session auth were both healthy the whole time via a fresh
    connection from elsewhere — leaving their dashboard frozen for over an hour with no
    visible sign anything was wrong). Returns the exact same payload the WebSocket sends as
    its first "init" message (see DashboardState.get_init_payload) so the frontend's fallback
    poll loop (dashboard.html, engaged automatically once the WebSocket drops) can keep
    everything live over plain HTTP regardless of what's blocking the WS upgrade on a given
    client — same auth gate as every other /api/ route, nothing new to secure."""
    return state.get_init_payload()


async def _reconstruct_missing_tp_fills(buys_by_ticker: dict) -> list[dict]:
    """Best-effort ESTIMATED entries for TP fills that genuinely happened but were never
    recorded anywhere (2026-07-21) -- check_take_profits() only wrote a trade_history row
    when a fill fully closed the position, until today's fix. Confirmed live: ADC (2
    targets remaining, meaning T1 already fired for real money) had zero trade_history
    rows for that fill. Scoped to currently-HELD positions only -- a closed position's
    hidden partial fills can't be reliably reconstructed (the final close's price is
    already a blended weighted average across however many tranches fired, with no way to
    separate them back out), so this deliberately does not attempt that harder case.

    For each held position missing fewer real SELL rows (since it was opened) than its
    target-count implies should have fired, synthesizes the difference: shares = the
    original buy's share count / 3 (the standard equal-thirds split), price = entry_price
    with the CURRENT config's t1_pct/t2_pct applied (the best available proxy -- the
    config active at the real fill time may have differed, which is exactly why this is
    flagged as an estimate, not real data)."""
    if not state.portfolio._db:
        return []
    tp_cfg = state.config.get("take_profit", {})
    t1_pct = tp_cfg.get("t1_pct", 5.0)
    t2_pct = tp_cfg.get("t2_pct", 7.0)
    estimated: list[dict] = []
    for ticker, pos in state.portfolio.positions.items():
        fired_count = 3 - len(pos.take_profit_targets or [])
        if fired_count <= 0:
            continue
        async with state.portfolio._db.execute(
            "SELECT COUNT(*) FROM trade_history WHERE ticker = ? AND action = 'SELL' "
            "AND timestamp >= ?",
            (ticker, pos.opened_at.isoformat()),
        ) as cur:
            (real_count,) = await cur.fetchone()
        missing = fired_count - real_count
        if missing <= 0:
            continue
        buy = buys_by_ticker.get(ticker)
        original_shares = buy["shares"] if buy else pos.shares * 3  # rough fallback
        tranche_shares = original_shares / 3
        pct_by_tranche = {1: t1_pct, 2: t2_pct}
        for tranche in range(1, missing + 1):
            pct = pct_by_tranche.get(tranche, t1_pct)
            price = round(pos.entry_price * (1 + pct / 100), 2)
            estimated.append({
                "ticker": ticker, "action": "SELL",
                "shares": round(tranche_shares, 4), "price": price,
                "pnl": round((price - pos.entry_price) * tranche_shares, 2),
                "reason": f"Take-Profit T{tranche} (estimated)",
                "timestamp": pos.opened_at.isoformat(),
                "is_estimated": True,
                "trade_id": pos.trade_id,
            })
    return estimated


@app.get("/api/trade-history")
async def get_trade_history():
    """Unified trade list (2026-07-21) -- BUYS come exclusively from the JSONL log
    (trade_logger), which has always carried conviction/reasoning/stop_loss for buys and
    never had a SQL counterpart. SELLS come exclusively from the SQL trade_history table,
    which has real P&L (the JSONL never did) and, as of today's fix, a complete record of
    every real fill including partial TP tranches -- using both would risk double-counting
    the same real sell, since most sell paths already wrote to both logs historically.
    ESTIMATED rows (see _reconstruct_missing_tp_fills) fill the one gap that's still
    reconstructable: TP fills on currently-held positions that predate today's fix."""
    jsonl_trades = state.trade_logger.get_trade_history(days=None)
    # state.live_account_start filter (2026-07-21 fix) -- the JSONL log still contains
    # ~69 leftover pre-migration dev/test buys (2026-06-25 through 2026-07-10, from the
    # local machine before the 2026-07-12 Hetzner go-live) that were already identified
    # and filtered out of /api/portfolio-summary on 2026-07-20 ("$209k Buy-Value" fix) --
    # missed applying that same filter here when this endpoint was rewritten today, so
    # the unified list silently included them again (confirmed live: buy_count of 86 vs
    # the real 17 the already-fixed portfolio-summary endpoint reports; user caught the
    # mismatch directly).
    buys = [t for t in jsonl_trades
            if t.get("signal") in ("BUY", "STRONG BUY")
            and t.get("timestamp", "") >= state.live_account_start]
    for b in buys:
        b["action"] = "BUY"
        b["is_estimated"] = False
        # real_fill_price (2026-07-22) -- every log_trade() call site now updates
        # signal.entry_price to the order's real fill price before logging when the
        # broker's response already includes it (fixed 2026-08-08, GitHub #53), so this
        # workaround is only load-bearing for trades logged BEFORE that fix -- the JSONL
        # log is append-only and those old rows still carry the AI's pre-trade estimate
        # forever (confirmed live at the time: differences of $0.40-$1.33 across several
        # held tickers, e.g. MET's suggested entry was $91.50 but the real fill was
        # $92.83). The real fill price only survives in Position.entry_price, and only
        # while the position is still held -- once sold, it's gone for a pre-fix trade.
        # Left in place rather than removed since it's still correct/needed for that
        # historical window.
        _held = state.portfolio.positions.get(b["ticker"])
        if _held is not None:
            b["real_fill_price"] = round(_held.entry_price, 2)
    buys_by_ticker: dict[str, dict] = {}
    for b in buys:
        # Keep the most recent buy per ticker as the "original shares" reference for
        # reconstruction below -- if a ticker was bought multiple times historically,
        # the latest buy is the one most likely to correspond to the position still held.
        prev = buys_by_ticker.get(b["ticker"])
        if prev is None or b["timestamp"] > prev["timestamp"]:
            buys_by_ticker[b["ticker"]] = b

    sells: list[dict] = []
    if state.portfolio._db:
        async with state.portfolio._db.execute(
            "SELECT ticker, action, shares, price, pnl, timestamp, reason, trade_id FROM trade_history "
            "WHERE action = 'SELL' AND timestamp >= ? ORDER BY timestamp",
            (state.live_account_start,),
        ) as cur:
            rows = await cur.fetchall()
        for ticker, action, shares, price, pnl, timestamp, reason, trade_id in rows:
            sells.append({
                "ticker": ticker, "action": action, "shares": shares, "price": price,
                "pnl": pnl, "timestamp": timestamp,
                "reason": reason or "Not recorded",
                "is_estimated": False,
                "trade_id": trade_id,
            })

    estimated = await _reconstruct_missing_tp_fills(buys_by_ticker)
    return buys + sells + estimated


@app.get("/api/performance-history")
async def get_performance_history(range: str = "all"):
    """Portfolio vs. the composition-weighted benchmark (see
    docs/superpowers/specs/2026-07-29-composition-weighted-benchmark-design.md),
    normalized to % change since the start of the requested range. Past days
    come from the settled benchmark_composition_history table (backfilled +
    updated once daily); today is stitched on live via weighted_intraday_series,
    chained onto yesterday's settled value so the join is continuous."""
    from src.analytics.benchmark_store import BenchmarkStore
    from src.analytics.composition_benchmark import weighted_intraday_series

    history = state.performance_history
    if range == "ytd":
        year_start = f"{state._now_et().year}-01-01"
        history = [p for p in history if p["date"] >= year_start]
    elif range == "week":
        # 2026-07-27: matches Total/YTD's shape exactly -- last 7 calendar days, not
        # trading days, same simple date-string comparison as the ytd branch above.
        week_start = (state._now_et() - timedelta(days=7)).strftime("%Y-%m-%d")
        history = [p for p in history if p["date"] >= week_start]
    if not history:
        return {"points": []}

    db_path = state.config.get("database", {}).get("path", "data/aitrading.db")
    store = BenchmarkStore(db_path)
    store.initialize()
    settled = store.get_settled_days()

    base = history[0]
    base_benchmark = settled.get(base["date"])
    points = []
    for p in history:
        benchmark_value = settled.get(p["date"])
        benchmark_pct = (
            (benchmark_value - base_benchmark) / base_benchmark * 100
            if benchmark_value is not None and base_benchmark else None
        )
        portfolio_pct = (
            (p["portfolio_value"] - base["portfolio_value"]) / base["portfolio_value"] * 100
            if base["portfolio_value"] else None
        )
        points.append({
            "date": p["date"],
            "portfolio_pct": round(portfolio_pct, 2) if portfolio_pct is not None else None,
            "benchmark_pct": round(benchmark_pct, 2) if benchmark_pct is not None else None,
        })

    today_str = state._now_et().strftime("%Y-%m-%d")
    if today_str not in settled and base_benchmark:
        try:
            holdings_value = _live_holdings_value()
            classifications = await _live_benchmark_classifications(holdings_value)
            etf_bars = await _fetch_today_etf_bars(classifications)
            intraday = weighted_intraday_series(holdings_value, classifications, etf_bars)
            prior_dates = sorted(d for d in settled if d < today_str)
            prior_value = settled[prior_dates[-1]] if prior_dates else base_benchmark
            portfolio_today_pct = (
                (state.portfolio.total_value - base["portfolio_value"]) / base["portfolio_value"] * 100
                if base["portfolio_value"] else None
            )
            # Exactly one point for today (fixed 2026-07-29, live bug: charts
            # rendered blank) -- this chart is keyed by plain "date" strings,
            # and Lightweight Charts requires every point in a series to have
            # a strictly unique, ascending time value. Appending one point
            # per intraday bar put several points on the identical date
            # string "today_str", which silently broke rendering for the
            # whole series. Using only the latest intraday value keeps this
            # genuinely live (recomputed on every request) without violating
            # that constraint -- unlike the Day popup's own chart, which
            # already uses real per-bar Alpaca timestamps and never had this
            # problem.
            if intraday:
                _, pct_since_open = intraday[-1]
                today_value = prior_value * (1 + pct_since_open / 100)
                benchmark_pct = (today_value - base_benchmark) / base_benchmark * 100
                points.append({
                    "date": today_str,
                    "portfolio_pct": round(portfolio_today_pct, 2) if portfolio_today_pct is not None else None,
                    "benchmark_pct": round(benchmark_pct, 2),
                })
        except Exception as e:
            logger.warning("Failed to stitch live intraday benchmark for today: %s", e, exc_info=True)

    return {"points": points}


def _live_holdings_value() -> dict:
    """Dollar value of every currently-held position, for the composition
    benchmark's live paths. Skips (rather than crashes on) any position
    whose shares/current_price isn't a real, finite number -- defensive
    hardening added 2026-07-29 after a live NoneType*float TypeError was
    caught here in production (non-fatal, already wrapped in the caller's
    own try/except, but this stops one bad position from blocking every
    other position's legitimate contribution)."""
    holdings_value = {}
    for t, pos in state.portfolio.positions.items():
        shares, price = pos.shares, pos.current_price
        if (shares is None or price is None
                or (isinstance(shares, float) and math.isnan(shares))
                or (isinstance(price, float) and math.isnan(price))):
            logger.warning(
                "_live_holdings_value: skipping %s -- shares=%r current_price=%r", t, shares, price)
            continue
        holdings_value[t] = shares * price
    return holdings_value


async def _live_benchmark_classifications(holdings_value: dict) -> dict:
    """Shared by get_performance_history's live "today" stitch and
    get_performance_today -- classifies every currently-held ticker (cached
    after the first lookup ever, see BenchmarkStore/classify_ticker)."""
    from src.analytics.benchmark_store import BenchmarkStore, classify_ticker
    db_path = state.config.get("database", {}).get("path", "data/aitrading.db")
    store = BenchmarkStore(db_path)
    store.initialize()
    sp500 = set(get_universe(["S&P 500"]))
    sp400 = set(get_universe(["S&P 400"]))
    sp600 = set(get_universe(["S&P 600"]))

    def get_sector(ticker):
        pos = state.portfolio.positions.get(ticker)
        return pos.sector if pos and pos.sector else None

    def get_cap_tier_membership():
        return (sp500, sp400, sp600)

    return {t: classify_ticker(t, store, get_sector, get_cap_tier_membership) for t in holdings_value}


async def _fetch_today_etf_bars(classifications: dict) -> dict:
    """Shared by get_performance_history's live "today" stitch and
    get_performance_today -- today's intraday closes for every sector/cap-tier
    ETF actually needed by the currently-classified holdings.

    Fetched concurrently, not one at a time (fixed 2026-08-14, owner report:
    the Day P/L popup "takes a long time to open, always has") -- each ETF's
    get_historical() call is already a real live yfinance fetch wrapped in
    asyncio.to_thread, so awaiting them in a plain for-loop serialized every
    single one; a live-timed real request measured 28.3s for this popup before
    the fix. asyncio.gather runs every ETF's fetch concurrently instead,
    turning that into roughly the time of the single slowest fetch. A failed
    fetch degrades that one ETF to an empty bar list rather than taking down
    the whole batch -- this loop had no per-ETF error handling before this
    fix, unlike get_performance_today's own sibling loop below, which already
    had it."""
    sector_etfs = {c[0] for c in classifications.values() if c[0]}
    cap_tier_etfs = {c[1] for c in classifications.values()}

    async def _fetch_one(etf: str):
        try:
            history = await state.market_data.get_historical(etf, period="1d", interval="1h")
        except Exception:
            history = []
        return etf, [(row["date"] + "T" + str(row["timestamp"]), row["close"]) for row in history]

    results = await asyncio.gather(*(_fetch_one(etf) for etf in sector_etfs | cap_tier_etfs))
    return dict(results)


@app.get("/api/win-loss-trades")
async def get_win_loss_trades():
    """Backs the Win Rate tile's popup (2026-07-29) -- the current-architecture-cutoff
    trade list from the same cache _refresh_win_rate_cache maintains, already
    trade_id-grouped (a position's T1/T2/final tranches count as one combined-outcome
    trade, not three) and most-recent-first. Zero-cost -- reads the cache, no live
    query."""
    return {
        "trades": state._win_rate_cache.get("trades", []),
        "win_rate_current_arch_pct": state._win_rate_cache.get("win_rate_current_arch_pct", 0.0),
        "closed_current_arch": state._win_rate_cache.get("closed_current_arch", 0),
        "win_rate_all_time_pct": state._win_rate_cache.get("win_rate_all_time_pct", 0.0),
        "closed_all_time": state._win_rate_cache.get("closed_all_time", 0),
    }


@app.get("/api/sell-analysis/{trade_id}")
async def get_sell_analysis(trade_id: str):
    """Backs the Recent Sells trade-detail popup's post-mortem section (2026-08-21) --
    the buy-side snapshot plus the AI's immediate and (once due) delayed follow-up
    judgment for one closed trade. Lazy-fetched by the frontend only when a Recent
    Sells row is actually opened, same pattern as every other on-demand popup in this
    app -- not embedded in the base /api/trade-history payload. 404 for a trade_id with
    no row (predates this feature, or never had a real buy_thesis to snapshot -- see
    close_position_async) or one still pending its first AI pass."""
    from fastapi import HTTPException
    record = await state.portfolio.get_sell_analysis(trade_id)
    if not record or not record.get("post_mortem_thesis"):
        raise HTTPException(status_code=404, detail="No sell analysis available for this trade")
    return record


@app.get("/api/weekly-pnl-history")
async def get_weekly_pnl_history():
    """Backs the Week P/L popup's historical list+bar breakdown (2026-07-27
    redesign) -- one entry per ISO calendar week that has real
    performance_history data, newest first. Uses the exact same
    _weekly_pnl_buckets() the live Week P/L tile's _week_pnl() bases its own
    current-week figure on, so the tile and this popup can never disagree
    about week boundaries or baseline values."""
    return {"weeks": state._weekly_pnl_buckets()}


@app.get("/api/daily-pnl-history")
async def get_daily_pnl_history():
    """Backs the Day P/L popup's running per-day history list (2026-08-14) -- one
    entry per settled trading day since genuine live-trading inception, newest
    first, plus today's own still-live figure (Portfolio.day_pnl/day_pnl_pct, the
    same numbers the Day P/L tile itself already shows) stitched on as the first
    entry so today is never missing just because tonight's snapshot hasn't run
    yet -- but only when today is actually a trading day (fixed 2026-08-15); a
    holiday/weekend "today" contributes no row at all rather than a noisy $0.00
    placeholder for a day the market never opened. See _daily_pnl_buckets()'s own
    docstring for why this doesn't need any further backdating."""
    days = state._daily_pnl_buckets()
    today_str = state._now_et().strftime("%Y-%m-%d")
    day_start = state.portfolio.day_start_value
    # Same real-trading-day check this codebase already uses for the pre-open batch/
    # daily report triggers. Originally only gated is_current (fixed 2026-08-15), but
    # the owner pointed out a further problem the same evening: stitching in a $0.00
    # "today" row for a day the market never even opened is just noise, not useful
    # history -- "it shouldnt show any saturday as saturday is not a market day." Now
    # skips the stitch entirely on a non-trading day, so the list simply starts from
    # the most recent real settled trading day instead of a placeholder. The daily
    # snapshot itself only ever fires on real trading days (same gate), so a settled
    # entry for today can never already exist in `days` when today isn't one --
    # nothing to preserve by still checking days[0]["date"] == today_str here.
    today_is_trading_day = not state._is_holiday and state._now_et().weekday() < 5
    if not today_is_trading_day:
        return {"days": days}
    today_entry = {
        "date": today_str,
        "pnl": round(state.portfolio.day_pnl, 2),
        "pnl_pct": round((state.portfolio.day_pnl / day_start * 100) if day_start else 0, 2),
        "is_current": True,
    }
    if days and days[0]["date"] == today_str:
        days[0] = today_entry
    else:
        days.insert(0, today_entry)
    return {"days": days}


@app.get("/api/performance-today")
async def get_performance_today():
    """Intraday version of the same comparison, for the Day P/L popup -- today's portfolio
    equity curve (Alpaca's own hourly portfolio history, filtered to today) vs. the
    composition-weighted benchmark (see
    docs/superpowers/specs/2026-07-29-composition-weighted-benchmark-design.md), both
    normalized to % change since market open. Real intraday granularity, not just two
    endpoints -- shows how today actually unfolded, not just where it landed.

    Alpaca's own equity-history timestamps don't align to yfinance's hourly bar grid, so
    (unlike get_performance_history's today-stitch, which builds its own aligned series
    via weighted_intraday_series) this matches each equity point to the nearest ETF bar
    at or before it, then blends via weighted_daily_return per point -- preserves the
    original "nearest bar at or before this timestamp" matching this endpoint already
    used for the old flat SPY/QQQ/DIA blend."""
    from src.analytics.composition_benchmark import weighted_daily_return

    today_str = state._now_et().strftime("%Y-%m-%d")
    try:
        equity_history = await state.order_manager.broker.get_portfolio_history()
    except Exception:
        equity_history = []
    today_equity = [p for p in equity_history if
                    datetime.fromtimestamp(p["t"]).strftime("%Y-%m-%d") == today_str]
    if not today_equity:
        return {"points": []}

    base_equity = today_equity[0]["equity"]
    holdings_value = _live_holdings_value()
    try:
        classifications = await _live_benchmark_classifications(holdings_value)
        sector_etfs = {c[0] for c in classifications.values() if c[0]}
        cap_tier_etfs = {c[1] for c in classifications.values()}

        # Concurrent, not sequential (fixed 2026-08-14, owner report: "takes a
        # long time to open, always has") -- see _fetch_today_etf_bars's own
        # docstring above for the live-measured 28.3s this loop's sequential
        # version produced. Same per-ETF-failure-degrades-to-[] semantics as
        # before, just no longer serialized.
        async def _fetch_one_15m(etf: str):
            try:
                return etf, await state.market_data.get_historical(etf, period="1d", interval="15m")
            except Exception:
                return etf, []

        etf_series: dict[str, list[dict]] = dict(await asyncio.gather(
            *(_fetch_one_15m(etf) for etf in sector_etfs | cap_tier_etfs)))
    except Exception as e:
        logger.warning("Failed to build composition-weighted intraday series: %s", e)
        classifications, etf_series = {}, {}

    # Every ETF weighted_daily_return will actually look up for these
    # holdings -- computed once, since classifications don't change per point.
    required_etfs = {c[0] for c in classifications.values() if c[0]} | {
        c[1] for c in classifications.values()}

    points = []
    for p in today_equity:
        t = p["t"]
        portfolio_pct = (p["equity"] - base_equity) / base_equity * 100 if base_equity else None
        etf_returns_since_open = {}
        for etf, hist in etf_series.items():
            if not hist:
                continue
            base_price = hist[0]["close"]
            # Nearest bar at or before this timestamp
            candidates = [h for h in hist if h.get("timestamp", 0) <= t]
            if candidates and base_price:
                nearest = candidates[-1]
                etf_returns_since_open[etf] = (nearest["close"] - base_price) / base_price
        # Fixed 2026-07-29, live 500 error (KeyError: 'MDY'): weighted_daily_return
        # looks up EVERY etf in required_etfs, not just whichever ones happen to
        # have a bar by this specific timestamp (an ETF's first 15m bar can post
        # later than another's, especially right at/before market open) -- the
        # old check only verified etf_returns_since_open was non-empty, which a
        # partial dict still satisfies right up until the missing key crashes
        # the lookup deep inside weighted_daily_return.
        if holdings_value and classifications and required_etfs <= etf_returns_since_open.keys():
            try:
                blended_pct = weighted_daily_return(
                    holdings_value, classifications, etf_returns_since_open) * 100
            except Exception as e:
                logger.warning("weighted_daily_return failed for a performance-today point: %s", e)
                blended_pct = None
        else:
            blended_pct = None
        points.append({
            "t": t,
            "portfolio_pct": round(portfolio_pct, 2) if portfolio_pct is not None else None,
            "benchmark_pct": round(blended_pct, 2) if blended_pct is not None else None,
        })
    return {"points": points}


@app.get("/api/portfolio-summary")
async def get_portfolio_summary():
    # 2026-07-20 fix: "all-time" trade_history JSONL files include ~3 weeks of leftover
    # local-machine dev/test data (2026-06-25 through 2026-07-10, owned by the pre-migration
    # dev user) that got copied over during the initial Hetzner deploy and never cleaned
    # out -- confirmed via file ownership on the server (uid 1000 = local dev, root = live
    # Hetzner account). Those files contain unrealistic full-capital-sized test trades
    # (e.g. a single $10,000 BUY) that inflated buy_value to ~$209k against a real ~$10k
    # account. Filtered by each trade's own timestamp rather than by filename, since ISO
    # timestamps string-sort correctly and this doesn't depend on any file ever being
    # cleaned up on disk. state.live_account_start (shared with /api/portfolio-health's
    # win-rate query below -- same pollution, same fix) is this install's own real
    # go-live date, self-initializing since 2026-08-20 -- see
    # _get_or_init_account_genesis's docstring.
    trades = [
        t for t in state.trade_logger.get_trade_history(days=None)
        if t.get("timestamp", "") >= state.live_account_start
    ]
    buy_count = 0
    buy_value = 0.0
    sell_count = 0
    sell_value = 0.0
    for t in trades:
        value = float(t.get("position_size", 0) or 0)
        if t.get("signal") == "BUY":
            buy_count += 1
            buy_value += value
        elif t.get("signal") == "SELL":
            sell_count += 1
            sell_value += value

    history = []
    broker = state.order_manager.broker
    if broker and hasattr(broker, "get_portfolio_history"):
        try:
            history = await broker.get_portfolio_history()
        except Exception as e:
            logger.warning("Portfolio history fetch failed: %s", e)

    return {
        "buy_count": buy_count, "buy_value": round(buy_value, 2),
        "sell_count": sell_count, "sell_value": round(sell_value, 2),
        "history": history,
    }


_MODEL_DISPLAY_NAMES = {
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-sonnet-5": "Sonnet 5",
    "claude-opus-4-8": "Opus 4.8",
    "claude-fable-5": "Fable 5",
}


@app.get("/api/portfolio-health-model")
async def get_portfolio_health_model():
    """Zero-cost, no-Claude-call lookup of which model a Portfolio Health Assessment run
    would currently use -- lets the dashboard show a confirmation prompt with the real
    model name before actually spending a Claude call (2026-07-24, user: accidentally
    opening the Portfolio Summary modal used to fire the assessment immediately with no
    way to back out)."""
    model = state.config["research"].get("model_portfolio_health", "claude-haiku-4-5")
    return {"model": model, "label": _MODEL_DISPLAY_NAMES.get(model, model)}


@app.get("/api/portfolio-health")
async def get_portfolio_health(force: bool = False):
    cache = state.portfolio_health_cache
    now_et = state._now_et()
    if not force and cache.get("generated_at"):
        # Widened 30 min -> effectively once per day (2026-07-20) per direct user feedback:
        # this only ever fires on-demand (no background job calls it), so the real cost
        # driver is how many times the modal gets reopened, not a fixed schedule. Cached
        # for the rest of the same ET trading day once generated, but always treated as
        # stale on the first check of a new day ("refresh in the morning") rather than a
        # flat rolling 24h window, since the underlying inputs (positions, thesis,
        # conviction) genuinely do refresh via the pre-open scan each morning. "Refresh
        # Assessment" (force=true) always bypasses this regardless of either check.
        cached_dt = datetime.fromisoformat(cache["generated_at"])
        if cached_dt.date() == now_et.date():
            return cache

    # Real win/loss determination needs the SQL trade_history table's pnl column —
    # the JSONL log (used by /api/portfolio-summary for buy/sell $ totals) doesn't carry
    # pnl. Same pre-migration pollution as the JSONL fix (2026-07-20): the whole
    # data/aitrading.db file was rsync'd over during the initial Hetzner deploy, and its
    # earliest SELL row is dated 2026-07-07 — before the real 2026-07-12 go-live. Filtered
    # by that cutoff so win_rate reflects only real live-account trades.
    #
    # Split into two figures (2026-07-20, per direct user feedback) rather than one blended
    # number: "all trades since go-live" mixes in trades decided by the pre-2026-07-17
    # Watchlist-based system, which no longer exists — a stat blending two different
    # trading systems' track records isn't a fair read on how the CURRENT logic performs.
    #
    # Grouped by trade_id (2026-07-29 fix) -- see _group_closed_trades's docstring; a
    # position's T1/T2/final tranches must count as one trade's combined outcome, not
    # three separate per-tranche wins/losses.
    async def _win_rate_since(cutoff: str) -> tuple[float, int]:
        trades = await state._closed_trades_since(cutoff)
        closed = len(trades)
        wins = sum(1 for t in trades if t["is_win"])
        return (wins / closed * 100) if closed else 0.0, closed

    win_rate_all_time_pct, closed_all_time = await _win_rate_since(state.live_account_start)
    win_rate_current_arch_pct, closed_current_arch = await _win_rate_since(_CURRENT_ARCHITECTURE_START)

    sector_counts: dict[str, int] = {}
    conviction_values = []
    positions_payload = []
    for ticker, pos in state.portfolio.positions.items():
        sector = pos.sector or "Unknown"
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        report = state.research_reports.get(ticker, {})
        conviction = report.get("conviction", 0) or 0
        if conviction:
            conviction_values.append(conviction)
        days_held = (now_et.date() - pos.opened_at.date()).days if pos.opened_at else 0
        positions_payload.append({
            "ticker": ticker,
            "thesis": report.get("thesis", ""),
            "fair_value_estimate": report.get("fair_value_estimate", 0.0) or 0.0,
            "margin_of_safety_pct": report.get("margin_of_safety_pct", 0.0) or 0.0,
            "conviction": conviction,
            "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            "days_held": days_held,
        })
    avg_conviction = sum(conviction_values) / len(conviction_values) if conviction_values else 0.0

    portfolio_summary = {
        "total_value": state.portfolio.total_value,
        "cash_pct": state.portfolio.cash_pct,
        "day_pnl_pct": (state.portfolio.day_pnl / state.portfolio.day_start_value * 100)
            if state.portfolio.day_start_value else 0.0,
        "total_pnl_pct": state.portfolio.total_pnl_pct,
        "drawdown_state": state.risk_manager.check_drawdown(state.portfolio),
        "win_rate_all_time_pct": win_rate_all_time_pct,
        "closed_all_time": closed_all_time,
        "win_rate_current_arch_pct": win_rate_current_arch_pct,
        "closed_current_arch": closed_current_arch,
        "avg_conviction": avg_conviction,
        "sector_counts": sector_counts,
        "min_conviction_score": state.config["research"]["min_conviction_score"],
    }

    result = await state.research_engine.recommend_portfolio_health(portfolio_summary, positions_payload)
    if result is None:
        return {"is_fallback": True}

    result["is_fallback"] = False
    result["generated_at"] = now_et.isoformat()
    state.portfolio_health_cache = result
    return result


@app.get("/api/orders")
async def get_orders():
    return [
        {
            "broker_order_id": o.broker_order_id,
            "ticker": o.ticker,
            "side": o.side.value,
            "order_type": o.order_type.value,
            "quantity": o.quantity,
            "limit_price": o.limit_price,
            "stop_price": o.stop_price,
            "status": o.status.value,
            "filled_price": o.filled_price,
            "filled_quantity": o.filled_quantity,
            "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
        }
        for o in state.order_manager.active_orders.values()
    ]


@app.post("/api/restart")
async def restart_server():
    logger.info("Server restart requested via UI")
    entry = state.add_ai_log("SYSTEM", "RESTART", "Server restart initiated from dashboard", "info")
    await state.broadcast({"type": "ai_log", "entry": entry})

    async def _do_restart():
        await asyncio.sleep(1.5)
        import subprocess
        subprocess.Popen(
            [sys.executable, str(Path(__file__).parent.parent / "start.py"), "--mode", "web", "--no-browser"],
            start_new_session=True,
        )

    asyncio.create_task(_do_restart())
    return {"status": "restarting"}


def _deep_dive_report_dict(report) -> dict:
    """Shared dict shape for a completed (non-fallback) DeepDiveReport, extracted (2026-07-19)
    from _run_manual_deep_dive so the new On-Deck-entry auto-trigger (_maybe_auto_deep_dive)
    can populate state.deep_dive_reports identically without a second copy of this ~25-field
    literal. Pure function, no side effects — caller handles caching/broadcasting."""
    return {
        "ticker": report.ticker,
        "company_name": report.company_name,
        "generated_at": report.generated_at.isoformat(),
        "signal": report.signal,
        "conviction_score": report.conviction_score,
        "thesis": report.thesis,
        "valuation_analysis": report.valuation_analysis,
        "fair_value_estimate": report.fair_value_estimate,
        "margin_of_safety_pct": report.margin_of_safety_pct,
        "entry_price": report.entry_price,
        "stop_loss": report.stop_loss,
        "risk_factors": report.risk_factors,
        "technical_analysis": report.technical_analysis,
        "competitive_moat": report.competitive_moat,
        "growth_outlook": report.growth_outlook,
        "catalysts": report.catalysts,
        "risk_scenarios": report.risk_scenarios,
        "entry_zone_low": report.entry_zone_low,
        "entry_zone_high": report.entry_zone_high,
        "monitoring_checklist": report.monitoring_checklist,
        "enhanced_reasoning": report.enhanced_reasoning,
        "peer_comparison": getattr(report, "peer_comparison", ""),
        "management_quality": getattr(report, "management_quality", ""),
        "macro_sensitivity": getattr(report, "macro_sensitivity", ""),
        "historical_patterns": getattr(report, "historical_patterns", ""),
        "expected_return_6m": getattr(report, "expected_return_6m", 0.0),
        "expected_return_12m": getattr(report, "expected_return_12m", 0.0),
        "is_deeper_dive": getattr(report, "is_deeper_dive", False),
        "is_fallback": False,
    }


@app.post("/api/deep-dive")
async def manual_deep_dive(payload: dict):
    ticker = payload.get("ticker", "").upper().strip()
    if not ticker:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="ticker required")
    asyncio.create_task(_run_manual_deep_dive(ticker))
    return {"status": "started", "ticker": ticker}


async def _run_manual_deep_dive(ticker: str):
    entry = state.add_ai_log(ticker, "DEEP DIVE", f"Manual deep dive started for {ticker}", "neutral")
    await state.broadcast({"type": "ai_log", "entry": entry})
    if ticker not in state.research_engine.reports:
        try:
            await state.research_engine.analyze_stock(ticker)
        except Exception as e:
            entry = state.add_ai_log(ticker, "DEEP DIVE", f"Pre-scan failed for {ticker}: {e}", "negative")
            await state.broadcast({"type": "ai_log", "entry": entry})
            return
    try:
        report = await state.research_engine.deep_dive_analysis(ticker)
        if report:
            if report.is_fallback:
                entry = state.add_ai_log(ticker, "DEEP DIVE",
                    f"⚠ Deep dive failed — AI unavailable. Reason: {report.fallback_reason}", "negative")
                await state.broadcast({"type": "ai_log", "entry": entry})
                await state.broadcast({"type": "deep_dive_failed", "ticker": ticker, "reason": report.fallback_reason})
                return
            state.deep_dive_reports[ticker] = _deep_dive_report_dict(report)
            asyncio.create_task(asyncio.to_thread(_save_dd_cache, state.deep_dive_reports))
            await state.broadcast({"type": "deep_dive_report", "ticker": ticker, "report": state.deep_dive_reports[ticker]})
            entry = state.add_ai_log(ticker, "DEEP DIVE", f"Deep dive complete for {ticker}", "positive")
            await state.broadcast({"type": "ai_log", "entry": entry})
    except Exception as e:
        entry = state.add_ai_log(ticker, "DEEP DIVE", f"Deep dive failed for {ticker}: {e}", "negative")
        await state.broadcast({"type": "ai_log", "entry": entry})


async def _run_manual_deeper_dive(ticker: str):
    entry = state.add_ai_log(ticker, "DEEPER DIVE", f"Deeper dive started for {ticker}...", "neutral")
    await state.broadcast({"type": "ai_log", "entry": entry})
    try:
        report = await state.research_engine.deeper_dive_analysis(ticker)
        if report:
            if report.is_fallback:
                entry = state.add_ai_log(ticker, "DEEPER DIVE",
                    f"⚠ Deeper dive failed — AI unavailable. Reason: {report.fallback_reason}", "negative")
                await state.broadcast({"type": "ai_log", "entry": entry})
                await state.broadcast({"type": "deep_dive_failed", "ticker": ticker, "reason": report.fallback_reason})
                return
            state.deep_dive_reports[ticker] = {
                "ticker": ticker,
                "company_name": report.company_name,
                "generated_at": report.generated_at.isoformat(),
                "signal": report.signal,
                "conviction_score": report.conviction_score,
                "thesis": report.thesis,
                "valuation_analysis": report.valuation_analysis,
                "fair_value_estimate": report.fair_value_estimate,
                "margin_of_safety_pct": report.margin_of_safety_pct,
                "entry_price": report.entry_price,
                "stop_loss": report.stop_loss,
                "risk_factors": report.risk_factors,
                "technical_analysis": report.technical_analysis,
                "competitive_moat": report.competitive_moat,
                "growth_outlook": report.growth_outlook,
                "catalysts": report.catalysts,
                "risk_scenarios": report.risk_scenarios,
                "entry_zone_low": report.entry_zone_low,
                "entry_zone_high": report.entry_zone_high,
                "monitoring_checklist": report.monitoring_checklist,
                "enhanced_reasoning": report.enhanced_reasoning,
                "peer_comparison": report.peer_comparison,
                "management_quality": report.management_quality,
                "macro_sensitivity": report.macro_sensitivity,
                "historical_patterns": report.historical_patterns,
                "expected_return_6m": report.expected_return_6m,
                "expected_return_12m": report.expected_return_12m,
                "is_deeper_dive": True,
                "is_fallback": False,
            }
            asyncio.create_task(asyncio.to_thread(_save_dd_cache, state.deep_dive_reports))
            await state.broadcast({"type": "deep_dive_report", "ticker": ticker, "report": state.deep_dive_reports[ticker]})
            entry = state.add_ai_log(ticker, "DEEPER DIVE",
                f"Deeper dive complete — E[r] 6m: {report.expected_return_6m:+.1f}% | 12m: {report.expected_return_12m:+.1f}%",
                "positive")
            await state.broadcast({"type": "ai_log", "entry": entry})
    except Exception as e:
        entry = state.add_ai_log(ticker, "DEEPER DIVE", f"Deeper dive failed for {ticker}: {e}", "negative")
        await state.broadcast({"type": "ai_log", "entry": entry})


@app.post("/api/deeper-dive")
async def deeper_dive(payload: dict):
    ticker = payload.get("ticker", "").upper().strip()
    if not ticker:
        return {"status": "error", "detail": "ticker required"}
    asyncio.create_task(_run_manual_deeper_dive(ticker))
    return {"status": "started", "ticker": ticker}


@app.post("/api/manual-buy")
async def manual_buy(payload: dict):
    from fastapi import HTTPException
    ticker = payload.get("ticker", "").upper().strip()
    # Guarded (fixed 2026-08-09, GitHub #69) -- every other validation failure in this
    # endpoint (missing ticker, non-positive dollars, broker not connected, bad quote,
    # invalid stop) is deliberately caught and turned into a clean HTTPException(400,
    # ...); a non-numeric dollars value used to raise an unguarded ValueError instead,
    # which FastAPI's default handler turns into a generic 500 with no indication of
    # what was actually wrong, unlike every neighboring check in this same function.
    try:
        dollars = float(payload.get("dollars", 0))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="dollars must be a number")
    # NaN guard (found in a same-day recheck) -- float() happily parses the literal
    # string "nan" with no exception, and NaN's broken comparison semantics
    # (nan <= 0 is always False) meant it would slip straight past the very check
    # right below this. Downstream, `min(dollars, max_size)` returns NaN itself
    # (Python's min/max just return whichever argument comes first when either side
    # is NaN), poisoning position_dollars/shares with NaN all the way into a real
    # trade confirmation -- the same class of bug as GitHub #65's conviction_score
    # NaN guard, just reachable through a different field.
    if math.isnan(dollars):
        raise HTTPException(status_code=400, detail="dollars must be a number")
    if not ticker or dollars <= 0:
        raise HTTPException(status_code=400, detail="ticker and dollars required")
    if not state.broker_connected:
        raise HTTPException(status_code=400, detail="Broker not connected")

    try:
        quote = await state.market_data.get_quote(ticker)
        price = quote.price
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch price for {ticker}: {e}")

    if not price or price <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid price for {ticker}")

    tp_cfg = state.config.get("take_profit", {})
    stop_loss_pct = tp_cfg.get("stop_loss_pct", 5.0)
    t1_pct = tp_cfg.get("t1_pct", 5.0)
    t2_pct = tp_cfg.get("t2_pct", 10.0)
    t3_pct = tp_cfg.get("t3_pct", 17.0)

    if stop_loss_pct <= 0:
        raise HTTPException(status_code=400, detail="stop_loss_pct must be > 0 — check Settings")

    stop = round(price * (1 - stop_loss_pct / 100), 2)
    if stop <= 0 or stop >= price:
        raise HTTPException(status_code=400, detail=f"Computed stop loss ${stop:.2f} is invalid for price ${price:.2f}")
    targets = [
        round(price * (1 + t1_pct / 100), 2),
        round(price * (1 + t2_pct / 100), 2),
        round(price * (1 + t3_pct / 100), 2),
    ]

    from src.research.engine import ResearchReport, Signal, RiskLevel
    report = ResearchReport(
        ticker=ticker,
        company_name=ticker,
        generated_at=datetime.now(),
        conviction_score=10,
        signal=Signal.BUY,
        risk_level=RiskLevel.MODERATE,
        thesis="Manual buy order",
        fundamental_summary="",
        insider_summary="",
        news_summary="",
        competitive_summary="",
        risk_factors="",
        recommended_action="BUY",
        entry_price=price,
        position_size_pct=0,
        stop_loss=stop,
        take_profit_targets=targets,
    )

    if not state.risk_manager.check_all_rules(report, state.portfolio):
        raise HTTPException(status_code=400, detail="Trade rejected by risk management (cash reserve, sector limits, or drawdown rules)")

    max_size = state.risk_manager.calculate_position_size(price, stop, state.portfolio.total_value)
    position_dollars = min(dollars, max_size)
    shares = round(position_dollars / price, 6)

    from src.decision.signal_generator import TradeSignal
    from src.research.engine import Signal as Sig
    signal = TradeSignal(
        ticker=ticker,
        signal=Sig.BUY,
        conviction=10,
        entry_price=price,
        stop_loss=stop,
        take_profit_targets=targets,
        position_size_pct=round(position_dollars / state.portfolio.total_value * 100, 2),
        position_size_dollars=position_dollars,
        shares=shares,
        reasoning="Manual buy",
        research_report=report,
        generated_at=datetime.now(),
        should_execute=True,
    )

    import uuid as _uuid_mod
    conf_id = f"manual_{ticker}_{_uuid_mod.uuid4().hex[:8]}"
    state.pending_confirmations[conf_id] = {"signal": signal, "created_at": datetime.now()}

    # Earnings warning (fixed 2026-08-08, GitHub #54) -- CLAUDE.md has documented this as
    # a real behavior ("Manual buys show an amber warning banner but still allow the user
    # to proceed") since _earnings_soon() was first built, but this endpoint -- the real,
    # live manual-buy path (the "+ Trade" button / submitManualBuy(), as opposed to the
    # dead WS execute_buy/requestTrade flow the pre-built #earningsWarning banner element
    # was actually wired to) -- never called it at all. Advisory only, matching the
    # documented behavior: never blocks the buy, just surfaces the date for the frontend
    # to render as a warning.
    earnings_soon, earnings_date = await state._earnings_soon(ticker)

    return {
        "confirmation_id": conf_id,
        "ticker": ticker,
        "price": price,
        "shares": shares,
        "dollars": position_dollars,
        "stop_loss": stop,
        "targets": targets,
        "capped": position_dollars < dollars,
        "earnings_warning": f"Earnings on {earnings_date}" if earnings_soon else None,
    }


@app.get("/api/keys/masked")
async def get_keys_masked():
    env_path = Path(__file__).parent.parent / ".env"
    keys = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            keys[k] = v
    result = {}
    for k in ["ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "FINNHUB_API_KEY", "NEWSAPI_API_KEY"]:
        result[k] = keys.get(k, "")
    return result


@app.post("/api/keys")
async def save_keys(payload: dict):
    env_path = Path(__file__).parent.parent / ".env"
    existing = {}
    lines = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                k, _, v = stripped.partition('=')
                existing[k.strip()] = len(lines)
            lines.append(line)

    allowed = {"ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "FINNHUB_API_KEY", "NEWSAPI_API_KEY"}
    changed: set[str] = set()
    for k, v in payload.items():
        if k not in allowed:
            continue
        v = v.strip()
        if not v:
            continue
        new_line = f"{k}={v}"
        if k in existing:
            lines[existing[k]] = new_line
        else:
            lines.append(new_line)
        os.environ[k] = v
        changed.add(k)

    env_path.write_text('\n'.join(lines) + '\n', encoding="utf-8")

    # Hot-swap live clients so the server doesn't need a restart
    if "ANTHROPIC_API_KEY" in changed:
        import anthropic as _anthropic
        state.research_engine.client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        state.has_claude = True
        state.stock_delay = 15

    if changed & {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"}:
        import alpaca_trade_api as tradeapi
        broker = state.order_manager.broker
        if broker is not None:
            api_key = os.getenv("ALPACA_API_KEY", "")
            secret  = os.getenv("ALPACA_SECRET_KEY", "")
            base_url = os.getenv("ALPACA_BASE_URL", "") or (
                "https://paper-api.alpaca.markets" if broker.paper else "https://api.alpaca.markets"
            )
            broker.api = tradeapi.REST(api_key, secret, base_url, api_version="v2")

    if "FINNHUB_API_KEY" in changed:
        state.market_data.finnhub_key = os.environ["FINNHUB_API_KEY"]
        state.news_feed.finnhub_key   = os.environ["FINNHUB_API_KEY"]

    if "NEWSAPI_API_KEY" in changed:
        state.news_feed.newsapi_key = os.environ["NEWSAPI_API_KEY"]

    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Auth gate (2026-07-22, GitHub #13) -- BaseHTTPMiddleware (what @app.middleware("http")
    # compiles to) only ever processes "http"-scope ASGI connections, so _auth_gate above
    # never runs for this route at all; the WebSocket needs its own explicit check.
    # websocket.session is already populated at this point -- SessionMiddleware operates on
    # both "http" and "websocket" scope types and runs before the route handler regardless of
    # accept() having been called yet, so this check (and the close-without-accepting on
    # failure) works correctly.
    if not websocket.session.get("authenticated"):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    state.connected_clients.append(websocket)

    # research_reports intentionally omitted — fetched lazily via GET /api/research-reports
    # (see that endpoint's docstring for why).
    await websocket.send_json(state.get_init_payload())

    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("command")

            if cmd == "pause":
                state.paused = True
                await state.broadcast({"type": "paused", "paused": True})
            elif cmd == "resume":
                state.paused = False
                await state.broadcast({"type": "paused", "paused": False})
            elif cmd == "get_portfolio":
                await websocket.send_json({"type": "portfolio", "portfolio": state.get_portfolio_snapshot()})
            elif cmd == "set_max_positions":
                val = int(data.get("value", 10))
                val = max(1, min(50, val))
                state.config.setdefault("portfolio", {})["max_positions"] = val
                await state.broadcast({"type": "max_positions", "value": val})
                entry = state.add_ai_log("SYSTEM", "CONFIG",
                    f"Max stock positions changed to {val}", "info")
                await state.broadcast({"type": "ai_log", "entry": entry})
                logger.info("Max positions updated to %d", val)
            elif cmd == "remove_on_deck":
                await state.remove_on_deck_candidate(
                    data.get("ticker", ""), data.get("permanent", False), data.get("days", 1),
                    data.get("note", ""))
            elif cmd in ("execute_buy", "confirm_buy", "execute_sell", "cancel_order"):
                await state.handle_trade_command(data, websocket)

    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


def _ensure_wal_mode(db_path: str) -> None:
    """Switches the shared SQLite database to WAL (write-ahead log) journal mode, once,
    at every startup (2026-08-19, live incident) -- see _SQLITE_TIMEOUT_SECS's own
    comment above for the real crash this pairs with: _run_pre_open_batch died with
    sqlite3.OperationalError: database is locked ~19 minutes into a live run, silently
    killing that morning's On Deck refresh. Rollback-journal mode (SQLite's default)
    lets one writer's transaction block every other reader AND writer for its duration;
    WAL mode lets readers and a single writer proceed concurrently without blocking
    each other, which is the dominant source of contention in an app this size --
    position_update_loop, ai_log persistence, trade history writes, and the watchlist
    cursor write that actually crashed all touch this same file from independent async
    loops with no coordination between them. WAL mode is a property of the database
    FILE itself (stored in its header), not the connection -- setting it once here
    covers every connection this app opens afterward (sync sqlite3 in
    watchlist_manager.py/benchmark_store.py/this file, and aiosqlite in portfolio.py),
    regardless of which module's connection happens to open the file first. Idempotent
    and cheap to call unconditionally on every startup -- querying an already-WAL
    database for its journal_mode is a no-op, not a real migration each time."""
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(db_path, timeout=_SQLITE_TIMEOUT_SECS)
    try:
        row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
        logger.info("Database journal mode: %s (%s)", row[0] if row else "?", db_path)
    finally:
        conn.close()


@app.on_event("startup")
async def startup():
    Path("data").mkdir(exist_ok=True)
    _ensure_wal_mode(state.config.get("database", {}).get("path", "data/aitrading.db"))
    state.deep_dive_reports = _load_dd_cache()
    if state.deep_dive_reports:
        logger.info("Restored %d deep dive report(s) from cache", len(state.deep_dive_reports))
    state.price_direction = _load_price_direction_cache()
    if state.price_direction:
        logger.info("Restored %d price direction(s) from cache", len(state.price_direction))
    state.research_reports = _load_report_cache()
    if state.research_reports:
        logger.info("Restored %d research report(s) from cache", len(state.research_reports))
        # Rebuild in-memory ResearchReport objects so deep_dive_analysis() works after restart
        from src.research.engine import ResearchReport, Signal, RiskLevel
        _sig_map = {s.value: s for s in Signal}
        _risk_map = {r.value: r for r in RiskLevel}
        for _ticker, _d in state.research_reports.items():
            try:
                state.research_engine.reports[_ticker] = ResearchReport(
                    ticker=_ticker,
                    company_name=_d.get("company_name", _ticker),
                    generated_at=datetime.fromisoformat(_d["generated_at"]) if "generated_at" in _d else datetime.now(),
                    conviction_score=_d.get("conviction", 0),
                    signal=_sig_map.get(_d.get("signal", "HOLD"), Signal.HOLD),
                    risk_level=_risk_map.get(_d.get("risk_level", "MODERATE"), RiskLevel.MODERATE),
                    thesis=_d.get("thesis", ""),
                    fundamental_summary=_d.get("fundamental_summary", ""),
                    insider_summary=_d.get("insider_summary", ""),
                    news_summary=_d.get("news_summary", ""),
                    competitive_summary=_d.get("competitive_summary", ""),
                    risk_factors=_d.get("risk_factors", ""),
                    recommended_action=_d.get("signal", "HOLD"),
                    entry_price=_d.get("entry_price", 0.0),
                    position_size_pct=_d.get("position_size_pct", 0.0),
                    stop_loss=_d.get("stop_loss", 0.0),
                    take_profit_targets=_d.get("take_profit_targets", []),
                    time_horizon=_d.get("time_horizon", ""),
                    reasoning=_d.get("reasoning", ""),
                    sector=_d.get("sector", ""),
                    fair_value_estimate=_d.get("fair_value_estimate", 0.0),
                    margin_of_safety_pct=_d.get("margin_of_safety_pct", 0.0),
                    is_fallback=False,
                )
            except Exception as _e:
                logger.warning("Could not restore ResearchReport for %s: %s", _ticker, _e)
        logger.info("Rebuilt %d ResearchReport objects in engine from cache", len(state.research_engine.reports))
    await state.portfolio.initialize()
    await state.connect_broker()
    await state.order_manager.start_trade_updates_stream()

    # Remove any watchlist stocks already held as positions — they don't need a slot
    held = set(state.portfolio.positions.keys())
    for ticker in held:
        if ticker in state.watchlist_manager.get_active_tickers():
            state.watchlist_manager.remove(ticker)
            logger.info("Startup cleanup: removed held position %s from watchlist", ticker)

    # Manual On Deck removals (2026-07-18) — loaded before near_miss_candidates so a ticker
    # that's both in the restored cache AND currently blocked (e.g. blocked, then the cache
    # was saved again by something else before the block was checked) gets caught below.
    state.on_deck_blocked = _load_on_deck_blocked()
    if state.on_deck_blocked:
        logger.info("Restored %d manually-blocked On Deck ticker(s)", len(state.on_deck_blocked))

    state.on_deck_notes = _load_on_deck_notes()
    if state.on_deck_notes:
        logger.info("Restored %d On Deck note(s)", len(state.on_deck_notes))

    state.on_deck_stale_dip_low = _load_on_deck_stale_dip_low()
    if state.on_deck_stale_dip_low:
        logger.info("Restored %d known-stale dip low(s)", len(state.on_deck_stale_dip_low))

    state.buy_reasoning = _load_buy_reasoning()
    if state.buy_reasoning:
        logger.info("Restored buy reasoning for %d ticker(s)", len(state.buy_reasoning))

    # Event-monitor cooldown (2026-07-30, real-cost fix) -- restoring this across a
    # restart is what actually matters: without it, ANY restart (even one deploying an
    # urgent, unrelated safety fix) wiped every ticker's cooldown clock, causing an
    # immediate real Claude re-fire for a condition that had just been checked minutes
    # earlier. Live-caught the same day: 3 same-day restarts each re-fired ONB's/GEN's
    # event-triggered re-analysis within seconds of startup.
    state._event_monitor_cooldown, state._loss_event_worst_pct = _load_event_monitor_cooldown()
    if state._event_monitor_cooldown:
        logger.info(
            "Restored event-monitor cooldown for %d ticker(s)", len(state._event_monitor_cooldown))

    # Mid-day re-scan fired tracker (2026-07-31 incident) -- restoring this across a
    # restart is what actually matters: without it, a restart occurring after a
    # configured slot's time had already passed for the day re-fires that slot again,
    # possibly concurrently with another still-running mid-day scan. Live-caught
    # deploying this very feature (see docs/CLAUDE_HISTORY.md).
    state._midday_scan_fired = _load_midday_scan_fired()
    if state._midday_scan_fired:
        logger.info("Restored mid-day scan fired state: %s", state._midday_scan_fired)

    state.promotion_attempts = _load_promotion_attempts()
    if state.promotion_attempts:
        logger.info("Restored %d promotion attempt(s) from cache", len(state.promotion_attempts))

    state.active_signals = _load_active_signals()
    if state.active_signals:
        logger.info("Restored %d active signal(s) from cache", len(state.active_signals))

    state.performance_history = _load_performance_history()
    if state.performance_history:
        logger.info("Restored %d day(s) of performance history", len(state.performance_history))

    # On Deck now persists across restarts and calendar days (2026-07-17) — load the
    # cross-day cache first (primary source), then run the existing same-day backfill as a
    # supplementary self-heal for tickers analyzed since the cache was last saved (e.g. a
    # restart mid-pre-open-run); that function already skips any ticker already present, so
    # loading the on_deck_cache first is what makes it "supplement, don't duplicate."
    state.near_miss_candidates = _load_on_deck_cache()
    for _t in list(state.near_miss_candidates.keys()):
        if state._is_on_deck_blocked(_t):
            state.near_miss_candidates.pop(_t, None)
    for _t in list(state.near_miss_candidates.keys()):
        if _t in state.portfolio.positions:
            state.near_miss_candidates.pop(_t, None)
    for _t in list(state.near_miss_candidates.keys()):
        # 2026-07-28 -- a ticker that went wash-sale-blocked before a restart (or was
        # already blocked when this exact cache was last saved) shouldn't come back from
        # the cache load either; every other population/eviction path is guarded the same
        # way, this closes the one restoration itself could otherwise reintroduce.
        if state._wash_sale_blocked(_t):
            state.near_miss_candidates.pop(_t, None)
    if state.near_miss_candidates:
        logger.info("Restored %d On Deck candidate(s) from cross-day cache",
                    len(state.near_miss_candidates))

    # Restored candidates from before the 2026-07-18 price-history-on-add change never got a
    # real backfill — they'd otherwise sit on the synthetic debug chart indefinitely (it only
    # gets replaced once 2+ real live ticks land). One-time catch-up: any restored candidate
    # still showing an empty real price_history gets the same real ~30-day backfill a brand
    # new candidate gets. Sequential, not concurrent — this list is small (On Deck candidates
    # only, not the full universe) and startup isn't latency-sensitive enough to need
    # asyncio.gather here.
    _catchup_ran = False
    for _t, _nm in state.near_miss_candidates.items():
        if not _nm.get("price_history"):
            _nm["price_history"] = await state._fetch_price_history(_t, _nm.get("last_price"))
            _catchup_ran = True
    if _catchup_ran:
        # FIXED 2026-07-18: this loop only ever updated the in-memory dict — nothing here
        # persisted the result, so every restart silently redid the same backfill in memory
        # and lost it again on the NEXT restart (the on-disk cache never actually gained the
        # real data). Caught live: WMB/JPM/META/PM had all been through this catch-up
        # multiple times today, yet their persisted price_history was still empty. Saving
        # here makes the backfill actually stick.
        await asyncio.to_thread(_save_on_deck_cache, dict(state.near_miss_candidates))
        logger.info("Persisted retroactive price_history backfill to on_deck_cache.json")

    _nm_added = await state._backfill_near_miss_from_cache()
    if _nm_added:
        logger.info("Backfilled %d additional On Deck candidate(s) from today's cached reports", _nm_added)
    # Defensive re-check (2026-07-19) -- e.g. if on_deck_max_size was lowered via Settings
    # since the cache was last saved, a straight restore could otherwise restart over cap.
    _nm_trimmed = await state._enforce_on_deck_cap("SYSTEM")
    if _nm_trimmed:
        logger.info("Trimmed %d On Deck candidate(s) over the size cap after restart", _nm_trimmed)
        # Without this, the trim above only ever updates the in-memory dict -- the stale,
        # still-over-cap set sitting in on_deck_cache.json survives untouched, so the very
        # next restart reloads it and re-drops (and re-logs) the identical tickers again,
        # forever. Same fix already applied to the Settings-triggered trim path above.
        await asyncio.to_thread(_save_on_deck_cache, dict(state.near_miss_candidates))

    # Win/Loss stat cache (2026-07-29) -- populate once here so the dashboard's Win Rate
    # tile shows real data immediately on startup instead of sitting empty for up to 60s
    # until near_miss_monitor_loop's own periodic refresh first fires.
    await state._refresh_win_rate_cache()

    asyncio.create_task(state.auto_scan_loop())
    asyncio.create_task(state.near_miss_monitor_loop())
    asyncio.create_task(state.watchlist_rr_loop())
    asyncio.create_task(state.position_update_loop())
    asyncio.create_task(state.position_monitor_loop())
    asyncio.create_task(state.position_deep_dive_loop())
    logger.info("Dashboard started — auto-scan running, position monitor active")

    # First-run bootstrap removed 2026-07-17 — it triggered on watchlist_manager.size() == 0,
    # which used to mean "fresh install" but now means nothing (the watchlist is never
    # populated at all since On Deck replaced it as the sole buy path), so this would have
    # fired its expensive 15-30 min full-universe scan on every single restart. On Deck
    # candidates are already restored via _backfill_near_miss_from_cache() above, and the
    # next scheduled pre-open scan repopulates fresh regardless — no gap for this to fill.

    # Fresh-install banner (2026-07-21) -- replaces the old bootstrap concept properly this
    # time: checks whether a scan has EVER completed (a real, install-lifetime marker file),
    # not a value that resets to "empty" on every restart. Never auto-fires anything itself
    # -- just tells the dashboard whether to show the "run your first scan?" banner, which
    # the user accepts or dismisses on their own terms (may still be adding API keys, etc.).
    state.needs_first_scan = not _FIRST_SCAN_MARKER.exists()
    if state.needs_first_scan:
        logger.info("No scan has ever completed on this install — dashboard will offer a first-scan banner")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
