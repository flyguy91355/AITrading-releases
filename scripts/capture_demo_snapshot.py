#!/usr/bin/env python3
"""Captures a full JSON snapshot of the live dashboard's real data for the
investor demo -- see
docs/superpowers/specs/2026-08-14-interactive-demo-design.md for the full
design. Run manually whenever the demo needs refreshing; never scheduled.

Usage: python3 scripts/capture_demo_snapshot.py <dashboard-password>
Writes: data/demo_snapshot.json (refuses to write if the sensitive-data
scan below finds a match -- see scan_for_sensitive_data).
"""
import json
import re
import sys
from pathlib import Path

import requests

HOST = "https://aitrading-hetzner.tail52228a.ts.net:8080"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "demo_snapshot.json"
ON_SHORE_DETAIL_CAP = 15

# Real patterns this specific artifact must never leak -- see the design
# spec's "No .env-derived secret" goal. Alpaca paper account IDs are always
# "PA" + 9 alphanumeric chars; Alpaca API key IDs start "AK"; Anthropic keys
# start "sk-ant-"; 100.x.x.x is this project's real Tailscale CGNAT range;
# the hostname is this specific box's real, private MagicDNS name.
_SENSITIVE_PATTERNS = [
    re.compile(r"\bPA[0-9A-Z]{8,14}\b"),
    re.compile(r"\bAK[0-9A-Z]{18,}\b"),
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"tail52228a\.ts\.net"),
]


def scan_for_sensitive_data(blob: str) -> list[str]:
    """Pure function, no network -- returns every pattern (not every match)
    that hit, so this module's own test suite and the demo builder can both
    call this without needing a live snapshot to test against."""
    return [p.pattern for p in _SENSITIVE_PATTERNS if p.search(blob)]


def login(password: str) -> requests.Session:
    session = requests.Session()
    resp = session.post(f"{HOST}/login", data={"password": password}, allow_redirects=True)
    if resp.url.rstrip("/").endswith("/login"):
        raise RuntimeError("Login failed -- check the password")
    return session


def get_json(session: requests.Session, path: str):
    resp = session.get(f"{HOST}{path}")
    resp.raise_for_status()
    return resp.json()


def capture_all(session: requests.Session) -> dict:
    snapshot: dict = {}

    def capture(path: str):
        print(f"Capturing {path}")
        snapshot[path] = get_json(session, path)

    for path in [
        "/api/dashboard-poll", "/api/near-miss", "/api/today-scan-rejects",
        "/api/research-reports", "/api/trade-history", "/api/weekly-pnl-history",
        "/api/daily-pnl-history", "/api/win-loss-trades", "/api/portfolio-summary",
        "/api/portfolio-health-model", "/api/chart-display-config",
        "/api/conviction-gate-config", "/api/ticker-tape-config", "/api/update-status",
        "/api/performance-history?range=all", "/api/performance-history?range=ytd",
        "/api/performance-today",
    ]:
        capture(path)

    init = snapshot["/api/dashboard-poll"]
    held_tickers = [p["ticker"] for p in init.get("portfolio", {}).get("positions", [])]
    on_deck_tickers = list(snapshot["/api/near-miss"].keys())
    on_shore_tickers = list(snapshot["/api/today-scan-rejects"].keys())

    for ticker in held_tickers:
        capture(f"/api/position/{ticker}/history")
        capture(f"/api/stock-report/{ticker}")
    for ticker in on_deck_tickers:
        capture(f"/api/near-miss/{ticker}/history")
    # Top 15 only -- see the design spec's Architecture section. on_shore_tickers
    # is already ranked best-first by the real endpoint (dict(ranked), Python
    # dicts preserve insertion order), so this slice is genuinely the top 15.
    for ticker in on_shore_tickers[:ON_SHORE_DETAIL_CAP]:
        capture(f"/api/today-scan-rejects/{ticker}/history")

    return snapshot


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/capture_demo_snapshot.py <dashboard-password>")
        sys.exit(1)

    session = login(sys.argv[1])
    snapshot = capture_all(session)

    blob = json.dumps(snapshot)
    hits = scan_for_sensitive_data(blob)
    if hits:
        print("REFUSING TO WRITE SNAPSHOT -- sensitive data detected:")
        for h in hits:
            print(f"  matched pattern: {h}")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(blob)
    print(f"Wrote {OUTPUT_PATH} ({len(blob):,} bytes, {len(snapshot)} endpoints captured)")


if __name__ == "__main__":
    main()
