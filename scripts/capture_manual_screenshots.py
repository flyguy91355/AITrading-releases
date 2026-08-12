#!/usr/bin/env python3
"""Capture static-page screenshots for USER_MANUAL.md via headless Chrome."""
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "docs" / "manual_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (url, output filename, window size)
SHOTS = [
    ("http://localhost:8080/", "dashboard_watchlist.png", "1400,900"),
    ("http://localhost:8080/settings", "settings_full.png", "1400,2400"),
]

def capture(url: str, filename: str, size: str) -> None:
    out_path = OUT_DIR / filename
    result = subprocess.run(
        [
            "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
            f"--screenshot={out_path}", f"--window-size={size}", url,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"FAILED {filename}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    if not out_path.exists() or out_path.stat().st_size < 1000:
        print(f"FAILED {filename}: output missing or too small", file=sys.stderr)
        sys.exit(1)
    print(f"OK {filename} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    for url, filename, size in SHOTS:
        capture(url, filename, size)
