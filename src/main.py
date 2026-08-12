"""System entry point and orchestrator for AITrading.

DEPRECATED: This module has been superseded by web/app.py, which runs the full
trading system plus the live dashboard in a single process. Running both
simultaneously causes double-trading on the same Alpaca account.

Start the system with:
    python -m uvicorn web.app:app --host 0.0.0.0 --port 8080
"""

raise SystemExit(
    "\n"
    "  ERROR: src/main.py is deprecated — use web/app.py instead.\n"
    "  Running both processes simultaneously causes double-trading.\n"
    "\n"
    "  Start the system with:\n"
    "    python -m uvicorn web.app:app --host 0.0.0.0 --port 8080\n"
)
