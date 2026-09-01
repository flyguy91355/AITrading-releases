"""Trade history and reasoning log."""

import json
from datetime import datetime
from pathlib import Path


class TradeLogger:
    def __init__(self, config: dict):
        self.config = config
        self.log_dir = Path("data/trade_history")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_trade(self, signal, is_paper: bool):
        entry = {
            "ticker": signal.ticker,
            "signal": signal.signal.value,
            "conviction": signal.conviction,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "shares": signal.shares,
            "position_size": signal.position_size_dollars,
            "reasoning": signal.reasoning,
            "timestamp": datetime.now().isoformat(),
            # 2026-07-27: links this buy record to every sell tranche that later closes
            # the same position (Position.trade_id) -- None for a signal that predates
            # this field, or that isn't a real executed buy.
            "trade_id": getattr(signal, "trade_id", None),
            # 2026-08-02: stamped from the broker's real paper/live state at the moment
            # this trade was logged -- required (no default) so a future call site that
            # forgets it fails loudly instead of silently mislabeling a real trade. See
            # docs/superpowers/specs/2026-08-02-tax-lot-tracking-design.md.
            "is_paper": is_paper,
        }

        log_file = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        # encoding= explicit (2026-08-31) -- the CLAUDE.md Windows-encoding rule
        # covers src/**, and bare open() is the exact variant the AST lock-in test
        # (read_text/write_text only) can't see.
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_sell_fill(self, ticker: str, shares: float, price: float, reason: str,
                      trade_id: str | None, is_paper: bool):
        """JSONL record for a sell executed INSIDE OrderManager (2026-08-31,
        full-codebase review) -- take-profit tranches, a stop-tranche fill, and
        reconciled/unreconciled untracked fills all wrote only the SQL
        trade_history table, but src/tax/trade_log_reader.read_tax_events and
        src/analytics/composition_benchmark.parse_trade_events reconstruct
        exclusively from these JSONL files: every such sell was invisible to
        Form 8949's FIFO lot matching (realized gains silently omitted, later
        same-ticker buys matched against already-sold shares) and to the
        composition benchmark's daily-holdings replay (shares overstated after
        every tranche). Same record shape log_trade produces for a sell --
        signal containing "SELL" (both readers' classify_trade_signal
        convention), entry_price carrying the REAL fill price (read_tax_events
        reads price from the entry_price field), plus shares/trade_id/is_paper.
        is_paper stays a required parameter, no default -- same fail-loudly rule
        as log_trade."""
        entry = {
            "ticker": ticker,
            "signal": "SELL",
            "conviction": 0,
            "entry_price": price,
            "stop_loss": 0.0,
            "shares": shares,
            "position_size": shares * price,
            "reasoning": reason,
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade_id,
            "is_paper": is_paper,
        }
        log_file = self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trade_history(self, days: int | None = 30) -> list[dict]:
        """days=None means no limit — glob every JSONL file, not just the most recent N.
        Relies on Python's [:None] slice already meaning "to the end"; the type hint just
        documents that as intentional rather than leaving it as an accident of slicing
        semantics a future refactor could break."""
        trades = []
        for log_file in sorted(self.log_dir.glob("*.jsonl"), reverse=True)[:days]:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    trades.append(json.loads(line))
        return trades
