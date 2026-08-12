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
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trade_history(self, days: int | None = 30) -> list[dict]:
        """days=None means no limit — glob every JSONL file, not just the most recent N.
        Relies on Python's [:None] slice already meaning "to the end"; the type hint just
        documents that as intentional rather than leaving it as an accident of slicing
        semantics a future refactor could break."""
        trades = []
        for log_file in sorted(self.log_dir.glob("*.jsonl"), reverse=True)[:days]:
            with open(log_file) as f:
                for line in f:
                    trades.append(json.loads(line))
        return trades
