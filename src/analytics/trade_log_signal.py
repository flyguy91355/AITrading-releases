"""The one shared BUY/SELL classification for data/trade_history/*.jsonl rows
(extracted 2026-08-31, GitHub #143).

Two separate readers consume that same log with deliberately different field
extraction -- src/analytics/composition_benchmark.py's parse_trade_events()
(ticker/is_buy/shares/timestamp, for daily-holdings reconstruction) and
src/tax/trade_log_reader.py's read_tax_events() (also price/trade_id/is_paper,
for cost basis and the paper-exclusion guarantee). That split is intentional
and stays. What was NOT intentional is that both independently re-implemented
the identical "BUY" in signal / "SELL" in signal detection plus the identical
warn-and-skip fallback -- and had already begun to drift (different warning
prefixes). A future change to the signal vocabulary would have needed making
in two places, with nothing to catch missing one: the two readers would just
silently start disagreeing about how many trades exist.

Only the classification convention lives here. Neither reader's own field
extraction moved."""

import logging

logger = logging.getLogger(__name__)


def classify_trade_signal(
    signal: str, *, source: str, ticker: str | None = None, file_name: str | None = None,
) -> bool | None:
    """Returns True for a buy signal, False for a sell, or None for anything
    unrecognized (e.g. "HOLD", which is never supposed to appear in an
    executed-trade log but is tolerated defensively -- skipped, not raised
    on). "BUY" matches "BUY" and "STRONG BUY"; "SELL" matches "SELL" and
    "STRONG SELL".

    An unrecognized signal is logged here, once, on the caller's behalf --
    `source` names the calling reader so the warning stays as traceable as
    each reader's own former message was."""
    if "BUY" in signal:
        return True
    if "SELL" in signal:
        return False
    logger.warning(
        "%s: unrecognized signal %r for %s in %s -- skipping",
        source, signal, ticker, file_name,
    )
    return None
