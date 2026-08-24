"""Generates buy/sell/hold signals from research reports."""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.research.engine import ResearchEngine, ResearchReport, Signal
from src.decision.risk_manager import RiskManager
from src.decision.portfolio import Portfolio

logger = logging.getLogger(__name__)


def _required_rr(conviction: float, min_conviction: float, base_rr: float, step: float, floor: float) -> float:
    """Conviction-scaled R/R threshold -- duplicated from web/app.py's own identical
    function (fixed 2026-08-09, GitHub #67), not imported from it: web/app.py already
    imports SignalGenerator at module load time, so importing back from web/app.py here
    would be circular; and web/app.py's own copy is exercised by tests/test_required_rr.py
    via direct AST extraction of a real ast.FunctionDef node in that file's own source,
    which an import statement wouldn't satisfy. Keep both copies identical -- this is the
    exact formula _required_rr in web/app.py uses for every other live buy gate (On Deck
    promotion, backfill, persist-check, cap-trim, swap). See that function's own docstring
    for the full reasoning: a flat min_risk_reward_ratio applied identically to every stock
    doesn't account for how much more confident Claude's thesis is; each conviction point
    above the minimum earns a small reduction in the required bar, floored so it never gets
    too lenient."""
    extra_conviction = max(0, conviction - min_conviction)
    return max(floor, base_rr - extra_conviction * step)


@dataclass
class TradeSignal:
    ticker: str
    signal: Signal
    conviction: int
    entry_price: float
    stop_loss: float
    take_profit_targets: list[float]
    position_size_pct: float
    position_size_dollars: float
    shares: float
    reasoning: str
    research_report: ResearchReport
    generated_at: datetime
    sector: str = ""
    should_execute: bool = False
    # Set by OrderManager._execute_buy right after a real fill (2026-07-27), echoing
    # Position.trade_id so trade_logger.log_trade(signal) can persist the same id to the
    # JSONL buy record. None until the buy actually executes; never set at all for a
    # signal that never becomes a real trade (e.g. a sell signal, or a rejected buy).
    trade_id: str | None = None
    # The exact R/R and its required gate at the moment THIS signal cleared to buy
    # (2026-08-21) -- not recomputed later, so Position.buy_rr/buy_required_rr (see
    # that field's own docstring) reflect the real numbers that allowed the purchase,
    # not a reconstruction from possibly-since-changed data. None for a sell signal or
    # any construction site that predates this field.
    rr: float | None = None
    required_rr: float | None = None


class SignalGenerator:
    def __init__(
        self,
        config: dict,
        research_engine: ResearchEngine,
        risk_manager: RiskManager,
        portfolio: Portfolio,
    ):
        self.config = config
        self.research_engine = research_engine
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        # No longer cached at __init__ (fixed 2026-08-09, GitHub #67) -- these used to be
        # snapshotted once here and never resynced on a live Settings change (unlike
        # RiskManager's own cached fields, which /api/settings explicitly resyncs). Read
        # fresh from self.config on every _evaluate_report call instead, matching how
        # every other live buy-gate call site in this codebase already reads
        # state.config[...] directly rather than caching it in an instance variable.

    def _evaluate_report(self, report: ResearchReport) -> TradeSignal | None:
        # AI Data Integrity guard (added 2026-08-24, GitHub #80) -- every other real
        # buy-decision call site in web/app.py checks is_fallback before treating a
        # report as buy-eligible; this was the sole exception. Currently latent (the
        # WS execute_buy/confirm_buy path this feeds has no live dashboard UI wired to
        # it today), but a real gap if that ever changes, or if a future fallback
        # source ever emits BUY (today's sole fallback, _rule_based_analysis, is
        # capped to HOLD/SELL/STRONG_SELL and never does).
        if getattr(report, "is_fallback", False):
            logger.info("  %s REJECTED: fallback (non-AI) report", report.ticker)
            return None

        min_conviction = self.config["research"]["min_conviction_score"]
        if report.conviction_score < min_conviction:
            logger.info("  %s REJECTED: conviction %d < %d", report.ticker, report.conviction_score, min_conviction)
            return None

        risk = report.entry_price - report.stop_loss
        if risk <= 0:
            logger.info("  %s REJECTED: risk <= 0 (entry $%.2f, stop $%.2f)", report.ticker, report.entry_price, report.stop_loss)
            return None

        # R/R reward side uses Claude's fair_value_estimate, not a fixed % of entry —
        # a flat percentage (e.g. the old T3 target) produces the identical ratio for
        # every stock regardless of actual upside, so it can never discriminate between
        # a genuinely undervalued stock and a fairly-valued one. fair_value_estimate is
        # real per-stock analysis output, so this makes R/R an actual quality filter.
        if not report.fair_value_estimate or report.fair_value_estimate <= 0:
            logger.info("  %s REJECTED: no valid fair_value_estimate for R/R check", report.ticker)
            return None
        reward = report.fair_value_estimate - report.entry_price
        rr = reward / risk if risk > 0 else 0
        # Conviction-scaled gate (fixed 2026-08-09, GitHub #67) -- this used to compare
        # against the flat, un-scaled min_risk_reward_ratio directly, unlike every other
        # live buy path (On Deck promotion, backfill, persist-check, cap-trim, swap),
        # which all use this same _required_rr formula. A high-conviction stock with a
        # real, system-qualifying R/R could be rejected here purely because it went
        # through this one stale code path instead of the others.
        research_cfg = self.config["research"]
        required_rr = _required_rr(
            report.conviction_score, min_conviction,
            research_cfg["min_risk_reward_ratio"],
            research_cfg.get("on_deck_rr_conviction_step", 0.1),
            research_cfg.get("on_deck_rr_floor", 1.5),
        )
        if reward <= 0 or rr < required_rr:
            logger.info("  %s REJECTED: R/R %.2f < %.2f (fair_value=$%.2f, entry=$%.2f, stop=$%.2f)", report.ticker, rr, required_rr, report.fair_value_estimate, report.entry_price, report.stop_loss)
            return None

        position_size = self.risk_manager.calculate_position_size(
            report.entry_price, report.stop_loss, self.portfolio.total_value
        )

        if not self.risk_manager.check_all_rules(report, self.portfolio):
            logger.info("  %s REJECTED: failed risk_manager.check_all_rules", report.ticker)
            return None

        # Use fractional shares — position_size_dollars is the notional amount
        shares = position_size / report.entry_price if report.entry_price > 0 else 0
        if shares < 0.001:
            logger.info("  %s REJECTED: position size too small ($%.2f)", report.ticker, position_size)
            return None

        return TradeSignal(
            ticker=report.ticker,
            signal=report.signal,
            conviction=report.conviction_score,
            entry_price=report.entry_price,
            stop_loss=report.stop_loss,
            take_profit_targets=report.take_profit_targets,
            position_size_pct=report.position_size_pct,
            position_size_dollars=position_size,
            shares=shares,
            reasoning=report.reasoning,
            research_report=report,
            generated_at=datetime.now(),
            sector=getattr(report, 'sector', ''),
            should_execute=report.signal in (Signal.STRONG_BUY, Signal.BUY),
            rr=rr,
            required_rr=required_rr,
        )

