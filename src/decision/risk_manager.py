"""Risk management and position sizing."""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.research.engine import ResearchReport
    from src.decision.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        rm = config["risk_management"]
        self.max_position_pct = rm["max_position_pct"] / 100
        self.max_loss_per_trade_pct = rm["max_loss_per_trade_pct"] / 100
        self.min_cash_reserve_pct = rm["min_cash_reserve_pct"] / 100
        self.max_sector_positions = rm["max_sector_positions"]
        self.sector_concentration_enabled = rm.get("sector_concentration_enabled", True)
        self.daily_loss_limit_pct = rm["daily_loss_limit_pct"] / 100
        self.drawdown_halt_pct = rm["drawdown_halt_pct"] / 100
        self.drawdown_defensive_pct = rm["drawdown_defensive_pct"] / 100
        self.drawdown_exit_review_pct = rm["drawdown_exit_review_pct"] / 100
        self.max_positions = max(1, config.get("portfolio", {}).get("max_positions", 10))
        # Wash-sale rebuy block (2026-07-27) -- IRC §1091 disallows a capital-loss
        # deduction if the same (or a substantially identical) security is rebought
        # within 30 days of a losing sale. Rather than detect this after the fact, this
        # system refuses the rebuy outright so every realized loss stays immediately
        # usable at tax time. Applies to every real buy path -- both automated (On Deck
        # promotion, rotation swap) and manual -- since both route through
        # check_all_rules. See docs/superpowers -- project-wash-sale-rebuy-block.
        self.wash_sale_cooldown_days = config.get("research", {}).get("wash_sale_cooldown_days", 30)

    def calculate_position_size(self, entry_price: float, stop_loss: float, portfolio_value: float) -> float:
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0.0

        max_risk_dollars = portfolio_value * self.max_loss_per_trade_pct
        shares_by_risk = max_risk_dollars / risk_per_share
        size_by_risk = shares_by_risk * entry_price

        # Cap per-position so filling all slots still respects cash reserve
        derived_pct = (1.0 - self.min_cash_reserve_pct) / self.max_positions
        max_position = portfolio_value * min(self.max_position_pct, derived_pct)
        return min(size_by_risk, max_position)

    def check_cash_reserve(self, portfolio: Portfolio, order_cost: float) -> bool:
        remaining_cash = portfolio.cash - order_cost
        return remaining_cash >= portfolio.total_value * self.min_cash_reserve_pct

    def check_sector_concentration(self, portfolio: Portfolio, sector: str) -> bool:
        if not self.sector_concentration_enabled:
            return True  # user has turned the check off entirely (Settings page)
        if not sector:
            return True  # unknown sector — cannot enforce concentration limit
        sector_count = sum(1 for p in portfolio.positions.values() if p.sector == sector)
        return sector_count < self.max_sector_positions

    def check_drawdown(self, portfolio: Portfolio) -> str:
        if portfolio.peak_value == 0:
            return "normal"
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        if drawdown >= self.drawdown_exit_review_pct:
            return "exit_review"
        if drawdown >= self.drawdown_defensive_pct:
            return "defensive"
        if drawdown >= self.drawdown_halt_pct:
            return "halt"
        return "normal"

    def check_daily_loss(self, portfolio: Portfolio) -> bool:
        if portfolio.total_value == 0:
            return False
        if portfolio.day_start_value == 0:
            return True  # no measurable loss yet; allow trading
        daily_loss = (portfolio.day_start_value - portfolio.total_value) / portfolio.day_start_value
        return daily_loss < self.daily_loss_limit_pct

    def check_wash_sale_cooldown(self, ticker: str, portfolio: Portfolio) -> bool:
        """True (passes) unless this ticker had a losing sale within the cooldown window
        -- see self.wash_sale_cooldown_days' docstring. portfolio.recent_losses is kept
        in memory, updated synchronously alongside every real trade_history SELL write
        (Portfolio.close_position_async and OrderManager's other insert sites), so this
        needs no DB access here and can run inside this otherwise-synchronous method."""
        last_loss = portfolio.recent_losses.get(ticker)
        if last_loss is None:
            return True
        return datetime.now() - last_loss >= timedelta(days=self.wash_sale_cooldown_days)

    def check_all_rules(self, report: ResearchReport, portfolio: Portfolio) -> bool:
        order_cost = self.calculate_position_size(report.entry_price, report.stop_loss, portfolio.total_value)

        if not self.check_wash_sale_cooldown(report.ticker, portfolio):
            last_loss = portfolio.recent_losses.get(report.ticker)
            logger.info("  %s RULE FAIL: wash-sale cooldown — lost money on this ticker on %s, "
                        "%d-day rebuy block still active",
                        report.ticker, last_loss.date() if last_loss else "?", self.wash_sale_cooldown_days)
            return False
        if not self.check_cash_reserve(portfolio, order_cost):
            remaining = portfolio.cash - order_cost
            required = portfolio.total_value * self.min_cash_reserve_pct
            logger.info("  %s RULE FAIL: cash reserve — need $%.0f, would have $%.0f after $%.0f order (cash: $%.0f)",
                        report.ticker, required, remaining, order_cost, portfolio.cash)
            return False
        if not self.check_daily_loss(portfolio):
            logger.info("  %s RULE FAIL: daily loss limit exceeded", report.ticker)
            return False
        drawdown_status = self.check_drawdown(portfolio)
        if drawdown_status != "normal":
            logger.info("  %s RULE FAIL: drawdown status = %s", report.ticker, drawdown_status)
            return False
        if not self.check_sector_concentration(portfolio, getattr(report, 'sector', '')):
            logger.info("  %s RULE FAIL: sector concentration limit (%d) reached for sector '%s'",
                        report.ticker, self.max_sector_positions, getattr(report, 'sector', ''))
            return False
        return True
