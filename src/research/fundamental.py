"""Fundamental analysis module — scores financials across five dimensions."""

from dataclasses import dataclass

from src.data.market_data import Financials


@dataclass
class FundamentalScore:
    ticker: str
    overall_score: float
    revenue_growth_score: float
    profitability_score: float
    valuation_score: float
    balance_sheet_score: float
    cash_flow_score: float
    summary: str = ""


class FundamentalAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    async def analyze(self, financials: Financials) -> FundamentalScore:
        rg = self._score_revenue_growth(financials)
        prof = self._score_profitability(financials)
        val = self._score_valuation(financials)
        bs = self._score_balance_sheet(financials)
        cf = self._score_cash_flow(financials)

        overall = (rg * 0.25 + prof * 0.20 + val * 0.25 + bs * 0.15 + cf * 0.15)

        summary_parts = []
        if rg >= 7:
            summary_parts.append(f"Strong revenue growth ({financials.revenue_growth:.0%})")
        elif rg <= 3:
            summary_parts.append("Weak or negative revenue growth")

        if prof >= 7:
            summary_parts.append(f"Healthy margins (net {financials.net_margin:.0%})")
        elif prof <= 3:
            summary_parts.append("Thin or negative margins")

        if val >= 7:
            summary_parts.append("Attractively valued")
        elif val <= 3:
            summary_parts.append(f"Expensive valuation (P/E {financials.pe_ratio:.1f})")

        if bs >= 7:
            summary_parts.append("Strong balance sheet")
        elif bs <= 3:
            summary_parts.append("Leveraged balance sheet")

        if cf >= 7:
            summary_parts.append("Strong free cash flow generation")

        summary = ". ".join(summary_parts) + "." if summary_parts else "Insufficient data for detailed assessment."

        return FundamentalScore(
            ticker=financials.ticker,
            overall_score=round(overall, 1),
            revenue_growth_score=round(rg, 1),
            profitability_score=round(prof, 1),
            valuation_score=round(val, 1),
            balance_sheet_score=round(bs, 1),
            cash_flow_score=round(cf, 1),
            summary=summary,
        )

    def _score_revenue_growth(self, f: Financials) -> float:
        g = f.revenue_growth
        if g >= 0.30:
            return 10.0
        if g >= 0.20:
            return 8.0
        if g >= 0.10:
            return 6.5
        if g >= 0.05:
            return 5.0
        if g >= 0.0:
            return 3.5
        if g >= -0.10:
            return 2.0
        return 1.0

    def _score_profitability(self, f: Financials) -> float:
        scores = []
        gm = f.gross_margin
        if gm >= 0.60:
            scores.append(10)
        elif gm >= 0.40:
            scores.append(7)
        elif gm >= 0.20:
            scores.append(5)
        elif gm > 0:
            scores.append(3)
        else:
            scores.append(1)

        nm = f.net_margin
        if nm >= 0.25:
            scores.append(10)
        elif nm >= 0.15:
            scores.append(7.5)
        elif nm >= 0.05:
            scores.append(5)
        elif nm > 0:
            scores.append(3)
        else:
            scores.append(1)

        roe = f.roe
        if roe >= 0.25:
            scores.append(10)
        elif roe >= 0.15:
            scores.append(7)
        elif roe >= 0.08:
            scores.append(5)
        elif roe > 0:
            scores.append(3)
        else:
            scores.append(1)

        return sum(scores) / len(scores) if scores else 5.0

    def _score_valuation(self, f: Financials) -> float:
        scores = []

        pe = f.pe_ratio
        if pe > 0:
            if pe <= 12:
                scores.append(10)
            elif pe <= 18:
                scores.append(7.5)
            elif pe <= 25:
                scores.append(5.5)
            elif pe <= 35:
                scores.append(4)
            elif pe <= 50:
                scores.append(2.5)
            else:
                scores.append(1.5)

        peg = f.peg_ratio
        if peg > 0:
            if peg <= 1.0:
                scores.append(10)
            elif peg <= 1.5:
                scores.append(7)
            elif peg <= 2.0:
                scores.append(5)
            elif peg <= 3.0:
                scores.append(3)
            else:
                scores.append(1.5)

        pb = f.pb_ratio
        if pb > 0:
            if pb <= 1.5:
                scores.append(9)
            elif pb <= 3:
                scores.append(7)
            elif pb <= 5:
                scores.append(5)
            elif pb <= 10:
                scores.append(3)
            else:
                scores.append(2)

        return sum(scores) / len(scores) if scores else 5.0

    def _score_balance_sheet(self, f: Financials) -> float:
        scores = []

        de = f.debt_to_equity
        if de >= 0:
            if de <= 0.3:
                scores.append(10)
            elif de <= 0.6:
                scores.append(8)
            elif de <= 1.0:
                scores.append(6)
            elif de <= 2.0:
                scores.append(4)
            else:
                scores.append(2)

        cr = f.current_ratio
        if cr > 0:
            if cr >= 2.0:
                scores.append(9)
            elif cr >= 1.5:
                scores.append(7)
            elif cr >= 1.0:
                scores.append(5)
            else:
                scores.append(2)

        qr = f.quick_ratio
        if qr > 0:
            if qr >= 1.5:
                scores.append(9)
            elif qr >= 1.0:
                scores.append(7)
            elif qr >= 0.5:
                scores.append(4)
            else:
                scores.append(2)

        return sum(scores) / len(scores) if scores else 5.0

    def _score_cash_flow(self, f: Financials) -> float:
        fcf = f.free_cash_flow
        rev = f.revenue

        if fcf <= 0:
            return 2.0

        if rev > 0:
            fcf_margin = fcf / rev
            if fcf_margin >= 0.20:
                return 10.0
            if fcf_margin >= 0.10:
                return 7.5
            if fcf_margin >= 0.05:
                return 5.5
            return 4.0

        return 6.0
