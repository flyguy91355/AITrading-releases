"""Insider activity analysis module — interprets insider transaction patterns."""

from dataclasses import dataclass

from src.data.insider_tracker import InsiderSummary, TransactionType


@dataclass
class InsiderAnalysisResult:
    ticker: str
    signal_strength: float
    cluster_buying_detected: bool
    notable_buys: int
    notable_sells: int
    net_insider_sentiment: str
    summary: str = ""


class InsiderAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    async def analyze(self, insider_summary: InsiderSummary) -> InsiderAnalysisResult:
        buys = insider_summary.buy_count_90d
        sells = insider_summary.sell_count_90d
        cluster = insider_summary.cluster_buying
        ratio = insider_summary.buy_sell_ratio

        signal = self._compute_signal(buys, sells, ratio, cluster, insider_summary)

        if signal >= 0.6:
            sentiment = "BULLISH"
        elif signal >= 0.2:
            sentiment = "SLIGHTLY BULLISH"
        elif signal <= -0.6:
            sentiment = "BEARISH"
        elif signal <= -0.2:
            sentiment = "SLIGHTLY BEARISH"
        else:
            sentiment = "NEUTRAL"

        notable_buys = sum(
            1 for t in insider_summary.notable_transactions
            if t.transaction_type == TransactionType.BUY and t.total_value >= 100_000
        )
        notable_sells = sum(
            1 for t in insider_summary.notable_transactions
            if t.transaction_type == TransactionType.SELL and t.total_value >= 100_000
        )

        summary = self._build_summary(buys, sells, cluster, notable_buys, notable_sells, sentiment, insider_summary)

        return InsiderAnalysisResult(
            ticker=insider_summary.ticker,
            signal_strength=round(signal, 2),
            cluster_buying_detected=cluster,
            notable_buys=notable_buys,
            notable_sells=notable_sells,
            net_insider_sentiment=sentiment,
            summary=summary,
        )

    def _compute_signal(
        self, buys: int, sells: int, ratio: float, cluster: bool, summary: InsiderSummary
    ) -> float:
        if buys == 0 and sells == 0:
            return 0.0

        ratio_signal = 0.0
        if ratio >= 3.0:
            ratio_signal = 0.8
        elif ratio >= 2.0:
            ratio_signal = 0.5
        elif ratio >= 1.0:
            ratio_signal = 0.2
        elif sells > 0 and buys == 0:
            ratio_signal = -0.5
        elif ratio < 0.5:
            ratio_signal = -0.3

        cluster_bonus = 0.3 if cluster else 0.0

        large_buy_bonus = 0.0
        for t in summary.notable_transactions:
            if t.transaction_type == TransactionType.BUY and t.total_value >= 500_000:
                large_buy_bonus = max(large_buy_bonus, 0.2)
            if t.transaction_type == TransactionType.BUY and t.total_value >= 1_000_000:
                large_buy_bonus = max(large_buy_bonus, 0.4)

        signal = ratio_signal + cluster_bonus + large_buy_bonus
        return max(-1.0, min(1.0, signal))

    def _build_summary(
        self, buys: int, sells: int, cluster: bool,
        notable_buys: int, notable_sells: int, sentiment: str,
        insider_summary: InsiderSummary,
    ) -> str:
        parts = [f"Last 90 days: {buys} insider buy(s), {sells} sell(s)."]

        if cluster:
            parts.append("CLUSTER BUYING detected — multiple insiders purchasing within a short window.")

        if notable_buys > 0:
            parts.append(f"{notable_buys} notable buy(s) over $100K.")
        if notable_sells > 0:
            parts.append(f"{notable_sells} notable sell(s) over $100K.")

        if insider_summary.notable_transactions:
            top = insider_summary.notable_transactions[0]
            action = "bought" if top.transaction_type == TransactionType.BUY else "sold"
            parts.append(
                f"Largest transaction: {top.insider_name} ({top.title}) "
                f"{action} {top.shares:,} shares at ${top.price:.2f} "
                f"(${top.total_value:,.0f})."
            )

        parts.append(f"Overall insider sentiment: {sentiment}.")
        return " ".join(parts)
