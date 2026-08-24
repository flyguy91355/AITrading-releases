"""Competitive analysis module — sector positioning and moat assessment."""

import asyncio
from dataclasses import dataclass

import yfinance as yf

from src.research.market_cap import market_cap_tier_label


SECTOR_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "SAMSUNG"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "AMZN": ["WMT", "SHOP", "MSFT"],
    "NVDA": ["AMD", "INTC", "AVGO"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "TSLA": ["F", "GM", "RIVN"],
}


@dataclass
class CompetitiveAnalysis:
    ticker: str
    market_position: str
    moat_assessment: str
    moat_score: float
    key_competitors: list[str]
    industry_outlook: str
    summary: str = ""


class CompetitorAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    async def analyze(self, ticker: str) -> CompetitiveAnalysis:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        competitors = SECTOR_PEERS.get(ticker, [])
        if not competitors:
            sector_key = info.get("sectorKey", "")
            industry_key = info.get("industryKey", "")
            competitors = await self._find_peers(ticker, sector_key, industry_key)

        market_cap = info.get("marketCap", 0) or 0
        market_position = self._assess_position(market_cap, info)

        moat_assessment, moat_score = self._assess_moat(info)

        peer_comparison = await self._compare_peers(ticker, competitors, info)

        industry_outlook = self._assess_industry(info)

        summary_parts = [
            f"{ticker} holds a {market_position} position in {info.get('industry', 'its industry')}.",
            f"Moat: {moat_assessment} ({moat_score:.1f}/10).",
        ]
        if competitors:
            summary_parts.append(f"Key competitors: {', '.join(competitors[:4])}.")
        if peer_comparison:
            summary_parts.append(peer_comparison)
        summary_parts.append(f"Industry outlook: {industry_outlook}.")

        return CompetitiveAnalysis(
            ticker=ticker,
            market_position=market_position,
            moat_assessment=moat_assessment,
            moat_score=moat_score,
            key_competitors=competitors[:5],
            industry_outlook=industry_outlook,
            summary=" ".join(summary_parts),
        )

    # Position phrases keyed off market_cap_tier_label's own shared tiers (fixed
    # 2026-08-24, GitHub #82) -- this used to have its own independent 6-tier
    # boundary set that drifted from engine.py's _market_cap_tier_label despite a
    # docstring there claiming they matched: the same $15B company was "large-cap"
    # in one and "mid-cap contender" in the other. Now both are always driven by
    # the same underlying tier, so they can't disagree on which bucket a company
    # falls into again -- only the descriptive wording differs by design.
    _POSITION_PHRASES = {
        "mega-cap": "dominant mega-cap leader",
        "large-cap": "major large-cap player",
        "mid-cap": "mid-cap contender",
        "small-cap": "small/micro-cap niche player",
        "": "unclassified-cap",
    }

    def _assess_position(self, market_cap: float, info: dict) -> str:
        tier = market_cap_tier_label(market_cap)
        return self._POSITION_PHRASES[tier]

    def _assess_moat(self, info: dict) -> tuple[str, float]:
        score = 5.0
        factors = []

        market_cap = info.get("marketCap", 0) or 0
        if market_cap >= 500_000_000_000:
            score += 1.5
            factors.append("scale advantage")
        elif market_cap >= 100_000_000_000:
            score += 0.5

        gross_margin = info.get("grossMargins", 0) or 0
        if gross_margin >= 0.60:
            score += 1.5
            factors.append("high-margin pricing power")
        elif gross_margin >= 0.40:
            score += 0.5

        roe = info.get("returnOnEquity", 0) or 0
        if roe >= 0.25:
            score += 1.0
            factors.append("superior returns on equity")
        elif roe >= 0.15:
            score += 0.5

        rev_growth = info.get("revenueGrowth", 0) or 0
        if rev_growth >= 0.20:
            score += 0.5
            factors.append("strong growth trajectory")

        score = min(10.0, max(1.0, score))

        if score >= 8:
            assessment = f"Wide moat — {', '.join(factors)}" if factors else "Wide moat"
        elif score >= 6:
            assessment = f"Moderate moat — {', '.join(factors)}" if factors else "Moderate moat"
        elif score >= 4:
            assessment = "Narrow moat — limited competitive advantages"
        else:
            assessment = "No meaningful moat — highly competitive space"

        return assessment, score

    async def _compare_peers(self, ticker: str, peers: list[str], info: dict) -> str:
        if not peers:
            return ""

        our_pe = info.get("trailingPE", 0) or 0

        peer_metrics = []
        for peer in peers[:3]:
            try:
                p = yf.Ticker(peer)
                pi = p.info or {}
                peer_pe = pi.get("trailingPE", 0) or 0
                peer_metrics.append((peer, peer_pe))
            except Exception:
                continue

        if not peer_metrics:
            return ""

        avg_pe = sum(m[1] for m in peer_metrics) / len(peer_metrics)

        if our_pe > 0 and avg_pe > 0:
            if our_pe < avg_pe * 0.8:
                return f"{ticker} trades at a discount to peers (P/E {our_pe:.1f} vs peer avg {avg_pe:.1f})."
            elif our_pe > avg_pe * 1.2:
                return f"{ticker} trades at a premium to peers (P/E {our_pe:.1f} vs peer avg {avg_pe:.1f})."
            else:
                return f"{ticker} trades in line with peers (P/E {our_pe:.1f} vs peer avg {avg_pe:.1f})."
        return ""

    async def _find_peers(self, ticker: str, sector_key: str, industry_key: str) -> list[str]:
        """Finds real peer tickers via yfinance's Industry/Sector top_companies data
        (fixed 2026-08-03 -- this was a permanent stub always returning [], silently
        disabling peer comparison for every ticker outside the 7 hardcoded SECTOR_PEERS
        mega-caps, i.e. essentially every stock this system actually scans). Takes the
        yfinance-native "key" slugs (e.g. "banks-diversified", not the human-readable
        "Banks - Diversified") -- callers must pass info["industryKey"]/["sectorKey"],
        not info["industry"]/["sector"]. Tries industry first (more specific/relevant
        for a peer comparison than the broader sector), falls back to sector if the
        industry lookup is empty or fails outright."""
        for key, cls in ((industry_key, yf.Industry), (sector_key, yf.Sector)):
            if not key:
                continue
            try:
                domain = await asyncio.to_thread(cls, key)
                top = await asyncio.to_thread(lambda d=domain: d.top_companies)
            except Exception:
                continue
            if top is None or top.empty:
                continue
            peers = [t for t in top.index.tolist() if t != ticker]
            if peers:
                return peers[:5]
        return []

    def _assess_industry(self, info: dict) -> str:
        growth = info.get("revenueGrowth", 0) or 0
        sector = info.get("sector", "Unknown")

        if growth >= 0.15:
            return f"{sector} sector showing strong growth momentum"
        if growth >= 0.05:
            return f"{sector} sector with moderate growth"
        if growth >= 0:
            return f"{sector} sector in stable/mature phase"
        return f"{sector} sector facing headwinds"
