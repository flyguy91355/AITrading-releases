"""Core research engine powered by Claude AI."""

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

from src.data.market_data import MarketDataFetcher, format_long_term_trend_summary
from src.data.insider_tracker import InsiderTracker
from src.data.news_feed import NewsFeed
from src.research.fundamental import FundamentalAnalyzer
from src.research.sentiment import SentimentAnalyzer
from src.research.insider_analysis import InsiderAnalyzer
from src.research.competitor import CompetitorAnalyzer

logger = logging.getLogger(__name__)


class Signal(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NO_ACTION = "NO ACTION"


class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


def _parse_json_bool(value, default: bool) -> bool:
    """Type-safe parse of a boolean-ish field from Claude's JSON response (fixed
    2026-08-09, GitHub #66) -- a bare bool(value) coercion silently flips the intended
    answer if Claude ever quotes the boolean as a JSON string instead of a raw JSON
    boolean: bool("false") is True in Python, since any non-empty string is truthy.
    Concrete reachable danger: _backfill_on_deck_from_on_shore calls
    recommend_on_deck_retention with fail_default=False -- a real Claude "no, not a
    good buy anymore" quoted as the string "false" would flip to True and the
    candidate would be re-admitted to the live On Deck buy pipeline, contrary to the
    AI's actual judgment. A real JSON bool parses through json.loads as a genuine
    Python bool already, so this only changes behavior for the malformed-string case;
    anything else (missing key, unrecognized string, wrong type) falls back to
    `default`, matching this codebase's existing AI Data Integrity principle of never
    inventing a new true/false answer the model didn't clearly give."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return default


def _clamp_ai_stop_loss(
    raw_stop: float, entry_price: float, min_pct: float, max_pct: float,
) -> float:
    """Bounds an AI-chosen stop-loss into [entry*(1 - max_pct/100), entry*(1 - min_pct/100)]
    (2026-07-31, AI-chosen stop-loss/TP feature) -- a real technical level (support,
    recent swing low) Claude identifies could still be unusually tight or wide due to a
    bad read; this guards against a degenerate stop reaching the actual order without
    discarding Claude's judgment for the normal case. Only applied when
    research.ai_chosen_stop_tp_enabled is true -- the existing fixed-percentage mode
    never calls this. entry_price <= 0 is a defensive no-op (never expected in
    practice) -- there's no sensible bound to compute against a zero/negative price, so
    the raw value passes through unchanged rather than raising."""
    if entry_price <= 0:
        return raw_stop
    floor = entry_price * (1 - max_pct / 100)
    ceiling = entry_price * (1 - min_pct / 100)
    return max(floor, min(raw_stop, ceiling))


@dataclass
class ResearchReport:
    ticker: str
    company_name: str
    generated_at: datetime
    conviction_score: float
    signal: Signal
    risk_level: RiskLevel
    thesis: str
    fundamental_summary: str
    insider_summary: str
    news_summary: str
    competitive_summary: str
    risk_factors: str
    recommended_action: str
    entry_price: float
    position_size_pct: float
    stop_loss: float
    take_profit_targets: list[float] = field(default_factory=list)
    time_horizon: str = ""
    reasoning: str = ""
    sector: str = ""
    is_fallback: bool = False
    fair_value_estimate: float = 0.0
    margin_of_safety_pct: float = 0.0
    business_summary: str = ""


@dataclass
class DeepDiveReport:
    ticker: str
    company_name: str
    generated_at: datetime
    signal: str
    conviction_score: float
    thesis: str
    valuation_analysis: str
    fair_value_estimate: float
    margin_of_safety_pct: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    risk_factors: str = ""
    technical_analysis: str = ""
    competitive_moat: str = ""
    growth_outlook: str = ""
    catalysts: list[dict] = field(default_factory=list)
    risk_scenarios: list[dict] = field(default_factory=list)
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    monitoring_checklist: list[str] = field(default_factory=list)
    enhanced_reasoning: str = ""
    # Deeper dive fields (populated by deeper_dive_analysis only)
    peer_comparison: str = ""
    management_quality: str = ""
    macro_sensitivity: str = ""
    historical_patterns: str = ""
    expected_return_6m: float = 0.0
    expected_return_12m: float = 0.0
    is_deeper_dive: bool = False
    is_fallback: bool = False
    fallback_reason: str = ""


DEEP_DIVE_PROMPT = """\
You are a senior equity research analyst producing an institutional-grade DEEP DIVE research report. This is a full comprehensive analysis — go beyond surface metrics to deliver a definitive investment verdict.

TODAY'S DATE: {current_date} — every catalyst timeframe or referenced event must be realistic relative to this date. Never reference a quarter, month, or year that has already passed.

STOCK: {ticker} — {company_name}
CURRENT PRICE: ${current_price:.2f}
INITIAL SCREEN SIGNAL: {signal} (Conviction: {conviction}/10)
INITIAL SCREEN FAIR VALUE ESTIMATE: ${prior_fair_value:.2f} (margin of safety {prior_mos:.1f}%) — your own quick-scan estimate for this same stock, moments ago. Refine this with the full valuation work below rather than deriving an unrelated new estimate from scratch; a material change from this number should be explainable by something specific in your reasoning, not an unexplained jump.

── FUNDAMENTAL DATA ──
{fundamental_summary}

── INSIDER ACTIVITY ──
{insider_summary}

── NEWS & SENTIMENT ──
{news_summary}

── COMPETITIVE POSITION ──
{competitive_summary}

── TECHNICAL CONTEXT ──
{technical_summary}

Perform a COMPREHENSIVE deep-dive. Evaluate ALL of the following dimensions:

1. SIGNAL & CONVICTION — your final verdict (STRONG BUY / BUY / HOLD / SELL / STRONG SELL) and why
2. INVESTMENT THESIS — the core 2-3 sentence case for or against this stock
3. VALUATION — DCF-style intrinsic value reasoning; is the stock cheap, fair, or expensive?
4. TECHNICAL ANALYSIS — trend direction, momentum, key support/resistance levels, volume patterns
5. COMPETITIVE MOAT — pricing power, switching costs, market share, barriers to entry
6. GROWTH OUTLOOK — revenue/earnings growth trajectory, margin expansion or compression
7. CATALYSTS — specific upcoming events that could move the stock
8. RISK SCENARIOS — bull, base, bear cases with price targets and probabilities
9. RISK FACTORS — what could invalidate the thesis
10. ENTRY & STOP — optimal entry zone, stop loss level
11. MONITORING — key metrics and events to watch weekly

Respond as JSON with these exact fields:
{{
    "signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>",
    "conviction_score": <1-10, one decimal place, e.g. 7.3>,
    "thesis": "<2-3 sentence investment thesis — the core reason to buy or avoid>",
    "valuation_analysis": "<detailed DCF-style valuation reasoning>",
    "fair_value_estimate": <intrinsic value estimate as a number>,
    "margin_of_safety_pct": <percentage below fair value the current price represents — negative if overvalued>,
    "entry_price": <recommended entry price>,
    "stop_loss": <stop loss price — typically 5-8% below entry>,
    "technical_analysis": "<trend, momentum, support/resistance, volume — 2-3 sentences>",
    "competitive_moat": "<moat quality: wide/narrow/none and why — pricing power, switching costs, market share>",
    "growth_outlook": "<revenue and earnings growth trajectory, margin trends — 2-3 sentences>",
    "risk_factors": "<key risks that could invalidate the thesis>",
    "catalysts": [
        {{"event": "<specific catalyst>", "timeframe": "<when>", "impact": "<expected price impact>"}}
    ],
    "risk_scenarios": [
        {{"scenario": "Bull", "probability_pct": <number>, "price_target": <number>, "description": "<what drives this outcome>"}},
        {{"scenario": "Base", "probability_pct": <number>, "price_target": <number>, "description": "<what drives this outcome>"}},
        {{"scenario": "Bear", "probability_pct": <number>, "price_target": <number>, "description": "<what drives this outcome>"}}
    ],
    "entry_zone_low": <optimal buy zone low price>,
    "entry_zone_high": <optimal buy zone high price>,
    "monitoring_checklist": ["<specific metric or event to track weekly — be concrete>"],
    "enhanced_reasoning": "<comprehensive narrative connecting all dimensions — valuation, technicals, moat, catalysts, risks — into a unified investment view>"
}}

Rules:
- STRONG BUY requires conviction >= 8, clear moat, MOS >= 20%
- BUY requires conviction >= 7, positive technicals, MOS >= 10%
- HOLD if thesis is intact but entry is stretched or risk/reward is borderline
- SELL if fundamentals deteriorating or price at/above fair value with no catalyst
- STRONG SELL if significant downside risk, broken thesis, or structural headwinds
- probability_pct across all three scenarios must sum to 100
- Every numeric field you output (stop_loss, entry_price, entry_zone_low, entry_zone_high) must be used consistently everywhere else in your response, including inside enhanced_reasoning — never state a different number in your narrative than what you output in these fields.
- fair_value_estimate must be the actual output of the valuation methodology you describe in valuation_analysis — never state one intrinsic value range or figure in valuation_analysis and then output a different number in the fair_value_estimate field.

Respond with ONLY the JSON object, no other text.
"""


DEEPER_DIVE_PROMPT = """\
You are a senior equity research analyst producing an institutional-grade DEEPER DIVE — the most comprehensive analysis possible on a single stock. You have already completed a standard deep dive. Now go significantly further.

STOCK: {ticker} — {company_name}
CURRENT PRICE: ${current_price:.2f}

── PRIOR DEEP DIVE FINDINGS ──
Signal: {prior_signal} | Conviction: {prior_conviction}/10
Thesis: {prior_thesis}
Fair Value: ${prior_fair_value:.2f} | Margin of Safety: {prior_mos:.1f}%
Entry Zone: ${prior_ez_low:.2f} – ${prior_ez_high:.2f} | Stop: ${prior_stop:.2f}
Valuation: {prior_valuation}
Technical: {prior_technical}
Moat: {prior_moat}
Growth: {prior_growth}
Risk Factors: {prior_risks}
Analysis: {prior_reasoning}

── LIVE MARKET DATA ──
{fundamental_summary}

── INSIDER ACTIVITY ──
{insider_summary}

── NEWS & SENTIMENT ──
{news_summary}

── COMPETITIVE POSITION ──
{competitive_summary}

── TECHNICAL CONTEXT ──
{technical_summary}

This is the DEEPEST possible analysis. Go well beyond the prior deep dive. Evaluate ALL dimensions below with maximum rigor and specificity:

1. REVISED SIGNAL & CONVICTION — does the deeper analysis confirm, strengthen, or change the prior verdict?
2. PEER COMPARISON — name 3-5 specific direct competitors; compare this stock's valuation (P/E, EV/EBITDA, P/S), growth rate, margins, and moat quality against each. Where does it rank?
3. MANAGEMENT QUALITY — track record of capital allocation (buybacks, dividends, acquisitions, debt management); insider buying/selling patterns; CEO tenure and credibility; any red flags in guidance history
4. MACRO SENSITIVITY — specific exposure to: interest rates, USD strength, commodity prices, consumer credit cycles, regulatory environment, geopolitical risk. What macro environment is ideal vs. dangerous for this stock?
5. HISTORICAL PATTERNS — how has this stock behaved in past bear markets, recessions, sector rotations? Does it tend to lead or lag the market? Seasonal patterns?
6. PROBABILITY-WEIGHTED RETURNS — given all three scenarios (bull/base/bear) with their probabilities, what is the expected return over 6 months and 12 months?
7. REVISED CATALYSTS — any new or refined catalysts beyond the prior analysis
8. REVISED RISK SCENARIOS — refine bull/base/bear with more specific price targets and drivers
9. POSITION SIZING — specific recommendation for position size % and why (account for volatility, liquidity, correlation with typical holdings)
10. FINAL COMPREHENSIVE VERDICT — a 4-6 sentence synthesis that a portfolio manager could read and act on

Respond as JSON with ALL these exact fields:
{{
    "signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>",
    "conviction_score": <1-10, one decimal place, e.g. 7.3>,
    "thesis": "<3-4 sentence revised investment thesis incorporating all findings>",
    "valuation_analysis": "<refined DCF-style valuation — more specific than prior dive>",
    "fair_value_estimate": <refined intrinsic value as a number>,
    "margin_of_safety_pct": <percentage below fair value — negative if overvalued>,
    "entry_price": <recommended entry price>,
    "stop_loss": <stop loss price>,
    "technical_analysis": "<refined technical analysis with specific price levels>",
    "competitive_moat": "<refined moat assessment with peer context>",
    "growth_outlook": "<refined growth trajectory with specific numbers where possible>",
    "risk_factors": "<comprehensive list of risks — more specific than prior dive>",
    "peer_comparison": "<name each peer, compare valuation/growth/moat, conclude where this stock ranks>",
    "management_quality": "<capital allocation track record, insider activity, CEO credibility, red flags>",
    "macro_sensitivity": "<specific macro exposures and what environment helps vs. hurts>",
    "historical_patterns": "<bear market behavior, recession performance, seasonality, market leadership>",
    "expected_return_6m": <probability-weighted expected return over 6 months as a percentage>,
    "expected_return_12m": <probability-weighted expected return over 12 months as a percentage>,
    "catalysts": [
        {{"event": "<specific catalyst>", "timeframe": "<when>", "impact": "<expected price impact>"}}
    ],
    "risk_scenarios": [
        {{"scenario": "Bull", "probability_pct": <number>, "price_target": <number>, "description": "<specific drivers>"}},
        {{"scenario": "Base", "probability_pct": <number>, "price_target": <number>, "description": "<specific drivers>"}},
        {{"scenario": "Bear", "probability_pct": <number>, "price_target": <number>, "description": "<specific drivers>"}}
    ],
    "entry_zone_low": <optimal buy zone low>,
    "entry_zone_high": <optimal buy zone high>,
    "monitoring_checklist": ["<specific, concrete metric or event to track — not generic>"],
    "enhanced_reasoning": "<4-6 sentence final verdict that synthesizes ALL dimensions — suitable for a portfolio manager to act on>"
}}

Rules:
- Be more specific than the prior dive — use actual numbers, name actual competitors, name actual macro factors
- probability_pct across scenarios must sum to 100
- expected_return_6m and expected_return_12m must be probability-weighted (multiply each scenario's return by its probability)
- If the deeper analysis changes the signal or conviction from the prior dive, explain why

Respond with ONLY the JSON object, no other text.
"""


ANALYSIS_PROMPT = """\
You are a senior equity research analyst. Analyze the following stock data and produce a structured investment recommendation.

TODAY'S DATE: {current_date} — any event, quarter, or timeframe you reference must be realistic relative to this date. Never reference a quarter, month, or year that has already passed.

IMPORTANT RULES:
- Be conservative. The cardinal rule is NEVER LOSE MONEY.
- Only recommend BUY or STRONG BUY if conviction is {min_conviction:.1f}/10 or higher.
- Every recommendation MUST include a stop loss.
- Risk/reward ratio must be at least {rr_base:.1f}:1 for a stock right at the {min_conviction:.1f}/10 conviction minimum, scaling down modestly for higher conviction — never below {rr_floor:.1f}:1 regardless of how high conviction is.
- If the data is insufficient or unclear, recommend NO ACTION.
- Every numeric field you output (entry_price, stop_loss, take_profit_targets) must be used consistently everywhere else in your response, including inside reasoning — if you cite risk/reward math or a specific target price in reasoning, it must match the actual entry_price/stop_loss/take_profit_targets values you output, and "first target" must mean take_profit_targets[0], not a later one.
- If a LONG-TERM TREND section is present below, weigh it explicitly: judge whether the current setup looks like a genuine fundamental turnaround versus a bounce back toward a level the stock has already failed at before.

STOCK: {ticker} — {company_name}
CURRENT PRICE: ${current_price:.2f}

── FUNDAMENTAL DATA ──
{fundamental_summary}

── INSIDER ACTIVITY ──
{insider_summary}

── NEWS & SENTIMENT ──
{news_summary}

── COMPETITIVE POSITION ──
{competitive_summary}

── TECHNICAL CONTEXT ──
{technical_summary}
{market_context_section}{long_term_trend_section}{trade_history_section}{user_note_section}
Based on all of the above, provide your analysis as JSON with these exact fields:
{{
    "conviction_score": <1-10, one decimal place, e.g. 7.3>,
    "signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL|NO ACTION>",
    "risk_level": "<LOW|MODERATE|HIGH>",
    "business_summary": "<1-2 factual sentences on what the company actually makes, sells, or operates — not an investment opinion, just what the business is>",
    "thesis": "<2-3 sentence investment thesis>",
    "risk_factors": "<key risks that could invalidate the thesis>",
    "recommended_action": "<specific action to take>",
    "entry_price": <recommended entry price as number>,
    {stop_tp_instructions}
    "position_size_pct": <recommended position size as % of portfolio, 1-10>,
    "time_horizon": "<days|weeks|months>",
    "reasoning": "<detailed reasoning connecting all research dimensions>",
    "fair_value_estimate": <quick intrinsic value estimate as a number>,
    "margin_of_safety_pct": <percentage below fair value the current price represents — negative if overvalued>
}}

Respond with ONLY the JSON object, no other text.
"""


def _build_user_note_section(user_note_summary: str) -> str:
    """Wraps a non-empty user-entered note in a clearly-labeled prompt section
    (2026-07-29, FNB discussion) -- a manual On Deck removal can attach a note (e.g. "this
    ticker's dip situation looks stuck, not resolving soon") that persists independent of
    the removal's own temporary/permanent block, so if that ticker is ever analyzed again
    the AI isn't starting cold. Framed as minor supplementary context, same "for reference
    only" pattern as _build_trade_history_section -- a past human judgment call shouldn't
    override fresh data, just inform it. Returns an empty string when there's no note, so
    the prompt renders byte-for-byte unchanged for any caller that doesn't pass one."""
    if not user_note_summary:
        return ""
    return (
        "\n── USER NOTE (for reference only — not a hard rule, weigh fresh data first) ──\n"
        f"{user_note_summary}\n"
    )


def _build_long_term_trend_section(long_term_trend_summary: str) -> str:
    """Wraps a non-empty multi-year trend summary in a clearly-labeled prompt section
    (2026-07-29, BEN discussion) -- see LongTermTrend/format_long_term_trend_summary in
    src/data/market_data.py for how the summary itself is built. Returns an empty string
    when there's nothing to show (an empty/failed fetch), so the prompt renders
    byte-for-byte unchanged for any caller that doesn't pass one."""
    if not long_term_trend_summary:
        return ""
    return f"\n── LONG-TERM TREND ──\n{long_term_trend_summary}\n"


def _build_market_context_section(market_change_pct: float | None) -> str:
    """Today's broad-market direction (S&P 500 proxy), with an explicit owner-directed
    interpretive lean (2026-08-04, "shouldn't the AI know if the market's up or down"
    discussion) -- every prompt in this system previously judged each stock in complete
    isolation from what the broader market was doing that day. Returns "" when the fetch
    failed (None), same graceful-omission pattern every other optional section here uses
    -- never fabricates a market condition.

    Owner's explicit direction, not left to open-ended AI judgment alone: a broadly up
    market should generally be a favorable day to take new positions (don't sit in cash
    waiting when the fundamentals/technicals already support a buy); a broadly down
    market calls for more caution and selectivity. Framed as a real, weighable factor
    alongside everything else in the prompt, not a mechanical override -- consistent with
    this project's standing preference for giving real data + framing and trusting
    Claude's judgment rather than hardcoding a rule."""
    if market_change_pct is None:
        return ""
    direction = "up" if market_change_pct >= 0 else "down"
    return (
        f"\n── TODAY'S MARKET ──\n"
        f"S&P 500 (SPY): {market_change_pct:+.2f}% today — the broad market is {direction} "
        f"today. Weigh this as a real factor: a stock's own price strength on a day the "
        f"broader market is also strongly up may partly reflect market-wide momentum "
        f"(beta) rather than something unique to this company -- but a broadly up market "
        f"is generally a favorable environment for taking new positions, so don't be "
        f"reflexively conservative if the fundamentals and technicals already support a "
        f"buy. On a broadly down market, be more selective and cautious, since risk is "
        f"elevated market-wide.\n"
    )


def _market_cap_tier_label(market_cap: float) -> str:
    """Human-readable size tier for a market cap in dollars (2026-08-04, "billion dollar
    stock" discussion) -- feeds recommend_dip_entry's staleness judgment, which previously
    had zero information about company size and applied the same generic "is this many
    days stale" instinct to a small volatile stock and a stable mega-cap alike. A large,
    steady company's support/resistance levels reasonably persist longer than a small
    volatile one's -- this gives Claude the context to calibrate for that instead of
    guessing. Same bucket boundaries CompetitorAnalyzer._assess_position already uses
    (src/research/competitor.py) for consistency across the codebase. Returns "" for a
    non-positive/unknown market cap so the caller can omit the context line entirely
    rather than asserting a tier it has no real data for."""
    if market_cap <= 0:
        return ""
    if market_cap >= 200_000_000_000:
        return "mega-cap"
    if market_cap >= 10_000_000_000:
        return "large-cap"
    if market_cap >= 2_000_000_000:
        return "mid-cap"
    return "small-cap"


def _build_trade_history_section(trade_history_summary: str) -> str:
    """Wraps a non-empty trade history summary in a clearly-labeled prompt section, framed
    as minor supplementary context so Claude weighs current fundamentals/technicals as the
    primary basis for its decision (2026-07-23, user's explicit calibration: "i dont want
    it a major contributing factor" -- see docs/superpowers/specs/
    2026-07-23-trade-history-context-design.md). Returns an empty string when there's no
    summary to show, so ANALYSIS_PROMPT renders byte-for-byte unchanged for any caller that
    doesn't pass one (in particular, submit_analysis_batch's universe-scan/persist-check
    path never passes one -- see the spec for why that's deliberately excluded)."""
    if not trade_history_summary:
        return ""
    return (
        "\n── PRIOR TRADING HISTORY (for reference only — weigh current fundamentals "
        "and technicals as the primary basis for your decision) ──\n"
        f"{trade_history_summary}\n"
    )


class ResearchEngine:
    def __init__(
        self,
        config: dict,
        market_data: MarketDataFetcher,
        insider_tracker: InsiderTracker,
        news_feed: NewsFeed,
    ):
        self.config = config
        self.market_data = market_data
        self.insider_tracker = insider_tracker
        self.news_feed = news_feed
        self.reports: dict[str, ResearchReport] = {}
        self.deep_dives: dict[str, DeepDiveReport] = {}

        self.fundamental_analyzer = FundamentalAnalyzer(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.insider_analyzer = InsiderAnalyzer(config)
        self.competitor_analyzer = CompetitorAnalyzer(config)

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

        self.reports_dir = Path("data/research_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        research_cfg = config.get("research", {})
        self._market_tz = ZoneInfo(research_cfg.get("market_timezone", "America/New_York"))

    def _now_et(self) -> datetime:
        """ET-aware now for report generated_at timestamps — mirrors web/app.py's
        DashboardState._now_et(). Added 2026-07-18: every generated_at here previously used
        naive datetime.now() (server wall-clock — UTC on Hetzner), which the Deep Dive modal
        displays directly as "Report generated: <timestamp>" with no timezone conversion,
        showing 4 hours ahead of real ET (same root cause as the TZ-1 AI-log fix)."""
        return datetime.now(self._market_tz)

    async def analyze_stock(
        self, ticker: str, trade_history_summary: str = "",
        user_note_summary: str = "", model: str | None = None,
    ) -> ResearchReport:
        logger.info("Starting research analysis for %s", ticker)

        quote = await self.market_data.get_quote(ticker)
        financials = await self.market_data.get_financials(ticker)
        technicals = await self.market_data.get_technicals(ticker)
        long_term_trend = await self.market_data.get_long_term_trend(
            ticker, quote.price,
            years=self.config.get("research", {}).get("long_term_trend_years", 5))
        insider_summary = await self.insider_tracker.get_insider_summary(ticker)
        news_items = await self.news_feed.get_company_news(ticker, days=7)

        fundamental_score = await self.fundamental_analyzer.analyze(financials)
        sentiment_analysis = await self.sentiment_analyzer.analyze(news_items)
        insider_analysis = await self.insider_analyzer.analyze(insider_summary)
        competitive_analysis = await self.competitor_analyzer.analyze(ticker)

        company_name = ""
        try:
            import yfinance as yf
            company_name = await asyncio.to_thread(
                lambda: yf.Ticker(ticker).info.get("shortName", ticker)
            )
        except Exception:
            company_name = ticker

        technical_summary = (
            f"Price: ${quote.price:.2f} | SMA50: ${technicals.sma_50:.2f} | "
            f"SMA200: ${technicals.sma_200:.2f} | RSI: {technicals.rsi:.1f} | "
            f"Support: ${technicals.support_level:.2f} | "
            f"Resistance: ${technicals.resistance_level:.2f} | "
            f"Avg Volume 30d: {technicals.avg_volume_30d:,}"
        )
        long_term_trend_summary = format_long_term_trend_summary(long_term_trend)

        if self.client:
            report = await self._claude_analysis(
                ticker, company_name, quote.price,
                fundamental_score.summary,
                insider_analysis.summary,
                sentiment_analysis.summary,
                competitive_analysis.summary,
                technical_summary,
                trade_history_summary,
                long_term_trend_summary,
                user_note_summary,
                model,
            )
        else:
            logger.warning("No ANTHROPIC_API_KEY — generating rule-based report for %s", ticker)
            report = self._rule_based_analysis(
                ticker, company_name, quote.price,
                fundamental_score, insider_analysis, sentiment_analysis,
                competitive_analysis, technicals,
            )

        report.sector = getattr(financials, 'sector', '')
        self.reports[ticker] = report
        await asyncio.to_thread(self._save_report, report)
        logger.info("Research complete for %s — Signal: %s, Conviction: %d/10",
                     ticker, report.signal.value, report.conviction_score)
        return report

    async def _call_claude_with_retry(
        self, model: str, max_tokens: int, messages: list,
        max_retries: int = 3, base_delay: float = 2.0,
    ) -> str:
        """Call Claude API with exponential backoff retry. Returns the text response.

        Some models reject the `temperature` parameter outright with a 400 "temperature is
        deprecated for this model" error rather than silently ignoring it — discovered
        2026-07-19 when claude-sonnet-5 (the deep-dive tier) hit this on every single call,
        the first time this session actually exercised the deep-dive model path since the
        2026-07-15 fix that added temperature=0.0 here for output-determinism (that fix was
        only verified against the quick-scan tier's Haiku model at the time; nobody had run
        a Deep Dive since, so it went undetected for 4 days). Handled generically rather than
        a hardcoded model-name check, since which models accept the parameter can change
        independently of this codebase: on that specific error, retries once immediately
        with temperature omitted entirely, without consuming one of the real retry attempts
        meant for genuine transient failures."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("Claude API retry %d/%d in %.0fs...", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
            try:
                try:
                    message = await asyncio.to_thread(
                        lambda: self.client.messages.create(
                            model=model, max_tokens=max_tokens, messages=messages,
                            temperature=0.0,
                        )
                    )
                except Exception as temp_err:
                    msg = str(temp_err).lower()
                    if "temperature" in msg and "deprecat" in msg:
                        logger.info(
                            "Model %s rejects temperature param — retrying without it", model)
                        message = await asyncio.to_thread(
                            lambda: self.client.messages.create(
                                model=model, max_tokens=max_tokens, messages=messages,
                            )
                        )
                    else:
                        raise
                text_block = next((b for b in message.content if hasattr(b, "text")), None)
                if not text_block:
                    raise ValueError("No text block in Claude response")
                return text_block.text
            except Exception as e:
                last_error = e
                logger.warning("Claude API attempt %d/%d failed: %s", attempt + 1, max_retries, e)
        raise last_error  # type: ignore[misc]

    async def recommend_dip_entry(
        self, ticker: str, company_name: str, thesis: str, fair_value_estimate: float,
        peak: float, low: float, current_price: float,
        peak_days_ago: float | None = None, low_days_ago: float | None = None,
        market_cap: float | None = None,
    ) -> tuple[float | None, str, bool] | None:
        """Recommends a specific entry price given a REAL, already-observed dip that has
        started recovering — added 2026-07-18 as the "ai" alternative to the mechanical
        peak-to-low retracement-% entry mode. Deliberately NOT called at routine pre-open
        analysis time: per the user's own reasoning, Claude can't meaningfully predict a
        pullback entry price before the dip has actually happened — a specific price
        recommendation is only informed once there's a real peak, a real low, and a real
        recovery already underway to reason about. near_miss_monitor_loop is the only
        caller, firing this once when it detects that shape (see its docstring for the
        exact trigger condition), not on every 60s tick.

        peak_days_ago/low_days_ago (2026-07-28, RRC/OVV incident) — how long ago the
        peak/low actually happened, in days. Before this, the prompt gave Claude only three
        bare prices with no sense of time at all, so it had no way to notice when a "dip"
        was actually just wherever a multi-week trend happened to start (the mechanical
        peak-to-low search over a long window finds the single lowest point regardless of
        how old it is — see dip_summary's docstring). Both RRC and OVV were bought this way:
        their "recent low" was several weeks stale, the stock had been trending steadily the
        whole time with no real recent pullback, and the system bought into (in OVV's case)
        an active decline because the mechanical "price is above the old low" check was
        already trivially true. Giving Claude the real elapsed time lets it judge staleness
        itself rather than trusting a hardcoded day cutoff — see the prompt's explicit
        instruction to decline when the pattern doesn't hold up. Optional (defaults to None
        -> omitted from the prompt) only so an old caller/test that hasn't been updated yet
        doesn't outright break; every real caller always supplies both.

        Returns None on any failure (no API key, malformed response, API error) — never
        fabricates a number, same AI Data Integrity principle as every other real trading
        figure in this codebase. Returns (None, reason, stale) when Claude explicitly
        judges this isn't a genuine, current uptrend worth entering — a real, reasoned
        "no", distinct from an outright failure, and the caller must NOT treat it as an
        error to retry blindly (see _compute_ai_dip_entry). `stale` (2026-07-30, On Deck
        auto-removal feature) is Claude's own judgment of whether this decline will
        never resolve itself (a genuinely stale reference point the stock has moved on
        from — worth evicting from On Deck so a fresh candidate can take the slot) versus
        one that's still legitimately developing (too early/fresh — worth continuing to
        monitor as-is). Always False when entry_price is a real number (irrelevant in
        that case). Uses its own model_dip_entry setting (2026-07-24, split off from the
        shared model_quick_scan default so this can be tuned independently of the live
        buy pipeline — see CLAUDE.md's "Model Selection" section).

        market_cap (2026-08-04, "billion dollar stock" discussion) — optional dollar
        market cap, converted to a human-readable size tier via _market_cap_tier_label.
        Before this, the staleness judgment above had zero information about company
        size and applied the same generic "is N days old too stale" instinct to a small
        volatile stock and a stable mega-cap alike, even though a larger company's
        support/resistance levels reasonably persist longer. This doesn't loosen or
        remove the staleness judgment itself (the user explicitly values the real,
        reasoned "AI declined this entry" message this function produces) — it gives
        Claude better information to calibrate that same judgment by. Omitted from the
        prompt entirely when None/non-positive (unknown), same graceful-omission pattern
        peak_days_ago/low_days_ago already use.

        Also fetches today's broad-market context (S&P 500 % change) via
        _build_market_context_section (2026-08-04, "shouldn't the AI know if the
        market's up or down" discussion) -- same shared section analyze_stock's prompt
        now includes."""
        if not self.client:
            return None
        market_change_pct = await self.market_data.get_market_change_pct()
        market_context_section = _build_market_context_section(market_change_pct)
        cap_tier = _market_cap_tier_label(market_cap) if market_cap else ""
        cap_line = f"\nCompany size: {cap_tier} (~${market_cap/1e9:.1f}B market cap)" if cap_tier else ""
        peak_age_line = (f"Recent peak (before the dip): ${peak:.2f}, {peak_days_ago:.1f} day(s) ago"
                          if peak_days_ago is not None else f"Recent peak (before the dip): ${peak:.2f}")
        low_age_line = (f"Recent low (the bottom of the dip): ${low:.2f}, {low_days_ago:.1f} day(s) ago"
                         if low_days_ago is not None else f"Recent low (the bottom of the dip): ${low:.2f}")
        prompt = f"""\
A stock you previously analyzed has dipped and is now showing real signs of recovery —
this is an observed price pattern, not a prediction. Given this actual price action,
judge whether this is a genuine, CURRENT uptrend worth entering, and if so, recommend a
specific entry price.

STOCK: {ticker} — {company_name}{cap_line}
Your prior thesis: {thesis}
Fair value estimate: ${fair_value_estimate:.2f}
{peak_age_line}
{low_age_line}
Current price (recovering off the low): ${current_price:.2f}
{market_context_section}
Use the elapsed time above as real information, not decoration: a low from many days or
weeks ago that the stock simply hasn't revisited since (because it's been trending steadily
in one direction the whole time) is NOT a current dip-and-recovery worth acting on, even
though the current price happens to sit above that old low. A genuine entry signal needs a
low that's recent enough to still be a meaningful, active reference point for where this
stock trades right now. If the pattern doesn't hold up under that scrutiny, say so and
decline — do not force a price recommendation onto a stale or unconvincing setup.

Weigh company size when judging what "recent enough" means: a large, stable company's
support levels typically persist and stay relevant far longer than a small, volatile
stock's — a low from several weeks ago can still be a genuine, active reference point for
a steady mega-cap, where the identical elapsed time might already be stale for a small-cap
that moves fast. Don't apply one generic staleness window to every stock regardless of size.

If you judge this a genuine current uptrend: considering technical support levels, how much
it has already recovered, and the fair value estimate, what price would represent a good
entry point? It should be a real, reasoned price level — not simply an arbitrary percentage
off the low.

Respond with ONLY a JSON object:
- If this is a genuine entry opportunity: {{"entry_price": <number>, "reasoning": "<one sentence>"}}
- If you decline because the reference low itself is the problem — genuinely stale,
  already fully priced in, or otherwise unlikely to ever become a meaningful CURRENT
  setup again (the stock has simply moved on from it): {{"entry_price": null, "reasoning": "<one sentence>", "stale": true}}
- If you decline for any other reason — e.g. the recovery is real but still too early/
  fresh to act on yet, and this same low may still develop into a genuine entry soon:
  {{"entry_price": null, "reasoning": "<one sentence>", "stale": false}}
"""
        model = self.config.get("research", {}).get("model_dip_entry", "claude-haiku-4-5")
        try:
            response_text = await self._call_claude_with_retry(
                model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            text = text.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
            data = json.loads(text)
            reasoning = str(data.get("reasoning", ""))
            raw_entry = data.get("entry_price")
            if raw_entry is None:
                # stale (2026-07-30, On Deck auto-removal feature) -- distinguishes a
                # decline that will never resolve itself (a genuinely stale reference
                # point, worth evicting so a fresh On Shore candidate can take the slot)
                # from one that's still legitimately developing (too early/fresh, worth
                # continuing to monitor). Defaults False on a malformed/missing key so an
                # ambiguous response never triggers an eviction it didn't clearly ask for.
                stale = _parse_json_bool(data.get("stale"), default=False)
                return None, reasoning, stale
            entry_price = float(raw_entry)
            if entry_price <= 0:
                return None
            return entry_price, reasoning, False
        except Exception as e:
            logger.warning("%s: dip-entry recommendation failed: %s", ticker, e)
            return None

    async def recommend_on_deck_retention(
        self, ticker: str, company_name: str, thesis: str, price: float,
        fair_value_estimate: float, stop_loss: float, rr: float, required_rr: float,
        conviction_score: float,
    ) -> tuple[bool, str] | None:
        """Judges whether an On Deck candidate whose R/R has risen above its own real
        gate is still a genuine buying opportunity, replacing a fixed grace-period timer
        (2026-08-05, owner design) -- owner: "if the r/r moves up to where the stock is
        not a good buy.. then remove it... whenever it isnt a good buy any more.. then
        evict it," but explicitly did not want a hardcoded numeric ceiling for what
        counts as "not a good buy" ("i dont know when it would not be a good buy..
        maybe something for ai to choose") -- same "trust AI judgment over mechanics"
        principle already used for the dip-entry staleness call above, rather than
        adding sample-size/statistical scaffolding.

        R/R clearing a candidate's own gate is ambiguous on its own: it can mean the
        stock has genuinely become undervalued, or it can simply mean price has kept
        falling toward the stop-loss level, which mechanically inflates the ratio
        (bigger numerator, smaller denominator) without any real improvement in the
        setup. This function gives Claude the real numbers and asks it to judge which
        one is actually happening, the same way recommend_dip_entry judges staleness --
        real reasoning text the owner can read, not a threshold.

        Deliberately a separate, targeted call rather than added to the shared batch
        prompt _run_on_deck_persist_check already uses for every candidate -- that batch
        infrastructure builds many prompts at once and is deliberately not customized
        per-ticker (same scoping precedent as the trade-history/user-note context
        sections, which also skip the batch path). Only fires for the subset of
        candidates actually found above their own gate, not the whole On Deck list.

        Returns None on any failure (no API key, malformed response, API error) --
        the caller must treat this as "keep, don't evict" (same AI Data Integrity
        principle as every other real trading figure in this codebase: never remove a
        candidate on missing/failed AI data). Uses model_dip_entry (shared with the
        thematically similar dip-entry judgment above, not a new dedicated dial)."""
        if not self.client:
            return None
        prompt = f"""\
STOCK: {ticker} — {company_name}
Original thesis: {thesis}
Current price: ${price:.2f}
Fair value estimate: ${fair_value_estimate:.2f}
Stop-loss level: ${stop_loss:.2f}
Computed reward/risk ratio: {rr:.2f} (this stock's own required minimum is
{required_rr:.2f}, conviction {conviction_score}/10)

This candidate's reward/risk ratio has risen above its own required minimum. That can
mean the stock has genuinely become a better value, but it can also simply mean price
has kept falling toward the stop-loss level, which mechanically inflates this ratio
(a bigger gap to fair value, a smaller gap to the stop) without any real improvement
in the setup.

Judge whether this is still a genuine, worthwhile buying opportunity right now, or
whether the elevated ratio reflects an overextended or deteriorating situation that's
no longer a good buy. Weigh whether the original thesis still holds and whether the
price action looks like continued decline versus a real recovery.

Respond with ONLY a JSON object:
{{"still_good_buy": true/false, "reasoning": "<a detailed 2-4 sentence explanation --
reference the real numbers above (price, fair value, stop, R/R) and the thesis
specifically, the same way you would explain a real investing decision to someone
reading it, not a generic one-liner>"}}
"""
        model = self.config.get("research", {}).get("model_dip_entry", "claude-haiku-4-5")
        try:
            response_text = await self._call_claude_with_retry(
                model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            text = text.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
            data = json.loads(text)
            reasoning = str(data.get("reasoning", ""))
            still_good_buy = _parse_json_bool(data.get("still_good_buy"), default=True)
            return still_good_buy, reasoning
        except Exception as e:
            logger.warning("%s: on-deck retention judgment failed: %s", ticker, e)
            return None

    async def recommend_portfolio_health(
        self, portfolio_summary: dict, positions: list[dict],
    ) -> dict | None:
        """One Claude call synthesizing the whole portfolio's mechanical signals plus
        every held position's own thesis/fair-value/conviction into a detailed health
        assessment and a probability-weighted annualized return estimate — the
        portfolio-level counterpart to the existing per-stock 6m/12m expected-return
        fields already used in Deeper Dive. Explicitly framed as a judgment call, not a
        guarantee. Returns None on any failure — never fabricates a number, same AI Data
        Integrity principle as every other real trading figure in this codebase. Uses its
        own model_portfolio_health setting (2026-07-24, split off from the shared
        model_deep_dive default -- see CLAUDE.md's "Model Selection" section), since this
        synthesizes many positions at once, unlike the cheap single-stock calls used
        elsewhere."""
        if not self.client:
            return None

        positions_text = "\n".join(
            f"- {p['ticker']}: conviction {p['conviction']}/10, thesis: {p['thesis']}, "
            f"fair value ${p['fair_value_estimate']:.2f}, margin of safety "
            f"{p['margin_of_safety_pct']:.1f}%, unrealized P&L {p['unrealized_pnl_pct']:+.1f}%, "
            f"held {p['days_held']} day(s)"
            for p in positions
        ) or "No open positions currently held."

        sector_text = ", ".join(
            f"{sector}: {count}" for sector, count in portfolio_summary["sector_counts"].items()
        ) or "No sector data available."

        prompt = f"""Assess the overall health of this stock trading portfolio and
predict its likely annualized return. Be direct and specific — flag real risks and
real strengths, don't hedge everything into meaninglessness.

IMPORTANT CONTEXT: This automated trading system has been under active development and
has undergone major rewrites to its buy/sell decision logic recently — most significantly
a complete replacement of the buy mechanism on 2026-07-17, plus further refinements to the
risk/reward gate and conviction thresholds through 2026-07-20. Trades before 2026-07-17
were decided by a fundamentally different, now-replaced system. Two win-rate figures are
given below for exactly this reason — weight the "current architecture" figure much more
heavily than the "all trades since go-live" figure when assessing likely forward
performance, since the older trades are not representative of how the system decides
trades today.

This is also a 100% automated trading system with no discretionary human buying — every
purchase happens the moment a candidate clears its own conviction/R/R/uptick bars via
continuous monitoring, not on a schedule and not held back for judgment calls. Don't treat
the cash % the way you would for a manual trader holding dry powder for optionality or
emotional comfort — that framing doesn't apply here, since the system deploys capital
automatically the instant a real opportunity qualifies. A large cash balance more likely
reflects either genuine risk discipline (position-sizing limits, cash reserve floor) or
that too few candidates are currently clearing the bar — comment on which one it looks
like here rather than defaulting to "healthy dry powder."

Conviction score is one of the system's real, hard buy-gate criteria, not just a
descriptive rating — a stock is only ever bought if its conviction clears
{portfolio_summary['min_conviction_score']:.1f}/10 (alongside separate R/R and uptick-
confirmation checks). Because of this, held positions clustering at or just above that
threshold is the gate working exactly as designed and a form of survivorship — anything
that scored lower was filtered out before ever becoming a position, so you're only seeing
the stocks that passed. Do NOT read tight conviction clustering among held positions as a
sign the underlying scoring model fails to differentiate opportunities; that would need
evidence from the full universe of scored-but-not-bought candidates, which isn't in front
of you here.

PORTFOLIO SNAPSHOT:
Total value: ${portfolio_summary['total_value']:,.2f}
Cash: {portfolio_summary['cash_pct']:.1f}% of portfolio
Day P&L: {portfolio_summary['day_pnl_pct']:+.2f}%
Total P&L since inception: {portfolio_summary['total_pnl_pct']:+.2f}%
Drawdown state: {portfolio_summary['drawdown_state']}
Win rate — all trades since go-live (2026-07-12): {portfolio_summary['win_rate_all_time_pct']:.1f}% ({portfolio_summary['closed_all_time']} closed trades, includes the old, now-replaced buy system)
Win rate — current architecture only (since 2026-07-17 rewrite): {portfolio_summary['win_rate_current_arch_pct']:.1f}% ({portfolio_summary['closed_current_arch']} closed trades) — this is the more meaningful figure
Average conviction of held positions: {portfolio_summary['avg_conviction']:.1f}/10
Sector concentration: {sector_text}

HELD POSITIONS:
{positions_text}

Respond with ONLY a JSON object in this exact shape:
{{"health_summary": "<detailed multi-paragraph assessment covering diversification, \
conviction quality, risk concentration, and anything that stands out — several sentences, \
not a one-liner>", "predicted_annual_return_pct": <signed number>, \
"prediction_reasoning": "<2-3 sentences explaining the prediction>"}}
"""
        model = self.config.get("research", {}).get("model_portfolio_health", "claude-haiku-4-5")
        # _call_claude_with_retry's own retry loop only covers the API call itself -- once
        # it returns text, a JSON-parse failure is outside that loop and would otherwise be
        # a permanent failure with no second attempt. This function asks for much longer
        # free-form prose (a "detailed multi-paragraph" health_summary) than any other
        # JSON-based Claude call in this codebase, making an unescaped quote/apostrophe
        # breaking strict JSON meaningfully more likely here -- confirmed live (2026-07-20):
        # "Expecting property name enclosed in double quotes" on the very first real call.
        # Added a small parse-specific retry (2 attempts) so one malformed response doesn't
        # waste the whole real Sonnet call with no recourse.
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response_text = await self._call_claude_with_retry(
                    model=model, max_tokens=16000, messages=[{"role": "user", "content": prompt}],
                    max_retries=2,
                )
                text = response_text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                text = text.rstrip()
                if text.endswith("```"):
                    text = text[:-3].rstrip()
                data = json.loads(text)
                return {
                    "health_summary": str(data["health_summary"]),
                    "predicted_annual_return_pct": float(data["predicted_annual_return_pct"]),
                    "prediction_reasoning": str(data.get("prediction_reasoning", "")),
                }
            except Exception as e:
                last_error = e
                logger.warning("Portfolio health assessment attempt %d/2 failed: %s", attempt + 1, e)
        logger.warning("Portfolio health assessment failed after 2 attempts: %s", last_error)
        return None

    def _build_quick_scan_prompt(
        self, ticker: str, company_name: str, current_price: float,
        fundamental_summary: str, insider_summary: str,
        news_summary: str, competitive_summary: str, technical_summary: str,
        trade_history_summary: str = "",
        long_term_trend_summary: str = "",
        user_note_summary: str = "",
        market_change_pct: float | None = None,
    ) -> str:
        """Shared by the sequential path (_claude_analysis) and the batch path
        (submit_analysis_batch) — both build the identical prompt from ANALYSIS_PROMPT,
        so there is no way for batch and sequential criteria to drift apart.
        trade_history_summary defaults to "" -- submit_analysis_batch's call site never
        passes one (2026-07-23, deliberately excluded, see docs/superpowers/specs/
        2026-07-23-trade-history-context-design.md), so the batch path's prompt is
        byte-for-byte unchanged from before this feature existed.

        market_change_pct (2026-08-04) -- today's S&P 500 % change, see
        _build_market_context_section's docstring for the full "should the AI know if
        the market's up or down" discussion. None (default) omits the section entirely,
        same as every other optional context piece here.

        stop_tp_instructions (2026-07-31, AI-chosen stop-loss/TP feature) switches
        between the original fixed-percentage language and a real-technicals-judgment
        variant based on research.ai_chosen_stop_tp_enabled -- see
        docs/superpowers/specs/2026-07-31-ai-chosen-stop-loss-tp-design.md. Defaults to
        the original mechanical language when the "research" config key is absent
        entirely (matches this method's own existing tests, several of which construct
        a bare engine with no "research" key at all)."""
        tp_cfg = self.config.get("take_profit", {})
        research_cfg = self.config.get("research", {})
        # Follows the live min_conviction_score setting (2026-08-03, owner request) rather
        # than a hardcoded "7/10" -- previously this prompt instruction never moved even
        # when the owner lowered the real gate, so Claude kept reserving BUY/STRONG BUY for
        # ~7+ regardless of what min_conviction_score was actually configured to, silently
        # making the setting change a no-op for anything below ~6.8 (confirmed live:
        # lowering the gate to 6.2 produced zero new BUY signals in that range until this
        # fix, since Claude was still being told 7/10 the whole time).
        min_conviction = research_cfg.get("min_conviction_score", 7)
        # Follows the live conviction-scaled R/R gate (2026-08-09, GitHub #63) --
        # ANALYSIS_PROMPT used to tell Claude a hardcoded, never-interpolated "at least
        # 3:1" regardless of the real gate, which is _required_rr's conviction-scaled
        # formula (base_rr - extra_conviction*step, floored at rr_floor), not a flat
        # 3:1 -- the exact same drift-out-of-sync-with-the-real-gate failure mode
        # already fixed above for the conviction number itself. _required_rr's own
        # per-conviction-point formula lives in web/app.py (the live pipeline), not
        # importable here without a circular dependency, so this states the real
        # base/floor bounds and the scaling relationship qualitatively rather than
        # duplicating the exact formula (which would just create a second place for
        # the two to drift apart again).
        rr_base = research_cfg.get("min_risk_reward_ratio", 2.1)
        rr_floor = research_cfg.get("on_deck_rr_floor", 1.5)
        if research_cfg.get("ai_chosen_stop_tp_enabled", False):
            stop_tp_instructions = (
                '"stop_loss": <stop loss price as number — judge a specific level from '
                'the real Support/Resistance/SMA/RSI data above; place it just below a '
                'genuine support level or recent swing low, not an arbitrary percentage '
                'off entry>,\n'
                '    "take_profit_targets": [<T1, T2, T3 — three ascending price levels '
                'judged from real resistance levels and the fair value estimate, not a '
                'fixed percentage above entry>],'
            )
        else:
            sl_pct = tp_cfg.get("stop_loss_pct", 7.0)
            t1_pct = tp_cfg.get("t1_pct", 5.0)
            t2_pct = tp_cfg.get("t2_pct", 10.0)
            t3_pct = tp_cfg.get("t3_pct", 17.0)
            stop_tp_instructions = (
                f'"stop_loss": <stop loss price as number — target {sl_pct:.0f}% below entry>,\n'
                f'    "take_profit_targets": [<T1 — ~{t1_pct:.0f}% above entry>, '
                f'<T2 — ~{t2_pct:.0f}% above entry>, <T3 — ~{t3_pct:.0f}% above entry>],'
            )
        return ANALYSIS_PROMPT.format(
            current_date=self._now_et().strftime("%Y-%m-%d"),
            min_conviction=min_conviction,
            rr_base=rr_base,
            rr_floor=rr_floor,
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            fundamental_summary=fundamental_summary,
            insider_summary=insider_summary,
            news_summary=news_summary,
            competitive_summary=competitive_summary,
            technical_summary=technical_summary,
            trade_history_section=_build_trade_history_section(trade_history_summary),
            long_term_trend_section=_build_long_term_trend_section(long_term_trend_summary),
            user_note_section=_build_user_note_section(user_note_summary),
            market_context_section=_build_market_context_section(market_change_pct),
            stop_tp_instructions=stop_tp_instructions,
        )

    def _parse_quick_scan_response(
        self, ticker: str, company_name: str, current_price: float,
        fundamental_summary: str, insider_summary: str,
        news_summary: str, competitive_summary: str, response_text: str,
    ) -> ResearchReport:
        """Shared by the sequential path and the batch path — parses a raw Claude text
        response (from either call type) into an identical ResearchReport. Raises on
        malformed input; callers are responsible for the is_fallback=True branch.

        Reads self.config.get("research", {}) directly (2026-07-31, AI-chosen
        stop-loss/TP feature) rather than taking a separate parameter -- this method is
        already an instance method with self.config available, unlike the standalone
        pure functions this feature also adds."""
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rstrip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].rstrip()

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = json.loads(response_text.replace('\r\n', ' ').replace('\n', ' '))

        entry_price = float(data["entry_price"]) if data.get("entry_price") is not None else current_price
        raw_stop_loss = float(data["stop_loss"]) if data.get("stop_loss") is not None else current_price * 0.95
        research_cfg = self.config.get("research", {})
        if research_cfg.get("ai_chosen_stop_tp_enabled", False):
            stop_loss = _clamp_ai_stop_loss(
                raw_stop_loss, entry_price,
                research_cfg.get("ai_stop_loss_min_pct", 2.0),
                research_cfg.get("ai_stop_loss_max_pct", 10.0),
            )
        else:
            stop_loss = raw_stop_loss

        # NaN guard (fixed 2026-08-09, GitHub #65) -- json.loads accepts bare NaN/Infinity
        # tokens by Python's own non-standard extension (confirmed: json.loads('{"x": NaN}')
        # succeeds with no exception), so a malformed Claude response containing
        # "conviction_score": NaN would otherwise parse silently. This matters because
        # every downstream conviction gate rejects via `score < min_conviction`, and in
        # Python `NaN < anything` is always False -- so the "too low, reject" branch would
        # never fire, letting a NaN-conviction report reach a real executed buy with a
        # permanently corrupted conviction value. Raising here routes it through the exact
        # same is_fallback=True/HOLD-capped path every other malformed response already
        # takes (this function's own docstring: "Raises on malformed input; callers are
        # responsible for the is_fallback=True branch"), rather than inventing a new one.
        conviction_score = round(float(data.get("conviction_score", 0)), 1)
        if math.isnan(conviction_score):
            raise ValueError(f"conviction_score parsed as NaN for {ticker} — malformed Claude response")

        return ResearchReport(
            ticker=ticker,
            company_name=company_name,
            generated_at=self._now_et(),
            conviction_score=conviction_score,
            signal=Signal(data.get("signal", "NO ACTION").strip().upper()),
            # .strip().upper() (fixed 2026-08-09, GitHub #66) -- matches the normalization
            # Signal already gets a line above. Without it, a response like "risk_level":
            # "Moderate" or "low" raises ValueError, which IS caught by both call sites'
            # broad except -- but that means a legitimate, well-reasoned BUY/STRONG BUY
            # with valid entry/stop/targets gets silently discarded as is_fallback=True
            # purely because of an unrelated field's casing, wasting the paid Claude call.
            risk_level=RiskLevel(data.get("risk_level", "HIGH").strip().upper()),
            thesis=data.get("thesis", ""),
            business_summary=data.get("business_summary", ""),
            fundamental_summary=fundamental_summary,
            insider_summary=insider_summary,
            news_summary=news_summary,
            competitive_summary=competitive_summary,
            risk_factors=data.get("risk_factors", ""),
            recommended_action=data.get("recommended_action", ""),
            entry_price=entry_price,
            position_size_pct=float(data["position_size_pct"]) if data.get("position_size_pct") is not None else 3,
            stop_loss=stop_loss,
            take_profit_targets=[float(t) for t in (data.get("take_profit_targets") or []) if t is not None and float(t) > 0],
            time_horizon=data.get("time_horizon", ""),
            reasoning=data.get("reasoning", ""),
            fair_value_estimate=float(data["fair_value_estimate"]) if data.get("fair_value_estimate") is not None else 0.0,
            margin_of_safety_pct=float(data["margin_of_safety_pct"]) if data.get("margin_of_safety_pct") is not None else 0.0,
        )

    def _fallback_report(self, ticker: str, company_name: str, current_price: float,
                          fundamental_summary: str, insider_summary: str,
                          news_summary: str, competitive_summary: str, error: Exception) -> ResearchReport:
        """Shared fallback construction — identical is_fallback=True report regardless
        of whether the failure happened in the sequential or batch path."""
        logger.error("Claude analysis failed for %s: %s", ticker, error)
        return ResearchReport(
            ticker=ticker,
            company_name=company_name,
            generated_at=self._now_et(),
            conviction_score=0.0,
            signal=Signal.NO_ACTION,
            risk_level=RiskLevel.HIGH,
            thesis=f"⚠ Claude API error — analysis could not be completed: {error}",
            fundamental_summary=fundamental_summary,
            insider_summary=insider_summary,
            news_summary=news_summary,
            competitive_summary=competitive_summary,
            risk_factors="AI analysis failed — do not trade on this stock until re-scanned successfully.",
            recommended_action="NO ACTION — AI error",
            entry_price=current_price,
            position_size_pct=0,
            stop_loss=current_price * 0.95,
            is_fallback=True,
        )

    async def _claude_analysis(
        self, ticker: str, company_name: str, current_price: float,
        fundamental_summary: str, insider_summary: str,
        news_summary: str, competitive_summary: str, technical_summary: str,
        trade_history_summary: str = "",
        long_term_trend_summary: str = "",
        user_note_summary: str = "",
        model: str | None = None,
    ) -> ResearchReport:
        market_change_pct = await self.market_data.get_market_change_pct()
        prompt = self._build_quick_scan_prompt(
            ticker, company_name, current_price, fundamental_summary,
            insider_summary, news_summary, competitive_summary, technical_summary,
            trade_history_summary, long_term_trend_summary, user_note_summary,
            market_change_pct,
        )
        try:
            response_text = await self._call_claude_with_retry(
                # 2026-07-24: model_position_monitor/model_dip_entry now split off from
                # this shared model_quick_scan default -- see CLAUDE.md's "Model Selection"
                # section for the full independent-settings list.
                model=model or self.config.get("research", {}).get("model_quick_scan", "claude-haiku-4-5"),
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_quick_scan_response(
                ticker, company_name, current_price, fundamental_summary,
                insider_summary, news_summary, competitive_summary, response_text,
            )
        except Exception as e:
            return self._fallback_report(
                ticker, company_name, current_price, fundamental_summary,
                insider_summary, news_summary, competitive_summary, e,
            )

    async def gather_analysis_inputs(self, ticker: str) -> dict:
        """Fetch and locally analyze everything analyze_stock() gathers, minus the Claude
        call itself. Used by submit_analysis_batch() so batch requests are built from the
        exact same data/prompt the sequential path would use for the same ticker — no
        separate or thinner data path for batch mode (the mistake PO-1 already fixed once)."""
        quote = await self.market_data.get_quote(ticker)
        financials = await self.market_data.get_financials(ticker)
        technicals = await self.market_data.get_technicals(ticker)
        long_term_trend = await self.market_data.get_long_term_trend(
            ticker, quote.price,
            years=self.config.get("research", {}).get("long_term_trend_years", 5))
        insider_summary_raw = await self.insider_tracker.get_insider_summary(ticker)
        news_items = await self.news_feed.get_company_news(ticker, days=7)

        fundamental_score = await self.fundamental_analyzer.analyze(financials)
        sentiment_analysis = await self.sentiment_analyzer.analyze(news_items)
        insider_analysis = await self.insider_analyzer.analyze(insider_summary_raw)
        competitive_analysis = await self.competitor_analyzer.analyze(ticker)

        company_name = ""
        try:
            import yfinance as yf
            company_name = await asyncio.to_thread(
                lambda: yf.Ticker(ticker).info.get("shortName", ticker)
            )
        except Exception:
            company_name = ticker

        technical_summary = (
            f"Price: ${quote.price:.2f} | SMA50: ${technicals.sma_50:.2f} | "
            f"SMA200: ${technicals.sma_200:.2f} | RSI: {technicals.rsi:.1f} | "
            f"Support: ${technicals.support_level:.2f} | "
            f"Resistance: ${technicals.resistance_level:.2f} | "
            f"Avg Volume 30d: {technicals.avg_volume_30d:,}"
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "current_price": quote.price,
            "sector": getattr(financials, "sector", ""),
            "fundamental_summary": fundamental_score.summary,
            "insider_summary": insider_analysis.summary,
            "news_summary": sentiment_analysis.summary,
            "competitive_summary": competitive_analysis.summary,
            "technical_summary": technical_summary,
            "long_term_trend_summary": format_long_term_trend_summary(long_term_trend),
        }

    async def submit_analysis_batch(self, tickers: list[str]) -> tuple[str | None, dict[str, dict]]:
        """Gather inputs for a chunk of tickers concurrently, build one Batch API request
        per ticker using the identical ANALYSIS_PROMPT the sequential path uses, and submit
        as a single Anthropic Batch API call. Returns (batch_id, inputs_by_ticker) — the
        inputs dict is needed later by fetch_batch_results() to reconstruct each
        ResearchReport, since the batch response doesn't echo back the summaries used to
        build the prompt. Tickers whose data-gathering fails are silently excluded from the
        batch — they're skipped for this run (available again next pre-open), not fatal."""
        if not self.client:
            return None, {}

        # Capped at 5 concurrent tickers' worth of data-gathering — unbounded concurrency
        # here (one asyncio.gather() per ticker, each firing several Finnhub/NewsAPI calls)
        # was found live on 2026-07-15 to blow through Finnhub's rate limit almost
        # immediately for any chunk above ~15 tickers, causing the majority of a chunk to
        # be silently dropped (see CLAUDE.md) before ever reaching Claude. This throttle
        # keeps concurrent gathering fast without triggering the same cascade.
        _gather_semaphore = asyncio.Semaphore(5)

        async def _gather_safe(ticker: str) -> tuple[str, dict | None]:
            async with _gather_semaphore:
                try:
                    return ticker, await self.gather_analysis_inputs(ticker)
                except Exception as e:
                    logger.warning("Batch input gathering failed for %s: %s", ticker, e)
                    return ticker, None

        gathered = await asyncio.gather(*(_gather_safe(t) for t in tickers))
        inputs_by_ticker = {t: inp for t, inp in gathered if inp is not None}
        if not inputs_by_ticker:
            return None, {}

        # 2026-07-24: previously hardcoded to model_quick_scan, which silently made the
        # long-standing model_pre_open_scan Settings-page dropdown dead (nothing ever read
        # it). Now the pre-open batch's own setting actually controls it.
        model = self.config.get("research", {}).get("model_pre_open_scan", "claude-haiku-4-5")
        # Fetched once for the whole batch, not once per ticker (2026-08-04) -- a batch
        # here can build 1,500+ prompts in one pass, and today's market direction is the
        # same value for every one of them; get_market_change_pct's own short cache would
        # naturally dedupe this anyway, but fetching it once up front avoids even the
        # first-ticker's real network call being on this loop's critical path repeatedly.
        market_change_pct = await self.market_data.get_market_change_pct()
        requests = []
        for ticker, inp in inputs_by_ticker.items():
            prompt = self._build_quick_scan_prompt(
                ticker, inp["company_name"], inp["current_price"],
                inp["fundamental_summary"], inp["insider_summary"],
                inp["news_summary"], inp["competitive_summary"], inp["technical_summary"],
                long_term_trend_summary=inp.get("long_term_trend_summary", ""),
                market_change_pct=market_change_pct,
            )
            requests.append({
                "custom_id": ticker,
                "params": {
                    "model": model,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            })

        try:
            batch = await asyncio.to_thread(
                self.client.beta.messages.batches.create, requests=requests,
            )
        except Exception as e:
            logger.error("Batch submission failed for %d tickers: %s", len(requests), e)
            return None, {}

        return batch.id, inputs_by_ticker

    async def poll_batch_status(self, batch_id: str):
        """Returns the raw Anthropic batch status object (.processing_status,
        .request_counts.succeeded/.processing/.errored)."""
        return await asyncio.to_thread(self.client.beta.messages.batches.retrieve, batch_id)

    async def cancel_batch(self, batch_id: str) -> None:
        """Best-effort cancel for a batch the caller has given up on (e.g. the
        zero-progress fallback timeout in _run_batched_chunk_loop). Failures are logged
        and swallowed — the caller has already moved on to the sequential fallback and
        doesn't need this to succeed to make progress."""
        try:
            await asyncio.to_thread(self.client.beta.messages.batches.cancel, batch_id)
        except Exception as e:
            logger.warning("Failed to cancel stuck batch %s: %s", batch_id, e)

    async def fetch_batch_results(
        self, batch_id: str, inputs_by_ticker: dict[str, dict],
    ) -> dict[str, ResearchReport]:
        """Retrieves and parses a completed batch's results into ResearchReport objects,
        keyed by ticker. Uses the same parsing and fallback construction the sequential
        path uses (_parse_quick_scan_response / _fallback_report) so results are
        structurally identical regardless of submission mechanism. Results are also stored
        into self.reports and persisted to disk, exactly as analyze_stock() already does,
        so downstream code (e.g. deep_dive_analysis's "run analyze_stock first" check)
        works identically for batch-analyzed tickers."""
        try:
            results = await asyncio.to_thread(
                lambda: list(self.client.beta.messages.batches.results(batch_id)))
        except Exception as e:
            logger.error("Batch result retrieval failed for batch %s: %s", batch_id, e)
            return {}

        reports: dict[str, ResearchReport] = {}
        for result in results:
            ticker = result.custom_id
            inp = inputs_by_ticker.get(ticker)
            if inp is None:
                continue

            if result.result.type == "succeeded":
                try:
                    text_block = next(
                        (b for b in result.result.message.content if hasattr(b, "text")), None)
                    if not text_block:
                        raise ValueError("No text block in batch response")
                    report = self._parse_quick_scan_response(
                        ticker, inp["company_name"], inp["current_price"],
                        inp["fundamental_summary"], inp["insider_summary"],
                        inp["news_summary"], inp["competitive_summary"], text_block.text,
                    )
                except Exception as e:
                    report = self._fallback_report(
                        ticker, inp["company_name"], inp["current_price"],
                        inp["fundamental_summary"], inp["insider_summary"],
                        inp["news_summary"], inp["competitive_summary"], e,
                    )
            else:
                report = self._fallback_report(
                    ticker, inp["company_name"], inp["current_price"],
                    inp["fundamental_summary"], inp["insider_summary"],
                    inp["news_summary"], inp["competitive_summary"],
                    Exception(f"batch result type={result.result.type}"),
                )

            report.sector = inp.get("sector", "")
            self.reports[ticker] = report
            asyncio.create_task(asyncio.to_thread(self._save_report, report))
            reports[ticker] = report
        return reports

    async def deep_dive_analysis(self, ticker: str, model: str | None = None) -> DeepDiveReport:
        report = self.reports.get(ticker)
        if not report:
            raise ValueError(f"No scan report for {ticker} — run analyze_stock first")

        logger.info("Starting deep-dive for %s", ticker)

        if not self.client:
            logger.warning("No ANTHROPIC_API_KEY — rule-based deep dive for %s", ticker)
            return self._rule_based_deep_dive(ticker, report, "No Anthropic API key configured")

        # Live quote + real technicals (fixed 2026-08-09, GitHub #62) -- this prompt used
        # to anchor `current_price` on `report.entry_price` (Claude's own previously
        # *recommended entry price* from the cached quick-scan, not a real quote -- wrong
        # even on a perfectly fresh report, and increasingly stale for a held position,
        # which per this project's own design only gets event-triggered re-analysis once
        # profitable, not periodic) and built `technical_summary` from nothing but that
        # same stale Entry/Stop pair, even though DEEP_DIVE_PROMPT explicitly asks Claude
        # to evaluate trend/momentum/support/resistance/volume from this section. Mirrors
        # the same live-quote + real-technicals pattern analyze_stock()/deeper_dive_analysis()
        # already use.
        #
        # Wrapped in its own try/except (caught in same-day recheck) -- this data fetch
        # sits before the function's own try/except below (which only ever covered the
        # Claude call/parse), so a live quote/technicals failure would otherwise raise
        # straight out of deep_dive_analysis() instead of degrading to the established
        # _rule_based_deep_dive() fallback every OTHER failure path in this function
        # already uses. Every real caller happens to wrap this call in its own broad
        # except too, so this wasn't a hard crash -- but it silently swapped a real,
        # structured is_fallback=True report for a bare logged error string, which is a
        # real behavior regression against this function's own established contract.
        try:
            quote = await self.market_data.get_quote(ticker)
            technicals = await self.market_data.get_technicals(ticker)
        except Exception as e:
            logger.warning("Live quote/technicals fetch failed for %s deep dive: %s", ticker, e)
            return self._rule_based_deep_dive(ticker, report, f"Market data fetch failed: {e}")
        technical_summary = (
            f"Price: ${quote.price:.2f} | SMA50: ${technicals.sma_50:.2f} | "
            f"SMA200: ${technicals.sma_200:.2f} | RSI: {technicals.rsi:.1f} | "
            f"Support: ${technicals.support_level:.2f} | "
            f"Resistance: ${technicals.resistance_level:.2f} | "
            f"Avg Volume 30d: {technicals.avg_volume_30d:,}"
        )

        prompt = DEEP_DIVE_PROMPT.format(
            current_date=self._now_et().strftime("%Y-%m-%d"),
            ticker=report.ticker,
            company_name=report.company_name,
            current_price=quote.price,
            signal=report.signal.value,
            conviction=report.conviction_score,
            prior_fair_value=report.fair_value_estimate,
            prior_mos=report.margin_of_safety_pct,
            fundamental_summary=report.fundamental_summary,
            insider_summary=report.insider_summary,
            news_summary=report.news_summary,
            competitive_summary=report.competitive_summary,
            technical_summary=technical_summary,
        )

        try:
            response_text = await self._call_claude_with_retry(
                # 2026-07-24: model_periodic_deep_dive can override this shared
                # model_deep_dive default -- see CLAUDE.md's "Model Selection" section.
                model=model or self.config.get("research", {}).get("model_deep_dive", "claude-sonnet-5"),
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rstrip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].rstrip()

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = json.loads(response_text.replace('\r\n', ' ').replace('\n', ' '))

            dd = DeepDiveReport(
                ticker=ticker,
                company_name=report.company_name,
                generated_at=self._now_et(),
                signal=data.get("signal", report.signal.value),
                conviction_score=round(float(data.get("conviction_score", report.conviction_score)), 1),
                thesis=data.get("thesis", report.thesis),
                valuation_analysis=data.get("valuation_analysis", ""),
                fair_value_estimate=float(data.get("fair_value_estimate", report.entry_price)),
                margin_of_safety_pct=float(data.get("margin_of_safety_pct", 0)),
                entry_price=float(data.get("entry_price", report.entry_price)),
                stop_loss=float(data.get("stop_loss", report.stop_loss)),
                risk_factors=data.get("risk_factors", report.risk_factors),
                technical_analysis=data.get("technical_analysis", ""),
                competitive_moat=data.get("competitive_moat", ""),
                growth_outlook=data.get("growth_outlook", ""),
                catalysts=data.get("catalysts", []),
                risk_scenarios=data.get("risk_scenarios", []),
                entry_zone_low=float(data.get("entry_zone_low", report.stop_loss)),
                entry_zone_high=float(data.get("entry_zone_high", report.entry_price)),
                monitoring_checklist=data.get("monitoring_checklist", []),
                enhanced_reasoning=data.get("enhanced_reasoning", ""),
            )

            self.deep_dives[ticker] = dd
            await asyncio.to_thread(self._save_deep_dive, dd)
            logger.info("Deep dive complete for %s — Fair value $%.2f, Margin %.0f%%",
                        ticker, dd.fair_value_estimate, dd.margin_of_safety_pct)
            return dd

        except Exception as e:
            reason = str(e)
            if "Unterminated string" in reason or "JSONDecodeError" in type(e).__name__:
                reason = f"Response truncated (JSON parse error) — try again: {reason}"
            logger.error("Deep dive failed for %s: %s", ticker, reason)
            return self._rule_based_deep_dive(ticker, report, reason)

    def _rule_based_deep_dive(self, ticker: str, report: ResearchReport, reason: str = "") -> DeepDiveReport:
        price = report.entry_price
        return DeepDiveReport(
            ticker=ticker,
            company_name=report.company_name,
            generated_at=self._now_et(),
            signal=report.signal.value,
            conviction_score=report.conviction_score,
            thesis=report.thesis,
            valuation_analysis="Rule-based estimate only — AI analysis unavailable.",
            fair_value_estimate=round(price * 1.15, 2),
            margin_of_safety_pct=13.0,
            entry_price=price,
            stop_loss=report.stop_loss,
            risk_factors=report.risk_factors,
            technical_analysis=f"Entry: ${price:.2f} | Stop: ${report.stop_loss:.2f}",
            competitive_moat="Not assessed — AI analysis unavailable.",
            growth_outlook="Not assessed — AI analysis unavailable.",
            catalysts=[{"event": "Earnings report", "timeframe": "Next quarter", "impact": "Moderate"}],
            risk_scenarios=[
                {"scenario": "Bull", "probability_pct": 25, "price_target": round(price * 1.35, 2),
                 "description": "Strong earnings beat and guidance raise"},
                {"scenario": "Base", "probability_pct": 50, "price_target": round(price * 1.10, 2),
                 "description": "In-line results, gradual appreciation"},
                {"scenario": "Bear", "probability_pct": 25, "price_target": round(price * 0.85, 2),
                 "description": "Earnings miss or macro headwinds"},
            ],
            entry_zone_low=round(price * 0.97, 2),
            entry_zone_high=round(price * 1.02, 2),
            monitoring_checklist=["Quarterly earnings", "Insider activity changes", "Sector rotation signals"],
            enhanced_reasoning=report.reasoning or report.thesis,
            is_fallback=True,
            fallback_reason=reason or "No AI client available (missing API key)",
        )

    async def deeper_dive_analysis(self, ticker: str) -> DeepDiveReport:
        """Deeper dive: uses prior deep dive as context for a much more comprehensive analysis."""
        report = self.reports.get(ticker)
        if not report:
            raise ValueError(f"No scan report for {ticker} — run analyze_stock first")

        prior = self.deep_dives.get(ticker)
        if not prior:
            logger.info("No cached deep dive for %s — running deep_dive_analysis first", ticker)
            prior = await self.deep_dive_analysis(ticker)
            if getattr(prior, "is_fallback", False):
                return prior

        logger.info("Starting DEEPER dive for %s", ticker)

        if not self.client:
            return self._rule_based_deep_dive(ticker, report, "No Anthropic API key configured")

        # Refresh live data so deeper analysis has current context
        quote = await self.market_data.get_quote(ticker)
        financials = await self.market_data.get_financials(ticker)
        technicals = await self.market_data.get_technicals(ticker)
        insider_summary = await self.insider_tracker.get_insider_summary(ticker)
        news_items = await self.news_feed.get_company_news(ticker, days=14)
        fundamental_score = await self.fundamental_analyzer.analyze(financials)
        sentiment_analysis = await self.sentiment_analyzer.analyze(news_items)
        insider_analysis = await self.insider_analyzer.analyze(insider_summary)
        competitive_analysis = await self.competitor_analyzer.analyze(ticker)

        technical_summary = (
            f"Price: ${quote.price:.2f} | SMA50: ${technicals.sma_50:.2f} | "
            f"SMA200: ${technicals.sma_200:.2f} | RSI: {technicals.rsi:.1f} | "
            f"Support: ${technicals.support_level:.2f} | Resistance: ${technicals.resistance_level:.2f}"
        )

        prompt = DEEPER_DIVE_PROMPT.format(
            ticker=ticker,
            company_name=report.company_name,
            current_price=quote.price,
            prior_signal=prior.signal,
            prior_conviction=prior.conviction_score,
            prior_thesis=prior.thesis,
            prior_fair_value=prior.fair_value_estimate,
            prior_mos=prior.margin_of_safety_pct,
            prior_ez_low=prior.entry_zone_low,
            prior_ez_high=prior.entry_zone_high,
            prior_stop=prior.stop_loss,
            prior_valuation=prior.valuation_analysis,
            prior_technical=prior.technical_analysis,
            prior_moat=prior.competitive_moat,
            prior_growth=prior.growth_outlook,
            prior_risks=prior.risk_factors,
            prior_reasoning=prior.enhanced_reasoning,
            fundamental_summary=fundamental_score.summary,
            insider_summary=insider_analysis.summary,
            news_summary=sentiment_analysis.summary,
            competitive_summary=competitive_analysis.summary,
            technical_summary=technical_summary,
        )

        try:
            response_text = await self._call_claude_with_retry(
                model=self.config.get("research", {}).get("model_deeper_dive", "claude-opus-4-8"),
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rstrip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].rstrip()

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = json.loads(response_text.replace('\r\n', ' ').replace('\n', ' '))

            dd = DeepDiveReport(
                ticker=ticker,
                company_name=report.company_name,
                generated_at=self._now_et(),
                signal=data.get("signal", prior.signal),
                conviction_score=round(float(data.get("conviction_score", prior.conviction_score)), 1),
                thesis=data.get("thesis", prior.thesis),
                valuation_analysis=data.get("valuation_analysis", prior.valuation_analysis),
                fair_value_estimate=float(data.get("fair_value_estimate", prior.fair_value_estimate)),
                margin_of_safety_pct=float(data.get("margin_of_safety_pct", prior.margin_of_safety_pct)),
                entry_price=float(data.get("entry_price", quote.price)),
                stop_loss=float(data.get("stop_loss", prior.stop_loss)),
                risk_factors=data.get("risk_factors", prior.risk_factors),
                technical_analysis=data.get("technical_analysis", prior.technical_analysis),
                competitive_moat=data.get("competitive_moat", prior.competitive_moat),
                growth_outlook=data.get("growth_outlook", prior.growth_outlook),
                catalysts=data.get("catalysts", prior.catalysts),
                risk_scenarios=data.get("risk_scenarios", prior.risk_scenarios),
                entry_zone_low=float(data.get("entry_zone_low", prior.entry_zone_low)),
                entry_zone_high=float(data.get("entry_zone_high", prior.entry_zone_high)),
                monitoring_checklist=data.get("monitoring_checklist", prior.monitoring_checklist),
                enhanced_reasoning=data.get("enhanced_reasoning", prior.enhanced_reasoning),
                peer_comparison=data.get("peer_comparison", ""),
                management_quality=data.get("management_quality", ""),
                macro_sensitivity=data.get("macro_sensitivity", ""),
                historical_patterns=data.get("historical_patterns", ""),
                expected_return_6m=float(data.get("expected_return_6m", 0)),
                expected_return_12m=float(data.get("expected_return_12m", 0)),
                is_deeper_dive=True,
            )

            self.deep_dives[ticker] = dd
            await asyncio.to_thread(self._save_deep_dive, dd)
            logger.info("Deeper dive complete for %s — FV $%.2f, E[r] 6m %.1f%% 12m %.1f%%",
                        ticker, dd.fair_value_estimate, dd.expected_return_6m, dd.expected_return_12m)
            return dd

        except Exception as e:
            reason = str(e)
            if "Unterminated string" in reason or "JSONDecodeError" in type(e).__name__:
                reason = f"Response truncated (JSON parse error) — try again: {reason}"
            logger.error("Deeper dive failed for %s: %s", ticker, reason)
            return self._rule_based_deep_dive(ticker, report, reason)

    def _save_deep_dive(self, report: DeepDiveReport):
        filename = f"{report.ticker}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}_deepdive.json"
        filepath = self.reports_dir / filename
        data = {
            "ticker": report.ticker,
            "company_name": report.company_name,
            "generated_at": report.generated_at.isoformat(),
            "signal": report.signal,
            "conviction_score": report.conviction_score,
            "thesis": report.thesis,
            "valuation_analysis": report.valuation_analysis,
            "fair_value_estimate": report.fair_value_estimate,
            "margin_of_safety_pct": report.margin_of_safety_pct,
            "entry_price": report.entry_price,
            "stop_loss": report.stop_loss,
            "risk_factors": report.risk_factors,
            "technical_analysis": report.technical_analysis,
            "competitive_moat": report.competitive_moat,
            "growth_outlook": report.growth_outlook,
            "catalysts": report.catalysts,
            "risk_scenarios": report.risk_scenarios,
            "entry_zone_low": report.entry_zone_low,
            "entry_zone_high": report.entry_zone_high,
            "monitoring_checklist": report.monitoring_checklist,
            "enhanced_reasoning": report.enhanced_reasoning,
            "peer_comparison": report.peer_comparison,
            "management_quality": report.management_quality,
            "macro_sensitivity": report.macro_sensitivity,
            "historical_patterns": report.historical_patterns,
            "expected_return_6m": report.expected_return_6m,
            "expected_return_12m": report.expected_return_12m,
            "is_deeper_dive": report.is_deeper_dive,
            "is_fallback": report.is_fallback,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _rule_based_analysis(
        self, ticker, company_name, price,
        fundamental, insider, sentiment, competitive, technicals,
    ) -> ResearchReport:
        score = 5

        if fundamental.overall_score >= 7:
            score += 1
        elif fundamental.overall_score <= 3:
            score -= 1

        if insider.signal_strength >= 0.5:
            score += 1
        elif insider.signal_strength <= -0.5:
            score -= 1

        if sentiment.overall_sentiment >= 0.3:
            score += 1
        elif sentiment.overall_sentiment <= -0.3:
            score -= 1

        if competitive.moat_score >= 7:
            score += 1
        elif competitive.moat_score <= 3:
            score -= 1

        if technicals.rsi < 30:
            score += 1
        elif technicals.rsi > 70:
            score -= 1

        score = max(1, min(10, score))

        # Rule-based analysis never generates buy signals — AI required for that decision.
        # SELL/STRONG_SELL are still allowed so the system can flag obvious problems.
        if score >= 5:
            signal = Signal.HOLD
        elif score >= 3:
            signal = Signal.SELL
        else:
            signal = Signal.STRONG_SELL

        tp_cfg = self.config.get("take_profit", {})
        sl_pct  = tp_cfg.get("stop_loss_pct", 7.0) / 100
        t1_pct  = tp_cfg.get("t1_pct",  5.0) / 100
        t2_pct  = tp_cfg.get("t2_pct", 10.0) / 100
        t3_pct  = tp_cfg.get("t3_pct", 17.0) / 100
        stop_loss = round(price * (1 - sl_pct), 2)
        t1 = round(price * (1 + t1_pct), 2)
        t2 = round(price * (1 + t2_pct), 2)
        t3 = round(price * (1 + t3_pct), 2)

        logger.warning("Rule-based fallback analysis for %s — AI unavailable, buy signals blocked", ticker)
        return ResearchReport(
            ticker=ticker,
            company_name=company_name,
            generated_at=self._now_et(),
            conviction_score=score,
            signal=signal,
            risk_level=RiskLevel.HIGH,
            thesis=f"⚠ Rule-based estimate only (AI unavailable). Score: {score}/10 — do not trade without AI confirmation.",
            fundamental_summary=fundamental.summary,
            insider_summary=insider.summary,
            news_summary=sentiment.summary,
            competitive_summary=competitive.summary,
            risk_factors="AI analysis unavailable — rule-based estimates only. Do not trade on this signal.",
            recommended_action="HOLD — await AI analysis",
            entry_price=price,
            position_size_pct=0.0,
            stop_loss=stop_loss,
            take_profit_targets=[t1, t2, t3],
            time_horizon="unknown",
            reasoning=(
                f"⚠ No AI analysis available. Rule-based score {score}/10 from: "
                f"Fundamental {fundamental.overall_score:.1f}/10, "
                f"Insider {insider.signal_strength:.2f}, "
                f"Sentiment {sentiment.overall_sentiment:.2f}, "
                f"Moat {competitive.moat_score:.1f}/10, "
                f"RSI {technicals.rsi:.1f}."
            ),
            is_fallback=True,
        )

    def _save_report(self, report: ResearchReport):
        filename = f"{report.ticker}_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        data = {
            "ticker": report.ticker,
            "company_name": report.company_name,
            "generated_at": report.generated_at.isoformat(),
            "conviction_score": report.conviction_score,
            "signal": report.signal.value,
            "risk_level": report.risk_level.value,
            "thesis": report.thesis,
            "fundamental_summary": report.fundamental_summary,
            "insider_summary": report.insider_summary,
            "news_summary": report.news_summary,
            "competitive_summary": report.competitive_summary,
            "risk_factors": report.risk_factors,
            "recommended_action": report.recommended_action,
            "entry_price": report.entry_price,
            "position_size_pct": report.position_size_pct,
            "stop_loss": report.stop_loss,
            "take_profit_targets": report.take_profit_targets,
            "time_horizon": report.time_horizon,
            "reasoning": report.reasoning,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
