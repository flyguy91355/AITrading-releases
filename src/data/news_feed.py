"""News aggregation and filtering via Finnhub and NewsAPI."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsItem:
    ticker: str
    headline: str
    summary: str
    source: str
    url: str
    published: datetime
    sentiment: SentimentScore = SentimentScore.NEUTRAL
    relevance_score: float = 0.0
    category: str = ""


class NewsFeed:
    def __init__(self, config: dict):
        self.config = config
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        self.newsapi_key = os.getenv("NEWSAPI_API_KEY", "")

    async def get_company_news(self, ticker: str, days: int = 7) -> list[NewsItem]:
        items = []

        if self.finnhub_key:
            items.extend(await self._finnhub_company_news(ticker, days))

        if self.newsapi_key and len(items) < 5:
            items.extend(await self._newsapi_search(ticker, days))

        items.sort(key=lambda n: n.published, reverse=True)
        return items

    async def _finnhub_company_news(self, ticker: str, days: int) -> list[NewsItem]:
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={
                        "symbol": ticker,
                        "from": from_date,
                        "to": to_date,
                        "token": self.finnhub_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("Finnhub news failed for %s: %s", ticker, e)
            return []

        items = []
        for article in data[:30]:
            try:
                published = datetime.fromtimestamp(article.get("datetime", 0))
            except (ValueError, TypeError, OSError):
                published = datetime.now()

            items.append(NewsItem(
                ticker=ticker,
                headline=article.get("headline", ""),
                summary=article.get("summary", ""),
                source=article.get("source", ""),
                url=article.get("url", ""),
                published=published,
                category=article.get("category", ""),
            ))
        return items

    async def _newsapi_search(self, query: str, days: int = 7) -> list[NewsItem]:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        # searchIn=title (fixed 2026-08-09, GitHub #64; corrected same
                        # day after a recheck caught the first attempt using a
                        # nonexistent "qInTitle" parameter -- confirmed against
                        # NewsAPI's real /v2/everything docs via WebFetch, not assumed
                        # from memory this time) -- without this, q matches anywhere in
                        # an article's title/description/content, a huge false-positive
                        # surface for a ticker that's also a common English word (MET,
                        # ALL, KEY, LOW, ARE, NOW, ONE -- several of which are real
                        # tickers in this system's actual stock universe). searchIn
                        # restricts matching to the headline only -- a real financial
                        # article about a specific company overwhelmingly names it in
                        # the title, while an unrelated article that merely contains
                        # the word somewhere in its body text won't match at all. This
                        # is a fallback path only (only fires when Finnhub, the
                        # symbol-scoped primary source, returns fewer than 5 items), so
                        # a real reduction in match volume here is the intended effect,
                        # not a regression.
                        "searchIn": "title",
                        "from": from_date,
                        "sortBy": "relevancy",
                        "language": "en",
                        "pageSize": 20,
                        "apiKey": self.newsapi_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("NewsAPI search failed for '%s': %s", query, e)
            return []

        items = []
        for article in data.get("articles", []):
            try:
                published = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError, KeyError):
                published = datetime.now()

            ticker = query.upper() if len(query) <= 5 and query.isalpha() else "MARKET"

            items.append(NewsItem(
                ticker=ticker,
                headline=article.get("title", ""),
                summary=article.get("description", "") or "",
                source=article.get("source", {}).get("name", ""),
                url=article.get("url", ""),
                published=published,
            ))
        return items
