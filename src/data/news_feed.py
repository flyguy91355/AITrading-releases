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


def _is_ticker_like(query: str) -> bool:
    """True when `query` looks like a real US stock symbol rather than a free-text
    search phrase (2026-08-31, GitHub #136).

    The previous `query.isalpha()` check was False for a real dot-class ticker
    (BRK.B, BF.B) because `.` isn't alphabetic, so every NewsAPI-fallback item for
    those symbols was mislabeled `ticker="MARKET"` — silently misattributing that
    company's news to the generic market bucket. `.` and `-` are both real US
    ticker-class characters (BRK.B, BRK-B), so both are allowed; at least one
    letter is still required, and the original 5-character ceiling is unchanged."""
    if not query or len(query) > 5:
        return False
    if not any(c.isalpha() for c in query):
        return False
    return all(c.isalpha() or c in ".-" for c in query)


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

        # A 200 response whose body isn't the documented JSON array (a plan-restricted
        # or error-shaped dict, which resp.raise_for_status() can't catch since the HTTP
        # status itself is fine) used to raise an uncaught exception out of `data[:30]`
        # below -- KeyError on Python 3.12 (slice objects are hashable there, so the
        # dict lookup simply misses), TypeError on a str/None body -- violating this
        # function's "never raise, return []" contract (2026-08-31, GitHub #135).
        if not isinstance(data, list):
            logger.warning(
                "Finnhub news for %s returned unexpected payload type %s — ignoring",
                ticker, type(data).__name__,
            )
            return []

        items = []
        for article in data[:30]:
            if not isinstance(article, dict):
                continue
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

        # Mirror image of the Finnhub guard above (2026-08-31, GitHub #166). That one
        # breaks when a 200 body is a dict where a list is expected; this one has the
        # opposite exposure -- a list/str/None body raises AttributeError on `.get`,
        # out of a function whose contract is likewise "never raise, return []".
        # resp.raise_for_status() can't catch it, since the HTTP status itself is fine.
        if not isinstance(data, dict):
            logger.warning(
                "NewsAPI search for '%s' returned unexpected payload type %s — ignoring",
                query, type(data).__name__,
            )
            return []

        articles = data.get("articles", [])
        if not isinstance(articles, list):
            logger.warning(
                "NewsAPI search for '%s' returned non-list 'articles' (%s) — ignoring",
                query, type(articles).__name__,
            )
            return []

        items = []
        for article in articles:
            # A non-dict element would escape the try below -- it only wraps the
            # publishedAt parse, while the .get() calls that build the NewsItem sit
            # outside it -- so skip it here, same as the Finnhub loop does.
            if not isinstance(article, dict):
                continue

            try:
                published = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError, KeyError):
                published = datetime.now()

            ticker = query.upper() if _is_ticker_like(query) else "MARKET"

            # NewsAPI documents "source" as an object, but a malformed payload can carry
            # a bare string there -- .get("name") on it would raise AttributeError out of
            # this same "never raise" contract, so resolve it defensively.
            source = article.get("source")
            source_name = source.get("name", "") if isinstance(source, dict) else ""

            items.append(NewsItem(
                ticker=ticker,
                headline=article.get("title", ""),
                summary=article.get("description", "") or "",
                source=source_name,
                url=article.get("url", ""),
                published=published,
            ))
        return items
