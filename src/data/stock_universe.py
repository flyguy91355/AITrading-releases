"""Stock universe for watchlist replacement scanning.

Dynamically fetches index constituents from Wikipedia, cached per-index for 24 hours.
Supported indexes: S&P 500, S&P 400, S&P 600, Nasdaq 100, Dow Jones 30.
Falls back to the hardcoded list if all fetches fail.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "universe_cache.json"
_CACHE_TTL_HOURS = 24
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AITrading/1.0; stock research bot)"}

# All available index sources. Label must match the universe_indexes config values.
_WIKI_SOURCES: list[tuple] = [
    ("S&P 500",
     "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
     {"id": "constituents"}, ("Symbol",)),
    ("S&P 400",
     "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
     {}, ("Ticker symbol", "Symbol", "Ticker")),
    ("S&P 600",
     "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
     {}, ("Ticker symbol", "Symbol", "Ticker")),
    ("Nasdaq 100",
     "https://en.wikipedia.org/wiki/Nasdaq-100",
     {"id": "constituents"}, ("Ticker", "Symbol")),
    ("Dow Jones 30",
     "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
     {}, ("Symbol", "Ticker")),
]

_SOURCE_MAP: dict[str, tuple] = {src[0]: src for src in _WIKI_SOURCES}

# Default indexes when none are configured
_DEFAULT_INDEXES = ["S&P 500", "S&P 400", "S&P 600"]


def _fetch_wiki_tickers(label: str, url: str, attrs: dict, col_names: tuple) -> list[str]:
    import io
    import requests
    import pandas as pd
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), attrs=attrs or None)
    for t in tables:
        for col in col_names:
            if col in t.columns:
                tickers = (t[col].dropna()
                           .astype(str)
                           .str.replace(".", "-", regex=False)
                           .str.strip()
                           .tolist())
                logger.info("Fetched %d %s tickers from Wikipedia", len(tickers), label)
                return tickers
    raise ValueError(f"No matching column found in {label} Wikipedia table")


def get_universe(enabled: list[str] | None = None) -> list[str]:
    """Return tickers for the enabled indexes, refreshing stale cache entries from Wikipedia.

    Cache is stored per-index so adding/removing an index only fetches what changed.
    Old single-blob cache format is detected and treated as stale.
    """
    if not enabled:
        enabled = _DEFAULT_INDEXES

    # Load per-index cache; detect and discard old flat format
    cache: dict[str, dict] = {}
    if _CACHE_PATH.exists():
        try:
            raw = json.loads(_CACHE_PATH.read_text())
            # New format has index-name keys; old format has top-level "cached_at"/"tickers"
            if isinstance(raw, dict) and not raw.get("cached_at"):
                cache = raw
        except Exception:
            pass

    combined: set[str] = set()
    cache_dirty = False

    for label in enabled:
        if label not in _SOURCE_MAP:
            logger.warning("Unknown universe index '%s' — skipping", label)
            continue

        # Use cached entry if fresh
        entry = cache.get(label, {})
        if entry:
            try:
                if datetime.now() - datetime.fromisoformat(entry["cached_at"]) < timedelta(hours=_CACHE_TTL_HOURS):
                    combined.update(entry["tickers"])
                    continue
            except Exception:
                pass

        # Fetch from Wikipedia
        _, url, attrs, col_names = _SOURCE_MAP[label]
        try:
            tickers = _fetch_wiki_tickers(label, url, attrs, col_names)
            if len(tickers) < 10:
                raise ValueError(f"Suspiciously few tickers ({len(tickers)}) for {label}")
            cache[label] = {"cached_at": datetime.now().isoformat(), "tickers": tickers}
            cache_dirty = True
            combined.update(tickers)
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", label, e)
            # Fall back to stale cache rather than nothing
            if entry and entry.get("tickers"):
                combined.update(entry["tickers"])
                logger.warning("Using stale cache for %s (%d stocks)", label, len(entry["tickers"]))

    if cache_dirty:
        try:
            _CACHE_PATH.parent.mkdir(exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(cache))
        except Exception as e:
            logger.warning("Failed to write universe cache: %s", e)

    if not combined:
        logger.warning("No tickers from any index — using hardcoded fallback (%d stocks)", len(STOCK_UNIVERSE))
        return list(STOCK_UNIVERSE)

    result = sorted(combined)
    logger.info("Universe ready: %d stocks from %s", len(result), ", ".join(enabled))
    return result


# ── Hardcoded fallback (~457 stocks) ──────────────────────────────────────────
# Used when Wikipedia fetch fails. Covers S&P 500 large-caps across all sectors.
STOCK_UNIVERSE = [
    # Mega-cap / current watchlist
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "LLY", "AVGO", "JPM", "V", "UNH", "MA", "XOM", "COST", "HD",
    "PG", "JNJ", "ABBV", "CRM", "NFLX", "AMD", "BAC", "KO", "MRK",
    "PEP", "TMO", "ORCL", "ACN", "LIN", "WMT", "CSCO", "MCD", "ABT",
    "DIS", "ADBE", "DHR", "WFC", "INTC", "QCOM", "INTU", "TXN", "PM",
    "NOW", "IBM", "GE", "CAT", "AMAT", "GS",
    # Technology
    "LRCX", "MU", "KLAC", "SNPS", "CDNS", "ANSS", "FTNT", "PANW",
    "CRWD", "ZS", "OKTA", "DDOG", "SNOW", "PLTR", "UBER", "LYFT",
    "TTD", "SHOP", "SQ", "PYPL", "TWLO", "ZM", "DOCU", "WORK",
    "NET", "CFLT", "MDB", "ESTC", "GTLB", "HUBS", "BILL", "COUP",
    "MSCI", "SPGI", "ICE", "CME", "CBOE", "MCO", "FIS", "FISV",
    "GPN", "WEX", "JKHY", "BR", "SSNC",
    # Healthcare
    "BMY", "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ILMN", "IQV",
    "ZBH", "SYK", "BSX", "MDT", "BDX", "EW", "ISRG", "ALGN",
    "DXCM", "IDXX", "PODD", "NVAX", "MRNA", "BNTX", "PFE", "CVS",
    "CI", "HUM", "ELV", "MOH", "CNC", "HCA", "THC", "UHS", "LPLA",
    "MCK", "CAH", "ABC", "PRGO", "ENDP", "PKI", "A", "RMD",
    # Financials
    "MS", "C", "USB", "PNC", "TFC", "COF", "AXP", "DFS", "SYF",
    "ALLY", "CMA", "FITB", "HBAN", "RF", "KEY", "CFG", "MTB",
    "ZION", "BOKF", "FHN", "SNV", "PBCT", "TCF", "WAL", "EWBC",
    "BK", "STT", "NTRS", "IVZ", "BEN", "AMG", "TROW", "VRTS",
    "PFG", "LNC", "AFL", "MET", "PRU", "AIG", "ALL", "TRV",
    "CB", "HIG", "WR", "MKL", "RLI", "CINF",
    # Consumer Discretionary
    "NKE", "SBUX", "LOW", "TJX", "ROST", "BURL", "GPS", "ANF",
    "AEO", "URBN", "LULU", "RH", "WSM", "BBY", "BBWI", "ETSY",
    "EBAY", "W", "CHWY", "CVNA", "KMX", "AZO", "ORLY",
    "AAP", "GPC", "GM", "F", "STLA", "TM", "HMC", "RIVN",
    "LCID", "NKLA", "BLNK", "MAR", "HLT", "H", "WH", "IHG",
    "CCL", "RCL", "NCLH", "LVS", "MGM", "WYNN", "CZR",
    # Consumer Staples
    "MO", "BTI", "MDLZ", "HSY", "GIS", "K", "CPB", "SJM",
    "CAG", "MKC", "HRL", "TSN", "KHC", "POST", "LANC", "BGS",
    "CLX", "CHD", "CL", "EL", "AVP", "COTY", "REV",
    "SFM", "KR", "ACI", "SYY", "PFGC", "USFD",
    # Energy
    "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "OXY", "DVN",
    "FANG", "MPC", "VLO", "PSX", "PBF", "DK", "WMB",
    "KMI", "OKE", "EPD", "ET", "MPLX", "PAA", "TRGP", "AM",
    "AR", "EQT", "RRC", "CNX", "SWN",
    # Industrials
    "HON", "MMM", "RTX", "LMT", "NOC", "GD", "BA", "LHX",
    "TDG", "HEI", "TXT", "DRS", "KTOS", "AVAV",
    "UPS", "FDX", "CHRW", "EXPD", "XPO", "ODFL", "SAIA",
    "JBHT", "WERN", "KNX", "URI",
    "PWR", "PRIM", "MTZ", "EME", "MYR", "J", "EXPO",
    "PH", "EMR", "ITW", "DOV", "AME", "GWW", "MSM",
    "ROP", "XYL", "MIDD",
    # Materials
    "FCX", "NEM", "GOLD", "AEM", "KGC", "AG", "HL", "CDE",
    "AA", "CENX", "ATI", "MP",
    "DD", "DOW", "LYB", "WLK", "OLN", "RPM",
    "SHW", "PPG", "AXTA",
    "NUE", "STLD", "CLF", "X", "CMC", "RS",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG",
    "PEG", "EIX", "XEL", "WEC", "DTE", "ETR", "FE", "CNP",
    "AES", "NI", "EVRG", "AWK",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "EQR", "AVB", "ESS",
    "MAA", "UDR", "CPT", "NNN", "O", "WPC", "SPG",
    "SLG", "BXP", "KIM", "REG", "FRT",
    "DRE", "FR", "EGP", "REXR", "TRNO", "COLD",
    # Communication Services
    "T", "VZ", "TMUS", "CHTR", "CMCSA", "DISH", "LUMN",
    "OMC", "IPG",
    "EA", "TTWO", "ATVI", "RBLX", "U", "MTCH", "BMBL",
    "SNAP", "PINS", "SPOT", "YELP", "IAC",
]
