"""Data fetching and caching for fractal price-volume analysis.

Fetches OHLCV data from Yahoo Finance via yfinance, caches locally as parquet.
Daily data is the default — it gives the deepest history and is the literature
standard for Hurst exponent estimation (which needs 500+ data points minimum).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_ticker(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker, caching as parquet.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL").
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date. None means today.
    interval : str
        Data interval. "1d" for daily (recommended), "1h" for hourly.
    cache_dir : Path
        Directory for cached parquet files.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        OHLCV dataframe indexed by date.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_{interval}_{start}_{end or 'latest'}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_path)
    return df


def fetch_universe(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_observations: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers, filtering by minimum length.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start, end, interval, cache_dir
        Passed to fetch_ticker.
    min_observations : int
        Minimum number of rows required. Tickers with fewer are skipped.
        500 is the minimum for reliable DFA estimation (~2 years of daily data).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV dataframe, only for tickers meeting the minimum.
    """
    universe = {}
    failed = []

    for ticker in tqdm(tickers, desc="Fetching"):
        try:
            df = fetch_ticker(ticker, start=start, end=end, interval=interval, cache_dir=cache_dir)
            if len(df) >= min_observations:
                universe[ticker] = df
        except Exception as e:
            failed.append((ticker, str(e)))

    if failed:
        print(f"\nFailed to fetch {len(failed)} tickers:")
        for t, err in failed[:10]:
            print(f"  {t}: {err}")

    print(f"\nFetched {len(universe)}/{len(tickers)} tickers with >= {min_observations} observations")
    return universe


# Common ticker universes for analysis
SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "DIS", "ADBE", "CRM", "NFLX", "CMCSA", "PFE",
    "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "NKE", "MRK", "WMT", "CVX",
    "LLY", "MCD", "DHR", "TXN", "NEE", "BMY", "UPS", "MS", "SCHW", "RTX",
    "LOW", "INTC", "QCOM", "INTU", "AMAT", "GS", "BLK", "ISRG", "MDLZ", "ADP",
]
