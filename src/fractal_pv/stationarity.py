"""Stationarity testing and time series preprocessing.

Before computing Hurst exponents, we need to know whether the series is
stationary (use directly) or nonstationary (transform first). This module
provides ADF/KPSS tests and the standard transforms for financial data.

Convention:
- Raw prices → nonstationary (unit root). Transform to log returns.
- Log returns → approximately stationary. Use for Hurst estimation.
- Absolute log returns → proxy for volatility. Has long memory (H > 0.5).
- Volume → often nonstationary in level. Transform to log(volume).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StationarityResult:
    """Result of a stationarity test."""

    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    detail: str

    def __str__(self) -> str:
        status = "stationary" if self.is_stationary else "nonstationary"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f} → {status}"


def adf_test(series: np.ndarray, significance: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller test for unit root.

    H0: series has a unit root (nonstationary).
    Reject H0 (p < significance) → stationary.

    Parameters
    ----------
    series : array-like
        1D time series.
    significance : float
        Significance level for the decision. Default 0.05.

    Returns
    -------
    StationarityResult
    """
    from statsmodels.tsa.stattools import adfuller

    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]

    result = adfuller(series, autolag="AIC")
    stat, p_value = result[0], result[1]
    lags_used = result[2]

    return StationarityResult(
        test_name="ADF",
        statistic=float(stat),
        p_value=float(p_value),
        is_stationary=p_value < significance,
        detail=f"Lags used: {lags_used}. H0: unit root. Reject if p < {significance}.",
    )


def kpss_test(series: np.ndarray, significance: float = 0.05) -> StationarityResult:
    """KPSS test for stationarity.

    H0: series is stationary.
    Reject H0 (p < significance) → nonstationary.
    Note: KPSS has the opposite null hypothesis from ADF.

    Parameters
    ----------
    series : array-like
        1D time series.
    significance : float
        Significance level for the decision. Default 0.05.

    Returns
    -------
    StationarityResult
    """
    from statsmodels.tsa.stattools import kpss

    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]

    stat, p_value, _, _ = kpss(series, regression="c", nlags="auto")

    return StationarityResult(
        test_name="KPSS",
        statistic=float(stat),
        p_value=float(p_value),
        is_stationary=p_value >= significance,  # fail to reject H0 → stationary
        detail=f"H0: stationary. Reject if p < {significance}. (Opposite of ADF.)",
    )


def diagnose_stationarity(series: np.ndarray) -> dict[str, StationarityResult]:
    """Run both ADF and KPSS and return a diagnostic interpretation.

    Interpretation matrix:
    - ADF rejects, KPSS doesn't reject → stationary
    - ADF doesn't reject, KPSS rejects → nonstationary
    - Both reject → trend-stationary (difference first)
    - Neither rejects → inconclusive (likely low power)
    """
    return {
        "ADF": adf_test(series),
        "KPSS": kpss_test(series),
    }


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series.

    log_return(t) = log(price(t)) - log(price(t-1))

    Parameters
    ----------
    prices : array-like
        1D price series. Must be positive.

    Returns
    -------
    np.ndarray
        Log returns (length = len(prices) - 1).
    """
    prices = np.asarray(prices, dtype=float).flatten()
    prices = prices[prices > 0]
    return np.diff(np.log(prices))


def abs_log_returns(prices: np.ndarray) -> np.ndarray:
    """Absolute log returns — proxy for volatility.

    This is the series with long memory (H ≈ 0.7-0.8) per Bollerslev & Jubinski 1999.
    Raw returns are approximately iid (H ≈ 0.5).
    """
    return np.abs(log_returns(prices))


def log_volume(volume: np.ndarray) -> np.ndarray:
    """Log-transform volume. Adds 1 to avoid log(0).

    Volume often has long memory (H ≈ 0.7-0.9) per Lobato & Velasco 2000.
    Log transform stabilizes variance.
    """
    volume = np.asarray(volume, dtype=float).flatten()
    return np.log(volume + 1)


def prepare_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Prepare all analysis series from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'Close' and 'Volume' columns.

    Returns
    -------
    dict with keys: 'log_returns', 'abs_log_returns', 'log_volume', 'close', 'volume'
    """
    close = df["Close"].dropna().values.flatten()
    vol = df["Volume"].dropna().values.flatten()

    # Align lengths (returns are 1 shorter than prices)
    min_len = min(len(close), len(vol))
    close = close[:min_len]
    vol = vol[:min_len]

    lr = log_returns(close)
    alr = np.abs(lr)
    lv = log_volume(vol[1:])  # align with returns (drop first)

    return {
        "log_returns": lr,
        "abs_log_returns": alr,
        "log_volume": lv,
        "close": close,
        "volume": vol,
    }
