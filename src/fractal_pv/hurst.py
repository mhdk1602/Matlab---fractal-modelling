"""Unified Hurst exponent estimation interface.

Wraps multiple estimation methods behind a consistent API so callers
don't need to know which backend is in use. Each method returns a
HurstResult dataclass with the estimate, method name, goodness-of-fit,
and the log-log data needed for diagnostic plots.

Primary: nolds.dfa (handles nonstationarity via detrending)
Secondary: nolds.hurst_rs (classical R/S, for comparison)
Tertiary: MFDFA at q=2 (should agree with DFA for monofractal series)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HurstResult:
    """Result of a Hurst exponent estimation."""

    H: float
    method: str
    r_squared: float = np.nan
    log_scales: np.ndarray = field(default_factory=lambda: np.array([]))
    log_fluctuations: np.ndarray = field(default_factory=lambda: np.array([]))
    error: str | None = None

    @property
    def fractal_dimension(self) -> float:
        """D = 2 - H. Bridges Hurst exponent to fractal geometry."""
        return 2.0 - self.H

    @property
    def is_persistent(self) -> bool:
        return self.H > 0.55

    @property
    def is_antipersistent(self) -> bool:
        return self.H < 0.45

    @property
    def interpretation(self) -> str:
        if self.H > 0.55:
            return "persistent (trending)"
        elif self.H < 0.45:
            return "anti-persistent (mean-reverting)"
        return "approximately random walk"


def _compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """R-squared of linear fit in log-log space."""
    if len(x) < 3:
        return np.nan
    coeffs = np.polyfit(x, y, 1)
    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def estimate_dfa(series: np.ndarray, min_scale: int = 10, max_scale: int | None = None) -> HurstResult:
    """Estimate Hurst exponent via Detrended Fluctuation Analysis.

    Uses nolds.dfa. DFA handles nonstationarity by removing local polynomial
    trends before computing fluctuation functions. Preferred for financial data.

    Parameters
    ----------
    series : array-like
        1D time series (log returns, absolute returns, or log volume).
    min_scale : int
        Minimum window size for DFA. Default 10.
    max_scale : int or None
        Maximum window size. Default len(series)//4.

    Returns
    -------
    HurstResult
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 50:
        return HurstResult(H=np.nan, method="DFA", error="Series too short (< 50)")

    if max_scale is None:
        max_scale = len(series) // 4

    try:
        import nolds

        nvals = nolds.logarithmic_n(min_scale, max_scale, 1.2)
        nvals = [int(n) for n in nvals if n <= max_scale]

        if len(nvals) < 4:
            return HurstResult(H=np.nan, method="DFA", error="Not enough scales")

        H = nolds.dfa(series, nvals=nvals)

        # Recompute fluctuation function for log-log data
        from nolds import measures as _m

        fluctuations = []
        valid_nvals = []
        for n in nvals:
            try:
                f = _m._dfa_fluctuation(series, n)
                if f > 0:
                    fluctuations.append(f)
                    valid_nvals.append(n)
            except Exception:
                continue

        log_scales = np.log(valid_nvals) if valid_nvals else np.array([])
        log_fluct = np.log(fluctuations) if fluctuations else np.array([])
        r2 = _compute_r_squared(log_scales, log_fluct) if len(log_scales) >= 3 else np.nan

        return HurstResult(
            H=float(H),
            method="DFA",
            r_squared=r2,
            log_scales=log_scales,
            log_fluctuations=log_fluct,
        )

    except ImportError:
        return HurstResult(H=np.nan, method="DFA", error="nolds not installed")
    except Exception as e:
        return HurstResult(H=np.nan, method="DFA", error=str(e))


def estimate_rs(series: np.ndarray) -> HurstResult:
    """Estimate Hurst exponent via Rescaled Range (R/S) analysis.

    Classical method from Hurst (1951). Known to overestimate H for series
    with short-range dependence (Lo 1991). Use DFA as primary; this is for
    comparison and method-agreement checks.

    Parameters
    ----------
    series : array-like
        1D time series.

    Returns
    -------
    HurstResult
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 50:
        return HurstResult(H=np.nan, method="R/S", error="Series too short (< 50)")

    try:
        import nolds

        H = nolds.hurst_rs(series)
        return HurstResult(H=float(H), method="R/S")

    except ImportError:
        return HurstResult(H=np.nan, method="R/S", error="nolds not installed")
    except Exception as e:
        return HurstResult(H=np.nan, method="R/S", error=str(e))


def estimate_mfdfa_q2(series: np.ndarray) -> HurstResult:
    """Estimate Hurst exponent via MFDFA at q=2 (equivalent to standard DFA).

    Uses the MFDFA package. At q=2, MFDFA reduces to standard DFA. Useful as
    a cross-check and as the entry point for multifractal analysis (higher q values).

    Parameters
    ----------
    series : array-like
        1D time series.

    Returns
    -------
    HurstResult
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 100:
        return HurstResult(H=np.nan, method="MFDFA(q=2)", error="Series too short (< 100)")

    try:
        from MFDFA import MFDFA as _MFDFA

        max_lag = len(series) // 4
        lag = np.unique(np.logspace(1, np.log10(max_lag), 50).astype(int))
        lag = lag[lag >= 10]

        if len(lag) < 4:
            return HurstResult(H=np.nan, method="MFDFA(q=2)", error="Not enough lag values")

        lag, dfa_vals = _MFDFA(series, lag=lag, q=2, order=1)
        dfa_vals = dfa_vals.flatten()

        mask = (dfa_vals > 0) & np.isfinite(dfa_vals)
        log_scales = np.log(lag[mask])
        log_fluct = np.log(dfa_vals[mask])

        if len(log_scales) < 3:
            return HurstResult(H=np.nan, method="MFDFA(q=2)", error="Not enough valid scales")

        H = np.polyfit(log_scales, log_fluct, 1)[0]
        r2 = _compute_r_squared(log_scales, log_fluct)

        return HurstResult(
            H=float(H),
            method="MFDFA(q=2)",
            r_squared=r2,
            log_scales=log_scales,
            log_fluctuations=log_fluct,
        )

    except ImportError:
        return HurstResult(H=np.nan, method="MFDFA(q=2)", error="MFDFA not installed")
    except Exception as e:
        return HurstResult(H=np.nan, method="MFDFA(q=2)", error=str(e))


def estimate_all(series: np.ndarray) -> dict[str, HurstResult]:
    """Run all Hurst estimators and return results keyed by method name."""
    return {
        "DFA": estimate_dfa(series),
        "R/S": estimate_rs(series),
        "MFDFA(q=2)": estimate_mfdfa_q2(series),
    }
