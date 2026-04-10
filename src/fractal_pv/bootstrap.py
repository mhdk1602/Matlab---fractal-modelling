"""Block bootstrap for Hurst exponent confidence intervals.

Standard (iid) bootstrap destroys temporal dependence, biasing Hurst
estimates downward. Block bootstrap (Politis & Romano 1994) preserves
dependence by resampling contiguous blocks.

This is a key differentiator of this project: no existing implementation
provides bootstrap CIs on fractal exponents.
"""

import numpy as np
from joblib import Parallel, delayed

from .hurst import estimate_dfa


def block_bootstrap_hurst(
    series: np.ndarray,
    n_bootstrap: int = 500,
    block_size: int | None = None,
    method=estimate_dfa,
    n_jobs: int = -1,
    seed: int | None = 42,
) -> dict:
    """Compute block bootstrap confidence interval for Hurst exponent.

    Parameters
    ----------
    series : array-like
        1D time series.
    n_bootstrap : int
        Number of bootstrap resamples. 500 is a reasonable default;
        1000+ for publication-quality CIs.
    block_size : int or None
        Size of contiguous blocks. Default n^(1/3) per Politis & Romano 1994.
    method : callable
        Hurst estimation function. Must accept np.ndarray and return HurstResult.
    n_jobs : int
        Number of parallel jobs. -1 = all cores.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        H_point: float — point estimate from full series
        H_boot: np.ndarray — bootstrap distribution (length n_bootstrap)
        ci_low: float — 2.5th percentile
        ci_high: float — 97.5th percentile
        std: float — bootstrap standard error
        p_value_05: float — p-value for H ≠ 0.5 (two-sided)
        block_size: int — actual block size used
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    if block_size is None:
        block_size = max(1, int(n ** (1 / 3)))

    # Point estimate
    H_point = method(series).H

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_size))

    def _one_bootstrap(i):
        local_rng = np.random.RandomState(seed + i if seed is not None else None)
        block_starts = local_rng.randint(0, n - block_size, size=n_blocks)
        boot_series = np.concatenate([series[s : s + block_size] for s in block_starts])[:n]
        result = method(boot_series)
        return result.H

    H_boot = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_bootstrap)(i) for i in range(n_bootstrap)
        )
    )

    # Remove NaN estimates
    H_boot = H_boot[np.isfinite(H_boot)]

    if len(H_boot) < 10:
        return {
            "H_point": H_point,
            "H_boot": H_boot,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std": np.nan,
            "p_value_05": np.nan,
            "block_size": block_size,
        }

    ci_low = float(np.percentile(H_boot, 2.5))
    ci_high = float(np.percentile(H_boot, 97.5))

    # Two-sided p-value for H ≠ 0.5:
    # Under H0: the bootstrap distribution is centered at 0.5.
    # Shift bootstrap to be centered at 0.5, then compute tail probability.
    H_boot_centered = H_boot - np.mean(H_boot) + 0.5
    p_value = float(np.mean(np.abs(H_boot_centered - 0.5) >= abs(H_point - 0.5)))

    return {
        "H_point": float(H_point),
        "H_boot": H_boot,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": float(np.std(H_boot)),
        "p_value_05": p_value,
        "block_size": block_size,
    }


def paired_hurst_test(
    series_a: np.ndarray,
    series_b: np.ndarray,
    n_bootstrap: int = 500,
    block_size: int | None = None,
    method=estimate_dfa,
    n_jobs: int = -1,
    seed: int | None = 42,
) -> dict:
    """Test whether two series have different Hurst exponents.

    H0: H(series_a) = H(series_b)
    Uses paired block bootstrap on the difference H_a - H_b.

    Useful for testing: is H(|returns|) significantly different from H(volume)?

    Parameters
    ----------
    series_a, series_b : array-like
        Two time series (should be same length for pairing).
    Other parameters same as block_bootstrap_hurst.

    Returns
    -------
    dict with keys:
        H_a, H_b: point estimates
        diff: H_a - H_b
        ci_low, ci_high: 95% CI for the difference
        p_value: two-sided p-value for H_a ≠ H_b
    """
    series_a = np.asarray(series_a, dtype=float)
    series_b = np.asarray(series_b, dtype=float)
    n = min(len(series_a), len(series_b))
    series_a = series_a[:n]
    series_b = series_b[:n]

    if block_size is None:
        block_size = max(1, int(n ** (1 / 3)))

    H_a = method(series_a).H
    H_b = method(series_b).H
    diff_point = H_a - H_b

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_size))

    def _one_bootstrap(i):
        local_rng = np.random.RandomState(seed + i if seed is not None else None)
        # Same block positions for both series (paired)
        block_starts = local_rng.randint(0, n - block_size, size=n_blocks)
        boot_a = np.concatenate([series_a[s : s + block_size] for s in block_starts])[:n]
        boot_b = np.concatenate([series_b[s : s + block_size] for s in block_starts])[:n]
        ha = method(boot_a).H
        hb = method(boot_b).H
        return ha - hb

    diffs = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_bootstrap)(i) for i in range(n_bootstrap)
        )
    )

    diffs = diffs[np.isfinite(diffs)]

    if len(diffs) < 10:
        return {
            "H_a": H_a, "H_b": H_b, "diff": diff_point,
            "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
        }

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    # Two-sided p-value: is the difference significantly different from 0?
    diffs_centered = diffs - np.mean(diffs)
    p_value = float(np.mean(np.abs(diffs_centered) >= abs(diff_point)))

    return {
        "H_a": float(H_a),
        "H_b": float(H_b),
        "diff": float(diff_point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }
