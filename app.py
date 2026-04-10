"""Fractal Price-Volume Explorer — Interactive Streamlit Dashboard.

Visualize Hurst exponents, rolling fractal dynamics, and price-volume
cross-correlation for any stock ticker.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fractal Price-Volume Explorer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2015-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Hurst Estimation")
window = st.sidebar.slider("Rolling window (trading days)", 200, 1000, 500, step=50)
step = st.sidebar.slider("Step size (trading days)", 5, 50, 20, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown(
    "Fractal analysis of stock price-volume dynamics using Hurst exponents. "
    "Built on [nolds](https://github.com/CSchoel/nolds) for DFA estimation. "
    "Source: [GitHub](https://github.com/mhdk1602/Matlab---fractal-modelling)"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


df = load_data(ticker, str(start_date), str(end_date))

if df.empty:
    st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
    st.stop()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
close = df["Close"].dropna().values.flatten()
volume = df["Volume"].dropna().values.flatten()

log_returns = np.diff(np.log(close))
abs_log_returns = np.abs(log_returns)
log_volume = np.log(volume[1:] + 1)  # +1 to avoid log(0)

dates = df.index[1:]  # align with returns

# ---------------------------------------------------------------------------
# Hurst estimation
# ---------------------------------------------------------------------------


def estimate_hurst_dfa(series: np.ndarray) -> float:
    """Estimate Hurst exponent via DFA using nolds."""
    try:
        import nolds
        return float(nolds.dfa(series))
    except Exception:
        return np.nan


def estimate_hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent via R/S analysis."""
    try:
        import nolds
        return float(nolds.hurst_rs(series))
    except Exception:
        return np.nan


@st.cache_data(show_spinner="Computing rolling Hurst exponents...")
def rolling_hurst(
    _series: np.ndarray, _window: int, _step: int, _dates: np.ndarray
) -> pd.DataFrame:
    results = []
    for i in range(0, len(_series) - _window, _step):
        segment = _series[i : i + _window]
        h = estimate_hurst_dfa(segment)
        mid_idx = i + _window // 2
        if mid_idx < len(_dates):
            results.append({"date": _dates[mid_idx], "H": h})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"📈 Fractal Price-Volume Explorer — {ticker}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Points", f"{len(close):,}")
col2.metric("Date Range", f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")

h_returns = estimate_hurst_dfa(abs_log_returns)
h_volume = estimate_hurst_dfa(log_volume)
col3.metric("H(|returns|) DFA", f"{h_returns:.3f}" if not np.isnan(h_returns) else "N/A")
col4.metric("H(log volume) DFA", f"{h_volume:.3f}" if not np.isnan(h_volume) else "N/A")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Price & Volume", "Rolling Hurst", "Cross-Correlation", "Method Comparison"]
)

# --- Tab 1: Price & Volume ---
with tab1:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#0e1117")

    axes[0].plot(df.index, close, color="#e94560", linewidth=0.8)
    axes[0].set_ylabel("Price", color="white")
    axes[0].set_facecolor("#0e1117")
    axes[0].tick_params(colors="white")
    axes[0].spines["bottom"].set_color("#333")
    axes[0].spines["left"].set_color("#333")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(df.index, volume, color="#1a1a2e", width=1.5, alpha=0.7)
    axes[1].set_ylabel("Volume", color="white")
    axes[1].set_facecolor("#0e1117")
    axes[1].tick_params(colors="white")
    axes[1].spines["bottom"].set_color("#333")
    axes[1].spines["left"].set_color("#333")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"""
        **Interpretation:** H(|returns|) = **{h_returns:.3f}** — {"persistent (trending)" if h_returns > 0.55 else "anti-persistent (mean-reverting)" if h_returns < 0.45 else "approximately random walk"}.
        H(log volume) = **{h_volume:.3f}** — {"strong long memory (expected per Lobato & Velasco 2000)" if h_volume > 0.65 else "moderate persistence" if h_volume > 0.55 else "surprisingly low persistence"}.
        """
    )

# --- Tab 2: Rolling Hurst ---
with tab2:
    if len(abs_log_returns) < window:
        st.warning(f"Need at least {window} data points for rolling analysis. Have {len(abs_log_returns)}.")
    else:
        roll_returns = rolling_hurst(abs_log_returns, window, step, dates.values)
        roll_volume = rolling_hurst(log_volume, window, step, dates.values)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.patch.set_facecolor("#0e1117")

        for ax in axes:
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#333")
            ax.spines["left"].set_color("#333")

        axes[0].plot(df.index, close, color="#e94560", linewidth=0.8)
        axes[0].set_ylabel("Price", color="white")

        if not roll_returns.empty:
            axes[1].plot(roll_returns["date"], roll_returns["H"], color="#e94560", linewidth=1.2, label="H(|returns|)")
        if not roll_volume.empty:
            axes[1].plot(roll_volume["date"], roll_volume["H"], color="#4ecdc4", linewidth=1.2, label="H(log volume)")
        axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="H=0.5 (random walk)")
        axes[1].set_ylabel("Hurst Exponent", color="white")
        axes[1].legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        axes[1].set_ylim(0.2, 1.0)

        if not roll_returns.empty and not roll_volume.empty:
            merged = roll_returns.merge(roll_volume, on="date", suffixes=("_ret", "_vol"))
            axes[2].scatter(merged["date"], merged["H_vol"] - merged["H_ret"],
                           color="#ffd93d", s=8, alpha=0.6)
            axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            axes[2].set_ylabel("H(volume) − H(|returns|)", color="white")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            """
            **What to look for:** Volume Hurst (teal) typically sits well above the 0.5 line,
            reflecting strong long memory in trading activity. Volatility Hurst (red) fluctuates
            closer to 0.5 but may show persistent regimes during market stress.
            The spread (yellow dots) reveals when price-volume fractal coupling tightens or diverges.
            """
        )

# --- Tab 3: Cross-Correlation ---
with tab3:
    if len(abs_log_returns) < window:
        st.warning(f"Need at least {window} data points. Have {len(abs_log_returns)}.")
    else:
        roll_returns = rolling_hurst(abs_log_returns, window, step, dates.values)
        roll_volume = rolling_hurst(log_volume, window, step, dates.values)

        if not roll_returns.empty and not roll_volume.empty:
            merged = roll_returns.merge(roll_volume, on="date", suffixes=("_ret", "_vol"))
            valid = merged.dropna()

            if len(valid) > 5:
                from scipy import stats

                pearson_r, pearson_p = stats.pearsonr(valid["H_ret"], valid["H_vol"])
                spearman_r, spearman_p = stats.spearmanr(valid["H_ret"], valid["H_vol"])

                col1, col2 = st.columns(2)
                col1.metric("Pearson r", f"{pearson_r:.3f}", f"p = {pearson_p:.4f}")
                col2.metric("Spearman ρ", f"{spearman_r:.3f}", f"p = {spearman_p:.4f}")

                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")

                scatter = ax.scatter(
                    valid["H_ret"], valid["H_vol"],
                    c=range(len(valid)), cmap="plasma", s=30, alpha=0.7, edgecolors="none"
                )
                plt.colorbar(scatter, ax=ax, label="Time →")

                z = np.polyfit(valid["H_ret"], valid["H_vol"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid["H_ret"].min(), valid["H_ret"].max(), 100)
                ax.plot(x_line, p(x_line), color="#e94560", linewidth=2, linestyle="--")

                ax.set_xlabel("H(|returns|)", color="white")
                ax.set_ylabel("H(log volume)", color="white")
                ax.tick_params(colors="white")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#333")
                ax.spines["left"].set_color("#333")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown(
                    f"""
                    **Interpretation:** The scatter shows how H(|returns|) and H(log volume) co-move over time
                    (color = time progression, dark → light). Pearson r = **{pearson_r:.3f}** (p = {pearson_p:.4f}).
                    {"Significant positive correlation — fractal properties of price and volume are coupled." if pearson_p < 0.05 and pearson_r > 0 else "Significant negative correlation — price and volume fractality diverge." if pearson_p < 0.05 and pearson_r < 0 else "No significant linear correlation detected at this window size."}
                    """
                )
            else:
                st.warning("Not enough rolling windows to compute correlation.")

# --- Tab 4: Method Comparison ---
with tab4:
    st.subheader("Hurst Estimation: DFA vs R/S")

    methods_data = []
    for label, series in [("Abs Log Returns", abs_log_returns), ("Log Volume", log_volume)]:
        h_dfa = estimate_hurst_dfa(series)
        h_rs = estimate_hurst_rs(series)
        methods_data.append({"Series": label, "DFA": f"{h_dfa:.4f}", "R/S": f"{h_rs:.4f}",
                             "Difference": f"{abs(h_dfa - h_rs):.4f}"})

    st.table(pd.DataFrame(methods_data))

    st.markdown(
        """
        **Notes:**
        - **DFA** (Detrended Fluctuation Analysis) handles nonstationarity via polynomial detrending. Preferred for financial data.
        - **R/S** (Rescaled Range) is the classical method but overestimates H for series with short-range dependence (Lo 1991).
        - Large disagreement between methods suggests the series has features (trends, structural breaks) that affect one estimator more than the other.
        """
    )

    st.markdown("---")
    st.subheader("Expected Values from Literature")
    st.table(pd.DataFrame([
        {"Measure": "H(raw returns)", "Expected": "≈ 0.5", "Interpretation": "Efficient market (random walk)", "Source": "Lo 1991"},
        {"Measure": "H(|returns| / volatility)", "Expected": "0.7 – 0.8", "Interpretation": "Long memory in volatility", "Source": "Bollerslev & Jubinski 1999"},
        {"Measure": "H(volume)", "Expected": "0.7 – 0.9", "Interpretation": "Very strong persistence", "Source": "Lobato & Velasco 2000"},
    ]))
