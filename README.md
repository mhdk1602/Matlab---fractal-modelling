# Fractal Price-Volume Correlation

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fractal-pv.streamlit.app)

Fractal analysis of the relationship between stock price dynamics and trading volume using Hurst exponents and multifractal detrended fluctuation analysis (MFDFA).

**[Launch the interactive explorer →](https://fractal-pv.streamlit.app)**

## The Question

Do stocks with more persistent price behavior (higher Hurst exponent) exhibit systematically different trading volume patterns? And does the multifractal complexity of price series correlate with volume dynamics?

The academic literature establishes that volume has strong long memory (H ≈ 0.7–0.9) while returns are approximately uncorrelated (H ≈ 0.5) in efficient markets. This project investigates the cross-correlation structure between these two Hurst regimes and whether rolling co-movement reveals regime-dependent behavior.

## Methods

- **Hurst estimation**: Rescaled range (R/S), Detrended Fluctuation Analysis (DFA), and Multifractal DFA via `nolds` and `MFDFA`
- **Statistical rigor**: Block bootstrap confidence intervals, ADF/KPSS stationarity tests, Benjamini-Hochberg correction for multiple comparisons
- **Data**: Daily OHLCV from Yahoo Finance (cached locally as parquet) for ~50 S&P 500 constituents
- **Validation**: All estimators tested against synthetic fractional Brownian motion with known H before applying to market data

## Setup

```bash
# Install
pip install -e ".[dev,test]"

# Fetch data
python -c "from fractal_pv.data import fetch_universe, SP500_SAMPLE; fetch_universe(SP500_SAMPLE)"

# Run tests
pytest
```

## Interactive Dashboard

The [Streamlit app](https://fractal-pv.streamlit.app) lets you explore fractal dynamics for any ticker:

- **Price & Volume** — time series with full-sample Hurst estimates
- **Rolling Hurst** — H(|returns|) and H(volume) over time with spread analysis
- **Cross-Correlation** — scatter of rolling Hurst values with Pearson/Spearman stats
- **Method Comparison** — DFA vs R/S with literature reference values

No installation required. Just open the link.

## Project Structure

```
app.py                 # Streamlit dashboard
src/fractal_pv/
  data.py            # Data fetching + parquet caching
  hurst.py           # Hurst estimation wrappers + CI    (Phase 2)
  mfdfa.py           # Multifractal analysis             (Phase 4)
  stationarity.py    # ADF, KPSS, preprocessing          (Phase 2)
  rolling.py         # Rolling window Hurst              (Phase 3)
  correlation.py     # Cross-correlation analysis        (Phase 3)
  viz.py             # Publication-ready plots            (Phase 5)
notebooks/           # Analysis notebooks (01-04)
tests/               # Validation against synthetic fBm
data/raw/            # Cached parquet files (gitignored)
figures/             # Generated plots
legacy/              # Original MATLAB code (2014, historical reference)
```

## Key References

- Mandelbrot, B. (1963). *The Variation of Certain Speculative Prices.* Journal of Business, 36(4), 394–419.
- Lo, A.W. (1991). *Long-term Memory in Stock Market Prices.* Econometrica, 59(5), 1279–1313.
- Peng, C.K., et al. (1994). *Mosaic Organization of DNA Nucleotide Sequences.* Physical Review E, 49(2), 1685.
- Di Matteo, T., Aste, T., & Dacorogna, M.M. (2005). *Long-term Memories of Developed and Emerging Markets.* European Physical Journal B, 46(2), 309–317.
- Kantelhardt, J.W., et al. (2002). *Multifractal Detrended Fluctuation Analysis of Nonstationary Time Series.* Physica A, 316(1–4), 87–114.
- Plerou, V., Gopikrishnan, P., & Stanley, H.E. (2003). *Two-phase Behaviour of Financial Markets.* Nature, 421, 130.
- Lobato, I.N. & Velasco, C. (2000). *Long Memory in Stock-Market Trading Volume.* JBES, 18(4), 410–427.

## History

This project started as a MATLAB prototype in 2014 (see `legacy/`). The original code explored the same hypothesis but used a now-defunct Yahoo Finance API and had several implementation bugs. This Python rewrite modernizes the methodology, adds statistical rigor, and makes the analysis reproducible.

## License

MIT
