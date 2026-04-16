# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-notebook quantitative finance analysis (`stockanalysis.ipynb`) that examines historical stock returns for AAPL, GOOGL, MSFT, and SPY from 2005-2025 using daily data from Yahoo Finance.

## Running the Notebook

Requires Python 3 with Jupyter. Run cells sequentially — later cells depend on variables computed in earlier ones (especially `data`, `log_returns`, `simple_returns`, `tickers`, `colors`).

```bash
jupyter notebook stockanalysis.ipynb
```

### Dependencies (no requirements.txt exists)

```
yfinance pandas numpy matplotlib seaborn scipy statsmodels
```

Install with: `pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels`

## Notebook Architecture

The notebook follows a sequential analytical pipeline where each stage builds on the previous:

1. **Data acquisition** (cell 3): Downloads daily closing prices via `yf.download()` into `data` DataFrame
2. **Return computation** (cell 4): Computes `simple_returns` (pct_change) and `log_returns` (log ratio), plus annualized metrics
3. **Rolling volatility** (cell 6): 30-day and 252-day rolling windows on log returns, annualized by `* sqrt(252)`
4. **Cumulative returns** (cell 7): `np.exp(log_returns.cumsum())` — growth of $1 invested
5. **Distribution analysis** (cell 8): Fits normal and t-distributions to log returns; annotates skewness/kurtosis
6. **Monthly heatmaps** (cell 9): Resamples to monthly, pivots to year x month grid
7. **Crisis regime classification** (cells 10-11): Labels months as `crisis_2008`, `crisis_covid`, or `non_crisis`; adds `is_crisis` binary and `crisis_identity` categorical columns to `monthly_log` DataFrame

## Key Conventions

- All statistical analysis uses **log returns**, not simple returns
- Annualization factor: **252 trading days** (multiply mean by 252, std by sqrt(252))
- Monthly resampling uses `'MS'` (month start) for crisis analysis, `'ME'` (month end) for heatmaps
- Visualization uses a `colors` list and `tickers` list that must stay in sync
- Cell 5 displays `vol_30` but defines it in cell 6 — cells 5 and 6 must both be present
