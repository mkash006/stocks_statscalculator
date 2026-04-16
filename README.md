# plot_metrics.py

A CLI tool that fetches stock price data and generates professional financial metric visualisations.

## Quick start

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy
python plot_metrics.py
```

This runs with defaults: tickers AAPL, MSFT, GOOGL, SPY from 2005 to 2025, displaying 8 interactive plot windows one after another.

## CLI options

```
--tickers TSLA NVDA AMD       # any number of Yahoo Finance symbols
--start 2015-01-01            # start date (YYYY-MM-DD)
--end 2025-12-31              # end date (YYYY-MM-DD)
--risk-free 0.05              # annualised risk-free rate for Sharpe/Sortino
--no-show                     # suppress plot windows (headless / CI use)
--save-dir ./charts           # write PNGs to this folder (created if missing)
```

Options combine freely. For example, to save charts for two tickers without opening any windows:

```bash
python plot_metrics.py --tickers TSLA NVDA --start 2018-01-01 --no-show --save-dir ./charts
```

## How the script works

The script is organised into three layers that run top to bottom: data, metrics, and plots.

### Data layer

`fetch_prices` pulls daily adjusted-close prices from Yahoo Finance using `yf.download`. If only one ticker is requested, the result comes back as a pandas Series, so the function wraps it into a single-column DataFrame to keep the downstream code consistent.

`compute_returns` takes that price DataFrame and produces two return DataFrames. Simple returns are the standard percent change day over day. Log returns are computed as `ln(price_today / price_yesterday)`. Every metric in the script uses log returns because they are additive over time (you can sum daily log returns to get a monthly return), which makes rolling windows and cumulative calculations correct without extra compounding logic.

### Metric layer

Each function takes the log return (or price) DataFrame and returns a new DataFrame of the same shape.

**Rolling volatility** computes the standard deviation of log returns over a sliding window, then multiplies by `sqrt(252)` to annualise. The 252 comes from the approximate number of trading days in a year. Two windows are computed — 30-day (short-term, noisy) and 252-day (full-year, smoother trend).

**Cumulative returns** exponentiates the running sum of log returns (`exp(cumsum(log_returns))`). This gives the growth-of-$1 curve. Exponentiation is needed because log returns live in log-space; `exp` brings them back to dollar values.

**Rolling Sharpe** subtracts a daily risk-free rate from log returns (the annual rate divided by 252), averages excess returns over the window, annualises that average, and divides by annualised volatility. The result is a unitless ratio: return-per-unit-of-risk.

**Rolling Sortino** works like Sharpe but the denominator uses only negative returns (downside deviation) instead of total standard deviation. The inner function `_sortino_slice` filters to negative observations, computes their standard deviation, and annualises. This is applied via `rolling().apply()` so it slides across the full time series.

**Maximum drawdown** first builds the cumulative return curve, then tracks the running maximum at each point. Drawdown is `(current - peak) / peak`, so it's always zero or negative. This tells you the worst loss from any prior high.

**Rolling beta** computes, for each stock, the rolling covariance of its returns with the benchmark (SPY by default) divided by the rolling variance of the benchmark. Beta = 1 means the stock moves in lockstep with the market; beta > 1 means it amplifies market moves.

**RSI** (Relative Strength Index) uses Wilder's exponential moving average to separately smooth daily gains and daily losses over a 14-day window. The ratio of average gain to average loss is converted to a 0–100 scale. Below 30 is considered oversold, above 70 overbought.

### Plot layer

Each plot function follows the same pattern: compute the metric, create a matplotlib figure, iterate over tickers, style with `_apply_style`, and hand off to `_save_or_show`. The helper `_save_or_show` checks whether to write a PNG, display interactively, or both.

All plots use a 5-colour IBM Design Library palette chosen for colourblind accessibility. The colour map is built by `_ticker_colors`, which cycles through the palette if there are more than 5 tickers.

Subplot-based plots (log returns, volatility, RSI) stack one row per ticker with a shared x-axis. Single-panel plots (cumulative returns, Sharpe, Sortino, drawdown, beta) overlay all tickers on one axis.

### CLI entry point

`build_parser` defines the argument schema. `main` orchestrates the pipeline: parse arguments, fetch data, compute returns, assign colours, call each plot function. The `if __name__ == "__main__"` guard means running the file directly triggers `main()`, but you can also `import plot_metrics` and call any function individually from another script or notebook.

## Using as a library

Since every function is a standalone callable, you can import them into a notebook or another script:

```python
from plot_metrics import fetch_prices, compute_returns, rolling_sharpe

prices = fetch_prices(["AAPL"], "2020-01-01", "2025-01-01")
_, log_ret = compute_returns(prices)
sharpe = rolling_sharpe(log_ret, window=126)  # 6-month rolling
```

## Dependencies

| Package    | Role                                    |
|------------|-----------------------------------------|
| yfinance   | Yahoo Finance price data                |
| pandas     | DataFrames, rolling windows, resampling |
| numpy      | Log/exp math, sqrt for annualisation    |
| matplotlib | All chart rendering                     |
| seaborn    | Imported for style (used in notebook)   |
| scipy      | stats module (used in notebook, available here for future distribution fits) |
