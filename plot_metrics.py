"""
plot_metrics.py
Produces key financial metric plots for a list of stock tickers.
Designed as a standalone CLI script so it slots cleanly into a git-based tool.

Usage:
    python plot_metrics.py                              # defaults: AAPL MSFT GOOGL SPY, 2005-2025
    python plot_metrics.py --tickers TSLA NVDA --start 2015-01-01 --end 2025-12-31
    python plot_metrics.py --no-show --save-dir ./charts  # save PNGs without opening windows
"""

import argparse
import sys
from pathlib import Path

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# colorblind-friendly palette (IBM Design Library order)
CB_PALETTE = ["#648FFF", "#DC267F", "#FE6100", "#FFB000", "#785EF0"]


# -- data layer ---------------------------------------------------------------

def fetch_prices(tickers, start, end):
    """Download adjusted close prices from Yahoo Finance."""
    df = yf.download(tickers, start=start, end=end)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df


def compute_returns(prices):
    """Daily simple and log returns from a price DataFrame."""
    simple = prices.pct_change().dropna()
    log = np.log(prices / prices.shift(1)).dropna()
    return simple, log


# -- metric computations ------------------------------------------------------

def rolling_volatility(log_returns, window=30, trading_days=252):
    """Annualised rolling standard deviation of log returns."""
    return log_returns.rolling(window).std() * np.sqrt(trading_days)


def cumulative_returns(log_returns):
    """Growth-of-$1 curve from cumulative log returns."""
    return np.exp(log_returns.cumsum())


def rolling_sharpe(log_returns, window=252, risk_free_annual=0.0, trading_days=252):
    """
    Rolling annualised Sharpe ratio.
    risk_free_annual is the annualised risk-free rate (e.g. 0.05 for 5%).
    """
    daily_rf = risk_free_annual / trading_days
    excess = log_returns - daily_rf
    roll_mean = excess.rolling(window).mean() * trading_days
    roll_std = log_returns.rolling(window).std() * np.sqrt(trading_days)
    return roll_mean / roll_std


def rolling_sortino(log_returns, window=252, risk_free_annual=0.0, trading_days=252):
    """
    Rolling Sortino ratio — like Sharpe but only penalises downside vol.
    Rewards strategies that have positive skew, which matters for dip-buying.
    """
    daily_rf = risk_free_annual / trading_days
    excess = log_returns - daily_rf

    def _sortino_slice(x):
        downside = x[x < 0]
        if len(downside) < 2:
            return np.nan
        down_std = downside.std() * np.sqrt(trading_days)
        ann_ret = x.mean() * trading_days
        return ann_ret / down_std if down_std > 0 else np.nan

    return log_returns.rolling(window).apply(_sortino_slice, raw=True)


def max_drawdown_series(log_returns):
    """
    Running maximum drawdown at every point in time.
    Shows the worst peak-to-trough decline experienced so far — essential
    for understanding the pain of holding through a dip.
    """
    cumret = np.exp(log_returns.cumsum())
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    return drawdown


def rolling_beta(log_returns, benchmark_col="SPY", window=252):
    """
    Rolling beta vs a benchmark (default SPY).
    Tells you how much a stock amplifies market moves — high-beta names
    dip harder but also recover faster if the thesis is right.
    """
    if benchmark_col not in log_returns.columns:
        return pd.DataFrame()

    bm = log_returns[benchmark_col]
    betas = {}
    for col in log_returns.columns:
        if col == benchmark_col:
            continue
        cov = log_returns[col].rolling(window).cov(bm)
        var = bm.rolling(window).var()
        betas[col] = cov / var
    return pd.DataFrame(betas, index=log_returns.index)


def rsi(prices, window=14):
    """
    Relative Strength Index (Wilder's smoothing).
    Classic momentum oscillator — values below 30 flag oversold conditions,
    which is the bread and butter of a buy-the-dip screen.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# -- plotting layer -----------------------------------------------------------

def _ticker_colors(tickers):
    """Map each ticker to a colorblind-safe color, cycling if needed."""
    return {t: CB_PALETTE[i % len(CB_PALETTE)] for i, t in enumerate(tickers)}


def _apply_style(ax, title="", ylabel=""):
    """Shared cosmetic tweaks — keeps every panel consistent."""
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, framealpha=0.8)


def plot_log_returns(log_returns, tickers, colors, show=True, save_dir=None):
    """Daily log return time series, one subplot per ticker."""
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Daily Log Returns", fontsize=14, weight="bold")

    for ax, t in zip(axes, tickers):
        ax.plot(log_returns[t], color=colors[t], alpha=0.6, linewidth=0.4, label=t)
        _apply_style(ax, title=t, ylabel="log return")

    plt.tight_layout()
    _save_or_show(fig, "log_returns", show, save_dir)


def plot_rolling_volatility(log_returns, tickers, colors,
                            short_window=30, long_window=252,
                            show=True, save_dir=None):
    """30-day and 252-day annualised volatility, stacked by ticker."""
    vol_short = rolling_volatility(log_returns, window=short_window)
    vol_long = rolling_volatility(log_returns, window=long_window)

    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Rolling Volatility — 30d vs 252d (annualised)", fontsize=14, weight="bold")

    for ax, t in zip(axes, tickers):
        ax.plot(vol_short[t], color=colors[t], alpha=0.7, linewidth=1, label="30d")
        ax.plot(vol_long[t], color="black", alpha=0.8, linewidth=1.4, label="252d")
        ax.fill_between(vol_short.index, vol_short[t], alpha=0.12, color=colors[t])
        _apply_style(ax, title=t, ylabel="annualised vol")

    plt.tight_layout()
    _save_or_show(fig, "rolling_volatility", show, save_dir)


def plot_cumulative_returns(log_returns, tickers, colors, show=True, save_dir=None):
    """Growth of $1 invested, all tickers on one panel."""
    cumret = cumulative_returns(log_returns)

    fig, ax = plt.subplots(figsize=(14, 6))
    for t in tickers:
        ax.plot(cumret[t], color=colors[t], linewidth=1.4, label=t)
        final = cumret[t].iloc[-1]
        ax.annotate(f"${final:.0f}", xy=(cumret.index[-1], final),
                    fontsize=9, color=colors[t],
                    xytext=(8, 0), textcoords="offset points")

    ax.axhline(1, color="black", linestyle="--", linewidth=0.7)
    _apply_style(ax, title="Cumulative Returns — Growth of $1", ylabel="value ($)")
    plt.tight_layout()
    _save_or_show(fig, "cumulative_returns", show, save_dir)


def plot_sharpe(log_returns, tickers, colors, window=252,
                risk_free=0.0, show=True, save_dir=None):
    """Rolling 1-year Sharpe ratio."""
    sharpe = rolling_sharpe(log_returns, window=window, risk_free_annual=risk_free)

    fig, ax = plt.subplots(figsize=(14, 5))
    for t in tickers:
        ax.plot(sharpe[t], color=colors[t], linewidth=1, alpha=0.8, label=t)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    _apply_style(ax, title=f"Rolling Sharpe Ratio ({window}d window)", ylabel="Sharpe")
    plt.tight_layout()
    _save_or_show(fig, "sharpe_ratio", show, save_dir)


def plot_sortino(log_returns, tickers, colors, window=252,
                 risk_free=0.0, show=True, save_dir=None):
    """Rolling 1-year Sortino ratio."""
    sortino = rolling_sortino(log_returns, window=window, risk_free_annual=risk_free)

    fig, ax = plt.subplots(figsize=(14, 5))
    for t in tickers:
        ax.plot(sortino[t], color=colors[t], linewidth=1, alpha=0.8, label=t)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    _apply_style(ax, title=f"Rolling Sortino Ratio ({window}d window)", ylabel="Sortino")
    plt.tight_layout()
    _save_or_show(fig, "sortino_ratio", show, save_dir)


def plot_drawdown(log_returns, tickers, colors, show=True, save_dir=None):
    """Maximum drawdown curve — how far each stock fell from its peak."""
    dd = max_drawdown_series(log_returns)

    fig, ax = plt.subplots(figsize=(14, 5))
    for t in tickers:
        ax.fill_between(dd.index, dd[t], alpha=0.3, color=colors[t], label=t)
        ax.plot(dd[t], color=colors[t], linewidth=0.6)
    ax.set_ylim(top=0)
    _apply_style(ax, title="Maximum Drawdown", ylabel="drawdown (%)")
    plt.tight_layout()
    _save_or_show(fig, "max_drawdown", show, save_dir)


def plot_beta(log_returns, tickers, colors, benchmark="SPY",
              window=252, show=True, save_dir=None):
    """Rolling beta vs the benchmark (SPY by default)."""
    betas = rolling_beta(log_returns, benchmark_col=benchmark, window=window)
    if betas.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    for t in betas.columns:
        ax.plot(betas[t], color=colors.get(t, "gray"), linewidth=1, alpha=0.8, label=t)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.7)
    _apply_style(ax, title=f"Rolling Beta vs {benchmark} ({window}d)", ylabel="beta")
    plt.tight_layout()
    _save_or_show(fig, "rolling_beta", show, save_dir)


def plot_rsi(prices, tickers, colors, window=14, show=True, save_dir=None):
    """RSI with standard overbought/oversold bands at 70/30."""
    rsi_df = rsi(prices, window=window)

    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"RSI ({window}-day)", fontsize=14, weight="bold")

    for ax, t in zip(axes, tickers):
        ax.plot(rsi_df[t], color=colors[t], linewidth=0.8)
        ax.axhline(70, color="gray", linestyle="--", linewidth=0.6)
        ax.axhline(30, color="gray", linestyle="--", linewidth=0.6)
        ax.fill_between(rsi_df.index, 30, rsi_df[t],
                        where=(rsi_df[t] < 30), alpha=0.25, color="green",
                        label="oversold (<30)")
        ax.fill_between(rsi_df.index, 70, rsi_df[t],
                        where=(rsi_df[t] > 70), alpha=0.25, color="red",
                        label="overbought (>70)")
        ax.set_ylim(0, 100)
        _apply_style(ax, title=t, ylabel="RSI")

    plt.tight_layout()
    _save_or_show(fig, "rsi", show, save_dir)


# -- utilities ----------------------------------------------------------------

def _save_or_show(fig, name, show, save_dir):
    """Either display the figure or write it to disk (or both)."""
    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -- CLI entry point ----------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Plot key financial metrics for a list of stock tickers."
    )
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "SPY"],
                   help="Space-separated ticker symbols (default: AAPL MSFT GOOGL SPY)")
    p.add_argument("--start", default="2005-01-01",
                   help="Start date YYYY-MM-DD (default: 2005-01-01)")
    p.add_argument("--end", default="2026-01-01",
                   help="End date YYYY-MM-DD (default: 2026-01-01)")
    p.add_argument("--risk-free", type=float, default=0.0,
                   help="Annualised risk-free rate for Sharpe/Sortino (default: 0.0)")
    p.add_argument("--no-show", action="store_true",
                   help="Suppress interactive plot windows (useful in CI / headless)")
    p.add_argument("--save-dir", default=None,
                   help="Directory to save PNG files (created if missing)")
    return p


def main(args=None):
    opts = build_parser().parse_args(args)
    tickers = opts.tickers
    show = not opts.no_show

    print(f"Fetching data for {tickers} from {opts.start} to {opts.end} ...")
    prices = fetch_prices(tickers, opts.start, opts.end)
    _, log_ret = compute_returns(prices)
    colors = _ticker_colors(tickers)

    print("Generating plots ...")
    plot_log_returns(log_ret, tickers, colors, show=show, save_dir=opts.save_dir)
    plot_rolling_volatility(log_ret, tickers, colors, show=show, save_dir=opts.save_dir)
    plot_cumulative_returns(log_ret, tickers, colors, show=show, save_dir=opts.save_dir)
    plot_sharpe(log_ret, tickers, colors, risk_free=opts.risk_free,
                show=show, save_dir=opts.save_dir)
    plot_sortino(log_ret, tickers, colors, risk_free=opts.risk_free,
                 show=show, save_dir=opts.save_dir)
    plot_drawdown(log_ret, tickers, colors, show=show, save_dir=opts.save_dir)
    plot_beta(log_ret, tickers, colors, show=show, save_dir=opts.save_dir)
    plot_rsi(prices, tickers, colors, show=show, save_dir=opts.save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
