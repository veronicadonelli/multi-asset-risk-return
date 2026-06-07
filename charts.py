"""
Visualization layer.

Produces 8 publication-quality charts for the multi-asset risk/return report.
All chart functions are pure: they accept data and return a Figure object
without side effects (no plt.show(), no file writes).
The caller decides how to display or save each figure.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from calculator import AssetMetrics, drawdown_series

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

STYLE: dict = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f9fc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#0d1b2a",
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}

PALETTE = [
    "#2196F3", "#e74c3c", "#27ae60", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#e91e63",
]


def _build_color_map(tickers: list[str]) -> dict[str, str]:
    return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(tickers)}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Chart 1 — Normalised cumulative returns
# ---------------------------------------------------------------------------

def chart_cumulative_returns(
    prices: pd.DataFrame,
    benchmark: str,
    title: str = "Normalized Cumulative Returns (Base = 100)",
) -> Figure:
    """
    All assets rebased to 100 at inception for direct visual comparison.
    Benchmark is rendered as a dashed line with higher linewidth.
    """
    _apply_style()
    tickers = list(prices.columns)
    colors = _build_color_map(tickers)

    norm = prices / prices.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    for t in tickers:
        lw = 2.5 if t == benchmark else 1.4
        ls = "--" if t == benchmark else "-"
        ax.plot(prices.index, norm[t], label=t, color=colors[t], lw=lw, ls=ls)

    ax.axhline(100, color="grey", lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Index Value")
    ax.legend(ncol=3, fontsize=8.5, framealpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Risk–Return scatter
# ---------------------------------------------------------------------------

def chart_risk_return_scatter(
    metrics: list[AssetMetrics],
    benchmark: str,
    title: str = "Risk–Return Profile (Annualized)",
) -> Figure:
    """
    Annualised volatility on X-axis vs. annualised return on Y-axis.
    Top-left quadrant (high return, low risk) is the efficient frontier ideal.
    """
    _apply_style()
    tickers = [m.ticker for m in metrics]
    colors = _build_color_map(tickers)

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in metrics:
        x = m.annualized_volatility * 100
        y = m.annualized_return * 100
        ax.scatter(x, y, color=colors[m.ticker], s=110, zorder=3)
        ax.annotate(
            m.ticker, (x, y),
            textcoords="offset points", xytext=(7, 4),
            fontsize=9, color=colors[m.ticker], fontweight="bold",
        )

    bm = next(m for m in metrics if m.ticker == benchmark)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(bm.annualized_volatility * 100, color="grey", lw=0.8, ls="--", alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Beta & R²
# ---------------------------------------------------------------------------

def chart_beta_r2(
    metrics: list[AssetMetrics],
    benchmark: str,
    title: str = "Beta & R² vs Benchmark",
) -> Figure:
    """
    Dual-panel bar chart.
    Beta panel: green for β < 1 (defensive), red for β > 1 (aggressive).
    R² panel: proportion of variance explained by the benchmark.
    """
    _apply_style()
    assets = [m for m in metrics if m.ticker != benchmark]
    labels = [m.ticker for m in assets]
    betas = [m.beta for m in assets]
    rsqs = [m.r_squared for m in assets]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Beta
    bar_colors = ["#27ae60" if b < 1 else "#e74c3c" for b in betas]
    bars1 = ax1.bar(x, betas, color=bar_colors, edgecolor="white", width=0.6)
    ax1.axhline(1, color="#1b3a5c", lw=1.5, ls="--", label="β = 1 (market)")
    ax1.axhline(0, color="grey", lw=0.7, ls=":")
    for bar, val in zip(bars1, betas):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.2f}", ha="center", fontsize=8.5, fontweight="bold",
        )
    ax1.set_title("Beta vs S&P 500")
    ax1.set_ylabel("Beta")
    ax1.legend(fontsize=8)

    # R²
    bars2 = ax2.bar(x, rsqs, color="#2196F3", edgecolor="white", alpha=0.85, width=0.6)
    for bar, val in zip(bars2, rsqs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}", ha="center", fontsize=8.5, fontweight="bold",
        )
    ax2.set_title("R² — Variance Explained by Benchmark")
    ax2.set_ylabel("R²")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.1)

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#0d1b2a")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — VaR vs CVaR
# ---------------------------------------------------------------------------

def chart_var_cvar(
    metrics: list[AssetMetrics],
    title: str = "VaR vs CVaR at 95% Confidence (Daily)",
) -> Figure:
    """
    Grouped bar chart comparing VaR and CVaR for each asset.
    CVaR (Expected Shortfall) is always >= VaR in absolute terms and reveals
    the average severity of tail losses beyond the VaR threshold.
    """
    _apply_style()
    tickers = [m.ticker for m in metrics]
    var_vals = [m.var_95 * 100 for m in metrics]
    cvar_vals = [m.cvar_95 * 100 for m in metrics]
    x = np.arange(len(tickers))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, var_vals, width=w, label="VaR 95%",
           color="#e74c3c", alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, cvar_vals, width=w, label="CVaR 95% (ES)",
           color="#c0392b", alpha=0.70, edgecolor="white")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Daily Loss (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — Sharpe vs Sortino
# ---------------------------------------------------------------------------

def chart_sharpe_sortino(
    metrics: list[AssetMetrics],
    title: str = "Sharpe vs Sortino Ratio (Annualized, Rf = 0%)",
) -> Figure:
    """
    Grouped bar chart. The gap between Sortino and Sharpe reveals
    the degree of return asymmetry: a larger Sortino implies the asset
    has meaningful upside volatility not penalised by Sortino.
    """
    _apply_style()
    tickers = [m.ticker for m in metrics]
    sharpes = [m.sharpe_ratio for m in metrics]
    sortini = [m.sortino_ratio for m in metrics]
    x = np.arange(len(tickers))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, sharpes, width=w, label="Sharpe",
           color="#27ae60", alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, sortini, width=w, label="Sortino",
           color="#1abc9c", alpha=0.75, edgecolor="white")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.axhline(1, color="#f39c12", lw=1.3, ls="--", label="Threshold = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 6 — Drawdown analysis
# ---------------------------------------------------------------------------

def chart_drawdown(
    prices: pd.DataFrame,
    metrics: list[AssetMetrics],
    benchmark: str,
    title: str = "Drawdown Analysis",
) -> Figure:
    """
    Two-panel figure:
    - Top: continuous drawdown time series for all assets.
    - Bottom: maximum drawdown bar chart ranked by severity.
    """
    _apply_style()
    tickers = list(prices.columns)
    colors = _build_color_map(tickers)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    for t in tickers:
        dd = drawdown_series(prices[t]) * 100
        lw = 2.2 if t == benchmark else 1.2
        ax1.plot(prices.index, dd, label=t, color=colors[t], lw=lw)

    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_title("Drawdown Time Series (%)")
    ax1.set_ylabel("Drawdown (%)")
    ax1.legend(ncol=3, fontsize=7.5, framealpha=0.6)

    mdd_vals = [m.max_drawdown * 100 for m in metrics]
    bars = ax2.bar(tickers, mdd_vals,
                   color=[colors[t] for t in tickers], edgecolor="white")
    for bar, val in zip(bars, mdd_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val - 0.5,
            f"{val:.1f}%", ha="center", va="top",
            fontsize=8.5, color="white", fontweight="bold",
        )
    ax2.set_title("Maximum Drawdown by Asset (%)")
    ax2.set_ylabel("Max Drawdown (%)")

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#0d1b2a")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 7 — Correlation heatmap
# ---------------------------------------------------------------------------

def chart_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Pearson Correlation Matrix (Daily Log Returns)",
) -> Figure:
    """
    Diverging colour scale centred at zero.
    Green = positive correlation, Red = negative, White = uncorrelated.
    Near-zero values for gold vs. equities confirm diversification benefits.
    """
    _apply_style()
    tickers = list(corr_matrix.columns)

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr_matrix.values, cmap="RdYlGn", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")

    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tickers, fontsize=9)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr_matrix.values[i, j]
            tc = "white" if abs(val) > 0.6 else "#2c3e50"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=tc, fontweight="bold")

    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 8 — Return distributions
# ---------------------------------------------------------------------------

def chart_return_distributions(
    log_returns: pd.DataFrame,
    title: str = "Daily Log Return Distributions",
) -> Figure:
    """
    3×3 grid of histograms (one per asset).
    Each panel marks the mean (dashed black) and 5th percentile VaR (dotted red).
    Fat tails and asymmetry are immediately visible.
    """
    _apply_style()
    tickers = list(log_returns.columns)
    colors = _build_color_map(tickers)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, t in enumerate(tickers):
        r = log_returns[t].dropna() * 100
        axes[i].hist(r, bins=50, color=colors[t], alpha=0.82,
                     edgecolor="white", lw=0.3)
        axes[i].axvline(r.mean(), color="black", lw=1.5, ls="--",
                        label=f"μ = {r.mean():.2f}%")
        axes[i].axvline(np.percentile(r, 5), color="red", lw=1.2, ls=":",
                        label=f"VaR = {np.percentile(r, 5):.2f}%")
        axes[i].set_title(t, fontsize=10, fontweight="bold", color="#0d1b2a")
        axes[i].set_xlabel("Daily Log Return (%)", fontsize=7.5)
        axes[i].legend(fontsize=7, framealpha=0.5)

    # Hide unused subplots if tickers < 9
    for j in range(len(tickers), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold",
                 color="#0d1b2a", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all 8 charts at once
# ---------------------------------------------------------------------------

def generate_all_charts(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    metrics: list[AssetMetrics],
    corr_matrix: pd.DataFrame,
    benchmark: str,
) -> dict[str, Figure]:
    """
    Generate all 8 standard charts and return them as a named dict.

    Keys: chart_1 … chart_8
    """
    logger.info("Generating all charts …")
    return {
        "chart_1_cumulative_returns": chart_cumulative_returns(prices, benchmark),
        "chart_2_risk_return_scatter": chart_risk_return_scatter(metrics, benchmark),
        "chart_3_beta_r2": chart_beta_r2(metrics, benchmark),
        "chart_4_var_cvar": chart_var_cvar(metrics),
        "chart_5_sharpe_sortino": chart_sharpe_sortino(metrics),
        "chart_6_drawdown": chart_drawdown(prices, metrics, benchmark),
        "chart_7_correlation_heatmap": chart_correlation_heatmap(corr_matrix),
        "chart_8_return_distributions": chart_return_distributions(log_returns),
    }
