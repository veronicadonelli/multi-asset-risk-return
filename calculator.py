"""
Financial metrics calculation engine.

All metrics are computed from first principles using standard financial theory.
Every function is pure (no side effects), fully typed, and independently testable.

Conventions
-----------
- Log returns are used throughout (time-additive, approximately normal).
- Annualisation uses the 252 trading-day convention.
- Volatility scaling follows square-root-of-time.
- Risk-free rate is configurable; defaults to 0.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AssetMetrics:
    """
    Full risk/return profile for a single asset.
    All return/volatility/drawdown fields are expressed as decimals (not %).
    Ratios are dimensionless.
    """
    ticker: str

    # Returns
    cumulative_return: float = float("nan")
    annualized_return: float = float("nan")

    # Risk
    annualized_volatility: float = float("nan")
    downside_volatility: float = float("nan")

    # Ratios
    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    calmar_ratio: float = float("nan")

    # Market sensitivity
    beta: float = float("nan")
    alpha_annualized: float = float("nan")
    r_squared: float = float("nan")

    # Tail risk
    var_95: float = float("nan")     # daily, historical
    cvar_95: float = float("nan")    # daily, historical (Expected Shortfall)

    # Drawdown
    max_drawdown: float = float("nan")

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if k != "ticker"}

    def to_display_dict(self) -> dict[str, str]:
        """Human-readable formatting for reporting."""
        pct = {
            "cumulative_return", "annualized_return", "annualized_volatility",
            "downside_volatility", "var_95", "cvar_95", "max_drawdown",
            "alpha_annualized",
        }
        out: dict[str, str] = {}
        for k, v in self.__dict__.items():
            if k == "ticker":
                continue
            if np.isnan(v):
                out[k] = "N/A"
            elif k in pct:
                out[k] = f"{v:.2%}"
            else:
                out[k] = f"{v:.4f}"
        return out


# ---------------------------------------------------------------------------
# Core calculation functions (pure, independently testable)
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns from a price series.

    Formula: r_t = ln(P_t / P_{t-1})

    Log returns are preferred over simple returns because they are:
    - Time-additive (multi-period return = sum of daily returns)
    - Approximately normally distributed
    - Symmetric around zero for equal up/down moves
    """
    if prices.empty:
        raise ValueError("Price series is empty.")
    if (prices <= 0).any():
        raise ValueError("Price series contains non-positive values.")
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(log_returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """
    Mean daily log return scaled to annual frequency.

    Note: this is the arithmetic mean of log returns × 252, which approximates
    the continuously compounded annual return. For the geometric (CAGR)
    equivalent, use: exp(mean * 252) - 1.
    """
    if log_returns.empty:
        return float("nan")
    return float(log_returns.mean() * trading_days)


def annualized_volatility(log_returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """
    Standard deviation of daily log returns scaled by sqrt(252).

    Uses ddof=1 (sample standard deviation) throughout.
    """
    if len(log_returns) < 2:
        return float("nan")
    return float(log_returns.std(ddof=1) * np.sqrt(trading_days))


def downside_volatility(
    log_returns: pd.Series,
    threshold: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Annualised volatility of returns below `threshold`.

    Used in the Sortino Ratio denominator. Only penalises negative deviations,
    unlike total volatility which penalises both up and down moves equally.
    """
    below = log_returns[log_returns < threshold]
    if len(below) < 2:
        return float("nan")
    return float(below.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    ann_return: float,
    ann_vol: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Excess return per unit of total risk.

    Sharpe = (R_p - R_f) / σ_p

    A ratio > 1.0 is generally considered acceptable.
    A ratio > 2.0 is considered excellent.
    """
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return float((ann_return - risk_free_rate) / ann_vol)


def sortino_ratio(
    ann_return: float,
    ann_down_vol: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Excess return per unit of downside risk.

    Sortino = (R_p - R_f) / σ_down

    More appropriate than Sharpe for asymmetric return distributions
    (e.g. options, trend-following strategies) because it does not
    penalise upside volatility.
    """
    if ann_down_vol == 0 or np.isnan(ann_down_vol):
        return float("nan")
    return float((ann_return - risk_free_rate) / ann_down_vol)


def beta_alpha_r2(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float, float]:
    """
    OLS regression of asset returns on benchmark returns.

    Returns
    -------
    beta  : systematic risk coefficient (slope)
    alpha : annualised Jensen's alpha (intercept × 252)
    r2    : coefficient of determination

    Beta interpretation:
    - β > 1  : amplifies benchmark moves (e.g. growth stocks)
    - β ≈ 1  : moves in line with the market
    - 0 < β < 1 : dampens benchmark moves (defensive assets)
    - β < 0  : inverse relationship (e.g. inverse ETFs, short positions)
    """
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan"), float("nan"), float("nan")

    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values

    slope, intercept, r_val, p_val, se = stats.linregress(x, y)

    beta = float(slope)
    alpha = float(intercept * TRADING_DAYS)   # annualise daily intercept
    r2 = float(r_val ** 2)

    logger.debug(
        "OLS %s: β=%.3f, α_ann=%.4f, R²=%.3f, p=%.4f",
        asset_returns.name, beta, alpha, r2, p_val,
    )
    return beta, alpha, r2


def historical_var(log_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Non-parametric Value-at-Risk at `confidence` level (daily).

    Historical simulation makes no distributional assumption —
    preferred over Gaussian VaR for fat-tailed assets (commodities, crypto).

    Returns a negative number (loss convention).
    """
    if log_returns.empty:
        return float("nan")
    return float(np.percentile(log_returns, (1 - confidence) * 100))


def historical_cvar(log_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) at `confidence` level (daily).

    CVaR = mean return in the worst (1-confidence)% of days.
    Provides a more complete picture of tail risk than VaR alone,
    as it captures the severity of losses beyond the VaR threshold.

    Returns a negative number (loss convention).
    """
    var = historical_var(log_returns, confidence)
    tail = log_returns[log_returns <= var]
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def maximum_drawdown(prices: pd.Series) -> float:
    """
    Maximum peak-to-trough percentage decline in price.

    MDD = min( (P_t - max(P_0..P_t)) / max(P_0..P_t) )

    Returns a negative number (e.g. -0.35 means -35% drawdown).
    Critical metric for risk-averse investors and strategy evaluation.
    """
    if prices.empty:
        return float("nan")
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    """
    Annualised return divided by absolute maximum drawdown.

    Calmar = R_ann / |MDD|

    Higher is better. Particularly relevant for strategies where
    drawdown risk is the primary concern (e.g. wealth management, retirees).
    """
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(ann_return / abs(max_dd))


def cumulative_return(prices: pd.Series) -> float:
    """
    Total price appreciation from first to last observation.

    Returns decimal (e.g. 0.45 = +45%).
    """
    if prices.empty or prices.iloc[0] == 0:
        return float("nan")
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Full time series of drawdown values (for plotting).

    Returns a Series of the same length as `prices`, values in [-1, 0].
    """
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix of log returns.

    Values in [-1, 1]. Values close to 0 indicate diversification potential.
    """
    return log_returns.corr(method="pearson")


def portfolio_metrics_table(
    metrics_list: list[AssetMetrics],
) -> pd.DataFrame:
    """
    Combine per-asset metrics into a single summary DataFrame.

    Index   : ticker
    Columns : all numeric metric fields
    """
    rows = {m.ticker: m.to_dict() for m in metrics_list}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

class RiskReturnCalculator:
    """
    Computes the complete risk/return metric set for every asset in a universe.

    Usage
    -----
    >>> calc = RiskReturnCalculator(prices, benchmark="^GSPC", risk_free_rate=0.04)
    >>> results = calc.compute_all()
    >>> df = portfolio_metrics_table(results)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        benchmark: str,
        risk_free_rate: float = 0.0,
        confidence: float = 0.95,
        trading_days: int = TRADING_DAYS,
    ) -> None:
        if benchmark not in prices.columns:
            raise ValueError(f"Benchmark '{benchmark}' not found in prices columns.")

        self.prices = prices
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.confidence = confidence
        self.trading_days = trading_days

        # Compute log returns once; reuse across all assets
        self.log_returns: pd.DataFrame = prices.apply(
            lambda col: compute_log_returns(col)
        )
        self.benchmark_returns: pd.Series = self.log_returns[benchmark]

    # ------------------------------------------------------------------ #
    def compute_all(self) -> list[AssetMetrics]:
        """Compute metrics for every ticker in the price universe."""
        results = []
        for ticker in self.prices.columns:
            try:
                results.append(self._compute_single(ticker))
            except Exception as exc:
                logger.error("Failed to compute metrics for %s: %s", ticker, exc)
                results.append(AssetMetrics(ticker=ticker))  # all NaN
        return results

    # ------------------------------------------------------------------ #
    def _compute_single(self, ticker: str) -> AssetMetrics:
        r = self.log_returns[ticker].dropna()
        p = self.prices[ticker].dropna()

        ann_ret = annualized_return(r, self.trading_days)
        ann_vol = annualized_volatility(r, self.trading_days)
        down_vol = downside_volatility(r, trading_days=self.trading_days)

        b, a, r2 = beta_alpha_r2(r, self.benchmark_returns)

        var = historical_var(r, self.confidence)
        cvar = historical_cvar(r, self.confidence)

        mdd = maximum_drawdown(p)

        return AssetMetrics(
            ticker=ticker,
            cumulative_return=cumulative_return(p),
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            downside_volatility=down_vol,
            sharpe_ratio=sharpe_ratio(ann_ret, ann_vol, self.risk_free_rate),
            sortino_ratio=sortino_ratio(ann_ret, down_vol, self.risk_free_rate),
            calmar_ratio=calmar_ratio(ann_ret, mdd),
            beta=b,
            alpha_annualized=a,
            r_squared=r2,
            var_95=var,
            cvar_95=cvar,
            max_drawdown=mdd,
        )

    # ------------------------------------------------------------------ #
    def correlation_matrix(self) -> pd.DataFrame:
        """Return the Pearson correlation matrix for the full universe."""
        return correlation_matrix(self.log_returns)

    def summary_table(self) -> pd.DataFrame:
        """Convenience method: compute all metrics and return as DataFrame."""
        return portfolio_metrics_table(self.compute_all())
