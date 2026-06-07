"""
Unit tests for src/metrics/calculator.py

Every metric function is tested in isolation with known inputs
whose expected outputs can be verified by hand or with a financial calculator.

Run with:
    pytest tests/unit/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculator import (
    RiskReturnCalculator,
    annualized_return,
    annualized_volatility,
    beta_alpha_r2,
    calmar_ratio,
    compute_log_returns,
    cumulative_return,
    downside_volatility,
    historical_cvar,
    historical_var,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns() -> pd.Series:
    """Series of identical positive returns → deterministic outputs."""
    return pd.Series([0.001] * 500, name="FLAT")


@pytest.fixture
def mixed_returns() -> pd.Series:
    """Series with alternating positive/negative returns."""
    vals = [0.02, -0.01, 0.03, -0.02, 0.01, -0.005] * 100
    return pd.Series(vals, name="MIXED")


@pytest.fixture
def trending_prices() -> pd.Series:
    """Monotonically increasing prices → MDD = 0, cumret > 0."""
    return pd.Series(np.linspace(100, 200, 252), name="TREND")


@pytest.fixture
def crash_prices() -> pd.Series:
    """Prices that drop 50% then recover → known MDD."""
    up = np.linspace(100, 200, 126)
    down = np.linspace(200, 100, 126)
    return pd.Series(np.concatenate([up, down]), name="CRASH")


@pytest.fixture
def benchmark_returns(mixed_returns) -> pd.Series:
    """Shifted version of mixed_returns as a benchmark."""
    return mixed_returns.shift(1).dropna().rename("BM")


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_basic_output_length(self, trending_prices):
        r = compute_log_returns(trending_prices)
        assert len(r) == len(trending_prices) - 1

    def test_values_are_finite(self, trending_prices):
        r = compute_log_returns(trending_prices)
        assert np.isfinite(r).all()

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_log_returns(pd.Series([], dtype=float))

    def test_non_positive_prices_raise(self):
        prices = pd.Series([100, 0, 50])
        with pytest.raises(ValueError, match="non-positive"):
            compute_log_returns(prices)

    def test_known_value(self):
        prices = pd.Series([100.0, 110.0])
        r = compute_log_returns(prices)
        expected = np.log(110.0 / 100.0)
        assert pytest.approx(r.iloc[0], rel=1e-9) == expected


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_flat_returns(self, flat_returns):
        r = annualized_return(flat_returns)
        assert pytest.approx(r, rel=1e-6) == 0.001 * TRADING_DAYS

    def test_empty_returns(self):
        assert np.isnan(annualized_return(pd.Series([], dtype=float)))

    def test_sign_consistency(self, mixed_returns):
        r = annualized_return(mixed_returns)
        # mean of mixed series is positive → annualized should also be positive
        assert r > 0


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVolatility:
    def test_flat_returns_zero_vol(self, flat_returns):
        vol = annualized_volatility(flat_returns)
        assert pytest.approx(vol, abs=1e-9) == 0.0

    def test_output_positive(self, mixed_returns):
        assert annualized_volatility(mixed_returns) > 0

    def test_single_observation(self):
        assert np.isnan(annualized_volatility(pd.Series([0.01])))


# ---------------------------------------------------------------------------
# downside_volatility
# ---------------------------------------------------------------------------

class TestDownsideVolatility:
    def test_only_positive_returns_gives_nan(self):
        r = pd.Series([0.01, 0.02, 0.005])
        assert np.isnan(downside_volatility(r))

    def test_mixed_returns_lower_than_total_vol(self, mixed_returns):
        total = annualized_volatility(mixed_returns)
        down = downside_volatility(mixed_returns)
        # Downside vol ≤ total vol (uses only the negative half)
        assert down <= total


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_zero_vol_returns_nan(self):
        assert np.isnan(sharpe_ratio(0.1, 0.0))

    def test_known_values(self):
        # 10% return, 20% vol, 0% Rf → Sharpe = 0.5
        assert pytest.approx(sharpe_ratio(0.10, 0.20, 0.0), rel=1e-9) == 0.5

    def test_with_risk_free_rate(self):
        # 10% return, 20% vol, 4% Rf → Sharpe = 0.3
        assert pytest.approx(sharpe_ratio(0.10, 0.20, 0.04), rel=1e-9) == 0.3


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_zero_downvol_returns_nan(self):
        assert np.isnan(sortino_ratio(0.1, 0.0))

    def test_greater_or_equal_to_sharpe_for_positive_skew(self):
        # When downside vol < total vol, Sortino ≥ Sharpe
        assert sortino_ratio(0.10, 0.12) >= sharpe_ratio(0.10, 0.20)


# ---------------------------------------------------------------------------
# beta_alpha_r2
# ---------------------------------------------------------------------------

class TestBetaAlphaR2:
    def test_identical_series_beta_one(self):
        r = pd.Series(np.random.randn(200) * 0.01, name="A")
        b, a, r2 = beta_alpha_r2(r, r)
        assert pytest.approx(b, abs=1e-6) == 1.0
        assert pytest.approx(r2, abs=1e-6) == 1.0

    def test_uncorrelated_series_r2_near_zero(self):
        np.random.seed(42)
        r1 = pd.Series(np.random.randn(500) * 0.01, name="A")
        r2 = pd.Series(np.random.randn(500) * 0.01, name="B")
        _, _, r_sq = beta_alpha_r2(r1, r2)
        assert r_sq < 0.05   # Near-zero correlation → near-zero R²

    def test_too_few_observations(self):
        r = pd.Series([0.01, 0.02], name="A")
        bm = pd.Series([0.01, 0.02], name="BM")
        b, a, r2 = beta_alpha_r2(r, bm)
        assert np.isnan(b)


# ---------------------------------------------------------------------------
# historical_var / historical_cvar
# ---------------------------------------------------------------------------

class TestVaRCVaR:
    def test_var_is_negative(self, mixed_returns):
        var = historical_var(mixed_returns)
        assert var < 0

    def test_cvar_leq_var(self, mixed_returns):
        var = historical_var(mixed_returns)
        cvar = historical_cvar(mixed_returns)
        # CVaR should be at least as bad as VaR
        assert cvar <= var

    def test_known_distribution(self):
        # Uniform [-1, 1]: VaR at 95% ≈ -0.9
        np.random.seed(0)
        r = pd.Series(np.random.uniform(-1, 1, 100_000))
        var = historical_var(r, confidence=0.95)
        assert pytest.approx(var, abs=0.02) == -0.9


# ---------------------------------------------------------------------------
# maximum_drawdown
# ---------------------------------------------------------------------------

class TestMaximumDrawdown:
    def test_trending_up_zero_mdd(self, trending_prices):
        mdd = maximum_drawdown(trending_prices)
        assert pytest.approx(mdd, abs=1e-9) == 0.0

    def test_crash_prices_known_mdd(self, crash_prices):
        mdd = maximum_drawdown(crash_prices)
        # Prices go from 200 → 100: drawdown = -50%
        assert pytest.approx(mdd, abs=0.01) == -0.50

    def test_mdd_always_nonpositive(self, mixed_returns):
        prices = (1 + mixed_returns).cumprod() * 100
        assert maximum_drawdown(prices) <= 0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_zero_mdd_returns_nan(self):
        assert np.isnan(calmar_ratio(0.10, 0.0))

    def test_known_values(self):
        # 10% return / 25% MDD = 0.4
        assert pytest.approx(calmar_ratio(0.10, -0.25), rel=1e-9) == 0.4


# ---------------------------------------------------------------------------
# cumulative_return
# ---------------------------------------------------------------------------

class TestCumulativeReturn:
    def test_double_prices(self):
        prices = pd.Series([100.0, 200.0])
        assert pytest.approx(cumulative_return(prices), rel=1e-9) == 1.0

    def test_flat_prices_zero_return(self):
        prices = pd.Series([100.0, 100.0, 100.0])
        assert pytest.approx(cumulative_return(prices), abs=1e-9) == 0.0


# ---------------------------------------------------------------------------
# RiskReturnCalculator (integration-style unit test)
# ---------------------------------------------------------------------------

class TestRiskReturnCalculator:
    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        np.random.seed(1)
        n = 500
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        bm = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
        asset = 100 * (1 + np.random.randn(n) * 0.015).cumprod()
        return pd.DataFrame({"^GSPC": bm, "AAPL": asset}, index=idx)

    def test_compute_all_returns_correct_count(self, sample_prices):
        calc = RiskReturnCalculator(sample_prices, benchmark="^GSPC")
        results = calc.compute_all()
        assert len(results) == 2

    def test_no_nan_in_well_behaved_data(self, sample_prices):
        calc = RiskReturnCalculator(sample_prices, benchmark="^GSPC")
        results = calc.compute_all()
        for m in results:
            for field, val in m.to_dict().items():
                # alpha/beta of benchmark vs itself may produce edge cases; skip those
                if m.ticker == "^GSPC" and field in ("beta", "alpha_annualized", "r_squared"):
                    continue
                assert not np.isnan(val), f"{m.ticker}.{field} is NaN"

    def test_benchmark_not_in_tickers_raises(self, sample_prices):
        with pytest.raises(ValueError, match="not found"):
            RiskReturnCalculator(sample_prices, benchmark="SPY")

    def test_correlation_matrix_shape(self, sample_prices):
        calc = RiskReturnCalculator(sample_prices, benchmark="^GSPC")
        corr = calc.correlation_matrix()
        assert corr.shape == (2, 2)
        assert (corr.values == corr.values.T).all()  # symmetric
