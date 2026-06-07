"""
Unit tests for src/data/provider.py

Tests cover configuration validation, data quality checks,
and caching behaviour without making real network calls
(yfinance is mocked throughout).

Run with:
    pytest tests/unit/test_provider.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from provider import (
    DataConfig,
    DataQualityError,
    DataQualityReport,
    MarketDataProvider,
    _is_cache_valid,
    _validate_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 500, tickers: list[str] | None = None) -> pd.DataFrame:
    """Generate synthetic price DataFrame for testing."""
    if tickers is None:
        tickers = ["^GSPC", "AAPL"]
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    data = {
        t: 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
        for t in tickers
    }
    return pd.DataFrame(data, index=idx)


def _make_config(**overrides) -> DataConfig:
    defaults = dict(
        tickers=("^GSPC", "AAPL"),
        benchmark="^GSPC",
        start="2022-01-01",
        end="2025-01-01",
    )
    defaults.update(overrides)
    return DataConfig(**defaults)


# ---------------------------------------------------------------------------
# DataConfig validation
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_valid_config(self):
        cfg = _make_config()
        assert cfg.benchmark == "^GSPC"
        assert "^GSPC" in cfg.all_tickers

    def test_benchmark_not_in_tickers_raises(self):
        with pytest.raises(ValueError, match="must be included"):
            _make_config(benchmark="SPY")

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="strictly after"):
            _make_config(start="2025-01-01", end="2022-01-01")

    def test_end_equal_start_raises(self):
        with pytest.raises(ValueError, match="strictly after"):
            _make_config(start="2022-01-01", end="2022-01-01")

    def test_deduplication_of_tickers(self):
        cfg = DataConfig(
            tickers=("^GSPC", "AAPL", "AAPL", "^GSPC"),
            benchmark="^GSPC",
            start="2022-01-01",
            end="2025-01-01",
        )
        assert len(cfg.all_tickers) == 2


# ---------------------------------------------------------------------------
# _validate_series
# ---------------------------------------------------------------------------

class TestValidateSeries:
    def test_valid_series_passes(self):
        cfg = _make_config(min_observations=10)
        series = pd.Series(np.linspace(100, 200, 200))
        ok, warns = _validate_series(series, "AAPL", cfg)
        assert ok
        assert warns == []

    def test_too_short_fails(self):
        cfg = _make_config(min_observations=500)
        series = pd.Series(np.linspace(100, 110, 50))
        ok, warns = _validate_series(series, "AAPL", cfg)
        assert not ok
        assert any("observations" in w for w in warns)

    def test_non_positive_price_fails(self):
        cfg = _make_config(min_observations=5)
        series = pd.Series([100, 0, 50, 60, 70, 80, 90, 100, 110, 120])
        ok, warns = _validate_series(series, "X", cfg)
        assert not ok
        assert any("non-positive" in w for w in warns)

    def test_too_many_nans_fails(self):
        cfg = _make_config(min_observations=5, max_missing_pct=0.01)
        series = pd.Series([100.0, np.nan, 102.0, np.nan, np.nan] * 40)
        ok, warns = _validate_series(series, "Y", cfg)
        assert not ok

    def test_forward_fill_warning_on_small_nan(self):
        cfg = _make_config(min_observations=5, max_missing_pct=0.10)
        # 1 NaN in 100 obs = 1% missing → below threshold
        vals = [float(i) + 100 for i in range(100)]
        vals[50] = np.nan
        series = pd.Series(vals)
        ok, warns = _validate_series(series, "Z", cfg)
        assert ok
        assert any("forward-filled" in w for w in warns)


# ---------------------------------------------------------------------------
# _is_cache_valid
# ---------------------------------------------------------------------------

class TestIsCacheValid:
    def test_nonexistent_file_is_invalid(self, tmp_path):
        assert not _is_cache_valid(tmp_path / "ghost.parquet")

    def test_fresh_file_is_valid(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"")
        assert _is_cache_valid(p, max_age_hours=24)

    def test_old_file_is_invalid(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"")
        old_time = (datetime.now() - timedelta(hours=48)).timestamp()
        import os
        os.utime(p, (old_time, old_time))
        assert not _is_cache_valid(p, max_age_hours=24)


# ---------------------------------------------------------------------------
# MarketDataProvider (mocked yfinance)
# ---------------------------------------------------------------------------

class TestMarketDataProvider:
    @pytest.fixture
    def cfg(self, tmp_path):
        return DataConfig(
            tickers=("^GSPC", "AAPL"),
            benchmark="^GSPC",
            start="2022-01-01",
            end="2025-01-01",
            cache_dir=tmp_path / ".cache",
            min_observations=10,
        )

    @pytest.fixture
    def mock_prices(self):
        return _make_prices(n=300, tickers=["^GSPC", "AAPL"])

    def _make_yf_mock(self, prices: pd.DataFrame):
        """Build a mock that mimics yfinance MultiIndex output."""
        mock_raw = MagicMock()
        multi_idx = pd.MultiIndex.from_tuples(
            [("Close", t) for t in prices.columns]
        )
        mock_raw.columns = multi_idx
        mock_raw.__getitem__ = lambda self, key: prices if key == "Close" else None
        return mock_raw

    def test_get_prices_returns_dataframe(self, cfg, mock_prices):
        with patch("src.data.provider.yf.download", return_value=self._make_yf_mock(mock_prices)):
            provider = MarketDataProvider(cfg, use_cache=False)
            prices, report = provider.get_prices()

        assert isinstance(prices, pd.DataFrame)
        assert set(prices.columns) == {"^GSPC", "AAPL"}
        assert report.is_clean

    def test_data_quality_error_on_bad_data(self, cfg):
        bad = _make_prices(n=5, tickers=["^GSPC", "AAPL"])  # Too few obs

        with patch("src.data.provider.yf.download", return_value=self._make_yf_mock(bad)):
            provider = MarketDataProvider(cfg, use_cache=False)
            with pytest.raises(DataQualityError):
                provider.get_prices()

    def test_cache_is_written_on_first_fetch(self, cfg, mock_prices, tmp_path):
        with patch("src.data.provider.yf.download", return_value=self._make_yf_mock(mock_prices)):
            provider = MarketDataProvider(cfg, use_cache=True)
            provider.get_prices()

        cache_files = list((tmp_path / ".cache").glob("*.parquet"))
        assert len(cache_files) == 1

    def test_cache_is_read_on_second_fetch(self, cfg, mock_prices):
        with patch("src.data.provider.yf.download", return_value=self._make_yf_mock(mock_prices)) as mock_dl:
            provider = MarketDataProvider(cfg, use_cache=True)
            provider.get_prices()   # first: writes cache
            provider.get_prices()   # second: should read from cache

        # yf.download should have been called only once
        assert mock_dl.call_count == 1

    def test_force_refresh_bypasses_cache(self, cfg, mock_prices):
        with patch("src.data.provider.yf.download", return_value=self._make_yf_mock(mock_prices)) as mock_dl:
            provider = MarketDataProvider(cfg, use_cache=True)
            provider.get_prices()
            provider.get_prices(force_refresh=True)  # should re-download

        assert mock_dl.call_count == 2
