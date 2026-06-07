"""
Data acquisition layer.
Handles fetching, caching, validation and cleaning of market price data
from Yahoo Finance. Designed for reproducibility and reliability.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for data acquisition."""
    tickers: tuple[str, ...]
    benchmark: str
    start: str
    end: str
    cache_dir: Path = Path(".cache")
    max_missing_pct: float = 0.02         # Allow up to 2 % NaN per series
    min_observations: int = 120           # Minimum trading days required
    auto_adjust: bool = True              # Corporate-action adjusted prices

    def __post_init__(self) -> None:
        if self.benchmark not in self.tickers:
            raise ValueError(
                f"Benchmark '{self.benchmark}' must be included in tickers."
            )
        start_dt = datetime.strptime(self.start, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end, "%Y-%m-%d")
        if end_dt <= start_dt:
            raise ValueError("'end' must be strictly after 'start'.")

    @property
    def all_tickers(self) -> list[str]:
        """Returns deduplicated, ordered ticker list."""
        seen: dict[str, None] = {}
        for t in self.tickers:
            seen[t] = None
        return list(seen)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    """Summary of data quality checks after fetching."""
    total_tickers: int
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        lines = [
            f"Data Quality Report — {self.total_tickers} tickers",
            f"  Passed : {len(self.passed)}",
            f"  Failed : {len(self.failed)}",
        ]
        for ticker, msgs in self.warnings.items():
            for msg in msgs:
                lines.append(f"  ⚠  {ticker}: {msg}")
        if self.failed:
            lines.append(f"  ✗  Failed tickers: {', '.join(self.failed)}")
        return "\n".join(lines)


def _validate_series(
    series: pd.Series,
    ticker: str,
    cfg: DataConfig,
) -> tuple[bool, list[str]]:
    """
    Run data quality checks on a single price series.

    Returns (is_valid, list_of_warnings).
    """
    warnings_out: list[str] = []

    # Minimum length
    if len(series.dropna()) < cfg.min_observations:
        return False, [
            f"Only {len(series.dropna())} observations; "
            f"minimum required: {cfg.min_observations}."
        ]

    # Missing value threshold
    missing_pct = series.isna().mean()
    if missing_pct > cfg.max_missing_pct:
        return False, [
            f"{missing_pct:.1%} missing values exceed threshold "
            f"({cfg.max_missing_pct:.1%})."
        ]
    elif missing_pct > 0:
        warnings_out.append(
            f"{missing_pct:.2%} missing values forward-filled."
        )

    # Negative or zero prices
    non_positive = (series.dropna() <= 0).sum()
    if non_positive > 0:
        return False, [f"{non_positive} non-positive price(s) found."]

    # Stale prices: more than 5 consecutive identical values
    consecutive_same = (series.diff() == 0).astype(int)
    max_run = consecutive_same.groupby(
        (consecutive_same != consecutive_same.shift()).cumsum()
    ).sum().max()
    if max_run > 5:
        warnings_out.append(
            f"Up to {max_run} consecutive unchanged prices detected "
            f"(possible data feed issue)."
        )

    return True, warnings_out


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _cache_key(cfg: DataConfig) -> str:
    key_str = f"{sorted(cfg.all_tickers)}{cfg.start}{cfg.end}{cfg.auto_adjust}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


def _cache_path(cfg: DataConfig) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"prices_{_cache_key(cfg)}.parquet"


def _is_cache_valid(path: Path, max_age_hours: int = 24) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MarketDataProvider:
    """
    Fetches and validates adjusted daily close prices for a given universe.

    Usage
    -----
    >>> cfg = DataConfig(tickers=(...), benchmark="^GSPC", start="2022-01-01", end="2025-01-01")
    >>> provider = MarketDataProvider(cfg)
    >>> prices, report = provider.get_prices()
    """

    def __init__(self, config: DataConfig, use_cache: bool = True) -> None:
        self.config = config
        self.use_cache = use_cache

    # ------------------------------------------------------------------ #
    def get_prices(
        self, *, force_refresh: bool = False
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Return a validated DataFrame of adjusted close prices.

        Columns  : one per ticker
        Index    : DatetimeIndex of trading days (UTC, tz-naive)
        """
        cache_path = _cache_path(self.config)

        if self.use_cache and not force_refresh and _is_cache_valid(cache_path):
            logger.info("Loading prices from cache: %s", cache_path)
            prices = pd.read_parquet(cache_path)
        else:
            logger.info(
                "Downloading %d tickers from Yahoo Finance …",
                len(self.config.all_tickers),
            )
            prices = self._download()
            if self.use_cache:
                prices.to_parquet(cache_path)
                logger.info("Prices cached to %s", cache_path)

        prices, report = self._validate(prices)

        if not report.is_clean:
            logger.warning(report.summary())
            raise DataQualityError(
                f"Data quality checks failed for: {report.failed}\n"
                f"{report.summary()}"
            )
        else:
            logger.info(report.summary())

        return prices, report

    # ------------------------------------------------------------------ #
    def _download(self) -> pd.DataFrame:
        raw = yf.download(
            self.config.all_tickers,
            start=self.config.start,
            end=self.config.end,
            auto_adjust=self.config.auto_adjust,
            progress=False,
            threads=True,
        )
        # yfinance returns MultiIndex columns when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": self.config.all_tickers[0]})

        # Normalise index to tz-naive UTC date
        close.index = pd.to_datetime(close.index).tz_localize(None)

        # Drop rows where ALL prices are missing (market holidays)
        return close.dropna(how="all")

    # ------------------------------------------------------------------ #
    def _validate(
        self, prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        report = DataQualityReport(total_tickers=len(prices.columns))
        cleaned: dict[str, pd.Series] = {}

        for ticker in prices.columns:
            series = prices[ticker].copy()
            # Forward-fill gaps (e.g. ETFs closed on certain holidays)
            series = series.ffill()

            ok, warns = _validate_series(series, ticker, self.config)
            if ok:
                report.passed.append(ticker)
                if warns:
                    report.warnings[ticker] = warns
                cleaned[ticker] = series
            else:
                report.failed.append(ticker)
                report.warnings[ticker] = warns

        cleaned_df = pd.DataFrame(cleaned).dropna()
        return cleaned_df, report


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataQualityError(RuntimeError):
    """Raised when data does not pass quality validation."""
