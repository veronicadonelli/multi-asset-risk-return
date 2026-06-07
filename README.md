# Multi-Asset Risk–Return Analysis

[![CI](https://github.com/veronicadonelli/multi-asset-risk-return/actions/workflows/ci.yml/badge.svg)](https://github.com/veronicadonelli/multi-asset-risk-return/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Systematic, production-grade quantitative risk-return evaluation of 8 financial assets — equities, ETFs, and commodities — benchmarked against the S&P 500. Built to the standards expected in institutional investment research.

---

## Overview

This project delivers a fully automated, end-to-end analysis pipeline covering:

- **Data acquisition** with caching, validation, and quality reporting
- **12 risk/return metrics** computed from first principles
- **8 publication-quality visualizations**
- **Self-contained HTML report** + CSV export
- **80%+ test coverage** with isolated, fast unit tests
- **CI/CD pipeline** with lint, type check, security scan, and test matrix

The analysis covers the period **March 2022 – March 2025**, capturing the Federal Reserve's fastest rate-hiking cycle since the 1980s — a demanding stress test for cross-asset risk and return dynamics.

---

## Asset Universe

| Ticker  | Name                    | Category         |
|---------|-------------------------|------------------|
| `^GSPC` | S&P 500 Index           | Benchmark        |
| `AAPL`  | Apple Inc.              | Equity           |
| `TSLA`  | Tesla Inc.              | Equity           |
| `JPM`   | JPMorgan Chase          | Equity           |
| `SPY`   | SPDR S&P 500 ETF        | ETF              |
| `QQQ`   | Invesco Nasdaq-100 ETF  | ETF              |
| `GLD`   | SPDR Gold Shares ETF    | ETF / Commodity  |
| `GC=F`  | Gold Futures            | Commodity        |
| `CL=F`  | Crude Oil WTI Futures   | Commodity        |

---

## Metrics Computed

| Metric                | Description                                           |
|-----------------------|-------------------------------------------------------|
| Annualized Return     | Mean daily log return × 252                           |
| Annualized Volatility | Std dev of log returns × √252                         |
| Sharpe Ratio          | Excess return / total volatility                      |
| Sortino Ratio         | Excess return / downside volatility                   |
| Beta                  | OLS slope vs. benchmark                               |
| Jensen's Alpha        | Annualised OLS intercept                              |
| R²                    | Variance explained by the market                      |
| VaR 95%               | Historical 5th percentile of daily log returns        |
| CVaR 95%              | Mean loss in the worst 5% of days (Expected Shortfall)|
| Maximum Drawdown      | Largest peak-to-trough decline                        |
| Calmar Ratio          | Annualized return / absolute MDD                      |
| Cumulative Return     | Total price appreciation over the period              |

---

## Project Structure

```
multi-asset-risk-return/
│
├── src/
│   ├── data/
│   │   └── provider.py          # Data fetch, cache, validation
│   ├── metrics/
│   │   └── calculator.py        # All risk/return metrics
│   ├── visualization/
│   │   └── charts.py            # 8 publication-quality charts
│   ├── reporting/
│   │   └── report.py            # CSV + HTML export
│   └── main.py                  # Pipeline entry point
│
├── tests/
│   └── unit/
│       ├── test_metrics.py      # 30+ metric unit tests
│       └── test_provider.py     # Provider tests with mocked I/O
│
├── config/
│   └── default.yaml             # All parameters in one place
│
├── .github/
│   └── workflows/
│       └── ci.yml               # Lint → test → security → typecheck
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start

**1. Clone**
```bash
git clone https://github.com/veronicadonelli/multi-asset-risk-return.git
cd multi-asset-risk-return
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the full pipeline**
```bash
python -m src.main
```

Reports are written to `reports/metrics.csv` and `reports/report.html`.

**5. Run with custom config**
```bash
python -m src.main --config config/my_custom.yaml
```

**6. Force a fresh data download**
```bash
python -m src.main --no-cache
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All parameters live in `config/default.yaml`. Key settings:

```yaml
tickers:        [list of Yahoo Finance ticker symbols]
benchmark:      "^GSPC"
start:          "2022-03-01"
end:            "2025-03-31"
risk_free_rate: 0.0          # Set to 0.04 for current T-bill rate
confidence_level: 0.95       # For VaR / CVaR
output_dir:     "reports"
```

---

## Key Findings (Mar 2022 – Mar 2025)

- **Gold (GLD / GC=F)** delivered the highest cumulative returns (~166–169%) with Sharpe Ratios above 1.35, while maintaining near-zero correlation with equities (R² = 0.006–0.014), confirming its diversification value.
- **JPMorgan (JPM)** was the top equity performer with a 144% cumulative return and positive Jensen's Alpha (+0.116), directly benefiting from the rate-hike cycle.
- **Tesla (TSLA)** carried the highest beta (2.05) but produced the worst risk-adjusted outcome: Sharpe of 0.14 with a 71.7% maximum drawdown.
- **Equity cluster correlation** (AAPL, TSLA, JPM, SPY, QQQ) is high (ρ = 0.60–0.99). Meaningful diversification requires structurally uncorrelated asset classes.

---

## Methodology

- **Log returns** used throughout for time-additivity and approximate normality.
- **252 trading-day convention** for annualisation; square-root-of-time for volatility.
- **Historical simulation VaR/CVaR** — non-parametric, no distributional assumptions.
- **OLS regression** for beta, alpha and R².
- **Risk-free rate** set to 0% for cross-asset comparability (configurable).

---

## Background

Developed to sharpen quantitative financial analysis skills with a focus on risk-return frameworks relevant to portfolio construction and investment research.

**Current role:** Wealth Management — Data & MiFID II Reporting

---

## License

MIT — free to use, adapt, and build upon with attribution.
