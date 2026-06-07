"""
Multi-Asset Risk–Return Analysis
=================================
Main entry point. Wires together data acquisition, metrics calculation,
visualization and reporting into a single reproducible pipeline.

Usage
-----
    python -m src.main                        # uses config/default.yaml
    python -m src.main --config my_config.yaml
    python -m src.main --no-cache             # force fresh data download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from provider import DataConfig, MarketDataProvider
from calculator import RiskReturnCalculator
from report import export_csv, export_html
from charts import generate_all_charts

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(config_path: str | Path = "config/default.yaml", use_cache: bool = True) -> None:
    cfg = load_config(config_path)

    data_cfg = DataConfig(
        tickers=tuple(cfg["tickers"]),
        benchmark=cfg["benchmark"],
        start=cfg["start"],
        end=cfg["end"],
        cache_dir=Path(cfg.get("cache_dir", ".cache")),
        max_missing_pct=cfg.get("max_missing_pct", 0.02),
        min_observations=cfg.get("min_observations", 120),
    )

    # ── Step 1: Fetch & validate data ─────────────────────────────────────
    logger.info("Step 1/4 — Fetching market data …")
    provider = MarketDataProvider(data_cfg, use_cache=use_cache)
    prices, quality_report = provider.get_prices()
    logger.info(quality_report.summary())

    # ── Step 2: Compute metrics ────────────────────────────────────────────
    logger.info("Step 2/4 — Computing risk/return metrics …")
    calculator = RiskReturnCalculator(
        prices=prices,
        benchmark=data_cfg.benchmark,
        risk_free_rate=cfg.get("risk_free_rate", 0.0),
        confidence=cfg.get("confidence_level", 0.95),
    )
    metrics = calculator.compute_all()
    corr_matrix = calculator.correlation_matrix()

    # ── Step 3: Generate charts ────────────────────────────────────────────
    logger.info("Step 3/4 — Generating charts …")
    charts = generate_all_charts(
        prices=prices,
        log_returns=calculator.log_returns,
        metrics=metrics,
        corr_matrix=corr_matrix,
        benchmark=data_cfg.benchmark,
    )

    # ── Step 4: Export reports ─────────────────────────────────────────────
    logger.info("Step 4/4 — Exporting reports …")
    output_dir = Path(cfg.get("output_dir", "reports"))

    csv_path = export_csv(metrics, output_dir / "metrics.csv")
    html_path = export_html(
        metrics=metrics,
        charts=charts,
        output_path=output_dir / "report.html",
        report_title=cfg.get("report_title", "Multi-Asset Risk–Return Analysis"),
        analysis_period=f"{data_cfg.start} → {data_cfg.end}",
        benchmark=data_cfg.benchmark,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("  CSV  → %s", csv_path)
    logger.info("  HTML → %s", html_path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Asset Risk–Return Analysis Pipeline"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data download, ignore cached prices",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(config_path=args.config, use_cache=not args.no_cache)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
