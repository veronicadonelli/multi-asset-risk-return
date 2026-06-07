"""
Reporting layer.

Exports the analysis results to:
- CSV  : machine-readable metrics table
- HTML : self-contained human-readable report with embedded charts

Both outputs are deterministic: given the same inputs they always
produce identical files (suitable for version control and CI).
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from calculator import AssetMetrics, portfolio_metrics_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    metrics: list[AssetMetrics],
    output_path: Path,
) -> Path:
    """
    Write the full metrics table to CSV.

    Values are stored as raw decimals (not %) for downstream use.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = portfolio_metrics_table(metrics)
    df.to_csv(output_path)

    logger.info("CSV report written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: Figure) -> str:
    """Encode a matplotlib Figure as a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _format_value(col: str, val: float) -> str:
    """Format a metric value for HTML display."""
    if pd.isna(val):
        return "N/A"
    pct_cols = {
        "cumulative_return", "annualized_return", "annualized_volatility",
        "downside_volatility", "var_95", "cvar_95", "max_drawdown",
        "alpha_annualized",
    }
    if col in pct_cols:
        return f"{val:.2%}"
    return f"{val:.4f}"


def _metrics_to_html_table(metrics: list[AssetMetrics]) -> str:
    """Render the metrics summary as an HTML table with conditional formatting."""
    df = portfolio_metrics_table(metrics)

    # Columns to display (in order)
    display_cols = [
        "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "beta", "alpha_annualized", "r_squared",
        "var_95", "cvar_95", "max_drawdown",
    ]
    display_labels = [
        "Cum. Return", "Ann. Return", "Ann. Volatility",
        "Sharpe", "Sortino", "Calmar",
        "Beta", "Alpha (ann.)", "R²",
        "VaR 95%", "CVaR 95%", "Max DD",
    ]

    rows_html = ""
    for ticker, row in df.iterrows():
        cells = f"<td><strong>{ticker}</strong></td>"
        for col in display_cols:
            val = row.get(col, float("nan"))
            formatted = _format_value(col, val)

            # Colour coding for key ratios
            style = ""
            if col in ("sharpe_ratio", "sortino_ratio", "calmar_ratio"):
                try:
                    if float(val) >= 1.0:
                        style = "color:#27ae60;font-weight:bold"
                    elif float(val) > 0:
                        style = "color:#f39c12"
                    else:
                        style = "color:#e74c3c"
                except (ValueError, TypeError):
                    pass

            cells += f'<td style="{style}">{formatted}</td>'
        rows_html += f"<tr>{cells}</tr>\n"

    header_html = "".join(f"<th>{lbl}</th>" for lbl in ["Ticker"] + display_labels)

    return f"""
    <table class="metrics-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def export_html(
    metrics: list[AssetMetrics],
    charts: dict[str, Figure],
    output_path: Path,
    report_title: str = "Multi-Asset Risk–Return Analysis",
    analysis_period: str = "",
    benchmark: str = "^GSPC",
) -> Path:
    """
    Write a self-contained HTML report with embedded charts and metrics table.

    The report requires no external dependencies to render — all CSS and
    chart images are inlined.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Encode all charts
    chart_sections = ""
    chart_titles = {
        "chart_1_cumulative_returns": "Normalized Cumulative Returns",
        "chart_2_risk_return_scatter": "Risk–Return Scatter",
        "chart_3_beta_r2": "Beta & R²",
        "chart_4_var_cvar": "VaR vs CVaR",
        "chart_5_sharpe_sortino": "Sharpe vs Sortino",
        "chart_6_drawdown": "Drawdown Analysis",
        "chart_7_correlation_heatmap": "Correlation Heatmap",
        "chart_8_return_distributions": "Return Distributions",
    }
    for key, fig in charts.items():
        title = chart_titles.get(key, key.replace("_", " ").title())
        b64 = _fig_to_base64(fig)
        chart_sections += f"""
        <section class="chart-section">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{b64}" alt="{title}" />
        </section>
        """

    metrics_table_html = _metrics_to_html_table(metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{report_title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f4f6f9;
            color: #2c3e50;
            line-height: 1.6;
        }}
        header {{
            background: #1b3a5c;
            color: white;
            padding: 2rem 3rem;
            border-bottom: 4px solid #2196F3;
        }}
        header h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
        header p  {{ font-size: 0.9rem; opacity: 0.75; }}
        main {{ max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }}
        h2 {{
            font-size: 1.3rem;
            color: #1b3a5c;
            margin: 2rem 0 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #2196F3;
        }}
        h3 {{ font-size: 1rem; color: #34495e; margin-bottom: 0.5rem; }}
        .chart-section {{ margin-bottom: 2.5rem; background: white;
                          border-radius: 8px; padding: 1.5rem;
                          box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
        .chart-section img {{ width: 100%; height: auto; border-radius: 4px; }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,.08);
        }}
        .metrics-table th {{
            background: #1b3a5c;
            color: white;
            padding: 0.6rem 0.8rem;
            text-align: right;
            white-space: nowrap;
        }}
        .metrics-table th:first-child {{ text-align: left; }}
        .metrics-table td {{
            padding: 0.5rem 0.8rem;
            text-align: right;
            border-bottom: 1px solid #ecf0f1;
        }}
        .metrics-table td:first-child {{ text-align: left; }}
        .metrics-table tr:hover {{ background: #f0f4ff; }}
        footer {{
            text-align: center;
            padding: 2rem;
            font-size: 0.8rem;
            color: #95a5a6;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 {report_title}</h1>
        <p>Benchmark: {benchmark} &nbsp;|&nbsp; {analysis_period}
           &nbsp;|&nbsp; Generated: {generated_at}</p>
    </header>
    <main>
        <h2>Performance Metrics Summary</h2>
        {metrics_table_html}

        <h2>Visualizations</h2>
        {chart_sections}
    </main>
    <footer>
        Generated by multi-asset-risk-return · MIT License
    </footer>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
    return output_path
