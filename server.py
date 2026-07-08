import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP, Image

# Initialize the MCP server
mcp = FastMCP("financial-analyst")

TICKER_ALIASES = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
}


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip()
    return TICKER_ALIASES.get(cleaned.lower(), cleaned.upper())


def fetch_ytd_gain_series(ticker: str) -> pd.Series:
    year_start = datetime.now().replace(month=1, day=1).strftime("%Y-%m-%d")
    data = yf.download(ticker, start=year_start, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    baseline = float(close.iloc[0])
    return ((close / baseline) - 1) * 100


def build_ytd_gain_line_chart(ticker: str, gain_series: pd.Series) -> Path:
    latest_gain = float(gain_series.iloc[-1])
    output_path = Path("output_chart.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    gain_series.plot(ax=ax, color="#E82127", linewidth=2)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_title(f"{ticker} YTD Stock Gain", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gain (%)")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"Latest YTD gain: {latest_gain:+.2f}%",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


@mcp.tool()
def test_connection() -> str:
    """A simple tool to verify Cursor is connected to the Python MCP server."""
    return "Connection successful! The MCP server is running."


@mcp.tool()
def analyze_and_plot(
    ticker: str,
    metric: str = "YTD stock gain",
    chart_type: str = "line",
) -> list:
    """Analyze a stock metric and generate a chart.

    Args:
        ticker: Stock ticker symbol or common company name (e.g. 'TSLA' or 'Tesla').
        metric: Metric to analyze. Supports YTD stock gain.
        chart_type: Chart style to render. Supports 'line'.
    """
    symbol = normalize_ticker(ticker)
    metric_lower = metric.lower()

    if chart_type.lower() != "line":
        raise ValueError(f"Unsupported chart_type '{chart_type}'. Use 'line'.")

    if "ytd" not in metric_lower or "gain" not in metric_lower:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'YTD stock gain'.")

    gain_series = fetch_ytd_gain_series(symbol)
    output_path = build_ytd_gain_line_chart(symbol, gain_series)
    latest_gain = float(gain_series.iloc[-1])

    summary = (
        f"Generated a {chart_type} chart for {symbol} {metric}. "
        f"Latest YTD gain: {latest_gain:+.2f}%. "
        f"Chart saved to {output_path.resolve()}."
    )

    return [summary, Image(path=output_path)]


if __name__ == "__main__":
    print("Starting the FastMCP server... Waiting for Cursor to connect.", file=sys.stderr)
    mcp.run()
