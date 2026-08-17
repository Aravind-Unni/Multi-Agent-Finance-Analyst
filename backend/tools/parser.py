"""
parser.py — validates ticker(s)/period, fetches price data for one or
more tickers, and merges them into ONE combined CSV so writer.py can
compare multiple stocks on a single plot without writing its own
multi-fetch/merge logic.
"""

import os
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf

from mcp_app import mcp

VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
OUTPUT_DIR = os.environ.get("AGENT_OUTPUT_DIR", "outputs")


@mcp.tool()
def parse_query_params(ticker: str, timeframe: str, action: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate ticker(s)/timeframe, fetch price data, save it to ONE CSV.

    Args:
        ticker: one ticker "TSLA", or multiple comma-separated "TSLA, AAPL"
        timeframe: e.g. "YTD", "1y", "max"
        action: optional, what to do with the data

    Returns (on success):
        tickers (list), period, action, data_path (CSV for writer.py to
        read — columns are prefixed per ticker, e.g. "TSLA_Close",
        "AAPL_Close"), columns, rows
    Returns (on failure):
        valid: False, error: str
    """
    tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if not tickers:
        return {"valid": False, "error": "No ticker provided."}

    period = timeframe.strip().lower()
    if period not in VALID_PERIODS:
        return {
            "valid": False,
            "error": f"'{timeframe}' is not a valid period. Use one of: {sorted(VALID_PERIODS)}",
        }

    frames = []
    for t in tickers:
        try:
            data = yf.download(t, period=period, progress=False, auto_adjust=True)
        except Exception as e:
            return {"valid": False, "error": f"Failed to fetch data for '{t}': {e}"}

        if data.empty:
            return {"valid": False, "error": f"No data found for ticker '{t}'."}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Prefix columns per ticker so multiple stocks can share one CSV
        # without column name collisions, e.g. "Close" -> "TSLA_Close".
        frames.append(data.add_prefix(f"{t}_"))

    combined = pd.concat(frames, axis=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = os.path.join(OUTPUT_DIR, f"{'_'.join(tickers)}_{period}.csv")
    combined.to_csv(data_path)

    return {
        "valid": True,
        "tickers": tickers,
        "period": period,
        "action": action,
        "data_path": data_path,
        "columns": [str(c) for c in combined.columns],
        "rows": len(combined),
    }


if __name__ == "__main__":
    print(parse_query_params("tsla, aapl", "YTD", "compare closing prices"))