"""
writer.py — code-writing tool. Writes plotting/analysis code against a CSV
that parser.py already fetched. Supports one or more tickers sharing that
CSV, with columns prefixed per ticker (e.g. "TSLA_Close", "AAPL_Close").
"""

import time
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from mcp_app import mcp

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)


def _prompt(tickers: List[str], data_path: str, action: str, chart_filename: str, error: str = "") -> str:
    tickers_str = ", ".join(tickers)
    multi_note = (
        f"This CSV contains {len(tickers)} tickers. Each column is prefixed "
        f"with its ticker, e.g. '{tickers[0]}_Close'. To compare them, plot "
        "each ticker's relevant column on the same axes.\n"
        if len(tickers) > 1 else ""
    )
    return (
        "Write a complete, standalone Python script using pandas and "
        "matplotlib to analyze financial data.\n"
        f"Ticker(s): {tickers_str}\n"
        f"Data is already fetched and saved at: {data_path}\n"
        f"{multi_note}"
        f"Requested action: {action}\n"
        "Rules:\n"
        "1. Start with: import matplotlib; matplotlib.use('Agg')\n"
        f"2. Load the data with: pd.read_csv('{data_path}', index_col=0, parse_dates=True)\n"
        "   Do NOT call yfinance or download anything — the data already exists.\n"
        f"3. Save the chart with EXACTLY: plt.savefig(os.path.join(OUTPUT_DIR, '{chart_filename}'))\n"
        "   Use this exact filename — do not use 'chart.png' or make up your own name.\n"
        "4. Never use plt.show() or input().\n"
        "5. Output ONLY raw Python code, no markdown, no explanation.\n"
        + (f"\nYour previous attempt failed with this error, fix it:\n{error}" if error else "")
    )


@mcp.tool()
def write_code(tickers: List[str], data_path: str, action: str, error: str = "") -> str:
    """
    Write a Python script (pandas + matplotlib) for a chart or metric,
    reading from an already-fetched CSV. Handles one or multiple tickers
    (e.g. comparing two stocks on one plot) sharing that same CSV.
    Returns raw code as a string. Does not execute it.

    Args:
        tickers: list of ticker symbols in the CSV, e.g. ["TSLA", "AAPL"]
        data_path: path to the CSV parser.py already fetched
        action: what to plot/compute, e.g. "compare closing prices"
        error: optional traceback from a previous failed attempt, to fix
    """
    chart_filename = f"chart_{'_'.join(tickers)}_{int(time.time())}.png"
    response = llm.invoke([
        SystemMessage(content=_prompt(tickers, data_path, action, chart_filename, error)),
        HumanMessage(content="Write the script now."),
    ])
    return response.content.replace("```python", "").replace("```", "").strip()