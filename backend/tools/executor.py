"""
executor.py — validates/fetches via parser, gets code from writer, runs
it, retries on error. Supports one or more tickers (Claude can pass a
comma-separated ticker string to compare stocks on one plot).
"""

import sys
import io
import os
import glob
import traceback
from typing import Any, Dict, Optional

from fastmcp.utilities.types import Image
from PIL import Image as PILImage
from mcp_app import mcp
from . import parser
from . import writer

MAX_RETRIES = 3
OUTPUT_DIR = os.environ.get("AGENT_OUTPUT_DIR", "outputs")
MAX_DIMENSION = 900   # px, longest side
MAX_FILE_BYTES = 300_000  # keep well under known client-side MCP image limits


def compress_chart(path: str) -> str:
    """Downscale/quantize a chart so it's safely under known MCP client
    image-size limits, regardless of the DPI the generated script used.
    Uses PNG palette quantization (not JPEG) since charts are mostly flat
    color/white background — quantization shrinks them a lot without the
    blocky artifacts JPEG would leave on thin lines and text."""
    img = PILImage.open(path).convert("RGB")

    if max(img.size) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), PILImage.LANCZOS)

    colors = 256
    while colors >= 16:
        quantized = img.convert("P", palette=PILImage.ADAPTIVE, colors=colors)
        quantized.save(path, format="PNG", optimize=True)
        if os.path.getsize(path) <= MAX_FILE_BYTES:
            break
        colors //= 2

    return path


def run_python(code: str, output_dir: str) -> Dict[str, Any]:
    """Executes code, captures stdout/stderr, reports new files created."""
    os.makedirs(output_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(output_dir, "*")))

    captured_stdout = io.StringIO()
    original_stdout, original_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = captured_stdout
        sys.stderr = captured_stdout

        exec(code, {"__name__": "__main__", "OUTPUT_DIR": output_dir}, {})

        after = set(glob.glob(os.path.join(output_dir, "*")))
        return {
            "success": True,
            "stdout": captured_stdout.getvalue(),
            "error": None,
            "artifacts": sorted(after - before),
        }

    except Exception:
        return {
            "success": False,
            "stdout": captured_stdout.getvalue(),
            "error": traceback.format_exc(),
            "artifacts": [],
        }

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@mcp.tool()
def generate_chart(ticker: str, timeframe: str, action: str):
    """
    Generate a chart for one or more tickers.

    Args:
        ticker: one ticker "TSLA", or multiple comma-separated to compare
            on one plot, e.g. "TSLA, AAPL"
        timeframe: e.g. "YTD", "1y", "max"
        action: what to plot/compute, e.g. "compare closing prices"
    """
    parsed = parser.parse_query_params(ticker, timeframe, action)
    if not parsed["valid"]:
        return [f"Invalid request: {parsed['error']}"]

    tickers = parsed["tickers"]
    data_path = parsed["data_path"]
    error = ""
    result: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_RETRIES + 1):
        code = writer.write_code(tickers, data_path, action, error)
        result = run_python(code, OUTPUT_DIR)
        result["attempts"] = attempt

        if result["success"]:
            chart_path = next((p for p in result["artifacts"] if p.endswith(".png")), None)
            if chart_path:
                compress_chart(chart_path)
                summary = f"Chart generated for {', '.join(tickers)} ({parsed['period']}) in {attempt} attempt(s)."
                return [summary, Image(path=chart_path)]

            # Ran without raising, but never actually saved a chart — this
            # must retry too, not just fall through and report failure.
            error = (
                "The script ran without errors but did NOT save any .png "
                "file to OUTPUT_DIR. You MUST end the script with exactly: "
                "plt.savefig(os.path.join(OUTPUT_DIR, 'chart.png')) "
                "— do not use plt.show(), and make sure this line actually "
                "executes (not inside an if-branch that might be skipped)."
            )
            continue

        error = result["error"]

    return [f"Failed after {MAX_RETRIES} attempts. Last issue:\n{error}"]


if __name__ == "__main__":
    print(generate_chart("TSLA, AAPL", "ytd", "compare closing prices"))