"""
server.py — entry point.

Resolves a writable OUTPUT_DIR (absolute, overridable via AGENT_OUTPUT_DIR
env var) so tools don't depend on whatever directory the MCP client
happens to launch this process from. Then imports each tool module so
their @mcp.tool() decorators register, then starts the server.
"""

import os
import sys
from pathlib import Path

from mcp_app import mcp

OUTPUT_DIR = Path(
    os.environ.get("AGENT_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["AGENT_OUTPUT_DIR"] = str(OUTPUT_DIR)  # so tool modules resolve the same path

import tools.parser   
import tools.writer   
import tools.executor 

if __name__ == "__main__":
    print("Starting MCP server... waiting for client to connect.", file=sys.stderr)
    mcp.run()