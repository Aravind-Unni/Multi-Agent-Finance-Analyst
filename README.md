# Market Intelligence Agent Using MCP

A Model Context Protocol (MCP) server that converts natural language financial queries into executable, visual market insights. The MCP client's own LLM (tested on **Claude Desktop** and **Cursor**) drives three MCP tools — parse, write, execute — in sequence to fetch price data, generate quantitative Python code, and safely execute it to produce a chart.


> Note: this README documents the *current* architecture. An earlier version of this project used a 4-agent LangGraph pipeline (Scout/Quant/Editor/Supervisor) with NVIDIA/Groq-based intent parsing and a LangGraph Supervisor node — that approach was abandoned in favor of the simpler tool-based design below, where the connecting MCP client (Claude Desktop or Cursor) handles orchestration and intent parsing itself, and the server just exposes three tools.

## 🌟 MCP Integration

This project is an MCP **server** — it doesn't run its own agent loop. Instead, it exposes three tools that any MCP-compatible client can call. The client's model decides which tool to call, in what order, and with what arguments; the server just executes them.

Tested and working with:

- **Claude Desktop** — connects via stdio, local process. See the transcript link above.
- **Cursor** — connects via the same stdio config under Settings → Features → MCP.

### MCP Configuration (Claude Desktop)

```jsonc
{
  "mcpServers": {
    "financial-analyst": {
      "command": "C:/Market Intelligence Agent/.venv/Scripts/python.exe",
      "args": ["C:/Market Intelligence Agent/backend/server.py"]
    }
  }
}
```

### MCP Configuration (Cursor)

```json
{
  "mcpServers": {
    "market-intelligence": {
      "command": "C:/Market Intelligence Agent/.venv/Scripts/python.exe",
      "args": [
        "C:/Market Intelligence Agent/backend/server.py"
      ],
      "env": {
        "PYTHONPATH": "C:/Market Intelligence Agent"
      }
    }
  }
}
```

> Ensure the paths match your local workspace directory in both cases.

## 🛠️ The Three MCP Tools

There is no LangGraph, no Supervisor, and no separate agent framework here — just three `@mcp.tool()`-decorated functions, called in sequence by the connecting client:

1. **`parse_query_params`** (`parser.py`) — validates the ticker(s) and timeframe, fetches price data via `yfinance` for one or more tickers, and merges everything into a single CSV (columns prefixed per ticker, e.g. `TSLA_Close`, `AAPL_Close`) so multi-ticker comparisons share one file.
2. **`write_code`** (`writer.py`) — acts as a code-writing **tool that is itself backed by an LLM**: it calls the Groq API (`openai/gpt-oss-120b` via `langchain_groq`) to generate a standalone pandas/matplotlib script against the CSV from step 1. It's strictly prompted to use `matplotlib.use('Agg')` for headless plotting and to save the chart to an exact filename.
3. **`generate_chart`** (`executor.py`) — the orchestrating tool. Calls `parse_query_params`, then `write_code`, executes the generated code in a sandboxed `exec()` call, and retries (up to 3 attempts) if the code errors out or fails to save a chart. On success it compresses the chart (PNG palette quantization, capped dimensions/file size) and returns it.

### Working around the MCP inline-image bug

Both Claude Desktop and claude.ai currently have an open, unresolved UI bug where MCP `image` content blocks never render inline in the visible chat response — they only ever show up inside the collapsed tool-call accordion. `executor.py` works around this by also base64-encoding the compressed chart into a `data:` URI and instructing the client to embed it directly in an HTML artifact (`<img src="data:image/png;base64,...">`), which isn't subject to the accordion bug or the artifact sandbox's CSP (which only blocks *external* image URLs, not data URIs).

## 🧠 LLM Usage

- **Groq (`openai/gpt-oss-120b`)** — used inside the `write_code` tool to generate the pandas/matplotlib analysis script.
- **Ollama (local, `ornith:35b`)** — used as the LLM-as-Judge in the MLflow evaluation suite (`backend/evaluation/evaluate.py`), scoring generated code against ground-truth rubrics.
- **Claude (Desktop or Cursor)** — the actual orchestrator at runtime. It reads each tool's docstring, decides which tools to call and in what order/arguments, and interprets the results back to the user.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | FastMCP (MCP Server) |
| Code-writing LLM | Groq (`openai/gpt-oss-120b`) via `langchain_groq` |
| Judge LLM | Ollama (`ornith:35b`) via `langchain_ollama` |
| Data Engine | yfinance, pandas, matplotlib |
| Evaluation | MLflow |
| Environment | Python 3.12+, `uv` package manager |

## ⚙️ Workflow

1. **Parse & fetch** — `parse_query_params` validates ticker(s)/timeframe and fetches+merges price data into one CSV.
2. **Write** — `write_code` prompts Groq's `openai/gpt-oss-120b` to generate a standalone script against that CSV.
3. **Execute** — `generate_chart` runs the script in a sandbox, retries on failure (up to 3 attempts, feeding the error back into the next `write_code` call), compresses the resulting chart, and returns it (plus a data-URI artifact instruction as an inline-image workaround).

## 📂 Project Structure

```
C:\Market Intelligence Agent\
├── backend\
│   ├── tools\
│   │   ├── mcp_app.py     # FastMCP app instance + .env loading
│   │   ├── parser.py      # Tool: ticker/timeframe validation + data fetch
│   │   ├── writer.py      # Tool: LLM-backed code generation (Groq)
│   │   └── executor.py    # Tool: sandboxed execution + retries + chart output
│   ├── evaluation\
│   │   ├── evaluate.py    # MLflow evaluation suite (Ollama judge)
│   │   └── evaluation_dataset.json
│   └── server.py          # FastMCP server entry point
├── outputs\                # Target directory for generated charts/CSVs
├── .env                    # API credentials
├── .gitignore
└── README.md
```

## 📋 Installation & Setup

### 1. Prerequisites

- Python 3.12+
- Ollama installed and running locally (for the evaluation judge model)
- `uv` for dependency management

### 2. Environment Configuration

```bash
uv venv
uv sync
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 3. Pull the Local Judge Model

```bash
ollama pull ornith:35b
```

## 📈 Running the Evaluation Suite

To run the MLflow evaluation suite against the dataset directly (without going through Claude Desktop or Cursor):

```bash
.venv\Scripts\python.exe backend\evaluation\evaluate.py
```

**Evaluation sequence:**

1. Loads `evaluation_dataset.json` (ticker/timeframe/action/rubric rows).
2. Runs each test case through the same parse → write → execute sequence the tools use at runtime.
3. Scores the generated code against its rubric using Ollama's `ornith:35b` as an LLM-as-Judge.
4. Logs performance metrics and execution traces to MLflow.
