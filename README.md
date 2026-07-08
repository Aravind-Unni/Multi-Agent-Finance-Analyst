# Market Intelligence Agent Using MCP

A Model Context Protocol (MCP) server built for the Cursor IDE, designed to convert natural language financial queries into executable, visual market insights. This system leverages a deterministic LangGraph architecture and a dynamic multi-LLM strategy to parse intent, generate quantitative Python code, and execute analysis safely within a sandboxed environment.

## 🌟 Key Highlight: MCP Integration for Cursor

This project acts as a custom MCP server, allowing Cursor's AI to natively understand, query, and trigger the financial LangGraph pipeline directly from your editor. It acts as a bridge between your IDE and the local `yfinance` analysis workflows.

### Cursor MCP Configuration

To connect this agent to Cursor, add the following configuration to your MCP settings file (or directly in the Cursor UI under **Settings > Features > MCP**).

**`cursor_mcp_config.json`**

```json
{
  "mcpServers": {
    "market-intelligence": {
      "command": "C:/Market Intelligence Agent/.venv/Scripts/python.exe",
      "args": [
        "server.py"
      ],
      "env": {
        "PYTHONPATH": "C:/Market Intelligence Agent"
      }
    }
  }
}
```

> Ensure the paths match your local workspace directory.

## 🧠 Multi-LLM Strategy

This project strategically routes tasks to different models based on compute constraints, rate limits, and reasoning requirements:

- **Ollama (Local — `ornith:35b`)**: Powers the Code Writer node. Bypasses cloud rate limits (like Groq) to handle heavy, repetitive code generation tasks securely on-device.
- **Groq / NVIDIA (Cloud — Llama 3.1 8B)**: Powers the Parser node. Handles high-speed, structured intent extraction (Pydantic JSON enforcement).
- **Groq (Llama 3.1 70B)**: Acts as the LLM-as-a-Judge in the MLflow evaluation suite, verifying code logic and mathematical accuracy against ground-truth rubrics.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | FastMCP (MCP Server), LangGraph, LangChain |
| LLM Inference | Ollama, Groq, NVIDIA AI Endpoints |
| Data Engine | yfinance, pandas, matplotlib |
| Evaluation | MLflow |
| Environment | Python 3.12+, uv package manager |

## ⚙️ Workflow Architecture

1. **Parser Node**: Analyzes raw queries to identify targets (e.g., `TSLA`), timeframes, and actions.
2. **Writer Node**: Generates pure Python code using `ornith:35b` and Tavily web search is used to fetch code or proper libraries from the web if llm fails at first attempt. It is strictly instructed to use `matplotlib.use('Agg')` to enable headless plotting, preventing GUI crashes in automated loops.
3. **Executor Node**: Safely executes the generated code, logs results, and writes `output_chart.png` to the local `/outputs` directory.

## 📂 Project Structure

```
C:\Market Intelligence Agent\
├── backend\
│   ├── agents\
│   │   ├── graph.py       # Compiled state machine
│   │   ├── parser.py      # Intent extraction (Llama 8B)
│   │   ├── writer.py      # Code generation (Ollama 35b)
│   │   └── executor.py    # Sandboxed execution
│   └── evaluation\
│       ├── evaluate.py    # MLflow evaluation suite
│       └── evaluation_dataset.json
├── outputs\               # Target directory for generated charts
├── .env                   # API Credentials
├── .gitignore             # Security and cache ignorance
├── server.py              # FastMCP Server Entry Point
└── README.md
```

## 📋 Installation & Setup

### 1. Prerequisites

- Python 3.12+
- Ollama installed and running locally.
- `uv` for dependency management (install the requirements.txt) .

### 2. Environment Configuration

Clone the repository and set up your virtual environment using `uv`:

```bash
uv venv
uv sync
```

Create a `.env` file in the root directory for your cloud LLMs and search tools:

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 3. Pull Local Models

Ensure your local Ollama instance has the required model downloaded:

```bash
ollama pull ornith:35b
```

## 📈 Running the Evaluation Suite

To run the automated MLflow evaluation suite against the dataset without triggering the MCP server:

```
.venv\Scripts\python.exe backend\evaluation\evaluate.py
```

**Evaluation Sequence:**

1. Loads `evaluation_dataset.json`.
2. Cycles through each query via the LangGraph pipeline.
3. Uses the Groq 70B Judge to score the generated code based on the expected rubric.
4. Outputs the performance metrics and execution traces via MLflow.
