from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from backend.schemas.state import AgentState
from langchain_ollama import ChatOllama

# 1. Initialize the Tavily Web Search Tool
tavily_tool = TavilySearchResults(
    max_results=3,
    description="Useful for searching the web for current Python library documentation, coding syntax, and debugging errors."
)

# 2. Initialize the Model
llm = ChatOllama(
    model="ornith:35b", 
    temperature=0.1
)

# 3. Create a Sub-Agent that knows how to use the tool
# This prebuilt agent handles the loop of: Search Web -> Read Results -> Write Answer
coding_assistant = create_react_agent(llm, tools=[tavily_tool])

def code_writer_node(state: AgentState) -> dict:
    """
    Agent 2: Code Writer with Web Search.
    Uses Tavily to look up syntax before writing the Python script.
    """
    params = state.get("parsed_params") or {}
    ticker = params.get("ticker", "UNKNOWN")
    metric = params.get("metric", "historical data")
    
    # Inject error traceback if routed back from the executor
    error_context = ""
    if state.get("error_traceback"):
        error_context = (
            f"\nWARNING: Your previous code crashed with this traceback:\n"
            f"{state['error_traceback']}\n"
            f"Please use the search tool to look up why this error occurred, then rewrite the code."
        )

    system_message = (
    "You are an expert Quantitative Python Developer. Write a complete, standalone Python "
    "script using `yfinance`, `pandas`, and `matplotlib` to analyze financial data.\n"
    f"Target Ticker: {ticker}\n"
    f"Target Metric/Analysis: {metric}\n"
    "CRITICAL INSTRUCTIONS FOR AUTOMATION:\n"
    "1. You must include these two lines at the very top of your script to prevent GUI crashes: \n"
    "   import matplotlib\n"
    "   matplotlib.use('Agg')\n"
    "2. Save the final plot exactly to the path 'outputs/output_chart.png'.\n"
    "3. NEVER use plt.show() as it will crash the automated pipeline.\n"
    "4. If you are unsure about any syntax, USE YOUR SEARCH TOOL to look up the documentation first.\n"
    "5. Do not use any input() prompts.\n"
    "6. Your final message must contain ONLY the raw, executable Python code. "
    "Do not wrap it in markdown blockquotes (no ```python). Just the raw code.\n"
    f"{error_context}"
    )
    # Invoke the ReAct agent with the instructions
    # The agent uses a standard "messages" state internally
    response = coding_assistant.invoke({
        "messages": [
            ("system", system_message),
            ("user", "Please write the python script now. Search the web if you need to verify library syntax.")
        ]
    })
    
    # Extract the final output from the agent's last message in the list
    final_output = response["messages"][-1].content
    
    # Clean up any stray markdown formatting
    clean_code = final_output.replace("```python", "").replace("```", "").strip()
    
    return {
        "generated_code": clean_code
    }