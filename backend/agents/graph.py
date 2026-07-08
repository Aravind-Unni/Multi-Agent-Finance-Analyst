from langgraph.graph import StateGraph, END
from backend.schemas.state import AgentState
from backend.agents.parser import query_parser_node
from backend.agents.writer import code_writer_node
from backend.agents.executor import code_executor_node

def route_execution(state: AgentState) -> str:
    """
    Conditional router that checks the state for execution errors.
    If an error is found and we haven't hit the retry limit, it sends
    the state back to the writer node for debugging.
    """
    if state.get("error_traceback"):
        if state.get("retry_count", 0) < 3:
            return "writer"  # Loop back to Code Writer to debug and refactor
        return END          # Stop trying after 3 attempts to prevent infinite loops
    return END              # Everything executed successfully, exit graph

# Initialize the StateGraph with your custom state schema
workflow = StateGraph(AgentState)

# 1. Define the Nodes
workflow.add_node("parser", query_parser_node)
workflow.add_node("writer", code_writer_node)
workflow.add_node("executor", code_executor_node)

# 2. Set up the Workflow Flow
workflow.set_entry_point("parser")
workflow.add_edge("parser", "writer")
workflow.add_edge("writer", "executor")

# 3. Add the Conditional Edge for Error Recovery
workflow.add_conditional_edges(
    "executor",
    route_execution,
    {
        "writer": "writer",
        END: END
    }
)

# Compile the workflow into an executable app
financial_analyst_graph = workflow.compile()