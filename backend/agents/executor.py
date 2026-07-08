import sys
import io
import os
import traceback
from backend.schemas.state import AgentState

def code_executor_node(state: AgentState) -> dict:
    """
    Agent 3: Code Executor.
    Safely executes the generated Python script in an isolated namespace.
    Captures outputs and errors to prevent breaking the MCP stdio stream.
    """
    code = state.get("generated_code")
    
    if not code:
        return {"error_traceback": "Error: No code was provided by the writer node."}

    # CRITICAL FIX FOR CLAUDE DESKTOP: Force the current working directory 
    # to your project folder so file writes don't hit system permission barriers.
    try:
        os.chdir("C:\\Market Intelligence Agent")
    except Exception as e:
        return {"error_traceback": f"Failed to set working directory: {str(e)}"}

    # Create in-memory string buffers to capture print statements and errors
    captured_output = io.StringIO()
    captured_error = io.StringIO()
    
    # Save the original stdio streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to our buffers
        sys.stdout = captured_output
        sys.stderr = captured_error
        
        # Execute the code in an isolated dictionary to prevent variable bleeding
        local_namespace = {}
        exec(code, {"__name__": "__main__"}, local_namespace)
        
        # If execution completes without throwing an exception, it was successful
        success_msg = "Code executed successfully. Chart saved as 'output_chart.png'."
        output_text = captured_output.getvalue()
        if output_text:
            success_msg += f"\nTerminal Output:\n{output_text}"
            
        return {
            "execution_result": success_msg,
            "error_traceback": None # Clear any previous errors
        }

    except Exception as e:
        # If the code crashes, capture the exact traceback for the LLM to read
        error_msg = traceback.format_exc()
        current_retries = state.get("retry_count", 0)
        
        return {
            "error_traceback": error_msg,
            "retry_count": current_retries + 1
        }
        
    finally:
        # CRITICAL: Always restore the original stdio streams so MCP can talk to Claude again
        sys.stdout = original_stdout
        sys.stderr = original_stderr