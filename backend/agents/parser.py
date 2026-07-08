import os
from typing import TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. Aligning with your official AgentState Schema
# ==========================================
class AgentState(TypedDict):
    query: str                          
    parsed_params: Optional[Dict[str, Any]] 
    generated_code: Optional[str]        
    execution_result: Optional[str]     
    error_traceback: Optional[str]      
    retry_count: int     

class FinancialQuery(BaseModel):
    """Schema for extracting financial data parameters from user queries."""
    ticker: str = Field(
        description="The stock ticker symbol. E.g., 'TSLA' for Tesla. If multiple, separate by comma (e.g., 'AAPL, MSFT')."
    )
    timeframe: str = Field(
        description="The time period for the analysis. E.g., 'YTD', '1Y', '6M', 'Max'."
    )
    action: str = Field(
        description="The core action requested by the user. E.g., 'plot line chart', 'histogram', 'calculate volatility'."
    )

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct", 
    temperature=0  
)

structured_parser = llm.with_structured_output(FinancialQuery)

prompt = PromptTemplate.from_template(
    "You are a precise financial data extraction agent.\n"
    "Extract the exact ticker, timeframe, and action from the following user query. Do not invent information.\n"
    "User Query: {query}"
)

extraction_chain = prompt | structured_parser

# ==========================================
# 2. Updated Node Function
# ==========================================
def query_parser_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the raw user query and extracts the required financial parameters.
    """
    print("---NODE 1: PARSING QUERY---")
    
    # LangGraph will no longer filter this out because 'query' matches AgentState!
    query = state.get("query", "")
    
    if not query:
        print("Warning: Received empty query.")
        return {"parsed_params": {"error": "Empty query provided."}}
        
    try:
        extracted_data = extraction_chain.invoke({"query": query})
        parsed_dict = extracted_data.model_dump()
        print(f"Extracted Parameters: {parsed_dict}")
        
        # Updating 'parsed_params' to match your official schema channel name
        return {"parsed_params": parsed_dict}
        
    except Exception as e:
        print(f"Parsing Failed: {e}")
        return {"parsed_params": {"error": f"Failed to parse query: {str(e)}"}}

if __name__ == "__main__":
    test_state = AgentState(
        query="Plot YTD stock gain of Tesla",
        parsed_params=None,
        generated_code=None,
        execution_result=None,
        error_traceback=None,
        retry_count=0
    )
    new_state = query_parser_node(test_state)
    